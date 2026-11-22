"""Cross the first three radio turns' saved history with the dose during the apology."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from typesafe_sdk import TypeSafeClient
from neuron_study.experiment import score_response
from neuron_study.gguf_run import resolve_engine, resolve_model


# Older standalone GGUF runs recorded a filename instead of a content identifier.
PUBLISHED_MODEL_SHA256 = "bb5b8b1308515427ca45544a46ec828f7a5472a505f2e8c9e60f26ce665a4a4d"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(folder, output, engine, model, env):
    source = json.loads((folder / "task-1.json").read_bytes())
    config = json.loads((folder / "config.json").read_bytes())
    turns = source["prompts"]
    if not source["completed"] or config["recipe"] != "combined":
        raise ValueError("A completed combined-recipe radio trajectory is required")
    if (list(config["tasks"]) != ["Community radio"] or config["max_tokens"] != 1536
            or source["settings"]["repetition_penalty"] != 1.05 or 50 not in config["doses"]):
        raise ValueError("Expected the published radio protocol including the half-dose branch")
    source_digest = re.fullmatch(r"sha256[-:]([0-9a-f]{64})", source["gguf_blob"])
    expected = source_digest.group(1) if source_digest else PUBLISHED_MODEL_SHA256
    model_digest = file_sha256(model)
    if model_digest != expected:
        raise ValueError(f"GGUF content differs from the radio model: expected SHA-256 {expected}, got {model_digest}")
    env = dict(env, QWEN_RUN_TRAJECTORY="1", QWEN_RUN_REPEAT_PENALTY="1.05", QWEN_RUN_TOKEN_CAP="1536")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("CUDA_VISIBLE_DEVICES", str(source.get("gpu_index", "0")))
    directions = folder / "task-1.directions.bin"
    provenance = {
        "source_run": folder.name,
        "source_results_sha256": file_sha256(folder / "task-1.json"),
        "source_config_sha256": file_sha256(folder / "config.json"),
        "directions_sha256": file_sha256(directions),
        "model_sha256": model_digest,
        "source_engine": source["engine"],
        "runner": Path(engine[0]).name,
        "runner_sha256": file_sha256(engine[0]),
        "cuda_device_order": env["CUDA_DEVICE_ORDER"],
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "judge": config["judge"],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.is_file():
        if json.loads(manifest.read_bytes()) != provenance:
            raise ValueError("Output directory belongs to another replay; choose a new directory")
    elif any(output.iterdir()):
        raise ValueError("Existing replay has no provenance manifest; choose a new output directory")
    else:
        manifest.write_text(json.dumps(provenance, indent=2), encoding="utf8")
    current = turns[3]["prompt"]
    inputs = output / "inputs.txt"
    inputs.write_text(source["calibration_text"] + "\n" + current + "\n", encoding="utf8")
    checks = []
    judge = TypeSafeClient(api_key=env["TYPESAFE_API_KEY"], timeout=90)
    metrics = []
    for prior, label in [(0, "baseline"), (0.5, "dose_050")]:
        prefix = output / f"history-{prior}.txt"
        lines, history = [], []
        for turn in turns[:3]:
            ans = turn["responses"][label]
            ids = ans["token_ids"][:-1] if ans["finish_reason"] == "stop" else ans["token_ids"]
            lines.extend([turn["prompt"], " ".join(map(str, ids))])
            history.append({"user": turn["prompt"], "assistant": ans["text"]})
        prefix.write_text("\n".join(lines) + "\n", encoding="utf8")
        cmd = [*engine, model, directions, inputs, "activation", "50", prefix]
        raw = output / f"history-{prior}.json"
        if not raw.exists():
            with (output / f"history-{prior}.log").open("wb") as log:
                result = subprocess.run(cmd, env=env, stderr=log, stdout=subprocess.PIPE)
            if result.returncode:
                raise RuntimeError(f"Native replay failed; see {log.name}. Reinstall the current checkout "
                                   "if your engine predates saved-history replay.")
            transcript = result.stdout.decode("utf8", errors="replace").splitlines()
            if "COMPLETE" not in transcript:
                raise RuntimeError("Native run did not finish")
            replies = [json.loads(line[7:]) for line in transcript if line.startswith("RESULT ")]
            raw.write_text(json.dumps({r["condition"]: r for r in replies}), encoding="utf8")
        replies = json.loads(raw.read_bytes())
        original = turns[3]["responses"][label]
        if replies[label]["token_ids"] != original["token_ids"]:
            raise RuntimeError("Unswitched answer differs from the original")
        if any(reply["input_tokens"] != original["input_tokens"] for reply in replies.values()):
            raise RuntimeError("Replayed input length differs from the original")
        if any(replies[k]["token_ids"] != replies["baseline"]["token_ids"] for k in ("zero_dose", "restored")):
            raise RuntimeError("Control tokens diverged")
        checks.append(dict(prior_dose=prior, input_tokens=original["input_tokens"],
                           unswitched_tokens_match=True, restored_tokens_match=True, zero_dose_tokens_match=True))
        for current_dose, condition in [(0,"baseline"), (0.5,"dose_050")]:
            reply = replies[condition]
            judging = json.dumps({"prior_turns":history,"current_user_request":current,"instruction":"Rate only the new assistant answer. Use prior turns to assess retained facts, constraints and goal."})
            scores, details = score_response(judge, config["judge"], judging, reply["text"], config["rubric"])
            (output / f"judge-{prior}-{current_dose}.json").write_text(json.dumps(details), encoding="utf8")
            metrics.append(dict(prior_dose=prior,current_dose=current_dose,input_tokens=reply["input_tokens"],
                                generated_tokens=len(reply["token_ids"]),finish=reply["finish_reason"],**scores))
    (output / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
    with (output / "crossover.csv").open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("radio_run", type=Path)
    parser.add_argument("output", type=Path, help="New directory for private transcripts and ratings")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    run(args.radio_run.resolve(), args.output.resolve(), [resolve_engine()],
        resolve_model("qwen3.5:4b-bf16"), os.environ)
