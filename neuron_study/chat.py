"""Open a fresh interactive conversation using a saved generation bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .gguf_run import resolve_engine, resolve_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.dose <= 2 or not 1 <= args.max_tokens <= 4096:
        parser.error("Use dose 0..2 and max-tokens 1..4096.")
    record = json.loads(args.run.read_bytes())
    env = dict(os.environ, QWEN_RUN_TOKEN_CAP=str(args.max_tokens),
               QWEN_RUN_REPEAT_PENALTY=str(record["settings"]["repetition_penalty"]))
    subprocess.run([resolve_engine(), resolve_model(record["model"]),
                    args.run.with_suffix(".directions.bin"), args.run.with_suffix(".inputs.txt"),
                    "chat", str(args.dose)], env=env, check=True)
