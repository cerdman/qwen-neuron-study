"""Check the two-turn smoke experiment without printing any model responses."""
import json
from pathlib import Path
import sys


def check(path):
    run = json.loads(path.read_bytes())
    assert run["completed"] and run["recipe"] == "combined"
    assert set(run["measurements"]) == {"5", "13", "14"}
    assert len(run["prompts"]) == 2
    lengths = []
    for turn in run["prompts"]:
        answers = turn["responses"]
        ids = answers["baseline"]["token_ids"]
        assert ids and ids == answers["zero_dose"]["token_ids"] == answers["restored"]["token_ids"]
        assert answers["dose_050"]["first_step_max_logit_delta"] > 0.0001
        lengths.append(answers["dose_050"]["input_tokens"])
    assert lengths[1] > lengths[0]
    print("PASS: three calibrated layers, changed logits, matched controls and growing history on two turns.")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
