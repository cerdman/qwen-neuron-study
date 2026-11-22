"""Print the published trajectory totals using only the Python standard library."""
import argparse
import csv
from itertools import groupby
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true", help="Also print the public result tables from a source checkout")
    args = parser.parse_args()
    if args.paper:
        docs = Path(__file__).resolve().parents[1] / "docs"
        if not (docs / "arithmetic.csv").is_file():
            parser.error("--paper requires the data in a source checkout")
        for filename in ("panel.csv", "checks.csv", "arithmetic.csv", "crossover.csv"):
            print(filename)
            with (docs / filename).open(encoding="utf8") as stream:
                for row in csv.reader(stream):
                    print(" | ".join(row))
            print()
    with Path(__file__).with_name("published.csv").open(encoding="utf8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: (r["task"], float(r["dose"]), int(r["turn"])))
    print("Task | Dose | Turns | Tokens | Capped | Final task progress (0-4)")
    for (task, dose), group in groupby(rows, key=lambda r: (r["task"], r["dose"])):
        turns = list(group)
        print(f"{task} | {dose} | {len(turns)} | {sum(int(r['tokens']) for r in turns)} | "
              f"{sum(r['finish'] == 'length' for r in turns)} | {float(turns[-1]['task_progress']):.2f}")


if __name__ == "__main__":
    main()
