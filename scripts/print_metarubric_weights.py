"""Print weight summary for metarubric JSON files.

Usage: python scripts/print_metarubric_weights.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "autorubric" / "meta" / "data"


def print_weights(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  {path.name}")
    print(f"{'=' * 60}")

    total_positive = 0.0
    total_negative = 0.0
    criteria_count = 0

    for section in data["rubric"]["sections"]:
        section_pos = 0.0
        section_neg = 0.0
        print(f"\n  Section: {section['name']}")
        for criterion in section["criteria"]:
            w = criterion["weight"]
            name = criterion["name"]
            print(f"    [{w:+.0f}] {name}")
            if w > 0:
                section_pos += w
                total_positive += w
            else:
                section_neg += abs(w)
                total_negative += abs(w)
            criteria_count += 1
        print(f"    --- section: +{section_pos:.0f} / -{section_neg:.0f}")

    print(f"\n  Total criteria: {criteria_count}")
    print(f"  Total positive weight: +{total_positive:.0f}")
    print(f"  Total negative weight: -{total_negative:.0f}")
    if total_positive:
        print(f"  Normalized penalty ratio: {total_negative / total_positive:.2%}")


def main() -> None:
    for name in ["meta_rubric_standalone.json", "meta_rubric_in_context.json"]:
        path = DATA_DIR / name
        if path.exists():
            print_weights(path)


if __name__ == "__main__":
    main()
