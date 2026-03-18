"""
Standalone re-validator for autoresearch-arabic dataset conditions.

Re-runs all mandatory validation checks on the built parquet shards without
rebuilding the data. Writes per-condition results to validation_report.json.

Usage:
    uv run python validate_dataset.py                  # validate all 3 conditions
    uv run python validate_dataset.py --condition d1   # validate only D1
    uv run python validate_dataset.py --condition d2   # validate only D2
    uv run python validate_dataset.py --condition d3   # validate only D3

Exit codes:
    0  — all requested conditions passed all mandatory checks
    1  — one or more conditions failed a mandatory check
"""

import argparse
import sys

from build_dataset import validate_condition, write_validation_report


def main(args: list[str] | None = None) -> None:
    """Parse arguments and validate requested conditions."""
    parser = argparse.ArgumentParser(
        description="Re-validate autoresearch-arabic dataset conditions without rebuilding.",
    )
    parser.add_argument(
        "--condition",
        choices=["d1", "d2", "d3", "all"],
        default="all",
        help="Which condition(s) to validate (default: all)",
    )
    parsed = parser.parse_args(args)

    if parsed.condition == "all":
        conditions = ["d1", "d2", "d3"]
    else:
        conditions = [parsed.condition]

    total = len(conditions)
    passed = 0
    failed = 0

    for cond in conditions:
        print(f"\nValidating condition: {cond}")
        try:
            val_results = validate_condition(cond)
            write_validation_report(cond, val_results)
            passed += 1
        except SystemExit:
            # validate_condition calls sys.exit(1) on hard failures.
            # Catch it so we can continue checking remaining conditions
            # and produce a complete failure report.
            failed += 1

    print(f"\nValidation complete: {passed}/{total} conditions passed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
