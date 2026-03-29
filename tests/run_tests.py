#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parents[0]

for entry in [str(TESTS_DIR), str(PROJECT_ROOT)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)


TEST_GROUPS: dict[str, list[str]] = {
    "data_prep": [
        "test_countix_helpers",
        "test_prepare_countix_manifest",
    ],
    "evaluation": [
        "test_bootstrap_confidence_intervals",
        "test_registry_and_routing",
    ],
    "review": [
        "test_hard_case_review_tools",
        "test_hard_case_review_server",
    ],
    "runtime": [
        "test_squat_counter_runtime",
        "test_live_squat_counter",
    ],
}
TEST_GROUPS["all"] = [
    module_name
    for group_name in ["data_prep", "evaluation", "review", "runtime"]
    for module_name in TEST_GROUPS[group_name]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped project test suites.")
    parser.add_argument(
        "group",
        nargs="?",
        default="all",
        choices=sorted(TEST_GROUPS.keys()),
        help="Which test group to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available test groups and their modules, then exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use unittest verbosity 2.",
    )
    return parser.parse_args()


def list_groups() -> None:
    print("Available test groups:")
    for group_name in sorted(TEST_GROUPS.keys()):
        print(f"- {group_name}")
        for module_name in TEST_GROUPS[group_name]:
            print(f"  - {module_name}")


def main() -> int:
    args = parse_args()
    if args.list:
        list_groups()
        return 0

    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromNames(TEST_GROUPS[args.group])
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
