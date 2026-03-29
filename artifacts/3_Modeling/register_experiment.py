from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_SENTINELS = ("Data/LLSP", "artifacts")
REGISTRY_FIELDS = [
    "stage",
    "scope",
    "question",
    "primary_result",
    "decision",
    "artifact_reference",
]


def resolve_project_dir(project_dir_arg: str | None = None) -> Path:
    if project_dir_arg:
        project_dir = Path(project_dir_arg).expanduser().resolve()
        if all((project_dir / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return project_dir
        raise FileNotFoundError(
            f"Provided --project-dir does not look like the project root: {project_dir}"
        )

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        cand = (base / "CV_Image_pose_detection").resolve()
        if all((cand / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return cand

    for base in [cwd, *cwd.parents]:
        if all((base / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return base.resolve()

    raise FileNotFoundError("Could not resolve project directory containing Data/LLSP and artifacts.")


def load_registry_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_registry_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_primary_result_from_metrics(metrics_path: Path) -> str:
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    valid = summary.get("valid_metrics", {})
    if not valid:
        raise ValueError(f"Could not find valid_metrics in {metrics_path}")

    parts: list[str] = []
    if "mae" in valid:
        parts.append(f"MAE={float(valid['mae']):.4f}")
    if "rmse" in valid:
        parts.append(f"RMSE={float(valid['rmse']):.4f}")
    if "within_1" in valid:
        parts.append(f"Within-1={float(valid['within_1']):.4f}")
    if not parts:
        raise ValueError(f"No supported metrics found in {metrics_path}")
    return "; ".join(parts)


def normalize_artifact_reference(path_text: str, project_dir: Path) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        return path.as_posix()

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_dir).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append or update a row in artifacts/3_Modeling/experiment_registry.csv."
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--registry-csv",
        default=None,
        help="Optional explicit experiment_registry.csv path.",
    )
    parser.add_argument("--stage", required=True, help="Stable stage id, e.g. 7B or 9_transformer.")
    parser.add_argument("--scope", required=True, help="Short scope label, e.g. selected_exercises.")
    parser.add_argument("--question", required=True, help="Research question answered by the experiment.")
    parser.add_argument("--decision", required=True, help="Decision taken after reviewing the result.")
    parser.add_argument(
        "--artifact-reference",
        required=True,
        help="Primary notebook, JSON, or artifact path for the experiment row.",
    )
    parser.add_argument(
        "--primary-result",
        default=None,
        help="Short result string. If omitted, use --metrics-json to build one automatically.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional metrics_summary.json path used to build --primary-result automatically.",
    )
    parser.add_argument(
        "--print-row",
        action="store_true",
        help="Print the final row JSON after updating the registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = resolve_project_dir(args.project_dir)
    registry_path = (
        Path(args.registry_csv).expanduser().resolve()
        if args.registry_csv
        else project_dir / "artifacts" / "3_Modeling" / "experiment_registry.csv"
    )

    primary_result = (args.primary_result or "").strip()
    if not primary_result:
        if not args.metrics_json:
            raise SystemExit("Provide --primary-result or --metrics-json.")
        metrics_path = Path(args.metrics_json).expanduser().resolve()
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics JSON: {metrics_path}")
        primary_result = build_primary_result_from_metrics(metrics_path)

    row = {
        "stage": args.stage.strip(),
        "scope": args.scope.strip(),
        "question": args.question.strip(),
        "primary_result": primary_result,
        "decision": args.decision.strip(),
        "artifact_reference": normalize_artifact_reference(args.artifact_reference, project_dir),
    }

    rows = load_registry_rows(registry_path)
    replaced = False
    for idx, existing in enumerate(rows):
        if existing.get("stage", "").strip() == row["stage"]:
            rows[idx] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    write_registry_rows(registry_path, rows)
    action = "Updated" if replaced else "Appended"
    print(f"{action} registry row for stage {row['stage']} in {registry_path}")
    if args.print_row:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
