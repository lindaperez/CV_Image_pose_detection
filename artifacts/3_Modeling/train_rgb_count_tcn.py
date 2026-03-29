#!/usr/bin/env python3
"""
Train a counting-only TCN on frozen RGB feature sequences.

This is the RGB analogue of the pose-sequence TCN baseline. It reuses the same
artifact format so results can be compared directly with the pose branch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("NumPy is required to train the RGB TCN model.") from exc

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("PyTorch is required to train the RGB TCN model.") from exc

import train_pose_count_tcn as scalar_tcn


@dataclass
class RGBSample:
    name: str
    split: str
    count: float
    exercise: str
    feature_path: Path
    backbone: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a counting-only TCN on frozen RGB features.")
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Optional explicit rgb_feature_index.csv path.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter, e.g. squat or push_up.",
    )
    parser.add_argument("--run-name", default="rgb_count_tcn_v1", help="Output subdirectory name.")
    parser.add_argument("--seq-len", type=int, default=192, help="Resampled sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--channels", type=int, default=96, help="TCN channel width.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Conv1d kernel size.")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual TCN blocks.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience on the selection metric.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--loss",
        default="l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Regression loss function.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use auto, cpu, cuda, or an explicit torch device like cuda:0.",
    )
    parser.add_argument(
        "--eval-transform",
        default="raw",
        choices=["raw", "round", "round_clip_nonneg"],
        help="How to transform predicted counts before computing evaluation metrics.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "within_1"],
        help="Validation metric used for early stopping and best-checkpoint selection.",
    )
    parser.add_argument(
        "--sampler",
        default="shuffle",
        choices=["shuffle", "balanced_count"],
        help="Train sampling strategy. balanced_count upsamples rare rep-count targets.",
    )
    parser.add_argument(
        "--time-warp-range",
        type=float,
        default=0.0,
        help="Max relative temporal warp applied during training augmentation.",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.0,
        help="Stddev of Gaussian noise added to normalized training features.",
    )
    parser.add_argument(
        "--frame-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of dropping a whole timestep during training augmentation.",
    )
    return parser


def resolve_feature_path(project_dir: Path, raw_path: str) -> Path:
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    raw = Path(raw_path)

    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    if not raw.is_absolute():
        joined = (annotation_dir / raw).resolve()
        if joined.exists():
            return joined

    direct_local = annotation_dir / raw.name
    if direct_local.exists():
        return direct_local.resolve()

    for candidate_dir in sorted(annotation_dir.glob("rgb_*_features")):
        candidate = (candidate_dir / raw.name).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve RGB feature path from {raw_path!r}")


def load_samples(project_dir: Path, index_path: Path, exercise_filter: str | None) -> list[RGBSample]:
    rows = scalar_tcn.load_csv_rows(index_path)
    samples: list[RGBSample] = []
    skipped_bad_count: list[str] = []
    for row in rows:
        split = row["split"].strip().lower()
        if split not in {"train", "valid"}:
            continue
        exercise = row["type"].strip()
        if exercise_filter and exercise != exercise_filter:
            continue
        count = scalar_tcn.parse_count(row.get("count"))
        if count is None:
            skipped_bad_count.append(row["name"].strip())
            continue
        samples.append(
            RGBSample(
                name=row["name"].strip(),
                split=split,
                count=count,
                exercise=exercise,
                feature_path=resolve_feature_path(project_dir, row["feature_path"]),
                backbone=row.get("backbone", "").strip(),
            )
        )
    if skipped_bad_count:
        preview = ", ".join(skipped_bad_count[:5])
        print(
            f"Skipped {len(skipped_bad_count)} row(s) with missing/non-numeric count"
            f" in {index_path.name}: {preview}"
        )
    if not samples:
        scope = exercise_filter or "all exercises"
        raise RuntimeError(f"No train/valid samples found for {scope} in {index_path}")
    return samples


def load_feature_array(path: Path, target_len: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, F] RGB feature array, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return scalar_tcn.resample_sequence(arr, target_len=target_len)


class RGBFeatureDataset(Dataset):
    def __init__(
        self,
        samples: list[RGBSample],
        target_len: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        augment: bool = False,
        time_warp_range: float = 0.0,
        feature_noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
    ) -> None:
        self.samples = samples
        self.target_len = target_len
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.augment = augment
        self.time_warp_range = time_warp_range
        self.feature_noise_std = feature_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[idx]
        arr = load_feature_array(sample.feature_path, self.target_len)
        arr = (arr - self.feature_mean) / self.feature_std
        if self.augment:
            arr = scalar_tcn.apply_training_augmentation(
                arr,
                time_warp_range=self.time_warp_range,
                feature_noise_std=self.feature_noise_std,
                frame_dropout_prob=self.frame_dropout_prob,
            )
        x = torch.from_numpy(arr)
        y = torch.tensor(sample.count, dtype=torch.float32)
        return x, y, sample.name, sample.exercise


def main() -> None:
    args = build_arg_parser().parse_args()
    scalar_tcn.set_seed(args.seed)

    project_dir = scalar_tcn.resolve_project_dir(args.project_dir)
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    index_path = (
        Path(args.index_csv).expanduser().resolve()
        if args.index_csv
        else (annotation_dir / "rgb_feature_index_selected.csv").resolve()
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Missing required file: {index_path}")

    output_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(project_dir, index_path, exercise_filter=args.exercise)
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid samples are required.")

    train_arrays = [load_feature_array(sample.feature_path, args.seq_len) for sample in train_samples]
    feature_mean, feature_std = scalar_tcn.compute_feature_stats(train_arrays)
    input_dim = int(train_arrays[0].shape[1])

    train_ds = RGBFeatureDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        augment=(
            args.time_warp_range > 0.0
            or args.feature_noise_std > 0.0
            or args.frame_dropout_prob > 0.0
        ),
        time_warp_range=args.time_warp_range,
        feature_noise_std=args.feature_noise_std,
        frame_dropout_prob=args.frame_dropout_prob,
    )
    valid_ds = RGBFeatureDataset(
        samples=valid_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    train_eval_ds = RGBFeatureDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_sampler = scalar_tcn.build_balanced_count_sampler(train_samples) if args.sampler == "balanced_count" else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = scalar_tcn.choose_device(args.device)
    model = scalar_tcn.PoseCountTCNRegressor(
        input_dim=input_dim,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = scalar_tcn.build_loss(args.loss)

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_metric = -math.inf if args.selection_metric == "within_1" else math.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true, _, _ = scalar_tcn.run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_pred, valid_true, _, _ = scalar_tcn.run_epoch(model, valid_loader, criterion, None, device)
        train_metrics = scalar_tcn.regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
        valid_metrics = scalar_tcn.regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_within_1": train_metrics["within_1"],
                "valid_mae": valid_metrics["mae"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_within_1": valid_metrics["within_1"],
                "selection_metric": valid_metrics[args.selection_metric],
            }
        )

        print(
            f"[{epoch:03d}] "
            f"train_mae={train_metrics['mae']:.4f} "
            f"valid_mae={valid_metrics['mae']:.4f} "
            f"valid_within_1={valid_metrics['within_1']:.4f} "
            f"selection({args.selection_metric})={valid_metrics[args.selection_metric]:.4f}"
        )

        current_valid_metric = valid_metrics[args.selection_metric]
        if scalar_tcn.is_better_metric(current_valid_metric, best_valid_metric, args.selection_metric):
            best_valid_metric = current_valid_metric
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    _, train_pred, train_true, train_names, train_exercises = scalar_tcn.run_epoch(
        model, train_eval_loader, criterion, None, device
    )
    _, valid_pred, valid_true, valid_names, valid_exercises = scalar_tcn.run_epoch(model, valid_loader, criterion, None, device)
    train_eval_pred = scalar_tcn.transform_predictions(train_pred, args.eval_transform)
    valid_eval_pred = scalar_tcn.transform_predictions(valid_pred, args.eval_transform)
    train_metrics = scalar_tcn.regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
    valid_metrics = scalar_tcn.regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

    combined_rows: list[dict[str, object]] = []
    for split, names, exercises, raw_pred, eval_pred, true in [
        ("train", train_names, train_exercises, train_pred, train_eval_pred, train_true),
        ("valid", valid_names, valid_exercises, valid_pred, valid_eval_pred, valid_true),
    ]:
        for name, exercise, raw_pred_value, eval_pred_value, true_value in zip(
            names, exercises, raw_pred, eval_pred, true
        ):
            combined_rows.append(
                {
                    "name": name,
                    "type": exercise,
                    "split": split,
                    "true_count": true_value,
                    "raw_pred_count": raw_pred_value,
                    "eval_pred_count": eval_pred_value,
                    "abs_error": abs(eval_pred_value - true_value),
                }
            )

    exercise_counter = {}
    backbone_counter = {}
    for sample in samples:
        exercise_counter[sample.exercise] = exercise_counter.get(sample.exercise, 0) + 1
        backbone_key = sample.backbone or "unknown"
        backbone_counter[backbone_key] = backbone_counter.get(backbone_key, 0) + 1

    config = {
        "run_name": args.run_name,
        "index_csv": str(index_path),
        "exercise_filter": args.exercise,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "channels": args.channels,
        "kernel_size": args.kernel_size,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "loss": args.loss,
        "eval_transform": args.eval_transform,
        "selection_metric": args.selection_metric,
        "sampler": args.sampler,
        "time_warp_range": args.time_warp_range,
        "feature_noise_std": args.feature_noise_std,
        "frame_dropout_prob": args.frame_dropout_prob,
        "device": str(device),
        "input_dim": input_dim,
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
        "exercise_counts": exercise_counter,
        "backbone_counts": backbone_counter,
    }

    summary = {
        "best_epoch": best_epoch,
        "best_selection_metric": best_valid_metric,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }

    scalar_tcn.write_csv(output_dir / "history.csv", list(history_rows[0].keys()), history_rows)
    scalar_tcn.write_csv(
        output_dir / "predictions.csv",
        ["name", "type", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
        combined_rows,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_dir / "feature_mean.npy", feature_mean)
    np.save(output_dir / "feature_std.npy", feature_std)
    torch.save(best_state, output_dir / "best_model.pt")

    print("\nSaved outputs:")
    print(" -", output_dir / "config.json")
    print(" -", output_dir / "history.csv")
    print(" -", output_dir / "predictions.csv")
    print(" -", output_dir / "metrics_summary.json")
    print(" -", output_dir / "best_model.pt")
    print("\nBest valid metrics:")
    print(json.dumps(valid_metrics, indent=2))


if __name__ == "__main__":
    main()
