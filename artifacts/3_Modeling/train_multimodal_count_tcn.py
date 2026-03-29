#!/usr/bin/env python3
"""
Train a simple multimodal RGB+pose counting TCN.

This trainer joins:
- normalized pose sequences from Stage 5
- frozen RGB feature sequences from Stage 7 / 7B

and applies a lightweight late-fusion architecture so the multimodal branch can
be compared directly against the current pose-only and RGB-only baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("NumPy is required to train the multimodal TCN model.") from exc

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("PyTorch is required to train the multimodal TCN model.") from exc

import train_pose_count_tcn as pose_tcn
import train_rgb_count_tcn as rgb_tcn


@dataclass
class MultiSample:
    name: str
    split: str
    count: float
    exercise: str
    pose_sequence_path: Path
    rgb_feature_path: Path
    rgb_backbone: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple multimodal RGB+pose counting TCN.")
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--pose-index-csv",
        default=None,
        help="Optional explicit pose_sequence_index.csv path.",
    )
    parser.add_argument(
        "--rgb-index-csv",
        default=None,
        help="Optional explicit rgb_feature_index.csv path.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter, e.g. squat or push_up.",
    )
    parser.add_argument("--run-name", default="multimodal_pose_rgb_tcn_v1", help="Output subdirectory name.")
    parser.add_argument("--seq-len", type=int, default=192, help="Resampled sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--pose-channels", type=int, default=48, help="Pose branch channel width.")
    parser.add_argument("--rgb-channels", type=int, default=48, help="RGB branch channel width.")
    parser.add_argument("--fusion-channels", type=int, default=128, help="Fusion branch channel width.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Conv1d kernel size.")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual TCN blocks per branch.")
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
        "--pose-feature-noise-std",
        type=float,
        default=0.02,
        help="Stddev of Gaussian noise added to normalized pose features during training.",
    )
    parser.add_argument(
        "--rgb-feature-noise-std",
        type=float,
        default=0.02,
        help="Stddev of Gaussian noise added to normalized RGB features during training.",
    )
    parser.add_argument(
        "--frame-dropout-prob",
        type=float,
        default=0.03,
        help="Probability of dropping a whole timestep during multimodal training augmentation.",
    )
    return parser


def time_warp_sequence_with_scale(array: np.ndarray, scale: float) -> np.ndarray:
    if array.shape[0] <= 1:
        return array
    src_x = np.linspace(0.0, 1.0, num=array.shape[0], dtype=np.float32)
    warped_x = np.clip(((src_x - 0.5) / scale) + 0.5, 0.0, 1.0)
    warped = np.empty_like(array, dtype=np.float32)
    for feat_idx in range(array.shape[1]):
        warped[:, feat_idx] = np.interp(warped_x, src_x, array[:, feat_idx]).astype(np.float32)
    return warped


def apply_multimodal_training_augmentation(
    pose_array: np.ndarray,
    rgb_array: np.ndarray,
    *,
    time_warp_range: float,
    pose_feature_noise_std: float,
    rgb_feature_noise_std: float,
    frame_dropout_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose_aug = pose_array.astype(np.float32, copy=False)
    rgb_aug = rgb_array.astype(np.float32, copy=False)

    if time_warp_range > 0.0 and pose_aug.shape[0] > 1:
        scale = random.uniform(max(1.0 - time_warp_range, 0.5), 1.0 + time_warp_range)
        pose_aug = time_warp_sequence_with_scale(pose_aug, scale)
        rgb_aug = time_warp_sequence_with_scale(rgb_aug, scale)

    if frame_dropout_prob > 0.0:
        keep_mask = (np.random.rand(pose_aug.shape[0], 1) >= frame_dropout_prob).astype(np.float32)
        pose_aug = pose_aug * keep_mask
        rgb_aug = rgb_aug * keep_mask

    if pose_feature_noise_std > 0.0:
        pose_noise = np.random.normal(0.0, pose_feature_noise_std, size=pose_aug.shape).astype(np.float32)
        pose_aug = pose_aug + pose_noise
    if rgb_feature_noise_std > 0.0:
        rgb_noise = np.random.normal(0.0, rgb_feature_noise_std, size=rgb_aug.shape).astype(np.float32)
        rgb_aug = rgb_aug + rgb_noise

    return pose_aug.astype(np.float32, copy=False), rgb_aug.astype(np.float32, copy=False)


def load_multimodal_samples(
    project_dir: Path,
    pose_index_path: Path,
    rgb_index_path: Path,
    exercise_filter: str | None,
) -> list[MultiSample]:
    pose_samples = pose_tcn.load_samples(project_dir, pose_index_path, exercise_filter=exercise_filter)
    rgb_samples = rgb_tcn.load_samples(project_dir, rgb_index_path, exercise_filter=exercise_filter)

    pose_by_key = {(sample.name, sample.split): sample for sample in pose_samples}
    rgb_by_key = {(sample.name, sample.split): sample for sample in rgb_samples}

    shared_keys = sorted(set(pose_by_key) & set(rgb_by_key))
    samples: list[MultiSample] = []
    mismatched_rows: list[str] = []
    for key in shared_keys:
        pose_sample = pose_by_key[key]
        rgb_sample = rgb_by_key[key]
        if pose_sample.exercise != rgb_sample.exercise:
            mismatched_rows.append(pose_sample.name)
            continue
        if abs(pose_sample.count - rgb_sample.count) > 1e-6:
            mismatched_rows.append(pose_sample.name)
            continue
        samples.append(
            MultiSample(
                name=pose_sample.name,
                split=pose_sample.split,
                count=pose_sample.count,
                exercise=pose_sample.exercise,
                pose_sequence_path=pose_sample.sequence_path,
                rgb_feature_path=rgb_sample.feature_path,
                rgb_backbone=rgb_sample.backbone,
            )
        )

    if mismatched_rows:
        preview = ", ".join(mismatched_rows[:5])
        print(
            f"Skipped {len(mismatched_rows)} row(s) with pose/RGB metadata mismatch: {preview}"
        )
    if not samples:
        scope = exercise_filter or "all exercises"
        raise RuntimeError(
            f"No shared train/valid pose+RGB samples found for {scope} using "
            f"{pose_index_path.name} and {rgb_index_path.name}."
        )
    return samples


def load_pose_array(path: Path, target_len: int) -> np.ndarray:
    return pose_tcn.load_sequence_array(path, target_len)


def load_rgb_array(path: Path, target_len: int) -> np.ndarray:
    return rgb_tcn.load_feature_array(path, target_len)


class MultimodalFeatureDataset(Dataset):
    def __init__(
        self,
        samples: list[MultiSample],
        target_len: int,
        pose_mean: np.ndarray,
        pose_std: np.ndarray,
        rgb_mean: np.ndarray,
        rgb_std: np.ndarray,
        augment: bool = False,
        time_warp_range: float = 0.0,
        pose_feature_noise_std: float = 0.0,
        rgb_feature_noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
    ) -> None:
        self.samples = samples
        self.target_len = target_len
        self.pose_mean = pose_mean
        self.pose_std = pose_std
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.augment = augment
        self.time_warp_range = time_warp_range
        self.pose_feature_noise_std = pose_feature_noise_std
        self.rgb_feature_noise_std = rgb_feature_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[idx]
        pose_arr = load_pose_array(sample.pose_sequence_path, self.target_len)
        rgb_arr = load_rgb_array(sample.rgb_feature_path, self.target_len)

        pose_arr = (pose_arr - self.pose_mean) / self.pose_std
        rgb_arr = (rgb_arr - self.rgb_mean) / self.rgb_std

        if self.augment:
            pose_arr, rgb_arr = apply_multimodal_training_augmentation(
                pose_arr,
                rgb_arr,
                time_warp_range=self.time_warp_range,
                pose_feature_noise_std=self.pose_feature_noise_std,
                rgb_feature_noise_std=self.rgb_feature_noise_std,
                frame_dropout_prob=self.frame_dropout_prob,
            )

        pose_x = torch.from_numpy(pose_arr)
        rgb_x = torch.from_numpy(rgb_arr)
        y = torch.tensor(sample.count, dtype=torch.float32)
        return pose_x, rgb_x, y, sample.name, sample.exercise


class MultimodalFusionTCNRegressor(nn.Module):
    def __init__(
        self,
        pose_input_dim: int,
        rgb_input_dim: int,
        pose_channels: int,
        rgb_channels: int,
        fusion_channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pose_proj = nn.Conv1d(pose_input_dim, pose_channels, kernel_size=1)
        self.rgb_proj = nn.Conv1d(rgb_input_dim, rgb_channels, kernel_size=1)
        self.pose_tcn = nn.Sequential(
            *[
                pose_tcn.TemporalBlock(
                    channels=pose_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
                for block_idx in range(num_blocks)
            ]
        )
        self.rgb_tcn = nn.Sequential(
            *[
                pose_tcn.TemporalBlock(
                    channels=rgb_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
                for block_idx in range(num_blocks)
            ]
        )
        self.fusion_proj = nn.Conv1d(pose_channels + rgb_channels, fusion_channels, kernel_size=1)
        self.fusion_tcn = nn.Sequential(
            *[
                pose_tcn.TemporalBlock(
                    channels=fusion_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
                for block_idx in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(fusion_channels, fusion_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_channels, 1),
        )

    def forward(self, pose_x: torch.Tensor, rgb_x: torch.Tensor) -> torch.Tensor:
        pose_x = pose_x.transpose(1, 2)
        rgb_x = rgb_x.transpose(1, 2)
        pose_feat = self.pose_tcn(self.pose_proj(pose_x))
        rgb_feat = self.rgb_tcn(self.rgb_proj(rgb_x))
        fused = torch.cat([pose_feat, rgb_feat], dim=1)
        fused = self.fusion_tcn(self.fusion_proj(fused))
        y = self.head(fused)
        return y.squeeze(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[float], list[float], list[str], list[str]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds: list[float] = []
    targets: list[float] = []
    names: list[str] = []
    exercises: list[str] = []

    for pose_x, rgb_x, y, batch_names, batch_exercises in loader:
        pose_x = pose_x.to(device)
        rgb_x = rgb_x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(pose_x, rgb_x)
        loss = criterion(pred, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * pose_x.shape[0]
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(y.detach().cpu().tolist())
        names.extend(list(batch_names))
        exercises.extend(list(batch_exercises))

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, preds, targets, names, exercises


def main() -> None:
    args = build_arg_parser().parse_args()
    pose_tcn.set_seed(args.seed)

    project_dir = pose_tcn.resolve_project_dir(args.project_dir)
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    pose_index_path = (
        Path(args.pose_index_csv).expanduser().resolve()
        if args.pose_index_csv
        else (annotation_dir / "pose_sequence_index.csv").resolve()
    )
    rgb_index_path = (
        Path(args.rgb_index_csv).expanduser().resolve()
        if args.rgb_index_csv
        else (annotation_dir / "rgb_feature_index_selected.csv").resolve()
    )
    if not pose_index_path.exists():
        raise FileNotFoundError(f"Missing required file: {pose_index_path}")
    if not rgb_index_path.exists():
        raise FileNotFoundError(f"Missing required file: {rgb_index_path}")

    output_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_multimodal_samples(
        project_dir,
        pose_index_path=pose_index_path,
        rgb_index_path=rgb_index_path,
        exercise_filter=args.exercise,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid shared pose+RGB samples are required.")

    pose_train_arrays = [load_pose_array(sample.pose_sequence_path, args.seq_len) for sample in train_samples]
    rgb_train_arrays = [load_rgb_array(sample.rgb_feature_path, args.seq_len) for sample in train_samples]
    pose_mean, pose_std = pose_tcn.compute_feature_stats(pose_train_arrays)
    rgb_mean, rgb_std = pose_tcn.compute_feature_stats(rgb_train_arrays)
    pose_input_dim = int(pose_train_arrays[0].shape[1])
    rgb_input_dim = int(rgb_train_arrays[0].shape[1])

    train_ds = MultimodalFeatureDataset(
        samples=train_samples,
        target_len=args.seq_len,
        pose_mean=pose_mean,
        pose_std=pose_std,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        augment=(
            args.time_warp_range > 0.0
            or args.pose_feature_noise_std > 0.0
            or args.rgb_feature_noise_std > 0.0
            or args.frame_dropout_prob > 0.0
        ),
        time_warp_range=args.time_warp_range,
        pose_feature_noise_std=args.pose_feature_noise_std,
        rgb_feature_noise_std=args.rgb_feature_noise_std,
        frame_dropout_prob=args.frame_dropout_prob,
    )
    valid_ds = MultimodalFeatureDataset(
        samples=valid_samples,
        target_len=args.seq_len,
        pose_mean=pose_mean,
        pose_std=pose_std,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
    )
    train_eval_ds = MultimodalFeatureDataset(
        samples=train_samples,
        target_len=args.seq_len,
        pose_mean=pose_mean,
        pose_std=pose_std,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
    )

    train_sampler = pose_tcn.build_balanced_count_sampler(train_samples) if args.sampler == "balanced_count" else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = pose_tcn.choose_device(args.device)
    model = MultimodalFusionTCNRegressor(
        pose_input_dim=pose_input_dim,
        rgb_input_dim=rgb_input_dim,
        pose_channels=args.pose_channels,
        rgb_channels=args.rgb_channels,
        fusion_channels=args.fusion_channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = pose_tcn.build_loss(args.loss)

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_metric = -math.inf if args.selection_metric == "within_1" else math.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_pred, valid_true, _, _ = run_epoch(model, valid_loader, criterion, None, device)
        train_metrics = pose_tcn.regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
        valid_metrics = pose_tcn.regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

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
        if pose_tcn.is_better_metric(current_valid_metric, best_valid_metric, args.selection_metric):
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

    _, train_pred, train_true, train_names, train_exercises = run_epoch(
        model, train_eval_loader, criterion, None, device
    )
    _, valid_pred, valid_true, valid_names, valid_exercises = run_epoch(model, valid_loader, criterion, None, device)
    train_eval_pred = pose_tcn.transform_predictions(train_pred, args.eval_transform)
    valid_eval_pred = pose_tcn.transform_predictions(valid_pred, args.eval_transform)
    train_metrics = pose_tcn.regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
    valid_metrics = pose_tcn.regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

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

    exercise_counter: dict[str, int] = {}
    backbone_counter: dict[str, int] = {}
    for sample in samples:
        exercise_counter[sample.exercise] = exercise_counter.get(sample.exercise, 0) + 1
        backbone_key = sample.rgb_backbone or "unknown"
        backbone_counter[backbone_key] = backbone_counter.get(backbone_key, 0) + 1

    config = {
        "run_name": args.run_name,
        "pose_index_csv": str(pose_index_path),
        "rgb_index_csv": str(rgb_index_path),
        "exercise_filter": args.exercise,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pose_channels": args.pose_channels,
        "rgb_channels": args.rgb_channels,
        "fusion_channels": args.fusion_channels,
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
        "pose_feature_noise_std": args.pose_feature_noise_std,
        "rgb_feature_noise_std": args.rgb_feature_noise_std,
        "frame_dropout_prob": args.frame_dropout_prob,
        "device": str(device),
        "pose_input_dim": pose_input_dim,
        "rgb_input_dim": rgb_input_dim,
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
        "exercise_counts": exercise_counter,
        "rgb_backbone_counts": backbone_counter,
    }

    summary = {
        "best_epoch": best_epoch,
        "best_selection_metric": best_valid_metric,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }

    pose_tcn.write_csv(output_dir / "history.csv", list(history_rows[0].keys()), history_rows)
    pose_tcn.write_csv(
        output_dir / "predictions.csv",
        ["name", "type", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
        combined_rows,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_dir / "pose_feature_mean.npy", pose_mean)
    np.save(output_dir / "pose_feature_std.npy", pose_std)
    np.save(output_dir / "rgb_feature_mean.npy", rgb_mean)
    np.save(output_dir / "rgb_feature_std.npy", rgb_std)
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
