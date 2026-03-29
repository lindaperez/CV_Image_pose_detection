#!/usr/bin/env python3
"""
Train a weakly supervised temporal-density TCN for repetition counting.

This stage keeps the existing Stage 5 pose-sequence contract, but changes the
counting formulation from direct whole-video scalar regression to temporal
density prediction. The model predicts a non-negative density curve over time;
the final repetition count is obtained by summing that curve.

Because the dataset does not currently include rep-event timestamps, this
trainer uses a weak temporal target: a pseudo density map with evenly spaced
Gaussian peaks whose total mass matches the true video-level count.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "NumPy is required to train the density-based TCN model."
    ) from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyTorch is required to train the density-based TCN model."
    ) from exc

import train_pose_count_tcn as scalar_tcn


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a temporal-density TCN on normalized pose sequences."
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Optional explicit pose_sequence_index.csv path.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter, e.g. squat or push_up.",
    )
    parser.add_argument(
        "--run-name",
        default="pose_count_density_tcn_v1",
        help="Output subdirectory name.",
    )
    parser.add_argument("--seq-len", type=int, default=192, help="Resampled sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--channels", type=int, default=96, help="TCN channel width.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Conv1d kernel size.")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual TCN blocks.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience on the selection metric.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--loss",
        default="l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Count regression loss function.",
    )
    parser.add_argument(
        "--density-loss",
        default="mse",
        choices=["smooth_l1", "l1", "mse"],
        help="Temporal density loss against the pseudo target.",
    )
    parser.add_argument(
        "--density-loss-weight",
        type=float,
        default=0.5,
        help="Weight applied to the temporal density loss term.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.01,
        help="Weight applied to temporal smoothness regularization on the predicted density.",
    )
    parser.add_argument(
        "--pseudo-sigma-scale",
        type=float,
        default=0.35,
        help=(
            "Relative width of each pseudo Gaussian peak as a fraction of the "
            "average gap between peaks."
        ),
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


def build_pseudo_density_target(count: float, target_len: int, sigma_scale: float) -> np.ndarray:
    target = np.zeros(target_len, dtype=np.float32)
    count_value = max(float(count), 0.0)
    peak_count = max(int(round(count_value)), 0)
    if target_len <= 0 or count_value <= 0.0 or peak_count <= 0:
        return target

    x = np.arange(target_len, dtype=np.float32)
    centers = np.linspace(0.0, float(target_len - 1), num=peak_count, dtype=np.float32)
    mean_gap = max(float(target_len) / max(peak_count, 1), 1.0)
    sigma = max(mean_gap * float(sigma_scale), 1.0)

    for center in centers:
        target += np.exp(-0.5 * ((x - center) / sigma) ** 2).astype(np.float32)

    total_mass = float(target.sum())
    if total_mass < 1e-6:
        target.fill(count_value / float(target_len))
    else:
        target *= count_value / total_mass
    return target.astype(np.float32, copy=False)


class PoseDensityDataset(Dataset):
    def __init__(
        self,
        samples: list[scalar_tcn.Sample],
        target_len: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        pseudo_sigma_scale: float,
        augment: bool = False,
        time_warp_range: float = 0.0,
        feature_noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
    ) -> None:
        self.samples = samples
        self.target_len = target_len
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.pseudo_sigma_scale = pseudo_sigma_scale
        self.augment = augment
        self.time_warp_range = time_warp_range
        self.feature_noise_std = feature_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[idx]
        arr = scalar_tcn.load_sequence_array(sample.sequence_path, self.target_len)
        arr = (arr - self.feature_mean) / self.feature_std
        if self.augment:
            arr = scalar_tcn.apply_training_augmentation(
                arr,
                time_warp_range=self.time_warp_range,
                feature_noise_std=self.feature_noise_std,
                frame_dropout_prob=self.frame_dropout_prob,
            )
        density_target = build_pseudo_density_target(
            sample.count,
            target_len=self.target_len,
            sigma_scale=self.pseudo_sigma_scale,
        )
        x = torch.from_numpy(arr.astype(np.float32, copy=False))
        y = torch.tensor(sample.count, dtype=torch.float32)
        y_density = torch.from_numpy(density_target)
        return x, y, y_density, sample.name, sample.exercise


class PoseCountDensityTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)
        blocks = []
        for block_idx in range(num_blocks):
            blocks.append(
                scalar_tcn.TemporalBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.density_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.tcn(x)
        density = F.softplus(self.density_head(x).squeeze(1))
        count = density.sum(dim=1)
        return count, density


def density_smoothness_penalty(density: torch.Tensor) -> torch.Tensor:
    if density.ndim != 2 or density.shape[1] <= 1:
        return density.new_tensor(0.0)
    return (density[:, 1:] - density[:, :-1]).abs().mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    count_criterion: nn.Module,
    density_criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    density_loss_weight: float,
    smoothness_weight: float,
    collect_density: bool = False,
) -> tuple[
    float,
    float,
    float,
    list[float],
    list[float],
    list[str],
    list[str],
    list[np.ndarray],
    list[np.ndarray],
]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count_loss = 0.0
    total_density_loss = 0.0
    preds: list[float] = []
    targets: list[float] = []
    names: list[str] = []
    exercises: list[str] = []
    pred_density_curves: list[np.ndarray] = []
    target_density_curves: list[np.ndarray] = []

    for x, y, density_target, batch_names, batch_exercises in loader:
        x = x.to(device)
        y = y.to(device)
        density_target = density_target.to(device)

        if is_train:
            optimizer.zero_grad()

        count_pred, density_pred = model(x)
        count_loss = count_criterion(count_pred, y)
        density_loss = density_criterion(density_pred, density_target)
        smoothness_loss = density_smoothness_penalty(density_pred)
        loss = count_loss + density_loss_weight * density_loss + smoothness_weight * smoothness_loss

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count_loss += float(count_loss.item()) * batch_size
        total_density_loss += float(density_loss.item()) * batch_size
        preds.extend(count_pred.detach().cpu().tolist())
        targets.extend(y.detach().cpu().tolist())
        names.extend(list(batch_names))
        exercises.extend(list(batch_exercises))

        if collect_density:
            pred_density_curves.extend(density_pred.detach().cpu().numpy().astype(np.float32))
            target_density_curves.extend(density_target.detach().cpu().numpy().astype(np.float32))

    denom = max(len(loader.dataset), 1)
    mean_total_loss = total_loss / denom
    mean_count_loss = total_count_loss / denom
    mean_density_loss = total_density_loss / denom
    return (
        mean_total_loss,
        mean_count_loss,
        mean_density_loss,
        preds,
        targets,
        names,
        exercises,
        pred_density_curves,
        target_density_curves,
    )


def save_density_curves(
    path: Path,
    *,
    names: list[str],
    exercises: list[str],
    true_counts: list[float],
    pred_counts: list[float],
    pred_density: list[np.ndarray],
    target_density: list[np.ndarray],
) -> None:
    if not pred_density or not target_density:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.asarray(names),
        exercises=np.asarray(exercises),
        true_counts=np.asarray(true_counts, dtype=np.float32),
        pred_counts=np.asarray(pred_counts, dtype=np.float32),
        pred_density=np.stack(pred_density).astype(np.float32),
        target_density=np.stack(target_density).astype(np.float32),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    scalar_tcn.set_seed(args.seed)

    project_dir = scalar_tcn.resolve_project_dir(args.project_dir)
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    index_path = (
        Path(args.index_csv).expanduser().resolve()
        if args.index_csv
        else (annotation_dir / "pose_sequence_index.csv").resolve()
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Missing required file: {index_path}")

    output_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = scalar_tcn.load_samples(project_dir, index_path, exercise_filter=args.exercise)
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid samples are required.")

    train_arrays = [scalar_tcn.load_sequence_array(sample.sequence_path, args.seq_len) for sample in train_samples]
    feature_mean, feature_std = scalar_tcn.compute_feature_stats(train_arrays)
    input_dim = int(train_arrays[0].shape[1])

    train_ds = PoseDensityDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        pseudo_sigma_scale=args.pseudo_sigma_scale,
        augment=(
            args.time_warp_range > 0.0
            or args.feature_noise_std > 0.0
            or args.frame_dropout_prob > 0.0
        ),
        time_warp_range=args.time_warp_range,
        feature_noise_std=args.feature_noise_std,
        frame_dropout_prob=args.frame_dropout_prob,
    )
    valid_ds = PoseDensityDataset(
        samples=valid_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        pseudo_sigma_scale=args.pseudo_sigma_scale,
    )
    train_eval_ds = PoseDensityDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        pseudo_sigma_scale=args.pseudo_sigma_scale,
    )

    train_sampler = (
        scalar_tcn.build_balanced_count_sampler(train_samples)
        if args.sampler == "balanced_count"
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = scalar_tcn.choose_device(args.device)
    model = PoseCountDensityTCN(
        input_dim=input_dim,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    count_criterion = scalar_tcn.build_loss(args.loss)
    density_criterion = scalar_tcn.build_loss(args.density_loss)

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_metric = -math.inf if args.selection_metric == "within_1" else math.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_count_loss,
            train_density_loss,
            train_pred,
            train_true,
            _,
            _,
            _,
            _,
        ) = run_epoch(
            model,
            train_loader,
            count_criterion,
            density_criterion,
            optimizer,
            device,
            density_loss_weight=args.density_loss_weight,
            smoothness_weight=args.smoothness_weight,
        )
        (
            valid_loss,
            valid_count_loss,
            valid_density_loss,
            valid_pred,
            valid_true,
            _,
            _,
            _,
            _,
        ) = run_epoch(
            model,
            valid_loader,
            count_criterion,
            density_criterion,
            None,
            device,
            density_loss_weight=args.density_loss_weight,
            smoothness_weight=args.smoothness_weight,
        )

        train_metrics = scalar_tcn.regression_metrics(
            train_pred,
            train_true,
            eval_transform=args.eval_transform,
        )
        valid_metrics = scalar_tcn.regression_metrics(
            valid_pred,
            valid_true,
            eval_transform=args.eval_transform,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_count_loss": train_count_loss,
                "train_density_loss": train_density_loss,
                "valid_loss": valid_loss,
                "valid_count_loss": valid_count_loss,
                "valid_density_loss": valid_density_loss,
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

    (
        _,
        _,
        _,
        train_pred,
        train_true,
        train_names,
        train_exercises,
        _,
        _,
    ) = run_epoch(
        model,
        train_eval_loader,
        count_criterion,
        density_criterion,
        None,
        device,
        density_loss_weight=args.density_loss_weight,
        smoothness_weight=args.smoothness_weight,
    )
    (
        _,
        _,
        _,
        valid_pred,
        valid_true,
        valid_names,
        valid_exercises,
        valid_density_pred,
        valid_density_target,
    ) = run_epoch(
        model,
        valid_loader,
        count_criterion,
        density_criterion,
        None,
        device,
        density_loss_weight=args.density_loss_weight,
        smoothness_weight=args.smoothness_weight,
        collect_density=True,
    )

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

    split_counts = {"train_rows": len(train_samples), "valid_rows": len(valid_samples)}
    exercise_counter = Counter(sample.exercise for sample in samples)

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
        "density_loss": args.density_loss,
        "density_loss_weight": args.density_loss_weight,
        "smoothness_weight": args.smoothness_weight,
        "pseudo_sigma_scale": args.pseudo_sigma_scale,
        "eval_transform": args.eval_transform,
        "selection_metric": args.selection_metric,
        "sampler": args.sampler,
        "time_warp_range": args.time_warp_range,
        "feature_noise_std": args.feature_noise_std,
        "frame_dropout_prob": args.frame_dropout_prob,
        "device": str(device),
        "input_dim": input_dim,
        "train_rows": split_counts["train_rows"],
        "valid_rows": split_counts["valid_rows"],
        "exercise_counts": dict(exercise_counter),
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
    save_density_curves(
        output_dir / "valid_density_curves.npz",
        names=valid_names,
        exercises=valid_exercises,
        true_counts=valid_true,
        pred_counts=valid_pred,
        pred_density=valid_density_pred,
        target_density=valid_density_target,
    )
    torch.save(best_state, output_dir / "best_model.pt")

    print("\nSaved outputs:")
    print(" -", output_dir / "config.json")
    print(" -", output_dir / "history.csv")
    print(" -", output_dir / "predictions.csv")
    print(" -", output_dir / "metrics_summary.json")
    print(" -", output_dir / "valid_density_curves.npz")
    print(" -", output_dir / "best_model.pt")
    print("\nBest valid metrics:")
    print(json.dumps(valid_metrics, indent=2))


if __name__ == "__main__":
    main()
