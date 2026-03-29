from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "NumPy is required to train the squat TCN model. "
        "Install it in the current environment before running this script."
    ) from exc

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyTorch is required to train the squat TCN model. "
        "Install it in the current environment before running this script."
    ) from exc


DEFAULT_FEATURE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "avg_knee_angle",
    "knee_flex",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "hip_center_y",
    "knee_center_y",
    "ankle_center_y",
    "hip_drop",
    "leg_extension",
    "hip_velocity",
    "frame_valid",
    "mean_conf",
]


def resolve_project_dir(project_dir_arg: str | None = None) -> Path:
    if project_dir_arg:
        project_dir = Path(project_dir_arg).expanduser().resolve()
        if (project_dir / "Data" / "LLSP").exists() and (project_dir / "artifacts").exists():
            return project_dir
        raise FileNotFoundError(
            f"Provided --project-dir does not look like the project root: {project_dir}"
        )

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        cand = (base / "CV_Image_pose_detection").resolve()
        if (cand / "Data" / "LLSP").exists() and (cand / "artifacts").exists():
            return cand

    for base in [cwd, *cwd.parents]:
        if (base / "Data" / "LLSP").exists() and (base / "artifacts").exists():
            return base.resolve()

    raise FileNotFoundError("Could not resolve project directory containing Data/LLSP and artifacts.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Sample:
    name: str
    split: str
    count: float
    feature_path: Path


def resolve_feature_path(project_dir: Path, raw_path: str) -> Path:
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    raw = Path(raw_path)

    # Local already-synced path.
    direct_local = annotation_dir / "squat_features" / raw.name
    if direct_local.exists():
        return direct_local.resolve()

    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    if not raw.is_absolute():
        joined = (annotation_dir / raw).resolve()
        if joined.exists():
            return joined

    raise FileNotFoundError(f"Could not resolve squat feature path from {raw_path!r}")


def load_samples(project_dir: Path, feature_index_path: Path) -> list[Sample]:
    rows = load_csv_rows(feature_index_path)
    samples: list[Sample] = []
    for row in rows:
        split = row["split"].strip().lower()
        if split not in {"train", "valid"}:
            continue
        feature_path = resolve_feature_path(project_dir, row["feature_path"])
        samples.append(
            Sample(
                name=row["name"].strip(),
                split=split,
                count=float(row["count"]),
                feature_path=feature_path,
            )
        )
    if not samples:
        raise RuntimeError(f"No train/valid samples found in {feature_index_path}")
    return samples


def select_feature_columns(array: np.ndarray, drop_frame_idx: bool) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected [T, F] feature array, got {array.shape}")
    if drop_frame_idx and array.shape[1] >= 2:
        return array[:, 1:]
    return array


def resample_sequence(array: np.ndarray, target_len: int) -> np.ndarray:
    if array.shape[0] == target_len:
        return array.astype(np.float32, copy=False)
    if array.shape[0] == 1:
        return np.repeat(array, target_len, axis=0).astype(np.float32, copy=False)

    src_x = np.linspace(0.0, 1.0, num=array.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, array.shape[1]), dtype=np.float32)
    for feat_idx in range(array.shape[1]):
        out[:, feat_idx] = np.interp(dst_x, src_x, array[:, feat_idx]).astype(np.float32)
    return out


def load_feature_array(path: Path, target_len: int, drop_frame_idx: bool) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = select_feature_columns(arr, drop_frame_idx=drop_frame_idx)
    return resample_sequence(arr, target_len=target_len)


def time_warp_sequence(array: np.ndarray, max_scale_delta: float) -> np.ndarray:
    if max_scale_delta <= 0.0 or array.shape[0] <= 1:
        return array

    scale = random.uniform(max(1.0 - max_scale_delta, 0.5), 1.0 + max_scale_delta)
    src_x = np.linspace(0.0, 1.0, num=array.shape[0], dtype=np.float32)
    warped_x = np.clip(((src_x - 0.5) / scale) + 0.5, 0.0, 1.0)
    warped = np.empty_like(array, dtype=np.float32)
    for feat_idx in range(array.shape[1]):
        warped[:, feat_idx] = np.interp(warped_x, src_x, array[:, feat_idx]).astype(np.float32)
    return warped


def apply_training_augmentation(
    array: np.ndarray,
    *,
    time_warp_range: float,
    feature_noise_std: float,
    frame_dropout_prob: float,
) -> np.ndarray:
    augmented = time_warp_sequence(array, max_scale_delta=time_warp_range)
    if frame_dropout_prob > 0.0:
        keep_mask = (np.random.rand(augmented.shape[0], 1) >= frame_dropout_prob).astype(np.float32)
        augmented = augmented * keep_mask
    if feature_noise_std > 0.0:
        noise = np.random.normal(0.0, feature_noise_std, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise
    return augmented.astype(np.float32, copy=False)


def compute_feature_stats(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(list(arrays), axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


class SquatTCNDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        target_len: int,
        drop_frame_idx: bool,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        augment: bool = False,
        time_warp_range: float = 0.0,
        feature_noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
    ) -> None:
        self.samples = samples
        self.target_len = target_len
        self.drop_frame_idx = drop_frame_idx
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.augment = augment
        self.time_warp_range = time_warp_range
        self.feature_noise_std = feature_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        arr = load_feature_array(sample.feature_path, self.target_len, self.drop_frame_idx)
        arr = (arr - self.feature_mean) / self.feature_std
        if self.augment:
            arr = apply_training_augmentation(
                arr,
                time_warp_range=self.time_warp_range,
                feature_noise_std=self.feature_noise_std,
                frame_dropout_prob=self.frame_dropout_prob,
            )
        x = torch.from_numpy(arr)
        y = torch.tensor(sample.count, dtype=torch.float32)
        return x, y, sample.name


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SquatTCNRegressor(nn.Module):
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
                TemporalBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, T, F]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.tcn(x)
        y = self.head(x)
        return y.squeeze(-1)


def transform_predictions(pred: Iterable[float], mode: str) -> list[float]:
    values = [float(x) for x in pred]
    if mode == "raw":
        return values
    if mode == "round":
        return [float(round(x)) for x in values]
    if mode == "round_clip_nonneg":
        return [float(max(0, round(x))) for x in values]
    raise ValueError(f"Unsupported eval transform: {mode}")


def regression_metrics(pred: list[float], true: list[float], eval_transform: str = "raw") -> dict[str, float]:
    if not pred:
        return {"rows": 0, "mae": math.nan, "rmse": math.nan, "within_1": math.nan}
    eval_pred = transform_predictions(pred, eval_transform)
    abs_err = [abs(a - b) for a, b in zip(eval_pred, true)]
    mse = sum((a - b) ** 2 for a, b in zip(eval_pred, true)) / len(eval_pred)
    within_1 = sum(1 for err in abs_err if err <= 1.0) / len(eval_pred)
    return {
        "rows": float(len(eval_pred)),
        "mae": float(sum(abs_err) / len(abs_err)),
        "rmse": float(math.sqrt(mse)),
        "within_1": float(within_1),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[float], list[float], list[str]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds: list[float] = []
    targets: list[float] = []
    names: list[str] = []

    for x, y, batch_names in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * x.shape[0]
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(y.detach().cpu().tolist())
        names.extend(list(batch_names))

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, preds, targets, names


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_balanced_count_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    count_freq = Counter(sample.count for sample in samples)
    weights = torch.tensor([1.0 / count_freq[sample.count] for sample in samples], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(samples), replacement=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a squat-only TCN rep-count regressor.")
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument("--run-name", default="squat_tcn_v1", help="Output subdirectory name.")
    parser.add_argument("--seq-len", type=int, default=128, help="Resampled sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--channels", type=int, default=64, help="TCN channel width.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Conv1d kernel size.")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual TCN blocks.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience on valid MAE.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--loss",
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Regression loss function.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device selection.",
    )
    parser.add_argument(
        "--keep-frame-idx",
        action="store_true",
        help="Keep the first feature column (frame_idx). By default it is dropped.",
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


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=1.0)
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def is_better_metric(current: float, best: float, metric_name: str) -> bool:
    if metric_name == "within_1":
        return current > best
    return current < best


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    project_dir = resolve_project_dir(args.project_dir)
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    feature_index_path = annotation_dir / "squat_feature_index.csv"
    if not feature_index_path.exists():
        raise FileNotFoundError(f"Missing required file: {feature_index_path}")

    output_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(project_dir, feature_index_path)
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid squat samples are required.")

    drop_frame_idx = not args.keep_frame_idx
    train_arrays = [
        load_feature_array(sample.feature_path, args.seq_len, drop_frame_idx=drop_frame_idx)
        for sample in train_samples
    ]
    feature_mean, feature_std = compute_feature_stats(train_arrays)
    input_dim = int(train_arrays[0].shape[1])

    train_ds = SquatTCNDataset(
        samples=train_samples,
        target_len=args.seq_len,
        drop_frame_idx=drop_frame_idx,
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
    valid_ds = SquatTCNDataset(
        samples=valid_samples,
        target_len=args.seq_len,
        drop_frame_idx=drop_frame_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    train_eval_ds = SquatTCNDataset(
        samples=train_samples,
        target_len=args.seq_len,
        drop_frame_idx=drop_frame_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_sampler = build_balanced_count_sampler(train_samples) if args.sampler == "balanced_count" else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    model = SquatTCNRegressor(
        input_dim=input_dim,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_loss(args.loss)

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    if args.selection_metric == "within_1":
        best_valid_metric = -math.inf
    else:
        best_valid_metric = math.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_pred, valid_true, _ = run_epoch(model, valid_loader, criterion, None, device)
        train_metrics = regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
        valid_metrics = regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

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
        if is_better_metric(current_valid_metric, best_valid_metric, args.selection_metric):
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

    _, train_pred, train_true, train_names = run_epoch(model, train_eval_loader, criterion, None, device)
    _, valid_pred, valid_true, valid_names = run_epoch(model, valid_loader, criterion, None, device)
    train_eval_pred = transform_predictions(train_pred, args.eval_transform)
    valid_eval_pred = transform_predictions(valid_pred, args.eval_transform)
    train_metrics = regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
    valid_metrics = regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

    combined_rows: list[dict[str, object]] = []
    for split, names, raw_pred, eval_pred, true in [
        ("train", train_names, train_pred, train_eval_pred, train_true),
        ("valid", valid_names, valid_pred, valid_eval_pred, valid_true),
    ]:
        for name, raw_pred_value, eval_pred_value, true_value in zip(names, raw_pred, eval_pred, true):
            combined_rows.append(
                {
                    "name": name,
                    "split": split,
                    "true_count": true_value,
                    "raw_pred_count": raw_pred_value,
                    "eval_pred_count": eval_pred_value,
                    "abs_error": abs(eval_pred_value - true_value),
                }
            )

    config = {
        "run_name": args.run_name,
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
        "drop_frame_idx": drop_frame_idx,
        "input_dim": input_dim,
        "feature_columns": DEFAULT_FEATURE_COLUMNS if drop_frame_idx else (["frame_idx"] + DEFAULT_FEATURE_COLUMNS),
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
    }

    summary = {
        "best_epoch": best_epoch,
        "best_selection_metric": best_valid_metric,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }

    write_csv(output_dir / "history.csv", list(history_rows[0].keys()), history_rows)
    write_csv(
        output_dir / "predictions.csv",
        ["name", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
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
