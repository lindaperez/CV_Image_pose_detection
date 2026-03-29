from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


try:
    import train_pose_count_tcn as base
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Could not import train_pose_count_tcn.py. Run this script from the "
        "artifacts/3_Modeling directory or keep the sibling trainer file available."
    ) from exc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PoseCountTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("--model-dim must be divisible by --num-heads.")

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.shape[1], :]
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a counting-only transformer encoder on normalized pose sequences."
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Optional explicit pose_sequence_index.csv path. Defaults to Data/LLSP/annotation_cleaned/pose_sequence_index.csv.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter, e.g. squat or push_up. If omitted, trains on all rows in the index.",
    )
    parser.add_argument("--run-name", default="pose_count_transformer_v1", help="Output subdirectory name.")
    parser.add_argument("--seq-len", type=int, default=192, help="Resampled sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--model-dim", type=int, default=192, help="Transformer embedding width.")
    parser.add_argument("--num-heads", type=int, default=6, help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of encoder blocks.")
    parser.add_argument("--ff-dim", type=int, default=384, help="Transformer feed-forward width.")
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
    parser.add_argument(
        "--camera-motion-std",
        type=float,
        default=0.0,
        help="Stddev of global x/y pose drift used to approximate camera motion in normalized pose coordinates.",
    )
    parser.add_argument(
        "--camera-zoom-std",
        type=float,
        default=0.0,
        help="Stddev of global pose scale drift used to approximate camera zoom in normalized pose coordinates.",
    )
    parser.add_argument(
        "--joint-occlusion-prob",
        type=float,
        default=0.0,
        help="Probability of masking a random set of joints for a contiguous span to mimic occlusion.",
    )
    parser.add_argument(
        "--joint-occlusion-min-ratio",
        type=float,
        default=0.10,
        help="Minimum occlusion span as a fraction of the sequence length.",
    )
    parser.add_argument(
        "--joint-occlusion-max-ratio",
        type=float,
        default=0.35,
        help="Maximum occlusion span as a fraction of the sequence length.",
    )
    parser.add_argument(
        "--joint-occlusion-max-joints",
        type=int,
        default=2,
        help="Maximum number of joints to mask in one occlusion event.",
    )
    parser.add_argument(
        "--keypoint-profile",
        default="none",
        help=(
            "Keypoint weighting profile. Use none, auto_exercise_v1, or a named "
            "profile such as squat_v1, push_up_v1, or pull_up_v1."
        ),
    )
    parser.add_argument(
        "--profile-strength",
        type=float,
        default=1.0,
        help="Interpolation strength toward the selected keypoint profile. 0.0 disables weighting.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base.set_seed(args.seed)

    project_dir = base.resolve_project_dir(args.project_dir)
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

    samples = base.load_samples(project_dir, index_path, exercise_filter=args.exercise)
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid samples are required.")

    train_arrays = [base.load_sequence_array(sample.sequence_path, args.seq_len) for sample in train_samples]
    feature_mean, feature_std = base.compute_feature_stats(train_arrays)
    input_dim = int(train_arrays[0].shape[1])
    feature_weight_vector, resolved_profile_name = base.build_feature_weight_vector(
        input_dim=input_dim,
        profile_name=args.keypoint_profile,
        exercise_name=args.exercise,
        strength=args.profile_strength,
    )
    if resolved_profile_name:
        print(
            f"Applying keypoint profile {resolved_profile_name} "
            f"(requested={args.keypoint_profile}, strength={args.profile_strength:.2f})"
        )

    train_ds = base.PoseSequenceDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_weight_vector=feature_weight_vector,
        augment=(
            args.time_warp_range > 0.0
            or args.feature_noise_std > 0.0
            or args.frame_dropout_prob > 0.0
            or args.camera_motion_std > 0.0
            or args.camera_zoom_std > 0.0
            or args.joint_occlusion_prob > 0.0
        ),
        time_warp_range=args.time_warp_range,
        camera_motion_std=args.camera_motion_std,
        camera_zoom_std=args.camera_zoom_std,
        feature_noise_std=args.feature_noise_std,
        frame_dropout_prob=args.frame_dropout_prob,
        joint_occlusion_prob=args.joint_occlusion_prob,
        joint_occlusion_min_ratio=args.joint_occlusion_min_ratio,
        joint_occlusion_max_ratio=args.joint_occlusion_max_ratio,
        joint_occlusion_max_joints=args.joint_occlusion_max_joints,
    )
    valid_ds = base.PoseSequenceDataset(
        samples=valid_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_weight_vector=feature_weight_vector,
    )
    train_eval_ds = base.PoseSequenceDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_weight_vector=feature_weight_vector,
    )

    train_sampler = base.build_balanced_count_sampler(train_samples) if args.sampler == "balanced_count" else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = base.choose_device(args.device)
    model = PoseCountTransformerRegressor(
        input_dim=input_dim,
        seq_len=args.seq_len,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = base.build_loss(args.loss)

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_metric = -math.inf if args.selection_metric == "within_1" else math.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true, _, _ = base.run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_pred, valid_true, _, _ = base.run_epoch(model, valid_loader, criterion, None, device)
        train_metrics = base.regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
        valid_metrics = base.regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

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
        if base.is_better_metric(current_valid_metric, best_valid_metric, args.selection_metric):
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

    _, train_pred, train_true, train_names, train_exercises = base.run_epoch(
        model, train_eval_loader, criterion, None, device
    )
    _, valid_pred, valid_true, valid_names, valid_exercises = base.run_epoch(
        model, valid_loader, criterion, None, device
    )
    train_eval_pred = base.transform_predictions(train_pred, args.eval_transform)
    valid_eval_pred = base.transform_predictions(valid_pred, args.eval_transform)
    train_metrics = base.regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
    valid_metrics = base.regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

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

    split_counts = {
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
    }
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
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
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
        "camera_motion_std": args.camera_motion_std,
        "camera_zoom_std": args.camera_zoom_std,
        "joint_occlusion_prob": args.joint_occlusion_prob,
        "joint_occlusion_min_ratio": args.joint_occlusion_min_ratio,
        "joint_occlusion_max_ratio": args.joint_occlusion_max_ratio,
        "joint_occlusion_max_joints": args.joint_occlusion_max_joints,
        "keypoint_profile": args.keypoint_profile,
        "resolved_keypoint_profile": resolved_profile_name,
        "profile_strength": args.profile_strength,
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

    base.write_csv(output_dir / "history.csv", list(history_rows[0].keys()), history_rows)
    base.write_csv(
        output_dir / "predictions.csv",
        ["name", "type", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
        combined_rows,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    base.np.save(output_dir / "feature_mean.npy", feature_mean)
    base.np.save(output_dir / "feature_std.npy", feature_std)
    if feature_weight_vector is not None:
        base.np.save(output_dir / "feature_weight_vector.npy", feature_weight_vector)
    torch.save(best_state, output_dir / "best_model.pt")

    print("\nSaved outputs:")
    print(" -", output_dir / "config.json")
    print(" -", output_dir / "history.csv")
    print(" -", output_dir / "predictions.csv")
    print(" -", output_dir / "metrics_summary.json")
    if feature_weight_vector is not None:
        print(" -", output_dir / "feature_weight_vector.npy")
    print(" -", output_dir / "best_model.pt")
    print("\nBest valid metrics:")
    print(json.dumps(valid_metrics, indent=2))


if __name__ == "__main__":
    main()
