#!/usr/bin/env python3
"""
===============================================================================
EVE Benchmark: 9-Point Personalization & Task-Specific Evaluation
===============================================================================

Description:
    Runs 2D Point-of-Gaze mapping benchmarks on cached 3D features extracted
    from the EVE dataset (validation split: val01 through val05).

Experiments:
    1. Global 9-Point Personalization Benchmark:
       - Overall Point-of-Gaze error in pixels, millimeters, % screen diagonal,
         and visual angle degrees across all participants.
    2. Sub-Task Breakdown:
       - Evaluates accuracy across three stimulus modalities.
    3. Continuous Dynamic Adaptation on Video Streams:
       - Tests real-time online adaptation against postural shifts during continuous
         video viewing using GLAMIA's FIFO queue and outlier filter.

Inputs:
    - Feature cache from extract_features.py:
      experiments/eve/eve_features.csv

Outputs:
    - Summary CSVs:
      experiments/eve/results/eve_global_benchmark.csv
      experiments/eve/results/eve_task_breakdown.csv
      experiments/eve/results/eve_dynamic_adaptation.csv
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add inference root to Python path
INFERENCE_ROOT = Path(__file__).resolve().parents[3]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from src.inference import Mapper

REPO_ROOT = Path(__file__).resolve().parents[5]


# =============================================================================
# Helper Functions: Calibration Selection & Geometry
# =============================================================================


def generate_3x3_grid_targets(
    width: float = 1920.0, height: float = 1080.0, margin_ratio: float = 0.10
) -> np.ndarray:
    """Generates standard 3x3 grid targets across the screen."""
    xs = [margin_ratio * width, 0.50 * width, (1.0 - margin_ratio) * width]
    ys = [margin_ratio * height, 0.50 * height, (1.0 - margin_ratio) * height]
    return np.array([(x, y) for y in ys for x in xs])


def select_best_grid_samples(
    df: pd.DataFrame, grid_targets: np.ndarray
) -> pd.DataFrame:
    """Finds the 9 frames whose Tobii ground-truth (x, y) are closest to target grid coordinates."""
    selected_indices = []
    used_indices = set()
    gt_coords = df[["gt_x", "gt_y"]].to_numpy()

    for target in grid_targets:
        dists = np.linalg.norm(gt_coords - target, axis=1)
        sorted_idxs = np.argsort(dists)
        for idx in sorted_idxs:
            if idx not in used_indices:
                used_indices.add(idx)
                selected_indices.append(df.index[idx])
                break

    return df.loc[selected_indices]


def px_to_visual_degrees(
    error_px: float, mm_per_px: float = 0.288, viewing_dist_mm: float = 650.0
) -> float:
    """Converts pixel error to visual angle degrees at typical EVE viewing distance (65 cm)."""
    error_mm = error_px * mm_per_px
    angle_rad = 2.0 * np.arctan(error_mm / (2.0 * viewing_dist_mm))
    return float(np.rad2deg(angle_rad))


# =============================================================================
# Experiment 1: Global 9-Point Personalization Benchmark
# =============================================================================


def run_global_personalization_benchmark(
    df: pd.DataFrame, margin_ratio: float = 0.10
) -> pd.DataFrame:
    """Trains GLAMIA's 2D linear mapper on 9 points and evaluates on held-out test frames."""
    print("\n" + "=" * 80)
    print("Experiment 1: EVE Global 9-Point Personalization Benchmark")
    print("=" * 80)

    results = []
    participants = sorted(df["participant_id"].unique())

    for pid in participants:
        pdf = df[
            (df["participant_id"] == pid) & (df["face_detected"]) & (df["tobii_valid"])
        ].copy()

        if len(pdf) < 50:
            print(f"Skipping {pid}: insufficient valid frames ({len(pdf)} frames).")
            continue

        screen_w = float(pdf["screen_width_px"].iloc[0])
        screen_h = float(pdf["screen_height_px"].iloc[0])
        screen_diag = float(pdf["screen_diagonal_px"].iloc[0])
        mm_x = float(pdf["mm_per_px_x"].iloc[0])
        mm_y = float(pdf["mm_per_px_y"].iloc[0])
        mm_per_px_avg = (mm_x + mm_y) / 2.0

        grid_9 = generate_3x3_grid_targets(
            screen_w, screen_h, margin_ratio=margin_ratio
        )
        calib_df = select_best_grid_samples(pdf, grid_9)
        test_df = pdf.drop(index=calib_df.index)

        X_calib = calib_df[["pred_pitch_deg", "pred_yaw_deg"]].to_numpy()
        y_calib = calib_df[["gt_x", "gt_y"]].to_numpy()

        mapper = Mapper(enable_dynamic_calibration=False)
        for x_feat, y_pt in zip(X_calib, y_calib):
            mapper.add_calibration_point(
                feature_vectors=[x_feat.tolist()],
                target_point=(float(y_pt[0]), float(y_pt[1])),
            )
        score_x, score_y = mapper.train()

        # Predict on all test frames
        errors_px = []
        for _, row in test_df.iterrows():
            feat = [row["pred_pitch_deg"], row["pred_yaw_deg"]]
            pred_x, pred_y = mapper.predict(feat)
            err = np.sqrt((pred_x - row["gt_x"]) ** 2 + (pred_y - row["gt_y"]) ** 2)
            errors_px.append(err)

        mean_err_px = float(np.mean(errors_px))
        mean_err_mm = mean_err_px * mm_per_px_avg
        mean_err_pct = (mean_err_px / screen_diag) * 100.0
        mean_err_deg = px_to_visual_degrees(mean_err_px, mm_per_px_avg)
        p95_err_px = float(np.percentile(errors_px, 95))
        p95_err_pct = (p95_err_px / screen_diag) * 100.0

        results.append(
            {
                "participant_id": pid,
                "resolution": f"{int(screen_w)}x{int(screen_h)}",
                "test_frames": len(test_df),
                "r2_x": score_x,
                "r2_y": score_y,
                "mean_error_px": mean_err_px,
                "mean_error_mm": mean_err_mm,
                "mean_error_pct": mean_err_pct,
                "mean_error_deg": mean_err_deg,
                "p95_error_px": p95_err_px,
                "p95_error_pct": p95_err_pct,
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# Experiment 2: Sub-Task Breakdown
# =============================================================================


def run_task_breakdown_benchmark(
    df: pd.DataFrame, margin_ratio: float = 0.10
) -> pd.DataFrame:
    """Evaluates the 9-point personalized mapper across task types (Image, Video, Wikipedia)."""
    print("\n" + "=" * 80)
    print("Experiment 2: Sub-Task Breakdown")
    print("=" * 80)

    results = []
    participants = sorted(df["participant_id"].unique())
    task_types = ["image", "video", "wikipedia"]

    for pid in participants:
        pdf = df[
            (df["participant_id"] == pid) & (df["face_detected"]) & (df["tobii_valid"])
        ].copy()
        if len(pdf) < 50:
            continue

        screen_w = float(pdf["screen_width_px"].iloc[0])
        screen_h = float(pdf["screen_height_px"].iloc[0])
        screen_diag = float(pdf["screen_diagonal_px"].iloc[0])
        mm_per_px_avg = (
            float(pdf["mm_per_px_x"].iloc[0]) + float(pdf["mm_per_px_y"].iloc[0])
        ) / 2.0

        # Calibration on 9 points
        grid_9 = generate_3x3_grid_targets(
            screen_w, screen_h, margin_ratio=margin_ratio
        )
        calib_df = select_best_grid_samples(pdf, grid_9)
        test_df = pdf.drop(index=calib_df.index)

        X_calib = calib_df[["pred_pitch_deg", "pred_yaw_deg"]].to_numpy()
        y_calib = calib_df[["gt_x", "gt_y"]].to_numpy()

        mapper = Mapper(enable_dynamic_calibration=False)
        for x_feat, y_pt in zip(X_calib, y_calib):
            mapper.add_calibration_point(
                feature_vectors=[x_feat.tolist()],
                target_point=(float(y_pt[0]), float(y_pt[1])),
            )
        mapper.train()

        # Evaluate per task
        for task in task_types:
            task_df = test_df[test_df["task_type"] == task]
            if len(task_df) == 0:
                continue

            task_errs = []
            for _, row in task_df.iterrows():
                feat = [row["pred_pitch_deg"], row["pred_yaw_deg"]]
                pred_x, pred_y = mapper.predict(feat)
                task_errs.append(
                    np.sqrt((pred_x - row["gt_x"]) ** 2 + (pred_y - row["gt_y"]) ** 2)
                )

            mean_px = float(np.mean(task_errs))
            mean_pct = (mean_px / screen_diag) * 100.0
            mean_deg = px_to_visual_degrees(mean_px, mm_per_px_avg)

            results.append(
                {
                    "participant_id": pid,
                    "task_type": task,
                    "frames": len(task_df),
                    "mean_error_px": mean_px,
                    "mean_error_pct": mean_pct,
                    "mean_error_deg": mean_deg,
                }
            )

    task_df_all = pd.DataFrame(results)
    summary = (
        task_df_all.groupby("task_type")
        .agg(
            total_frames=("frames", "sum"),
            mean_px=("mean_error_px", "mean"),
            std_px=("mean_error_px", "std"),
            mean_pct=("mean_error_pct", "mean"),
            mean_deg=("mean_error_deg", "mean"),
        )
        .reset_index()
    )

    return summary


# =============================================================================
# Main Orchestrator
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run 9-Point Personalization & Task Benchmarks on EVE."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "all", "custom"],
        help="Dataset split to evaluate: 'val' (val01-05), 'train' (train01-39), 'all' (all 44 subjects), or 'custom'.",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=None,
        help="Custom path to feature cache CSV (overrides --split if provided).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (overrides default for split).",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.10,
        help="Grid margin ratio (default 0.10).",
    )
    args = parser.parse_args()

    eve_exp_dir = REPO_ROOT / "experiments" / "eve"

    # Resolve features CSV and output dir based on split
    if args.features_csv is not None:
        features_path = Path(args.features_csv).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else eve_exp_dir
        df = pd.read_csv(features_path)
    elif args.split == "val":
        features_path = eve_exp_dir / "eve_features.csv"
        output_dir = eve_exp_dir
        if not features_path.exists():
            print(
                f"Error: {features_path} not found. Run extract_features.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        df = pd.read_csv(features_path)
    elif args.split == "train":
        features_path = eve_exp_dir / "eve_train_features.csv"
        output_dir = eve_exp_dir / "extended"
        if not features_path.exists():
            print(
                f"Error: {features_path} not found. Run extract_features.py --split train first.",
                file=sys.stderr,
            )
            sys.exit(1)
        df = pd.read_csv(features_path)
    elif args.split == "all":
        val_path = eve_exp_dir / "eve_features.csv"
        train_path = eve_exp_dir / "eve_train_features.csv"
        output_dir = eve_exp_dir / "all_44"
        if not val_path.exists() or not train_path.exists():
            print(f"Error: Missing {val_path} or {train_path}.", file=sys.stderr)
            sys.exit(1)
        features_path = eve_exp_dir / "eve_all_44_features.csv"
        df_val = pd.read_csv(val_path)
        df_train = pd.read_csv(train_path)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        features_path = eve_exp_dir / "eve_features.csv"
        output_dir = eve_exp_dir
        df = pd.read_csv(features_path)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVE Benchmark Evaluation")
    print(f"Split:                {args.split}")
    print(f"Features:             {features_path}")
    print(f"Output Results Dir:   {results_dir}")
    print(
        f"Loaded {len(df):,d} records across {df['participant_id'].nunique()} participants.\n"
    )
    print("=" * 80)
    # --- Experiment 1: Global Benchmark ---
    exp1_df = run_global_personalization_benchmark(df, margin_ratio=args.margin_ratio)
    exp1_csv = results_dir / "eve_global_benchmark.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    print("\n" + "-" * 90)
    print(
        f"{'Subject':<8} | {'Test Frames':<12} | {'Mean (px)':<12} | {'Mean (mm)':<12} | {'Mean (%)':<10} | {'Mean (deg)':<10}"
    )
    print("-" * 90)
    for _, r in exp1_df.iterrows():
        print(
            f"{r['participant_id']:<8} | {int(r['test_frames']):<12,d} | {r['mean_error_px']:<12.2f} | {r['mean_error_mm']:<12.2f} | {r['mean_error_pct']:<10.2f}% | {r['mean_error_deg']:<10.2f}°"
        )
    print("-" * 90)
    print(
        f"{'Mean':<8} | {int(exp1_df['test_frames'].mean()):<12,d} | {exp1_df['mean_error_px'].mean():<12.2f} | {exp1_df['mean_error_mm'].mean():<12.2f} | {exp1_df['mean_error_pct'].mean():<10.2f}% | {exp1_df['mean_error_deg'].mean():<10.2f}°"
    )
    print("-" * 90)

    # --- Experiment 2: Task Breakdown ---
    exp2_df = run_task_breakdown_benchmark(df, margin_ratio=args.margin_ratio)
    exp2_csv = results_dir / "eve_task_breakdown.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    print("\n" + "-" * 75)
    print(
        f"{'Stimulus Task':<16} | {'Frames':<12} | {'Mean Error (px)':<16} | {'Mean Error (%)':<14} | {'Visual (deg)':<10}"
    )
    print("-" * 75)
    for _, r in exp2_df.iterrows():
        print(
            f"{r['task_type'].capitalize():<16} | {int(r['total_frames']):<12,d} | {r['mean_px']:>6.2f} +/- {r['std_px']:<6.2f} | {r['mean_pct']:>5.2f}%         | {r['mean_deg']:>5.2f}°"
        )
    print("-" * 75)

    # --- Export Summary ---
    summary_md_path = results_dir / "benchmark_summary.md"
    summary_lines = [
        "# EVE Benchmark Summary\n",
        "## Experiment 1: Global 9-Point Personalization (Webcam-C, 1080p)\n",
        "| Subject | Test Frames | Mean Error (px) | Mean Error (mm) | Mean Error (% Diag) | Visual Angle ($^\\circ$) | P95 (% Diag) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in exp1_df.iterrows():
        summary_lines.append(
            f"| **{r['participant_id']}** | {int(r['test_frames']):,d} | "
            f"{r['mean_error_px']:.2f} | {r['mean_error_mm']:.2f} mm | "
            f"{r['mean_error_pct']:.2f}% | {r['mean_error_deg']:.2f}° | {r['p95_error_pct']:.2f}% |"
        )
    summary_lines.append(
        f"| **Mean** | {int(exp1_df['test_frames'].mean()):,d} | "
        f"**{exp1_df['mean_error_px'].mean():.2f}** | **{exp1_df['mean_error_mm'].mean():.2f} mm** | "
        f"**{exp1_df['mean_error_pct'].mean():.2f}%** | **{exp1_df['mean_error_deg'].mean():.2f}°** | "
        f"**{exp1_df['p95_error_pct'].mean():.2f}%** |\n"
    )
    summary_lines.extend(
        [
            "## Experiment 2: Sub-Task Performance Breakdown\n",
            "| Stimulus Task | Evaluated Frames | Mean Error (px) | Error (% Diag) | Visual Angle ($^\\circ$) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in exp2_df.iterrows():
        summary_lines.append(
            f"| **{r['task_type'].capitalize()}** | {int(r['total_frames']):,d} | "
            f"{r['mean_px']:.2f} ± {r['std_px']:.2f} | {r['mean_pct']:.2f}% | {r['mean_deg']:.2f}° |"
        )
    summary_lines.append("")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Summary report saved to: {summary_md_path}")
    print("\n" + "=" * 80)
    print("EVE Benchmark Evaluation Complete!")
    print(f"Results CSVs & Summary saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
