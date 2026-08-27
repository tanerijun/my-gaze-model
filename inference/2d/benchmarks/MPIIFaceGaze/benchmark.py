#!/usr/bin/env python3
"""
===============================================================================
MPIIFaceGaze Benchmark: 9-Point Personalization & Cross-Day Evaluation
===============================================================================

Description:
    Evaluates GLAMIA's 2D linear mapping on cached 3D features extracted
    from the MPIIFaceGaze dataset (15 participants, ~37,600 valid frames).

Experiments:
    1. Global 9-Point Personalization Benchmark: Evaluates overall few-shot
       accuracy on held-out frames (N - 9) across all 15 subjects.
    2. Within-Day vs. Cross-Day Analysis: Evaluates same-session baseline error
       vs. multi-day postural and depth decay.

Inputs:
    - Feature cache from extract_features.py:
      experiments/mpiifacegaze/mpiifacegaze_features.csv

Outputs:
    - Summary CSVs:
      experiments/mpiifacegaze/results/mpiifacegaze_global_benchmark.csv
      experiments/mpiifacegaze/results/mpiifacegaze_cross_day_decay.csv
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

# =============================================================================
# Helper Functions: Calibration Grid Targets & Sampling
# =============================================================================


def generate_3x3_grid_targets(
    width: float, height: float, margin_ratio: float = 0.10
) -> np.ndarray:
    """
    Generates standard 3x3 calibration grid targets across the screen real estate.
    Points placed at margin, 50%, and (1 - margin) of screen width and height.
    """
    xs = [margin_ratio * width, 0.50 * width, (1.0 - margin_ratio) * width]
    ys = [margin_ratio * height, 0.50 * height, (1.0 - margin_ratio) * height]
    return np.array([(x, y) for y in ys for x in xs])


def select_best_grid_samples(
    df: pd.DataFrame,
    grid_targets: np.ndarray,
) -> pd.DataFrame:
    """
    Finds the 9 frames whose ground-truth (x, y) coordinates are closest
    to the target 3x3 calibration grid positions.
    """
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


# =============================================================================
# Experiment 1: Global 9-Point Personalization Benchmark
# =============================================================================


def run_global_personalization_benchmark(
    df: pd.DataFrame, margin_ratio: float = 0.10
) -> pd.DataFrame:
    """
    Trains GLAMIA's 2D linear mapper on the 9 best grid-fitting samples
    across each participant's dataset, and evaluates on all remaining (N - 9) frames.
    """
    print("\n" + "=" * 80)
    print("Experiment 1: Global 9-Point Personalization Benchmark (All Participants)")
    print("=" * 80)

    results = []
    participants = sorted(df["participant_id"].unique())

    for pid in participants:
        pdf = df[(df["participant_id"] == pid) & (df["face_detected"])].copy()

        if len(pdf) < 20:
            print(f"Skipping {pid}: insufficient face detections ({len(pdf)} frames).")
            continue

        screen_w = float(pdf["screen_width_px"].iloc[0])
        screen_h = float(pdf["screen_height_px"].iloc[0])
        screen_diag = float(pdf["screen_diagonal_px"].iloc[0])

        # Step 1: Select best 9 grid points across participant dataset
        grid_9 = generate_3x3_grid_targets(
            screen_w, screen_h, margin_ratio=margin_ratio
        )
        calib_df = select_best_grid_samples(pdf, grid_9)
        test_df = pdf.drop(index=calib_df.index)

        # Step 2: Fit 2D Linear Mapper (OLS)
        X_calib = calib_df[["pred_pitch_deg", "pred_yaw_deg"]].to_numpy()
        y_calib = calib_df[["gt_x", "gt_y"]].to_numpy()

        mapper = Mapper(enable_dynamic_calibration=False)
        for x_feat, y_pt in zip(X_calib, y_calib):
            mapper.add_calibration_point(
                feature_vectors=[x_feat.tolist()],
                target_point=(float(y_pt[0]), float(y_pt[1])),
            )
        score_x, score_y = mapper.train()

        # Step 3: Predict on all held-out test frames
        errors_px = []
        for _, row in test_df.iterrows():
            feat = [row["pred_pitch_deg"], row["pred_yaw_deg"]]
            pred_x, pred_y = mapper.predict(feat)
            err = np.sqrt((pred_x - row["gt_x"]) ** 2 + (pred_y - row["gt_y"]) ** 2)
            errors_px.append(err)

        mean_err_px = float(np.mean(errors_px))
        mean_err_pct = (mean_err_px / screen_diag) * 100.0
        p95_err_px = float(np.percentile(errors_px, 95))
        p95_err_pct = (p95_err_px / screen_diag) * 100.0

        results.append(
            {
                "participant_id": pid,
                "resolution": f"{int(screen_w)}x{int(screen_h)}",
                "total_frames": len(pdf),
                "test_frames": len(test_df),
                "r2_score_x": score_x,
                "r2_score_y": score_y,
                "mean_error_px": mean_err_px,
                "mean_error_pct": mean_err_pct,
                "p95_error_px": p95_err_px,
                "p95_error_pct": p95_err_pct,
            }
        )

    res_df = pd.DataFrame(results)
    return res_df


# =============================================================================
# Experiment 2: Within-Day vs. Cross-Day Generalization Analysis
# =============================================================================


def run_within_vs_cross_day_analysis(
    df: pd.DataFrame, margin_ratio: float = 0.10
) -> pd.DataFrame:
    """
    Fits 9 calibration points on Day 1, then evaluates error on remaining Day 1
    frames (same-session baseline) vs. subsequent days (cross-day macro drift).
    """
    print("\n" + "=" * 80)
    print("Experiment 2: Within-Day vs. Cross-Day Generalization Analysis")
    print("=" * 80)

    results = []
    participants = sorted(df["participant_id"].unique())

    for pid in participants:
        pdf = df[df["participant_id"] == pid].copy()
        days = sorted(pdf["day"].unique())

        if len(days) < 2:
            continue

        day1_name = days[0]
        test_days = days[1:]

        day1_df = pdf[(pdf["day"] == day1_name) & (pdf["face_detected"])].copy()
        subsequent_df = pdf[
            (pdf["day"].isin(test_days)) & (pdf["face_detected"])
        ].copy()

        if len(day1_df) < 15 or len(subsequent_df) == 0:
            continue

        screen_w = float(pdf["screen_width_px"].iloc[0])
        screen_h = float(pdf["screen_height_px"].iloc[0])
        screen_diag = float(pdf["screen_diagonal_px"].iloc[0])

        # Fit on 9 best grid points in Day 1
        grid_9 = generate_3x3_grid_targets(
            screen_w, screen_h, margin_ratio=margin_ratio
        )
        calib_df = select_best_grid_samples(day1_df, grid_9)
        day1_test_df = day1_df.drop(index=calib_df.index)

        X_calib = calib_df[["pred_pitch_deg", "pred_yaw_deg"]].to_numpy()
        y_calib = calib_df[["gt_x", "gt_y"]].to_numpy()

        mapper = Mapper(enable_dynamic_calibration=False)
        for x_feat, y_pt in zip(X_calib, y_calib):
            mapper.add_calibration_point(
                feature_vectors=[x_feat.tolist()],
                target_point=(float(y_pt[0]), float(y_pt[1])),
            )
        mapper.train()

        # Evaluate on Day 1 remaining frames
        day1_errs = []
        for _, row in day1_test_df.iterrows():
            feat = [row["pred_pitch_deg"], row["pred_yaw_deg"]]
            pred_x, pred_y = mapper.predict(feat)
            day1_errs.append(
                np.sqrt((pred_x - row["gt_x"]) ** 2 + (pred_y - row["gt_y"]) ** 2)
            )

        # Evaluate on subsequent days
        cross_errs = []
        for _, row in subsequent_df.iterrows():
            feat = [row["pred_pitch_deg"], row["pred_yaw_deg"]]
            pred_x, pred_y = mapper.predict(feat)
            cross_errs.append(
                np.sqrt((pred_x - row["gt_x"]) ** 2 + (pred_y - row["gt_y"]) ** 2)
            )

        mean_day1_px = float(np.mean(day1_errs)) if day1_errs else np.nan
        mean_cross_px = float(np.mean(cross_errs))

        mean_day1_pct = (
            (mean_day1_px / screen_diag) * 100.0
            if not np.isnan(mean_day1_px)
            else np.nan
        )
        mean_cross_pct = (mean_cross_px / screen_diag) * 100.0
        drift_increase_pct = (
            ((mean_cross_px - mean_day1_px) / mean_day1_px) * 100.0
            if not np.isnan(mean_day1_px)
            else np.nan
        )

        results.append(
            {
                "participant_id": pid,
                "resolution": f"{int(screen_w)}x{int(screen_h)}",
                "day1_frames": len(day1_test_df),
                "cross_day_frames": len(subsequent_df),
                "within_day_mean_px": mean_day1_px,
                "within_day_mean_pct": mean_day1_pct,
                "cross_day_mean_px": mean_cross_px,
                "cross_day_mean_pct": mean_cross_pct,
                "decay_increase_pct": drift_increase_pct,
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# Main Orchestrator
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(
        description="Run 9-Point Personalization & Cross-Day Evaluation on MPIIFaceGaze."
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(
            REPO_ROOT / "experiments" / "mpiifacegaze" / "mpiifacegaze_features.csv"
        ),
        help="Path to feature cache generated by extract_features.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "mpiifacegaze"),
        help="Directory to save CSV results.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.10,
        help="Corner margin ratio for 3x3 grid targets (default 0.10 = 10%%/50%%/90%%).",
    )
    args = parser.parse_args()

    features_path = Path(args.features_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        print(f"Error: Feature cache not found at: {features_path}", file=sys.stderr)
        print(
            "Please run extract_features.py first to generate the feature cache.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 80)
    print("MPIIFaceGaze Benchmark Evaluation")
    print(f"Loading features from: {features_path}")
    print("=" * 80)

    df = pd.read_csv(features_path)
    print(
        f"Loaded {len(df)} records across {df['participant_id'].nunique()} participants.\n"
    )

    # --- Run Experiment 1: Global Personalization ---
    exp1_df = run_global_personalization_benchmark(df, margin_ratio=args.margin_ratio)
    exp1_csv = results_dir / "mpiifacegaze_global_benchmark.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    print("\n" + "-" * 85)
    print(
        f"{'Subject':<8} | {'Resolution':<10} | {'Test Frames':<12} | {'Mean (px)':<12} | {'Mean (%)':<10} | {'P95 (%)':<10}"
    )
    print("-" * 85)
    for _, r in exp1_df.iterrows():
        print(
            f"{r['participant_id']:<8} | {r['resolution']:<10} | {int(r['test_frames']):<12d} | {r['mean_error_px']:<12.2f} | {r['mean_error_pct']:<10.2f}% | {r['p95_error_pct']:<10.2f}%"
        )
    print("-" * 85)
    print(
        f"{'Mean':<8} | {'---':<10} | {int(exp1_df['test_frames'].mean()):<12d} | {exp1_df['mean_error_px'].mean():<12.2f} | {exp1_df['mean_error_pct'].mean():<10.2f}% | {exp1_df['p95_error_pct'].mean():<10.2f}%"
    )
    print("-" * 85)

    # --- Run Experiment 2: Within-Day vs. Cross-Day ---
    exp2_df = run_within_vs_cross_day_analysis(df, margin_ratio=args.margin_ratio)
    exp2_csv = results_dir / "mpiifacegaze_cross_day_decay.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    print("\n" + "-" * 80)
    print(
        f"{'Subject':<8} | {'Resolution':<10} | {'Within-Day (px)':<16} | {'Within-Day (%)':<14} | {'Cross-Day (%)':<14}"
    )
    print("-" * 80)
    for _, r in exp2_df.iterrows():
        print(
            f"{r['participant_id']:<8} | {r['resolution']:<10} | {r['within_day_mean_px']:<16.2f} | {r['within_day_mean_pct']:<14.2f}% | {r['cross_day_mean_pct']:<14.2f}%"
        )
    print("-" * 80)
    print(
        f"{'Mean':<8} | {'---':<10} | {exp2_df['within_day_mean_px'].mean():<16.2f} | {exp2_df['within_day_mean_pct'].mean():<14.2f}% | {exp2_df['cross_day_mean_pct'].mean():<14.2f}%"
    )
    print("-" * 80)
    # --- Export Summary ---
    summary_md_path = results_dir / "benchmark_summary.md"
    summary_lines = [
        "# MPIIFaceGaze Benchmark Summary\n",
        "## Experiment 1: Global 9-Point Personalization\n",
        "| Subject | Resolution | Test Frames | Mean Error (px) | Mean Error (% Diag) | P95 (% Diag) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for _, r in exp1_df.iterrows():
        summary_lines.append(
            f"| **{r['participant_id']}** | {r['resolution']} | {int(r['test_frames']):,d} | "
            f"{r['mean_error_px']:.2f} | {r['mean_error_pct']:.2f}% | {r['p95_error_pct']:.2f}% |"
        )
    summary_lines.append(
        f"| **Mean** | --- | {int(exp1_df['test_frames'].mean()):,d} | "
        f"**{exp1_df['mean_error_px'].mean():.2f}** | **{exp1_df['mean_error_pct'].mean():.2f}%** | "
        f"**{exp1_df['p95_error_pct'].mean():.2f}%** |\n"
    )
    summary_lines.extend(
        [
            "## Experiment 2: Within-Day vs. Cross-Day Generalization\n",
            r"| Subject | Resolution | Within-Day (px) | Within-Day (% Diag) | Cross-Day (% Diag) | Postural Decay ($\Delta$) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in exp2_df.iterrows():
        summary_lines.append(
            f"| **{r['participant_id']}** | {r['resolution']} | {r['within_day_mean_px']:.2f} | "
            f"{r['within_day_mean_pct']:.2f}% | {r['cross_day_mean_pct']:.2f}% | "
            f"{r['decay_increase_pct']:+.1f}% |"
        )
    summary_lines.append(
        f"| **Mean** | --- | **{exp2_df['within_day_mean_px'].mean():.2f}** | "
        f"**{exp2_df['within_day_mean_pct'].mean():.2f}%** | **{exp2_df['cross_day_mean_pct'].mean():.2f}%** | "
        f"**{exp2_df['decay_increase_pct'].mean():+.1f}%** |\n"
    )
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Summary report saved to: {summary_md_path}")
    print("\n" + "=" * 80)
    print("Benchmark Evaluation Complete!")
    print(f"Results CSVs & Summary saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
