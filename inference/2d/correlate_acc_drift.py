#!/usr/bin/env python3
"""
Correlate gaze accuracy (error_px) with head movement (total_deviation).

Example:
python 2d/correlate_acc_drift.py \
  --pose-data experiment_results/sample1/head_pose_data.json \
  --eval-results experiment_results/sample1/static_evaluation_results.json \
  --output-dir experiment_results/sample1
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _resolve_eval_results_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".json":
        return input_path

    if input_path.suffix.lower() != ".txt":
        return input_path

    name = input_path.name
    candidate: Path | None = None

    if name.startswith("dynamic_evaluation_summary_BUF_"):
        candidate_name = (
            name.replace(
                "dynamic_evaluation_summary_BUF_",
                "dynamic_evaluation_results_BUF_",
                1,
            ).removesuffix(".txt")
            + ".json"
        )
        candidate = input_path.with_name(candidate_name)
    elif name == "static_evaluation_summary.txt":
        candidate = input_path.with_name("static_evaluation_results.json")

    if candidate is not None and candidate.exists():
        print(f"Info: Resolved summary file to evaluation JSON: {candidate}")
        return candidate

    return input_path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Expected a JSON evaluation file, but got invalid JSON: {path}. "
                "If you passed a summary .txt, pass the corresponding *_results*.json file instead."
            ) from exc


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _extract_pose_series(pose_payload: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pose_payload, list):
        raise ValueError("Head pose data must be a JSON array.")

    rows: list[tuple[float, float]] = []
    for entry in pose_payload:
        if not isinstance(entry, dict):
            continue
        timestamp_ms = _to_float(entry.get("timestamp_ms"))
        deviation = _to_float(entry.get("total_deviation"))
        if timestamp_ms is None or deviation is None:
            continue
        rows.append((timestamp_ms, deviation))

    if not rows:
        raise ValueError(
            "No valid head pose rows with timestamp_ms + total_deviation found."
        )

    rows.sort(key=lambda x: x[0])
    timestamps = np.array([r[0] for r in rows], dtype=float)
    deviations = np.array([r[1] for r in rows], dtype=float)
    return timestamps, deviations


def _extract_eval_rows(eval_payload: Any) -> list[dict[str, float]]:
    if isinstance(eval_payload, dict):
        results = eval_payload.get("evaluation_results")
        if results is None:
            results = eval_payload.get("results")
    elif isinstance(eval_payload, list):
        results = eval_payload
    else:
        results = None

    if not isinstance(results, list):
        raise ValueError(
            "Evaluation results must be a JSON array or a dict with 'evaluation_results'."
        )

    extracted: list[dict[str, float]] = []

    for row in results:
        if not isinstance(row, dict):
            continue

        timestamp_ms = None
        for key in ("timestamp_ms", "videoTimestamp", "timestamp"):
            timestamp_ms = _to_float(row.get(key))
            if timestamp_ms is not None:
                break

        error_px = None
        for key in ("error_px", "mean_error_px", "median_error_px"):
            error_px = _to_float(row.get(key))
            if error_px is not None:
                break

        if timestamp_ms is None or error_px is None:
            continue

        extracted.append(
            {
                "timestamp_ms": timestamp_ms,
                "error_px": error_px,
            }
        )

    if not extracted:
        raise ValueError("No valid evaluation rows with timestamp + error found.")

    return extracted


def _normalize_click_timestamps(
    click_ts: np.ndarray, pose_ts: np.ndarray
) -> np.ndarray:
    """
    Align click timeline to pose timeline when ranges do not overlap (e.g., absolute epoch timestamps).
    """
    click_min = float(np.min(click_ts))
    click_max = float(np.max(click_ts))
    pose_min = float(np.min(pose_ts))
    pose_max = float(np.max(pose_ts))

    has_overlap = not (click_max < pose_min or click_min > pose_max)
    if has_overlap:
        return click_ts

    return click_ts - click_min + pose_min


def _nearest_indices(sorted_reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_reference, queries, side="left")
    idx = np.clip(idx, 0, len(sorted_reference) - 1)
    prev_idx = np.clip(idx - 1, 0, len(sorted_reference) - 1)

    dist_prev = np.abs(queries - sorted_reference[prev_idx])
    dist_curr = np.abs(sorted_reference[idx] - queries)
    use_prev = dist_prev <= dist_curr
    return np.where(use_prev, prev_idx, idx)


def _extract_participant_id(
    pose_payload: Any, eval_payload: Any, eval_path: Path
) -> str:
    candidates: list[Any] = []

    for payload in (eval_payload, pose_payload):
        if not isinstance(payload, dict):
            continue

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            participant = metadata.get("participant")
            if isinstance(participant, dict):
                for key in ("id", "participantId", "name"):
                    if participant.get(key):
                        candidates.append(participant.get(key))
            for key in ("participantId", "participant_id", "sessionId"):
                if metadata.get(key):
                    candidates.append(metadata.get(key))

        for key in ("participantId", "participant_id", "sessionId"):
            if payload.get(key):
                candidates.append(payload.get(key))

    for value in candidates:
        if value is not None and str(value).strip():
            return str(value)

    return eval_path.parent.name


def _format_stat(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def _compute_binned_stats(
    x: np.ndarray, y: np.ndarray, num_bins: int
) -> tuple[list[dict[str, float]], np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, num_bins + 1))
    rows: list[dict[str, float]] = []

    for i in range(num_bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == num_bins - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)

        if not np.any(mask):
            continue

        x_bin = x[mask]
        y_bin = y[mask]
        mean_y = float(np.mean(y_bin))
        std_y = float(np.std(y_bin, ddof=1)) if len(y_bin) > 1 else 0.0
        sem_y = std_y / np.sqrt(len(y_bin)) if len(y_bin) > 1 else 0.0
        ci95 = 1.96 * sem_y

        rows.append(
            {
                "bin_index": float(i),
                "drift_bin_left_deg": left,
                "drift_bin_right_deg": right,
                "drift_bin_center_deg": float(np.mean(x_bin)),
                "count": float(len(y_bin)),
                "mean_error_px": mean_y,
                "std_error_px": std_y,
                "ci95_error_px": float(ci95),
            }
        )

    return rows, edges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate gaze error (px) with head deviation (deg)."
    )
    parser.add_argument(
        "--pose-data",
        type=Path,
        required=True,
        help="Path to head_pose_data.json",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        required=True,
        help="Path to static_evaluation_results.json or dynamic_evaluation_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plot/report",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of quantile bins for binned drift-error plot (default: 5)",
    )
    parser.add_argument(
        "--high-drift-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold for high-drift subgroup boxplot (default: 0.75)",
    )
    args = parser.parse_args()

    pose_path = args.pose_data
    eval_path = _resolve_eval_results_path(args.eval_results)
    output_dir = args.output_dir

    if not pose_path.exists():
        raise FileNotFoundError(f"Pose data not found: {pose_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_path}")
    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2")
    if not (0.0 < args.high_drift_quantile < 1.0):
        raise ValueError("--high-drift-quantile must be in (0, 1)")

    output_dir.mkdir(parents=True, exist_ok=True)

    pose_payload = _load_json(pose_path)
    eval_payload = _load_json(eval_path)

    pose_ts, pose_dev = _extract_pose_series(pose_payload)
    eval_rows = _extract_eval_rows(eval_payload)

    click_ts = np.array([r["timestamp_ms"] for r in eval_rows], dtype=float)
    error_px = np.array([r["error_px"] for r in eval_rows], dtype=float)

    click_ts_aligned = _normalize_click_timestamps(click_ts, pose_ts)
    nearest_idx = _nearest_indices(pose_ts, click_ts_aligned)

    matched_deviation = pose_dev[nearest_idx]
    timestamp_gap_ms = np.abs(click_ts_aligned - pose_ts[nearest_idx])

    valid_mask = np.isfinite(matched_deviation) & np.isfinite(error_px)
    x = matched_deviation[valid_mask]
    y = error_px[valid_mask]
    gaps = timestamp_gap_ms[valid_mask]

    if len(x) < 2:
        raise ValueError("Not enough aligned points for correlation (need at least 2).")

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        r_val = float("nan")
        p_val = float("nan")
        spearman_rho = float("nan")
        spearman_p = float("nan")
    else:
        pearson_values = np.asarray(pearsonr(x, y), dtype=float).reshape(-1)
        r_val = float(pearson_values[0])
        p_val = float(pearson_values[1])
        spearman_values = np.asarray(spearmanr(x, y), dtype=float).reshape(-1)
        spearman_rho = float(spearman_values[0])
        spearman_p = float(spearman_values[1])

    trend_slope = float("nan")
    trend_intercept = float("nan")
    if len(x) >= 2 and np.ptp(x) > 0:
        trend_slope, trend_intercept = np.polyfit(x, y, 1)

    participant_id = _extract_participant_id(pose_payload, eval_payload, eval_path)

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.75, edgecolor="black", linewidth=0.5)

    if np.isfinite(trend_slope) and np.isfinite(trend_intercept):
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = trend_slope * x_line + trend_intercept
        plt.plot(x_line, y_line, color="red", linewidth=2, label="Linear trend")
        plt.legend()

    plt.xlabel("Head Deviation (degrees)")
    plt.ylabel("Gaze Error (pixels)")
    plt.title("Head Deviation vs Gaze Error")
    plt.grid(True, alpha=0.3)

    info_text = "\n".join(
        [
            f"Participant: {participant_id}",
            f"n = {len(x)}",
            f"r = {_format_stat(r_val)}",
            f"p = {_format_stat(p_val)}",
        ]
    )
    plt.gca().text(
        0.02,
        0.98,
        info_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plot_path = output_dir / "correlation_acc_drift.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    # Plot 2: Hexbin density view
    hexbin_plot_path = output_dir / "correlation_acc_drift_hexbin.png"
    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(x, y, gridsize=22, mincnt=1, cmap="viridis")
    plt.colorbar(hb, label="Count")
    if np.isfinite(trend_slope) and np.isfinite(trend_intercept):
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = trend_slope * x_line + trend_intercept
        plt.plot(x_line, y_line, color="red", linewidth=2, label="Linear trend")
        plt.legend()
    plt.xlabel("Head Deviation (degrees)")
    plt.ylabel("Gaze Error (pixels)")
    plt.title("Head Deviation vs Gaze Error (Density)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(hexbin_plot_path, dpi=180)
    plt.close()

    # Plot 3: Binned drift-error trend with 95% CI
    binned_rows, _ = _compute_binned_stats(x, y, args.num_bins)
    binned_plot_path = output_dir / "correlation_acc_drift_binned.png"
    plt.figure(figsize=(10, 7))
    if binned_rows:
        bx = np.array([row["drift_bin_center_deg"] for row in binned_rows], dtype=float)
        by = np.array([row["mean_error_px"] for row in binned_rows], dtype=float)
        be = np.array([row["ci95_error_px"] for row in binned_rows], dtype=float)
        plt.errorbar(
            bx,
            by,
            yerr=be,
            fmt="o-",
            capsize=5,
            linewidth=2,
            markersize=7,
            label="Binned mean ±95% CI",
        )
        plt.legend()
    plt.xlabel("Head Deviation (degrees)")
    plt.ylabel("Gaze Error (pixels)")
    plt.title("Binned Error vs Head Deviation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(binned_plot_path, dpi=180)
    plt.close()

    # Plot 4: High vs low drift error distribution
    drift_threshold = float(np.quantile(x, args.high_drift_quantile))
    high_mask = x >= drift_threshold
    low_errors = y[~high_mask]
    high_errors = y[high_mask]

    box_plot_path = output_dir / "correlation_acc_drift_highlow_boxplot.png"
    plt.figure(figsize=(8, 7))
    plot_data = []
    labels = []
    if len(low_errors) > 0:
        plot_data.append(low_errors)
        labels.append(f"Low drift (n={len(low_errors)})")
    if len(high_errors) > 0:
        plot_data.append(high_errors)
        labels.append(f"High drift (n={len(high_errors)})")
    if plot_data:
        plt.boxplot(plot_data)
        plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.ylabel("Gaze Error (pixels)")
    plt.title(
        f"Error Distribution by Drift Group (q={args.high_drift_quantile:.2f}, threshold={drift_threshold:.3f}°)"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(box_plot_path, dpi=180)
    plt.close()

    # CSV outputs
    metrics_csv_path = output_dir / "correlation_metrics.csv"
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "evaluation_results_path",
                "num_points",
                "mean_timestamp_gap_ms",
                "median_timestamp_gap_ms",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
                "trend_slope_px_per_deg",
                "trend_intercept_px",
                "drift_threshold_deg",
                "high_drift_mean_error_px",
                "low_drift_mean_error_px",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "participant_id": participant_id,
                "evaluation_results_path": str(eval_path),
                "num_points": len(x),
                "mean_timestamp_gap_ms": float(np.mean(gaps)),
                "median_timestamp_gap_ms": float(np.median(gaps)),
                "pearson_r": r_val,
                "pearson_p": p_val,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "trend_slope_px_per_deg": trend_slope,
                "trend_intercept_px": trend_intercept,
                "drift_threshold_deg": drift_threshold,
                "high_drift_mean_error_px": float(np.mean(high_errors))
                if len(high_errors) > 0
                else np.nan,
                "low_drift_mean_error_px": float(np.mean(low_errors))
                if len(low_errors) > 0
                else np.nan,
            }
        )

    binned_csv_path = output_dir / "correlation_binned_stats.csv"
    with binned_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_index",
                "drift_bin_left_deg",
                "drift_bin_right_deg",
                "drift_bin_center_deg",
                "count",
                "mean_error_px",
                "std_error_px",
                "ci95_error_px",
            ],
        )
        writer.writeheader()
        for row in binned_rows:
            writer.writerow(row)

    summary_lines = [
        "Head Pose vs Gaze Error Correlation Summary",
        "=" * 48,
        f"Participant: {participant_id}",
        f"Pose data: {pose_path}",
        f"Evaluation results: {eval_path}",
        f"Total evaluation rows: {len(eval_rows)}",
        f"Aligned valid points: {len(x)}",
        f"Mean timestamp gap: {float(np.mean(gaps)):.3f} ms",
        f"Median timestamp gap: {float(np.median(gaps)):.3f} ms",
        f"Pearson r: {_format_stat(r_val)}",
        f"P-value: {_format_stat(p_val)}",
        f"Spearman rho: {_format_stat(spearman_rho)}",
        f"Spearman p-value: {_format_stat(spearman_p)}",
        f"Trend line: y = {trend_slope:.6f}x + {trend_intercept:.6f}",
        f"High-drift quantile threshold ({args.high_drift_quantile:.2f}): {drift_threshold:.6f}°",
        f"Low-drift mean error: {float(np.mean(low_errors)):.6f} px"
        if len(low_errors) > 0
        else "Low-drift mean error: nan",
        f"High-drift mean error: {float(np.mean(high_errors)):.6f} px"
        if len(high_errors) > 0
        else "High-drift mean error: nan",
        f"Plot (scatter): {plot_path}",
        f"Plot (hexbin): {hexbin_plot_path}",
        f"Plot (binned): {binned_plot_path}",
        f"Plot (high/low box): {box_plot_path}",
        f"CSV (metrics): {metrics_csv_path}",
        f"CSV (binned): {binned_csv_path}",
    ]

    report_path = output_dir / "correlation_summary.txt"
    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
