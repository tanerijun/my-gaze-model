#!/usr/bin/env python3
"""
Compare static vs dynamic calibration robustness against head pose drift.

Example:
python 2d/compare_static_dynamic_drift.py \
  --pose-data experiment_results/sample1/head_pose_data.json \
  --static-results experiment_results/sample1/static_evaluation_results.json \
  --dynamic-results experiment_results/sample1/dynamic_evaluation_results_BUF_buffer_90.json \
  --output-dir experiment_results/sample1
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _extract_pose_series(payload: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(payload, list):
        raise ValueError("Head pose data must be a JSON array.")

    rows: list[tuple[float, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        ts = _to_float(item.get("timestamp_ms"))
        dev = _to_float(item.get("total_deviation"))
        if ts is None or dev is None:
            continue
        rows.append((ts, dev))

    if not rows:
        raise ValueError("No valid pose entries with timestamp_ms and total_deviation.")

    rows.sort(key=lambda x: x[0])
    return (
        np.asarray([r[0] for r in rows], dtype=float),
        np.asarray([r[1] for r in rows], dtype=float),
    )


def _extract_eval_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        results = payload.get("evaluation_results")
        if results is None:
            results = payload.get("results")
    elif isinstance(payload, list):
        results = payload
    else:
        results = None

    if not isinstance(results, list):
        raise ValueError(
            "Evaluation payload must be list or dict with 'evaluation_results'."
        )

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            continue

        ts = None
        for key in ("timestamp_ms", "videoTimestamp", "timestamp"):
            ts = _to_float(item.get(key))
            if ts is not None:
                break

        err = None
        for key in ("error_px", "mean_error_px", "median_error_px"):
            err = _to_float(item.get(key))
            if err is not None:
                break

        if ts is None or err is None:
            continue

        click_id = item.get("click_id")
        if click_id is None:
            click_id = item.get("id")

        rows.append(
            {
                "index": i,
                "click_id": str(click_id) if click_id is not None else None,
                "timestamp_ms": ts,
                "error_px": err,
            }
        )

    if not rows:
        raise ValueError("No valid evaluation rows with timestamp and error.")

    return rows


def _nearest_indices(reference_sorted: np.ndarray, queries: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(reference_sorted, queries, side="left")
    idx = np.clip(idx, 0, len(reference_sorted) - 1)
    prev_idx = np.clip(idx - 1, 0, len(reference_sorted) - 1)
    choose_prev = np.abs(queries - reference_sorted[prev_idx]) <= np.abs(
        reference_sorted[idx] - queries
    )
    return np.where(choose_prev, prev_idx, idx)


def _align_timeline(click_ts: np.ndarray, pose_ts: np.ndarray) -> np.ndarray:
    click_min = float(np.min(click_ts))
    click_max = float(np.max(click_ts))
    pose_min = float(np.min(pose_ts))
    pose_max = float(np.max(pose_ts))

    has_overlap = not (click_max < pose_min or click_min > pose_max)
    if has_overlap:
        return click_ts

    return click_ts - click_min + pose_min


def _match_eval_to_pose(
    eval_rows: list[dict[str, Any]], pose_ts: np.ndarray, pose_dev: np.ndarray
) -> list[dict[str, Any]]:
    click_ts = np.asarray([float(r["timestamp_ms"]) for r in eval_rows], dtype=float)
    aligned_ts = _align_timeline(click_ts, pose_ts)
    idx = _nearest_indices(pose_ts, aligned_ts)

    matched: list[dict[str, Any]] = []
    for i, row in enumerate(eval_rows):
        d = float(pose_dev[idx[i]])
        gap = float(abs(aligned_ts[i] - pose_ts[idx[i]]))
        matched.append(
            {
                **row,
                "aligned_timestamp_ms": float(aligned_ts[i]),
                "head_deviation_deg": d,
                "timestamp_gap_ms": gap,
            }
        )

    return matched


def _safe_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "std_population": None,
            "std_sample": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_population": float(np.std(values, ddof=0)),
        "std_sample": float(np.std(values, ddof=1)) if values.size > 1 else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _build_key(row: dict[str, Any]) -> str:
    click_id = row.get("click_id")
    if click_id is not None:
        return f"id:{click_id}"
    return f"ts:{float(row['timestamp_ms']):.3f}"


def _extract_participant_id(*payloads: Any, fallback: str) -> str:
    candidates: list[Any] = []

    for payload in payloads:
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

    return fallback


def _compute_error_binned_curve(
    drift: np.ndarray, error: np.ndarray, n_bins: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    if drift.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if np.ptp(drift) == 0:
        return np.array([float(drift[0])]), np.array([float(np.mean(error))])

    edges = np.quantile(drift, np.linspace(0, 1, n_bins + 1))
    centers: list[float] = []
    means: list[float] = []

    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (drift >= lo) & (drift <= hi)
        else:
            mask = (drift >= lo) & (drift < hi)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(drift[mask])))
        means.append(float(np.mean(error[mask])))

    return np.asarray(centers, dtype=float), np.asarray(means, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare static vs dynamic gaze accuracy under head-pose drift."
    )
    parser.add_argument("--pose-data", type=Path, required=True)
    parser.add_argument("--static-results", type=Path, required=True)
    parser.add_argument("--dynamic-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--high-drift-quantile",
        type=float,
        default=0.75,
        help="Quantile to define high-drift subset (default: 0.75)",
    )
    args = parser.parse_args()

    if not (0.0 < args.high_drift_quantile < 1.0):
        raise ValueError("--high-drift-quantile must be in (0, 1).")

    for path in (args.pose_data, args.static_results, args.dynamic_results):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pose_payload = _load_json(args.pose_data)
    static_payload = _load_json(args.static_results)
    dynamic_payload = _load_json(args.dynamic_results)

    pose_ts, pose_dev = _extract_pose_series(pose_payload)
    static_rows = _extract_eval_rows(static_payload)
    dynamic_rows = _extract_eval_rows(dynamic_payload)

    static_matched = _match_eval_to_pose(static_rows, pose_ts, pose_dev)
    dynamic_matched = _match_eval_to_pose(dynamic_rows, pose_ts, pose_dev)

    static_err = np.asarray([float(r["error_px"]) for r in static_matched], dtype=float)
    static_drift = np.asarray(
        [float(r["head_deviation_deg"]) for r in static_matched], dtype=float
    )
    dynamic_err = np.asarray(
        [float(r["error_px"]) for r in dynamic_matched], dtype=float
    )
    dynamic_drift = np.asarray(
        [float(r["head_deviation_deg"]) for r in dynamic_matched], dtype=float
    )

    static_map = {_build_key(r): r for r in static_matched}
    dynamic_map = {_build_key(r): r for r in dynamic_matched}
    common_keys = sorted(set(static_map.keys()) & set(dynamic_map.keys()))

    paired_static_err: list[float] = []
    paired_dynamic_err: list[float] = []
    paired_drift: list[float] = []

    for key in common_keys:
        s = static_map[key]
        d = dynamic_map[key]
        s_err = _to_float(s.get("error_px"))
        d_err = _to_float(d.get("error_px"))
        s_dev = _to_float(s.get("head_deviation_deg"))
        d_dev = _to_float(d.get("head_deviation_deg"))
        if s_err is None or d_err is None or s_dev is None or d_dev is None:
            continue

        paired_static_err.append(s_err)
        paired_dynamic_err.append(d_err)
        paired_drift.append((s_dev + d_dev) / 2.0)

    paired_static_arr = np.asarray(paired_static_err, dtype=float)
    paired_dynamic_arr = np.asarray(paired_dynamic_err, dtype=float)
    paired_drift_arr = np.asarray(paired_drift, dtype=float)

    if paired_drift_arr.size == 0:
        drift_threshold = float("nan")
        high_mask = np.zeros(0, dtype=bool)
    else:
        drift_threshold = float(np.quantile(paired_drift_arr, args.high_drift_quantile))
        high_mask = paired_drift_arr >= drift_threshold

    delta_err = paired_static_arr - paired_dynamic_arr

    overall_improvement_mean = float(np.mean(delta_err)) if delta_err.size > 0 else None
    overall_improvement_pct = (
        float(np.mean(delta_err / paired_static_arr) * 100.0)
        if delta_err.size > 0 and np.all(paired_static_arr != 0)
        else None
    )

    high_delta = (
        delta_err[high_mask] if delta_err.size > 0 else np.asarray([], dtype=float)
    )
    high_improvement_mean = float(np.mean(high_delta)) if high_delta.size > 0 else None
    high_improvement_pct = (
        float(np.mean(high_delta / paired_static_arr[high_mask]) * 100.0)
        if high_delta.size > 0 and np.all(paired_static_arr[high_mask] != 0)
        else None
    )

    participant_id = _extract_participant_id(
        static_payload,
        dynamic_payload,
        fallback=args.output_dir.name,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].scatter(static_drift, static_err, alpha=0.65, label="Static", s=28)
    axes[0].scatter(dynamic_drift, dynamic_err, alpha=0.65, label="Dynamic", s=28)

    if static_drift.size >= 2 and np.ptp(static_drift) > 0:
        sx = np.linspace(float(np.min(static_drift)), float(np.max(static_drift)), 100)
        sslope, sint = np.polyfit(static_drift, static_err, 1)
        axes[0].plot(sx, sslope * sx + sint, "--", linewidth=2)

    if dynamic_drift.size >= 2 and np.ptp(dynamic_drift) > 0:
        dx = np.linspace(
            float(np.min(dynamic_drift)), float(np.max(dynamic_drift)), 100
        )
        dslope, dint = np.polyfit(dynamic_drift, dynamic_err, 1)
        axes[0].plot(dx, dslope * dx + dint, "--", linewidth=2)

    axes[0].set_xlabel("Head Deviation (degrees)")
    axes[0].set_ylabel("Gaze Error (pixels)")
    axes[0].set_title("Error vs Drift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    static_x, static_y = _compute_error_binned_curve(static_drift, static_err)
    dynamic_x, dynamic_y = _compute_error_binned_curve(dynamic_drift, dynamic_err)

    if static_x.size > 0:
        axes[1].plot(static_x, static_y, marker="o", label="Static")
    if dynamic_x.size > 0:
        axes[1].plot(dynamic_x, dynamic_y, marker="o", label="Dynamic")

    if np.isfinite(drift_threshold):
        axes[1].axvline(
            drift_threshold,
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"High-drift threshold ({args.high_drift_quantile:.2f})",
        )

    axes[1].set_xlabel("Head Deviation (degrees)")
    axes[1].set_ylabel("Mean Gaze Error (pixels)")
    axes[1].set_title("Binned Error by Drift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Static vs Dynamic Robustness — Participant: {participant_id}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)

    plot_path = args.output_dir / "static_dynamic_drift_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "participant_id": participant_id,
        "inputs": {
            "pose_data": str(args.pose_data),
            "static_results": str(args.static_results),
            "dynamic_results": str(args.dynamic_results),
        },
        "counts": {
            "static_points": int(static_err.size),
            "dynamic_points": int(dynamic_err.size),
            "paired_points": int(delta_err.size),
            "high_drift_paired_points": int(high_delta.size),
        },
        "drift": {
            "high_drift_quantile": float(args.high_drift_quantile),
            "high_drift_threshold_deg": drift_threshold,
            "static_head_deviation": _safe_stats(static_drift),
            "dynamic_head_deviation": _safe_stats(dynamic_drift),
            "paired_head_deviation": _safe_stats(paired_drift_arr),
        },
        "errors": {
            "static_error_px": _safe_stats(static_err),
            "dynamic_error_px": _safe_stats(dynamic_err),
            "paired_delta_error_px_static_minus_dynamic": _safe_stats(delta_err),
            "high_drift_delta_error_px_static_minus_dynamic": _safe_stats(high_delta),
        },
        "improvement": {
            "overall_mean_error_improvement_px": overall_improvement_mean,
            "overall_mean_error_improvement_pct": overall_improvement_pct,
            "high_drift_mean_error_improvement_px": high_improvement_mean,
            "high_drift_mean_error_improvement_pct": high_improvement_pct,
            "positive_improvement_rate": float(np.mean(delta_err > 0))
            if delta_err.size > 0
            else None,
        },
        "outputs": {
            "plot": str(plot_path),
        },
    }

    summary_json_path = args.output_dir / "static_dynamic_drift_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "STATIC VS DYNAMIC ROBUSTNESS SUMMARY",
        "=" * 48,
        f"Participant: {participant_id}",
        f"Static points: {static_err.size}",
        f"Dynamic points: {dynamic_err.size}",
        f"Paired points: {delta_err.size}",
        f"High-drift threshold (deg): {drift_threshold:.6f}"
        if np.isfinite(drift_threshold)
        else "High-drift threshold (deg): nan",
        f"High-drift paired points: {high_delta.size}",
        "",
        f"Static mean error: {float(np.mean(static_err)):.4f} px"
        if static_err.size
        else "Static mean error: nan",
        f"Dynamic mean error: {float(np.mean(dynamic_err)):.4f} px"
        if dynamic_err.size
        else "Dynamic mean error: nan",
        f"Overall mean improvement (static-dynamic): {overall_improvement_mean:.4f} px"
        if overall_improvement_mean is not None
        else "Overall mean improvement (static-dynamic): nan",
        f"High-drift mean improvement (static-dynamic): {high_improvement_mean:.4f} px"
        if high_improvement_mean is not None
        else "High-drift mean improvement (static-dynamic): nan",
        f"Plot: {plot_path}",
        f"JSON Summary: {summary_json_path}",
    ]

    summary_txt_path = args.output_dir / "static_dynamic_drift_summary.txt"
    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"TXT Summary: {summary_txt_path}")


if __name__ == "__main__":
    main()
