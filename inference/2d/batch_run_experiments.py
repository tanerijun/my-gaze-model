#!/usr/bin/env python3
"""
Batch runner for per-session experiment processing.

Features:
- Discovers session folders containing metadata.json, screen.mp4, webcam.mp4
- Runs static_evaluation + head_pose_analysis + linearity_analysis once per session
- Searches dynamic calibration buffer size (exhaustive or binary-like local search)
- Stores all outputs in a single output root with one subfolder per session
- Runs correlate_acc_drift.py and compare_static_dynamic_drift.py in comprehensive mode
- Saves aggregate CSV/JSON summary across sessions

Example:
python 2d/batch_run_experiments.py \
  --sessions-root collected_data \
  --weights weights/prod.pth \
  --output-root experiment_results/batch_run_01 \
  --device cpu \
  --search-strategy binary \
  --buffer-min 10 --buffer-max 150 --buffer-step 10 \
  --include-accumulate
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_SESSION_FILES = ("metadata.json", "screen.mp4", "webcam.mp4")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _get_nested(mapping: Any, keys: list[str]) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_single_row_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        return row
    except Exception:
        return None


def _session_dirs_from_root(sessions_root: Path) -> list[Path]:
    if not sessions_root.exists():
        raise FileNotFoundError(f"Sessions root not found: {sessions_root}")

    discovered: set[Path] = set()
    for meta_path in sessions_root.rglob("metadata.json"):
        candidate = meta_path.parent
        if all(
            (candidate / file_name).exists() for file_name in REQUIRED_SESSION_FILES
        ):
            discovered.add(candidate)

    return sorted(discovered)


def _buffer_suffix(buffer_size: int) -> str:
    if buffer_size == -1:
        return "accumulate"
    return f"buffer_{buffer_size}"


def _dynamic_results_path(output_dir: Path, buffer_size: int) -> Path:
    suffix = _buffer_suffix(buffer_size)
    return output_dir / f"dynamic_evaluation_results_BUF_{suffix}.json"


def _score_eval_results(results_json_path: Path, score_metric: str) -> float | None:
    if not results_json_path.exists():
        return None

    payload = _load_json(results_json_path)
    if not isinstance(payload, dict):
        return None

    eval_rows = payload.get("evaluation_results")
    if not isinstance(eval_rows, list) or not eval_rows:
        return None

    errors = [
        _to_float(row.get("error_px")) for row in eval_rows if isinstance(row, dict)
    ]
    arr = np.asarray([e for e in errors if e is not None], dtype=float)
    if arr.size == 0:
        return None

    if score_metric == "mean":
        return float(np.mean(arr))
    if score_metric == "median":
        return float(np.median(arr))
    if score_metric == "p95":
        return float(np.percentile(arr, 95))
    raise ValueError(f"Unknown score metric: {score_metric}")


def _eval_summary_from_results(
    results_json_path: Path,
) -> dict[str, float | int | None]:
    payload = _load_json(results_json_path)
    if not isinstance(payload, dict):
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "std_population": None,
            "std_sample": None,
        }

    eval_rows = payload.get("evaluation_results")
    if not isinstance(eval_rows, list):
        eval_rows = []

    errors = [
        _to_float(row.get("error_px")) for row in eval_rows if isinstance(row, dict)
    ]
    arr = np.asarray([e for e in errors if e is not None], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "std_population": None,
            "std_sample": None,
        }

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std_population": float(np.std(arr, ddof=0)),
        "std_sample": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
    }


def _run_exp_collector(
    script_path: Path,
    data_dir: Path,
    weights_path: Path,
    output_dir: Path,
    tasks: list[str],
    context_frames: int,
    buffer_size: int,
    device: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir",
        str(data_dir),
        "--weights",
        str(weights_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--context-frames",
        str(context_frames),
        "--buffer-size",
        str(buffer_size),
        "--tasks",
        *tasks,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _search_best_buffer(
    candidates: list[int],
    score_fn,
    strategy: str,
) -> tuple[int | None, dict[int, float | None]]:
    if not candidates:
        return None, {}

    tested: dict[int, float | None] = {}

    def get_score(buffer_value: int) -> float | None:
        if buffer_value in tested:
            return tested[buffer_value]
        tested[buffer_value] = score_fn(buffer_value)
        return tested[buffer_value]

    if strategy == "exhaustive" or len(candidates) <= 3:
        for value in candidates:
            get_score(value)
    else:
        lo = 0
        hi = len(candidates) - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            mid_score = get_score(candidates[mid])

            left_score = None
            if mid > lo:
                left_score = get_score(candidates[mid - 1])

            right_score = None
            if mid < hi:
                right_score = get_score(candidates[mid + 1])

            best_local = mid_score
            move_left = left_score is not None and (
                best_local is None or left_score < best_local
            )
            move_right = right_score is not None and (
                best_local is None or right_score < best_local
            )

            if not move_left and not move_right:
                break
            if move_left and (
                not move_right
                or (
                    left_score is not None
                    and right_score is not None
                    and left_score <= right_score
                )
            ):
                hi = mid - 1
            else:
                lo = mid + 1

        # Always evaluate boundaries for safety.
        get_score(candidates[0])
        get_score(candidates[-1])

    valid = [(k, v) for k, v in tested.items() if v is not None]
    if not valid:
        return None, tested

    best_buffer = min(valid, key=lambda kv: kv[1])[0]
    return best_buffer, tested


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run static/head-pose/dynamic-buffer experiments over many sessions."
    )
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument(
        "--exp-script",
        type=Path,
        default=Path(__file__).with_name("exp_data_collector.py"),
        help="Path to exp_data_collector.py",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--context-frames", type=int, default=5)

    parser.add_argument(
        "--search-strategy",
        choices=["exhaustive", "binary"],
        default="binary",
        help="Buffer search strategy. Binary is faster but assumes local smoothness.",
    )
    parser.add_argument(
        "--score-metric",
        choices=["mean", "median", "p95"],
        default="mean",
        help="Metric minimized when picking best dynamic buffer.",
    )

    parser.add_argument(
        "--buffer-values",
        type=int,
        nargs="+",
        default=None,
        help="Explicit buffer candidates. Example: 30 50 70 90",
    )
    parser.add_argument("--buffer-min", type=int, default=10)
    parser.add_argument("--buffer-max", type=int, default=150)
    parser.add_argument("--buffer-step", type=int, default=10)
    parser.add_argument(
        "--include-accumulate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include unlimited dynamic buffer mode (equivalent to --buffer-size -1, saved as 'accumulate').",
    )
    parser.add_argument(
        "--comprehensive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run full analysis bundle (correlation + robustness comparison) after static/dynamic/head-pose/linearity.",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun tasks even if expected output files already exist.",
    )
    parser.add_argument(
        "--include-correlation",
        action="store_true",
        help="Force-run correlate_acc_drift.py for static and best dynamic outputs.",
    )
    parser.add_argument(
        "--include-robustness-compare",
        action="store_true",
        help="Force-run compare_static_dynamic_drift.py using best dynamic buffer.",
    )

    args = parser.parse_args()

    sessions_root = args.sessions_root
    weights_path = args.weights
    output_root = args.output_root
    exp_script = args.exp_script

    if not exp_script.exists():
        raise FileNotFoundError(f"exp_data_collector.py not found: {exp_script}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    sessions = _session_dirs_from_root(sessions_root)
    if not sessions:
        raise ValueError(
            "No valid session directories found (requires metadata.json, screen.mp4, webcam.mp4)."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    if args.buffer_values:
        candidates = sorted({int(v) for v in args.buffer_values})
    else:
        if args.buffer_step <= 0:
            raise ValueError("--buffer-step must be > 0")
        if args.buffer_min > args.buffer_max:
            raise ValueError("--buffer-min cannot be greater than --buffer-max")
        candidates = list(range(args.buffer_min, args.buffer_max + 1, args.buffer_step))

    if args.include_accumulate:
        candidates = sorted(set(candidates + [-1]))

    corr_script = Path(__file__).with_name("correlate_acc_drift.py")
    compare_script = Path(__file__).with_name("compare_static_dynamic_drift.py")

    aggregate_rows: list[dict[str, Any]] = []

    print(f"Found {len(sessions)} sessions")
    print(f"Dynamic buffer candidates: {candidates}")

    run_correlation = args.comprehensive or args.include_correlation
    run_robustness_compare = args.comprehensive or args.include_robustness_compare

    if args.comprehensive:
        print("Comprehensive mode: enabled")
    print(f"Run correlation: {run_correlation}")
    print(f"Run static-vs-dynamic compare: {run_robustness_compare}")

    for index, session_dir in enumerate(sessions, start=1):
        session_id = session_dir.name
        session_out = output_root / session_id
        session_out.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"[{index}/{len(sessions)}] Session: {session_id}")
        print(f"Data dir: {session_dir}")
        print(f"Output dir: {session_out}")

        static_results_path = session_out / "static_evaluation_results.json"
        head_pose_stats_path = session_out / "head_pose_stats.json"
        head_pose_data_path = session_out / "head_pose_data.json"
        linearity_report_path = session_out / "linearity_report.txt"

        if args.force_rerun or not (
            static_results_path.exists()
            and head_pose_stats_path.exists()
            and linearity_report_path.exists()
        ):
            base_run = _run_exp_collector(
                script_path=exp_script,
                data_dir=session_dir,
                weights_path=weights_path,
                output_dir=session_out,
                tasks=["static_evaluation", "head_pose_analysis", "linearity_analysis"],
                context_frames=args.context_frames,
                buffer_size=90,
                device=args.device,
            )
            (session_out / "batch_base_stdout.log").write_text(
                base_run.stdout or "", encoding="utf-8"
            )
            (session_out / "batch_base_stderr.log").write_text(
                base_run.stderr or "", encoding="utf-8"
            )
            if base_run.returncode != 0:
                print(
                    f"Base run failed (return code {base_run.returncode}). Skipping session."
                )
                aggregate_rows.append(
                    {
                        "session_id": session_id,
                        "status": "base_run_failed",
                        "best_buffer": None,
                    }
                )
                continue
        else:
            print("Reusing existing static/head-pose/linearity outputs")

        def dynamic_score(buffer_value: int) -> float | None:
            dynamic_path = _dynamic_results_path(session_out, buffer_value)
            if args.force_rerun or not dynamic_path.exists():
                run = _run_exp_collector(
                    script_path=exp_script,
                    data_dir=session_dir,
                    weights_path=weights_path,
                    output_dir=session_out,
                    tasks=["dynamic_evaluation"],
                    context_frames=args.context_frames,
                    buffer_size=buffer_value,
                    device=args.device,
                )
                suffix = _buffer_suffix(buffer_value)
                (session_out / f"batch_dynamic_{suffix}_stdout.log").write_text(
                    run.stdout or "", encoding="utf-8"
                )
                (session_out / f"batch_dynamic_{suffix}_stderr.log").write_text(
                    run.stderr or "", encoding="utf-8"
                )
                if run.returncode != 0:
                    print(f"  Buffer {buffer_value}: run failed")
                    return None

            score = _score_eval_results(dynamic_path, args.score_metric)
            score_text = "nan" if score is None else f"{score:.4f}"
            print(f"  Buffer {buffer_value}: {args.score_metric}={score_text}")
            return score

        tested_scores: dict[int, float | None] = {}

        if -1 in candidates:
            print("  Testing accumulate mode first...")
            tested_scores[-1] = dynamic_score(-1)

        finite_candidates = [c for c in candidates if c != -1]
        if finite_candidates:
            best_buffer_finite, finite_scores = _search_best_buffer(
                candidates=finite_candidates,
                score_fn=dynamic_score,
                strategy=args.search_strategy,
            )
            tested_scores.update(finite_scores)
        else:
            best_buffer_finite = None

        valid_scores = [(k, v) for k, v in tested_scores.items() if v is not None]
        best_buffer = (
            min(valid_scores, key=lambda kv: kv[1])[0] if valid_scores else None
        )

        if best_buffer is None:
            print("No valid dynamic result for this session.")
            aggregate_rows.append(
                {
                    "session_id": session_id,
                    "status": "dynamic_failed",
                    "best_buffer": None,
                }
            )
            continue

        static_summary = _eval_summary_from_results(static_results_path)
        best_dynamic_path = _dynamic_results_path(session_out, best_buffer)
        dynamic_summary = _eval_summary_from_results(best_dynamic_path)

        static_mean = _to_float(static_summary.get("mean"))
        dynamic_mean = _to_float(dynamic_summary.get("mean"))
        static_p95 = _to_float(static_summary.get("p95"))
        dynamic_p95 = _to_float(dynamic_summary.get("p95"))

        mean_improvement = (
            static_mean - dynamic_mean
            if static_mean is not None and dynamic_mean is not None
            else None
        )
        p95_improvement = (
            static_p95 - dynamic_p95
            if static_p95 is not None and dynamic_p95 is not None
            else None
        )

        drift_rate = None
        headpose_detection_rate = None
        headpose_game_mean_dev = None
        headpose_game_p95_dev = None
        headpose_game_std_sample = None
        headpose_pitch_p95_dev = None
        headpose_yaw_p95_dev = None
        headpose_roll_p95_dev = None
        headpose_total_velocity_deg_sec = None

        if head_pose_stats_path.exists():
            hp_stats = _load_json(head_pose_stats_path)
            if isinstance(hp_stats, dict):
                headpose_detection_rate = _to_float(hp_stats.get("detection_rate"))
                game_phase = hp_stats.get("game_phase")
                if isinstance(game_phase, dict):
                    drift_rate = _to_float(game_phase.get("drift_rate_deg_min"))
                    headpose_game_mean_dev = _to_float(game_phase.get("mean_deviation"))
                    headpose_game_p95_dev = _to_float(game_phase.get("p95_deviation"))
                    headpose_game_std_sample = _to_float(
                        game_phase.get("std_deviation_sample")
                    )
                    headpose_pitch_p95_dev = _to_float(
                        _get_nested(
                            game_phase, ["per_axis_abs_deviation", "pitch", "p95"]
                        )
                    )
                    headpose_yaw_p95_dev = _to_float(
                        _get_nested(
                            game_phase, ["per_axis_abs_deviation", "yaw", "p95"]
                        )
                    )
                    headpose_roll_p95_dev = _to_float(
                        _get_nested(
                            game_phase, ["per_axis_abs_deviation", "roll", "p95"]
                        )
                    )
                    headpose_total_velocity_deg_sec = _to_float(
                        _get_nested(
                            game_phase,
                            [
                                "stability",
                                "total_deviation_mean_abs_velocity_deg_per_sec",
                            ],
                        )
                    )

        corr_static_metrics = _load_single_row_csv(
            session_out / "correlation_static" / "correlation_metrics.csv"
        )
        corr_dynamic_metrics = _load_single_row_csv(
            session_out / "correlation_dynamic_best" / "correlation_metrics.csv"
        )

        corr_static_pearson_r = _to_float(
            corr_static_metrics.get("pearson_r") if corr_static_metrics else None
        )
        corr_static_pearson_p = _to_float(
            corr_static_metrics.get("pearson_p") if corr_static_metrics else None
        )
        corr_static_spearman_rho = _to_float(
            corr_static_metrics.get("spearman_rho") if corr_static_metrics else None
        )
        corr_static_spearman_p = _to_float(
            corr_static_metrics.get("spearman_p") if corr_static_metrics else None
        )
        corr_static_slope = _to_float(
            corr_static_metrics.get("trend_slope_px_per_deg")
            if corr_static_metrics
            else None
        )

        corr_dynamic_pearson_r = _to_float(
            corr_dynamic_metrics.get("pearson_r") if corr_dynamic_metrics else None
        )
        corr_dynamic_pearson_p = _to_float(
            corr_dynamic_metrics.get("pearson_p") if corr_dynamic_metrics else None
        )
        corr_dynamic_spearman_rho = _to_float(
            corr_dynamic_metrics.get("spearman_rho") if corr_dynamic_metrics else None
        )
        corr_dynamic_spearman_p = _to_float(
            corr_dynamic_metrics.get("spearman_p") if corr_dynamic_metrics else None
        )
        corr_dynamic_slope = _to_float(
            corr_dynamic_metrics.get("trend_slope_px_per_deg")
            if corr_dynamic_metrics
            else None
        )

        compare_summary = _load_json_if_exists(
            session_out / "static_dynamic_drift_summary.json"
        )
        compare_overall_improve_px = _to_float(
            _get_nested(
                compare_summary, ["improvement", "overall_mean_error_improvement_px"]
            )
        )
        compare_high_drift_improve_px = _to_float(
            _get_nested(
                compare_summary, ["improvement", "high_drift_mean_error_improvement_px"]
            )
        )
        compare_positive_rate = _to_float(
            _get_nested(compare_summary, ["improvement", "positive_improvement_rate"])
        )
        compare_high_drift_threshold = _to_float(
            _get_nested(compare_summary, ["drift", "high_drift_threshold_deg"])
        )

        if run_correlation and corr_script.exists() and head_pose_data_path.exists():
            corr_static_out = session_out / "correlation_static"
            corr_static_out.mkdir(parents=True, exist_ok=True)
            corr_static_run = subprocess.run(
                [
                    sys.executable,
                    str(corr_script),
                    "--pose-data",
                    str(head_pose_data_path),
                    "--eval-results",
                    str(static_results_path),
                    "--output-dir",
                    str(corr_static_out),
                ],
                capture_output=True,
                text=True,
            )
            (session_out / "batch_correlation_static_stdout.log").write_text(
                corr_static_run.stdout or "", encoding="utf-8"
            )
            (session_out / "batch_correlation_static_stderr.log").write_text(
                corr_static_run.stderr or "", encoding="utf-8"
            )
            if corr_static_run.returncode != 0:
                print("Warning: static correlation run failed")

            corr_dynamic_out = session_out / "correlation_dynamic_best"
            corr_dynamic_out.mkdir(parents=True, exist_ok=True)
            corr_dynamic_run = subprocess.run(
                [
                    sys.executable,
                    str(corr_script),
                    "--pose-data",
                    str(head_pose_data_path),
                    "--eval-results",
                    str(best_dynamic_path),
                    "--output-dir",
                    str(corr_dynamic_out),
                ],
                capture_output=True,
                text=True,
            )
            (session_out / "batch_correlation_dynamic_stdout.log").write_text(
                corr_dynamic_run.stdout or "", encoding="utf-8"
            )
            (session_out / "batch_correlation_dynamic_stderr.log").write_text(
                corr_dynamic_run.stderr or "", encoding="utf-8"
            )
            if corr_dynamic_run.returncode != 0:
                print("Warning: dynamic correlation run failed")

        if (
            run_robustness_compare
            and compare_script.exists()
            and head_pose_data_path.exists()
        ):
            compare_run = subprocess.run(
                [
                    sys.executable,
                    str(compare_script),
                    "--pose-data",
                    str(head_pose_data_path),
                    "--static-results",
                    str(static_results_path),
                    "--dynamic-results",
                    str(best_dynamic_path),
                    "--output-dir",
                    str(session_out),
                ],
                capture_output=True,
                text=True,
            )
            (session_out / "batch_compare_stdout.log").write_text(
                compare_run.stdout or "", encoding="utf-8"
            )
            (session_out / "batch_compare_stderr.log").write_text(
                compare_run.stderr or "", encoding="utf-8"
            )
            if compare_run.returncode != 0:
                print("Warning: static-vs-dynamic comparison run failed")

        search_trace_path = session_out / "dynamic_buffer_search_trace.json"
        search_trace_payload = {
            "strategy": args.search_strategy,
            "metric": args.score_metric,
            "candidates": candidates,
            "accumulate_tested_first": -1 in candidates,
            "best_buffer_finite": best_buffer_finite,
            "tested_scores": tested_scores,
            "best_buffer": best_buffer,
            "best_score": tested_scores.get(best_buffer),
        }
        search_trace_path.write_text(
            json.dumps(search_trace_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            "Best buffer: "
            f"{best_buffer} ({args.score_metric}={tested_scores.get(best_buffer):.4f})"
            if tested_scores.get(best_buffer) is not None
            else f"Best buffer: {best_buffer}"
        )

        aggregate_rows.append(
            {
                "session_id": session_id,
                "status": "ok",
                "best_buffer": best_buffer,
                "best_dynamic_score": tested_scores.get(best_buffer),
                "drift_rate_deg_min": drift_rate,
                "headpose_detection_rate": headpose_detection_rate,
                "headpose_game_mean_deviation": headpose_game_mean_dev,
                "headpose_game_p95_deviation": headpose_game_p95_dev,
                "headpose_game_std_sample": headpose_game_std_sample,
                "headpose_pitch_p95_deviation": headpose_pitch_p95_dev,
                "headpose_yaw_p95_deviation": headpose_yaw_p95_dev,
                "headpose_roll_p95_deviation": headpose_roll_p95_dev,
                "headpose_total_velocity_deg_sec": headpose_total_velocity_deg_sec,
                "static_n": static_summary.get("n"),
                "static_mean_error_px": static_summary.get("mean"),
                "static_p95_error_px": static_summary.get("p95"),
                "dynamic_n": dynamic_summary.get("n"),
                "dynamic_mean_error_px": dynamic_summary.get("mean"),
                "dynamic_p95_error_px": dynamic_summary.get("p95"),
                "mean_improvement_px": mean_improvement,
                "p95_improvement_px": p95_improvement,
                "corr_static_pearson_r": corr_static_pearson_r,
                "corr_static_pearson_p": corr_static_pearson_p,
                "corr_static_spearman_rho": corr_static_spearman_rho,
                "corr_static_spearman_p": corr_static_spearman_p,
                "corr_static_slope_px_per_deg": corr_static_slope,
                "corr_dynamic_pearson_r": corr_dynamic_pearson_r,
                "corr_dynamic_pearson_p": corr_dynamic_pearson_p,
                "corr_dynamic_spearman_rho": corr_dynamic_spearman_rho,
                "corr_dynamic_spearman_p": corr_dynamic_spearman_p,
                "corr_dynamic_slope_px_per_deg": corr_dynamic_slope,
                "compare_overall_mean_improvement_px": compare_overall_improve_px,
                "compare_high_drift_mean_improvement_px": compare_high_drift_improve_px,
                "compare_positive_improvement_rate": compare_positive_rate,
                "compare_high_drift_threshold_deg": compare_high_drift_threshold,
                "session_output_dir": str(session_out),
            }
        )

    aggregate_json = output_root / "batch_summary.json"
    aggregate_json.write_text(
        json.dumps(aggregate_rows, indent=2) + "\n", encoding="utf-8"
    )

    aggregate_csv = output_root / "batch_summary.csv"
    csv_fields = [
        "session_id",
        "status",
        "best_buffer",
        "best_dynamic_score",
        "drift_rate_deg_min",
        "headpose_detection_rate",
        "headpose_game_mean_deviation",
        "headpose_game_p95_deviation",
        "headpose_game_std_sample",
        "headpose_pitch_p95_deviation",
        "headpose_yaw_p95_deviation",
        "headpose_roll_p95_deviation",
        "headpose_total_velocity_deg_sec",
        "static_n",
        "static_mean_error_px",
        "static_p95_error_px",
        "dynamic_n",
        "dynamic_mean_error_px",
        "dynamic_p95_error_px",
        "mean_improvement_px",
        "p95_improvement_px",
        "corr_static_pearson_r",
        "corr_static_pearson_p",
        "corr_static_spearman_rho",
        "corr_static_spearman_p",
        "corr_static_slope_px_per_deg",
        "corr_dynamic_pearson_r",
        "corr_dynamic_pearson_p",
        "corr_dynamic_spearman_rho",
        "corr_dynamic_spearman_p",
        "corr_dynamic_slope_px_per_deg",
        "compare_overall_mean_improvement_px",
        "compare_high_drift_mean_improvement_px",
        "compare_positive_improvement_rate",
        "compare_high_drift_threshold_deg",
        "session_output_dir",
    ]
    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({k: _fmt(row.get(k)) for k in csv_fields})

    ok_count = sum(1 for r in aggregate_rows if r.get("status") == "ok")
    print("\n" + "=" * 72)
    print(f"Completed batch run: {ok_count}/{len(aggregate_rows)} sessions successful")
    print(f"Aggregate JSON: {aggregate_json}")
    print(f"Aggregate CSV:  {aggregate_csv}")


if __name__ == "__main__":
    main()
