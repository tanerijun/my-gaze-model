#!/usr/bin/env python3
"""
===============================================================================
EVE Benchmark: 3D Feature Extraction & Tobii Ground-Truth Caching
===============================================================================

Description:
    Processes the raw frontal center webcam videos (webcam_c.mp4) from the
    EVE dataset (validation split: val01 through val05) using GLAMIA's
    pre-trained MobileOne-S1 3D gaze estimator and BlazeFace detector.

    Aligns predicted 3D gaze angles (pitch, yaw in degrees) with ground-truth
    Point-of-Gaze screen coordinates from the Tobii Pro Spectrum eye tracker
    stored in webcam_c.h5 (face_PoG_tobii).

    Supports incremental saving and resuming so processing can be safely
    interrupted and restarted.

Inputs:
    - EVE dataset path: datasets/extracted/EVE/
    - Pretrained weights: weights/prod.pth

Outputs:
    - Feature cache: experiments/eve/eve_features.csv
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add inference root to Python path
INFERENCE_ROOT = Path(__file__).resolve().parents[3]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from src.inference import GazePipeline3D

REPO_ROOT = Path(__file__).resolve().parents[5]


def parse_task_type(step_name: str) -> str:
    """Classifies stimulus step into 'image', 'video', or 'wikipedia'."""
    name_lower = step_name.lower()
    if "image" in name_lower:
        return "image"
    elif "video" in name_lower:
        return "video"
    elif "wikipedia" in name_lower:
        return "wikipedia"
    return "other"


def process_stimulus_step(
    step_dir: Path,
    participant_id: str,
    pipeline_3d: GazePipeline3D,
    stride: int = 1,
) -> list[dict]:
    """
    Processes a single stimulus step folder for one participant.
    Extracts frames from webcam_c.mp4 and matches them with webcam_c.h5 ground truth.
    """
    h5_path = step_dir / "webcam_c.h5"
    mp4_path = step_dir / "webcam_c.mp4"

    if not h5_path.exists() or not mp4_path.exists():
        return []

    # Read ground truth from HDF5
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "face_PoG_tobii" not in f:
                return []
            pog_group = f["face_PoG_tobii"]
            gt_pog_data = pog_group["data"][:]  # Shape (N, 2)
            gt_validity = pog_group["validity"][:]  # Shape (N,)

            # Read millimeter scaling if available
            mm_per_px = (
                f["millimeters_per_pixel"][:]
                if "millimeters_per_pixel" in f
                else np.array([0.288, 0.288])
            )
    except Exception as e:
        print(f"Warning: Failed to read {h5_path}: {e}")
        return []

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = min(total_frames, len(gt_pog_data))
    task_type = parse_task_type(step_dir.name)

    records = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            is_valid_tobii = bool(gt_validity[frame_idx])
            gt_x = float(gt_pog_data[frame_idx, 0])
            gt_y = float(gt_pog_data[frame_idx, 1])

            # Run GLAMIA's 3D Gaze Pipeline (BlazeFace + MobileOne-S1)
            results = pipeline_3d(frame)
            face_detected = len(results) > 0

            if face_detected:
                first_face = results[0]
                pred_pitch_deg = float(first_face["gaze"]["pitch"])
                pred_yaw_deg = float(first_face["gaze"]["yaw"])
                bbox = first_face["bbox"]
            else:
                pred_pitch_deg = np.nan
                pred_yaw_deg = np.nan
                bbox = [0, 0, 0, 0]

            records.append(
                {
                    "participant_id": participant_id,
                    "step_name": step_dir.name,
                    "task_type": task_type,
                    "frame_idx": frame_idx,
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                    "tobii_valid": is_valid_tobii,
                    "face_detected": face_detected,
                    "pred_pitch_deg": pred_pitch_deg,
                    "pred_yaw_deg": pred_yaw_deg,
                    "pred_pitch_rad": np.deg2rad(pred_pitch_deg)
                    if face_detected
                    else np.nan,
                    "pred_yaw_rad": np.deg2rad(pred_yaw_deg)
                    if face_detected
                    else np.nan,
                    "screen_width_px": 1920.0,
                    "screen_height_px": 1080.0,
                    "screen_diagonal_px": float(np.sqrt(1920.0**2 + 1080.0**2)),
                    "mm_per_px_x": float(mm_per_px[0]),
                    "mm_per_px_y": float(mm_per_px[1]),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                }
            )

        frame_idx += 1

    cap.release()
    return records


def get_completed_steps(output_csv: Path) -> set[tuple[str, str]]:
    """Returns set of (participant_id, step_name) already saved in CSV."""
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=["participant_id", "step_name"])
        unique_pairs = df.drop_duplicates()[["participant_id", "step_name"]].to_numpy()
        return {(str(p[0]), str(p[1])) for p in unique_pairs}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache 3D gaze features for EVE benchmark."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "extracted" / "EVE"),
        help="Path to extracted EVE dataset root.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(REPO_ROOT / "weights" / "prod.pth"),
        help="Path to MobileOne-S1 weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "eve"),
        help="Directory to save feature cache CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="eve_features.csv",
        help="Filename for feature cache CSV (default: eve_features.csv).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "all", "custom"],
        help="Dataset split to evaluate: 'val' (val01-05), 'train' (train01-39), 'all' (all 44 subjects), or 'custom'.",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=None,
        help="Custom list of participant folders (used if --split custom).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame sampling stride (default 1 = every frame; 2 = every 2nd frame).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for inference.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / args.output_file

    # Resolve participants based on split
    if args.split == "val":
        participants = [f"val{i:02d}" for i in range(1, 6)]
    elif args.split == "train":
        participants = [f"train{i:02d}" for i in range(1, 40)]
    elif args.split == "all":
        participants = [f"train{i:02d}" for i in range(1, 40)] + [
            f"val{i:02d}" for i in range(1, 6)
        ]
    elif args.split == "custom":
        participants = (
            args.participants
            if args.participants
            else [f"val{i:02d}" for i in range(1, 6)]
        )
    else:
        participants = [f"val{i:02d}" for i in range(1, 6)]
    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not weights_path.exists():
        print(f"Error: Weights file does not exist: {weights_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"EVE 3D Feature Extraction (Split: {args.split})")
    print(f"Dataset Dir:    {data_dir}")
    print(f"Weights:        {weights_path}")
    print(f"Output CSV:     {output_csv}")
    print(
        f"Participants:   {len(participants)} subjects ({participants[0]}..{participants[-1]})"
    )
    print(f"Frame Stride:   {args.stride}")
    print(f"Device:         {args.device}")
    print("=" * 80)

    # Initialize GazePipeline3D
    print("\nInitializing GazePipeline3D...")
    pipeline_3d = GazePipeline3D(
        weights_path=str(weights_path),
        device=args.device,
        smooth_facebbox=False,
        smooth_gaze=False,
        enable_landmarker_features=False,
    )

    completed_steps = get_completed_steps(output_csv)
    if completed_steps:
        print(
            f"Resuming: Found {len(completed_steps)} already processed steps in {output_csv.name}"
        )

    total_new_records = 0

    for pid in participants:
        p_dir = data_dir / pid
        if not p_dir.exists():
            print(f"Warning: Participant directory {p_dir} not found. Skipping.")
            continue
        step_dirs = sorted(
            [d for d in p_dir.iterdir() if d.is_dir() and d.name.startswith("step")]
        )
        print(f"\nProcessing {pid} ({len(step_dirs)} total stimulus steps):")

        for step_dir in tqdm(step_dirs, desc=f"  {pid}", leave=True):
            if (pid, step_dir.name) in completed_steps:
                continue

            records = process_stimulus_step(
                step_dir=step_dir,
                participant_id=pid,
                pipeline_3d=pipeline_3d,
                stride=args.stride,
            )

            if records:
                step_df = pd.DataFrame(records)
                # Append incrementally to CSV
                header = not output_csv.exists()
                step_df.to_csv(output_csv, mode="a", index=False, header=header)
                total_new_records += len(records)
                completed_steps.add((pid, step_dir.name))

    print("\n" + "=" * 80)
    print("EVE Feature Extraction Complete!")
    print(f"Features saved to: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
