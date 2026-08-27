#!/usr/bin/env python3
"""
===============================================================================
Stage 1: 3D Feature Extraction & Caching for MPIIFaceGaze Benchmark
===============================================================================

Description:
    Processes raw MPIIFaceGaze images using the pre-trained 3D Gaze Estimator
    (MobileOne-S1 trained on Gaze360) and MediaPipe BlazeFace face detector.

    Extracts predicted 3D gaze angles (pitch, yaw in degrees) and aligns them
    with ground-truth 2D screen coordinates, screen metadata, and recording day.

    The resulting feature cache enables instantaneous execution of downstream
    2D mapping and dynamic calibration experiments (Stage 2) without needing
    to re-run heavy neural network inference.

Inputs:
    - Raw dataset path: datasets/extracted/MPIIFaceGaze/
    - Pretrained weights: weights/prod.pth

Outputs:
    - Feature cache: experiments/mpiifacegaze/mpiifacegaze_features.csv
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

# Add inference root to Python path so we can import src.inference modules
INFERENCE_ROOT = Path(__file__).resolve().parents[3]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from src.inference import GazePipeline3D


def load_screen_metadata(calibration_dir: Path) -> dict[str, float]:
    """
    Loads screen dimensions and resolution metadata from screenSize.mat.

    Args:
        calibration_dir: Path to participant's Calibration/ directory.

    Returns:
        Dictionary with screen resolution (pixels) and physical size (mm).
    """
    screen_size_file = calibration_dir / "screenSize.mat"
    if not screen_size_file.exists():
        raise FileNotFoundError(f"Missing screen metadata: {screen_size_file}")

    mat = sio.loadmat(str(screen_size_file))

    width_px = float(mat["width_pixel"][0, 0])
    height_px = float(mat["height_pixel"][0, 0])
    width_mm = float(mat["width_mm"][0, 0])
    height_mm = float(mat["height_mm"][0, 0])

    diagonal_px = np.sqrt(width_px**2 + height_px**2)

    return {
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "diagonal_px": diagonal_px,
    }


def parse_participant_annotations(annotation_file: Path) -> list[dict]:
    """
    Parses MPIIFaceGaze ground-truth annotation file (e.g. p00.txt).

    File format per line:
    - Col 0: Image relative path (e.g. 'day01/0005.jpg')
    - Col 1: Ground-truth screen X coordinate (pixels)
    - Col 2: Ground-truth screen Y coordinate (pixels)
    - Col 3..14: 2D facial landmark positions
    - Col 15..20: 3D head pose (rotation & translation)
    - Col 21..23: 3D face center (fc)
    - Col 24..26: 3D gaze target (gt)
    - Col 27: Evaluation eye selection ('left' or 'right')
    """
    samples = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            image_rel_path = parts[0]
            gt_x = float(parts[1])
            gt_y = float(parts[2])

            # Extract recording day (e.g. 'day01' from 'day01/0005.jpg')
            day = image_rel_path.split("/")[0] if "/" in image_rel_path else "day01"

            samples.append(
                {
                    "image_rel_path": image_rel_path,
                    "day": day,
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                }
            )
    return samples


def extract_features_for_participant(
    participant_id: str,
    participant_dir: Path,
    pipeline_3d: GazePipeline3D,
) -> list[dict]:
    """
    Runs 3D gaze estimation on all images for a single participant.
    """
    annotation_file = participant_dir / f"{participant_id}.txt"
    calibration_dir = participant_dir / "Calibration"

    if not annotation_file.exists():
        print(f"Warning: Annotation file {annotation_file} not found. Skipping.")
        return []

    screen_meta = load_screen_metadata(calibration_dir)
    samples = parse_participant_annotations(annotation_file)

    records = []
    print(
        f"Processing {participant_id}: {len(samples)} frames (Screen: {int(screen_meta['width_px'])}x{int(screen_meta['height_px'])})"
    )

    for sample in tqdm(samples, desc=f"  {participant_id}", leave=False):
        img_path = participant_dir / sample["image_rel_path"]
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Run GLAMIA's 3D Gaze Pipeline (BlazeFace face detection + MobileOne-S1)
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
                "day": sample["day"],
                "image_rel_path": sample["image_rel_path"],
                "gt_x": sample["gt_x"],
                "gt_y": sample["gt_y"],
                "screen_width_px": screen_meta["width_px"],
                "screen_height_px": screen_meta["height_px"],
                "screen_diagonal_px": screen_meta["diagonal_px"],
                "screen_width_mm": screen_meta["width_mm"],
                "screen_height_mm": screen_meta["height_mm"],
                "face_detected": face_detected,
                "pred_pitch_deg": pred_pitch_deg,
                "pred_yaw_deg": pred_yaw_deg,
                "pred_pitch_rad": np.deg2rad(pred_pitch_deg)
                if face_detected
                else np.nan,
                "pred_yaw_rad": np.deg2rad(pred_yaw_deg) if face_detected else np.nan,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
            }
        )

    return records


REPO_ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache 3D gaze features for MPIIFaceGaze benchmark."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "extracted" / "MPIIFaceGaze"),
        help="Path to the extracted MPIIFaceGaze dataset root.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(REPO_ROOT / "weights" / "prod.pth"),
        help="Path to MobileOne-S1 pretrained weights (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "mpiifacegaze"),
        help="Directory to save the extracted feature cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for 3D model inference.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "mpiifacegaze_features.csv"

    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not weights_path.exists():
        print(f"Error: Weights file does not exist: {weights_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("Stage 1: MPIIFaceGaze 3D Feature Extraction")
    print(f"Dataset Dir: {data_dir}")
    print(f"Weights:     {weights_path}")
    print(f"Output CSV:  {output_csv}")
    print(f"Device:      {args.device}")
    print("=" * 80)

    # Initialize GazePipeline3D
    # Note: For static per-frame extraction, we disable temporal smoothing
    # to avoid state carryover between discontinuous frames across days.
    print("\nInitializing GazePipeline3D...")
    pipeline_3d = GazePipeline3D(
        weights_path=str(weights_path),
        device=args.device,
        smooth_facebbox=False,
        smooth_gaze=False,
        enable_landmarker_features=False,
    )

    # Discover participants (p00 through p14)
    participants = sorted(
        [
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("p") and (d / f"{d.name}.txt").exists()
        ]
    )

    if not participants:
        print(
            f"Error: No valid participant folders found in {data_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Found {len(participants)} participants: {', '.join(participants)}\n")

    all_records = []
    for pid in participants:
        p_dir = data_dir / pid
        records = extract_features_for_participant(pid, p_dir, pipeline_3d)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("Feature Extraction Complete!")
    print(f"Total samples processed: {len(df)}")
    print(f"Face detection success rate: {df['face_detected'].mean() * 100:.2f}%")
    print(f"Features saved to: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
