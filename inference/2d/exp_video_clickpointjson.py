"""
Base script for running experiment using dataset from dataset collection platform. Modify as needed.
Expected data: a video and a JSON file containing click timestamp & ground truth (for both calibration and test).
Always save the modified version in /experiments along with experiment data for archive.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.inference import GazePipeline2D, GazePipeline3D, Mapper

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)


def process_video_for_features(video_path: str, pipeline: GazePipeline3D) -> List[Dict]:
    """
    Processes a video frame-by-frame to extract gaze and head pose features.

    Args:
        video_path: Path to the video file.
        pipeline: Initialized GazePipeline3D instance.

    Returns:
        A list of dictionaries, where each dictionary contains the
        frame index and extracted features for a single frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {os.path.basename(video_path)} ({frame_count} frames)")

    all_frame_features = []
    frame_idx = 0
    with tqdm(total=frame_count, unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results_3d = pipeline(frame)

            # Use the first detected person's data
            if results_3d:
                all_frame_features.append({"frame_idx": frame_idx, **results_3d[0]})

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return all_frame_features


def prepare_calibration_data(
    mapper: Mapper,
    pipeline_2d: GazePipeline2D,
    all_frame_features: List[Dict],
    click_data_path: str,
    fps: float,
    frame_window: int = 5,
):
    """
    Populates the Mapper by extracting configured features from the 3D pipeline results.
    """
    with open(click_data_path, "r") as f:
        click_data = json.load(f)

    features_by_frame = {f["frame_idx"]: f for f in all_frame_features}
    total_points = 0

    for click_event in click_data:
        click_time_sec = click_event["timestamp"] / 1000
        click_frame = int(click_time_sec * fps)
        start_frame, end_frame = click_frame - frame_window, click_frame + frame_window

        vectors_for_point = []
        for i in range(start_frame, end_frame + 1):
            if i in features_by_frame:
                try:
                    # Use the pipeline's public utility to ensure consistency
                    feature_vector = pipeline_2d.extract_feature_vector(
                        features_by_frame[i]
                    )
                    vectors_for_point.append(feature_vector)
                except ValueError:
                    pass  # Skip frames with missing features

        if not vectors_for_point:
            continue

        target_coords = (
            click_event["coordinates"]["x"],
            click_event["coordinates"]["y"],
        )
        mapper.add_calibration_point(vectors_for_point, target_coords)
        total_points += 1

    print(f"Populated mapper with data from {total_points} calibration points.")


def prepare_test_data(
    click_data_path: str,
    fps: float,
    frame_window: int = 20,
) -> Tuple[List[int], np.ndarray]:
    """
    Identifies the frame indices to be tested and their corresponding ground truth labels.

    Args:
        click_data_path: Path to the JSON file with click data.
        fps: Frames per second of the video.
        frame_window: Number of frames to take before and after the click frame.

    Returns:
        A tuple of (test_frame_indices, y_test).
    """
    with open(click_data_path, "r") as f:
        click_data = json.load(f)

    test_frame_indices = []
    y_test = []
    for click_event in click_data:
        click_time_sec = click_event["timestamp"] / 1000
        click_frame = int(click_time_sec * fps)

        start_frame = click_frame - frame_window
        end_frame = click_frame + frame_window

        gt_coords = [
            click_event["coordinates"]["x"],
            click_event["coordinates"]["y"],
        ]

        for i in range(start_frame, end_frame + 1):
            test_frame_indices.append(i)
            y_test.append(gt_coords)

    return test_frame_indices, np.array(y_test)


def analyze_and_plot_results(results_df: pd.DataFrame, output_dir: str):
    """
    Calculates summary statistics and generates a prediction vs. ground truth plot
    with unique colors for each point pair.
    """
    if results_df.empty:
        print("Cannot generate analysis. The results DataFrame is empty.")
        return

    # --- Print Summary Statistics ---
    mean_error = results_df["error"].mean()
    median_error = results_df["error"].median()
    std_error = results_df["error"].std()
    percentile_95 = results_df["error"].quantile(0.95)

    stats_string = (
        f"--- Experiment Results ---\n"
        f"Mean Pixel Error:     {mean_error:.2f} pixels\n"
        f"Median Pixel Error:   {median_error:.2f} pixels\n"
        f"Std Dev of Error:     {std_error:.2f} pixels\n"
        f"95th Percentile Error: {percentile_95:.2f} pixels\n"
        f"--------------------------"
    )

    print(f"\n{stats_string}\n")

    stats_path = os.path.join(output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_string)
    print(f"Summary statistics saved to: {stats_path}")

    # --- Generate Plot ---
    num_points = len(results_df)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, num_points))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot ground truth points
    plt.scatter(
        results_df["gt_x"],
        results_df["gt_y"],
        c=colors,
        label="Ground Truth",
        s=100,
        marker="o",
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )

    # Plot predicted points
    plt.scatter(
        results_df["pred_x"],
        results_df["pred_y"],
        c=colors,
        label="Prediction",
        s=80,
        marker="x",
        alpha=0.8,
    )

    # This avoids using the DataFrame index 'i' which caused the type error.
    for color, (index, row) in zip(colors, results_df.iterrows()):
        plt.plot(
            [row["gt_x"], row["pred_x"]],
            [row["gt_y"], row["pred_y"]],
            color=color,
            alpha=0.4,
            linewidth=1,
        )

    # Plot styling
    all_x = pd.concat([results_df["gt_x"], results_df["pred_x"]])
    all_y = pd.concat([results_df["gt_y"], results_df["pred_y"]])
    plt.xlim(0, all_x.max() * 1.05)
    plt.ylim(0, all_y.max() * 1.05)
    plt.gca().invert_yaxis()
    plt.title("Gaze Point Prediction vs. Ground Truth")
    plt.xlabel("Screen X Coordinate")
    plt.ylabel("Screen Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plot_path = os.path.join(output_dir, "pred_vs_gt.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Analysis plot saved to: {plot_path}")
    plt.close()


def get_video_fps(video_path: str) -> float:
    """Helper to get the FPS of a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file to get FPS: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def main(args):
    pipeline_3d = GazePipeline3D(
        weights_path=args.weights,
        device="cuda" if torch.cuda.is_available() else "cpu",
        smooth_facebbox=True,
        smooth_gaze=False,
    )
    mapper = Mapper()
    pipeline_2d = GazePipeline2D(
        pipeline_3d=pipeline_3d, mapper=mapper, feature_keys=["pitch", "yaw"]
    )

    # Calibration Phase
    print("\n--- Preparing Calibration Data ---")
    calib_fps = get_video_fps(args.calib_video)
    calib_features = process_video_for_features(args.calib_video, pipeline_3d)

    # Populate the mapper with training data
    prepare_calibration_data(
        mapper=mapper,
        pipeline_2d=pipeline_2d,
        all_frame_features=calib_features,
        click_data_path=args.calib_json,
        fps=calib_fps,
    )

    try:
        score_x, score_y = mapper.train()
        print(f"Mapper trained. R² scores -> X: {score_x:.3f}, Y: {score_y:.3f}")
    except ValueError as e:
        print(f"Error during training: {e}")
        return

    # Evaluation Phase
    print("\n--- Evaluating on Test Data ---")
    test_fps = get_video_fps(args.test_video)
    test_frame_indices, y_test = prepare_test_data(args.test_json, test_fps)

    cap = cv2.VideoCapture(args.test_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open test video: {args.test_video}")

    frame_target_map = {frame_idx: i for i, frame_idx in enumerate(test_frame_indices)}

    current_frame = 0
    predictions: List[Tuple[float, float] | None] = [None] * len(y_test)

    print(
        f"Running end-to-end pipeline on {len(frame_target_map)} specific test frames..."
    )
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_target_map:
                results_2d = pipeline_2d.predict(frame)
                result_idx = frame_target_map[current_frame]

                if results_2d:
                    pog = results_2d[0]["pog"]
                    predictions[result_idx] = (pog["x"], pog["y"])
                else:
                    predictions[result_idx] = (np.nan, np.nan)

            current_frame += 1
            pbar.update(1)

    cap.release()

    results = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        if prediction is None or np.isnan(prediction[0]):
            continue

        pred_x, pred_y = prediction
        gt_x, gt_y = y_test[i]
        error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        results.append(
            {
                "gt_x": gt_x,
                "gt_y": gt_y,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "error": error,
            }
        )
    results_df = pd.DataFrame(results)

    os.makedirs(args.output_dir, exist_ok=True)
    analyze_and_plot_results(results_df, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a streamlined gaze estimation experiment."
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to the 3D gaze model weights (.pth file).",
    )
    parser.add_argument(
        "--calib-video",
        required=True,
        type=str,
        help="Path to the calibration video file.",
    )
    parser.add_argument(
        "--calib-json",
        required=True,
        type=str,
        help="Path to the calibration JSON file.",
    )
    parser.add_argument(
        "--test-video", required=True, type=str, help="Path to the test video file."
    )
    parser.add_argument(
        "--test-json", required=True, type=str, help="Path to the test JSON file."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save the analysis plot.",
    )

    args = parser.parse_args()
    main(args)
