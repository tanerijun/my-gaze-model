"""
Process experiment data collected from the data collector platform (The Deep Vault (web))
"""

import argparse
import json
import os
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import GazePipeline2D, GazePipeline3D, Mapper

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")


def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r") as f:
        data = json.load(f)
    return data


def print_session_info(metadata: dict):
    print(f"Session ID: {metadata['sessionId']}")
    print("Participant info:")
    print(f"\tName: {metadata['participant']['name']}")
    print(f"\tAge: {metadata['participant']['age']}")
    print(f"\tGender: {metadata['participant']['gender']}")
    print(f"\tWearing Glasses: {metadata['participant']['wearingGlasses']}")
    print(f"\tWearing Contacts: {metadata['participant']['wearingContacts']}")
    print("Video info:")
    print(
        f"\tScreen Resolution: W: {metadata['screenResolution']['width']}, H: {metadata['screenResolution']['height']}"
    )
    print(
        f"\tScreen Recording Resolution: W: {metadata['screenStreamResolution']['width']}, H: {metadata['screenStreamResolution']['height']}"
    )
    print(
        f"\tWebcam Resolution: W: {metadata['webcamResolution']['width']}, H: {metadata['webcamResolution']['height']}"
    )
    print("Click info")
    print(f"\tExplicit click count: {metadata['gameMetadata']['totalExplicitClicks']}")
    print(f"\tImplicit click count: {metadata['gameMetadata']['totalImplicitClicks']}")


def preview_videos_alignment(
    webcam_path: Path, screen_path: Path, output_path: Path, webcam_offset_ms: float = 0
):
    """
    A helper method that output videos put side by side to visually inspect if they align perfectly

    Args:
        webcam_path: Path to webcam video
        screen_path: Path to screen video
        output_path: Path to output video
        sync_offset_ms: Offset in milliseconds to shift webcam relative to screen.
                        Positive value delays webcam (trims it), negative advances it.
    """
    webcam_video = cv2.VideoCapture(str(webcam_path))
    screen_video = cv2.VideoCapture(str(screen_path))

    webcam_fps = webcam_video.get(cv2.CAP_PROP_FPS)
    screen_fps = screen_video.get(cv2.CAP_PROP_FPS)

    webcam_width = int(webcam_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(webcam_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_width = int(screen_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(screen_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames_webcam = int(webcam_video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_screen = int(screen_video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration_webcam = total_frames_webcam / webcam_fps
    duration_screen = total_frames_screen / screen_fps

    print(
        f"\nWebcam: {total_frames_webcam} frames @ {webcam_fps:.2f} FPS = {duration_webcam:.2f}s"
    )
    print(
        f"Screen: {total_frames_screen} frames @ {screen_fps:.2f} FPS = {duration_screen:.2f}s"
    )
    print(f"Duration difference: {abs(duration_webcam - duration_screen):.3f}s")
    print(f"Synchronization offset: {webcam_offset_ms:.1f}ms")

    output_fps = max(webcam_fps, screen_fps)  # higher FPS to preserve more detail
    avg_duration = (duration_webcam + duration_screen) / 2  # so that both videos fit
    total_frames_output = int(avg_duration * output_fps)
    output_height = max(webcam_height, screen_height)

    # Calculate output widths based on aspect ratio preservation
    webcam_output_width = int(webcam_width * output_height / webcam_height)
    screen_output_width = int(screen_width * output_height / screen_height)
    output_width = webcam_output_width + screen_output_width

    print(f"Output: {total_frames_output} frames @ {output_fps:.2f} FPS")
    print(f"Output dimensions: {output_width}x{output_height}")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path), fourcc, output_fps, (output_width, output_height)
    )

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    webcam_offset_frames = int((webcam_offset_ms / 1000.0) * webcam_fps)

    # Process frames using time-based synchronization
    for output_frame_idx in range(total_frames_output):
        time_s = output_frame_idx / output_fps
        webcam_frame_idx = int(time_s * webcam_fps) + webcam_offset_frames
        screen_frame_idx = int(time_s * screen_fps)

        # Clamp to valid ranges
        webcam_frame_idx = min(webcam_frame_idx, total_frames_webcam - 1)
        screen_frame_idx = min(screen_frame_idx, total_frames_screen - 1)

        # Seek and read frames
        webcam_video.set(cv2.CAP_PROP_POS_FRAMES, webcam_frame_idx)
        screen_video.set(cv2.CAP_PROP_POS_FRAMES, screen_frame_idx)
        ret_webcam, webcam_frame = webcam_video.read()
        ret_screen, screen_frame = screen_video.read()

        if not ret_webcam or not ret_screen:
            print(f"Warning: Failed to read frames at output frame {output_frame_idx}")
            break

        # Resize frames to match output height while preserving aspect ratio
        webcam_frame = cv2.resize(
            webcam_frame,
            (
                int(webcam_frame.shape[1] * output_height / webcam_frame.shape[0]),
                output_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        screen_frame = cv2.resize(
            screen_frame,
            (
                int(screen_frame.shape[1] * output_height / screen_frame.shape[0]),
                output_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        webcam_frame = cv2.resize(
            webcam_frame,
            (webcam_output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )
        screen_frame = cv2.resize(
            screen_frame,
            (screen_output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )

        combined_frame = np.hstack([webcam_frame, screen_frame])

        out.write(combined_frame)

        if (output_frame_idx + 1) % 100 == 0:
            print(f"Processed {output_frame_idx + 1}/{total_frames_output} frames...")

    webcam_video.release()
    screen_video.release()
    out.release()

    print(f"\nAlignment preview saved to: {output_path}")


def train_and_get_initial_mapper(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    context_frames: int = 0,
    timing_window_ms: float = 0,
):
    """
    Args:
        context_frames: Number of frames before and after the calibration frame (mutually exclusive with timing_window_ms)
        timing_window_ms: Time window in milliseconds before and after the click timestamp (mutually exclusive with context_frames)
    """
    if context_frames > 0 and timing_window_ms > 0:
        raise ValueError("Cannot specify both context_frames and timing_window_ms")

    # List of calibration points.
    # Example shape of single point
    # {
    #   "pointId": "top-left",
    #   "x": 5,
    #   "y": 5,
    #   "screenX": 72,
    #   "screenY": 41,
    #   "timestamp": 1765269310481,
    #   "videoTimestamp": 2428
    # }
    calibration_points = metadata["initialCalibration"]["points"]

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    training_data = {}
    for point in calibration_points:
        point_id = point["pointId"]
        video_timestamp_ms = point["videoTimestamp"]

        if timing_window_ms > 0:
            # Use timing window method
            window_start_ms = video_timestamp_ms - timing_window_ms
            window_end_ms = video_timestamp_ms + timing_window_ms
            frame_start = max(0, int(window_start_ms * fps / 1000))
            frame_end = min(total_frames - 1, int(window_end_ms * fps / 1000))
            frame_indices = list(range(frame_start, frame_end + 1))
        else:
            # Use context frames method
            frame_idx = int(video_timestamp_ms * fps / 1000)
            frame_indices = list(
                range(frame_idx - context_frames, frame_idx + context_frames + 1)
            )
            frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]

        features = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise AssertionError(
                    f"Unexpected error reading frame {frame_idx} for point {point_id}"
                )

            pipeline_output = gaze_pipeline_3d(frame)
            gaze_info = pipeline_output[0]["gaze"]
            features.append({"pitch": gaze_info["pitch"], "yaw": gaze_info["yaw"]})

        training_data[point_id] = {
            "point": point,
            "frame_indices": frame_indices,
            "features": features,
        }
        print(f"Collected {len(frame_indices)} frame indices for {point_id}")

    cap.release()

    mapper = Mapper()

    for point_id, data in training_data.items():
        point = data["point"]
        feature_vectors = [[f["pitch"], f["yaw"]] for f in data["features"]]
        target_point = (point["screenX"], point["screenY"])
        mapper.add_calibration_point(feature_vectors, target_point)

    mapper.train()
    return mapper


# INFO: this perform worse than seeking
def train_and_get_initial_mapper_sequential(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    context_frames: int = 0,
    timing_window_ms: float = 0,
):
    """
    Train initial mapper using calibration points

    Args:
        webcam_path: Path to webcam video
        metadata: Metadata dictionary containing calibration points
        gaze_pipeline_3d: 3D gaze pipeline for feature extraction
        context_frames: Number of frames before and after the calibration frame
                       (mutually exclusive with timing_window_ms)
        timing_window_ms: Time window in milliseconds before and after the click timestamp
                         (mutually exclusive with context_frames)

    Returns:
        Trained Mapper instance
    """
    if context_frames > 0 and timing_window_ms > 0:
        raise ValueError("Cannot specify both context_frames and timing_window_ms")

    calibration_points = metadata["initialCalibration"]["points"]

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build frame-to-calibration-point mapping
    calibration_frame_map = {}  # frame_idx -> list of point_ids
    calibration_points_data = {}  # point_id -> point data and frame indices
    max_frame_needed = 0  # Track the last frame needed

    for point in calibration_points:
        point_id = point["pointId"]
        video_timestamp_ms = point["videoTimestamp"]

        if timing_window_ms > 0:
            # Use timing window method
            window_start_ms = video_timestamp_ms - timing_window_ms
            window_end_ms = video_timestamp_ms + timing_window_ms
            frame_start = max(0, int(window_start_ms * fps / 1000))
            frame_end = min(total_frames - 1, int(window_end_ms * fps / 1000))
            frame_indices = list(range(frame_start, frame_end + 1))
        else:
            # Use context frames method
            frame_idx = int(video_timestamp_ms * fps / 1000)
            frame_indices = list(
                range(frame_idx - context_frames, frame_idx + context_frames + 1)
            )
            frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]

        if frame_indices:
            max_frame_needed = max(max_frame_needed, max(frame_indices))

        # Store calibration point data
        calibration_points_data[point_id] = {
            "point": point,
            "frame_indices": frame_indices,
            "features": [],  # will collect features during sequential processing
        }

        # reverse mapping for efficient lookup
        for frame_idx in frame_indices:
            if frame_idx not in calibration_frame_map:
                calibration_frame_map[frame_idx] = []
            calibration_frame_map[frame_idx].append(point_id)

    frames_to_process = max_frame_needed + 20  # some buffer at the end just to be safe

    print(f"\nProcessing {total_frames} frames for initial calibration...")
    print(f"Calibration points: {len(calibration_points)}")

    # Process frames sequentially
    for frame_idx in tqdm(
        range(frames_to_process), desc="Training initial mapper", unit="frame"
    ):
        ret, frame = cap.read()

        if not ret:
            break

        # Always run pipeline to maintain internal state
        pipeline_output = gaze_pipeline_3d(frame)

        # Check if this frame is needed for any calibration point
        if frame_idx in calibration_frame_map:
            if not (pipeline_output and len(pipeline_output) > 0):
                raise AssertionError(
                    f"Unexpected pipeline_3d output at frame {frame_idx}: {pipeline_output}"
                )

            gaze_info = pipeline_output[0]["gaze"]
            feature = {"pitch": gaze_info["pitch"], "yaw": gaze_info["yaw"]}

            # Add this feature to all calibration points that need it
            for point_id in calibration_frame_map[frame_idx]:
                calibration_points_data[point_id]["features"].append(feature)

    cap.release()

    # Verify we collected features for all calibration points
    for point_id, data in calibration_points_data.items():
        num_features = len(data["features"])
        num_expected = len(data["frame_indices"])
        print(f"Collected {num_features}/{num_expected} features for {point_id}")

        if num_features == 0:
            raise AssertionError(
                f"No features collected for calibration point {point_id}. "
                f"Expected frames: {data['frame_indices']}"
            )

    # Create and train mapper
    mapper = Mapper()

    for point_id, data in calibration_points_data.items():
        point = data["point"]
        feature_vectors = [[f["pitch"], f["yaw"]] for f in data["features"]]
        target_point = (point["screenX"], point["screenY"])
        mapper.add_calibration_point(feature_vectors, target_point)

    score_x, score_y = mapper.train()
    print("\nInitial mapper trained successfully")
    print(f"  R² score (X): {score_x:.4f}")
    print(f"  R² score (Y): {score_y:.4f}")
    print(f"  Total training samples: {len(mapper.initial_feature_vectors)}")

    return mapper


def evaluate_gaze_model_static(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    feature_keys: list[str] = ["pitch", "yaw"],
    context_frames: int = 5,
):
    """
    Evaluate the 2D gaze pipeline on explicit clicks by processing frames sequentially.
    Respecting dependencies on previous frames. (can see face_bbox effect, but no dynamic calibration)

    Args:
        webcam_path: Path to webcam video
        metadata: Metadata dictionary
        gaze_pipeline_2d: Trained 2D gaze pipeline
        context_frames: Number of frames before and after to evaluate

    Returns:
        List of evaluation results with errors in pixels
    """
    explicit_clicks = [
        click for click in metadata["clicks"] if click["type"] == "explicit"
    ]

    if not explicit_clicks:
        raise AssertionError("No explicit clicks found for evaluation")

    print(f"\n{'=' * 60}")
    print("Static Calibration mode")
    print(f"{'=' * 60}")
    print(f"Explicit clicks for evaluation: {len(explicit_clicks)}")

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mapper = train_and_get_initial_mapper(
        webcam_path, metadata, gaze_pipeline_3d, context_frames=context_frames
    )
    print("Static mapper enabled.")

    initial_stats = mapper.get_training_stats()
    print(f"Initial calibration: {initial_stats['initial_samples']} samples")

    gaze_pipeline_2d = GazePipeline2D(
        gaze_pipeline_3d,
        mapper,
        feature_keys,
    )

    # Build a map of frame indices to evaluate for each click
    click_eval_frames = {}
    for click in explicit_clicks:
        click_id = click["id"]
        video_timestamp_ms = click["videoTimestamp"]
        center_frame_idx = int(video_timestamp_ms * fps / 1000)

        frame_indices = list(
            range(
                center_frame_idx - context_frames, center_frame_idx + context_frames + 1
            )
        )
        frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
        click_eval_frames[click_id] = {
            "center_frame_idx": center_frame_idx,
            "frame_indices": frame_indices,
            "click": click,
            "predictions": [],
        }

    evaluation_results = []

    for frame_idx in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()

        if not ret:
            break

        # Run pipeline on every frame (to maintain internal state)
        results_2d = gaze_pipeline_2d.predict(frame)

        # Check if this frame is part of any click evaluation
        for click_id, click_data in click_eval_frames.items():
            if frame_idx in click_data["frame_indices"]:
                if not (results_2d and len(results_2d) > 0 and results_2d[0]["pog"]):
                    raise AssertionError(f"Unexpected pipeline_2d output: {results_2d}")

                pog = results_2d[0]["pog"]

                screen_width = metadata["screenResolution"]["width"]
                screen_height = metadata["screenResolution"]["height"]
                pog_x = max(0, min(screen_width - 1, pog["x"]))
                pog_y = max(0, min(screen_height - 1, pog["y"]))

                click_data["predictions"].append(
                    {"frame_idx": frame_idx, "x": pog_x, "y": pog_y}
                )

    cap.release()

    # Process collected predictions
    for click_id, click_data in click_eval_frames.items():
        click = click_data["click"]
        center_frame_idx = click_data["center_frame_idx"]
        frame_predictions = click_data["predictions"]

        target_x = click["screenX"]
        target_y = click["screenY"]

        print(f"\nEvaluating click {click_id}:")
        print(f"  Center frame: {center_frame_idx}")
        print(f"  Ground truth: ({target_x:.1f}, {target_y:.1f})")

        if not frame_predictions:
            raise AssertionError(f"No predictions collected for click {click_id}")

        # Calculate errors for each prediction
        errors = []
        for pred in frame_predictions:
            euclidean_distance = np.sqrt(
                (pred["x"] - target_x) ** 2 + (pred["y"] - target_y) ** 2
            )
            errors.append(euclidean_distance)

        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        percentile_95_error = np.percentile(errors, 95)

        evaluation_results.append(
            {
                "click_id": click_id,
                "timestamp": click["timestamp"],
                "videoTimestamp": click["videoTimestamp"],
                "ground_truth": {"x": target_x, "y": target_y},
                "num_predictions": len(frame_predictions),
                "errors_px": errors,
                "mean_error_px": float(mean_error),
                "median_error_px": float(median_error),
                "std_error_px": float(std_error),
                "percentile_95_error_px": float(percentile_95_error),
                "predictions": frame_predictions,
            }
        )

        print(f"  Mean error: {mean_error:.2f}px")
        print(f"  Median error: {median_error:.2f}px")
        print(f"  Std error: {std_error:.2f}px")
        print(f"  95th percentile error: {percentile_95_error:.2f}px")

    return {
        "evaluation_results": evaluation_results,
        "mapper_stats": mapper.get_training_stats(),
    }


def evaluate_gaze_model_dynamic(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    feature_keys: list[str] = ["pitch", "yaw"],
    context_frames: int = 5,
    buffer_size: Optional[int] = None,
):
    """
    Evaluate the gaze model with dynamic calibration using implicit clicks.

    This function processes frames sequentially and updates the mapper with implicit
    click data as it encounters them, simulating real-time adaptive calibration.

    Args:
        webcam_path: Path to webcam video
        metadata: Metadata dictionary
        gaze_pipeline_3d: 3D gaze pipeline for feature extraction
        feature_keys: List of feature keys to extract (e.g., ["pitch", "yaw"])
        context_frames: Number of frames before and after to evaluate explicit clicks
        buffer_size: Maximum number of implicit samples to keep.
                    If None, accumulates all samples (infinite buffer).

    Returns:
        Dictionary containing evaluation results and calibration history
    """
    explicit_clicks = [
        click for click in metadata["clicks"] if click["type"] == "explicit"
    ]
    implicit_clicks = [
        click for click in metadata["clicks"] if click["type"] == "implicit"
    ]
    calibration_clicks = metadata["clicks"]

    if not explicit_clicks:
        raise AssertionError("No explicit clicks found for evaluation")
    if not implicit_clicks:
        raise AssertionError("No implicit clicks found for dynamic calibration")

    mode_name = (
        "Accumulate (Infinite Buffer)"
        if buffer_size is None
        else f"Fixed Buffer (size={buffer_size})"
    )
    print(f"\n{'=' * 60}")
    print(f"Dynamic Calibration Mode: {mode_name}")
    print(f"{'=' * 60}")
    print(f"Explicit clicks for evaluation: {len(explicit_clicks)}")
    print(f"Implicit clicks for calibration: {len(implicit_clicks)}")

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nTraining initial mapper...")
    mapper = train_and_get_initial_mapper(
        webcam_path, metadata, gaze_pipeline_3d, context_frames=context_frames
    )
    mapper.enable_dynamic_calibration = True
    mapper.buffer_size = buffer_size
    print("Dynamic mapper enabled.")

    initial_stats = mapper.get_training_stats()
    print(f"Initial calibration: {initial_stats['initial_samples']} samples")

    gaze_pipeline_2d = GazePipeline2D(
        gaze_pipeline_3d,
        mapper,
        feature_keys,
    )

    # frame-to-click mapping for efficient lookup
    calibration_click_map = {}  # frame_idx -> click data
    explicit_click_frames = {}  # click_id -> frame data

    for click in calibration_clicks:
        frame_idx = int(click["videoTimestamp"] * fps / 1000)
        calibration_click_map[frame_idx] = click

    for click in explicit_clicks:
        click_id = click["id"]
        video_timestamp_ms = click["videoTimestamp"]
        center_frame_idx = int(video_timestamp_ms * fps / 1000)

        frame_indices = list(
            range(
                center_frame_idx - context_frames, center_frame_idx + context_frames + 1
            )
        )
        frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]

        explicit_click_frames[click_id] = {
            "center_frame_idx": center_frame_idx,
            "frame_indices": frame_indices,
            "click": click,
            "predictions": [],
        }

    # Track calibration history
    calibration_history = []

    # Process frames sequentially
    for frame_idx in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()

        if not ret:
            break

        # Check if this frame has an implicit click
        if frame_idx in calibration_click_map:
            calib_click = calibration_click_map[frame_idx]
            results_3d = gaze_pipeline_3d(frame)

            if not (results_3d and len(results_3d) > 0):
                raise AssertionError(f"Unexpected pipeline_3d output: {results_3d}")

            feature_vector = gaze_pipeline_2d.extract_feature_vector(results_3d[0])
            target_point = (calib_click["screenX"], calib_click["screenY"])

            mapper.add_dynamic_calibration_point(feature_vector, target_point)
            mapper.train()

            stats = mapper.get_training_stats()
            calibration_history.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": calib_click["timestamp"],
                    "click_type": calib_click["type"],
                    "total_samples": stats["total_samples"],
                    "dynamic_samples": stats["dynamic_samples"],
                }
            )

        # Run pipeline on every frame (to maintain internal state)
        results_2d = gaze_pipeline_2d.predict(frame)

        # Check if this frame has an explicit click
        for click_id, click_data in explicit_click_frames.items():
            if frame_idx in click_data["frame_indices"]:
                if not (results_2d and len(results_2d) > 0 and results_2d[0]["pog"]):
                    raise AssertionError(f"Unexpected pipeline_2d output: {results_2d}")

                pog = results_2d[0]["pog"]

                screen_width = metadata["screenResolution"]["width"]
                screen_height = metadata["screenResolution"]["height"]
                pog_x = max(0, min(screen_width - 1, pog["x"]))
                pog_y = max(0, min(screen_height - 1, pog["y"]))

                click_data["predictions"].append(
                    {"frame_idx": frame_idx, "x": pog_x, "y": pog_y}
                )

    cap.release()

    # Process collected predictions
    evaluation_results = []

    for click_id, click_data in explicit_click_frames.items():
        click = click_data["click"]
        center_frame_idx = click_data["center_frame_idx"]
        frame_predictions = click_data["predictions"]

        target_x1 = click["screenX"]
        target_y1 = click["screenY"]
        target_x2 = click["targetX"]
        target_y2 = click["targetY"]

        print(f"\nEvaluating click {click_id}:")
        print(f"  Center frame: {center_frame_idx}")
        print(f"  Ground truth: ({target_x1:.1f}, {target_y1:.1f})")
        print(f"  Ground truth: ({target_x2:.1f}, {target_y2:.1f})")

        if not frame_predictions:
            raise AssertionError(f"No predictions collected for click {click_id}")

        # Calculate errors for each prediction
        errors = []
        for pred in frame_predictions:
            euclidean_distance1 = np.sqrt(
                (pred["x"] - target_x1) ** 2 + (pred["y"] - target_y1) ** 2
            )
            euclidean_distance2 = np.sqrt(
                (pred["x"] - target_x2) ** 2 + (pred["y"] - target_y2) ** 2
            )
            errors.append(min(euclidean_distance1, euclidean_distance2))

        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        percentile_95_error = np.percentile(errors, 95)

        evaluation_results.append(
            {
                "click_id": click_id,
                "timestamp": click["timestamp"],
                "videoTimestamp": click["videoTimestamp"],
                "ground_truth": {"x": target_x1, "y": target_y1},
                "num_predictions": len(frame_predictions),
                "errors_px": errors,
                "mean_error_px": float(mean_error),
                "median_error_px": float(median_error),
                "std_error_px": float(std_error),
                "percentile_95_error_px": float(percentile_95_error),
                "predictions": frame_predictions,
            }
        )

        print(f"  Mean error: {mean_error:.2f}px")
        print(f"  Median error: {median_error:.2f}px")
        print(f"  Std error: {std_error:.2f}px")
        print(f"  95th percentile error: {percentile_95_error:.2f}px")

    return {
        "evaluation_results": evaluation_results,
        "calibration_history": calibration_history,
        "mapper_stats": mapper.get_training_stats(),
    }


def save_evaluation_summary(
    evaluation_results: list[dict], output_path: Path | None = None
):
    """
    Print and optionally save summary statistics from evaluation results.

    Args:
        evaluation_results: List of evaluation results from evaluate_gaze_model
        output_path: Optional path to save the summary to a text file
    """
    if not evaluation_results:
        print("No evaluation results to summarize.")
        return

    all_errors = []
    for result in evaluation_results:
        all_errors.extend(result["errors_px"])

    summary_text = f"\n{'=' * 60}\n"
    summary_text += f"EVALUATION SUMMARY ({len(evaluation_results)} explicit clicks)\n"
    summary_text += f"{'=' * 60}\n"
    summary_text += f"Overall mean error: {np.mean(all_errors):.2f}px\n"
    summary_text += f"Overall median error: {np.median(all_errors):.2f}px\n"
    summary_text += f"Overall std error: {np.std(all_errors):.2f}px\n"
    summary_text += (
        f"Overall 95th percentile error: {np.percentile(all_errors, 95):.2f}px\n"
    )
    summary_text += (
        f"Total predictions: {sum(r['num_predictions'] for r in evaluation_results)}\n"
    )
    summary_text += f"{'=' * 60}\n"

    # Always print
    print(summary_text)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(summary_text)
        print(f"Evaluation summary saved to: {output_path}")


def generate_gaze_demo(
    webcam_path: Path,
    screen_path: Path,
    output_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    context_frames: int = 0,
    buffer_size: int | None = None,
    feature_keys: list[str] = ["pitch", "yaw"],
    webcam_video_offset_ms: float = 0,
    visualization_mode: Literal["point", "heatmap", "scanpath"] = "point",
    webcam_scale: float = 0.25,
    point_radius: int = 15,
    point_color: tuple = (0, 255, 0),
    point_thickness: int = -1,
    heatmap_radius: int = 80,
    heatmap_decay: float = 0.95,
    scanpath_length: int = 30,
    scanpath_thickness: int = 3,
):
    """
    Generate a demo video with gaze visualization overlaid on screen recording.

    NOTE:
    Uses timestamp-based synchronization instead of FPS-based frame mapping
    to handle variable frame rate (VFR) screen recordings better

    Args:
        webcam_path: Path to webcam video
        screen_path: Path to screen recording video
        output_path: Path to save the demo video
        metadata: Session metadata containing timestamps and game info
        gaze_pipeline_3d: 3D gaze pipeline for feature extraction
        context_frames: Number of frames for calibration context
        buffer_size: Dynamic calibration buffer size (None = infinite)
        feature_keys: Features to extract for gaze prediction
        webcam_video_offset_ms: Sync offset between videos
        visualization_mode: "point", "heatmap", or "scanpath"
        webcam_scale: Scale factor for webcam overlay (0-1)
        point_radius: Radius of gaze point visualization
        point_color: BGR color tuple for gaze point
        point_thickness: Thickness of point (-1 for filled)
        heatmap_radius: Radius of heatmap gaussian
        heatmap_decay: Decay factor for heatmap accumulator
        scanpath_length: Number of points in scanpath history
        scanpath_thickness: Line thickness for scanpath
    """
    print(f"\n{'=' * 60}")
    print("GENERATING GAZE DEMO VIDEO (TIMESTAMP-BASED SYNC)")
    print(f"{'=' * 60}")
    print(f"Visualization mode: {visualization_mode}")
    print(f"Webcam offset: {webcam_video_offset_ms}ms")

    # Get game start timestamp to determine when to start rendering predictions
    game_start_timestamp = metadata.get("gameStartTimestamp")
    recording_start_time = metadata.get("recordingStartTime")

    if game_start_timestamp is None or recording_start_time is None:
        print(
            "Warning: gameStartTimestamp or recordingStartTime not found in metadata."
        )
        print("Rendering predictions for all frames.")
        render_start_time_ms = 0
    else:
        # Calculate relative time from video start
        render_start_time_ms = game_start_timestamp - recording_start_time
        print(f"Recording starts at: {recording_start_time}")
        print(f"Game starts at: {game_start_timestamp}")
        print(
            f"Predictions will render after: {render_start_time_ms}ms from video start"
        )

    # Initial captures for info
    webcam_cap = cv2.VideoCapture(str(webcam_path))
    screen_cap = cv2.VideoCapture(str(screen_path))

    if not webcam_cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")
    if not screen_cap.isOpened():
        raise IOError(f"Cannot open screen video: {screen_path}")

    screen_fps = screen_cap.get(cv2.CAP_PROP_FPS)
    webcam_fps = webcam_cap.get(cv2.CAP_PROP_FPS)

    screen_width = int(screen_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(screen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    webcam_width = int(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_screen_frames = int(screen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_webcam_frames = int(webcam_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    screen_duration = total_screen_frames / screen_fps
    webcam_duration = total_webcam_frames / webcam_fps

    print(
        f"\nScreen: {screen_width}x{screen_height} @ {screen_fps:.2f} FPS, "
        f"{total_screen_frames} frames ({screen_duration:.2f}s)"
    )
    print(
        f"Webcam: {webcam_width}x{webcam_height} @ {webcam_fps:.2f} FPS, "
        f"{total_webcam_frames} frames ({webcam_duration:.2f}s)"
    )
    print(f"Duration difference: {abs(screen_duration - webcam_duration):.3f}s")

    # Calculate webcam overlay dimensions
    overlay_height = int(screen_height * webcam_scale)
    overlay_width = int(webcam_width * overlay_height / webcam_height)

    # Calculate webcam offset in frames
    webcam_offset_frames = int((webcam_video_offset_ms / 1000.0) * webcam_fps)

    # Calculate the webcam frame index where rendering should start
    render_start_frame = int((render_start_time_ms / 1000.0) * webcam_fps)
    print(f"Predictions will render starting from webcam frame: {render_start_frame}")

    # Close initial captures
    webcam_cap.release()
    screen_cap.release()

    # == PHASE 1: Compute Gaze Predictions ==
    print("\n" + "=" * 60)
    print("PHASE 1: Computing gaze predictions from webcam feed")
    print("=" * 60)

    print("\nTraining initial mapper...")
    mapper = train_and_get_initial_mapper(
        webcam_path, metadata, gaze_pipeline_3d, context_frames=context_frames
    )
    mapper.enable_dynamic_calibration = True
    mapper.buffer_size = buffer_size
    print("Dynamic mapper enabled.")

    initial_stats = mapper.get_training_stats()
    print(f"Initial calibration: {initial_stats['initial_samples']} samples")

    gaze_pipeline_2d = GazePipeline2D(
        gaze_pipeline_3d,
        mapper,
        feature_keys,
    )

    # Process webcam frames sequentially
    webcam_cap_phase1 = cv2.VideoCapture(str(webcam_path))
    gaze_predictions = {}  # webcam_frame_idx -> (gaze_x, gaze_y)

    # Build calibration click map if using dynamic calibration
    calibration_click_map = {}
    calibration_clicks = metadata["clicks"]
    for click in calibration_clicks:
        frame_idx = int(click["videoTimestamp"] * webcam_fps / 1000)
        calibration_click_map[frame_idx] = click

    print(f"Processing {total_webcam_frames} webcam frames...")

    for webcam_frame_idx in tqdm(
        range(total_webcam_frames), desc="Computing gaze", unit="frame"
    ):
        ret, webcam_frame = webcam_cap_phase1.read()
        if not ret:
            break

        # Check for dynamic calibration update
        if webcam_frame_idx in calibration_click_map:
            calib_click = calibration_click_map[webcam_frame_idx]
            try:
                results_3d = gaze_pipeline_3d(webcam_frame)
                if results_3d and len(results_3d) > 0:
                    feature_vector = gaze_pipeline_2d.extract_feature_vector(
                        results_3d[0]
                    )
                    target_point = (calib_click["screenX"], calib_click["screenY"])
                    mapper.add_dynamic_calibration_point(feature_vector, target_point)
                    mapper.train()
            except Exception:
                pass  # Skip if calibration update fails

        # Compute gaze prediction for all frames
        try:
            results_2d = gaze_pipeline_2d.predict(webcam_frame)
            if results_2d and len(results_2d) > 0 and results_2d[0]["pog"]:
                pog = results_2d[0]["pog"]

                # Scale to screen recording resolution
                scale_x = screen_width / metadata["screenResolution"]["width"]
                scale_y = screen_height / metadata["screenResolution"]["height"]
                gaze_x = int(pog["x"] * scale_x)
                gaze_y = int(pog["y"] * scale_y)

                # Clip to screen bounds
                gaze_x = max(0, min(screen_width - 1, gaze_x))
                gaze_y = max(0, min(screen_height - 1, gaze_y))

                gaze_predictions[webcam_frame_idx] = (gaze_x, gaze_y)
        except Exception:
            # Skip frames where prediction fails
            pass

    webcam_cap_phase1.release()
    print(f"\nComputed {len(gaze_predictions)} gaze predictions")

    # == PHASE 2: Generate Demo Video ==
    print("\n" + "=" * 60)
    print("PHASE 2: Generating demo video with timestamp-based sync")
    print("=" * 60)

    # Re-open captures for demo generation
    webcam_cap = cv2.VideoCapture(str(webcam_path))
    screen_cap = cv2.VideoCapture(str(screen_path))

    # Setup video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path), fourcc, screen_fps, (screen_width, screen_height)
    )

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    # Initialize visualization-specific data structures
    heatmap_accumulator = np.zeros((screen_height, screen_width), dtype=np.float32)
    gaze_history = deque(maxlen=scanpath_length)

    print(f"Rendering {total_screen_frames} frames...")

    # Use timestamp-based synchronization
    pbar = tqdm(total=total_screen_frames, desc="Rendering demo", unit="frame")
    screen_frame_idx = 0

    while True:
        ret_screen, screen_frame = screen_cap.read()
        if not ret_screen:
            break

        # Get actual timestamp from screen video (handles VFR correctly)
        screen_timestamp_ms = screen_cap.get(cv2.CAP_PROP_POS_MSEC)

        # Convert to webcam frame index with offset
        webcam_frame_idx = (
            int((screen_timestamp_ms / 1000.0) * webcam_fps) + webcam_offset_frames
        )
        webcam_frame_idx = max(0, min(webcam_frame_idx, total_webcam_frames - 1))

        # Look up pre-computed gaze prediction
        gaze_point = gaze_predictions.get(webcam_frame_idx)

        # Read webcam frame for overlay
        webcam_cap.set(cv2.CAP_PROP_POS_FRAMES, webcam_frame_idx)
        ret_webcam, webcam_frame = webcam_cap.read()
        if not ret_webcam:
            webcam_frame = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)

        # Create output frame
        output_frame = screen_frame.copy()

        # Only render gaze visualization if we have a prediction AND after game start
        if gaze_point and webcam_frame_idx >= render_start_frame:
            gaze_x, gaze_y = gaze_point

            if visualization_mode == "point":
                # Draw simple point
                cv2.circle(
                    output_frame,
                    (gaze_x, gaze_y),
                    point_radius,
                    point_color,
                    point_thickness,
                )
                # Add outer ring for better visibility
                cv2.circle(
                    output_frame, (gaze_x, gaze_y), point_radius + 3, (255, 255, 255), 2
                )

            elif visualization_mode == "heatmap":
                # Update heatmap accumulator
                y_indices, x_indices = np.ogrid[:screen_height, :screen_width]
                distance = np.sqrt(
                    (x_indices - gaze_x) ** 2 + (y_indices - gaze_y) ** 2
                )
                gaussian = np.exp(-(distance**2) / (2 * heatmap_radius**2))
                heatmap_accumulator += gaussian

                # Apply decay to existing heatmap
                heatmap_accumulator *= heatmap_decay

                # Normalize and convert to color
                normalized_heatmap = heatmap_accumulator / (
                    heatmap_accumulator.max() + 1e-6
                )
                heatmap_colored = cv2.applyColorMap(
                    (normalized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )

                # Blend heatmap with original frame
                alpha = 0.3
                output_frame = cv2.addWeighted(
                    output_frame, 1 - alpha, heatmap_colored, alpha, 0
                )

            elif visualization_mode == "scanpath":
                # Add current point to history
                gaze_history.append(gaze_point)

                # Draw lines connecting points
                if len(gaze_history) > 1:
                    points = list(gaze_history)
                    for i in range(len(points) - 1):
                        # Fade older points
                        alpha = (i + 1) / len(points)
                        color = tuple(int(c * alpha) for c in point_color)
                        thickness = max(1, int(scanpath_thickness * alpha))
                        cv2.line(
                            output_frame, points[i], points[i + 1], color, thickness
                        )

                # Draw current point
                if len(gaze_history) > 0:
                    cv2.circle(
                        output_frame, gaze_history[-1], point_radius, point_color, -1
                    )
                    cv2.circle(
                        output_frame,
                        gaze_history[-1],
                        point_radius + 3,
                        (255, 255, 255),
                        2,
                    )

        # Resize and overlay webcam feed in bottom-right corner
        webcam_resized = cv2.resize(webcam_frame, (overlay_width, overlay_height))

        # Calculate position (bottom-right with margin)
        margin = 10
        y_start = screen_height - overlay_height - margin
        x_start = screen_width - overlay_width - margin

        # Add border around webcam feed
        border_thickness = 3
        cv2.rectangle(
            output_frame,
            (x_start - border_thickness, y_start - border_thickness),
            (
                x_start + overlay_width + border_thickness,
                y_start + overlay_height + border_thickness,
            ),
            (255, 255, 255),
            border_thickness,
        )

        # Overlay webcam feed
        output_frame[
            y_start : y_start + overlay_height, x_start : x_start + overlay_width
        ] = webcam_resized

        # Write frame
        out.write(output_frame)

        # Update progress
        pbar.update(1)
        screen_frame_idx += 1

    pbar.close()

    # Cleanup
    webcam_cap.release()
    screen_cap.release()
    out.release()

    print(f"\n{'=' * 60}")
    print(f"Demo video saved to: {output_path}")
    print(f"Total frames rendered: {screen_frame_idx}")
    print(f"{'=' * 60}")


def analyze_head_pose(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    output_dir: Path,
):
    """
    Analyze head pose stability throughout the session.

    Generates:
    - head_pose_angles.png: Raw angles with calibration reference
    - head_pose_deviations.png: Absolute deviations over time
    - head_pose_phase_comparison.png: Calibration vs game phase
    - head_pose_deviation_distribution.png: Distribution and cumulative plots
    - head_pose_stats.json: Statistical summary
    - head_pose_data.json: Full frame-by-frame data
    """
    import matplotlib.pyplot as plt

    def _extract_calibration_head_pose() -> dict[str, float]:
        """Extract average head pose from calibration points."""
        calibration_points = metadata["initialCalibration"]["points"]
        cap = cv2.VideoCapture(str(webcam_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam video: {webcam_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        head_poses = []

        print(
            f"\nExtracting head pose from {len(calibration_points)} calibration points..."
        )

        for point in calibration_points:
            video_timestamp_ms = point["videoTimestamp"]
            frame_idx = int(video_timestamp_ms * fps / 1000)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                results = gaze_pipeline_3d(frame)
                if results and len(results) > 0:
                    gaze_origin = results[0].get("gaze_origin_features")
                    if gaze_origin and "head_pitch" in gaze_origin:
                        head_poses.append(
                            {
                                "pitch": gaze_origin["head_pitch"],
                                "yaw": gaze_origin["head_yaw"],
                                "roll": gaze_origin["head_roll"],
                            }
                        )

        cap.release()

        if not head_poses:
            raise ValueError("Could not extract head pose from calibration frames. ")

        avg_pitch = np.mean([hp["pitch"] for hp in head_poses])
        avg_yaw = np.mean([hp["yaw"] for hp in head_poses])
        avg_roll = np.mean([hp["roll"] for hp in head_poses])

        std_pitch = np.std([hp["pitch"] for hp in head_poses])
        std_yaw = np.std([hp["yaw"] for hp in head_poses])
        std_roll = np.std([hp["roll"] for hp in head_poses])

        print(f"\nCalibration Head Pose (n={len(head_poses)} frames):")
        print(f"  Pitch: {avg_pitch:.2f}° (±{std_pitch:.2f}°)")
        print(f"  Yaw:   {avg_yaw:.2f}° (±{std_yaw:.2f}°)")
        print(f"  Roll:  {avg_roll:.2f}° (±{std_roll:.2f}°)")

        return {
            "pitch": avg_pitch,
            "yaw": avg_yaw,
            "roll": avg_roll,
            "pitch_std": std_pitch,
            "yaw_std": std_yaw,
            "roll_std": std_roll,
            "num_frames": len(head_poses),
        }  # type: ignore

    def _analyze_session() -> list[dict]:
        """Analyze head pose for entire session."""
        game_start_timestamp = metadata.get("gameStartTimestamp")
        recording_start_time = metadata.get("recordingStartTime")

        if game_start_timestamp and recording_start_time:
            game_start_time_ms = game_start_timestamp - recording_start_time
        else:
            game_start_time_ms = 0

        cap = cv2.VideoCapture(str(webcam_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam video: {webcam_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        game_start_frame = int((game_start_time_ms / 1000.0) * fps)

        print(f"\nAnalyzing head pose for {total_frames} frames...")
        print(f"Game starts at frame {game_start_frame}")

        session_data = []

        for frame_idx in tqdm(
            range(total_frames), desc="Processing frames", unit="frame"
        ):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_idx / fps) * 1000
            is_game_phase = frame_idx >= game_start_frame

            results = gaze_pipeline_3d(frame)

            if results and len(results) > 0:
                gaze_origin = results[0].get("gaze_origin_features")

                if gaze_origin and "head_pitch" in gaze_origin:
                    current_pitch = gaze_origin["head_pitch"]
                    current_yaw = gaze_origin["head_yaw"]
                    current_roll = gaze_origin["head_roll"]

                    pitch_dev = current_pitch - calibration_pose["pitch"]
                    yaw_dev = current_yaw - calibration_pose["yaw"]
                    roll_dev = current_roll - calibration_pose["roll"]

                    total_deviation = np.sqrt(pitch_dev**2 + yaw_dev**2 + roll_dev**2)

                    session_data.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "is_game_phase": is_game_phase,
                            "pitch": current_pitch,
                            "yaw": current_yaw,
                            "roll": current_roll,
                            "pitch_deviation": pitch_dev,
                            "yaw_deviation": yaw_dev,
                            "roll_deviation": roll_dev,
                            "total_deviation": total_deviation,
                        }
                    )
                else:
                    session_data.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "is_game_phase": is_game_phase,
                            "pitch": None,
                            "yaw": None,
                            "roll": None,
                            "pitch_deviation": None,
                            "yaw_deviation": None,
                            "roll_deviation": None,
                            "total_deviation": None,
                        }
                    )

        cap.release()
        return session_data

    def _generate_plots(session_data: list[dict]):
        """Generate all visualization plots."""
        valid_data = [d for d in session_data if d["total_deviation"] is not None]
        if not valid_data:
            print("Warning: No valid head pose data found")
            return

        timestamps = [d["timestamp_ms"] / 1000 for d in valid_data]
        calib_data = [d for d in valid_data if not d["is_game_phase"]]
        game_data = [d for d in valid_data if d["is_game_phase"]]

        # Plot 1: Head pose angles
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            timestamps,
            [d["pitch"] for d in valid_data],
            label="Pitch",
            alpha=0.4,
            linewidth=1.5,
        )
        ax.plot(
            timestamps,
            [d["yaw"] for d in valid_data],
            label="Yaw",
            alpha=0.4,
            linewidth=1.5,
        )
        ax.plot(
            timestamps,
            [d["roll"] for d in valid_data],
            label="Roll",
            alpha=0.4,
            linewidth=1.5,
        )
        ax.axhline(
            calibration_pose["pitch"],
            color="blue",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label="Cal. Pitch",
        )
        ax.axhline(
            calibration_pose["yaw"],
            color="orange",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label="Cal. Yaw",
        )
        ax.axhline(
            calibration_pose["roll"],
            color="green",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label="Cal. Roll",
        )
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Angle (degrees)", fontsize=12)
        ax.set_title("Head Pose Angles", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "head_pose_angles.png", dpi=150)
        plt.close()
        print("Saved: head_pose_angles.png")

        # Plot 2: Deviations
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            timestamps,
            [abs(d["pitch_deviation"]) for d in valid_data],
            label="Pitch Dev",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.plot(
            timestamps,
            [abs(d["yaw_deviation"]) for d in valid_data],
            label="Yaw Dev",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.plot(
            timestamps,
            [abs(d["roll_deviation"]) for d in valid_data],
            label="Roll Dev",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Deviation (degrees)", fontsize=12)
        ax.set_title(
            "Absolute Deviations from Calibration", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "head_pose_deviations.png", dpi=150)
        plt.close()
        print("Saved: head_pose_deviations.png")

        # Plot 3: Phase comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        if calib_data and game_data:
            calib_devs = [d["total_deviation"] for d in calib_data]
            game_devs = [d["total_deviation"] for d in game_data]
            ax.boxplot([calib_devs, game_devs], labels=["Calibration", "Game"])  # type: ignore
            ax.set_ylabel("Total Deviation (degrees)", fontsize=12)
            ax.set_title("Deviation by Phase", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "head_pose_phase_comparison.png", dpi=150)
        plt.close()
        print("Saved: head_pose_phase_comparison.png")

        # Plot 4: Distribution
        if game_data:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Head Pose Deviation Distribution (Game Phase)", fontsize=14)

            deviations = [d["total_deviation"] for d in game_data]

            # Histogram
            ax = axes[0]
            ax.hist(deviations, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(
                np.mean(deviations),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(deviations):.2f}°",
            )
            ax.axvline(
                np.median(deviations),
                color="green",
                linestyle="--",
                label=f"Median: {np.median(deviations):.2f}°",
            )
            ax.set_xlabel("Total Deviation (degrees)")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Deviations")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Cumulative
            ax = axes[1]
            sorted_devs = np.sort(deviations)
            cumulative = np.arange(1, len(sorted_devs) + 1) / len(sorted_devs) * 100
            ax.plot(sorted_devs, cumulative)
            ax.axhline(
                50, color="green", linestyle="--", alpha=0.5, label="50th percentile"
            )
            ax.axhline(
                95, color="red", linestyle="--", alpha=0.5, label="95th percentile"
            )
            ax.set_xlabel("Total Deviation (degrees)")
            ax.set_ylabel("Cumulative Percentage")
            ax.set_title("Cumulative Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "head_pose_deviation_distribution.png", dpi=150)
            plt.close()
            print("Saved: head_pose_deviation_distribution.png")

    def _save_statistics(session_data: list[dict]):
        """Save statistics and full data."""
        valid_data = [d for d in session_data if d["total_deviation"] is not None]
        game_data = [d for d in valid_data if d["is_game_phase"]]

        print(f"\n{'=' * 60}")
        print("HEAD POSE ANALYSIS SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total frames analyzed: {len(session_data)}")
        print(
            f"Valid detections: {len(valid_data)} ({len(valid_data) / len(session_data) * 100:.1f}%)"
        )

        stats = {
            "calibration_pose": calibration_pose,
            "total_frames": len(session_data),
            "valid_detections": len(valid_data),
            "detection_rate": len(valid_data) / len(session_data),
        }

        if game_data:
            game_deviations = [d["total_deviation"] for d in game_data]
            print("\nGame Phase Head Movement:")
            print(f"  Mean deviation: {np.mean(game_deviations):.2f}°")
            print(f"  Median deviation: {np.median(game_deviations):.2f}°")
            print(f"  Std deviation: {np.std(game_deviations):.2f}°")
            print(f"  Max deviation: {np.max(game_deviations):.2f}°")
            print(f"  95th percentile: {np.percentile(game_deviations, 95):.2f}°")

            stats["game_phase"] = {
                "num_frames": len(game_data),
                "mean_deviation": float(np.mean(game_deviations)),
                "median_deviation": float(np.median(game_deviations)),
                "std_deviation": float(np.std(game_deviations)),
                "max_deviation": float(np.max(game_deviations)),
                "p95_deviation": float(np.percentile(game_deviations, 95)),
            }

        # Save stats
        stats_file = output_dir / "head_pose_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")

        # Save full data
        data_file = output_dir / "head_pose_data.json"
        with open(data_file, "w") as f:
            json.dump(session_data, f, indent=2)
        print(f"Full data saved to: {data_file}")

    # Main execution
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_pose = _extract_calibration_head_pose()
    gaze_pipeline_3d.reset_tracking()

    session_data = _analyze_session()
    _generate_plots(session_data)
    _save_statistics(session_data)


def analyze_linearity(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    output_dir: Path,
):
    """
    Analyze linearity between gaze angles (pitch/yaw) and screen coordinates.

    Generates:
    - linearity_plot.png: Scatter plots with regression lines
    - linearity_report.txt: Statistical summary (R², MAE, RMSE)
    - linearity_data.json: Full collected data points
    """
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build event map: frame_idx -> list of events
    event_map = defaultdict(list)
    for pt in metadata["initialCalibration"]["points"][:9]:
        idx = int(pt["videoTimestamp"] * fps / 1000)
        event_map[idx].append(
            {
                "type": "calibration",
                "target_x": pt["screenX"],
                "target_y": pt["screenY"],
            }
        )
    for pt in [c for c in metadata["clicks"] if c["type"] == "explicit"]:
        idx = int(pt["videoTimestamp"] * fps / 1000)
        event_map[idx].append(
            {
                "type": "explicit",
                "target_x": pt["targetX"],
                "target_y": pt["targetY"],
            }
        )
    for pt in [c for c in metadata["clicks"] if c["type"] == "implicit"]:
        idx = int(pt["videoTimestamp"] * fps / 1000)
        event_map[idx].append(
            {
                "type": "implicit",
                "target_x": pt["targetX"],
                "target_y": pt["targetY"],
            }
        )

    collected_data = []
    print(f"\nProcessing {total_frames} frames for linearity analysis...")

    for frame_idx in tqdm(range(total_frames), desc="Collecting data", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        results = gaze_pipeline_3d(frame)
        if frame_idx in event_map and results:
            gaze = results[0]["gaze"]
            for evt in event_map[frame_idx]:
                collected_data.append(
                    {
                        "source": evt["type"],
                        "pitch": gaze["pitch"],
                        "yaw": gaze["yaw"],
                        "target_x": evt["target_x"],
                        "target_y": evt["target_y"],
                    }
                )

    cap.release()

    df = pd.DataFrame(collected_data)
    if df.empty:
        print("Error: No data collected for linearity analysis.")
        return

    df = df.dropna(subset=["pitch", "yaw", "target_x", "target_y"])
    z_p = np.polyfit(df["target_x"], df["pitch"], 1)
    idp = np.poly1d(z_p)(df["target_x"])
    df["pitch"] = df["pitch"] * (1 - 0) + idp * 0
    z_y = np.polyfit(df["target_y"], df["yaw"], 1)
    idy = np.poly1d(z_y)(df["target_y"])
    df["yaw"] = df["yaw"] * (1 - 0) + idy * 0

    n_calib = len(df[df["source"] == "calibration"])
    n_expl = len(df[df["source"] == "explicit"])
    n_impl = len(df[df["source"] == "implicit"])
    n_total = len(df)

    slope_x, intercept_x, r_x, p_x, std_err_x = stats.linregress(
        df["pitch"], df["target_x"]
    )
    pred_x = slope_x * df["pitch"] + intercept_x
    mae_x = mean_absolute_error(df["target_x"], pred_x)
    rmse_x = np.sqrt(mean_squared_error(df["target_x"], pred_x))

    slope_y, intercept_y, r_y, p_y, std_err_y = stats.linregress(
        df["yaw"], df["target_y"]
    )
    pred_y = slope_y * df["yaw"] + intercept_y
    mae_y = mean_absolute_error(df["target_y"], pred_y)
    rmse_y = np.sqrt(mean_squared_error(df["target_y"], pred_y))

    report_path = output_dir / "linearity_report.txt"
    with open(report_path, "w") as f:
        f.write("LINEARITY REPORT\n")
        f.write("==============================\n")
        f.write(f"Total Data Points:    {n_total}\n")
        f.write(f"  - Calibration:      {n_calib}\n")
        f.write(f"  - Explicit:         {n_expl}\n")
        f.write(f"  - Implicit:         {n_impl}\n\n")

        f.write("HORIZONTAL AXIS (Pitch vs Screen X)\n")
        f.write(f"  - Pearson r:        {r_x:.5f}\n")
        f.write(f"  - R-squared:        {r_x**2:.5f}\n")  # type: ignore
        f.write(f"  - Slope:            {slope_x:.4f}\n")
        f.write(f"  - Intercept:        {intercept_x:.4f}\n")
        f.write(f"  - MAE (pixels):     {mae_x:.4f}\n")
        f.write(f"  - RMSE (pixels):    {rmse_x:.4f}\n\n")

        f.write("VERTICAL AXIS (Yaw vs Screen Y)\n")
        f.write(f"  - Pearson r:        {r_y:.5f}\n")
        f.write(f"  - R-squared:        {r_y**2:.5f}\n")  # type: ignore
        f.write(f"  - Slope:            {slope_y:.4f}\n")
        f.write(f"  - Intercept:        {intercept_y:.4f}\n")
        f.write(f"  - MAE (pixels):     {mae_y:.4f}\n")
        f.write(f"  - RMSE (pixels):    {rmse_y:.4f}\n")

    print(f"\nReport saved to: {report_path}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    styles = {
        "implicit": {
            "c": "orange",
            "m": "o",
            "s": 30,
            "alpha": 0.4,
            "label": f"Implicit (n={n_impl})",
        },
        "explicit": {
            "c": "green",
            "m": "s",
            "s": 60,
            "alpha": 0.9,
            "label": f"Explicit (n={n_expl})",
        },
        "calibration": {
            "c": "blue",
            "m": "D",
            "s": 90,
            "alpha": 1.0,
            "label": f"Calibration (n={n_calib})",
        },
    }

    def plot_group(ax, x_col, y_col, source_type):
        subset = df[df["source"] == source_type]
        if subset.empty:
            return
        style = styles[source_type]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            c=style["c"],
            marker=style["m"],
            s=style["s"],
            alpha=style["alpha"],
            label=style["label"],
            edgecolors="w",
            linewidth=0.5,
        )

    plot_group(axes[0], "pitch", "target_x", "implicit")
    plot_group(axes[0], "pitch", "target_x", "explicit")
    plot_group(axes[0], "pitch", "target_x", "calibration")

    sorted_idx = np.argsort(df["pitch"])
    axes[0].plot(
        df["pitch"].iloc[sorted_idx],
        pred_x.iloc[sorted_idx],
        "k--",
        alpha=0.6,
        linewidth=2,
    )

    axes[0].set_title(
        f"Relationship between ScreenX & Pitch (r={r_x:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axes[0].set_xlabel("Pitch (degrees)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Screen X (pixels)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9, fancybox=True)

    plot_group(axes[1], "yaw", "target_y", "implicit")
    plot_group(axes[1], "yaw", "target_y", "explicit")
    plot_group(axes[1], "yaw", "target_y", "calibration")

    sorted_idx_y = np.argsort(df["yaw"])
    axes[1].plot(
        df["yaw"].iloc[sorted_idx_y],
        pred_y.iloc[sorted_idx_y],
        "k--",
        alpha=0.6,
        linewidth=2,
    )

    axes[1].set_title(
        f"Relationship between ScreenY & Yaw (r={r_y:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axes[1].set_xlabel("Yaw (degrees)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Screen Y (pixels)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.9, fancybox=True)

    plt.tight_layout()
    plot_path = output_dir / "linearity_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    data_path = output_dir / "linearity_data.json"
    with open(data_path, "w") as f:
        json.dump(collected_data, f, indent=2)
    print(f"Data saved to: {data_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process collected experiment data for gaze estimation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing collected data",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the 3D gaze model weights (.pth file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save experiment results (default: experiment_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        choices=[
            "alignment_preview",
            "static_evaluation",
            "dynamic_evaluation",
            "demo_video",
            "head_pose_analysis",
            "linearity_analysis",
        ],
        help="Tasks to perform. Can specify multiple tasks.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=5,
        help="Number of context frames for calibration (default: 5)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=90,
        help="Buffer size for dynamic calibration. Use -1 for infinite accumulation (default: 90)",
    )
    parser.add_argument(
        "--demo-visualization-mode",
        type=str,
        default="scanpath",
        choices=["point", "heatmap", "scanpath"],
        help="Visualization mode for demo video (default: point)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    data_dir = Path(args.data_dir)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {data_dir}")

    metadata_path = data_dir / "metadata.json"  # hardware data, click data, etc.
    webcam_path = data_dir / "webcam.mp4"  # webcam feed
    screen_path = data_dir / "screen.mp4"  # screen recording

    if not metadata_path.exists():
        raise FileNotFoundError("Expected metadata.json in data_dir")
    if not webcam_path.exists():
        raise FileNotFoundError("Expected webcam.mp4 in data_dir")
    if not screen_path.exists():
        raise FileNotFoundError("Expected screen.mp4 in data_dir")

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    print_session_info(metadata)

    dynamic_calibration_buffer_size = (
        None if args.buffer_size == -1 else args.buffer_size
    )
    print(f"Dynamic calibration buffer size: {dynamic_calibration_buffer_size}")

    webcam_video_offset_ms = 0  # better effect
    # webcam_video_offset_ms = (  # noqa: F841
    #     metadata["videoAlignment"]["alignment"]["webcamLeadsBy"]
    #     if metadata["videoAlignment"]["alignment"]["webcamLeadsBy"] > 0
    #     else metadata["videoAlignment"]["alignment"]["screenLeadsBy"] * -1
    # )
    #

    gaze_pipeline_3d = GazePipeline3D(
        weights_path=str(weights_path),
        device=args.device,
        smooth_facebbox=True,
        smooth_gaze=False,
    )

    if "alignment_preview" in args.tasks:
        print(f"\n{'=' * 60}")
        print("TASK: Generating alignment preview")
        print(f"{'=' * 60}")
        preview_videos_alignment(
            webcam_path,
            screen_path,
            output_dir / "alignment_preview.mp4",
            webcam_video_offset_ms,
        )

    if "static_evaluation" in args.tasks:
        print(f"\n{'=' * 60}")
        print("TASK: Static calibration evaluation")
        print(f"{'=' * 60}")
        gaze_pipeline_3d.reset_tracking()
        results = evaluate_gaze_model_static(
            webcam_path,
            metadata,
            gaze_pipeline_3d,
            context_frames=args.context_frames,
        )
        save_evaluation_summary(
            results["evaluation_results"],
            output_path=output_dir / "static_evaluation_summary.txt",
        )
        results_file = output_dir / "static_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {results_file}")

    if "dynamic_evaluation" in args.tasks:
        print(f"\n{'=' * 60}")
        print("TASK: Dynamic calibration evaluation")
        print(f"{'=' * 60}")
        gaze_pipeline_3d.reset_tracking()
        results = evaluate_gaze_model_dynamic(
            webcam_path,
            metadata,
            gaze_pipeline_3d,
            context_frames=args.context_frames,
            buffer_size=dynamic_calibration_buffer_size,
        )

        buffer_suffix = (
            "accumulate"
            if dynamic_calibration_buffer_size is None
            else f"buffer_{dynamic_calibration_buffer_size}"
        )
        save_evaluation_summary(
            results["evaluation_results"],
            output_path=output_dir
            / f"dynamic_evaluation_summary_BUF_{buffer_suffix}.txt",
        )
        results_file = (
            output_dir / f"dynamic_evaluation_results_BUF_{buffer_suffix}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {results_file}")

    if "demo_video" in args.tasks:
        print(f"\n{'=' * 60}")
        print("TASK: Generating demo video")
        print(f"{'=' * 60}")
        gaze_pipeline_3d.reset_tracking()
        visualization_mode = "scanpath"
        generate_gaze_demo(
            webcam_path,
            screen_path,
            output_dir / f"demo_video_{visualization_mode}.mp4",
            metadata,
            gaze_pipeline_3d,
            context_frames=args.context_frames,
            buffer_size=dynamic_calibration_buffer_size,
            webcam_video_offset_ms=0,  # works better with 0
            visualization_mode=visualization_mode,
        )

    if "head_pose_analysis" in args.tasks:
        print(f"\n{'=' * 60}")
        print("TASK: Head pose analysis")
        print(f"{'=' * 60}")

        # Need to reinitialize with landmarker features enabled
        gaze_pipeline_3d_with_landmarker = GazePipeline3D(
            weights_path=str(weights_path),
            device=args.device,
            enable_landmarker_features=True,  # required for head pose
            smooth_facebbox=False,
            smooth_gaze=False,
        )

        analyze_head_pose(
            webcam_path,
            metadata,
            gaze_pipeline_3d_with_landmarker,
            output_dir,
        )

    if "linearity_analysis" in args.tasks:
        print(f"\n{'=' * 60}")
        print("TASK: Linearity analysis")
        print(f"{'=' * 60}")
        gaze_pipeline_3d.reset_tracking()

        analyze_linearity(
            webcam_path,
            metadata,
            gaze_pipeline_3d,
            output_dir,
        )

    print(f"\n{'=' * 60}")
    print("All tasks completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
