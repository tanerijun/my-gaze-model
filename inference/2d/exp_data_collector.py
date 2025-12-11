"""
Process experiment data collected from the data collector platform (The Deep Value (web))
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
    webcam_path: Path, screen_path: Path, webcam_offset_ms: float = 0
):
    """
    A helper method that output videos put side by side to visually inspect if they align perfectly

    Args:
        webcam_path: Path to webcam video
        screen_path: Path to screen video
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
    output_path = webcam_path.parent / "alignment_preview.mp4"
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
    webcam_scale: float = 0.25,  # scale factor for webcam overlay
    point_radius: int = 15,
    point_color: tuple = (0, 255, 0),  # BGR
    point_thickness: int = -1,  # -1 for filled circle
    heatmap_radius: int = 80,
    heatmap_decay: float = 0.95,
    scanpath_length: int = 30,
    scanpath_thickness: int = 3,
):
    """
    Generate a demo video with gaze visualization overlaid on screen recording,
    with webcam feed in the bottom-right corner.
    """
    print(f"\n{'=' * 60}")
    print("GENERATING GAZE DEMO VIDEO")
    print(f"{'=' * 60}")
    print(f"Visualization mode: {visualization_mode}")
    print(f"Webcam offset: {webcam_video_offset_ms}ms")

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

    print(
        f"\nScreen: {screen_width}x{screen_height} @ {screen_fps:.2f} FPS, {total_screen_frames} frames"
    )
    print(
        f"Webcam: {webcam_width}x{webcam_height} @ {webcam_fps:.2f} FPS, {total_webcam_frames} frames"
    )

    # Calculate webcam overlay dimensions
    overlay_height = int(screen_height * webcam_scale)
    overlay_width = int(webcam_width * overlay_height / webcam_height)

    # Calculate webcam offset in frames
    webcam_offset_frames = int((webcam_video_offset_ms / 1000.0) * webcam_fps)

    # Close initial captures
    webcam_cap.release()
    screen_cap.release()

    # == Compute Gaze Predictions ==
    print("\n" + "=" * 60)
    print("Computing gaze predictions from webcam feed")
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

        # Compute gaze prediction
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
    print("PHASE 2: Generating demo video")
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

    for screen_frame_idx in tqdm(
        range(total_screen_frames), desc="Rendering demo", unit="frame"
    ):
        ret_screen, screen_frame = screen_cap.read()
        if not ret_screen:
            print(f"Warning: Could not read screen frame {screen_frame_idx}")
            break

        # Calculate corresponding webcam frame (with offset)
        time_s = screen_frame_idx / screen_fps
        webcam_frame_idx = int(time_s * webcam_fps) + webcam_offset_frames
        webcam_frame_idx = max(0, min(webcam_frame_idx, total_webcam_frames - 1))

        # Look up pre-computed gaze prediction
        gaze_point = gaze_predictions.get(webcam_frame_idx)

        # Read webcam frame for overlay (it's ok to seek now)
        webcam_cap.set(cv2.CAP_PROP_POS_FRAMES, webcam_frame_idx)
        ret_webcam, webcam_frame = webcam_cap.read()
        if not ret_webcam:
            webcam_frame = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)

        # Create output frame
        output_frame = screen_frame.copy()

        if gaze_point:
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

    # Cleanup
    webcam_cap.release()
    screen_cap.release()
    out.release()

    print(f"\n{'=' * 60}")
    print(f"Demo video saved to: {output_path}")
    print(f"{'=' * 60}")


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

    metadata = load_metadata(metadata_path)
    print_session_info(metadata)

    webcam_video_offset_ms = (  # noqa: F841
        metadata["videoAlignment"]["alignment"]["webcamLeadsBy"]
        if metadata["videoAlignment"]["alignment"]["webcamLeadsBy"] > 0
        else metadata["videoAlignment"]["alignment"]["screenLeadsBy"]
    )

    # Uncomment to generate a side by side preview
    # preview_videos_alignment(
    #     webcam_path,
    #     screen_path,
    #     webcam_video_offset_ms,
    # )

    gaze_pipeline_3d = GazePipeline3D(
        weights_path=str(weights_path),
        device=args.device,
        smooth_facebbox=True,
        smooth_gaze=False,
    )

    ### STATIC MODE ###
    # results = evaluate_gaze_model_static(
    #     webcam_path,
    #     metadata,
    #     gaze_pipeline_3d,
    #     context_frames=5,
    # )

    # save_evaluation_summary(
    #     results["evaluation_results"],
    #     output_path=output_dir / "evaluation_summary_static.txt",
    # )

    # results_file = output_dir / "evaluation_results_static.json"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # with open(results_file, "w") as f:
    #     json.dump(results, f, indent=2)
    # print(f"\nEvaluation results saved to: {results_file}")
    ####################

    buffer_size = 110

    ### DYNAMIC MODE ###
    gaze_pipeline_3d.reset_tracking()  # in case it's used before
    results = evaluate_gaze_model_dynamic(
        webcam_path,
        metadata,
        gaze_pipeline_3d,
        context_frames=5,
        buffer_size=buffer_size,
    )

    buffer_suffix = "accumulate" if buffer_size is None else f"buffer_{buffer_size}"
    save_evaluation_summary(
        results["evaluation_results"],
        output_path=output_dir / f"evaluation_symmary_dynamic_{buffer_suffix}.txt",
    )

    results_file = output_dir / f"evaluation_results_dynamic_{buffer_suffix}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_file}")
    ###################

    ### DEMO ###
    # gaze_pipeline_3d.reset_tracking()
    # visualization_mode = "scanpath"
    # generate_gaze_demo(
    #     webcam_path,
    #     screen_path,
    #     output_dir / f"demo_{visualization_mode}.mp4",
    #     metadata,
    #     gaze_pipeline_3d,
    #     context_frames=5,
    #     buffer_size=buffer_size,
    #     webcam_video_offset_ms=webcam_video_offset_ms,
    #     visualization_mode=visualization_mode,
    # )
    ############


if __name__ == "__main__":
    main()
