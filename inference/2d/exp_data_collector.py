"""
Process experiment data collected from the data collector platform (The Deep Value (web))
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

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


def train_and_get_mapper(
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


def evaluate_gaze_model(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_2d: GazePipeline2D,
    context_frames: int = 5,
):
    """
    Evaluate the 2D gaze pipeline on explicit clicks. (with seek, no implicit calibration)

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

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    evaluation_results = []

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

        target_x = click["screenX"]
        target_y = click["screenY"]

        print(f"\nEvaluating click {click_id}:")
        print(f"  Center frame: {center_frame_idx}")
        print(f"  Ground truth: ({target_x:.1f}, {target_y:.1f})")

        frame_predictions = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise AssertionError(f"Error reading frame {frame_idx}")

            results_2d = gaze_pipeline_2d.predict(frame)

            if not (results_2d and len(results_2d) > 0 and results_2d[0]["pog"]):
                raise AssertionError(f"Unexpected pipeline_2d output: {results_2d}")

            pog = results_2d[0]["pog"]
            frame_predictions.append(
                {"frame_idx": frame_idx, "x": pog["x"], "y": pog["y"]}
            )

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
                "videoTimestamp": video_timestamp_ms,
                "ground_truth": {"x": target_x, "y": target_y},
                "num_predictions": len(frame_predictions),
                "errors_px": errors,
                "mean_error_px": float(mean_error),
                "median_error_px": float(median_error),
                "std_error_px": float(std_error),
                "predictions": frame_predictions,
            }
        )

        print(f"  Mean error: {mean_error:.2f}px")
        print(f"  Median error: {median_error:.2f}px")
        print(f"  Std error: {std_error:.2f}px")
        print(f"  95th percentile error: {percentile_95_error:.2f}px")

    cap.release()
    return evaluation_results


def evaluate_gaze_model_1(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_2d: GazePipeline2D,
    context_frames: int = 5,
):
    """
    Evaluate the 2D gaze pipeline on explicit clicks by processing frames sequentially.
    Respecting dependencies on previous frames. (can see face_bbox effect, but no still no dynamic calibration)

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

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                click_data["predictions"].append(
                    {"frame_idx": frame_idx, "x": pog["x"], "y": pog["y"]}
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

    return evaluation_results


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

    webcam_video_offset_ms = (
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

    mapper = train_and_get_mapper(
        webcam_path, metadata, gaze_pipeline_3d, context_frames=5
    )

    gaze_pipeline_2d = GazePipeline2D(
        gaze_pipeline_3d,
        mapper,
        ["pitch", "yaw"],
    )

    evaluation_results = evaluate_gaze_model_1(
        webcam_path,
        metadata,
        gaze_pipeline_2d,
        context_frames=5,
    )

    save_evaluation_summary(
        evaluation_results, output_path=output_dir / "evalutation_summary.txt"
    )

    # Save evaluation result
    results_file = output_dir / "evaluation_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_file}")


if __name__ == "__main__":
    main()
