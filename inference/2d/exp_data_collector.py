"""
Process experiment data collected from the data collector platform (The Deep Value (web))
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


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
    preview_videos_alignment(
        webcam_path,
        screen_path,
        webcam_video_offset_ms,
    )


if __name__ == "__main__":
    main()
