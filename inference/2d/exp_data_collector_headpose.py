"""
Head Pose Analysis for Experiment Data
Analyzes how user's head pose changes throughout the session relative to calibration.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import GazePipeline3D


def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r") as f:
        data = json.load(f)
    return data


def extract_calibration_head_pose(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
) -> dict[str, float]:
    """
    Extract average head pose during initial calibration phase.

    Args:
        webcam_path: Path to webcam video
        metadata: Session metadata
        gaze_pipeline_3d: 3D gaze pipeline with landmarker features enabled

    Returns:
        Dictionary with average pitch, yaw, roll during calibration
    """
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
        raise ValueError(
            "Could not extract head pose from calibration frames. "
            "Make sure enable_landmarker_features=True"
        )

    # Calculate average and std
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


def analyze_session_head_pose(
    webcam_path: Path,
    metadata: dict,
    gaze_pipeline_3d: GazePipeline3D,
    calibration_pose: dict[str, float],
) -> list[dict]:
    """
    Analyze head pose throughout the entire session.

    Args:
        webcam_path: Path to webcam video
        metadata: Session metadata
        gaze_pipeline_3d: 3D gaze pipeline with landmarker features enabled
        calibration_pose: Reference head pose from calibration

    Returns:
        List of head pose data for each frame
    """
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

    for frame_idx in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
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

                # Calculate deviations from calibration
                pitch_dev = current_pitch - calibration_pose["pitch"]
                yaw_dev = current_yaw - calibration_pose["yaw"]
                roll_dev = current_roll - calibration_pose["roll"]

                # Calculate total angular deviation (Euclidean distance in angle space)
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
            # No face detected
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


def generate_head_pose_report(
    session_data: list[dict],
    calibration_pose: dict[str, float],
    output_dir: Path,
):
    """
    Generate visualizations and statistics for head pose analysis.

    Args:
        session_data: Head pose data for each frame
        calibration_pose: Reference calibration pose
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out frames with no detection
    valid_data = [d for d in session_data if d["total_deviation"] is not None]

    if not valid_data:
        print("Warning: No valid head pose data found")
        return

    # Separate calibration and game phases
    calibration_data = [d for d in valid_data if not d["is_game_phase"]]
    game_data = [d for d in valid_data if d["is_game_phase"]]

    print(f"\n{'=' * 60}")
    print("HEAD POSE ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total frames analyzed: {len(session_data)}")
    print(
        f"Valid detections: {len(valid_data)} ({len(valid_data) / len(session_data) * 100:.1f}%)"
    )
    print(f"Calibration phase: {len(calibration_data)} frames")
    print(f"Game phase: {len(game_data)} frames")

    if game_data:
        game_deviations = [d["total_deviation"] for d in game_data]
        print("\nGame Phase Head Movement:")
        print(f"  Mean deviation: {np.mean(game_deviations):.2f}°")
        print(f"  Median deviation: {np.median(game_deviations):.2f}°")
        print(f"  Std deviation: {np.std(game_deviations):.2f}°")
        print(f"  Max deviation: {np.max(game_deviations):.2f}°")
        print(f"  95th percentile: {np.percentile(game_deviations, 95):.2f}°")

    # Generate plots
    _plot_head_pose_over_time(valid_data, calibration_pose, output_dir)
    _plot_deviation_distribution(game_data, output_dir)
    _plot_3d_trajectory(valid_data, calibration_pose, output_dir)

    # Save detailed statistics
    stats = {
        "calibration_pose": calibration_pose,
        "total_frames": len(session_data),
        "valid_detections": len(valid_data),
        "detection_rate": len(valid_data) / len(session_data),
    }

    if game_data:
        game_deviations = [d["total_deviation"] for d in game_data]
        stats["game_phase"] = {
            "num_frames": len(game_data),
            "mean_deviation": float(np.mean(game_deviations)),
            "median_deviation": float(np.median(game_deviations)),
            "std_deviation": float(np.std(game_deviations)),
            "max_deviation": float(np.max(game_deviations)),
            "p95_deviation": float(np.percentile(game_deviations, 95)),
        }

    stats_file = output_dir / "head_pose_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

    # Save full session data
    data_file = output_dir / "head_pose_data.json"
    with open(data_file, "w") as f:
        json.dump(session_data, f, indent=2)
    print(f"Full data saved to: {data_file}")


def _plot_head_pose_over_time(
    session_data: list[dict],
    calibration_pose: dict[str, float],
    output_dir: Path,
):
    """Plot head pose angles and deviations over time."""
    timestamps = [d["timestamp_ms"] / 1000 for d in session_data]  # Convert to seconds

    # Separate calibration and game phases for the boxplot
    calib_data = [d for d in session_data if not d["is_game_phase"]]
    game_data = [d for d in session_data if d["is_game_phase"]]

    # Plot 1: Raw angles
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        timestamps,
        [d["pitch"] for d in session_data],
        label="Pitch",
        alpha=0.4,
        linewidth=1.5,
    )
    ax.plot(
        timestamps,
        [d["yaw"] for d in session_data],
        label="Yaw",
        alpha=0.4,
        linewidth=1.5,
    )
    ax.plot(
        timestamps,
        [d["roll"] for d in session_data],
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
    output_file = output_dir / "head_pose_angles.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")

    # Plot 2: Deviations
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        timestamps,
        [abs(d["pitch_deviation"]) for d in session_data],
        label="Pitch Dev",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        timestamps,
        [abs(d["yaw_deviation"]) for d in session_data],
        label="Yaw Dev",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        timestamps,
        [abs(d["roll_deviation"]) for d in session_data],
        label="Roll Dev",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Deviation (degrees)", fontsize=12)
    ax.set_title("Absolute Deviations from Calibration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = output_dir / "head_pose_deviations.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")

    # Plot 3: Total deviation
    fig, ax = plt.subplots(figsize=(12, 6))
    total_devs = [d["total_deviation"] for d in session_data]
    ax.plot(timestamps, total_devs, color="red", alpha=0.7, linewidth=1.5)
    ax.axhline(
        np.mean(total_devs),  # type: ignore
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(total_devs):.2f}°",
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Total Deviation (degrees)", fontsize=12)
    ax.set_title("Total Angular Deviation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = output_dir / "head_pose_total_deviation.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")

    # Plot 4: Phase comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    if calib_data and game_data:
        calib_devs = [d["total_deviation"] for d in calib_data]
        game_devs = [d["total_deviation"] for d in game_data]

        ax.boxplot([calib_devs, game_devs], labels=["Calibration", "Game"])  # type: ignore
        ax.set_ylabel("Total Deviation (degrees)", fontsize=12)
        ax.set_title("Deviation by Phase", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "head_pose_phase_comparison.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")


def _plot_deviation_distribution(game_data: list[dict], output_dir: Path):
    """Plot distribution of head pose deviations during game phase."""
    if not game_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Head Pose Deviation Distribution (Game Phase)", fontsize=14)

    # Histogram
    ax = axes[0]
    deviations = [d["total_deviation"] for d in game_data]
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

    # Cumulative distribution
    ax = axes[1]
    sorted_devs = np.sort(deviations)
    cumulative = np.arange(1, len(sorted_devs) + 1) / len(sorted_devs) * 100
    ax.plot(sorted_devs, cumulative)
    ax.axhline(50, color="green", linestyle="--", alpha=0.5, label="50th percentile")
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, label="95th percentile")
    ax.set_xlabel("Total Deviation (degrees)")
    ax.set_ylabel("Cumulative Percentage")
    ax.set_title("Cumulative Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "deviation_distribution.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")


def _plot_3d_trajectory(
    session_data: list[dict],
    calibration_pose: dict[str, float],
    output_dir: Path,
):
    """Plot 3D trajectory of head pose in pitch-yaw-roll space."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Separate by phase
    calib_data = [d for d in session_data if not d["is_game_phase"]]
    game_data = [d for d in session_data if d["is_game_phase"]]

    if calib_data:
        ax.scatter(
            [d["pitch"] for d in calib_data],
            [d["yaw"] for d in calib_data],
            [d["roll"] for d in calib_data],  # type: ignore
            c="blue",
            alpha=0.3,
            s=10,
            label="Calibration",
        )

    if game_data:
        # Color by deviation magnitude
        deviations = [d["total_deviation"] for d in game_data]
        scatter = ax.scatter(
            [d["pitch"] for d in game_data],
            [d["yaw"] for d in game_data],
            [d["roll"] for d in game_data],  # type: ignore
            c=deviations,
            cmap="hot",
            alpha=0.5,
            s=10,
            label="Game",
        )
        plt.colorbar(scatter, ax=ax, label="Deviation (degrees)")

    # Mark calibration center
    ax.scatter(
        [calibration_pose["pitch"]],
        [calibration_pose["yaw"]],
        [calibration_pose["roll"]],  # type: ignore
        c="green",
        marker="*",
        s=200,
        label="Calibration Center",
        edgecolors="black",
    )

    ax.set_xlabel("Pitch (degrees)")
    ax.set_ylabel("Yaw (degrees)")
    ax.set_zlabel("Roll (degrees)")
    ax.set_title("Head Pose Trajectory in 3D Space")
    ax.legend()

    output_file = output_dir / "head_pose_3d_trajectory.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze head pose changes throughout experiment session"
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
        help="Directory to save analysis results",
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
    output_dir = Path(args.output_dir)

    metadata_path = data_dir / "metadata.json"
    webcam_path = data_dir / "webcam.mp4"

    if not metadata_path.exists():
        raise FileNotFoundError("Expected metadata.json in data_dir")
    if not webcam_path.exists():
        raise FileNotFoundError("Expected webcam.mp4 in data_dir")

    metadata = load_metadata(metadata_path)

    print(f"\n{'=' * 60}")
    print("HEAD POSE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Session ID: {metadata['sessionId']}")

    # Initialize pipeline with landmarker features enabled
    gaze_pipeline_3d = GazePipeline3D(
        weights_path=args.weights,
        device=args.device,
        enable_landmarker_features=True,  # Required for head pose
        smooth_facebbox=False,  # Disable smoothing for analysis
        smooth_gaze=False,
    )

    # Extract calibration head pose
    calibration_pose = extract_calibration_head_pose(
        webcam_path, metadata, gaze_pipeline_3d
    )

    # Reset tracking before full session analysis
    gaze_pipeline_3d.reset_tracking()

    # Analyze entire session
    session_data = analyze_session_head_pose(
        webcam_path, metadata, gaze_pipeline_3d, calibration_pose
    )

    # Generate report and visualizations
    generate_head_pose_report(session_data, calibration_pose, output_dir)

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
