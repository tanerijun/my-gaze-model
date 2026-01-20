"""
Experiment with different calibration grid densities using 5x5 grid data.
(collected_data/calib_grid_density) or record a new one in calib_grid_alt branch of data collector project.
Evaluates how calibration point density affects gaze estimation accuracy.
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


CALIBRATION_CONFIGS = {
    "4-point": {
        "name": "Config 1 (n = 4)",
        "point_ids": ["r0c0", "r0c4", "r4c0", "r4c4"],
    },
    "5-point": {
        "name": "Config 2 (n = 5)",
        "point_ids": ["r0c0", "r0c4", "r2c2", "r4c0", "r4c4"],
    },
    "9-point": {
        "name": "Config 3 (n = 9)",
        "point_ids": [
            "r0c0",
            "r0c2",
            "r0c4",
            "r2c0",
            "r2c2",
            "r2c4",
            "r4c0",
            "r4c2",
            "r4c4",
        ],
    },
    "12-point": {
        "name": "Config 4 (n = 12)",
        "point_ids": [
            "r0c1",
            "r0c3",
            "r1c0",
            "r1c2",
            "r1c4",
            "r2c1",
            "r2c3",
            "r3c0",
            "r3c2",
            "r3c4",
            "r4c1",
            "r4c3",
        ],
    },
    "13-point": {
        "name": "Config 5 (n = 13)",
        "point_ids": [
            "r0c0",
            "r0c2",
            "r0c4",
            "r1c1",
            "r1c3",
            "r2c0",
            "r2c2",
            "r2c4",
            "r3c1",
            "r3c3",
            "r4c0",
            "r4c2",
            "r4c4",
        ],
    },
    "17-point": {
        "name": "Config 6 (n = 17)",
        "point_ids": [
            "r0c0",
            "r0c1",
            "r0c2",
            "r0c3",
            "r0c4",
            "r1c0",
            "r1c4",
            "r2c0",
            "r2c2",
            "r2c4",
            "r3c0",
            "r3c4",
            "r4c0",
            "r4c1",
            "r4c2",
            "r4c3",
            "r4c4",
        ],
    },
    "21-point": {
        "name": "Config 7 (n = 21)",
        "point_ids": [
            "r0c0",
            "r0c1",
            "r0c2",
            "r0c3",
            "r0c4",
            "r1c0",
            "r1c2",
            "r1c4",
            "r2c0",
            "r2c1",
            "r2c2",
            "r2c3",
            "r2c4",
            "r3c0",
            "r3c2",
            "r3c4",
            "r4c0",
            "r4c1",
            "r4c2",
            "r4c3",
            "r4c4",
        ],
    },
    "25-point": {
        "name": "Config 8 (n = 25)",
        "point_ids": [
            "r0c0",
            "r0c1",
            "r0c2",
            "r0c3",
            "r0c4",
            "r1c0",
            "r1c1",
            "r1c2",
            "r1c3",
            "r1c4",
            "r2c0",
            "r2c1",
            "r2c2",
            "r2c3",
            "r2c4",
            "r3c0",
            "r3c1",
            "r3c2",
            "r3c3",
            "r3c4",
            "r4c0",
            "r4c1",
            "r4c2",
            "r4c3",
            "r4c4",
        ],
    },
}


def load_metadata(metadata_path: Path) -> dict:
    """Load session metadata from JSON file."""
    with open(metadata_path, "r") as f:
        data = json.load(f)
    return data


def print_session_info(metadata: dict):
    print(f"Session ID: {metadata['sessionId']}")
    print("Participant info:")
    print(f"\tName: {metadata['participant']['name']}")
    print(f"\tAge: {metadata['participant']['age']}")
    print(f"\tGender: {metadata['participant']['gender']}")
    print("Video info:")
    print(
        f"\tScreen Resolution: W: {metadata['screenResolution']['width']}, "
        f"H: {metadata['screenResolution']['height']}"
    )
    print(
        f"\tWebcam Resolution: W: {metadata['webcamResolution']['width']}, "
        f"H: {metadata['webcamResolution']['height']}"
    )
    print("Click info:")
    print(f"\tExplicit click count: {metadata['gameMetadata']['totalExplicitClicks']}")


def filter_calibration_points(
    all_points: list[dict], point_ids: list[str]
) -> list[dict]:
    """
    Filter calibration points based on point IDs.

    Args:
        all_points: List of all calibration points (25 points from 5x5 grid)
        point_ids: List of point IDs to include (e.g., ['r0c0', 'r0c4', ...])

    Returns:
        Filtered list of calibration points
    """
    return [point for point in all_points if point["pointId"] in point_ids]


def train_mapper_with_config(
    webcam_path: Path,
    calibration_points: list[dict],
    gaze_pipeline_3d: GazePipeline3D,
    context_frames: int = 0,
) -> Mapper:
    """
    Train mapper using specified calibration points.

    Args:
        webcam_path: Path to webcam video
        calibration_points: List of calibration points to use
        gaze_pipeline_3d: 3D gaze pipeline for feature extraction
        context_frames: Number of frames before and after calibration frame

    Returns:
        Trained Mapper instance
    """
    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam video: {webcam_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    training_data = {}
    for point in calibration_points:
        point_id = point["pointId"]
        video_timestamp_ms = point["videoTimestamp"]

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

    cap.release()

    mapper = Mapper()
    for point_id, data in training_data.items():
        point = data["point"]
        feature_vectors = [[f["pitch"], f["yaw"]] for f in data["features"]]
        target_point = (point["screenX"], point["screenY"])
        mapper.add_calibration_point(feature_vectors, target_point)

    score_x, score_y = mapper.train()
    return mapper


def evaluate_with_mapper(
    webcam_path: Path,
    metadata: dict,
    mapper: Mapper,
    gaze_pipeline_3d: GazePipeline3D,
    feature_keys: list[str] = ["pitch", "yaw"],
    context_frames: int = 5,
) -> dict:
    """
    Evaluate gaze model using provided mapper on explicit clicks.

    Args:
        webcam_path: Path to webcam video
        metadata: Metadata dictionary
        mapper: Trained mapper to use for predictions
        gaze_pipeline_3d: 3D gaze pipeline
        feature_keys: Features to extract
        context_frames: Number of frames around click to evaluate

    Returns:
        Dictionary with evaluation results
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

    gaze_pipeline_2d = GazePipeline2D(
        gaze_pipeline_3d,
        mapper,
        feature_keys,
    )

    evaluation_results = []

    for click in tqdm(explicit_clicks, desc="Evaluating clicks", unit="click"):
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
        screen_width = metadata["screenResolution"]["width"]
        screen_height = metadata["screenResolution"]["height"]

        predictions = []
        errors = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise AssertionError(
                    f"Unexpected error reading frame {frame_idx} for click {click_id}"
                )

            results_2d = gaze_pipeline_2d.predict(frame)

            if not (results_2d and len(results_2d) > 0 and results_2d[0]["pog"]):
                raise AssertionError(f"Unexpected pipeline_2d output: {results_2d}")

            pog = results_2d[0]["pog"]

            pog_x = max(0, min(screen_width - 1, pog["x"]))
            pog_y = max(0, min(screen_height - 1, pog["y"]))

            predictions.append({"frame_idx": frame_idx, "x": pog_x, "y": pog_y})

            euclidean_distance = np.sqrt(
                (pog_x - target_x) ** 2 + (pog_y - target_y) ** 2
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
                "num_predictions": len(predictions),
                "errors_px": errors,
                "mean_error_px": float(mean_error),
                "median_error_px": float(median_error),
                "std_error_px": float(std_error),
                "percentile_95_error_px": float(percentile_95_error),
            }
        )

    cap.release()
    return {"evaluation_results": evaluation_results}


def save_text_report(results_by_config: dict, output_dir: Path):
    report_path = output_dir / "grid_density_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CALIBRATION GRID DENSITY EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Configuration':<30} {'Points':<8} {'Mean (px)':<12} {'Median (px)':<12} {'95th %ile (px)':<12}\n"
        )
        f.write("-" * 80 + "\n")

        sorted_configs = sorted(
            results_by_config.items(), key=lambda x: x[1]["num_calibration_points"]
        )

        for config_key, result in sorted_configs:
            all_errors = []
            for eval_result in result["evaluation_results"]:
                all_errors.extend(eval_result["errors_px"])

            mean_err = np.mean(all_errors)
            median_err = np.median(all_errors)
            p95_err = np.percentile(all_errors, 95)
            num_points = result["num_calibration_points"]
            config_name = result["config_name"]

            f.write(
                f"{config_name:<30} {num_points:<8} {mean_err:<12.2f} {median_err:<12.2f} {p95_err:<12.2f}\n"
            )

        f.write("-" * 80 + "\n\n")

        f.write("DETAILED STATISTICS BY CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")

        for config_key, result in sorted_configs:
            all_errors = []
            for eval_result in result["evaluation_results"]:
                all_errors.extend(eval_result["errors_px"])

            f.write(f"{result['config_name']}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Number of calibration points: {result['num_calibration_points']}\n"
            )
            f.write(
                f"Calibration point IDs: {', '.join(result['calibration_point_ids'])}\n"
            )
            f.write(f"Number of test clicks: {len(result['evaluation_results'])}\n")
            f.write(f"Total predictions: {len(all_errors)}\n")
            f.write("\nError Statistics:\n")
            f.write(f"  Mean:             {np.mean(all_errors):.2f} px\n")
            f.write(f"  Median:           {np.median(all_errors):.2f} px\n")
            f.write(f"  Std Dev:          {np.std(all_errors):.2f} px\n")
            f.write(f"  Min:              {np.min(all_errors):.2f} px\n")
            f.write(f"  Max:              {np.max(all_errors):.2f} px\n")
            f.write(f"  25th Percentile:  {np.percentile(all_errors, 25):.2f} px\n")
            f.write(f"  75th Percentile:  {np.percentile(all_errors, 75):.2f} px\n")
            f.write(f"  95th Percentile:  {np.percentile(all_errors, 95):.2f} px\n")
            f.write(f"  99th Percentile:  {np.percentile(all_errors, 99):.2f} px\n")

            if "mapper_stats" in result:
                stats = result["mapper_stats"]
                f.write("\nMapper Training Statistics:\n")
                f.write(f"  Training samples: {stats.get('initial_samples', 'N/A')}\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"\nText report saved to: {report_path}")


def generate_accuracy_bar_chart(results_by_config: dict, output_dir: Path):
    """Generate bar chart showing mean accuracy for each configuration."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping plot generation.")
        return

    config_names = []
    mean_errors = []
    num_points = []

    sorted_configs = sorted(
        results_by_config.items(), key=lambda x: x[1]["num_calibration_points"]
    )

    for config_key, result in sorted_configs:
        config_names.append(result["config_name"])
        num_points.append(result["num_calibration_points"])

        errors = []
        for eval_result in result["evaluation_results"]:
            errors.extend(eval_result["errors_px"])

        mean_errors.append(np.mean(errors))

    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(
        range(len(config_names)),
        mean_errors,
        color="steelblue",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
        width=0.6,
    )

    for i, (bar, error) in enumerate(zip(bars, mean_errors)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{error:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    ax.set_xlabel("Calibration Configuration", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Error (pixels)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, fontsize=11, rotation=0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    min_error = min(mean_errors)
    ax.axhline(
        y=min_error,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label=f"Best: {min_error:.1f}px",
    )
    ax.legend(loc="upper right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "grid_density_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Accuracy chart saved to: {output_dir / 'grid_density_accuracy.png'}")


def generate_grid_visualization(output_dir: Path):
    """Generate visualization of different grid configurations for thesis."""
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping visualization generation.")
        return

    # Define grid positions with proper margins
    # Make horizontal spacing wider for 16:9-like aspect ratio
    grid_x_positions = [1, 2.6, 4.2, 5.8, 7.4]  # Wider horizontal spacing
    grid_y_positions = [1, 2, 3, 4, 5]  # Normal vertical spacing

    configs_to_show = {
        "4-point": CALIBRATION_CONFIGS["4-point"],
        "5-point": CALIBRATION_CONFIGS["5-point"],
        "9-point": CALIBRATION_CONFIGS["9-point"],
        "12-point": CALIBRATION_CONFIGS["12-point"],
        "13-point": CALIBRATION_CONFIGS["13-point"],
        "17-point": CALIBRATION_CONFIGS["17-point"],
        "21-point": CALIBRATION_CONFIGS["21-point"],
        "25-point": CALIBRATION_CONFIGS["25-point"],
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    point_radius = 0.15

    for idx, (config_key, config) in enumerate(configs_to_show.items()):
        ax = axes[idx]

        for i in range(5):  # rows
            for j in range(5):  # cols
                point_id = f"r{i}c{j}"
                x_pos = grid_x_positions[j]
                y_pos = grid_y_positions[4 - i]  # Flip y-axis so r0 is at top

                # Check if this point is in the configuration
                if point_id in config["point_ids"]:
                    circle = plt.Circle(
                        (x_pos, y_pos), point_radius, color="red", zorder=3, alpha=0.9
                    )
                    ax.add_patch(circle)
                    circle_border = plt.Circle(
                        (x_pos, y_pos),
                        point_radius,
                        color="white",
                        fill=False,
                        linewidth=2,
                        zorder=4,
                    )
                    ax.add_patch(circle_border)
                else:
                    circle = plt.Circle(
                        (x_pos, y_pos),
                        point_radius,
                        color="lightgray",
                        zorder=2,
                        alpha=0.5,
                    )
                    ax.add_patch(circle)

        for y_pos in grid_y_positions:
            ax.plot(
                [grid_x_positions[0], grid_x_positions[-1]],
                [y_pos, y_pos],
                "k-",
                linewidth=0.5,
                alpha=0.3,
                zorder=1,
            )
        for x_pos in grid_x_positions:
            ax.plot(
                [x_pos, x_pos],
                [grid_y_positions[0], grid_y_positions[-1]],
                "k-",
                linewidth=0.5,
                alpha=0.3,
                zorder=1,
            )

        screen_margin = 0.25
        screen_x_start = grid_x_positions[0] - screen_margin
        screen_y_start = grid_y_positions[0] - screen_margin
        screen_width = grid_x_positions[-1] - grid_x_positions[0] + 2 * screen_margin
        screen_height = grid_y_positions[-1] - grid_y_positions[0] + 2 * screen_margin

        rect = patches.Rectangle(
            (screen_x_start, screen_y_start),
            screen_width,
            screen_height,
            linewidth=3,
            edgecolor="black",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

        x_padding = 0.2
        y_padding = 0.2
        ax.set_xlim(
            screen_x_start - x_padding, screen_x_start + screen_width + x_padding
        )
        ax.set_ylim(
            screen_y_start - y_padding, screen_y_start + screen_height + y_padding
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(config["name"], fontsize=15, fontweight="bold", pad=5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.suptitle(
        "Calibration Grid Configurations",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(h_pad=1)
    plt.savefig(
        output_dir / "grid_configurations_visualization.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Grid visualization saved to: {output_dir / 'grid_configurations_visualization.png'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experiment with different calibration grid densities using 5x5 grid data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing collected data (with 25-point calibration)",
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
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        choices=list(CALIBRATION_CONFIGS.keys()),
        help="Calibration configurations to test. If not specified, tests all configurations.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=5,
        help="Number of context frames for calibration (default: 5)",
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

    metadata_path = data_dir / "metadata.json"
    webcam_path = data_dir / "webcam.mp4"

    if not metadata_path.exists():
        raise FileNotFoundError("Expected metadata.json in data_dir")
    if not webcam_path.exists():
        raise FileNotFoundError("Expected webcam.mp4 in data_dir")

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    print_session_info(metadata)

    # Get all 25 calibration points
    all_calibration_points = metadata["initialCalibration"]["points"]
    print(f"\nTotal calibration points collected: {len(all_calibration_points)}")

    # Determine which configurations to test
    configs_to_test = args.configs if args.configs else list(CALIBRATION_CONFIGS.keys())
    print(
        f"\nTesting {len(configs_to_test)} configurations: {', '.join(configs_to_test)}"
    )

    print(f"\n{'=' * 80}")
    print("Generating grid configuration visualization...")
    print(f"{'=' * 80}")
    generate_grid_visualization(output_dir)

    gaze_pipeline_3d = GazePipeline3D(
        weights_path=str(weights_path),
        device=args.device,
        smooth_facebbox=True,
        smooth_gaze=False,
    )

    results_by_config = {}

    for config_key in configs_to_test:
        config = CALIBRATION_CONFIGS[config_key]
        config_name = config["name"]
        point_ids = config["point_ids"]

        print(f"\n{'=' * 80}")
        print(f"Testing Configuration: {config_name} ({len(point_ids)} points)")
        print(f"{'=' * 80}")

        # Filter calibration points for this configuration
        filtered_points = filter_calibration_points(all_calibration_points, point_ids)
        print(f"Using calibration points: {[p['pointId'] for p in filtered_points]}")

        # Reset pipeline state
        gaze_pipeline_3d.reset_tracking()

        print("\nTraining mapper...")
        mapper = train_mapper_with_config(
            webcam_path,
            filtered_points,
            gaze_pipeline_3d,
            context_frames=args.context_frames,
        )

        stats = mapper.get_training_stats()
        print("Training complete:")
        print(f"  Total training samples: {stats['initial_samples']}")

        gaze_pipeline_3d.reset_tracking()

        print("\nEvaluating on explicit clicks...")
        results = evaluate_with_mapper(
            webcam_path,
            metadata,
            mapper,
            gaze_pipeline_3d,
            context_frames=args.context_frames,
        )

        all_errors = []
        for eval_result in results["evaluation_results"]:
            all_errors.extend(eval_result["errors_px"])

        print(f"\nResults for {config_name}:")
        print(f"  Mean error: {np.mean(all_errors):.2f}px")
        print(f"  Median error: {np.median(all_errors):.2f}px")
        print(f"  Std error: {np.std(all_errors):.2f}px")
        print(f"  95th percentile: {np.percentile(all_errors, 95):.2f}px")

        results_by_config[config_key] = {
            "config_name": config_name,
            "num_calibration_points": len(point_ids),
            "calibration_point_ids": point_ids,
            "evaluation_results": results["evaluation_results"],
            "mapper_stats": stats,
        }

    print(f"\n{'=' * 80}")
    print("Generating final outputs...")
    print(f"{'=' * 80}")

    save_text_report(results_by_config, output_dir)
    generate_accuracy_bar_chart(results_by_config, output_dir)

    print(f"\n{'=' * 80}")
    print("Grid density experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("  - grid_density_report.txt")
    print("  - grid_density_accuracy.png")
    print("  - grid_configurations_visualization.png")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
