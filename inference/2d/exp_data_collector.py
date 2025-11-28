"""
Process experiment data collected from the data collector platform.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from src.inference import GazePipeline2D, GazePipeline3D, Mapper


class DataCollectorProcessor:
    """Process data collected from the data collection platform."""

    def __init__(self, data_dir: str, weights_path: str, device: str = "cpu"):
        """
        Initialize the processor.

        Args:
            data_dir: Directory containing the collected data (JSON and video files)
            weights_path: Path to the 3D gaze model weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.data_dir = Path(data_dir)
        self.weights_path = weights_path
        self.device = device

        # Check if directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")

        # Find session JSON files
        json_files = list(self.data_dir.glob("session_*.json"))

        # Check for exactly 1 JSON file
        if len(json_files) == 0:
            raise FileNotFoundError(
                f"No session JSON files found in {data_dir}. "
                f"Expected file pattern: session_*.json"
            )
        elif len(json_files) > 1:
            found_files = "\n  ".join([f.name for f in json_files])
            raise ValueError(
                f"Found {len(json_files)} session JSON files in {data_dir}, "
                f"but expected exactly 1.\nFiles found:\n  {found_files}"
            )

        print(f"Found session file: {json_files[0].name}")
        self.json_path = json_files[0]

        self.pipeline_3d = GazePipeline3D(
            weights_path=weights_path,
            device=device,
            smooth_facebbox=True,
            smooth_gaze=True,
        )
        self.mapper = Mapper()
        self.pipeline_2d = GazePipeline2D(
            pipeline_3d=self.pipeline_3d,
            mapper=self.mapper,
            feature_keys=["pitch", "yaw"],
        )

    def load_session_data(self, json_path: Path) -> Dict:
        """
        Load session data from JSON file.

        Args:
            json_path: Path to the session JSON file

        Returns:
            Dictionary containing session data
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def train_initial_mapper(self, session_data: Dict) -> int:
        """
        Extract calibration data from session and train the mapper.

        Args:
            session_data: Session data dictionary

        Returns:
            num_points
        """
        calibration_points = session_data["calibration"]["calibration_points"]
        print(f"\nProcessing {len(calibration_points)} calibration points...")

        # Populate mapper with calibration data from stored gaze_result
        num_valid_points = 0
        for point in calibration_points:
            target_coords = (point["target"]["x"], point["target"]["y"])

            # Use the stored gaze_result instead of re-running inference
            if "gaze_result" not in point:
                raise AssertionError("Expected gaze_result in calibration point")

            gaze_result = point["gaze_result"]

            try:
                feature_vector = self.pipeline_2d.extract_feature_vector(gaze_result)
                self.mapper.add_calibration_point([feature_vector], target_coords)
                num_valid_points += 1
            except ValueError as e:
                print(f"Warning: Could not extract features: {e}")
                continue

        print(f"Added {num_valid_points} calibration points to mapper")
        print("Training mapper...")

        try:
            score_x, score_y = self.mapper.train()
            print(f"Mapper trained. R² scores -> X: {score_x:.3f}, Y: {score_y:.3f}")
            return num_valid_points
        except ValueError as e:
            print(f"Error during training: {e}")
            raise

    def evaluate_without_dynamic_calibration(self, session_data: Dict) -> pd.DataFrame:
        """
        Evaluate the trained mapper without dynamic calibration.

        This is the baseline approach: train once on calibration data,
        then evaluate on all test points without updating the mapper.

        Args:
            session_data: Session data dictionary

        Returns:
            DataFrame with evaluation results
        """
        explicit_points = session_data["click_events"]["explicit_points"]
        print(
            f"\nEvaluating without dynamic calibration on {len(explicit_points)} test points..."
        )

        # Evaluate predictions using stored gaze_result
        results = []
        for point in explicit_points:
            gt_x = point["target"]["x"]
            gt_y = point["target"]["y"]

            # Use the stored gaze_result instead of re-running inference
            if "gaze_result" not in point:
                raise AssertionError("Expected gaze_result for each explicit point")

            gaze_result = point["gaze_result"]

            # Extract feature vector and make prediction
            try:
                feature_vector = self.pipeline_2d.extract_feature_vector(gaze_result)
                pred_coords = self.mapper.predict(feature_vector)
                pred_x, pred_y = pred_coords[0], pred_coords[1]
            except ValueError as e:
                print(f"Warning: Could not predict for point: {e}")
                continue

            # Calculate error
            error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            results.append(
                {
                    "timestamp": point["video_timestamp"],
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "error": error,
                }
            )

        return pd.DataFrame(results)

    def evaluate_with_dynamic_calibration_A(self, session_data: Dict) -> pd.DataFrame:
        """
        Evaluate with dynamic calibration A.

        For each test point, treat it as a new calibration point:
        1. Predict with current mapper
        2. Add the test point's gaze_result and ground truth to calibration data
        3. Retrain mapper
        4. Continue to next point

        This simulates perfect dynamic calibration where we know the ground truth
        for each click and use it to improve the mapper.

        Args:
            session_data: Session data dictionary

        Returns:
            DataFrame with evaluation results
        """
        # TODO: Implement simulated dynamic calibration
        raise NotImplementedError("Not yet implemented")

    def evaluate_with_dynamic_calibration_B(self, session_data: Dict) -> pd.DataFrame:
        """
        Evaluate with dynamic calibration using implicit clicks.

        Uses implicit clicks (natural user interactions) to continuously
        update the mapper during evaluation. This is the realistic dynamic
        calibration scenario where we don't have ground truth labels.

        Args:
            session_data: Session data dictionary

        Returns:
            DataFrame with evaluation results
        """
        # TODO: Implement implicit dynamic calibration
        raise NotImplementedError("Not yet implemented")

    def evaluate_with_dynamic_calibration_C(self, session_data: Dict) -> pd.DataFrame:
        """
        Just like B, but with explicit clicks as well.

        Args:
            session_data: Session data dictionary

        Returns:
            DataFrame with evaluation results
        """
        # TODO: Implement implicit dynamic calibration
        raise NotImplementedError("Not yet implemented")

    def evaluate_with_dynamic_calibration_D(self, session_data: Dict) -> pd.DataFrame:
        """
        Just like C, but we have a FIFO queue of size N for new data.
        (With original 9 points as fixed anchor)

        Args:
            session_data: Session data dictionary

        Returns:
            DataFrame with evaluation results
        """
        # TODO: Implement implicit dynamic calibration
        raise NotImplementedError("Not yet implemented")

    def print_evaluation_stats(self, results_df: pd.DataFrame):
        """Print evaluation statistics."""
        if results_df.empty:
            raise AssertionError("Expected non-empty results_df at this point.")

        mean_error = results_df["error"].mean()
        median_error = results_df["error"].median()
        std_error = results_df["error"].std()
        percentile_95 = results_df["error"].quantile(0.95)
        min_error = results_df["error"].min()
        max_error = results_df["error"].max()

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Number of test points:    {len(results_df)}")
        print(f"Mean Pixel Error:         {mean_error:.2f} pixels")
        print(f"Median Pixel Error:       {median_error:.2f} pixels")
        print(f"Std Dev of Error:         {std_error:.2f} pixels")
        print(f"Min Error:                {min_error:.2f} pixels")
        print(f"Max Error:                {max_error:.2f} pixels")
        print(f"95th Percentile Error:    {percentile_95:.2f} pixels")
        print("=" * 50 + "\n")

    def process_session(self, output_dir: Path):
        """
        Process a complete session.

        Args:
            output_dir: Directory to save results
        """
        print(f"\n{'=' * 70}")
        print(f"Processing session: {self.json_path.name}")
        print(f"{'=' * 70}\n")

        # Load session data
        session_data = self.load_session_data(self.json_path)
        session_id = session_data["session_id"]

        # Find video file
        video_filename = session_data["metadata"]["video_files"]["webcam"]
        video_path = self.data_dir / video_filename

        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return

        print(f"Session ID: {session_id}")
        print(f"Participant: {session_data['participant_name']}")
        print(f"Video file: {video_filename}")
        print(
            f"Screen size: {session_data['metadata']['screen_size']['width']}x{session_data['metadata']['screen_size']['height']}"
        )
        print(
            f"Calibration points: {len(session_data['calibration']['calibration_points'])}"
        )
        print(
            f"Explicit clicks: {len(session_data['click_events']['explicit_points'])}"
        )
        print(
            f"Implicit clicks: {len(session_data['click_events']['implicit_clicks'])}"
        )

        # Process calibration data
        try:
            num_calib_points = self.train_initial_mapper(session_data)
        except Exception as e:
            print(f"Error during calibration: {e}")
            return

        # Evaluate without dynamic calibration (baseline)
        try:
            results_df = self.evaluate_without_dynamic_calibration(session_data)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return

        # Print statistics
        self.print_evaluation_stats(results_df)

        # Create session-specific output directory
        session_output_dir = output_dir / session_id
        session_output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results_path = session_output_dir / "results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

        # Save summary
        summary = {
            "session_id": session_id,
            "participant_name": session_data["participant_name"],
            "num_calibration_points": num_calib_points,
            "num_test_points": len(results_df),
            "mean_error": results_df["error"].mean() if not results_df.empty else None,
            "median_error": results_df["error"].median()
            if not results_df.empty
            else None,
            "std_error": results_df["error"].std() if not results_df.empty else None,
            "percentile_95": results_df["error"].quantile(0.95)
            if not results_df.empty
            else None,
        }

        summary_path = session_output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process collected experiment data for gaze estimation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing collected data (e.g., collected_data/session_1)",
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
        default="experiment_results",
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

    processor = DataCollectorProcessor(
        data_dir=args.data_dir,
        weights_path=args.weights,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    processor.process_session(output_dir)

    print("\n" + "=" * 70)
    print("All sessions processed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
