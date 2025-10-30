"""
Data Manager for Session Data Collection

This module handles all session data management, including:
- Session initialization with metadata
- Recording click events (calibration, explicit, and implicit)
- Saving data to JSON files
- Exporting sessions as ZIP archives
"""

import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, Optional


class DataManager:
    """
    Manages session data collection and storage.

    The DataManager stores all session metadata and click events in memory,
    then writes everything to a JSON file when the session ends.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the DataManager.

        Args:
            output_dir: Root directory for storing collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_data: Dict = {}
        self.session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.video_filename: Optional[str] = None

    def start_session(
        self, metadata: Dict, participant_name: Optional[str] = None
    ) -> str:
        """
        Initialize a new session with metadata.

        Args:
            metadata: Dictionary containing system_info, screen_size, camera_resolution,
                     performance benchmarks, etc.
            participant_name: Optional participant name/ID to prefix the session

        Returns:
            session_id: Unique identifier for this session (timestamp-based)
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Add participant name prefix if provided
        if participant_name:
            # Sanitize name for filename
            safe_name = "".join(
                c for c in participant_name if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            safe_name = safe_name.replace(" ", "-")
            self.session_id = f"{safe_name}_{timestamp}"
        else:
            self.session_id = timestamp

        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session data structure
        self.session_data = {
            "session_id": self.session_id,
            "participant_name": participant_name or "Anonymous",
            "start_time": datetime.datetime.now().isoformat(),
            "metadata": metadata,
            "calibration": {"baseline_head_pose": None, "calibration_points": []},
            "click_events": {
                "explicit_points": [],  # Random points during collection
                "implicit_clicks": [],  # Natural user clicks
            },
        }

        print(f"DataManager: Session started with ID: {self.session_id}")
        return self.session_id

    def set_baseline_head_pose(self, baseline_pose: Dict):
        """
        Store the baseline head pose calculated from calibration data.

        Args:
            baseline_pose: Dictionary containing averaged head pose features
        """
        if not self.session_data:
            raise RuntimeError("Session not started. Call start_session() first.")

        self.session_data["calibration"]["baseline_head_pose"] = baseline_pose
        print("DataManager: Baseline head pose stored")

    def add_calibration_point(
        self,
        target_x: int,
        target_y: int,
        click_x: int,
        click_y: int,
        gaze_result: Optional[Dict],
        video_timestamp: Optional[float] = None,
    ):
        """
        Record a calibration point click.

        Args:
            target_x, target_y: Ground truth location of the calibration point
            click_x, click_y: Actual location where the user clicked
            gaze_result: Head pose and gaze data at the time of click
            video_timestamp: Time offset in the video (seconds)
        """
        if not self.session_data:
            raise RuntimeError("Session not started. Call start_session() first.")

        event = {
            "target": {"x": target_x, "y": target_y},
            "click": {"x": click_x, "y": click_y},
            "gaze_result": gaze_result,
            "timestamp": datetime.datetime.now().isoformat(),
            "video_timestamp": video_timestamp,
        }

        self.session_data["calibration"]["calibration_points"].append(event)
        num_points = len(self.session_data["calibration"]["calibration_points"])
        print(f"DataManager: Calibration point {num_points} recorded")

    def add_explicit_click(
        self,
        target_x: int,
        target_y: int,
        click_x: int,
        click_y: int,
        gaze_result: Optional[Dict],
        video_timestamp: Optional[float] = None,
    ):
        """
        Record an explicit click (random point during collection).

        Args:
            target_x, target_y: Ground truth location of the explicit point
            click_x, click_y: Actual location where the user clicked
            gaze_result: Head pose and gaze data at the time of click
            video_timestamp: Time offset in the video (seconds)
        """
        if not self.session_data:
            raise RuntimeError("Session not started. Call start_session() first.")

        event = {
            "target": {"x": target_x, "y": target_y},
            "click": {"x": click_x, "y": click_y},
            "gaze_result": gaze_result,
            "timestamp": datetime.datetime.now().isoformat(),
            "video_timestamp": video_timestamp,
        }

        self.session_data["click_events"]["explicit_points"].append(event)
        num_explicit = len(self.session_data["click_events"]["explicit_points"])
        print(f"DataManager: Explicit click {num_explicit} recorded")

    def add_implicit_click(
        self,
        click_x: int,
        click_y: int,
        gaze_result: Optional[Dict],
        video_timestamp: Optional[float] = None,
    ):
        """
        Record an implicit click (natural user click during normal usage).

        Args:
            click_x, click_y: Location where the user clicked
            gaze_result: Head pose and gaze data at the time of click
            video_timestamp: Time offset in the video (seconds)
        """
        if not self.session_data:
            raise RuntimeError("Session not started. Call start_session() first.")

        event = {
            "click": {"x": click_x, "y": click_y},
            "gaze_result": gaze_result,
            "timestamp": datetime.datetime.now().isoformat(),
            "video_timestamp": video_timestamp,
        }

        self.session_data["click_events"]["implicit_clicks"].append(event)
        # Don't print for every implicit click (can be noisy)

    def set_video_filename(self, filename: str):
        """
        Store the video filename for this session.

        Args:
            filename: Name of the video file (without path)
        """
        self.video_filename = filename
        if self.session_data:
            self.session_data["video_file"] = filename

    def save_to_disk(self) -> Optional[Path]:
        """
        Write the complete session data to a JSON file.

        Returns:
            Path to the saved JSON file, or None if session not started
        """
        if not self.session_data or not self.session_id or not self.session_dir:
            print("DataManager: No session data to save")
            return None

        # Add end time
        self.session_data["end_time"] = datetime.datetime.now().isoformat()

        # Calculate statistics
        num_calibration = len(self.session_data["calibration"]["calibration_points"])
        num_explicit = len(self.session_data["click_events"]["explicit_points"])
        num_implicit = len(self.session_data["click_events"]["implicit_clicks"])
        total_clicks = num_calibration + num_explicit + num_implicit

        self.session_data["statistics"] = {
            "total_clicks": total_clicks,
            "calibration_points": num_calibration,
            "explicit_points": num_explicit,
            "implicit_clicks": num_implicit,
        }

        # Write to JSON file
        json_path = self.session_dir / f"session_{self.session_id}.json"
        with open(json_path, "w") as f:
            json.dump(self.session_data, f, indent=2)

        print("\n=== Session Data Saved ===")
        print(f"JSON File: {json_path}")
        print(f"Total Clicks: {total_clicks}")
        print(f"  - Calibration: {num_calibration}")
        print(f"  - Explicit: {num_explicit}")
        print(f"  - Implicit: {num_implicit}")
        print("==========================\n")

        return json_path

    def export_session_as_zip(self, session_id: Optional[str] = None) -> Optional[Path]:
        """
        Export a session (video + JSON) as a ZIP file.

        Args:
            session_id: ID of the session to export. If None, exports current session.

        Returns:
            Path to the created ZIP file, or None if failed
        """
        if session_id is None:
            session_id = self.session_id

        if not session_id:
            print("DataManager: No session to export")
            return None

        session_dir = self.output_dir / f"session_{session_id}"
        if not session_dir.exists():
            print(f"DataManager: Session directory not found: {session_dir}")
            return None

        # Create ZIP file
        zip_path = self.output_dir / f"session_{session_id}.zip"
        shutil.make_archive(
            str(zip_path.with_suffix("")),  # Remove .zip, make_archive adds it
            "zip",
            session_dir,
        )

        print(f"DataManager: Session exported to {zip_path}")
        return zip_path

    def clear_all_data(self):
        """
        Delete all collected session data.

        WARNING: This is destructive and cannot be undone!
        """
        if not self.output_dir.exists():
            print("DataManager: No data directory to clear")
            return

        # Delete all session directories
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("session_"):
                shutil.rmtree(item)
                print(f"DataManager: Deleted {item.name}")
            elif item.is_file() and item.name.endswith(".zip"):
                item.unlink()
                print(f"DataManager: Deleted {item.name}")

        print("DataManager: All session data cleared")

    def get_session_statistics(self) -> Optional[Dict]:
        """
        Get statistics for the current session.

        Returns:
            Dictionary with session statistics, or None if no session active
        """
        if not self.session_data:
            return None

        return {
            "session_id": self.session_id,
            "calibration_points": len(
                self.session_data["calibration"]["calibration_points"]
            ),
            "explicit_clicks": len(
                self.session_data["click_events"]["explicit_points"]
            ),
            "implicit_clicks": len(
                self.session_data["click_events"]["implicit_clicks"]
            ),
            "total_clicks": (
                len(self.session_data["calibration"]["calibration_points"])
                + len(self.session_data["click_events"]["explicit_points"])
                + len(self.session_data["click_events"]["implicit_clicks"])
            ),
        }
