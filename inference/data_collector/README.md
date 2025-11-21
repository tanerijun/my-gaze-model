# Gaze Estimation Data Collection Tool

## Project Overview

A **native desktop data collection application** for gaze estimation research. It captures synchronized video and click data for gaze estimation models.

**Tech Stack:**

- **Python 3.12.9** (strict requirement)
- **PyQt6** for GUI and system tray
- **OpenCV** for video capture and encoding
- **PyTorch** for gaze inference
- **MediaPipe BlazeFace** for face detection
- **pynput** for system-wide mouse tracking
- **uv** for dependency management

**Features:**

- System benchmarking and profiling
- 3x3 grid calibration with baseline head pose calculation
- Continuous data collection with explicit and implicit clicks
- Head pose drift detection and warnings
- Multi-threaded architecture for smooth capture
- Synchronized video and JSON metadata storage

## Project Structure

```
inference/
├── data_collector/              # Main application package
│   ├── main.py                  # Entry point
│   ├── app.py                   # SystemTrayApp (Qt app wrapper)
│   ├── config.py                # All configuration constants
│   ├── NOTE.md                  # This file
│   ├── core/
│   │   ├── app_controller.py   # Main orchestrator (state machine)
│   │   ├── data_manager.py     # Session data storage and JSON export
│   │   └── workers.py          # CameraWorker, InferenceWorker, StorageWorker
│   ├── ui/
│   │   ├── calibration_overlay.py  # Frameless full-screen overlay
│   │   ├── system_tray.py          # System tray menu
│   │   └── assets/
│   │       └── icon.png
│   └── utils/
│       └── system_info.py      # Hardware profiling functions
├── collected_data/              # Output directory for sessions
│   └── session_YYYY-MM-DD_HH-MM-SS/
│       ├── session_*.mp4       # Recorded video
│       └── session_*.json      # All metadata and click events
├── src/                         # Gaze inference pipeline
│   ├── inference/
│   │   ├── gaze_pipeline_3d.py
│   │   └── mapper.py
│   ├── models/
│   └── utils/
├── weights/
│   └── prod.pth                # Trained model weights
├── mediapipe_models/
│   └── blaze_face_short_range.tflite
└── pyproject.toml              # uv dependency configuration
```

## Application Flow

### Phase 1: Benchmarking (Automatic)

1. User clicks tray icon → "Start Session"
2. App captures system information (CPU, RAM, OS, Python/PyTorch versions)
3. Captures single webcam frame
4. Runs model inference 100 times to measure FPS
5. Saves all metadata to session

### Phase 2: Calibration (9-point grid)

1. User clicks → "Start Calibration"
2. Full-screen overlay appears with first calibration point
3. User clicks on each point (9 total)
4. After each click, press SPACE to continue
5. App calculates baseline head pose from all 9 points
6. Transitions to data collection

### Phase 3: Continuous Collection

1. Overlay closes, app runs in background
2. **Explicit Points:** Random point appears every X seconds (configurable)
3. **Implicit Clicks:** All mouse clicks are silently recorded
4. **Drift Detection:** Real-time head pose monitoring
   - If drift exceeds thresholds → overlay warning appears
   - Data collection pauses until user returns to position
5. User clicks tray icon → "Stop Session" when done
6. Video and JSON saved to session

## Data Format

### Session JSON Structure

```json
{
  "session_id": "2025-10-30_14-30-00",
  "start_time": "2025-10-30T14:30:00.123456",
  "end_time": "2025-10-30T14:35:30.789012",
  "metadata": {
    "system_info": {
      "os": "Darwin 25.0.0 (arm64)",
      "python_version": "3.12.9",
      "torch_version": "2.8.0",
      "cpu": { "brand": "Apple M3", "cores": 8, ... },
      "ram": { "total_gb": 16.0, ... }
    },
    "screen_size": { "width": 1512, "height": 982 },
    "camera_resolution": { "width": 1280, "height": 720 },
    "performance": { "inference_fps": 49.81 }
  },
  "video_file": "session_2025-10-30_14-30-00.mp4",
  "calibration": {
    "baseline_head_pose": {
      "roll": 2.5,
      "eye_distance": 145.6,
      "eye_center_x": 640.0,
      "eye_center_y": 360.0,
      "num_samples": 9
    },
    "calibration_points": [
      {
        "target": { "x": 60, "y": 78 },
        "click": { "x": 62, "y": 80 },
        "gaze_result": {
          "gaze": { "pitch": 12.5, "yaw": -3.2 },
          "eye_distance": 145.3,
          "eye_center": [640.2, 360.1],
          "face_bbox": [100, 50, 300, 250]
        },
        "timestamp": "2025-10-30T14:30:15.123456",
        "video_timestamp": 5.234
      }
      // ... 8 more points
    ]
  },
  "click_events": {
    "explicit_points": [
      {
        "target": { "x": 800, "y": 400 },
        "click": { "x": 803, "y": 402 },
        "gaze_result": { /* same structure */ },
        "timestamp": "2025-10-30T14:30:45.123456",
        "video_timestamp": 35.234
      }
      // ... more explicit points
    ],
    "implicit_clicks": [
      {
        "click": { "x": 500, "y": 300 },
        "gaze_result": { /* same structure */ },
        "timestamp": "2025-10-30T14:30:47.456789",
        "video_timestamp": 37.567
      }
      // ... many implicit clicks
    ]
  },
  "statistics": {
    "total_clicks": 42,
    "calibration_points": 9,
    "explicit_points": 6,
    "implicit_clicks": 27
  }
}
```

### Gaze Result Fields (Filtered)

**Saved to JSON:**

- `gaze.pitch` - Vertical gaze angle (degrees)
- `gaze.yaw` - Horizontal gaze angle (degrees)
- `eye_distance` - Inter-pupillary distance in pixels
- `eye_center` - [x, y] center point between eyes
- `face_bbox` - [x, y, width, height] face bounding box

### Timestamp Fields

1. **`timestamp`** (ISO 8601 string): Absolute wall-clock time
   - Example: `"2025-10-30T14:30:45.123456"`
   - Purpose: Know when the session happened in real-world time

2. **`video_timestamp`** (float, seconds): Time offset from video start
   - Example: `35.234`
   - Purpose: Seek to exact frame in video where click occurred
   - Critical for synchronizing visual data with click locations

### Installation & Running

```bash
# Navigate to project directory

# Install/sync dependencies (uv reads pyproject.toml)
uv sync

# Run the application
uv run data_collector/main.py

# The app will appear in your system tray/menu bar
```

### Adding Dependencies

```bash
# Add a new package
uv add package-name

# Add with version constraint
uv add "package-name>=1.2.3"

# Remove a package
uv remove package-name

# Update all packages
uv sync --upgrade
```
