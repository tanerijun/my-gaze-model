# Gaze Estimation Inference

A streamlined, production-ready gaze estimation system using MobileOne S1 backbone for real-time inference applications.

## Features

- **Fast Inference**: Fused MobileOne S1 backbone with TorchScript compilation
- **Real-time Processing**: Optimized pipeline with threaded frame capture
- **Smooth Tracking**: Kalman filter-based bounding box smoothing (gaze vectors are raw predictions)
- **Easy Integration**: Simple API for embedding in applications
- **Minimal Dependencies**: Streamlined package requirements
- **Gaze360 Configuration**: 90 bins covering ±180° range

## Quick Start

### Installation

1. Install dependencies:

```bash
uv sync
```

2. Required files:
   - MediaPipe face detection model (already included)
   - Your trained fused MobileOne S1 model weights (.pth file)

### Basic Usage

```bash
# Run demo with webcam
python demo.py --weights path/to/your/model.pth

# Use specific webcam or video file
python demo.py --weights model.pth --source 1
python demo.py --weights model.pth --source video.mp4

# Force CPU inference
python demo.py --weights model.pth --device cpu
```

### Programmatic Usage

```python
from src import GazePipeline
import cv2

# Initialize pipeline
pipeline = GazePipeline("model.pth", device="auto")

# Process frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

results = pipeline(frame)
for result in results:
    bbox = result["bbox"]  # [x1, y1, x2, y2]
    gaze = result["gaze"]  # {"pitch": float, "yaw": float} in degrees

    print(f"Gaze: pitch={gaze['pitch']:.1f}°, yaw={gaze['yaw']:.1f}°")
```

## Model Requirements

- **Fused MobileOne S1** model from the training repository
- **Gaze360 configuration**: 90 bins, ±180° range
- **Format**: PyTorch state dict (.pth file)

## Architecture

```
src/
├── models/
│   ├── mobileone.py      # MobileOne S1 backbone (fused-only)
│   ├── gaze_head.py      # 90-bin classification head
│   └── gaze_model.py     # Complete model
├── inference/
│   └── gaze_pipeline.py  # Main inference pipeline
└── utils/
    ├── transforms.py     # Image preprocessing
    └── kalman_tracker.py # Bounding box smoothing
```

## Performance

- **CPU**: ~15-20 FPS
- **GPU**: ~60+ FPS
- **Model Size**: ~8MB
- **Memory**: ~200MB

## API Reference

### GazePipeline

```python
class GazePipeline:
    def __init__(self, weights_path: str, device: str = "auto", image_size: int = 224)
    def __call__(self, frame: np.ndarray) -> List[Dict]
    def reset_tracking(self) -> None
```

**Returns**: List of detections with `bbox` and `gaze` (pitch/yaw in degrees)

## Demo Controls

- **Q**: Quit
- **R**: Reset tracking

## Troubleshooting

1. **Face detector model not found**: Ensure `blaze_face_short_range.tflite` is in `mediapipe_models/`
2. **Model weights not found**: Verify the .pth file path
3. **Poor CPU performance**: Use GPU with `--device cuda`
4. **Unstable tracking**: Reset with 'R' key

## Integration Example

```python
# Simple real-time processing
import cv2
from src import GazePipeline

pipeline = GazePipeline("model.pth")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pipeline(frame)
    # Process results...
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy 1.24+
