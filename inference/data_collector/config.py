from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ICON_PATH = PROJECT_ROOT / "data_collector" / "ui" / "assets" / "icon.png"

GAZE_MODEL_PATH = PROJECT_ROOT / "weights" / "prod.pth"

FACE_DETECTOR_PATH = PROJECT_ROOT / "mediapipe_models" / "blaze_face_short_range.tflite"

PIPELINE_CONFIG = {
    "enable_landmarker_features": False,  # Use BlazeFace keypoints for head pose
    "smooth_facebbox": True,  # Enable Kalman filter for face bbox
    "smooth_gaze": False,  # Disable Kalman filter for gaze vector
}

DATA_OUTPUT_DIR = PROJECT_ROOT / "collected_data"

CAMERA_ID = 0  # Default webcam
CAMERA_RESOLUTION = (1280, 720)  # Width, Height
