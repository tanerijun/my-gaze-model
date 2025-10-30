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
VIDEO_FPS = 30  # The target FPS for the output video file
BENCHMARK_FRAMES = 100  # Number of frames to process for FPS calculation

# R2 (Cloudflare) Upload Configuration
R2_ENDPOINT_URL = "https://6e4e2c5c3f487a52988adf2a46fe200f.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "gaze-estimation-research"
# Credentials should be set via environment variables:
# R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY

CAMERA_ID = 0  # Default webcam
DESIRED_CAMERA_RESOLUTION = (1280, 720)  # Width, Height

# Head Pose Drift Detection Thresholds
DRIFT_THRESHOLDS = {
    "roll_degrees": 20.0,  # Maximum roll deviation in degrees
    "eye_distance_ratio": 0.25,  # Maximum IPD change ratio (e.g., 0.25 = 25%)
    "eye_center_shift_pixels": 50.0,  # Maximum eye center position shift in pixels
}

# Continuous Collection Settings
EXPLICIT_POINT_INTERVAL_SECONDS = 20  # Time between random explicit points
ENABLE_IMPLICIT_CLICKS = True  # Enable tracking of natural mouse clicks
