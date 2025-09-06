import os

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..models import GazeModel
from ..utils import GazeKalmanTracker, KalmanBoxTracker, batch_preprocess_faces


class GazePipeline3D:
    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        image_size: int = 224,
        use_landmarker: bool = False,
        smooth_gaze: bool = False,
    ):
        """
        Initialize the gaze pipeline.

        Args:
            weights_path: Path to the trained model weights (.pth file)
            device: Device to run inference on ("cpu", "cuda", or "auto")
            image_size: Input image size for the model (default 224 to match training)
            smooth_gaze: Enable Kalman filtering for gaze vectors (default False)
        """
        self.image_size = image_size
        self.device = self._setup_device(device)
        self.use_landmarker = use_landmarker

        self.model = GazeModel()
        self._load_model_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

        # JIT compile for faster inference
        self._compile_model()

        if self.use_landmarker:
            self._setup_face_landmarker()
            print("Pipeline configured to use FaceLandmarker.")
        else:
            self._setup_face_detector()
            print("Pipeline configured to use FaceDetector.")

        self.bbox_tracker = KalmanBoxTracker()  # for bbox smoothing

        self.gaze_tracker = GazeKalmanTracker(enabled=smooth_gaze)
        if smooth_gaze:
            print("Gaze smoothing enabled - set smooth_gaze=False to disable")

        print("Gaze estimation pipeline initialized successfully")

    def _decode_predictions(self, predictions):
        """
        Converts binned model output to continuous angular predictions.
        Matches the original training code.
        """
        pitch_pred, yaw_pred = predictions
        num_bins = 90  # Gaze360 configuration
        idx_tensor = torch.arange(num_bins, dtype=torch.float32, device=self.device)

        pitch_probs = F.softmax(pitch_pred, dim=1)
        yaw_probs = F.softmax(yaw_pred, dim=1)

        # Calculate expected value (continuous angle)
        # Gaze360 formula: bin_index * 4 - 180
        pitch = torch.sum(pitch_probs * idx_tensor, 1) * 4 - 180
        yaw = torch.sum(yaw_probs * idx_tensor, 1) * 4 - 180

        return torch.stack([pitch, yaw], dim=1)

    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_device = torch.device(device)
        print(f"Using device: {torch_device}")
        return torch_device

    def _load_model_weights(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded model weights from: {weights_path}")

    def _compile_model(self):
        """TorchScript JIT compilation for faster inference."""
        try:
            self.model = torch.jit.script(self.model)
            print("Model compiled with TorchScript for faster inference")
        except Exception as e:
            print(f"TorchScript compilation failed: {e}")
            print("Continuing with standard PyTorch model")

    def _setup_face_detector(self):
        """Initialize MediaPipe face detector."""
        model_path = os.path.join("mediapipe_models", "blaze_face_short_range.tflite")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face detector model not found at {model_path}. "
                f"Please download blaze_face_short_range.tflite and place it in the mediapipe_models folder."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        print("Face detector initialized successfully")

    def _setup_face_landmarker(self):
        """Initialize MediaPipe FaceLandmarker."""
        model_path = os.path.join("mediapipe_models", "face_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                f"Please download face_landmarker.task and place it in the mediapipe_models folder."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_facial_transformation_matrixes=True,  # for head pose
            num_faces=1,  # optimize for single-person use
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("Face landmarker initialized successfully")

    @torch.no_grad()
    def __call__(self, frame):
        """
        Process a frame to detect faces and estimate gaze directions.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            list: List of detection results, each containing:
                - "bbox": Bounding box [x1, y1, x2, y2]
                - "gaze": Dict with "pitch" and "yaw" angles in degrees
        """
        h, w, _ = frame.shape

        if self.use_landmarker:
            face_crops, bboxes, head_poses, all_landmarks = (
                self._process_frame_with_landmarker(frame)
            )
        else:
            face_crops, bboxes = self._process_frame_with_detector(frame)
            head_poses = [None] * len(bboxes)  # placeholder
            all_landmarks = [None] * len(bboxes)  # placeholder

        if not face_crops:
            return []

        # Batch process faces through gaze model
        face_batch = batch_preprocess_faces(face_crops, self.image_size)
        if face_batch.size(0) == 0:
            return []

        face_batch = face_batch.to(self.device)
        predictions = self.model(face_batch)
        decoded_preds = self._decode_predictions(predictions).cpu().numpy()

        # Package results
        results = []
        for i, bbox in enumerate(bboxes):
            # Apply gaze smoothing
            pitch = float(decoded_preds[i][0])
            yaw = float(decoded_preds[i][1])
            smoothed_pitch, smoothed_yaw = self.gaze_tracker.update(pitch, yaw)

            results.append(
                {
                    "bbox": bbox,
                    "gaze": {
                        "pitch": smoothed_pitch,
                        "yaw": smoothed_yaw,
                    },
                    "head_pose_matrix": head_poses[i],  # None if not using landmarker
                    "landmarks": all_landmarks[i],
                }
            )

        return results

    def _detect_faces_detector(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.face_detector.detect(mp_image)
        return detection_result.detections

    def _extract_face_crops_detector(self, frame, detections):
        h, w, _ = frame.shape
        face_crops, bboxes = [], []

        for detection in detections:
            bbox = detection.bounding_box

            # Ensure bbox is within frame boundaries
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)

            # Apply Kalman filter for bbox smoothing
            sx1, sy1, sx2, sy2 = self.bbox_tracker.update([x1, y1, x2, y2])

            # Ensure smoothed bbox is still within bounds
            sx1 = max(0, min(w - 1, sx1))
            sy1 = max(0, min(h - 1, sy1))
            sx2 = max(sx1 + 1, min(w, sx2))
            sy2 = max(sy1 + 1, min(h, sy2))

            # Extract face crop
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                face_crops.append(crop)
                bboxes.append((sx1, sy1, sx2, sy2))

        return face_crops, bboxes

    def _process_frame_with_detector(self, frame):
        detections = self._detect_faces_detector(frame)
        return self._extract_face_crops_detector(frame, detections)

    def _process_frame_with_landmarker(self, frame):
        h, w, _ = frame.shape
        face_crops, bboxes, head_poses, all_landmarks = [], [], [], []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return [], [], [], []

        landmarks = detection_result.face_landmarks[0]
        transform_matrix = detection_result.facial_transformation_matrixes[0]

        bbox = self._get_bbox_from_landmarks(landmarks, (h, w))
        sx1, sy1, sx2, sy2 = self.bbox_tracker.update(bbox)

        crop = frame[sy1:sy2, sx1:sx2]
        if crop.size > 0:
            face_crops.append(crop)
            bboxes.append((sx1, sy1, sx2, sy2))
            head_poses.append(transform_matrix)
            all_landmarks.append(landmarks)

        return face_crops, bboxes, head_poses, all_landmarks

    def _get_bbox_from_landmarks(self, landmarks, frame_shape, margin=0.1):
        """Derive bounding box from landmarks"""
        h, w = frame_shape
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        box_w, box_h = x_max - x_min, y_max - y_min
        margin_w, margin_h = box_w * margin, box_h * margin
        x1 = int(max(0, x_min - margin_w))
        y1 = int(max(0, y_min - margin_h))
        x2 = int(min(w - 1, x_max + margin_w))
        y2 = int(min(h - 1, y_max + margin_h))
        return x1, y1, x2, y2

    def reset_tracking(self):
        """Reset both bbox and gaze tracking states."""
        self.bbox_tracker.reset()
        self.gaze_tracker.reset()
