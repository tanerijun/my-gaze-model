import math
import os

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..models import GazeModel
from ..utils import FaceKalmanTracker, GazeKalmanTracker, batch_preprocess_faces


class SmoothedKeypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class GazePipeline3D:
    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        image_size: int = 224,
        enable_landmarker_features: bool = False,
        smooth_facebbox: bool = False,
        smooth_gaze: bool = False,
    ):
        """
        Initialize the gaze pipeline.

        Args:
            weights_path: Path to the trained model weights (.pth file)
            device: Device to run inference on ("cpu", "cuda", or "auto")
            image_size: Input image size for the model (default 224 to match training)
            enable_landmarker_features: If True, runs FaceLandmarker to get head pose and landmarks (default false for speed) (assume only 1 face)
            smooth_facebbox: Enable Kalman filtering for face bounding box and keypoints (default false)
            smooth_gaze: Enable Kalman filtering for gaze vectors (default False)
        """
        self.image_size = image_size
        self.device = self._setup_device(device)
        self.enable_landmarker_features = enable_landmarker_features

        self._setup_face_detector()

        if self.enable_landmarker_features:
            self._setup_face_landmarker()
            print("Landmarker features ENABLED (for head pose and landmarks).")
        else:
            print("Landmarker features DISABLED (default, max performance).")
            self.face_landmarker = None

        self.model = GazeModel()
        self._load_model_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

        # JIT compile for faster inference
        self._compile_model()

        self.face_tracker = FaceKalmanTracker(enabled=smooth_facebbox)
        if smooth_facebbox:
            print("Face bounding box smoothing ENABLED.")
        else:
            print("Face bounding box smoothing DISABLED.")

        self.gaze_tracker = GazeKalmanTracker(enabled=smooth_gaze)
        if smooth_gaze:
            print("Gaze smoothing ENABLED.")
        else:
            print("Gaze smoothing DISABLED.")

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
            num_faces=1,  # optimize for single-person use; Smoothing is only applied when n_face == 1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("Face landmarker initialized successfully")

    @torch.no_grad()
    def __call__(self, frame):
        """
        Process a frame to detect faces and estimate gaze directions.

        Args:
            frame: The input video frame as a NumPy array (H, W, 3) in BGR format.

        Returns:
            A list of dictionaries, where each dictionary contains the results for one
            detected face. The list is empty if no faces are found. The dictionary
            has the following structure:
            ```
            {
                "bbox": Tuple[int, int, int, int],
                "gaze": Dict[str, float],
                "head_pose_matrix": Optional[np.ndarray],
                "landmarks": Optional[List[mp.tasks.python.vision.NormalizedLandmark]]
            }
            ```
            - `bbox`: The smoothed bounding box (x1, y1, x2, y2) of the face.
            - `gaze`: A dictionary with "pitch" and "yaw" keys in degrees.
            - `gaze_origin_features`: Dict[str, float],
            - `head_pose_matrix`: A 4x4 NumPy array representing the 3D head
                transformation matrix. Is `None` if `enable_landmarker_features` is False.
            - `landmarks`: A list of MediaPipe NormalizedLandmark objects. Is `None`
                if `enable_landmarker_features` is False.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_detector.detect(mp_image)
        detections = detection_result.detections
        if not detections:
            return []

        face_crops, bboxes, keypoints_list = self._extract_face_crops_detector(
            frame, detections
        )
        if not face_crops:
            return []

        head_pose, mediapipe_landmarks = None, None
        if self.enable_landmarker_features and self.face_landmarker:
            landmarker_result = self.face_landmarker.detect(mp_image)
            if landmarker_result.facial_transformation_matrixes:
                head_pose = landmarker_result.facial_transformation_matrixes[0]
                # Ensure we have landmarks before trying to access them
                if landmarker_result.face_landmarks:
                    mediapipe_landmarks = landmarker_result.face_landmarks[0]

        face_batch = batch_preprocess_faces(face_crops, self.image_size)
        if face_batch.size(0) == 0:
            return []

        face_batch = face_batch.to(self.device)
        predictions = self.model(face_batch)
        decoded_preds = self._decode_predictions(predictions).cpu().numpy()

        # Package results
        results = []
        for i, (bbox, keypoints) in enumerate(zip(bboxes, keypoints_list)):
            # Apply gaze smoothing
            pitch = float(decoded_preds[i][0])
            yaw = float(decoded_preds[i][1])
            smoothed_pitch, smoothed_yaw = self.gaze_tracker.update(pitch, yaw)
            origin_features = self._calculate_gaze_origin_features(keypoints, w, h)

            results.append(
                {
                    "bbox": bbox,
                    "gaze": {
                        "pitch": smoothed_pitch,
                        "yaw": smoothed_yaw,
                    },
                    "gaze_origin_features": origin_features,
                    "blaze_keypoints": keypoints,
                    "head_pose_matrix": head_pose,
                    "mediapipe_landmarks": mediapipe_landmarks,
                }
            )

        return results

    def _extract_face_crops_detector(self, frame, detections):
        h, w, _ = frame.shape
        face_crops, final_bboxes, final_keypoints = [], [], []

        for detection in detections:
            # Extract BBox
            bbox = detection.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)
            bbox_coords = [x1, y1, x2, y2]

            # Extract and flatten keypoints
            keypoints = detection.keypoints
            kp_coords = [coord for kp in keypoints for coord in (kp.x, kp.y)]

            # Ensure we have 6 keypoints (12 coords) before proceeding
            if len(kp_coords) != 12:
                continue

            # Form the unified measurement vector and update tracker
            measurement = bbox_coords + kp_coords
            smoothed_state = self.face_tracker.update(measurement)

            # Unpack the smoothed state
            smoothed_bbox = smoothed_state[:4]
            smoothed_kp_coords = smoothed_state[4:]

            sx1, sy1, sx2, sy2 = map(int, smoothed_bbox)

            # Ensure smoothed bbox is still within bounds
            sx1 = max(0, min(w - 1, sx1))
            sy1 = max(0, min(h - 1, sy1))
            sx2 = max(sx1 + 1, min(w, sx2))
            sy2 = max(sy1 + 1, min(h, sy2))

            # Reconstruct keypoints into usable format
            reconstructed_kps = []
            for i in range(0, len(smoothed_kp_coords), 2):
                reconstructed_kps.append(
                    SmoothedKeypoint(
                        x=smoothed_kp_coords[i], y=smoothed_kp_coords[i + 1]
                    )
                )

            # Extract face crop
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                face_crops.append(crop)
                final_bboxes.append((sx1, sy1, sx2, sy2))
                final_keypoints.append(reconstructed_kps)

        return face_crops, final_bboxes, final_keypoints

    def _calculate_gaze_origin_features(self, keypoints, frame_w, frame_h):
        """
        Calculates gaze origin features from BlazeFace keypoints.

        Keypoint indices for BlazeFace:
        0: Right Eye
        1: Left Eye
        """
        if len(keypoints) < 2:
            return {
                "eye_center_x": 0.0,
                "eye_center_y": 0.0,
                "ipd": 0.0,
                "roll_angle": 0.0,
            }

        # De-normalize keypoints
        right_eye = (keypoints[0].x * frame_w, keypoints[0].y * frame_h)
        left_eye = (keypoints[1].x * frame_w, keypoints[1].y * frame_h)

        # Eye Center (normalized by frame dimensions for consistency)
        eye_center_x = ((left_eye[0] + right_eye[0]) / 2.0) / frame_w
        eye_center_y = ((left_eye[1] + right_eye[1]) / 2.0) / frame_h

        # Inter-pupillary distance (IPD) as a proxy for Z-depth
        ipd = math.sqrt(
            (left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2
        )
        # Normalize by frame width
        normalized_ipd = ipd / frame_w

        # Head Roll Angle
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        roll_angle = math.degrees(math.atan2(dy, dx))

        return {
            "eye_center_x": eye_center_x,
            "eye_center_y": eye_center_y,
            "ipd": normalized_ipd,
            "roll_angle": roll_angle,
        }

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
        self.face_tracker.reset()
        self.gaze_tracker.reset()
