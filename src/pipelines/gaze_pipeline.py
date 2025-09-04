import os

import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms

from src.models import build_model


class KalmanBoxTracker:
    def __init__(self):
        # State: [x1, y1, x2, y2]
        self.state = None
        self.P = np.eye(4) * 100  # initial uncertainty
        self.F = np.eye(4)
        self.H = np.eye(4)

        # Base parameters - tuned for stability
        self.base_Q = np.eye(4) * 0.05  # process noise
        self.base_R = np.eye(4) * 25.0  # measurement noise

        # Current parameters (will be adjusted)
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()

        self.movement_threshold = 50  # pixels

    def update(self, bbox):
        z = np.array(bbox)

        if self.state is None:
            self.state = z
            return bbox

        # Calculate movement
        movement = np.linalg.norm(z - self.state)

        # Adaptive parameters
        if movement > self.movement_threshold:
            # High movement: trust measurements more
            self.Q = self.base_Q * 20.0
            self.R = self.base_R * 0.1
        else:
            # Low movement: prioritize stability
            self.Q = self.base_Q * 0.2
            self.R = self.base_R * 2.0

        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return [int(x) for x in self.state]


class GazePipeline:
    def __init__(self, config, device, weights_path):
        self.config = config
        self.device = device
        is_fused = "fused" in config and config["fused"]
        model_kwargs = {"inference_mode": is_fused} if is_fused else {}
        self.model = build_model(config, **model_kwargs).to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()
        print("Gaze estimation model loaded successfully.")

        # Initialize Kalman Filter for bounding box smoothing
        self.bbox_tracker = KalmanBoxTracker()

        # --- JIT compilation for faster CPU inference ---
        try:
            self.model = torch.jit.script(self.model)
            print("Model successfully compiled with TorchScript (JIT).")
        except Exception as e:
            print(f"TorchScript JIT compilation failed: {e}")
            print("Proceeding with standard PyTorch model.")

        # --- Initialize Face Detector (New MediaPipe Tasks API) ---
        model_path = os.path.join("mediapipe_models", "blaze_face_short_range.tflite")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face detector model not found at {model_path}. Please download it."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        print("Face detector model loaded successfully.")

        # --- Define Preprocessing ---
        image_size = config.get("image_size", 224)
        print(f"Initializing GazePipeline with image size: {image_size}x{image_size}")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _decode_predictions(self, predictions):
        """
        Converts binned model output to continuous angular predictions.
        """
        pitch_pred, yaw_pred = predictions
        num_bins = self.config["num_bins"]
        idx_tensor = torch.arange(num_bins, dtype=torch.float32).to(self.device)

        pitch_probs = F.softmax(pitch_pred, dim=1)
        yaw_probs = F.softmax(yaw_pred, dim=1)

        # Calculate expected value (continuous angle)
        pitch = torch.sum(pitch_probs * idx_tensor, 1) * 4 - 180
        yaw = torch.sum(yaw_probs * idx_tensor, 1) * 4 - 180

        return torch.stack([pitch, yaw], dim=1)

    @torch.no_grad()
    def __call__(self, frame):
        """
        Processes a single frame to detect faces and estimate gaze.
        """
        h, w, _ = frame.shape

        # --- Face Detection ---
        rgb_frame = frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_detector.detect(mp_image)

        if not detection_result.detections:
            return []

        face_crops = []
        bboxes = []

        for detection in detection_result.detections:
            bbox = detection.bounding_box

            # Ensure bbox is within frame boundaries
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)

            # Apply Kalman Filter bounding box smoothing
            smoothed_bbox = self.bbox_tracker.update([x1, y1, x2, y2])
            sx1, sy1, sx2, sy2 = smoothed_bbox

            # Ensure smoothed bbox is still within bounds
            sx1 = max(0, min(w - 1, sx1))
            sy1 = max(0, min(h - 1, sy1))
            sx2 = max(sx1 + 1, min(w, sx2))
            sy2 = max(sy1 + 1, min(h, sy2))

            # Crop face and preprocess
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue

            face_crops.append(self.transform(crop))
            bboxes.append((sx1, sy1, sx2, sy2))

        if not face_crops:
            return []

        # --- Gaze Estimation on Batch ---
        face_batch = torch.stack(face_crops).to(self.device)
        predictions = self.model(face_batch)
        decoded_preds = self._decode_predictions(predictions).cpu().numpy()

        # --- Package Results ---
        output = []
        for i in range(len(bboxes)):
            output.append(
                {
                    "bbox": bboxes[i],
                    "gaze": {
                        "pitch": decoded_preds[i][0],
                        "yaw": decoded_preds[i][1],
                    },
                }
            )

        return output
