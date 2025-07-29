import os
import torch
import torch.nn.functional as F
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from src.models import build_model

class GazePipeline:
    def __init__(self, config, device, weights_path):
        self.config = config
        self.device = device

        is_fused = 'fused' in config and config['fused']
        model_kwargs = {'inference_mode': is_fused} if is_fused else {}
        self.model = build_model(config, **model_kwargs).to(device)

        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()
        print("Gaze estimation model loaded successfully.")

        # --- Initialize Face Detector (New MediaPipe Tasks API) ---
        model_path = os.path.join('mediapipe_models', 'blaze_face_short_range.tflite')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face detector model not found at {model_path}. Please download it.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE # Process one image at a time
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        print("Face detector model loaded successfully.")

        # --- Define Preprocessing ---
        image_size = config.get('image_size', 224)
        print(f"Initializing GazePipeline with image size: {image_size}x{image_size}")

        # This must be IDENTICAL to the transform used during training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _decode_predictions(self, predictions):
        """
        Converts binned model output to continuous angular predictions.
        This logic MUST match the regression loss calculation in GazeLoss.
        """
        pitch_pred, yaw_pred = predictions
        num_bins = self.config['num_bins']
        idx_tensor = torch.arange(num_bins, dtype=torch.float32).to(self.device)

        pitch_probs = F.softmax(pitch_pred, dim=1)
        yaw_probs = F.softmax(yaw_pred, dim=1)

        # Calculate expected value (continuous angle) using the L2CS-Net logic
        pitch = torch.sum(pitch_probs * idx_tensor, 1) * 4 - 180
        yaw = torch.sum(yaw_probs * idx_tensor, 1) * 4 - 180

        return torch.stack([pitch, yaw], dim=1)

    @torch.no_grad()
    def __call__(self, frame):
        """
        Processes a single frame to detect faces and estimate gaze.
        Args:
            frame (np.ndarray): The input video frame (BGR).
        Returns:
            list: A list of dictionaries, one for each detected face.
                  Each dict contains 'bbox' and 'gaze' (pitch, yaw in degrees).
        """
        h, w, _ = frame.shape

        # --- Face Detection (New MediaPipe Tasks API) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            # Crop face and preprocess
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            face_crops.append(self.transform(crop))
            bboxes.append((x1, y1, x2, y2))

        if not face_crops:
            return []

        # --- Gaze Estimation on Batch ---
        face_batch = torch.stack(face_crops).to(self.device)
        predictions = self.model(face_batch)
        decoded_preds = self._decode_predictions(predictions).cpu().numpy()

        # --- Package Results ---
        output = []
        for i in range(len(bboxes)):
            output.append({
                "bbox": bboxes[i],
                "gaze": {
                    "pitch": decoded_preds[i][0],
                    "yaw": decoded_preds[i][1],
                }
            })
        return output
