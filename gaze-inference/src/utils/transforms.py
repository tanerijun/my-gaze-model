import cv2
import torch
from torchvision import transforms


def get_gaze_transforms(image_size: int = 224):
    """
    Get the standard preprocessing transforms for gaze estimation.

    Args:
        image_size: Target image size (default 224 for MobileOne S1)

    Returns:
        torchvision.transforms.Compose: Preprocessing pipeline
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def preprocess_face_crop(face_crop, image_size: int = 224):
    """
    Preprocess a face crop for gaze estimation.

    Args:
        face_crop: Face crop as numpy array (H, W, 3) in BGR format
        image_size: Target size for the model

    Returns:
        torch.Tensor: Preprocessed tensor ready for model input
    """
    if face_crop.size == 0:
        raise ValueError("Empty face crop provided")

    # Convert BGR to RGB
    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # Apply transforms
    transform = get_gaze_transforms(image_size)
    return transform(face_crop_rgb)


def batch_preprocess_faces(face_crops, image_size: int = 224):
    """
    Batch preprocess multiple face crops.

    Args:
        face_crops: List of face crops as numpy arrays
        image_size: Target size for the model

    Returns:
        torch.Tensor: Batched tensor [N, 3, H, W]
    """
    if not face_crops:
        return torch.empty(0, 3, image_size, image_size)

    processed_faces = []
    for crop in face_crops:
        if crop.size > 0:
            processed_faces.append(preprocess_face_crop(crop, image_size))

    if not processed_faces:
        return torch.empty(0, 3, image_size, image_size)

    return torch.stack(processed_faces)
