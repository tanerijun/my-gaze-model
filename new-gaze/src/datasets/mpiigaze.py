# src/datasets/mpiigaze.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MPIIGazeDataset(Dataset):
    def __init__(self, data_root, split, num_bins, angle_range, bin_width):
        # For MPIIGaze, 'split' will refer to the person ID (e.g., 'p00', 'p01')
        self.data_root = data_root
        self.person_id = split
        self.num_bins = num_bins
        self.angle_range = angle_range
        self.bin_width = bin_width

        # Updated transform to match Gaze360 for consistency
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- CORRECTED PATHS ---
        self.image_dir = os.path.join(self.data_root, 'Image')
        label_file = os.path.join(self.data_root, 'Label', f'{self.person_id}.label')

        self.lines = []
        with open(label_file) as f:
            # The Mobile-Gaze labels have a header, so we skip it
            lines = f.readlines()[1:]
            print(f"Loading labels for person {self.person_id} from {label_file}...")
            for line in tqdm(lines):
                parts = line.strip().split()
                # --- CORRECTED LABEL PARSING (Mobile-Gaze format) ---
                # Gaze is in column 7 (index 6) for this format
                gaze_2d = np.array(parts[7].split(",")).astype(float) # pitch, yaw in radians

                # Convert to degrees
                pitch_deg = gaze_2d[0] * 180 / np.pi
                yaw_deg = gaze_2d[1] * 180 / np.pi

                # Filter out samples with extreme angles, as is standard practice
                if abs(pitch_deg) < 42 and abs(yaw_deg) < 42:
                     # Store the full path to the image and the gaze data
                     # Path is in column 0 (index 0)
                     full_image_path = os.path.join(self.image_dir, parts[0])
                     self.lines.append((full_image_path, gaze_2d))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_path, gaze_2d = self.lines[idx]

        # --- Continuous labels in degrees ---
        pitch_deg = gaze_2d[0] * 180 / np.pi
        yaw_deg = gaze_2d[1] * 180 / np.pi
        cont_labels = torch.FloatTensor([pitch_deg, yaw_deg])

        # --- Binned labels for classification ---
        pitch_binned = np.floor((pitch_deg + self.angle_range / 2) / self.bin_width).astype(int)
        yaw_binned = np.floor((yaw_deg + self.angle_range / 2) / self.bin_width).astype(int)

        pitch_binned = np.clip(pitch_binned, 0, self.num_bins - 1)
        yaw_binned = np.clip(yaw_binned, 0, self.num_bins - 1)
        binned_labels = torch.LongTensor([pitch_binned, yaw_binned])

        # --- Load and transform image ---
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, binned_labels, cont_labels
