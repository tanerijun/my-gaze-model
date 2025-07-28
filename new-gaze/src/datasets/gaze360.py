import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Gaze360Dataset(Dataset):
    def __init__(self, data_root, split, num_bins, angle_range, bin_width):
        self.data_root = data_root
        self.split = split
        self.num_bins = num_bins
        self.angle_range = angle_range
        self.bin_width = bin_width

        self.transform = transforms.Compose([
            # transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # The folder structure of Gaze360 is assumed to be /path/to/Gaze360/Image and /path/to/Gaze360/Label
        self.image_dir = os.path.join(self.data_root, 'Image')
        label_file = os.path.join(self.data_root, 'Label', f'{split}.label')

        self.lines = []
        with open(label_file) as f:
            lines = f.readlines()[1:] # skip header
            print(f"Loading {split} labels from {label_file}...")

            for line in tqdm(lines):
                parts = line.strip().split()
                gaze_2d = np.array(parts[5].split(",")).astype(float) # pitch, yaw in radians

                # Convert to degrees
                pitch_deg = gaze_2d[0] * 180 / np.pi
                yaw_deg = gaze_2d[1] * 180 / np.pi

                # Filter out samples with extreme angles
                if abs(pitch_deg) < 90 and abs(yaw_deg) < 90:
                    self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()

        image_path = os.path.join(self.image_dir, line[0])
        gaze_2d = np.array(line[5].split(",")).astype(float) # pitch, yaw in radians

        # Continuous labels in degrees for regression
        pitch_deg = gaze_2d[0] * 180 / np.pi
        yaw_deg = gaze_2d[1] * 180 / np.pi
        cont_labels = torch.FloatTensor([pitch_deg, yaw_deg])

        # Binned labels for classification
        # The range [-90, 90] is shifted to [0, 180] before binning
        pitch_binned = np.floor((pitch_deg + self.angle_range / 2) / self.bin_width).astype(int)
        yaw_binned = np.floor((yaw_deg + self.angle_range / 2) / self.bin_width).astype(int)

        # Clamp to handle edge cases where angle is exactly +90 or -90
        pitch_binned = np.clip(pitch_binned, 0, self.num_bins - 1)
        yaw_binned = np.clip(yaw_binned, 0, self.num_bins - 1)
        binned_labels = torch.LongTensor([pitch_binned, yaw_binned])

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, binned_labels, cont_labels
