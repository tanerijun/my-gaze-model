import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MPIIGazeDataset(Dataset):
    def __init__(self, data_root, split, num_bins, angle_range, image_size=224):
        self.data_root = data_root
        self.split = split  # For MPII, split can be 'train', 'val', or a person ID like 'p00'
        self.num_bins = num_bins
        self.angle_range = angle_range
        self.bin_width = self.angle_range / self.num_bins

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.image_dir = os.path.join(self.data_root, 'Image')
        # The label file is named after the split (e.g., 'train.label' or 'p14.label')
        label_file = os.path.join(self.data_root, 'Label', f'{self.split}.label')

        self.lines = []
        with open(label_file) as f:
            lines_raw = f.readlines()[1:] # skip header

        # The MPIIFaceGaze dataset is naturally filtered during preprocessing to be within ~42 degrees.
        self.lines = lines_raw
        print(f"Loading {len(self.lines)} samples for split '{self.split}' from {label_file}...")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()

        # The image path is relative to the Image/ directory
        image_path = os.path.join(self.image_dir, line[0])
        # Gaze is in column 8 (index 7), in radians
        gaze_2d = np.array(line[7].split(",")).astype(float)

        # Continuous labels in degrees
        pitch_deg = gaze_2d[0] * 180 / np.pi
        yaw_deg = gaze_2d[1] * 180 / np.pi
        cont_labels = torch.FloatTensor([pitch_deg, yaw_deg])

        # Binned labels for classification. For MPII, range is [-42, 42]
        pitch_binned = np.floor((pitch_deg + self.angle_range / 2) / self.bin_width).astype(int)
        yaw_binned = np.floor((yaw_deg + self.angle_range / 2) / self.bin_width).astype(int)

        pitch_binned = np.clip(pitch_binned, 0, self.num_bins - 1)
        yaw_binned = np.clip(yaw_binned, 0, self.num_bins - 1)
        binned_labels = torch.LongTensor([pitch_binned, yaw_binned])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, binned_labels, cont_labels
