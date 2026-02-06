"""
PyTorch Dataset for KiNet training format.

Expects directory structure:
    data_dir/
        images/{split}/     - RGB images (PNG)
        labels_postm/{split}/  - Positive tumor proximity maps
        labels_negtm/{split}/  - Negative tumor proximity maps
        labels_other/{split}/  - Non-tumor proximity maps
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class KiNetDataset(Dataset):
    """Dataset for KiNet format training data."""

    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Args:
            data_dir: Root directory containing images/ and labels_*/
            split: 'train' or 'val'
            transform: Optional transform applied to (image, labels) pair
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Find all image files
        image_dir = os.path.join(data_dir, 'images', split)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        if not self.image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Verify label directories exist
        self.label_dirs = {
            'postm': os.path.join(data_dir, 'labels_postm', split),
            'negtm': os.path.join(data_dir, 'labels_negtm', split),
            'other': os.path.join(data_dir, 'labels_other', split),
        }
        for name, d in self.label_dirs.items():
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Label directory not found: {d}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        # Load image as float32 [0, 1]
        img_path = os.path.join(self.data_dir, 'images', self.split, filename)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

        # Load 3 label channels as float32 [0, 1]
        label_channels = []
        for key in ('postm', 'negtm', 'other'):
            label_path = os.path.join(self.label_dirs[key], filename)
            if os.path.exists(label_path):
                lbl = Image.open(label_path).convert('L')
                lbl_np = np.array(lbl, dtype=np.float32) / 255.0  # (H, W)
            else:
                # Fallback: zero label if not found
                lbl_np = np.zeros(img_np.shape[:2], dtype=np.float32)
            label_channels.append(lbl_np)

        labels_np = np.stack(label_channels, axis=0)  # (3, H, W)

        # Apply transforms (expect numpy arrays)
        if self.transform:
            img_np, labels_np = self.transform(img_np, labels_np)

        # Convert to tensors
        # Image: (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(img_np.transpose(2, 0, 1).copy())
        labels = torch.from_numpy(labels_np.copy())

        return image, labels
