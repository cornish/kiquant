"""
Joint image+label augmentations for KiNet training.

All transforms operate on numpy arrays:
  - image: float32 (H, W, 3) in [0, 1]
  - labels: float32 (3, H, W) in [0, 1]

Geometric transforms are applied to both image and labels.
Color transforms are applied to image only.
"""

import numpy as np


class Compose:
    """Chain multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, labels):
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels


class RandomCrop:
    """Random crop of (image, labels) pair."""

    def __init__(self, size):
        """Args: size: int or (height, width)"""
        if isinstance(size, int):
            self.height = self.width = size
        else:
            self.height, self.width = size

    def __call__(self, image, labels):
        h, w = image.shape[:2]

        if h <= self.height or w <= self.width:
            # Image smaller than crop: pad with zeros
            pad_h = max(0, self.height - h)
            pad_w = max(0, self.width - w)
            if pad_h > 0 or pad_w > 0:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                labels = np.pad(labels, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
                h, w = image.shape[:2]

        y = np.random.randint(0, h - self.height + 1)
        x = np.random.randint(0, w - self.width + 1)

        image = image[y:y+self.height, x:x+self.width]
        labels = labels[:, y:y+self.height, x:x+self.width]

        return image, labels


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, labels):
        if np.random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            labels = np.flip(labels, axis=2).copy()
        return image, labels


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, labels):
        if np.random.random() < self.p:
            image = np.flip(image, axis=0).copy()
            labels = np.flip(labels, axis=1).copy()
        return image, labels


class RandomRotation90:
    """Random 90-degree rotation (0, 90, 180, or 270 degrees)."""

    def __call__(self, image, labels):
        k = np.random.randint(0, 4)
        if k > 0:
            # image: (H, W, 3) - rotate on axes (0, 1)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            # labels: (3, H, W) - rotate on axes (1, 2)
            labels = np.rot90(labels, k, axes=(1, 2)).copy()
        return image, labels


class ColorJitter:
    """Random brightness, contrast, and saturation adjustment (image only)."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, labels):
        # Brightness
        if self.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            image = image * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Saturation
        if self.saturation > 0:
            factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            gray = image.mean(axis=2, keepdims=True)
            image = gray + (image - gray) * factor

        image = np.clip(image, 0, 1)
        return image, labels


def get_train_transform(crop_size=256):
    """Get default training augmentation pipeline."""
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomRotation90(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])


def get_val_transform(crop_size=256):
    """Get validation transform (center crop only)."""
    return Compose([
        RandomCrop(crop_size),  # For val, still need fixed-size inputs
    ])
