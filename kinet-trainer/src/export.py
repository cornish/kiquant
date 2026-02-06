"""
Export annotated data to KiNet training format.

Creates the directory structure:
  output_dir/
    images/train/     - Training images (PNG)
    images/val/       - Validation images (PNG)
    labels_postm/train/  - Positive tumor proximity maps
    labels_postm/val/
    labels_negtm/train/  - Negative tumor proximity maps
    labels_negtm/val/
    labels_other/train/  - Non-tumor proximity maps
    labels_other/val/
"""

import os
import random
import shutil
import numpy as np
from PIL import Image

from state import MarkerClass, ReviewStatus


def generate_proximity_map(width: int, height: int, centroids: list,
                           sigma: float = 6.0) -> np.ndarray:
    """
    Generate a proximity map (voting map) from centroid coordinates.

    Each centroid creates a Gaussian blob: exp(-d^2 / 2*sigma^2) * 255.
    Overlapping blobs are merged with np.maximum.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        centroids: List of (x, y) tuples
        sigma: Gaussian sigma (default 6.0)

    Returns:
        uint8 ndarray of shape (height, width)
    """
    result = np.zeros((height, width), dtype=np.float32)

    if not centroids:
        return result.astype(np.uint8)

    # Bound radius to 4*sigma for efficiency
    radius = int(4 * sigma)

    for cx, cy in centroids:
        # Calculate bounding box (clipped to image bounds)
        x_min = max(0, cx - radius)
        x_max = min(width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(height, cy + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Create coordinate grids for the bounding box
        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(xs, ys)

        # Compute Gaussian
        d_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        blob = np.exp(-d_sq / (2 * sigma * sigma)) * 255.0

        # Accumulate with maximum
        result[y_min:y_max, x_min:x_max] = np.maximum(
            result[y_min:y_max, x_min:x_max], blob
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def export_training_data(fields: list, output_dir: str,
                         val_split: float = 0.2, sigma: float = 6.0,
                         seed: int = 42,
                         progress_callback=None,
                         reviewed_only: bool = False) -> dict:
    """
    Export annotated fields to KiNet training directory structure.

    Args:
        fields: List of Field objects with annotations
        output_dir: Root output directory
        val_split: Fraction of images for validation (0.0-1.0)
        sigma: Gaussian sigma for proximity maps
        seed: Random seed for train/val split
        progress_callback: Optional callable(message, progress_fraction)
        reviewed_only: If True, only export fields with review_status == REVIEWED

    Returns:
        Dict with export statistics
    """
    # Filter to fields that have at least one annotation
    annotated = [f for f in fields if f.get_total_count() > 0]

    if reviewed_only:
        annotated = [f for f in annotated
                     if f.review_status == ReviewStatus.REVIEWED]

    if not annotated:
        return {'success': False, 'message': 'No annotated images to export'}

    # Train/val split at image level
    random.seed(seed)
    indices = list(range(len(annotated)))
    random.shuffle(indices)
    val_count = max(1, int(len(annotated) * val_split))
    val_indices = set(indices[:val_count])

    # Create directory structure
    dirs = {}
    for split in ('train', 'val'):
        dirs[f'images_{split}'] = os.path.join(output_dir, 'images', split)
        dirs[f'postm_{split}'] = os.path.join(output_dir, 'labels_postm', split)
        dirs[f'negtm_{split}'] = os.path.join(output_dir, 'labels_negtm', split)
        dirs[f'other_{split}'] = os.path.join(output_dir, 'labels_other', split)

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    stats = {
        'train_images': 0, 'val_images': 0,
        'total_positive': 0, 'total_negative': 0, 'total_other': 0
    }

    for i, field in enumerate(annotated):
        if progress_callback:
            progress_callback(
                f"Exporting image {i + 1}/{len(annotated)}...",
                i / len(annotated)
            )

        split = 'val' if i in val_indices else 'train'
        stats[f'{split}_images'] += 1

        # Load image to get dimensions
        img = Image.open(field.filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.width, img.height

        # Use consistent filename (PNG)
        base_name = os.path.splitext(os.path.basename(field.filepath))[0] + '.png'

        # Copy image as PNG
        img_out = os.path.join(dirs[f'images_{split}'], base_name)
        img.save(img_out, format='PNG')

        # Collect centroids per class
        pos_centroids = [(m.x, m.y) for m in field.markers
                         if m.marker_class == MarkerClass.POSITIVE]
        neg_centroids = [(m.x, m.y) for m in field.markers
                         if m.marker_class == MarkerClass.NEGATIVE]
        other_centroids = [(m.x, m.y) for m in field.markers
                           if m.marker_class == MarkerClass.OTHER]

        stats['total_positive'] += len(pos_centroids)
        stats['total_negative'] += len(neg_centroids)
        stats['total_other'] += len(other_centroids)

        # Generate and save proximity maps
        for centroids, label_key in [
            (pos_centroids, f'postm_{split}'),
            (neg_centroids, f'negtm_{split}'),
            (other_centroids, f'other_{split}')
        ]:
            pmap = generate_proximity_map(w, h, centroids, sigma)
            label_path = os.path.join(dirs[label_key], base_name)
            Image.fromarray(pmap).save(label_path, format='PNG')

    if progress_callback:
        progress_callback("Export complete", 1.0)

    stats['success'] = True
    stats['output_dir'] = output_dir
    stats['message'] = (
        f"Exported {stats['train_images']} train + {stats['val_images']} val images "
        f"({stats['total_positive']} pos, {stats['total_negative']} neg, "
        f"{stats['total_other']} other markers)"
    )
    return stats
