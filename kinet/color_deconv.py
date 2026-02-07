"""
Color deconvolution for H-DAB stained images.
Used to classify detected nuclei as positive (DAB+) or negative (DAB-).
"""

import numpy as np
from typing import List, Tuple

# H-DAB stain vectors (from skimage.color.hdx_from_rgb)
# These are the optical density vectors for Hematoxylin and DAB
# Reference: Ruifrok & Johnston, Analytical and Quantitative Cytology and Histology

# Stain matrix columns: [Hematoxylin, DAB, Residual]
# Each column is the OD contribution of that stain in R, G, B channels
HDAB_MATRIX = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin
    [0.268, 0.570, 0.776],  # DAB
    [0.711, 0.423, 0.500]   # Residual/background
]).T

# Invert the matrix for deconvolution
HDAB_MATRIX_INV = np.linalg.inv(HDAB_MATRIX)


def separate_stains(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform H-DAB color deconvolution.

    Args:
        image: RGB image as numpy array (H, W, 3) with dtype uint8.

    Returns:
        Tuple of (hematoxylin_channel, dab_channel) as float arrays.
        Values are optical densities, higher = more stain.
    """
    # Convert to float and normalize to [0, 1]
    img_float = image.astype(np.float64) / 255.0

    # Avoid log(0) by clamping minimum values
    img_float = np.maximum(img_float, 1e-6)

    # Convert to optical density (Beer-Lambert law)
    od = -np.log(img_float)

    # Reshape for matrix multiplication: (H*W, 3)
    h, w = od.shape[:2]
    od_flat = od.reshape(-1, 3)

    # Deconvolve: multiply by inverse stain matrix
    stains_flat = od_flat @ HDAB_MATRIX_INV

    # Reshape back: (H, W, 3) where channels are [H, DAB, Residual]
    stains = stains_flat.reshape(h, w, 3)

    # Extract channels (clamp negative values to 0)
    hematoxylin = np.maximum(stains[:, :, 0], 0)
    dab = np.maximum(stains[:, :, 1], 0)

    return hematoxylin, dab


def get_dab_intensity_at_point(
    dab_channel: np.ndarray,
    x: int,
    y: int,
    radius: int = 5
) -> float:
    """
    Get average DAB intensity around a point.

    Args:
        dab_channel: DAB channel from separate_stains().
        x: X coordinate (column).
        y: Y coordinate (row).
        radius: Radius around point to average.

    Returns:
        Average DAB intensity (optical density) at point.
    """
    h, w = dab_channel.shape

    # Define bounding box
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)

    # Extract region and compute mean
    region = dab_channel[y_min:y_max, x_min:x_max]
    return float(np.mean(region))


def classify_by_dab_intensity(
    dab_channel: np.ndarray,
    centroids: List[Tuple[int, int]],
    threshold: float = 0.3,
    sampling_radius: int = 5
) -> List[bool]:
    """
    Classify nuclei as positive or negative based on DAB intensity.

    Args:
        dab_channel: DAB channel from separate_stains().
        centroids: List of (x, y) centroid coordinates.
        threshold: DAB intensity threshold (0-1 range, typical values 0.2-0.5).
                   Nuclei with intensity >= threshold are classified as positive.
        sampling_radius: Radius around centroid to sample for intensity.

    Returns:
        List of booleans, True = positive (DAB+), False = negative (DAB-).
    """
    # Normalize DAB channel to [0, 1] for threshold comparison
    dab_max = np.max(dab_channel)
    if dab_max > 0:
        dab_normalized = dab_channel / dab_max
    else:
        dab_normalized = dab_channel

    classifications = []
    for x, y in centroids:
        intensity = get_dab_intensity_at_point(
            dab_normalized, x, y, sampling_radius
        )
        classifications.append(intensity >= threshold)

    return classifications
