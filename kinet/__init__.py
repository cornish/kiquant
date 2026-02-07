"""
KiNet - Shared detection module for kiQuant and kinet-trainer.
Provides nucleus detection using KiNet, CellPose, and StarDist deep learning models.
"""

# Export commonly used classes
from .detector import BaseDetector, DetectedNucleus, DetectorFactory
from .model import Ki67Net

# Check for available detection libraries
_cellpose_available = False
_stardist_available = False
_kinet_available = False

try:
    import cellpose
    _cellpose_available = True
except ImportError:
    pass

try:
    import stardist
    _stardist_available = True
except ImportError:
    pass

try:
    import torch
    _kinet_available = True
except ImportError:
    pass


def is_cellpose_available() -> bool:
    """Check if CellPose is installed."""
    return _cellpose_available


def is_stardist_available() -> bool:
    """Check if StarDist is installed."""
    return _stardist_available


def is_kinet_available() -> bool:
    """Check if KiNet (PyTorch) is available."""
    return _kinet_available


def is_detection_available() -> bool:
    """Check if any detection model is available."""
    return _cellpose_available or _stardist_available or _kinet_available


def get_available_models() -> list:
    """
    Get list of available detection models.

    Returns:
        List of dicts with 'id' and 'name' keys for each available model.
    """
    models = []

    # KiNet first - it's specifically designed for Ki-67 IHC
    if _kinet_available:
        models.append({
            'id': 'kinet',
            'name': 'KiNet (Ki-67 IHC)'
        })

    if _cellpose_available:
        models.append({
            'id': 'cellpose',
            'name': 'CellPose (nuclei)'
        })

    if _stardist_available:
        models.append({
            'id': 'stardist',
            'name': 'StarDist (2D versatile)'
        })

    return models
