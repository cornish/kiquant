"""
Base detector classes and factory for nucleus detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np


@dataclass
class DetectedNucleus:
    """A detected nucleus with centroid coordinates and confidence score."""
    x: int
    y: int
    confidence: float = 1.0


class BaseDetector(ABC):
    """Abstract base class for nucleus detectors."""

    def __init__(self):
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @abstractmethod
    def load_model(self, progress_callback: Optional[Callable] = None) -> None:
        """
        Load the detection model.

        Args:
            progress_callback: Optional callback for progress updates.
                              Called with (message: str, progress: float 0-1)
        """
        pass

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable] = None,
        settings: Optional[dict] = None
    ) -> List[DetectedNucleus]:
        """
        Detect nuclei in the image.

        Args:
            image: RGB image as numpy array (H, W, 3) with dtype uint8.
            progress_callback: Optional callback for progress updates.
                              Called with (message: str, progress: float 0-1)
            settings: Optional dict with algorithm parameters.

        Returns:
            List of DetectedNucleus objects with centroid coordinates.
        """
        pass

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None


class DetectorFactory:
    """Factory for creating and caching detector instances."""

    _detectors: dict = {}

    @classmethod
    def get_detector(cls, model_name: str) -> BaseDetector:
        """
        Get a detector instance by model name.

        Args:
            model_name: 'cellpose' or 'stardist'

        Returns:
            Detector instance (cached).

        Raises:
            ValueError: If model name is not recognized.
            ImportError: If required library is not installed.
        """
        if model_name in cls._detectors:
            return cls._detectors[model_name]

        if model_name == 'cellpose':
            from .cellpose_detector import CellPoseDetector
            detector = CellPoseDetector()
        elif model_name == 'stardist':
            from .stardist_detector import StarDistDetector
            detector = StarDistDetector()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        cls._detectors[model_name] = detector
        return detector

    @classmethod
    def clear_cache(cls) -> None:
        """Unload all cached detectors."""
        for detector in cls._detectors.values():
            detector.unload_model()
        cls._detectors.clear()
