"""
StarDist-based nucleus detector.
"""

from typing import List, Optional, Callable
import numpy as np

from .detector import BaseDetector, DetectedNucleus


class StarDistDetector(BaseDetector):
    """Nucleus detector using StarDist."""

    def __init__(self):
        super().__init__()

    def load_model(self, progress_callback: Optional[Callable] = None) -> None:
        """Load the StarDist 2D versatile model."""
        if self._model is not None:
            return

        if progress_callback:
            progress_callback("Loading StarDist model...", 0.0)

        from stardist.models import StarDist2D

        # Load pre-trained versatile fluorescence model
        # This model works well on various nuclear stains
        try:
            self._model = StarDist2D.from_pretrained('2D_versatile_fluo')
        except OSError as e:
            # Windows symlink issue - try to fix it
            if 'WinError 1314' in str(e) or 'privilege' in str(e).lower():
                self._fix_windows_model_extraction()
                self._model = StarDist2D.from_pretrained('2D_versatile_fluo')
            else:
                raise

        if progress_callback:
            progress_callback("StarDist model loaded", 1.0)

    def _fix_windows_model_extraction(self) -> None:
        """Fix StarDist model extraction on Windows (symlink privilege issue)."""
        import os

        base_dir = os.path.expanduser('~/.keras/models/StarDist2D/2D_versatile_fluo')
        extracted_dir = os.path.join(base_dir, '2D_versatile_fluo_extracted')
        target_dir = os.path.join(base_dir, '2D_versatile_fluo')

        # If extracted exists but target doesn't, rename it
        if os.path.exists(extracted_dir) and not os.path.exists(target_dir):
            os.rename(extracted_dir, target_dir)

    def detect(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable] = None
    ) -> List[DetectedNucleus]:
        """
        Detect nuclei using StarDist.

        Args:
            image: RGB image as numpy array (H, W, 3) with dtype uint8.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of DetectedNucleus objects.
        """
        if self._model is None:
            self.load_model(progress_callback)

        if progress_callback:
            progress_callback("Running StarDist detection...", 0.1)

        # StarDist expects a single-channel grayscale image
        if image.ndim == 3 and image.shape[2] == 3:
            # Use luminance formula for grayscale conversion
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # Normalize to [0, 1]
        gray_min, gray_max = gray.min(), gray.max()
        if gray_max > gray_min:
            gray = (gray - gray_min) / (gray_max - gray_min)

        if progress_callback:
            progress_callback("Predicting instances...", 0.3)

        # Run prediction
        labels, details = self._model.predict_instances(
            gray,
            prob_thresh=0.5,
            nms_thresh=0.4
        )

        if progress_callback:
            progress_callback("Extracting centroids...", 0.8)

        # Extract centroids from prediction details
        nuclei = self._extract_centroids(labels, details)

        if progress_callback:
            progress_callback(f"Detected {len(nuclei)} nuclei", 1.0)

        return nuclei

    def _extract_centroids(
        self,
        labels: np.ndarray,
        details: dict
    ) -> List[DetectedNucleus]:
        """
        Extract centroid coordinates from StarDist prediction.

        Args:
            labels: Label image.
            details: Prediction details containing 'points' and 'prob'.

        Returns:
            List of DetectedNucleus with centroid coordinates.
        """
        nuclei = []

        # StarDist returns 'points' (polygon centers) and 'prob' (probabilities)
        points = details.get('points', [])
        probs = details.get('prob', [])

        for i, point in enumerate(points):
            # points are in (y, x) format
            cy, cx = point
            prob = probs[i] if i < len(probs) else 1.0

            nuclei.append(DetectedNucleus(
                x=int(round(cx)),
                y=int(round(cy)),
                confidence=float(prob)
            ))

        return nuclei
