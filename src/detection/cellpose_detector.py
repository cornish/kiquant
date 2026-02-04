"""
CellPose-based nucleus detector.
"""

from typing import List, Optional, Callable
import numpy as np

from .detector import BaseDetector, DetectedNucleus


class CellPoseDetector(BaseDetector):
    """Nucleus detector using CellPose."""

    def __init__(self):
        super().__init__()
        self._gpu = False

    def load_model(self, progress_callback: Optional[Callable] = None) -> None:
        """Load the CellPose nuclei model."""
        if self._model is not None:
            return

        if progress_callback:
            progress_callback("Loading CellPose model...", 0.0)

        from cellpose import models

        # Check for GPU availability
        try:
            from cellpose import core
            self._gpu = core.use_gpu()
        except Exception:
            self._gpu = False

        # Load nuclei model (CellPose 3.x API)
        self._model = models.Cellpose(model_type='nuclei', gpu=self._gpu)

        if progress_callback:
            progress_callback("CellPose model loaded", 1.0)

    def detect(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable] = None
    ) -> List[DetectedNucleus]:
        """
        Detect nuclei using CellPose.

        Args:
            image: RGB image as numpy array (H, W, 3) with dtype uint8.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of DetectedNucleus objects.
        """
        if self._model is None:
            self.load_model(progress_callback)

        if progress_callback:
            progress_callback("Running CellPose detection...", 0.1)

        # Convert to grayscale for nuclei detection
        if image.ndim == 3 and image.shape[2] == 3:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = image

        # Run segmentation (CellPose 3.x API)
        # channels=[0, 0] means grayscale input
        masks, flows, styles, diams = self._model.eval(
            gray,
            diameter=None,  # Auto-estimate
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )

        if progress_callback:
            progress_callback("Extracting centroids...", 0.8)

        # Extract centroids from masks
        nuclei = self._extract_centroids(masks)

        if progress_callback:
            progress_callback(f"Detected {len(nuclei)} nuclei", 1.0)

        return nuclei

    def _extract_centroids(self, masks: np.ndarray) -> List[DetectedNucleus]:
        """
        Extract centroid coordinates from segmentation masks.

        Args:
            masks: Label image where each unique value > 0 is a cell.

        Returns:
            List of DetectedNucleus with centroid coordinates.
        """
        nuclei = []

        # Get unique labels (excluding background = 0)
        labels = np.unique(masks)
        labels = labels[labels > 0]

        for label in labels:
            # Find pixels belonging to this cell
            ys, xs = np.where(masks == label)

            if len(xs) == 0:
                continue

            # Calculate centroid
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            nuclei.append(DetectedNucleus(x=cx, y=cy, confidence=1.0))

        return nuclei
