"""
KiNet-based nucleus detector for Ki-67 IHC images.

Based on: Xing et al. "Pixel-to-pixel Learning with Weak Supervision for
Single-stage Nucleus Recognition in Ki67 Images" IEEE TBME, 2019.
"""

import os
from typing import List, Optional, Callable
import numpy as np

from .detector import BaseDetector, DetectedNucleus


# Pretrained model URL
# TODO: Move to GitHub Releases when ready for release
# Future URL: https://github.com/cornish/kiquant/releases/download/vX.X.X/ki67net-best.pth
MODEL_URL = "https://www.dropbox.com/s/sl2l5z3d65l983t/ki67net-best.pth?dl=1"
MODEL_FILENAME = "ki67net-best.pth"


class KiNetDetector(BaseDetector):
    """
    Nucleus detector using KiNet.

    KiNet is specifically trained on Ki-67 IHC images and performs
    joint detection and classification (positive/negative) in a single pass.
    """

    def __init__(self):
        super().__init__()
        self._device = None

    def _get_model_path(self) -> str:
        """Get path to model weights, downloading if necessary."""
        # Store in user's home directory
        model_dir = os.path.expanduser('~/.kiquant/models')
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, MODEL_FILENAME)

    def _safe_load_state_dict(self, net, state_dict, progress_callback=None):
        """
        Load state dict, skipping any keys that don't match.
        Based on original KiNet's safe_load_state_dict.
        """
        own_state = net.state_dict()
        skipped = []
        loaded = []

        for name, param in state_dict.items():
            if name not in own_state:
                skipped.append(name)
                continue
            if hasattr(param, 'data'):
                param = param.data
            if own_state[name].size() != param.size():
                skipped.append(f"{name} (size mismatch)")
                continue
            own_state[name].copy_(param)
            loaded.append(name)

        if skipped and progress_callback:
            progress_callback(f"Loaded {len(loaded)} params, skipped {len(skipped)}", 0.8)
        elif progress_callback:
            progress_callback(f"Loaded {len(loaded)} parameters", 0.8)

    def _download_model(self, progress_callback: Optional[Callable] = None) -> str:
        """Download pretrained model weights."""
        import urllib.request

        model_path = self._get_model_path()

        if os.path.exists(model_path):
            return model_path

        if progress_callback:
            progress_callback("Downloading KiNet model...", 0.0)

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0 and progress_callback:
                progress = min(block_num * block_size / total_size, 1.0)
                progress_callback(f"Downloading KiNet model... {int(progress*100)}%", progress * 0.5)

        urllib.request.urlretrieve(MODEL_URL, model_path, report_progress)

        if progress_callback:
            progress_callback("Download complete", 0.5)

        return model_path

    def load_model(self, progress_callback: Optional[Callable] = None) -> None:
        """Load the KiNet model."""
        if self._model is not None:
            return

        if progress_callback:
            progress_callback("Loading KiNet model...", 0.0)

        import torch
        from .model import Ki67Net

        # Check for GPU
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Download model if needed
        model_path = self._download_model(progress_callback)

        if progress_callback:
            progress_callback("Initializing KiNet...", 0.6)

        # Create model (Ki67Net has fixed 3-class output)
        self._model = Ki67Net()

        # Load weights
        if progress_callback:
            progress_callback("Loading weights...", 0.7)

        state_dict = torch.load(model_path, map_location=self._device, weights_only=False)

        # Use safe loading that skips mismatched keys
        self._safe_load_state_dict(self._model, state_dict, progress_callback)

        self._model = self._model.to(self._device)
        self._model.eval()

        if progress_callback:
            progress_callback("KiNet model loaded", 1.0)

    def detect(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable] = None,
        settings: Optional[dict] = None
    ) -> List[DetectedNucleus]:
        """
        Detect nuclei using KiNet.

        KiNet simultaneously detects AND classifies nuclei as positive, negative,
        or other (non-tumor), so the returned DetectedNucleus objects include classification.

        Args:
            image: RGB image as numpy array (H, W, 3) with dtype uint8.
            progress_callback: Optional callback for progress updates.
            settings: Optional dict with 'min_distance', 'threshold', 'tile_size', 'include_other'.

        Returns:
            List of DetectedNucleus objects with classification.
        """
        if settings is None:
            settings = {}

        if self._model is None:
            self.load_model(progress_callback)

        if progress_callback:
            progress_callback("Running KiNet detection...", 0.1)

        import torch

        # Get parameters
        min_distance = settings.get('min_distance', 5)
        threshold = settings.get('threshold', 0.3)
        # Tile size for processing large images (default 1024, 0 = no tiling)
        tile_size = settings.get('tile_size', 1024)
        # Overlap between tiles to avoid boundary artifacts
        tile_overlap = settings.get('tile_overlap', 128)
        # Whether to include non-tumor (class 2) detections (default True for trainer, False for kiQuant)
        include_other = settings.get('include_other', True)

        # Prepare image for model
        # KiNet expects RGB float32 normalized to [0, 1]
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
            if image_float.max() > 1.0:
                image_float = image_float / 255.0

        h, w = image_float.shape[:2]

        # Clear CUDA cache before processing
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        # Decide whether to use tiled processing
        # Use tiling if image is larger than tile_size and tiling is enabled
        use_tiling = tile_size > 0 and (h > tile_size or w > tile_size)

        if use_tiling:
            voting_maps = self._process_tiled(
                image_float, tile_size, tile_overlap, progress_callback
            )
        else:
            voting_maps = self._process_whole(image_float, progress_callback)

        if progress_callback:
            progress_callback("Extracting nuclei...", 0.85)

        # Extract nuclei from voting maps
        nuclei = self._extract_nuclei(
            voting_maps,
            min_distance=min_distance,
            threshold=threshold,
            include_other=include_other
        )

        # Final CUDA cache cleanup
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(f"Detected {len(nuclei)} nuclei", 1.0)

        return nuclei

    def _process_whole(
        self,
        image_float: np.ndarray,
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """Process the entire image at once (for small images)."""
        import torch

        # Convert to tensor (B, C, H, W)
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self._device)

        if progress_callback:
            progress_callback("Running inference...", 0.3)

        # Run inference
        with torch.no_grad():
            voting_maps = self._model(image_tensor)
            voting_maps = voting_maps.cpu().numpy()[0]  # (3, H, W)

        return voting_maps

    def _process_tiled(
        self,
        image_float: np.ndarray,
        tile_size: int,
        overlap: int,
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Process image in overlapping tiles to handle large images.

        Uses overlapping tiles and blends results to avoid boundary artifacts.
        """
        import torch

        h, w = image_float.shape[:2]

        # Initialize output voting maps
        voting_maps = np.zeros((3, h, w), dtype=np.float32)
        # Weight map for blending overlapping regions
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Calculate tile positions with overlap
        step = tile_size - overlap
        y_positions = list(range(0, h - overlap, step))
        x_positions = list(range(0, w - overlap, step))

        # Ensure we cover the entire image
        if y_positions[-1] + tile_size < h:
            y_positions.append(h - tile_size)
        if x_positions[-1] + tile_size < w:
            x_positions.append(w - tile_size)

        total_tiles = len(y_positions) * len(x_positions)
        tile_idx = 0

        if progress_callback:
            progress_callback(f"Processing {total_tiles} tiles...", 0.2)

        # Create blending weights (cosine taper at edges)
        blend_weights = self._create_blend_weights(tile_size, overlap)

        for y_start in y_positions:
            for x_start in x_positions:
                tile_idx += 1

                # Extract tile (handle edge cases)
                y_end = min(y_start + tile_size, h)
                x_end = min(x_start + tile_size, w)
                actual_h = y_end - y_start
                actual_w = x_end - x_start

                tile = image_float[y_start:y_end, x_start:x_end]

                # Pad tile if it's smaller than tile_size (edge tiles)
                if actual_h < tile_size or actual_w < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                    padded_tile[:actual_h, :actual_w] = tile
                    tile = padded_tile

                # Convert to tensor
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
                tile_tensor = tile_tensor.to(self._device)

                # Run inference on tile
                with torch.no_grad():
                    tile_output = self._model(tile_tensor)
                    tile_output = tile_output.cpu().numpy()[0]  # (3, H, W)

                # Clear intermediate tensors
                del tile_tensor
                if self._device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Crop output to actual tile size if padded
                tile_output = tile_output[:, :actual_h, :actual_w]

                # Get weights for this tile
                tile_weights = blend_weights[:actual_h, :actual_w]

                # Adjust weights at image boundaries (no blending needed at edges)
                if y_start == 0:
                    tile_weights[:overlap//2, :] = 1.0
                if x_start == 0:
                    tile_weights[:, :overlap//2] = 1.0
                if y_end == h:
                    tile_weights[-(overlap//2):, :] = 1.0
                if x_end == w:
                    tile_weights[:, -(overlap//2):] = 1.0

                # Accumulate results with weights
                for c in range(3):
                    voting_maps[c, y_start:y_end, x_start:x_end] += tile_output[c] * tile_weights
                weight_map[y_start:y_end, x_start:x_end] += tile_weights

                # Update progress
                if progress_callback:
                    progress = 0.2 + 0.6 * (tile_idx / total_tiles)
                    progress_callback(f"Processing tile {tile_idx}/{total_tiles}...", progress)

        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-8)  # Avoid division by zero
        for c in range(3):
            voting_maps[c] /= weight_map

        return voting_maps

    def _create_blend_weights(self, tile_size: int, overlap: int) -> np.ndarray:
        """
        Create blending weights with cosine taper at edges.

        Returns weights that smoothly transition from 0 at edges to 1 in the center,
        ensuring seamless blending of overlapping tiles.
        """
        # Create 1D weight profile
        weights_1d = np.ones(tile_size, dtype=np.float32)

        # Cosine taper for overlap region
        taper_length = overlap // 2
        if taper_length > 0:
            taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_length)))
            weights_1d[:taper_length] = taper
            weights_1d[-taper_length:] = taper[::-1]

        # Create 2D weights
        weights_2d = np.outer(weights_1d, weights_1d)

        return weights_2d

    def _extract_nuclei(
        self,
        voting_maps: np.ndarray,
        min_distance: int = 5,
        threshold: float = 0.3,
        include_other: bool = True
    ) -> List[DetectedNucleus]:
        """
        Extract nuclei coordinates and classifications from voting maps.

        Args:
            voting_maps: Shape (3, H, W) with channels for positive, negative, non-tumor.
            min_distance: Minimum distance between detected peaks.
            threshold: Detection threshold (0-1).
            include_other: Whether to include non-tumor (class 2) detections.

        Returns:
            List of DetectedNucleus with x, y, confidence, and marker_class.
        """
        from skimage.feature import peak_local_max

        nuclei = []

        # Channel 0: positive tumor nuclei
        # Channel 1: negative tumor nuclei
        # Channel 2: non-tumor / other

        # Create combined detection map (max of all channels)
        if include_other:
            detect_map = np.maximum(np.maximum(voting_maps[0], voting_maps[1]), voting_maps[2])
        else:
            detect_map = np.maximum(voting_maps[0], voting_maps[1])

        # Normalize to 0-1 range
        detect_min = detect_map.min()
        detect_max = detect_map.max()
        if detect_max > detect_min:
            detect_map_norm = (detect_map - detect_min) / (detect_max - detect_min)
        else:
            detect_map_norm = detect_map

        # Find local maxima (potential nuclei)
        coordinates = peak_local_max(
            detect_map_norm,
            min_distance=min_distance,
            threshold_abs=threshold
        )

        # For each detected point, determine class based on which channel is strongest
        for coord in coordinates:
            y, x = coord
            pos_score = voting_maps[0, y, x]
            neg_score = voting_maps[1, y, x]
            other_score = voting_maps[2, y, x]

            # Confidence is the detection strength
            confidence = float(detect_map_norm[y, x])

            # Class is determined by which channel is strongest
            # 0 = positive (green), 1 = negative (red), 2 = other (blue)
            if include_other:
                scores = [pos_score, neg_score, other_score]
                marker_class = int(np.argmax(scores))
            else:
                marker_class = 0 if pos_score >= neg_score else 1

            nuclei.append(DetectedNucleus(
                x=int(x),
                y=int(y),
                confidence=confidence,
                marker_class=marker_class
            ))

        return nuclei
