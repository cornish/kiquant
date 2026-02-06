"""
Image preparation utilities for KiNet training.

Tiles large microscopy images into training-ready patches.
Can be used standalone (CLI) or imported by the GUI.
"""

import argparse
import os

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}


def tile_images(source_dir, output_dir, tile_size=512, overlap=0,
                skip_blank=True, blank_threshold=0.9,
                progress_callback=None):
    """
    Tile large images into fixed-size patches for annotation and training.

    Args:
        source_dir: Directory containing source images.
        output_dir: Directory to write tile images.
        tile_size: Width and height of each tile in pixels.
        overlap: Pixel overlap between adjacent tiles (0 for non-overlapping).
        skip_blank: If True, skip tiles that are mostly white/background.
        blank_threshold: Fraction of near-white pixels to consider a tile blank.
        progress_callback: Optional callable(message, progress_fraction).

    Returns:
        Dict with stats: total_tiles, skipped_blank, saved, source_images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather source images
    source_files = []
    for filename in sorted(os.listdir(source_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            source_files.append(os.path.join(source_dir, filename))

    if not source_files:
        return {
            'success': False,
            'message': 'No images found in source directory',
            'total_tiles': 0, 'skipped_blank': 0, 'saved': 0,
            'source_images': 0
        }

    stats = {
        'total_tiles': 0,
        'skipped_blank': 0,
        'saved': 0,
        'source_images': len(source_files)
    }

    step = tile_size - overlap if overlap > 0 else tile_size

    for file_idx, filepath in enumerate(source_files):
        if progress_callback:
            progress_callback(
                f"Tiling image {file_idx + 1}/{len(source_files)}...",
                file_idx / len(source_files)
            )

        img = Image.open(filepath).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        base_name = os.path.splitext(os.path.basename(filepath))[0]

        row = 0
        for y_start in range(0, h, step):
            col = 0
            for x_start in range(0, w, step):
                y_end = y_start + tile_size
                x_end = x_start + tile_size

                # Discard partial edge tiles
                if y_end > h or x_end > w:
                    col += 1
                    continue

                stats['total_tiles'] += 1
                tile = img_np[y_start:y_end, x_start:x_end]

                # Skip blank tiles
                if skip_blank and _is_blank(tile, blank_threshold):
                    stats['skipped_blank'] += 1
                    col += 1
                    continue

                # Save tile
                tile_name = f"{base_name}_r{row}_c{col}.png"
                tile_path = os.path.join(output_dir, tile_name)
                Image.fromarray(tile).save(tile_path, format='PNG')
                stats['saved'] += 1

                col += 1
            row += 1

    if progress_callback:
        progress_callback("Tiling complete", 1.0)

    stats['success'] = True
    stats['message'] = (
        f"Tiled {stats['source_images']} images: "
        f"{stats['saved']} tiles saved, {stats['skipped_blank']} blank tiles skipped"
    )
    return stats


def _is_blank(tile_np, threshold=0.9):
    """
    Check if a tile is mostly blank (near-white background).

    A pixel is considered near-white if all RGB channels are > 230.
    Returns True if the fraction of near-white pixels exceeds the threshold.
    """
    white_mask = np.all(tile_np > 230, axis=2)
    fraction = white_mask.sum() / white_mask.size
    return fraction > threshold


def main():
    """CLI entry point for image tiling."""
    parser = argparse.ArgumentParser(description='Tile images for KiNet training')
    parser.add_argument('--source-dir', required=True, help='Source image directory')
    parser.add_argument('--output-dir', required=True, help='Output tile directory')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size in pixels (default: 512)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Overlap between tiles in pixels (default: 0)')
    parser.add_argument('--no-skip-blank', action='store_true',
                        help='Keep blank/white tiles')
    parser.add_argument('--blank-threshold', type=float, default=0.9,
                        help='Fraction of white pixels to consider a tile blank (default: 0.9)')
    args = parser.parse_args()

    def progress(message, fraction):
        print(f"  [{int(fraction * 100):3d}%] {message}")

    print(f"Tiling images from: {args.source_dir}")
    print(f"Output to: {args.output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    if args.overlap > 0:
        print(f"Overlap: {args.overlap}px")
    print()

    result = tile_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        skip_blank=not args.no_skip_blank,
        blank_threshold=args.blank_threshold,
        progress_callback=progress
    )

    print()
    print(result['message'])


if __name__ == '__main__':
    main()
