"""
Model evaluation for KiNet.

Compares predicted nucleus detections against ground truth annotations
using distance-based matching (Hungarian algorithm).

Can be used standalone (CLI) or imported by the GUI.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# Add parent src/detection to path for Ki67Net import
_kiquant_detection = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'detection'
)
if _kiquant_detection not in sys.path:
    sys.path.insert(0, _kiquant_detection)


def load_model(weights_path, device=None):
    """Load Ki67Net model with weights."""
    import torch
    from kinet_model import Ki67Net

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Ki67Net()
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, device


def predict_image(model, image_np, device):
    """
    Run model on a single image.

    Args:
        model: Ki67Net model
        image_np: RGB numpy array (H, W, 3) uint8 or float32
        device: torch device

    Returns:
        voting_maps: numpy array (3, H, W) float32
    """
    import torch

    if image_np.dtype == np.uint8:
        image_np = image_np.astype(np.float32) / 255.0

    # (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        output = torch.clamp(output, 0, 1)

    return output[0].cpu().numpy()  # (3, H, W)


def extract_peaks(voting_map, min_distance=5, threshold=0.3):
    """Extract peak coordinates from a single-channel voting map."""
    from skimage.feature import peak_local_max

    if voting_map.max() == 0:
        return np.zeros((0, 2), dtype=int)

    # Normalize
    vmax = voting_map.max()
    vmin = voting_map.min()
    if vmax > vmin:
        normalized = (voting_map - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(voting_map)

    coords = peak_local_max(normalized, min_distance=min_distance,
                            threshold_abs=threshold)
    return coords  # (N, 2) as (y, x)


def extract_detections(voting_maps, min_distance=5, threshold=0.3):
    """
    Extract classified detections from 3-channel voting maps.

    Returns list of (y, x, class) tuples.
    """
    detections = []

    # Combine all channels for detection
    combined = np.max(voting_maps, axis=0)

    if combined.max() == 0:
        return detections

    coords = extract_peaks(combined, min_distance, threshold)

    for y, x in coords:
        scores = voting_maps[:, y, x]
        cls = int(np.argmax(scores))
        detections.append((int(y), int(x), cls))

    return detections


def match_detections(gt_points, pred_points, max_distance=10):
    """
    Match predicted points to ground truth using distance-based matching.

    Args:
        gt_points: List of (y, x) tuples — ground truth
        pred_points: List of (y, x) tuples — predictions
        max_distance: Maximum distance for a valid match

    Returns:
        matches: List of (gt_idx, pred_idx) pairs
        unmatched_gt: List of gt indices with no match
        unmatched_pred: List of pred indices with no match
    """
    if not gt_points or not pred_points:
        return [], list(range(len(gt_points))), list(range(len(pred_points)))

    from scipy.optimize import linear_sum_assignment

    gt_arr = np.array(gt_points, dtype=np.float64)
    pred_arr = np.array(pred_points, dtype=np.float64)

    # Distance matrix
    diff = gt_arr[:, np.newaxis, :] - pred_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

    # Set distances > max_distance to a large value
    cost_matrix = dist_matrix.copy()
    cost_matrix[cost_matrix > max_distance] = 1e6

    # Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for gi, pi in zip(gt_indices, pred_indices):
        if dist_matrix[gi, pi] <= max_distance:
            matches.append((int(gi), int(pi)))
            matched_gt.add(int(gi))
            matched_pred.add(int(pi))

    unmatched_gt = [i for i in range(len(gt_points)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_points)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_metrics(gt_detections, pred_detections, max_distance=10):
    """
    Compute per-class precision, recall, F1 using distance-based matching.

    Args:
        gt_detections: List of (y, x, class) tuples
        pred_detections: List of (y, x, class) tuples
        max_distance: Max distance for matching

    Returns:
        Dict with per-class and overall metrics
    """
    class_names = {0: 'positive', 1: 'negative', 2: 'other'}
    results = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in range(3):
        gt_cls = [(y, x) for y, x, c in gt_detections if c == cls]
        pred_cls = [(y, x) for y, x, c in pred_detections if c == cls]

        matches, unmatched_gt, unmatched_pred = match_detections(
            gt_cls, pred_cls, max_distance
        )

        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[class_names[cls]] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'gt_count': len(gt_cls), 'pred_count': len(pred_cls)
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results['overall'] = {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1
    }

    return results


def evaluate_on_fields(weights_path, fields, min_distance=5, threshold=0.3,
                       max_match_distance=10, progress_callback=None):
    """
    Evaluate model on annotated Fields (from trainer GUI).

    Args:
        weights_path: Path to model weights
        fields: List of Field objects with ground truth markers
        min_distance: Peak detection min distance
        threshold: Peak detection threshold
        max_match_distance: Max distance for matching
        progress_callback: Optional callable(message, progress)

    Returns:
        Dict with evaluation results
    """
    # Only evaluate fields with annotations
    annotated = [f for f in fields if f.get_total_count() > 0]
    if not annotated:
        return {'success': False, 'message': 'No annotated images to evaluate'}

    if progress_callback:
        progress_callback("Loading model...", 0.0)

    model, device = load_model(weights_path)

    all_gt = []
    all_pred = []

    for i, field in enumerate(annotated):
        if progress_callback:
            progress_callback(
                f"Evaluating image {i+1}/{len(annotated)}...",
                (i + 1) / (len(annotated) + 1)
            )

        # Load image
        img = Image.open(field.filepath).convert('RGB')
        img_np = np.array(img)

        # Get ground truth from markers
        for m in field.markers:
            all_gt.append((m.y, m.x, m.marker_class))

        # Run model
        voting_maps = predict_image(model, img_np, device)
        detections = extract_detections(voting_maps, min_distance, threshold)
        all_pred.extend(detections)

    if progress_callback:
        progress_callback("Computing metrics...", 0.95)

    results = compute_metrics(all_gt, all_pred, max_match_distance)

    if progress_callback:
        progress_callback("Done", 1.0)

    results['success'] = True
    results['per_class'] = {
        k: v for k, v in results.items()
        if k in ('positive', 'negative', 'other')
    }
    return results


def main():
    """CLI evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate KiNet model')
    parser.add_argument('--model', required=True, help='Path to model weights')
    parser.add_argument('--data-dir', required=True, help='Data directory (KiNet format)')
    parser.add_argument('--split', default='val', help='Split to evaluate on')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--min-distance', type=int, default=5, help='Min peak distance')
    parser.add_argument('--max-match-distance', type=int, default=10,
                        help='Max distance for matching')
    parser.add_argument('--output', default=None, help='Output JSON file')
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model, device = load_model(args.model)

    # Load data
    from training.dataset import KiNetDataset
    dataset = KiNetDataset(args.data_dir, args.split)

    print(f"Evaluating on {len(dataset)} images from '{args.split}' split...")

    all_gt = []
    all_pred = []

    for i in range(len(dataset)):
        image, labels = dataset[i]

        # Get ground truth peaks from label maps
        for cls in range(3):
            label_map = labels[cls].numpy()
            gt_coords = extract_peaks(label_map, args.min_distance, args.threshold)
            for y, x in gt_coords:
                all_gt.append((int(y), int(x), cls))

        # Run model
        img_np = image.numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        voting_maps = predict_image(model, img_np, device)
        detections = extract_detections(voting_maps, args.min_distance, args.threshold)
        all_pred.extend(detections)

        print(f"  [{i+1}/{len(dataset)}] GT: {len([g for g in all_gt if True])} "
              f"Pred: {len(all_pred)}")

    results = compute_metrics(all_gt, all_pred, args.max_match_distance)

    # Print results
    print("\n=== Evaluation Results ===\n")
    for cls_name in ('positive', 'negative', 'other'):
        m = results[cls_name]
        print(f"  {cls_name:>10s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  (GT={m['gt_count']}, Pred={m['pred_count']}, "
              f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    m = results['overall']
    print(f"\n  {'overall':>10s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(args.model), 'evaluation_results.json'
    )
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
