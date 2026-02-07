"""
KiNet training script.

Usage:
    Fine-tune from base model (auto-downloads if needed):
    python -m training.train --data-dir data/ --output-dir runs/exp01 --epochs 50

    Fine-tune from a registered model or weights file:
    python -m training.train --data-dir data/ --weights base --epochs 50
    python -m training.train --data-dir data/ --weights ft-20260206-143022 --epochs 50
    python -m training.train --data-dir data/ --weights path/to/model.pth --epochs 50

    Train from scratch (not recommended without large dataset):
    python -m training.train --data-dir data/ --output-dir runs/exp01 \
        --no-pretrained --epochs 200

    Resume interrupted training:
    python -m training.train --data-dir data/ --output-dir runs/exp01 \
        --resume runs/exp01/latest_checkpoint.pth
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repo root to path for kinet package
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from kinet import Ki67Net
from training.dataset import KiNetDataset
from training.augment import get_train_transform, get_val_transform

import model_registry


class WeightedMSELoss(nn.Module):
    """MSE loss with foreground weighting to counteract class imbalance."""

    def __init__(self, fg_weight=5.0, threshold=0.05):
        super().__init__()
        self.fg_weight = fg_weight
        self.threshold = threshold

    def forward(self, pred, target):
        # Weight foreground pixels more heavily
        # target: (B, 3, H, W), values in [0, 1]
        fg_mask = (target > self.threshold).float()
        weights = 1.0 + (self.fg_weight - 1.0) * fg_mask

        loss = weights * (pred - target) ** 2
        return loss.mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Clamp output to [0, 1] for proximity maps
        outputs = torch.clamp(outputs, 0, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """Validate model. Returns average loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = torch.clamp(outputs, 0, 1)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            count += images.size(0)

    return total_loss / count if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train KiNet model')
    parser.add_argument('--data-dir', required=True, help='Training data directory')
    parser.add_argument('--output-dir', default='runs/exp', help='Output directory')
    parser.add_argument('--weights', default=None,
                        help='Pre-trained weights: model ID (e.g. "base", "ft-20260206-143022") or file path')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume training from')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch (skip auto-loading pretrained weights)')
    parser.add_argument('--model-name', default=None,
                        help='Human-friendly name for the trained model (default: auto-generated)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--crop-size', type=int, default=256, help='Training crop size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--fg-weight', type=float, default=5.0, help='Foreground loss weight')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Datasets
    print("Loading datasets...")
    train_transform = get_train_transform(args.crop_size)
    val_transform = get_val_transform(args.crop_size)

    train_dataset = KiNetDataset(args.data_dir, 'train', transform=train_transform)
    val_dataset = KiNetDataset(args.data_dir, 'val', transform=val_transform)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    print("Creating model...")
    model = Ki67Net()

    # Load pre-trained weights
    start_epoch = 0
    best_val_loss = float('inf')
    parent_model_id = None  # Track lineage

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        parent_model_id = checkpoint.get('parent_model_id')
    elif args.weights:
        # Resolve via registry (accepts model IDs or file paths)
        weights_path, resolved_id = model_registry.resolve_weights_path(args.weights)
        if weights_path is None:
            print(f"Error: Could not resolve weights '{args.weights}'")
            print("  Provide a registered model ID or a valid file path.")
            sys.exit(1)
        parent_model_id = resolved_id or model_registry.find_model_id_by_path(weights_path)
        print(f"Loading pre-trained weights: {weights_path}")
        if resolved_id:
            print(f"  (registry model: {resolved_id})")
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    elif not args.no_pretrained:
        # Auto-download base model via registry
        print("Looking for base model weights...")
        try:
            cached = model_registry.ensure_base_model(
                progress_callback=lambda msg, p: print(f"  {msg}")
            )
            parent_model_id = 'base'
            print(f"Using base model: {cached}")
            print("  (use --no-pretrained to train from scratch)")
            state_dict = torch.load(cached, map_location=device, weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Could not obtain base model: {e}")
            print("Training from scratch.")
            print("  Or pass --weights <path> to specify weights manually.")

    model = model.to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = WeightedMSELoss(fg_weight=args.fg_weight)

    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training log
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    log_exists = os.path.exists(log_path) and args.resume
    log_file = open(log_path, 'a' if log_exists else 'w', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'time_sec'])

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Crop size: {args.crop_size}")
    print(f"  Foreground weight: {args.fg_weight}")
    print()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={current_lr:.6f}  time={epoch_time:.1f}s")

        # Log
        log_writer.writerow([epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                             f'{current_lr:.6f}', f'{epoch_time:.1f}'])
        log_file.flush()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> New best val loss: {val_loss:.6f}")

        # Save latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'parent_model_id': parent_model_id,
        }, os.path.join(args.output_dir, 'latest_checkpoint.pth'))

    log_file.close()

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'final_model.pth'))

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Models saved to: {args.output_dir}")

    # Register best model in the registry
    try:
        new_model_id = model_registry.generate_model_id()
        best_model_src = os.path.join(args.output_dir, 'best_model.pth')

        if os.path.exists(best_model_src):
            # Copy to canonical registry location
            registry_path = os.path.join(
                model_registry.MODEL_DIR, f'{new_model_id}.pth'
            )
            os.makedirs(model_registry.MODEL_DIR, exist_ok=True)
            shutil.copy2(best_model_src, registry_path)

            # Gather training data summary
            training_data = {
                'data_dir': os.path.abspath(args.data_dir),
                'train_images': len(train_dataset),
                'val_images': len(val_dataset),
            }

            model_name = args.model_name or f"Fine-tuned {new_model_id.replace('ft-', '')}"

            entry = model_registry.register_model(
                model_id=new_model_id,
                name=model_name,
                path=registry_path,
                parent_model=parent_model_id,
                training_data=training_data,
                metrics={
                    'best_val_loss': float(best_val_loss),
                    'epochs_trained': args.epochs - start_epoch,
                },
                description=f"Trained on {len(train_dataset)} images for {args.epochs - start_epoch} epochs"
            )

            # Print lineage
            lineage = model_registry.get_model_lineage(new_model_id)
            lineage_str = ' -> '.join(m['id'] for m in lineage)
            print(f"\n  Model registered: {new_model_id}")
            print(f"  Lineage: {lineage_str}")
            print(f"  Registry path: {registry_path}")
    except Exception as e:
        print(f"\n  Warning: Could not register model: {e}")
        print(f"  Best model is still saved at: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
