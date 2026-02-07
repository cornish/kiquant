"""
KiNet fine-tuning script.

Fine-tunes the base KiNet model on custom annotated data.

Usage:
    python -m training.train --data-dir ./exported_data --epochs 50

The script expects data in the format created by Export Training Data:
    data_dir/
        images/train/     - Training images
        images/val/       - Validation images
        labels_postm/     - Positive tumor proximity maps
        labels_negtm/     - Negative tumor proximity maps
        labels_other/     - Non-tumor proximity maps
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Add kinet module to path
_repo_root = os.path.dirname(os.path.dirname(_src_dir))
_kinet_dir = os.path.join(_repo_root, 'kinet')
if _kinet_dir not in sys.path:
    sys.path.insert(0, _kinet_dir)

from training.dataset import KiNetDataset
from training.augment import get_train_transform, get_val_transform
import model_registry


def get_model():
    """Import and return the Ki67Net model class."""
    from model import Ki67Net
    return Ki67Net()


def load_weights(model, weights_path, device, freeze_encoder=False):
    """
    Load weights into model.

    Args:
        model: Ki67Net model instance
        weights_path: Path to .pth weights file
        device: torch device
        freeze_encoder: If True, freeze encoder layers for fine-tuning

    Returns:
        Number of parameters loaded
    """
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)

    # Safe loading - skip mismatched keys
    own_state = model.state_dict()
    loaded = 0

    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if hasattr(param, 'data'):
            param = param.data
        if own_state[name].size() != param.size():
            continue
        own_state[name].copy_(param)
        loaded += 1

    # Optionally freeze encoder layers
    if freeze_encoder:
        encoder_prefixes = ['in_tr', 'down_tr']
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in encoder_prefixes):
                param.requires_grad = False

    return loaded


def count_parameters(model, only_trainable=True):
    """Count model parameters."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Epoch {epoch}/{total_epochs} | "
                  f"Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss_meter.avg:.4f}")

    return loss_meter.avg


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def train(
    data_dir: str,
    weights: str = 'base',
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    crop_size: int = 256,
    freeze_encoder: bool = False,
    output_dir: str = None,
    model_name: str = None,
    progress_callback=None
):
    """
    Fine-tune KiNet on custom data.

    Args:
        data_dir: Path to exported training data
        weights: Starting weights ('base', model ID, or path)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        crop_size: Random crop size for training
        freeze_encoder: Freeze encoder layers (transfer learning)
        output_dir: Where to save model (default: ~/.kiquant/models/)
        model_name: Name for the model in registry
        progress_callback: Optional callback(message, progress, metrics)

    Returns:
        Dict with training results and model info
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Resolve starting weights
    weights_path, parent_model_id = model_registry.resolve_weights_path(weights)
    if weights_path is None:
        # Try to ensure base model is downloaded
        if weights == 'base':
            weights_path = model_registry.ensure_base_model(
                progress_callback=lambda msg, p: print(f"  {msg}")
            )
            parent_model_id = 'base'
        else:
            raise ValueError(f"Could not resolve weights: {weights}")

    print(f"Starting from: {parent_model_id or weights_path}")

    # Load model
    if progress_callback:
        progress_callback("Loading model...", 0.05, None)

    model = get_model()
    loaded = load_weights(model, weights_path, device, freeze_encoder)
    model = model.to(device)

    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    print(f"Loaded {loaded} parameter tensors")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if freeze_encoder:
        print("Encoder layers frozen for transfer learning")

    # Setup data loaders
    if progress_callback:
        progress_callback("Loading datasets...", 0.1, None)

    train_transform = get_train_transform(crop_size)
    val_transform = get_val_transform(crop_size)

    train_dataset = KiNetDataset(data_dir, split='train', transform=train_transform)
    val_dataset = KiNetDataset(data_dir, split='val', transform=val_transform)

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=device.type == 'cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 50)

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    # Setup output directory
    if output_dir is None:
        output_dir = model_registry.MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Generate model ID
    model_id = model_registry.generate_model_id()
    checkpoint_path = os.path.join(output_dir, f"{model_id}.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, epochs
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

        # Progress callback
        if progress_callback:
            progress = 0.1 + 0.85 * (epoch / epochs)
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'lr': current_lr
            }
            progress_callback(f"Epoch {epoch}/{epochs}", progress, metrics)

    total_time = time.time() - start_time
    print("-" * 50)
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Register model
    if progress_callback:
        progress_callback("Registering model...", 0.98, None)

    if model_name is None:
        model_name = f"Fine-tuned {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    entry = model_registry.register_model(
        model_id=model_id,
        name=model_name,
        path=checkpoint_path,
        parent_model=parent_model_id,
        training_data=data_dir,
        metrics={
            'best_val_loss': float(best_val_loss),
            'best_epoch': best_epoch,
            'final_train_loss': float(train_losses[-1]),
            'epochs_trained': epochs,
            'train_images': len(train_dataset),
            'val_images': len(val_dataset),
        },
        description=f"Fine-tuned from {parent_model_id or 'custom weights'}"
    )

    print(f"\nModel registered: {model_id}")
    print(f"  Name: {model_name}")
    print(f"  Path: {checkpoint_path}")

    if progress_callback:
        progress_callback("Training complete!", 1.0, None)

    return {
        'success': True,
        'model_id': model_id,
        'model_name': model_name,
        'model_path': checkpoint_path,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'epochs_trained': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time_seconds': total_time
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune KiNet on custom annotated data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning from base model
  python -m training.train --data-dir ./exported_data --epochs 50

  # Fine-tune with frozen encoder (faster, less overfitting)
  python -m training.train --data-dir ./exported_data --freeze-encoder

  # Fine-tune from a previous fine-tuned model
  python -m training.train --data-dir ./exported_data --weights ft-20260207-143022

  # Custom learning rate and batch size
  python -m training.train --data-dir ./exported_data --lr 5e-5 --batch-size 8
        """
    )

    parser.add_argument(
        '--data-dir', required=True,
        help='Path to exported training data directory'
    )
    parser.add_argument(
        '--weights', default='base',
        help='Starting weights: "base", model ID, or path to .pth file'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--crop-size', type=int, default=256,
        help='Random crop size for training (default: 256)'
    )
    parser.add_argument(
        '--freeze-encoder', action='store_true',
        help='Freeze encoder layers (transfer learning mode)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for model (default: ~/.kiquant/models/)'
    )
    parser.add_argument(
        '--name',
        help='Name for the fine-tuned model'
    )

    args = parser.parse_args()

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Check for required subdirectories
    required = ['images/train', 'images/val', 'labels_postm/train']
    for subdir in required:
        path = os.path.join(args.data_dir, subdir)
        if not os.path.isdir(path):
            print(f"Error: Required directory not found: {path}")
            print("Did you export training data from KiNet Trainer?")
            sys.exit(1)

    try:
        result = train(
            data_dir=args.data_dir,
            weights=args.weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            crop_size=args.crop_size,
            freeze_encoder=args.freeze_encoder,
            output_dir=args.output_dir,
            model_name=args.name
        )

        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Model ID: {result['model_id']}")
        print(f"Best validation loss: {result['best_val_loss']:.4f}")
        print(f"Training time: {result['total_time_seconds']/60:.1f} minutes")
        print(f"\nModel saved to: {result['model_path']}")
        print("\nTo use this model for detection in KiNet Trainer,")
        print("select it from the model dropdown in the detection dialog.")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
