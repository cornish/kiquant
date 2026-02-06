# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

kiQuant is a cell marker quantification tool for immunohistochemistry analysis. It provides a simple interface for manually counting positive and negative cell markers on microscopy images.

**Technology Stack:**
- Backend: Python 3.11+ with Eel (web-based GUI framework)
- Frontend: HTML5 Canvas + vanilla JavaScript
- Packaging: PyInstaller for standalone Windows executable
- Image handling: Pillow

## Project Structure

```
kiquant/
├── src/                    # Source code
│   ├── main.py            # Entry point, Eel backend, exposed functions
│   ├── marker_state.py    # Data classes (Marker, Field, State)
│   └── web/               # Frontend files
│       ├── index.html     # UI layout
│       ├── style.css      # Styling (dark theme)
│       └── app.js         # Canvas rendering, event handling
├── scripts/               # Development scripts
│   ├── dev-setup.bat      # Create venv, install deps
│   ├── run.bat            # Run the application
│   └── build.bat          # Build executable
├── .github/workflows/     # GitHub Actions
│   ├── build.yml          # CI build on push
│   └── release.yml        # Create release on tag
├── requirements.txt       # Python dependencies
├── README.md              # User documentation
├── LICENSE                # MIT license
└── CHANGELOG.md           # Version history
```

## Running the Application

```batch
# First time setup
scripts\dev-setup.bat

# Run the app
scripts\run.bat

# Build executable
scripts\build.bat
```

## Key Architecture Decisions

### Eel Communication
- Python functions exposed via `@eel.expose` decorator
- Frontend calls these via `eel.function_name()()`
- All marker operations return updated state to keep frontend in sync

### State Management
- `State` class holds all project data (image list, markers, current index)
- Per-field undo/redo history stored in memory
- Auto-save on image navigation
- JSON format for project persistence

### Canvas Rendering
- Image drawn at native resolution
- CSS transform used for zoom/pan (no image resampling)
- Markers scaled inversely with zoom for consistent visual size
- Overview map rendered separately with scaled coordinates

### Modes
- POSITIVE (0): Click to add positive marker
- NEGATIVE (1): Click to add negative marker
- SELECT (2): Rectangle or lasso selection
- ERASER (3): Click to delete markers
- PAN (4): Drag to pan image

## AI Detection (experimental)

The `src/detection/` module provides optional AI-powered nucleus detection:

### Available Models
- **KiNet**: Purpose-built for Ki-67 IHC images. Provides joint detection AND classification (positive/negative) in a single pass. Based on Xing et al. "Pixel-to-pixel Learning with Weak Supervision for Single-stage Nucleus Recognition in Ki67 Images" (IEEE TBME, 2019). Original code: https://github.com/exhh/KiNet
- **CellPose**: General nucleus segmentation (requires hematoxylin extraction for IHC)
- **StarDist**: General nucleus detection (requires hematoxylin extraction for IHC)

### Model Weights
KiNet weights (~45MB) are currently hosted on Dropbox (from original repo).

**TODO for release**: Move weights to GitHub Releases for long-term reliability:
1. Create release (e.g., `v0.2.0` or `kinet-weights`)
2. Upload `ki67net-best.pth` as release asset
3. Update `MODEL_URL` in `src/detection/kinet_detector.py`

### Retraining KiNet
To improve the model with additional training data:

**Training data format:**
- RGB images in `images/train/` and `images/val/`
- Three label images per training image (proximity maps = Gaussian blobs at each nucleus):
  - `labels_postm/` - Positive tumor nuclei
  - `labels_negtm/` - Negative tumor nuclei
  - `labels_other/` - Non-tumor nuclei

**Proximity map generation:** `Label(x,y) = exp(-d²/2σ²) × 255` where d = distance from nucleus centroid

**Requirements:**
- CUDA GPU for training (~hours for 100k iterations)
- PyTorch 0.4.1+ (original) or modern PyTorch with minor adjustments
- Training script: `train.py` in original KiNet repo

**Potential workflow:**
1. Use kiQuant to manually annotate images (creates point annotations)
2. Export annotations to KiNet label format (would need export function)
3. Fine-tune from existing weights using `--weights` flag

See: https://github.com/exhh/KiNet for original training code.

### Dependencies
AI detection requires additional packages not in base requirements:
- KiNet: `torch`, `scikit-image`
- CellPose: `cellpose>=3.0,<4.0`
- StarDist: `stardist`, `tensorflow>=2.10`

Install via `scripts/install-ai.bat` (CPU) or `scripts/install-ai-gpu.bat` (GPU).

## KiNet Trainer (separate app)

The `kinet-trainer/` directory contains a separate Eel app for creating KiNet training data:

```
kinet-trainer/
├── src/
│   ├── main.py          # Eel backend: annotation, export, model management
│   ├── state.py         # 3-class annotation state (positive, negative, other)
│   ├── export.py        # Proximity map generation + KiNet directory export
│   ├── evaluate.py      # Model evaluation & metrics (CLI + GUI)
│   ├── training/
│   │   ├── train.py     # CLI training script (argparse)
│   │   ├── dataset.py   # PyTorch Dataset for KiNet format
│   │   └── augment.py   # Joint image+label augmentations
│   └── web/             # Annotation GUI (3-class, dark theme)
├── requirements.txt     # torch, scikit-image, Pillow, Eel, numpy, scipy
└── scripts/
    ├── run.bat          # Launch the GUI
    └── train.bat        # Run training from CLI
```

**Key differences from kiQuant:**
- 3 marker classes (positive tumor, negative tumor, non-tumor) vs 2
- Exports proximity maps for KiNet training format
- CLI training pipeline with weighted MSE loss
- Import from kiQuant projects (maps 2 classes, no "other")
- Shares `Ki67Net` model architecture from `src/detection/kinet_model.py`

**Running:**
```batch
cd kinet-trainer\scripts
run.bat          # GUI annotation tool
train.bat --data-dir <exported> --epochs 100  # CLI training
```

## Testing

No automated tests. Manual testing required:
1. Load images, verify display
2. Place markers, verify counts
3. Navigate images, verify markers persist
4. Export CSV, verify counts match
5. Build exe, test on clean Windows machine

## Version Management

Version is defined in `src/main.py` as `__version__`. Update this when releasing:
1. Update `__version__` in main.py
2. Update CHANGELOG.md
3. Commit changes
4. Tag with `git tag v0.x.x`
5. Push tag to trigger release workflow
