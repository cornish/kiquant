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

### Dependencies
AI detection requires additional packages not in base requirements:
- KiNet: `torch`, `scikit-image`
- CellPose: `cellpose>=3.0,<4.0`
- StarDist: `stardist`, `tensorflow>=2.10`

Install via `scripts/install-ai.bat` (CPU) or `scripts/install-ai-gpu.bat` (GPU).

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
