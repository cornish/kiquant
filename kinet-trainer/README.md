# KiNet Trainer

Fine-tuning and training tool for the KiNet Ki-67 detection model.

KiNet ships with pre-trained weights that work reasonably well out of the box. This trainer lets you **improve the model on your own images** through a detect-then-correct workflow:

1. **Prepare** large images by tiling them into training-ready patches
2. **Detect** nuclei using the existing model (or a fine-tuned model)
3. **Review & correct** predictions — fix errors, add missed nuclei
4. **Export** reviewed annotations as KiNet training data
5. **Train** an improved model on your corrected data
6. **Iterate** — detect with the new model, review, export, retrain

This is a companion app to [kiQuant](../README.md). While kiQuant handles 2-class quantification (positive/negative), the trainer adds the third "non-tumor" class needed for training. You can import existing kiQuant projects as a starting point.

## Quick Start

```batch
# 1. Set up environment
scripts\dev-setup.bat

# 2. (Optional) Install GPU support for training
scripts\install-gpu.bat

# 3. Launch the annotation GUI
scripts\run.bat
```

## Pre-trained Weights & Model Registry

The trainer manages models through a **model registry** stored at `~/.kiquant/models/registry.json`. Each model tracks its lineage (which parent model and training data produced it).

The base KiNet weights are **automatically downloaded** the first time you run detection — no need to run kiQuant first. You can also:
- Use `--weights base` or `--weights <model-id>` with the training script to reference registered models
- Use `--weights path/to/file.pth` for unregistered weight files
- After training completes, the best model is automatically registered with lineage tracking

To train from scratch instead of fine-tuning (not recommended without a large dataset):
```batch
scripts\train.bat --data-dir exported_data --no-pretrained --epochs 200
```

## Three Marker Classes

| Class | Shortcut | Color | Label Directory | Description |
|-------|----------|-------|-----------------|-------------|
| Positive tumor | P | Green | `labels_postm/` | Ki-67+ stained tumor nuclei |
| Negative tumor | N | Red | `labels_negtm/` | Unstained tumor nuclei |
| Non-tumor / Other | O | Blue | `labels_other/` | Stromal, immune, other cells |

## End-to-End Workflow

### 0. Prepare Images (Tile)

If your source images are large whole-slide crops, tile them first:

**From the GUI:** **File > Prepare Images (Tile)**
- Select source directory and output directory
- Choose tile size (512x512 recommended, 720x720 matches original KiNet)
- Blank/white background tiles are automatically skipped
- After tiling, optionally create a new project from the output

**From the command line:**
```batch
scripts\tile.bat --source-dir C:\slides --output-dir C:\tiles --tile-size 512
```

**Recommended tile sizes:**
- **512x512** — good balance of context and memory (default)
- **720x720** — matches original KiNet training patches
- **256x256** — minimal, for memory-constrained training

### 1. Create a Project

Launch the GUI with `scripts\run.bat`.

- **File > New Project** — select a directory of images (or tiled patches), then choose where to save the project file
- **File > Import kiQuant Project** — import an existing kiQuant project (positive and negative markers carry over; add "Other" annotations in the trainer)

### 2. Detect Nuclei

Use the existing model to get initial predictions, then correct them:

- **File > Detect Current Image** — detect on the current image only
- **File > Detect All Images** — detect on every image in the project

The detection dialog lets you:
- Choose which model to use (base model or any fine-tuned model)
- Adjust detection threshold (0.1-0.9, default 0.3)
- Set minimum distance between detections

After detection, images are marked with yellow "needs review" status.

### 3. Review & Correct

The sidebar shows review status for each image:
- **Gray dot** = not started (no annotations or detections)
- **Yellow dot** = needs review (has auto-detected markers)
- **Green dot** = reviewed (corrections complete, ready for export)

**Review workflow:**
1. Navigate to a yellow (needs review) image
2. Delete false positives using the eraser
3. Add missed nuclei by clicking to place markers
4. Add "Other" class annotations as needed
5. Press **R** to mark as reviewed and jump to the next unreviewed image
6. Press **Shift+R** to revert a reviewed image back to needs-review

**Filter buttons** in the sidebar header let you show: All | Needs Review | Reviewed

The review header shows progress: `12 reviewed / 45 images (27 need review)`

### 4. Export Training Data

**File > Export Training Data** opens the export dialog:

- **Validation Split** — fraction of images held out for validation (default 20%)
- **Proximity Map Sigma** — Gaussian sigma for label blobs (default 6.0)
- **Export only reviewed images** — default on; only exports green (reviewed) images

The dialog shows how many images are reviewed vs. pending.

```
output_dir/
├── images/
│   ├── train/          # Training images (PNG)
│   └── val/            # Validation images (PNG)
├── labels_postm/
│   ├── train/          # Positive tumor proximity maps
│   └── val/
├── labels_negtm/
│   ├── train/          # Negative tumor proximity maps
│   └── val/
└── labels_other/
    ├── train/          # Non-tumor proximity maps
    └── val/
```

Each label image is a grayscale PNG where pixel intensity encodes proximity to the nearest annotated centroid: `intensity = exp(-d² / 2σ²) × 255`.

### 5. Train the Model

Training runs from the command line (not the GUI) — it's better for long runs and more flexible.

**Fine-tune the base model (recommended):**
```batch
scripts\train.bat --data-dir path\to\exported --output-dir runs\exp01 --epochs 50
```
The base model is automatically downloaded if not already cached.

**Fine-tune from a registered model:**
```batch
scripts\train.bat --data-dir path\to\exported --weights ft-20260206-143022 --epochs 50
```

**Fine-tune from a specific weights file:**
```batch
scripts\train.bat --data-dir path\to\exported --weights path\to\model.pth --output-dir runs\exp01 --epochs 50
```

**Give the model a friendly name:**
```batch
scripts\train.bat --data-dir path\to\exported --model-name "Round 1 - Liver biopsies" --epochs 50
```

**Train from scratch (large dataset needed):**
```batch
scripts\train.bat --data-dir path\to\exported --no-pretrained --output-dir runs\exp01 --epochs 200
```

**Resume interrupted training:**
```batch
scripts\train.bat --data-dir path\to\exported --output-dir runs\exp01 --resume runs\exp01\latest_checkpoint.pth
```

After training completes, the best model is automatically:
- Copied to `~/.kiquant/models/` with a unique ID (e.g., `ft-20260206-143022.pth`)
- Registered in the model registry with parent lineage and training metadata
- Available for detection in the GUI

**All training options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | *(required)* | Exported training data directory |
| `--output-dir` | `runs/exp` | Where to save models and logs |
| `--weights` | Auto (base) | Model ID or file path to fine-tune from |
| `--model-name` | Auto | Human-friendly name for the trained model |
| `--no-pretrained` | Off | Train from scratch (skip auto-loading cached weights) |
| `--resume` | None | Checkpoint to resume training from |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4 | Batch size (reduce if out of memory) |
| `--crop-size` | 256 | Random crop size for training patches |
| `--lr` | 0.001 | Learning rate (Adam optimizer) |
| `--fg-weight` | 5.0 | Foreground loss weight (higher = penalize missed nuclei more) |
| `--num-workers` | 2 | DataLoader worker threads |
| `--seed` | 42 | Random seed for reproducibility |

**Training outputs:**

| File | Description |
|------|-------------|
| `best_model.pth` | Weights with lowest validation loss |
| `final_model.pth` | Weights after last epoch |
| `latest_checkpoint.pth` | Full checkpoint for resuming (model + optimizer + epoch) |
| `training_log.csv` | Per-epoch train/val loss, learning rate, timing |
| `config.json` | Training configuration used |

**Tips:**
- A CUDA GPU is strongly recommended. CPU training is possible but very slow.
- Start with a small number of epochs (5-10) to verify everything works before long runs.
- Watch `val_loss` in the log — if it stops decreasing, training has converged.
- Lower `--batch-size` if you get CUDA out-of-memory errors.
- The `--fg-weight` parameter helps with class imbalance (most pixels are background). Increase it if the model misses nuclei; decrease it if you get too many false positives.

### 6. Model Browser

**File > Model Browser** shows all registered models with:
- Model name and unique ID
- Creation date
- Lineage chain (e.g., `base -> ft-20260206 -> ft-20260210`)
- Training data summary (image counts)
- Metrics (validation loss, epochs trained)
- "Set as Default" button to select which model is used for detection

### 7. Evaluate the Model

**From the GUI:**
- **File > Evaluate Model** — select a `.pth` weights file
- The model runs inference on all annotated images and reports per-class precision, recall, and F1

**From the command line:**
```batch
scripts\evaluate.bat --model runs\exp01\best_model.pth --data-dir path\to\exported
```

**Evaluation options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *(required)* | Path to model weights |
| `--data-dir` | *(required)* | Data directory (KiNet format) |
| `--split` | `val` | Which split to evaluate (`train` or `val`) |
| `--threshold` | 0.3 | Peak detection threshold |
| `--min-distance` | 5 | Minimum distance between detected peaks (pixels) |
| `--max-match-distance` | 10 | Maximum distance for matching prediction to ground truth |
| `--output` | Auto | Output JSON file path |

**Metrics:**
- **Precision** — fraction of predictions that match a ground truth nucleus
- **Recall** — fraction of ground truth nuclei that were detected
- **F1** — harmonic mean of precision and recall
- Matching uses the Hungarian algorithm with a distance threshold (default 10px)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| P | Positive marker mode |
| N | Negative marker mode |
| O | Other marker mode |
| E | Eraser mode |
| H | Pan mode |
| R | Mark reviewed & advance to next unreviewed |
| Shift+R | Mark current as needs review |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z / Ctrl+Y | Redo |
| Ctrl+S | Save project |
| Left Arrow | Previous image |
| Right Arrow | Next image |
| F | Zoom to fit |
| 1 | Zoom to 100% |
| +/- | Zoom in/out |
| Scroll wheel | Zoom at cursor |
| Middle-click drag | Pan (any mode) |

## Testing Checklist

Use this sequence to verify the app is working correctly:

### GUI Basics
- [ ] Launch with `scripts\run.bat` — browser window opens with welcome screen
- [ ] **New Project** — select image directory, save project file — images load
- [ ] Sidebar shows image list with filenames and status dots
- [ ] Click images in sidebar to navigate

### Image Tiling
- [ ] **File > Prepare Images** — select large image dir — tiles created in output dir
- [ ] Blank tiles are skipped, `{name}_r0_c0.png` naming format
- [ ] After tiling, "Create project" prompt works

### Annotation
- [ ] **Positive mode (P)** — click places green markers, count updates
- [ ] **Negative mode (N)** — click places red markers, count updates
- [ ] **Other mode (O)** — click places blue markers, count updates
- [ ] **Eraser mode (E)** — click/drag deletes markers, cursor shows eraser circle
- [ ] **Pan mode (H)** — drag to pan, cursor shows grab hand
- [ ] Middle-click panning works in any mode
- [ ] Scroll wheel zoom works, zoom controls work (Fit, 1:1, +/-)
- [ ] Overview map shows marker positions and viewport rectangle

### Detection
- [ ] **File > Detect All Images** — base model downloads if needed — markers appear on all images
- [ ] All images show yellow "needs review" status dots
- [ ] Sidebar filter "Needs Review" shows only yellow images
- [ ] **File > Detect Current Image** — works on single image

### Review Workflow
- [ ] Press **R** — marks current as reviewed (green dot), advances to next unreviewed
- [ ] Press **Shift+R** — reverts to needs review (yellow dot)
- [ ] Review summary header updates counts
- [ ] Filter buttons (All / Needs Review / Reviewed) filter sidebar correctly
- [ ] Close and reopen project — review status persists

### Undo/Redo & Persistence
- [ ] **Ctrl+Z** undoes last marker add/delete
- [ ] **Ctrl+Y** redoes undone action
- [ ] Navigate to next image (Right Arrow) and back — markers persist
- [ ] **Ctrl+S** saves — close and reopen with **Load Project** — state restored
- [ ] Sidebar counts update after adding/removing markers

### Import
- [ ] **File > Import kiQuant Project** — select a kiQuant `.json` file
- [ ] Positive and negative markers appear correctly
- [ ] Other class count is 0 (ready for annotation)

### Export
- [ ] **File > Export Training Data** — modal shows review summary
- [ ] "Export only reviewed" checkbox works — only green images exported
- [ ] Select output directory — progress shows, export completes
- [ ] Verify output directory structure

### Training (requires exported data)
- [ ] `scripts\train.bat --data-dir <exported> --epochs 5` — starts training, base model auto-downloads
- [ ] Loss values print each epoch, loss decreases
- [ ] After training: model registered in `~/.kiquant/models/registry.json`
- [ ] Lineage chain printed: `base -> ft-XXXXXXXX-XXXXXX`
- [ ] Resume works: `scripts\train.bat --data-dir <exported> --output-dir <same> --resume <same>\latest_checkpoint.pth`

### Model Browser
- [ ] **File > Model Browser** — lists all models with lineage
- [ ] "Set as Default" works — detect uses the new default
- [ ] After training a second model, lineage shows `base -> ft-1 -> ft-2`

### Evaluation (requires trained model)
- [ ] `scripts\evaluate.bat --model <path>\best_model.pth --data-dir <exported>` — prints per-class P/R/F1
- [ ] Output `evaluation_results.json` is created
- [ ] From GUI: **File > Evaluate Model** — select weights — results display

## Project Structure

```
kinet-trainer/
├── src/
│   ├── main.py              # Eel backend: annotation, detection, export, evaluation
│   ├── state.py             # Data model: 3-class Marker, Field, State, ReviewStatus
│   ├── export.py            # Proximity map generation, KiNet format export
│   ├── evaluate.py          # Model evaluation with distance-based matching
│   ├── image_prep.py        # Image tiling for training preparation
│   ├── model_registry.py    # Model registry with download, lineage tracking
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py         # CLI training script with auto-registration
│   │   ├── dataset.py       # PyTorch Dataset for KiNet format
│   │   └── augment.py       # Joint image+label augmentations
│   └── web/
│       ├── index.html       # Annotation GUI layout
│       ├── app.js           # Canvas rendering, event handling, review workflow
│       └── style.css        # Dark theme (matches kiQuant)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── scripts/
    ├── dev-setup.bat        # Create venv, install dependencies
    ├── install-gpu.bat      # Replace CPU PyTorch with CUDA version
    ├── run.bat              # Launch the annotation GUI
    ├── train.bat            # Run training from CLI
    ├── evaluate.bat         # Run evaluation from CLI
    └── tile.bat             # Tile images from CLI
```

## Architecture Notes

- **Shared model**: Uses `Ki67Net` from `src/detection/kinet_model.py` (the main kiQuant repo). The training script adds it to `sys.path` automatically.
- **Model registry**: All trained models are tracked at `~/.kiquant/models/registry.json` with lineage, metrics, and training data references.
- **Self-contained weights**: Base model weights are downloaded independently by the trainer — no kiQuant dependency.
- **Same tech stack as kiQuant**: Python + Eel + HTML5 Canvas. Dark theme. Same zoom/pan/overlay rendering patterns.
- **Separate venv**: The trainer has its own `venv` inside `kinet-trainer/src/` (created by `dev-setup.bat`), independent from kiQuant's venv.
- **CLI training**: Training runs from the command line, not the GUI. This is intentional — long training runs are better managed in a terminal where you can see live output, pipe to a log file, or run in `screen`/`tmux`.
