# kiQuant

Cell marker quantification tool for immunohistochemistry analysis.

kiQuant provides a simple interface for counting positive and negative cell markers on microscopy images, with automatic calculation of proliferation indices. It supports both manual annotation and AI-assisted nucleus detection.

## Features

- **Marker Annotation**: Click to place positive (green) or negative (red) markers
- **AI Detection** (optional): Automatic nucleus detection with CellPose or StarDist
- **DAB Classification**: Auto-classify detected nuclei by DAB staining intensity
- **Multiple Selection Modes**: Rectangle and lasso selection for bulk operations
- **Undo/Redo**: Full history support with Ctrl+Z / Ctrl+Y
- **Quick Mode**: Load images directly without project setup
- **WSI-style Viewer**: Overview map, zoom/pan controls
- **Field Guide**: Visual bounding box showing marker coverage
- **CSV Export**: Tab-delimited results compatible with Excel

## Installation

### From Release (Recommended for Manual Counting)

Download `kiQuant.exe` from the [Releases](https://github.com/cornish/kiquant/releases) page. No installation required - just run the executable.

> **Note**: The standalone exe does not include AI detection. For AI features, install from source (see below).

### From Source (Required for AI Detection)

Requires Python 3.11+

```bash
# Clone the repository
git clone https://github.com/cornish/kiquant.git
cd kiquant

# Run setup script
scripts\dev-setup.bat

# Run the application
scripts\run.bat
```

## AI Detection Setup (Optional)

The AI detection feature requires additional dependencies that are not included in the base installation due to their size (~2-3GB total).

### Quick Install

Run the helper script and follow the prompts:

```bash
scripts\install-ai.bat
```

### Manual Installation

#### Option 1: CellPose (Recommended)

CellPose uses PyTorch and works well on both CPU and GPU.

```bash
# Activate the virtual environment first
cd src
venv\Scripts\activate

# Install CellPose
pip install cellpose>=3.0.0
```

#### Option 2: StarDist

StarDist uses TensorFlow and is particularly good for fluorescence images.

```bash
# Activate the virtual environment first
cd src
venv\Scripts\activate

# Install StarDist with TensorFlow
pip install stardist>=0.8.0 tensorflow>=2.10.0
```

#### Option 3: Both

You can install both models to compare results:

```bash
pip install cellpose>=3.0.0 stardist>=0.8.0 tensorflow>=2.10.0
```

### GPU Acceleration (Optional)

For faster detection on NVIDIA GPUs, use the GPU install script:

```bash
scripts\install-ai-gpu.bat
```

This installs CUDA-enabled versions of PyTorch and/or TensorFlow. Requires:
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed

### Verifying Installation

After installation, restart kiQuant. If AI models are available, the detection controls will be enabled in the toolbar.

## Using AI Detection

1. Load a project or use Quick Mode to open images
2. Select a detection model from the dropdown (CellPose or StarDist)
3. Choose a classification mode:
   - **Auto (DAB)**: Automatically classify nuclei based on DAB staining intensity
   - **All Positive**: Mark all detected nuclei as positive
   - **All Negative**: Mark all detected nuclei as negative
4. Adjust the DAB threshold slider (for Auto mode) - higher values = stricter positive classification
5. Click **Detect** or press **D**
6. Review and correct the results using the standard marker tools
7. Use **Ctrl+Z** to undo detection if needed

## Usage

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| P | Positive marker mode |
| N | Negative marker mode |
| S | Select mode |
| E | Eraser mode |
| H | Pan mode |
| D | Run AI detection |
| G | Toggle field guide |
| F | Fit to window |
| 1 | Zoom to 100% |
| +/- | Zoom in/out |
| Delete | Delete selected markers |
| Left/Right | Navigate images |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+A | Select all |
| Escape | Deselect all |

### Mouse Controls

- **Left-click**: Place marker (in marker mode) or select (in select mode)
- **Right-click**: Context menu
- **Middle-click**: Pan
- **Scroll wheel**: Zoom toward cursor
- **Drag on overview**: Navigate large images

## Building

To create a standalone executable (without AI features):

```bash
scripts\build.bat
```

The executable will be created at `src\dist\kiQuant.exe`.

## Troubleshooting

### AI Detection Not Available

If the detection controls show "No AI models":
1. Ensure you installed from source (not the standalone exe)
2. Verify CellPose or StarDist is installed: `pip list | findstr cellpose`
3. Check for import errors: `python -c "import cellpose; print('OK')"`

### Detection is Slow

- First detection may take 30-60 seconds while the model loads
- Subsequent detections on the same session are faster
- Consider installing GPU support for large images

### Out of Memory Errors

- Close other applications
- Use CellPose instead of StarDist (lower memory usage)
- Process smaller images or downscale before detection

## License

MIT License - see [LICENSE](LICENSE) for details.
