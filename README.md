# kiQuant

Cell marker quantification tool for immunohistochemistry analysis.

kiQuant provides a simple interface for manually counting positive and negative cell markers on microscopy images, with automatic calculation of proliferation indices.

## Features

- **Marker Annotation**: Click to place positive (green) or negative (red) markers
- **Multiple Selection Modes**: Rectangle and lasso selection for bulk operations
- **Undo/Redo**: Full history support with Ctrl+Z / Ctrl+Y
- **Quick Mode**: Load images directly without project setup
- **WSI-style Viewer**: Overview map, zoom/pan controls
- **Field Guide**: Visual bounding box showing marker coverage
- **CSV Export**: Tab-delimited results compatible with Excel

## Installation

### From Release (Recommended)

Download `kiQuant.exe` from the [Releases](https://github.com/cornish/kiquant/releases) page. No installation required - just run the executable.

### From Source

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

## Usage

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| P | Positive marker mode |
| N | Negative marker mode |
| S | Select mode |
| E | Eraser mode |
| H | Pan mode |
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

To create a standalone executable:

```bash
scripts\build.bat
```

The executable will be created at `src\dist\kiQuant.exe`.

## License

MIT License - see [LICENSE](LICENSE) for details.
