"""
kiQuant - Cell marker quantification tool.
Main entry point with Eel backend.
"""

import eel
import os
import sys
import base64
import json
from io import BytesIO
from tkinter import Tk, filedialog, messagebox
from PIL import Image

from marker_state import State, Field, Marker, MarkerClass, Mode

__version__ = "0.1.0"

# Initialize Eel with the web folder
eel.init('web')

# Global state
state: State = State()

# Undo/Redo history - stores snapshots of markers per field
# Structure: { field_index: { 'undo': [snapshots], 'redo': [snapshots] } }
history = {}
MAX_HISTORY = 50


def _get_field_history(field_index):
    """Get or create history for a field."""
    if field_index not in history:
        history[field_index] = {'undo': [], 'redo': []}
    return history[field_index]


def _save_to_history():
    """Save current markers state to undo history."""
    field = state.get_current_field()
    if not field:
        return

    field_index = state.current_index
    h = _get_field_history(field_index)

    # Save snapshot of current markers
    snapshot = [m.to_dict() for m in field.markers]
    h['undo'].append(snapshot)

    # Limit history size
    if len(h['undo']) > MAX_HISTORY:
        h['undo'].pop(0)

    # Clear redo stack on new action
    h['redo'].clear()


def _restore_markers(snapshot):
    """Restore markers from a snapshot."""
    field = state.get_current_field()
    if not field:
        return

    field.markers = [Marker.from_dict(m) for m in snapshot]


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# ============== File Dialog Helpers ==============

def _hide_tk_root():
    """Create and hide a tkinter root window for file dialogs."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    return root


@eel.expose
def select_directory():
    """Open directory selection dialog."""
    root = _hide_tk_root()
    directory = filedialog.askdirectory(title='Select Image Directory')
    root.destroy()
    return directory if directory else None


@eel.expose
def select_save_file(title='Save File', default_name='', file_types=None):
    """Open save file dialog."""
    root = _hide_tk_root()
    if file_types is None:
        file_types = [('JSON files', '*.json'), ('All files', '*.*')]
    filepath = filedialog.asksaveasfilename(
        title=title,
        initialfile=default_name,
        filetypes=file_types,
        defaultextension=file_types[0][1].replace('*', '')
    )
    root.destroy()
    return filepath if filepath else None


@eel.expose
def select_open_file(title='Open File', file_types=None):
    """Open file selection dialog."""
    root = _hide_tk_root()
    if file_types is None:
        file_types = [('JSON files', '*.json'), ('All files', '*.*')]
    filepath = filedialog.askopenfilename(
        title=title,
        filetypes=file_types
    )
    root.destroy()
    return filepath if filepath else None


# ============== Project Management ==============

@eel.expose
def new_project():
    """Create a new project. Returns success status and message."""
    global state

    # Select image directory
    directory = select_directory()
    if not directory:
        return {'success': False, 'message': 'No directory selected'}

    # Select project save location
    project_path = select_save_file(
        title='Save New Project',
        default_name='project.json',
        file_types=[('JSON files', '*.json')]
    )
    if not project_path:
        return {'success': False, 'message': 'No project file selected'}

    # Select results output file
    output_path = select_save_file(
        title='Select Results Output File',
        default_name='results.csv',
        file_types=[('CSV files', '*.csv'), ('Excel-compatible', '*.xls')]
    )
    if not output_path:
        return {'success': False, 'message': 'No output file selected'}

    # Initialize state
    state = State()
    state.project_path = project_path
    state.output_path = output_path
    count = state.load_image_directory(directory)

    if count == 0:
        return {'success': False, 'message': 'No images found in directory'}

    # Save initial project
    state.save_to_file()

    return {
        'success': True,
        'message': f'Project created with {count} images',
        'image_count': count
    }


@eel.expose
def load_project():
    """Load an existing project. Returns success status and message."""
    global state

    project_path = select_open_file(
        title='Open Project File',
        file_types=[('JSON files', '*.json')]
    )
    if not project_path:
        return {'success': False, 'message': 'No project file selected'}

    try:
        state = State.load_from_file(project_path)

        # Verify image directory still exists
        if not os.path.isdir(state.image_dir):
            return {
                'success': False,
                'message': f'Image directory not found: {state.image_dir}'
            }

        return {
            'success': True,
            'message': f'Project loaded with {state.get_total()} images',
            'image_count': state.get_total()
        }
    except Exception as e:
        return {'success': False, 'message': f'Failed to load project: {str(e)}'}


@eel.expose
def save_project():
    """Save current project state."""
    global state
    if not state.project_path:
        return {'success': False, 'message': 'No project file set'}

    try:
        state.save_to_file()
        return {'success': True, 'message': 'Project saved'}
    except Exception as e:
        return {'success': False, 'message': f'Failed to save: {str(e)}'}


@eel.expose
def export_csv():
    """Export results to CSV file."""
    global state

    if not state.fields:
        return {'success': False, 'message': 'No images loaded'}

    try:
        filepath = state.export_csv()
        summary = state.get_summary()
        return {
            'success': True,
            'message': f'Results exported to {filepath}',
            'summary': summary
        }
    except Exception as e:
        return {'success': False, 'message': f'Export failed: {str(e)}'}


# ============== Image Navigation ==============

@eel.expose
def get_image_list():
    """Get list of all image filenames."""
    return [os.path.basename(f.filepath) for f in state.fields]


@eel.expose
def get_current_index():
    """Get current image index."""
    return state.current_index


@eel.expose
def get_total_images():
    """Get total number of images."""
    return state.get_total()


@eel.expose
def get_image_data(index=None):
    """Get image data as base64 and its markers."""
    global state

    if index is not None:
        state.go_to_field(index)

    field = state.get_current_field()
    if not field:
        return None

    try:
        # Load and convert image to base64
        img = Image.open(field.filepath)

        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # Encode as JPEG for smaller size
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'index': state.current_index,
            'total': state.get_total(),
            'filename': os.path.basename(field.filepath),
            'image': f'data:image/jpeg;base64,{img_base64}',
            'width': img.width,
            'height': img.height,
            'markers': [m.to_dict() for m in field.markers],
            'positive_count': field.get_positive_count(),
            'negative_count': field.get_negative_count()
        }
    except Exception as e:
        return {'error': str(e)}


@eel.expose
def next_image():
    """Navigate to next image."""
    state.next_field()
    save_project()  # Auto-save on navigation
    return get_image_data()


@eel.expose
def previous_image():
    """Navigate to previous image."""
    state.previous_field()
    save_project()  # Auto-save on navigation
    return get_image_data()


@eel.expose
def go_to_image(index):
    """Navigate to specific image index."""
    state.go_to_field(index)
    save_project()
    return get_image_data()


# ============== Marker Operations ==============

@eel.expose
def get_mode():
    """Get current marking mode."""
    return state.mode


@eel.expose
def set_mode(mode):
    """Set marking mode (0=positive, 1=negative, 2=select, 3=eraser)."""
    state.mode = mode
    return mode


@eel.expose
def add_marker(x, y, marker_class=None):
    """Add a marker at position (x, y)."""
    field = state.get_current_field()
    if not field:
        return None

    # Use current mode if marker_class not specified
    if marker_class is None:
        if state.mode == Mode.POSITIVE:
            marker_class = MarkerClass.POSITIVE
        elif state.mode == Mode.NEGATIVE:
            marker_class = MarkerClass.NEGATIVE
        else:
            return None  # Can't add markers in select/eraser mode

    # Save state for undo
    _save_to_history()

    # Deselect all existing markers
    field.deselect_all()

    # Create and add new marker
    marker = Marker(marker_class=marker_class, x=int(x), y=int(y), selected=True)
    field.add_marker(marker)

    return {
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'can_undo': len(_get_field_history(state.current_index)['undo']) > 0,
        'can_redo': len(_get_field_history(state.current_index)['redo']) > 0
    }


@eel.expose
def delete_marker_at(x, y):
    """Delete marker at position (x, y). Used in eraser mode."""
    field = state.get_current_field()
    if not field:
        return None

    for i, marker in enumerate(field.markers):
        if marker.contains(int(x), int(y)):
            # Save state for undo before deletion
            _save_to_history()
            field.remove_marker(i)
            break

    return {
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'can_undo': len(_get_field_history(state.current_index)['undo']) > 0,
        'can_redo': len(_get_field_history(state.current_index)['redo']) > 0
    }


@eel.expose
def delete_selected_markers():
    """Delete all selected markers."""
    field = state.get_current_field()
    if not field:
        return None

    # Only save to history if there are selected markers
    selected = field.get_selected_markers()
    if selected:
        _save_to_history()

    removed = field.remove_selected_markers()

    return {
        'removed': removed,
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'can_undo': len(_get_field_history(state.current_index)['undo']) > 0,
        'can_redo': len(_get_field_history(state.current_index)['redo']) > 0
    }


@eel.expose
def select_marker_at(x, y):
    """Select marker at position (x, y)."""
    field = state.get_current_field()
    if not field:
        return None

    field.deselect_all()
    field.select_marker_at(int(x), int(y))

    return {'markers': [m.to_dict() for m in field.markers]}


@eel.expose
def select_markers_in_rect(x, y, width, height):
    """Select all markers within rectangle."""
    field = state.get_current_field()
    if not field:
        return None

    field.deselect_all()
    count = field.select_markers_in_rect(int(x), int(y), int(width), int(height))

    return {
        'selected_count': count,
        'markers': [m.to_dict() for m in field.markers]
    }


@eel.expose
def deselect_all():
    """Deselect all markers."""
    field = state.get_current_field()
    if not field:
        return None

    field.deselect_all()
    return {'markers': [m.to_dict() for m in field.markers]}


@eel.expose
def get_markers():
    """Get all markers for current field."""
    field = state.get_current_field()
    if not field:
        return []

    return [m.to_dict() for m in field.markers]


@eel.expose
def get_summary():
    """Get overall project statistics."""
    return state.get_summary()


# ============== Undo/Redo ==============

@eel.expose
def undo():
    """Undo the last action on current field."""
    field = state.get_current_field()
    if not field:
        return None

    h = _get_field_history(state.current_index)

    if not h['undo']:
        return None

    # Save current state to redo stack
    current_snapshot = [m.to_dict() for m in field.markers]
    h['redo'].append(current_snapshot)

    # Restore previous state
    previous_snapshot = h['undo'].pop()
    _restore_markers(previous_snapshot)

    return {
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'can_undo': len(h['undo']) > 0,
        'can_redo': len(h['redo']) > 0
    }


@eel.expose
def redo():
    """Redo the last undone action on current field."""
    field = state.get_current_field()
    if not field:
        return None

    h = _get_field_history(state.current_index)

    if not h['redo']:
        return None

    # Save current state to undo stack
    current_snapshot = [m.to_dict() for m in field.markers]
    h['undo'].append(current_snapshot)

    # Restore redo state
    redo_snapshot = h['redo'].pop()
    _restore_markers(redo_snapshot)

    return {
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'can_undo': len(h['undo']) > 0,
        'can_redo': len(h['redo']) > 0
    }


@eel.expose
def get_undo_redo_state():
    """Get current undo/redo availability."""
    h = _get_field_history(state.current_index)
    return {
        'can_undo': len(h['undo']) > 0,
        'can_redo': len(h['redo']) > 0
    }


# ============== Selection & Conversion ==============

@eel.expose
def select_markers_by_indices(indices):
    """Select markers by their indices."""
    field = state.get_current_field()
    if not field:
        return None

    field.deselect_all()
    for idx in indices:
        if 0 <= idx < len(field.markers):
            field.markers[idx].selected = True

    return {'markers': [m.to_dict() for m in field.markers]}


@eel.expose
def select_all_markers():
    """Select all markers on current field."""
    field = state.get_current_field()
    if not field:
        return None

    for m in field.markers:
        m.selected = True

    return {'markers': [m.to_dict() for m in field.markers]}


@eel.expose
def convert_selected_markers(new_class):
    """Convert selected markers to a new class (0=positive, 1=negative)."""
    field = state.get_current_field()
    if not field:
        return None

    selected = field.get_selected_markers()
    if not selected:
        return None

    # Save state for undo
    _save_to_history()

    for m in selected:
        m.marker_class = new_class

    return {
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'can_undo': len(_get_field_history(state.current_index)['undo']) > 0,
        'can_redo': len(_get_field_history(state.current_index)['redo']) > 0
    }


# ============== Quick Mode ==============

@eel.expose
def quick_mode():
    """Start quick mode - select images directly without full project setup."""
    global state

    root = _hide_tk_root()
    filepaths = filedialog.askopenfilenames(
        title='Select Images for Quick Mode',
        filetypes=[
            ('Image files', '*.tif *.tiff *.jpg *.jpeg *.png *.bmp'),
            ('All files', '*.*')
        ]
    )
    root.destroy()

    if not filepaths:
        return {'success': False, 'message': 'No images selected'}

    # Initialize state without project file
    state = State()
    state.project_path = ''  # No project file in quick mode

    # Add selected files as fields
    for filepath in sorted(filepaths):
        state.fields.append(Field(filepath=filepath))

    state.current_index = 0

    # Set default output path (same directory as first image)
    first_dir = os.path.dirname(filepaths[0])
    state.output_path = os.path.join(first_dir, 'quick_results.csv')

    return {
        'success': True,
        'message': f'Quick mode started with {len(filepaths)} images',
        'image_count': len(filepaths),
        'is_quick_mode': True
    }


@eel.expose
def is_quick_mode():
    """Check if currently in quick mode."""
    return state.project_path == '' and len(state.fields) > 0


# ============== About Dialog ==============

@eel.expose
def get_version():
    """Get application version."""
    return __version__


@eel.expose
def show_about():
    """Show about dialog."""
    root = _hide_tk_root()
    messagebox.showinfo(
        "About kiQuant",
        f"kiQuant v{__version__}\n\n"
        "Cell marker quantification tool for\n"
        "immunohistochemistry analysis.\n\n"
        "https://github.com/your-username/kiquant"
    )
    root.destroy()


# ============== Main Entry Point ==============

def main():
    """Start the application."""
    # Start Eel with Chrome/Edge in app mode
    try:
        eel.start(
            'index.html',
            size=(1200, 800),
            port=0,  # Auto-select port
            mode='edge'  # Try Edge first (common on Windows)
        )
    except EnvironmentError:
        # Fall back to default browser if Edge not available
        try:
            eel.start(
                'index.html',
                size=(1200, 800),
                port=0,
                mode='chrome'
            )
        except EnvironmentError:
            # Last resort: default browser
            eel.start(
                'index.html',
                size=(1200, 800),
                port=0,
                mode=None
            )


if __name__ == '__main__':
    main()
