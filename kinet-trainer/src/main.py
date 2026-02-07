"""
KiNet Trainer - Annotation and training management GUI.
Eel backend for 3-class annotation of Ki-67 IHC images.
"""

import eel
import os
import sys
import base64
import json
from io import BytesIO
from tkinter import Tk, filedialog
from PIL import Image

# Add repo root to path for kinet package
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from state import State, Field, Marker, MarkerClass, Mode, ReviewStatus
from export import export_training_data
from image_prep import tile_images
import model_registry

__version__ = "0.1.0"

# Initialize Eel with the web folder
eel.init('web')

# Global state
state: State = State()

# Undo/Redo history per field
history = {}
MAX_HISTORY = 50


def _get_field_history(field_index):
    if field_index not in history:
        history[field_index] = {'undo': [], 'redo': []}
    return history[field_index]


def _save_to_history():
    field = state.get_current_field()
    if not field:
        return
    h = _get_field_history(state.current_index)
    snapshot = [m.to_dict() for m in field.markers]
    h['undo'].append(snapshot)
    if len(h['undo']) > MAX_HISTORY:
        h['undo'].pop(0)
    h['redo'].clear()


def _restore_markers(snapshot):
    field = state.get_current_field()
    if not field:
        return
    field.markers = [Marker.from_dict(m) for m in snapshot]


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# ============== File Dialog Helpers ==============

def _hide_tk_root():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    return root


@eel.expose
def select_directory(title='Select Directory'):
    root = _hide_tk_root()
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory if directory else None


@eel.expose
def select_save_file(title='Save File', default_name='', file_types=None):
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
    """Create a new trainer project from an image directory."""
    global state

    directory = select_directory(title='Select Image Directory')
    if not directory:
        return {'success': False, 'message': 'No directory selected'}

    project_path = select_save_file(
        title='Save Trainer Project',
        default_name='trainer_project.json',
        file_types=[('JSON files', '*.json')]
    )
    if not project_path:
        return {'success': False, 'message': 'No project file selected'}

    state = State()
    state.project_path = project_path
    count = state.load_image_directory(directory)

    if count == 0:
        return {'success': False, 'message': 'No images found in directory'}

    state.save_to_file()

    return {
        'success': True,
        'message': f'Project created with {count} images',
        'image_count': count
    }


@eel.expose
def new_project_from_dir(directory):
    """Create a new trainer project from a specific directory (used after tiling)."""
    global state

    if not directory or not os.path.isdir(directory):
        return {'success': False, 'message': 'Invalid directory'}

    project_path = select_save_file(
        title='Save Trainer Project',
        default_name='trainer_project.json',
        file_types=[('JSON files', '*.json')]
    )
    if not project_path:
        return {'success': False, 'message': 'No project file selected'}

    state = State()
    state.project_path = project_path
    count = state.load_image_directory(directory)

    if count == 0:
        return {'success': False, 'message': 'No images found in directory'}

    state.save_to_file()

    return {
        'success': True,
        'message': f'Project created with {count} images',
        'image_count': count
    }


@eel.expose
def load_project():
    """Load an existing trainer project."""
    global state

    project_path = select_open_file(
        title='Open Trainer Project',
        file_types=[('JSON files', '*.json')]
    )
    if not project_path:
        return {'success': False, 'message': 'No project file selected'}

    try:
        state = State.load_from_file(project_path)

        if not os.path.isdir(state.image_dir):
            return {
                'success': False,
                'message': f'Image directory not found: {state.image_dir}'
            }

        # Check for missing and new images in the folder
        missing_images = state.find_missing_images()
        new_images = state.find_new_images()

        return {
            'success': True,
            'message': f'Project loaded with {state.get_total()} images',
            'image_count': state.get_total(),
            'missing_images': [os.path.basename(f) for f in missing_images],
            'missing_image_paths': missing_images,
            'new_images': [os.path.basename(f) for f in new_images],
            'new_image_paths': new_images
        }
    except Exception as e:
        return {'success': False, 'message': f'Failed to load project: {str(e)}'}


@eel.expose
def import_kiquant_project():
    """Import annotations from a kiQuant project."""
    global state

    kiquant_path = select_open_file(
        title='Import kiQuant Project',
        file_types=[('JSON files', '*.json')]
    )
    if not kiquant_path:
        return {'success': False, 'message': 'No file selected'}

    project_path = select_save_file(
        title='Save Trainer Project As',
        default_name='trainer_project.json',
        file_types=[('JSON files', '*.json')]
    )
    if not project_path:
        return {'success': False, 'message': 'No save location selected'}

    try:
        state = State.import_from_kiquant(kiquant_path)
        state.project_path = project_path

        if not os.path.isdir(state.image_dir):
            return {
                'success': False,
                'message': f'Image directory not found: {state.image_dir}'
            }

        state.save_to_file()

        summary = state.get_summary()
        return {
            'success': True,
            'message': (
                f'Imported {summary["total_images"]} images with '
                f'{summary["positive"]} positive, {summary["negative"]} negative markers. '
                f'Add "Other" class annotations in the trainer.'
            ),
            'image_count': state.get_total()
        }
    except Exception as e:
        return {'success': False, 'message': f'Import failed: {str(e)}'}


@eel.expose
def check_for_new_images():
    """Check if there are new images in the project folder."""
    if not state.image_dir:
        return {'success': False, 'message': 'No project loaded'}

    new_images = state.find_new_images()

    return {
        'success': True,
        'new_images': [os.path.basename(f) for f in new_images],
        'new_image_paths': new_images,
        'count': len(new_images)
    }


@eel.expose
def add_new_images(filepaths):
    """Add new images to the project."""
    if not state.project_path:
        return {'success': False, 'message': 'No project loaded'}

    added = state.add_images(filepaths)

    if added > 0:
        state.save_to_file()

    return {
        'success': True,
        'added': added,
        'total': state.get_total(),
        'message': f'Added {added} new images'
    }


@eel.expose
def remove_missing_images():
    """Remove images from project that no longer exist on disk."""
    if not state.project_path:
        return {'success': False, 'message': 'No project loaded'}

    removed = state.remove_missing_images()

    if removed > 0:
        state.save_to_file()

    return {
        'success': True,
        'removed': removed,
        'total': state.get_total(),
        'message': f'Removed {removed} missing images'
    }


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


# ============== Image Navigation ==============

@eel.expose
def get_image_list():
    """Get list of all image filenames with annotation counts."""
    result = []
    for f in state.fields:
        result.append({
            'filename': os.path.basename(f.filepath),
            'total': f.get_total_count(),
            'positive': f.get_positive_count(),
            'negative': f.get_negative_count(),
            'other': f.get_other_count(),
            'review_status': f.review_status
        })
    return result


@eel.expose
def get_current_index():
    return state.current_index


@eel.expose
def get_total_images():
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
        img = Image.open(field.filepath)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # Store dimensions
        field.image_width = img.width
        field.image_height = img.height

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
            'negative_count': field.get_negative_count(),
            'other_count': field.get_other_count(),
            'review_status': field.review_status
        }
    except Exception as e:
        return {'error': str(e)}


@eel.expose
def next_image():
    state.next_field()
    save_project()
    return get_image_data()


@eel.expose
def previous_image():
    state.previous_field()
    save_project()
    return get_image_data()


@eel.expose
def go_to_image(index):
    state.go_to_field(index)
    save_project()
    return get_image_data()


# ============== Marker Operations ==============

@eel.expose
def get_mode():
    return state.mode


@eel.expose
def set_mode(mode):
    state.mode = mode
    return mode


def _marker_response():
    """Helper to build standard marker response."""
    field = state.get_current_field()
    if not field:
        return None
    h = _get_field_history(state.current_index)
    return {
        'markers': [m.to_dict() for m in field.markers],
        'positive_count': field.get_positive_count(),
        'negative_count': field.get_negative_count(),
        'other_count': field.get_other_count(),
        'can_undo': len(h['undo']) > 0,
        'can_redo': len(h['redo']) > 0
    }


@eel.expose
def add_marker(x, y, marker_class=None):
    """Add a marker at position (x, y)."""
    field = state.get_current_field()
    if not field:
        return None

    if marker_class is None:
        if state.mode == Mode.POSITIVE:
            marker_class = MarkerClass.POSITIVE
        elif state.mode == Mode.NEGATIVE:
            marker_class = MarkerClass.NEGATIVE
        elif state.mode == Mode.OTHER:
            marker_class = MarkerClass.OTHER
        else:
            return None

    _save_to_history()
    field.deselect_all()

    marker = Marker(marker_class=int(marker_class), x=int(x), y=int(y), selected=True)
    field.add_marker(marker)

    return _marker_response()


@eel.expose
def delete_marker_at(x, y):
    """Delete marker at position (x, y). Used in eraser mode."""
    field = state.get_current_field()
    if not field:
        return None

    for i, marker in enumerate(field.markers):
        if marker.contains(int(x), int(y)):
            _save_to_history()
            field.remove_marker(i)
            break

    return _marker_response()


@eel.expose
def delete_markers_in_radius(x, y, radius, save_history=True):
    """Delete all markers within radius of point (x, y). Used for brush eraser."""
    field = state.get_current_field()
    if not field:
        return None

    x, y, radius = int(x), int(y), int(radius)
    radius_sq = radius * radius

    to_delete = []
    for i, marker in enumerate(field.markers):
        dx = marker.x - x
        dy = marker.y - y
        if dx * dx + dy * dy <= radius_sq:
            to_delete.append(i)

    if to_delete and save_history:
        _save_to_history()

    for i in reversed(to_delete):
        field.remove_marker(i)

    resp = _marker_response()
    if resp:
        resp['deleted_count'] = len(to_delete)
    return resp


@eel.expose
def change_markers_in_radius(x, y, radius, new_class, save_history=True):
    """Change class of all markers within radius of point (x, y). Used for brush tool."""
    field = state.get_current_field()
    if not field:
        return None

    x, y, radius = int(x), int(y), int(radius)
    new_class = int(new_class)
    radius_sq = radius * radius

    changed_count = 0
    markers_to_change = []
    for marker in field.markers:
        dx = marker.x - x
        dy = marker.y - y
        if dx * dx + dy * dy <= radius_sq:
            if marker.marker_class != new_class:
                markers_to_change.append(marker)

    if markers_to_change and save_history:
        _save_to_history()

    for marker in markers_to_change:
        marker.marker_class = new_class
        changed_count += 1

    resp = _marker_response()
    if resp:
        resp['changed_count'] = changed_count
    return resp


# ============== Undo/Redo ==============

@eel.expose
def undo():
    field = state.get_current_field()
    if not field:
        return None

    h = _get_field_history(state.current_index)
    if not h['undo']:
        return None

    current_snapshot = [m.to_dict() for m in field.markers]
    h['redo'].append(current_snapshot)

    previous_snapshot = h['undo'].pop()
    _restore_markers(previous_snapshot)

    return _marker_response()


@eel.expose
def redo():
    field = state.get_current_field()
    if not field:
        return None

    h = _get_field_history(state.current_index)
    if not h['redo']:
        return None

    current_snapshot = [m.to_dict() for m in field.markers]
    h['undo'].append(current_snapshot)

    redo_snapshot = h['redo'].pop()
    _restore_markers(redo_snapshot)

    return _marker_response()


@eel.expose
def get_undo_redo_state():
    h = _get_field_history(state.current_index)
    return {
        'can_undo': len(h['undo']) > 0,
        'can_redo': len(h['redo']) > 0
    }


# ============== Selection Operations ==============

@eel.expose
def select_all():
    """Select all markers on current field."""
    field = state.get_current_field()
    if not field:
        return None
    for m in field.markers:
        m.selected = True
    return _marker_response()


@eel.expose
def deselect_all():
    """Deselect all markers on current field."""
    field = state.get_current_field()
    if not field:
        return None
    for m in field.markers:
        m.selected = False
    return _marker_response()


@eel.expose
def invert_selection():
    """Invert selection on current field."""
    field = state.get_current_field()
    if not field:
        return None
    for m in field.markers:
        m.selected = not m.selected
    return _marker_response()


@eel.expose
def select_markers_in_rect(x, y, width, height, additive=False):
    """Select all markers within rectangle. If not additive, deselect others first."""
    field = state.get_current_field()
    if not field:
        return None
    if not additive:
        for m in field.markers:
            m.selected = False
    for m in field.markers:
        if x <= m.x <= x + width and y <= m.y <= y + height:
            m.selected = True
    return _marker_response()


def _point_in_polygon(x, y, polygon):
    """Check if point (x, y) is inside polygon using ray casting."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]['x'], polygon[i]['y']
        xj, yj = polygon[j]['x'], polygon[j]['y']
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@eel.expose
def select_markers_in_polygon(points, additive=False):
    """Select all markers within polygon (lasso selection). If not additive, deselect others first."""
    field = state.get_current_field()
    if not field:
        return None
    if len(points) < 3:
        return _marker_response()
    if not additive:
        for m in field.markers:
            m.selected = False
    for m in field.markers:
        if _point_in_polygon(m.x, m.y, points):
            m.selected = True
    return _marker_response()


@eel.expose
def select_marker_at_index(index, additive=False):
    """Select/toggle marker at given index. If not additive, deselect others first."""
    field = state.get_current_field()
    if not field:
        return None
    if index < 0 or index >= len(field.markers):
        return _marker_response()
    if not additive:
        for m in field.markers:
            m.selected = False
        field.markers[index].selected = True
    else:
        # Toggle selection when additive
        field.markers[index].selected = not field.markers[index].selected
    return _marker_response()


@eel.expose
def change_selected_class(new_class):
    """Change class of all selected markers."""
    field = state.get_current_field()
    if not field:
        return None

    selected = [m for m in field.markers if m.selected]
    if not selected:
        return _marker_response()

    _save_to_history()
    for m in selected:
        m.marker_class = new_class
        m.selected = False

    return _marker_response()


@eel.expose
def delete_selected():
    """Delete all selected markers."""
    field = state.get_current_field()
    if not field:
        return None

    selected = [m for m in field.markers if m.selected]
    if not selected:
        return _marker_response()

    _save_to_history()
    field.markers = [m for m in field.markers if not m.selected]

    return _marker_response()


# ============== Summary ==============

@eel.expose
def get_summary():
    return state.get_summary()


# ============== Export ==============

@eel.expose
def export_data(val_split=0.2, sigma=6.0, reviewed_only=False):
    """Export training data to KiNet format."""
    if not state.fields:
        return {'success': False, 'message': 'No images loaded'}

    output_dir = select_directory(title='Select Export Output Directory')
    if not output_dir:
        return {'success': False, 'message': 'No output directory selected'}

    try:
        def progress_callback(message, progress):
            eel.onExportProgress(message, progress)()

        result = export_training_data(
            state.fields, output_dir,
            val_split=val_split, sigma=sigma,
            progress_callback=progress_callback,
            reviewed_only=reviewed_only
        )
        return result
    except Exception as e:
        return {'success': False, 'message': f'Export failed: {str(e)}'}


# ============== Image Tiling ==============

@eel.expose
def tile_source_images(tile_size=512, skip_blank=True, blank_threshold=0.9):
    """Tile large images into training-ready patches."""
    source_dir = select_directory(title='Select Source Image Directory')
    if not source_dir:
        return {'success': False, 'message': 'No source directory selected'}

    output_dir = select_directory(title='Select Output Directory for Tiles')
    if not output_dir:
        return {'success': False, 'message': 'No output directory selected'}

    try:
        def progress_callback(message, progress):
            eel.onTileProgress(message, progress)()

        result = tile_images(
            source_dir=source_dir,
            output_dir=output_dir,
            tile_size=tile_size,
            skip_blank=skip_blank,
            blank_threshold=blank_threshold,
            progress_callback=progress_callback
        )
        if result.get('success'):
            result['output_dir'] = output_dir
        return result
    except Exception as e:
        return {'success': False, 'message': f'Tiling failed: {str(e)}'}


# ============== Model Management ==============

@eel.expose
def get_available_models():
    """Get list of registered models."""
    try:
        models = model_registry.list_models()
        default_id = model_registry.get_default_model_id()
        return {
            'success': True,
            'models': models,
            'default_model': default_id
        }
    except Exception as e:
        return {'success': False, 'message': str(e), 'models': []}


@eel.expose
def ensure_model_available():
    """Download base model if needed."""
    try:
        def progress_callback(message, progress):
            eel.onDetectProgress(message, progress)()

        path = model_registry.ensure_base_model(progress_callback)
        return {'success': True, 'path': path}
    except Exception as e:
        return {'success': False, 'message': str(e)}


@eel.expose
def get_model_lineage(model_id):
    """Get the lineage chain for a model."""
    lineage = model_registry.get_model_lineage(model_id)
    return lineage


@eel.expose
def set_default_model(model_id):
    """Set the default model for detection."""
    return model_registry.set_default_model(model_id)


@eel.expose
def get_annotation_stats():
    """Get statistics about existing annotations for warning before detect all."""
    if not state.fields:
        return {
            'total_images': 0,
            'images_with_markers': 0,
            'images_reviewed': 0,
            'total_markers': 0
        }

    images_with_markers = 0
    images_reviewed = 0
    total_markers = 0

    for field in state.fields:
        if field.markers:
            images_with_markers += 1
            total_markers += len(field.markers)
        if field.review_status == ReviewStatus.REVIEWED:
            images_reviewed += 1

    return {
        'total_images': len(state.fields),
        'images_with_markers': images_with_markers,
        'images_reviewed': images_reviewed,
        'total_markers': total_markers
    }


# ============== Detection ==============

def _run_detection_on_field(field_obj, threshold=0.3, min_distance=5):
    """Run detection on a single field using KiNetDetector."""
    import numpy as np
    from PIL import Image as PILImage
    from kinet.kinet_detector import KiNetDetector

    img = PILImage.open(field_obj.filepath).convert('RGB')
    img_np = np.array(img)

    # Use shared KiNet detector with 3-class output
    detector = KiNetDetector()
    settings = {
        'threshold': threshold,
        'min_distance': min_distance,
        'tile_size': 1024,
        'include_other': True  # Trainer uses all 3 classes
    }
    nuclei = detector.detect(img_np, settings=settings)

    # Convert DetectedNucleus to Marker objects
    new_markers = []
    for nucleus in nuclei:
        # KiNet returns marker_class: 0=positive, 1=negative, 2=other
        mc = nucleus.marker_class if nucleus.marker_class is not None else 1
        new_markers.append(Marker(
            marker_class=int(mc),
            x=nucleus.x,
            y=nucleus.y,
            selected=False
        ))

    return new_markers


@eel.expose
def detect_current_image(model_id, threshold=0.3, min_distance=5):
    """Run detection on the current image, replacing its markers."""
    field = state.get_current_field()
    if not field:
        return {'success': False, 'message': 'No image loaded'}

    try:
        eel.onDetectProgress("Running detection...", 0.1)()
        new_markers = _run_detection_on_field(field, threshold, min_distance)

        # Save history and replace markers
        _save_to_history()
        field.markers = new_markers
        field.review_status = ReviewStatus.NEEDS_REVIEW

        eel.onDetectProgress("Done", 1.0)()

        resp = _marker_response()
        resp['success'] = True
        resp['detected_count'] = len(new_markers)
        resp['review_status'] = field.review_status
        return resp
    except ImportError as e:
        return {'success': False, 'message': f'Missing dependency: {str(e)}'}
    except Exception as e:
        return {'success': False, 'message': f'Detection failed: {str(e)}'}


@eel.expose
def detect_all_images(model_id, threshold=0.3, min_distance=5):
    """Run detection on all images."""
    if not state.fields:
        return {'success': False, 'message': 'No images loaded'}

    try:
        total = len(state.fields)
        total_detected = 0

        for i, field_obj in enumerate(state.fields):
            eel.onDetectProgress(
                f"Detecting image {i + 1}/{total}...",
                0.1 + 0.85 * (i / total)
            )()

            new_markers = _run_detection_on_field(field_obj, threshold, min_distance)

            # Save history for current field only (others are batch operation)
            if i == state.current_index:
                _save_to_history()

            field_obj.markers = new_markers
            field_obj.review_status = ReviewStatus.NEEDS_REVIEW
            total_detected += len(new_markers)

        eel.onDetectProgress("Detection complete", 1.0)()

        return {
            'success': True,
            'message': f'Detected {total_detected} markers across {total} images',
            'total_detected': total_detected
        }
    except ImportError as e:
        return {'success': False, 'message': f'Missing dependency: {str(e)}'}
    except Exception as e:
        return {'success': False, 'message': f'Detection failed: {str(e)}'}


# ============== Review Status ==============

@eel.expose
def set_review_status(status):
    """Set review status for the current image."""
    field = state.get_current_field()
    if not field:
        return None
    field.review_status = int(status)
    return {'review_status': field.review_status}


@eel.expose
def mark_reviewed_and_next():
    """Mark current image as reviewed and advance to next unreviewed."""
    field = state.get_current_field()
    if not field:
        return None

    field.review_status = ReviewStatus.REVIEWED
    save_project()

    # Find next unreviewed
    next_idx = state.find_next_unreviewed(state.current_index)
    if next_idx is not None:
        state.go_to_field(next_idx)
        return get_image_data()
    else:
        # No more unreviewed — stay on current
        return get_image_data()


@eel.expose
def toggle_reviewed():
    """Toggle review status: if reviewed, unlock; if not, mark reviewed and advance."""
    field = state.get_current_field()
    if not field:
        return None

    if field.review_status == ReviewStatus.REVIEWED:
        # Unlock - set back to needs review
        field.review_status = ReviewStatus.NEEDS_REVIEW
        save_project()
        return {
            'action': 'unlocked',
            'data': get_image_data()
        }
    else:
        # Lock - mark as reviewed and advance
        field.review_status = ReviewStatus.REVIEWED
        save_project()

        # Find next unreviewed
        next_idx = state.find_next_unreviewed(state.current_index)
        if next_idx is not None:
            state.go_to_field(next_idx)
        # Stay on current if no more unreviewed

        return {
            'action': 'locked',
            'data': get_image_data()
        }


@eel.expose
def get_review_summary():
    """Get review status counts."""
    return state.get_review_summary()


# ============== Model Evaluation ==============

@eel.expose
def evaluate_model(weights_path=None):
    """Run model evaluation on annotated images."""
    if weights_path is None:
        weights_path = select_open_file(
            title='Select Model Weights',
            file_types=[('PyTorch weights', '*.pth'), ('All files', '*.*')]
        )
    if not weights_path:
        return {'success': False, 'message': 'No weights file selected'}

    if not state.fields:
        return {'success': False, 'message': 'No images loaded'}

    try:
        # evaluate.py handles its own path setup for kinet_model import
        from evaluate import evaluate_on_fields

        def progress_callback(message, progress):
            eel.onEvalProgress(message, progress)()

        results = evaluate_on_fields(
            weights_path, state.fields,
            progress_callback=progress_callback
        )
        return results
    except ImportError as e:
        return {'success': False, 'message': f'Missing dependency: {str(e)}'}
    except Exception as e:
        return {'success': False, 'message': f'Evaluation failed: {str(e)}'}


# ============== About ==============

@eel.expose
def get_version():
    return __version__


@eel.expose
def open_url(url: str):
    import webbrowser
    webbrowser.open(url)


# ============== Main Entry Point ==============

def main():
    try:
        eel.start(
            'index.html',
            size=(1400, 900),
            port=0,
            mode='edge'
        )
    except EnvironmentError:
        try:
            eel.start(
                'index.html',
                size=(1400, 900),
                port=0,
                mode='chrome'
            )
        except EnvironmentError:
            eel.start(
                'index.html',
                size=(1400, 900),
                port=0,
                mode=None
            )


if __name__ == '__main__':
    main()
