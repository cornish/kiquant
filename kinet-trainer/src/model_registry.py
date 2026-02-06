"""
Model registry for KiNet trained models.

Tracks model lineage (which training data and parent model produced each
fine-tuned model), stores metadata, and manages base model download.

Registry storage: ~/.kiquant/models/registry.json
"""

import json
import os
import urllib.request
from datetime import datetime


# Same URL and path used by kiQuant's kinet_detector.py
MODEL_URL = "https://www.dropbox.com/s/sl2l5z3d65l983t/ki67net-best.pth?dl=1"
MODEL_DIR = os.path.join(os.path.expanduser('~'), '.kiquant', 'models')
BASE_MODEL_FILENAME = "ki67net-best.pth"


def get_registry_path():
    """Return path to the registry JSON file."""
    return os.path.join(MODEL_DIR, 'registry.json')


def load_registry():
    """Load the registry, creating an empty one if it doesn't exist."""
    path = get_registry_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'models': {}, 'default_model': 'base'}


def save_registry(registry):
    """Write the registry to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = get_registry_path()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)


def ensure_base_model(progress_callback=None):
    """
    Download base KiNet weights if not cached, register as 'base'.

    Returns the path to the base model weights.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, BASE_MODEL_FILENAME)

    # Download if not present
    if not os.path.exists(model_path):
        if progress_callback:
            progress_callback("Downloading KiNet base model...", 0.0)

        def report_progress(block_num, block_size, total_size):
            if total_size > 0 and progress_callback:
                frac = min(block_num * block_size / total_size, 1.0)
                progress_callback(
                    f"Downloading KiNet base model... {int(frac * 100)}%",
                    frac * 0.9
                )

        urllib.request.urlretrieve(MODEL_URL, model_path, report_progress)

        if progress_callback:
            progress_callback("Download complete", 0.95)

    # Ensure base is registered
    registry = load_registry()
    if 'base' not in registry['models']:
        registry['models']['base'] = {
            'id': 'base',
            'name': 'KiNet Base (Xing et al.)',
            'path': model_path,
            'created': '2024-01-01T00:00:00',
            'parent_model': None,
            'training_data': None,
            'metrics': None,
            'description': 'Original pre-trained weights'
        }
        save_registry(registry)

    if progress_callback:
        progress_callback("Base model ready", 1.0)

    return model_path


def register_model(model_id, name, path, parent_model=None,
                   training_data=None, metrics=None, description=''):
    """
    Register a new model in the registry.

    Returns the model entry dict.
    """
    registry = load_registry()

    entry = {
        'id': model_id,
        'name': name,
        'path': path,
        'created': datetime.now().isoformat(timespec='seconds'),
        'parent_model': parent_model,
        'training_data': training_data,
        'metrics': metrics,
        'description': description
    }

    registry['models'][model_id] = entry
    save_registry(registry)
    return entry


def list_models():
    """Return all registered models sorted by creation date (newest first)."""
    registry = load_registry()
    models = list(registry['models'].values())
    # Sort: base first, then by date descending
    models.sort(key=lambda m: (
        m['id'] != 'base',
        m.get('created', '') or ''
    ), reverse=False)
    # Re-sort non-base by date descending
    base = [m for m in models if m['id'] == 'base']
    rest = [m for m in models if m['id'] != 'base']
    rest.sort(key=lambda m: m.get('created', ''), reverse=True)
    return base + rest


def get_model(model_id):
    """Return a single model entry, or None if not found."""
    registry = load_registry()
    return registry['models'].get(model_id)


def get_default_model_id():
    """Return the ID of the default model."""
    registry = load_registry()
    return registry.get('default_model', 'base')


def set_default_model(model_id):
    """Set the default model ID."""
    registry = load_registry()
    if model_id not in registry['models']:
        return False
    registry['default_model'] = model_id
    save_registry(registry)
    return True


def get_model_lineage(model_id):
    """
    Walk the parent_model chain from root to the given model.

    Returns a list of model entries from the oldest ancestor to the
    requested model (e.g. [base, ft-1, ft-2]).
    """
    registry = load_registry()
    chain = []
    visited = set()
    current_id = model_id

    while current_id and current_id not in visited:
        visited.add(current_id)
        entry = registry['models'].get(current_id)
        if not entry:
            break
        chain.append(entry)
        current_id = entry.get('parent_model')

    chain.reverse()
    return chain


def generate_model_id():
    """Generate a unique model ID based on timestamp: ft-YYYYMMDD-HHMMSS."""
    return 'ft-' + datetime.now().strftime('%Y%m%d-%H%M%S')


def resolve_weights_path(weights_arg):
    """
    Resolve a --weights argument to a file path.

    Accepts:
      - A model ID (e.g. 'base', 'ft-20260206-143022') -> resolves via registry
      - A file path -> returned as-is

    Returns (path, model_id) tuple. model_id is None if the arg was a path.
    """
    if weights_arg is None:
        return None, None

    # Check if it's a registered model ID
    model = get_model(weights_arg)
    if model:
        return model['path'], model['id']

    # Check if it's 'base' but not yet registered
    if weights_arg == 'base':
        path = ensure_base_model()
        return path, 'base'

    # Treat as a file path
    if os.path.exists(weights_arg):
        return weights_arg, None

    return None, None


def find_model_id_by_path(path):
    """Find the model ID for a given weights path, or None."""
    if not path:
        return None
    registry = load_registry()
    # Normalize path for comparison
    norm_path = os.path.normpath(os.path.abspath(path))
    for model_id, entry in registry['models'].items():
        entry_path = os.path.normpath(os.path.abspath(entry['path']))
        if entry_path == norm_path:
            return model_id
    return None
