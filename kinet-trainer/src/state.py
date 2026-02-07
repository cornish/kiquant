"""
Data classes for KiNet Trainer state management.
3-class annotation: positive tumor, negative tumor, non-tumor.
All classes are JSON-serializable for project persistence.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import IntEnum


class MarkerClass(IntEnum):
    POSITIVE = 0   # Ki-67+ stained tumor nuclei
    NEGATIVE = 1   # Unstained tumor nuclei
    OTHER = 2      # Non-tumor (stromal, immune, etc.)


class Mode(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    OTHER = 2
    ERASER = 3
    PAN = 4


class ReviewStatus(IntEnum):
    NOT_STARTED = 0   # No annotations, not detected
    NEEDS_REVIEW = 1  # Has auto-detected markers, not yet reviewed
    REVIEWED = 2      # User has reviewed and corrected this image


@dataclass
class Marker:
    """A single marker annotation on an image."""
    marker_class: int  # MarkerClass.POSITIVE, NEGATIVE, or OTHER
    x: int
    y: int
    selected: bool = False

    RADIUS: int = field(default=6, repr=False)

    def contains(self, px: int, py: int) -> bool:
        """Check if point (px, py) is within this marker's hit-box."""
        dx = px - self.x
        dy = py - self.y
        return (dx * dx + dy * dy) <= (self.RADIUS * self.RADIUS)

    def to_dict(self) -> dict:
        return {
            'marker_class': self.marker_class,
            'x': self.x,
            'y': self.y,
            'selected': self.selected
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Marker':
        return cls(
            marker_class=data['marker_class'],
            x=data['x'],
            y=data['y'],
            selected=data.get('selected', False)
        )


@dataclass
class Field:
    """A single image field with its markers."""
    filepath: str
    markers: List[Marker] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    review_status: int = 0  # ReviewStatus.NOT_STARTED

    def get_count_by_class(self, marker_class: int) -> int:
        return sum(1 for m in self.markers if m.marker_class == marker_class)

    def get_positive_count(self) -> int:
        return self.get_count_by_class(MarkerClass.POSITIVE)

    def get_negative_count(self) -> int:
        return self.get_count_by_class(MarkerClass.NEGATIVE)

    def get_other_count(self) -> int:
        return self.get_count_by_class(MarkerClass.OTHER)

    def get_total_count(self) -> int:
        return len(self.markers)

    def add_marker(self, marker: Marker) -> None:
        self.markers.append(marker)

    def remove_marker(self, index: int) -> None:
        if 0 <= index < len(self.markers):
            self.markers.pop(index)

    def deselect_all(self) -> None:
        for m in self.markers:
            m.selected = False

    def to_dict(self) -> dict:
        return {
            'filepath': self.filepath,
            'markers': [m.to_dict() for m in self.markers],
            'image_width': self.image_width,
            'image_height': self.image_height,
            'review_status': self.review_status
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Field':
        return cls(
            filepath=data['filepath'],
            markers=[Marker.from_dict(m) for m in data.get('markers', [])],
            image_width=data.get('image_width', 0),
            image_height=data.get('image_height', 0),
            review_status=data.get('review_status', 0)
        )


@dataclass
class State:
    """Complete trainer project state."""
    image_dir: str = ''
    project_path: str = ''
    current_index: int = 0
    mode: int = Mode.POSITIVE
    fields: List[Field] = field(default_factory=list)

    IMAGE_EXTENSIONS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}

    def load_image_directory(self, directory: str) -> int:
        """Load all images from directory. Returns count of images found."""
        self.image_dir = directory
        self.fields = []

        if not os.path.isdir(directory):
            return 0

        files = []
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.IMAGE_EXTENSIONS:
                filepath = os.path.join(directory, filename)
                files.append(filepath)

        files.sort()
        for filepath in files:
            self.fields.append(Field(filepath=filepath))

        self.current_index = 0
        return len(self.fields)

    def find_new_images(self) -> List[str]:
        """Find images in directory that aren't in the project yet."""
        if not self.image_dir or not os.path.isdir(self.image_dir):
            return []

        # Get existing filepaths
        existing = {f.filepath for f in self.fields}

        new_images = []
        for filename in os.listdir(self.image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.IMAGE_EXTENSIONS:
                filepath = os.path.join(self.image_dir, filename)
                if filepath not in existing:
                    new_images.append(filepath)

        new_images.sort()
        return new_images

    def find_missing_images(self) -> List[str]:
        """Find images in project that no longer exist on disk."""
        missing = []
        for field in self.fields:
            if not os.path.isfile(field.filepath):
                missing.append(field.filepath)
        return missing

    def remove_missing_images(self) -> int:
        """Remove fields for images that no longer exist. Returns count removed."""
        original_count = len(self.fields)
        self.fields = [f for f in self.fields if os.path.isfile(f.filepath)]
        removed = original_count - len(self.fields)

        # Adjust current_index if needed
        if self.current_index >= len(self.fields):
            self.current_index = max(0, len(self.fields) - 1)

        return removed

    def add_images(self, filepaths: List[str]) -> int:
        """Add new images to the project. Returns count added."""
        added = 0
        existing = {f.filepath for f in self.fields}

        for filepath in filepaths:
            if filepath not in existing and os.path.isfile(filepath):
                self.fields.append(Field(filepath=filepath))
                added += 1

        # Re-sort fields by filepath to maintain order
        self.fields.sort(key=lambda f: f.filepath)

        # Update current_index if needed to stay on same image
        return added

    def get_current_field(self) -> Optional[Field]:
        if 0 <= self.current_index < len(self.fields):
            return self.fields[self.current_index]
        return None

    def get_total(self) -> int:
        return len(self.fields)

    def next_field(self) -> int:
        if self.fields:
            self.current_index = (self.current_index + 1) % len(self.fields)
        return self.current_index

    def previous_field(self) -> int:
        if self.fields:
            self.current_index = (self.current_index - 1) % len(self.fields)
        return self.current_index

    def go_to_field(self, index: int) -> int:
        if self.fields:
            self.current_index = max(0, min(index, len(self.fields) - 1))
        return self.current_index

    def get_summary(self) -> dict:
        """Get overall statistics."""
        total_pos = 0
        total_neg = 0
        total_other = 0
        for f in self.fields:
            total_pos += f.get_positive_count()
            total_neg += f.get_negative_count()
            total_other += f.get_other_count()

        total = total_pos + total_neg + total_other
        annotated = sum(1 for f in self.fields if f.get_total_count() > 0)

        return {
            'positive': total_pos,
            'negative': total_neg,
            'other': total_other,
            'total': total,
            'annotated_images': annotated,
            'total_images': len(self.fields)
        }

    def get_review_summary(self) -> dict:
        """Get counts by review status."""
        not_started = 0
        needs_review = 0
        reviewed = 0
        for f in self.fields:
            if f.review_status == ReviewStatus.REVIEWED:
                reviewed += 1
            elif f.review_status == ReviewStatus.NEEDS_REVIEW:
                needs_review += 1
            else:
                not_started += 1
        return {
            'not_started': not_started,
            'needs_review': needs_review,
            'reviewed': reviewed,
            'total': len(self.fields)
        }

    def find_next_unreviewed(self, from_index: int = 0) -> Optional[int]:
        """Find the next image needing review, starting from from_index+1. Wraps around."""
        n = len(self.fields)
        if n == 0:
            return None
        for offset in range(1, n + 1):
            idx = (from_index + offset) % n
            if self.fields[idx].review_status == ReviewStatus.NEEDS_REVIEW:
                return idx
        return None

    def to_dict(self) -> dict:
        return {
            'app': 'kinet-trainer',
            'version': 1,
            'image_dir': self.image_dir,
            'project_path': self.project_path,
            'current_index': self.current_index,
            'mode': self.mode,
            'fields': [f.to_dict() for f in self.fields]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'State':
        state = cls(
            image_dir=data.get('image_dir', ''),
            project_path=data.get('project_path', ''),
            current_index=data.get('current_index', 0),
            mode=data.get('mode', Mode.POSITIVE)
        )
        state.fields = [Field.from_dict(f) for f in data.get('fields', [])]
        return state

    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """Save state to JSON file. Returns filepath used."""
        if filepath:
            self.project_path = filepath

        with open(self.project_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

        return self.project_path

    @classmethod
    def load_from_file(cls, filepath: str) -> 'State':
        """Load state from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = cls.from_dict(data)
        state.project_path = filepath
        return state

    @classmethod
    def import_from_kiquant(cls, filepath: str) -> 'State':
        """Import a kiQuant project. Maps class 0->0, class 1->1, no class 2."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = cls(
            image_dir=data.get('image_dir', ''),
            current_index=data.get('current_index', 0),
            mode=Mode.POSITIVE
        )

        for field_data in data.get('fields', []):
            markers = []
            for m in field_data.get('markers', []):
                # kiQuant: class 0 = positive, class 1 = negative
                # Trainer: class 0 = positive, class 1 = negative, class 2 = other
                markers.append(Marker(
                    marker_class=m['marker_class'],  # 0 and 1 map directly
                    x=m['x'],
                    y=m['y'],
                    selected=False
                ))
            state.fields.append(Field(
                filepath=field_data['filepath'],
                markers=markers,
                image_width=field_data.get('image_width', 0),
                image_height=field_data.get('image_height', 0)
            ))

        return state
