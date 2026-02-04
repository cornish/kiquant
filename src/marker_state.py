"""
Data classes for kiQuant state management.
All classes are JSON-serializable for project persistence.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import IntEnum


class MarkerClass(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1


class Mode(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    SELECT = 2
    ERASER = 3


@dataclass
class Marker:
    """A single marker annotation on an image."""
    marker_class: int  # MarkerClass.POSITIVE or MarkerClass.NEGATIVE
    x: int
    y: int
    selected: bool = False

    # Marker hit-box radius for click detection
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

    def get_count_by_class(self, marker_class: int) -> int:
        return sum(1 for m in self.markers if m.marker_class == marker_class)

    def get_positive_count(self) -> int:
        return self.get_count_by_class(MarkerClass.POSITIVE)

    def get_negative_count(self) -> int:
        return self.get_count_by_class(MarkerClass.NEGATIVE)

    def get_total_count(self) -> int:
        return len(self.markers)

    def add_marker(self, marker: Marker) -> None:
        self.markers.append(marker)

    def remove_marker(self, index: int) -> None:
        if 0 <= index < len(self.markers):
            self.markers.pop(index)

    def remove_selected_markers(self) -> int:
        """Remove all selected markers. Returns count of removed markers."""
        original_count = len(self.markers)
        self.markers = [m for m in self.markers if not m.selected]
        return original_count - len(self.markers)

    def get_selected_markers(self) -> List[Marker]:
        return [m for m in self.markers if m.selected]

    def deselect_all(self) -> None:
        for m in self.markers:
            m.selected = False

    def select_marker_at(self, x: int, y: int) -> Optional[Marker]:
        """Select marker at point. Returns the marker if found."""
        for m in self.markers:
            if m.contains(x, y):
                m.selected = True
                return m
        return None

    def select_markers_in_rect(self, x: int, y: int, width: int, height: int) -> int:
        """Select all markers within rectangle. Returns count selected."""
        count = 0
        for m in self.markers:
            if x <= m.x <= x + width and y <= m.y <= y + height:
                m.selected = True
                count += 1
        return count

    def to_dict(self) -> dict:
        return {
            'filepath': self.filepath,
            'markers': [m.to_dict() for m in self.markers]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Field':
        return cls(
            filepath=data['filepath'],
            markers=[Marker.from_dict(m) for m in data.get('markers', [])]
        )


@dataclass
class State:
    """Complete project state."""
    image_dir: str = ''
    output_path: str = ''
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

    def get_current_field(self) -> Optional[Field]:
        if 0 <= self.current_index < len(self.fields):
            return self.fields[self.current_index]
        return None

    def get_total(self) -> int:
        return len(self.fields)

    def next_field(self) -> int:
        """Move to next field (wraps around). Returns new index."""
        if self.fields:
            self.current_index = (self.current_index + 1) % len(self.fields)
        return self.current_index

    def previous_field(self) -> int:
        """Move to previous field (wraps around). Returns new index."""
        if self.fields:
            self.current_index = (self.current_index - 1) % len(self.fields)
        return self.current_index

    def go_to_field(self, index: int) -> int:
        """Go to specific field index. Returns actual index."""
        if self.fields:
            self.current_index = max(0, min(index, len(self.fields) - 1))
        return self.current_index

    def get_summary(self) -> dict:
        """Get overall statistics."""
        total_pos = 0
        total_neg = 0
        for f in self.fields:
            total_pos += f.get_positive_count()
            total_neg += f.get_negative_count()

        total = total_pos + total_neg
        pi = (total_pos / total * 100) if total > 0 else 0.0

        return {
            'positive': total_pos,
            'negative': total_neg,
            'total': total,
            'proliferation_index': round(pi, 1)
        }

    def to_dict(self) -> dict:
        return {
            'image_dir': self.image_dir,
            'output_path': self.output_path,
            'project_path': self.project_path,
            'current_index': self.current_index,
            'mode': self.mode,
            'fields': [f.to_dict() for f in self.fields]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'State':
        state = cls(
            image_dir=data.get('image_dir', ''),
            output_path=data.get('output_path', ''),
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

    def export_csv(self, filepath: Optional[str] = None) -> str:
        """Export results to CSV. Returns filepath used."""
        if filepath is None:
            filepath = self.output_path

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write('image\tpositive\tnegative\ttotal\n')
            for field in self.fields:
                pos = field.get_positive_count()
                neg = field.get_negative_count()
                total = pos + neg
                # Use just the filename, not full path
                filename = os.path.basename(field.filepath)
                f.write(f'{filename}\t{pos}\t{neg}\t{total}\n')

        return filepath
