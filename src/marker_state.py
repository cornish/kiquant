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
    PAN = 4
    ROI = 5


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
class ROI:
    """Region of Interest for quantification."""
    x: int
    y: int
    width: int
    height: int

    def contains(self, px: int, py: int) -> bool:
        """Check if point (px, py) is within the ROI."""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height}

    @classmethod
    def from_dict(cls, data: dict) -> 'ROI':
        return cls(x=data['x'], y=data['y'], width=data['width'], height=data['height'])


@dataclass
class Field:
    """A single image field with its markers."""
    filepath: str
    markers: List[Marker] = field(default_factory=list)
    roi: Optional[ROI] = None

    def get_markers_in_roi(self) -> List[Marker]:
        """Get markers within ROI, or all markers if no ROI set."""
        if self.roi is None:
            return self.markers
        return [m for m in self.markers if self.roi.contains(m.x, m.y)]

    def get_count_by_class(self, marker_class: int, use_roi: bool = True) -> int:
        markers = self.get_markers_in_roi() if use_roi else self.markers
        return sum(1 for m in markers if m.marker_class == marker_class)

    def get_positive_count(self, use_roi: bool = True) -> int:
        return self.get_count_by_class(MarkerClass.POSITIVE, use_roi)

    def get_negative_count(self, use_roi: bool = True) -> int:
        return self.get_count_by_class(MarkerClass.NEGATIVE, use_roi)

    def get_total_count(self, use_roi: bool = True) -> int:
        return len(self.get_markers_in_roi() if use_roi else self.markers)

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

    def find_hotspot(self, target_count: int = 500, image_width: int = None,
                     image_height: int = None) -> Optional[ROI]:
        """
        Find square ROI with ~target_count nuclei having highest positive ratio.

        Uses a sliding window approach to find the region with the most
        positive nuclei while containing approximately target_count total nuclei.

        Args:
            target_count: Target number of nuclei in the ROI (default 500)
            image_width: Image width to constrain ROI bounds
            image_height: Image height to constrain ROI bounds

        Returns:
            ROI with best hotspot, or None if not enough markers
        """
        if len(self.markers) < target_count // 2:
            return None

        # Estimate square size based on marker density
        if len(self.markers) == 0:
            return None

        xs = [m.x for m in self.markers]
        ys = [m.y for m in self.markers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate density and estimate square size for target count
        area = max((max_x - min_x) * (max_y - min_y), 1)
        density = len(self.markers) / area
        target_area = target_count / density if density > 0 else area
        square_size = int(target_area ** 0.5)

        # Ensure minimum size
        square_size = max(square_size, 50)

        # Constrain to image bounds if provided
        if image_width:
            square_size = min(square_size, image_width)
        if image_height:
            square_size = min(square_size, image_height)

        # Sliding window search with step size
        step = max(square_size // 10, 20)
        best_roi = None
        best_score = -1

        # Define search bounds
        search_min_x = min_x - square_size // 2
        search_max_x = max_x
        search_min_y = min_y - square_size // 2
        search_max_y = max_y

        if image_width:
            search_min_x = max(0, search_min_x)
            search_max_x = min(image_width - square_size, search_max_x)
        if image_height:
            search_min_y = max(0, search_min_y)
            search_max_y = min(image_height - square_size, search_max_y)

        for x in range(search_min_x, search_max_x + 1, step):
            for y in range(search_min_y, search_max_y + 1, step):
                # Count markers in this window
                pos_count = 0
                total_count = 0
                for m in self.markers:
                    if x <= m.x <= x + square_size and y <= m.y <= y + square_size:
                        total_count += 1
                        if m.marker_class == MarkerClass.POSITIVE:
                            pos_count += 1

                # Score: prioritize regions with ~target_count and high positive ratio
                if total_count > 0:
                    # Penalize being far from target count
                    count_factor = 1.0 - abs(total_count - target_count) / max(target_count, total_count)
                    count_factor = max(0, count_factor)

                    # Positive ratio
                    pos_ratio = pos_count / total_count

                    # Combined score: weight positive ratio more if we have enough cells
                    score = pos_ratio * (0.5 + 0.5 * count_factor)

                    if total_count >= target_count * 0.5:  # At least half target
                        score *= 1.5  # Bonus for having enough cells

                    if score > best_score:
                        best_score = score
                        best_roi = ROI(x=x, y=y, width=square_size, height=square_size)

        return best_roi

    def to_dict(self) -> dict:
        result = {
            'filepath': self.filepath,
            'markers': [m.to_dict() for m in self.markers]
        }
        if self.roi is not None:
            result['roi'] = self.roi.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'Field':
        roi = None
        if 'roi' in data and data['roi'] is not None:
            roi = ROI.from_dict(data['roi'])
        return cls(
            filepath=data['filepath'],
            markers=[Marker.from_dict(m) for m in data.get('markers', [])],
            roi=roi
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
