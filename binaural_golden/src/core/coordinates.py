"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COORDINATE SYSTEM - Unified Conventions                   ║
║                                                                              ║
║   SINGLE SOURCE OF TRUTH for coordinate transformations across the system.  ║
║   This module defines the spatial conventions used throughout the           ║
║   vibroacoustic plate optimization framework.                               ║
║                                                                              ║
║   By centralizing coordinates here, the framework becomes:                  ║
║   - Agnostic to specific physics implementations                            ║
║   - Portable to different visualization frameworks (web, Qt, etc.)          ║
║   - Easily adaptable to different body orientations                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

COORDINATE CONVENTIONS
══════════════════════

1. NORMALIZED COORDINATES (0-1 range, domain-agnostic)
   Used in: Genome classes (CutoutGene, ExciterPosition, SpringSupportGene, etc.)
   
   x_norm: Lateral position
           0.0 = left edge (person's left side when lying supine)
           1.0 = right edge (person's right side)
           
   y_norm: Longitudinal position
           0.0 = feet end of table
           1.0 = head end of table

2. PHYSICAL/FEM COORDINATES (meters, physics domain)
   Used in: FEM solvers, STL export, CNC paths
   
   x_phys: Length axis (along the body, longitudinal)
           0.0 = feet end
           L = head end (plate length in meters)
           
   y_phys: Width axis (across the body, lateral)
           0.0 = left edge
           W = right edge (plate width in meters)
   
   TRANSFORMATION: (x_norm, y_norm) → (y_norm * L, x_norm * W)
                   Note the swap! Normalized Y → Physical X, Normalized X → Physical Y

3. CANVAS/DISPLAY COORDINATES (pixels, visualization domain)
   Used in: Evolution canvas, UI components
   
   x_canvas: Horizontal position
             0 = left edge of canvas (feet end of plate)
             max = right edge (head end)
             
   y_canvas: Vertical position
             0 = top of canvas (person's right side)
             max = bottom (person's left side)
   
   TRANSFORMATION: (x_norm, y_norm) → (y_norm * canvas_w, (1-x_norm) * canvas_h)
                   Y-axis is inverted for screen coordinates!

USAGE EXAMPLE
═════════════

    from core.coordinates import (
        norm_to_physical, physical_to_norm,
        norm_to_canvas, canvas_to_norm,
        CoordinateSpace, Position
    )
    
    # Create a position in normalized space
    pos = Position(x=0.5, y=0.8)  # Center-lateral, 80% toward head
    
    # Convert to physical for FEM
    phys_x, phys_y = norm_to_physical(pos.x, pos.y, plate_length=2.0, plate_width=0.6)
    # Result: (1.6, 0.3) - 1.6m from feet, 0.3m from left edge
    
    # Convert to canvas for display
    canvas_x, canvas_y = norm_to_canvas(pos.x, pos.y, canvas_w=800, canvas_h=300)
    # Result: (640, 150) - 80% right, center vertically

"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
from enum import Enum, auto


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE SPACES
# ══════════════════════════════════════════════════════════════════════════════

class CoordinateSpace(Enum):
    """Available coordinate spaces in the system."""
    NORMALIZED = auto()   # (0-1, 0-1) - domain agnostic
    PHYSICAL = auto()     # (meters, meters) - FEM/physics
    CANVAS = auto()       # (pixels, pixels) - visualization


@dataclass
class Position:
    """
    A 2D position with its coordinate space.
    
    For normalized space: x=lateral (0=left, 1=right), y=longitudinal (0=feet, 1=head)
    """
    x: float
    y: float
    space: CoordinateSpace = CoordinateSpace.NORMALIZED
    
    def to_physical(self, plate_length: float, plate_width: float) -> 'Position':
        """Convert to physical coordinates."""
        if self.space == CoordinateSpace.PHYSICAL:
            return self
        elif self.space == CoordinateSpace.NORMALIZED:
            px, py = norm_to_physical(self.x, self.y, plate_length, plate_width)
            return Position(px, py, CoordinateSpace.PHYSICAL)
        else:
            raise ValueError(f"Cannot convert from {self.space} to PHYSICAL directly")
    
    def to_normalized(self, plate_length: float = 1.0, plate_width: float = 1.0) -> 'Position':
        """Convert to normalized coordinates."""
        if self.space == CoordinateSpace.NORMALIZED:
            return self
        elif self.space == CoordinateSpace.PHYSICAL:
            nx, ny = physical_to_norm(self.x, self.y, plate_length, plate_width)
            return Position(nx, ny, CoordinateSpace.NORMALIZED)
        else:
            raise ValueError(f"Cannot convert from {self.space} to NORMALIZED directly")
    
    def to_canvas(self, canvas_w: int, canvas_h: int, 
                  plate_coords: Optional[Tuple[float, float, float, float]] = None) -> 'Position':
        """
        Convert to canvas coordinates.
        
        Args:
            canvas_w, canvas_h: Canvas dimensions in pixels
            plate_coords: (x0, y0, x1, y1) plate bounding box on canvas (optional)
        """
        if self.space == CoordinateSpace.CANVAS:
            return self
        elif self.space == CoordinateSpace.NORMALIZED:
            if plate_coords:
                x0, y0, x1, y1 = plate_coords
                cx = x0 + self.y * (x1 - x0)  # y_norm → canvas X
                cy = y0 + (1 - self.x) * (y1 - y0)  # x_norm → canvas Y (inverted)
            else:
                cx, cy = norm_to_canvas(self.x, self.y, canvas_w, canvas_h)
            return Position(cx, cy, CoordinateSpace.CANVAS)
        else:
            raise ValueError(f"Cannot convert from {self.space} to CANVAS directly")


@dataclass
class BoundingBox:
    """A bounding box with min/max in a coordinate space."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    space: CoordinateSpace = CoordinateSpace.NORMALIZED
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def norm_to_physical(
    x_norm: float, 
    y_norm: float, 
    plate_length: float, 
    plate_width: float
) -> Tuple[float, float]:
    """
    Convert normalized coordinates to physical (FEM) coordinates.
    
    IMPORTANT: This performs a coordinate SWAP!
    - Normalized X (lateral) → Physical Y (width axis)
    - Normalized Y (longitudinal) → Physical X (length axis)
    
    Args:
        x_norm: Lateral position (0=left, 1=right)
        y_norm: Longitudinal position (0=feet, 1=head)
        plate_length: Plate length in meters (along body)
        plate_width: Plate width in meters (across body)
    
    Returns:
        (x_phys, y_phys) in meters
    """
    x_phys = y_norm * plate_length  # y_norm → x_phys (longitudinal)
    y_phys = x_norm * plate_width   # x_norm → y_phys (lateral)
    return (x_phys, y_phys)


def physical_to_norm(
    x_phys: float, 
    y_phys: float, 
    plate_length: float, 
    plate_width: float
) -> Tuple[float, float]:
    """
    Convert physical (FEM) coordinates to normalized.
    
    Inverse of norm_to_physical.
    
    Args:
        x_phys: Position along length axis (0=feet, L=head)
        y_phys: Position along width axis (0=left, W=right)
        plate_length: Plate length in meters
        plate_width: Plate width in meters
    
    Returns:
        (x_norm, y_norm) in range [0, 1]
    """
    if plate_length <= 0 or plate_width <= 0:
        return (0.5, 0.5)  # Fallback to center
    
    y_norm = x_phys / plate_length  # x_phys → y_norm
    x_norm = y_phys / plate_width   # y_phys → x_norm
    return (np.clip(x_norm, 0, 1), np.clip(y_norm, 0, 1))


def norm_to_canvas(
    x_norm: float, 
    y_norm: float, 
    canvas_w: int, 
    canvas_h: int,
    margin: float = 0.0
) -> Tuple[float, float]:
    """
    Convert normalized coordinates to canvas pixel coordinates.
    
    Note: Canvas Y is inverted (0 at top, increases downward).
    
    Args:
        x_norm: Lateral position (0=left, 1=right body)
        y_norm: Longitudinal position (0=feet, 1=head)
        canvas_w: Canvas width in pixels
        canvas_h: Canvas height in pixels
        margin: Margin in pixels (optional)
    
    Returns:
        (x_canvas, y_canvas) in pixels
    """
    usable_w = canvas_w - 2 * margin
    usable_h = canvas_h - 2 * margin
    
    x_canvas = margin + y_norm * usable_w       # y_norm → x_canvas (head at right)
    y_canvas = margin + (1 - x_norm) * usable_h  # x_norm → y_canvas (inverted)
    
    return (x_canvas, y_canvas)


def canvas_to_norm(
    x_canvas: float, 
    y_canvas: float, 
    canvas_w: int, 
    canvas_h: int,
    margin: float = 0.0
) -> Tuple[float, float]:
    """
    Convert canvas pixel coordinates to normalized.
    
    Inverse of norm_to_canvas.
    """
    usable_w = canvas_w - 2 * margin
    usable_h = canvas_h - 2 * margin
    
    if usable_w <= 0 or usable_h <= 0:
        return (0.5, 0.5)
    
    y_norm = (x_canvas - margin) / usable_w
    x_norm = 1 - (y_canvas - margin) / usable_h
    
    return (np.clip(x_norm, 0, 1), np.clip(y_norm, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# BATCH TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════

def transform_points(
    points: np.ndarray,
    from_space: CoordinateSpace,
    to_space: CoordinateSpace,
    plate_length: float = 1.0,
    plate_width: float = 1.0,
    canvas_w: int = 800,
    canvas_h: int = 600
) -> np.ndarray:
    """
    Transform array of points between coordinate spaces.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        from_space: Source coordinate space
        to_space: Target coordinate space
        plate_length, plate_width: Plate dimensions (for PHYSICAL)
        canvas_w, canvas_h: Canvas dimensions (for CANVAS)
    
    Returns:
        (N, 2) array of transformed coordinates
    """
    if from_space == to_space:
        return points.copy()
    
    result = np.zeros_like(points)
    
    for i, (x, y) in enumerate(points):
        if from_space == CoordinateSpace.NORMALIZED:
            if to_space == CoordinateSpace.PHYSICAL:
                result[i] = norm_to_physical(x, y, plate_length, plate_width)
            elif to_space == CoordinateSpace.CANVAS:
                result[i] = norm_to_canvas(x, y, canvas_w, canvas_h)
        elif from_space == CoordinateSpace.PHYSICAL:
            if to_space == CoordinateSpace.NORMALIZED:
                result[i] = physical_to_norm(x, y, plate_length, plate_width)
            elif to_space == CoordinateSpace.CANVAS:
                nx, ny = physical_to_norm(x, y, plate_length, plate_width)
                result[i] = norm_to_canvas(nx, ny, canvas_w, canvas_h)
        elif from_space == CoordinateSpace.CANVAS:
            if to_space == CoordinateSpace.NORMALIZED:
                result[i] = canvas_to_norm(x, y, canvas_w, canvas_h)
            elif to_space == CoordinateSpace.PHYSICAL:
                nx, ny = canvas_to_norm(x, y, canvas_w, canvas_h)
                result[i] = norm_to_physical(nx, ny, plate_length, plate_width)
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BODY ZONE HELPERS (domain-specific but useful for vibroacoustic therapy)
# ══════════════════════════════════════════════════════════════════════════════

class BodyZone(Enum):
    """Standard body zones for vibroacoustic therapy."""
    HEAD = "head"
    CERVICAL = "cervical"
    THORACIC = "thoracic"
    LUMBAR = "lumbar"
    PELVIS = "pelvis"
    LEGS = "legs"
    FEET = "feet"


# Default zone boundaries (normalized y coordinates)
BODY_ZONE_BOUNDARIES = {
    BodyZone.HEAD: (0.90, 1.00),
    BodyZone.CERVICAL: (0.80, 0.90),
    BodyZone.THORACIC: (0.55, 0.80),
    BodyZone.LUMBAR: (0.40, 0.55),
    BodyZone.PELVIS: (0.25, 0.40),
    BodyZone.LEGS: (0.10, 0.25),
    BodyZone.FEET: (0.00, 0.10),
}


def get_zone_at_position(y_norm: float) -> Optional[BodyZone]:
    """Get the body zone at a normalized longitudinal position."""
    for zone, (y_min, y_max) in BODY_ZONE_BOUNDARIES.items():
        if y_min <= y_norm <= y_max:
            return zone
    return None


def get_zone_center(zone: BodyZone) -> float:
    """Get the center y_norm position of a body zone."""
    y_min, y_max = BODY_ZONE_BOUNDARIES[zone]
    return (y_min + y_max) / 2


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("COORDINATE SYSTEM - Unit Tests")
    print("=" * 60)
    
    # Test 1: Normalized → Physical
    print("\n1. Normalized → Physical:")
    nx, ny = 0.5, 0.8  # Center-lateral, 80% toward head
    px, py = norm_to_physical(nx, ny, plate_length=2.0, plate_width=0.6)
    print(f"   Norm ({nx}, {ny}) → Phys ({px:.2f}m, {py:.2f}m)")
    assert abs(px - 1.6) < 0.01, "Physical X should be 1.6m"
    assert abs(py - 0.3) < 0.01, "Physical Y should be 0.3m"
    print("   ✅ PASS")
    
    # Test 2: Physical → Normalized (round-trip)
    print("\n2. Physical → Normalized (round-trip):")
    nx2, ny2 = physical_to_norm(px, py, plate_length=2.0, plate_width=0.6)
    print(f"   Phys ({px:.2f}, {py:.2f}) → Norm ({nx2:.2f}, {ny2:.2f})")
    assert abs(nx2 - nx) < 0.01, "Round-trip X failed"
    assert abs(ny2 - ny) < 0.01, "Round-trip Y failed"
    print("   ✅ PASS")
    
    # Test 3: Normalized → Canvas
    print("\n3. Normalized → Canvas:")
    cx, cy = norm_to_canvas(nx, ny, canvas_w=800, canvas_h=300)
    print(f"   Norm ({nx}, {ny}) → Canvas ({cx:.0f}px, {cy:.0f}px)")
    assert abs(cx - 640) < 1, "Canvas X should be ~640px (80% of 800)"
    assert abs(cy - 150) < 1, "Canvas Y should be ~150px (center of 300)"
    print("   ✅ PASS")
    
    # Test 4: Position class
    print("\n4. Position class transformations:")
    pos = Position(x=0.3, y=0.9)  # Left-ish, near head
    pos_phys = pos.to_physical(plate_length=2.0, plate_width=0.6)
    print(f"   Norm (0.3, 0.9) → Phys ({pos_phys.x:.2f}m, {pos_phys.y:.2f}m)")
    assert pos_phys.space == CoordinateSpace.PHYSICAL
    print("   ✅ PASS")
    
    # Test 5: Body zones
    print("\n5. Body zone detection:")
    test_positions = [0.05, 0.35, 0.5, 0.75, 0.95]
    for y in test_positions:
        zone = get_zone_at_position(y)
        print(f"   y={y:.2f} → {zone.value if zone else 'None'}")
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
