"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           PLATE CANVAS                                      ║
║                                                                              ║
║   Professional canvas for drawing plates with rulers, grid, and layers    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from typing import Optional, Callable, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from ..theme import PlateLabStyle
from .genome_card import PlateGenome, ContourType

STYLE = PlateLabStyle()


class LayerType(Enum):
    """Canvas layer types."""
    GRID = "grid"
    RULERS = "rulers"
    PLATE = "plate"
    HUMAN = "human"
    MODES = "modes"
    EXCITERS = "exciters"
    ANNOTATIONS = "annotations"


@dataclass
class Person:
    """Human body representation for overlay."""
    height: float = 1.75  # meters
    width: float = 0.45   # meters
    position: Tuple[float, float] = (0.0, 0.0)  # x, y offset from plate center
    visible: bool = True


@dataclass
class Exciter:
    """Exciter/transducer representation."""
    x: float
    y: float
    radius: float = 0.05  # meters
    active: bool = True
    label: str = ""


@dataclass
class CanvasConfig:
    """Configuration for PlateCanvas."""
    width: int = 800
    height: int = 600
    show_grid: bool = True
    show_rulers: bool = True
    grid_size: float = 0.1  # meters
    ruler_size: int = 30    # pixels
    min_zoom: float = 0.1
    max_zoom: float = 10.0
    snap_to_grid: bool = True
    anti_aliasing: bool = True
    
    # Layer visibility
    layer_visibility: Dict[LayerType, bool] = None
    
    def __post_init__(self):
        if self.layer_visibility is None:
            self.layer_visibility = {
                layer: True for layer in LayerType
            }


class PlateCanvas(tk.Canvas):
    """
    Canvas professionale per disegnare tavole vibroacustiche.
    
    Features:
    - Grid con snap to grid
    - Rulers (righe metriche) sui bordi
    - Zoom/pan con mouse
    - Layers separati (plate, human, modes, exciters)
    - Anti-aliasing per forme smooth
    - Coordinate fisiche (metri)
    - Tools per misurazioni
    
    Example:
        canvas = PlateCanvas(
            parent,
            config=CanvasConfig(width=1000, show_grid=True)
        )
        
        genome = PlateGenome(length=2.0, width=1.0, thickness=0.02)
        canvas.draw_plate(genome)
        
        person = Person(height=1.75, position=(0.0, 0.1))
        canvas.draw_human_overlay(person, opacity=0.3)
    """
    
    def __init__(self, parent, *, config: Optional[CanvasConfig] = None,
                 on_click: Optional[Callable[[float, float], None]] = None,
                 on_drag: Optional[Callable[[float, float, float, float], None]] = None):
        
        self.config = config or CanvasConfig()
        
        super().__init__(
            parent,
            width=self.config.width,
            height=self.config.height,
            bg=STYLE.BG_DARK,
            highlightthickness=0
        )
        
        # Coordinate system
        self._zoom = 1.0
        self._pan_x = 0.0  # Physical offset in meters
        self._pan_y = 0.0
        self._pixels_per_meter = 200  # Base scale
        
        # Current data
        self._current_plate: Optional[PlateGenome] = None
        self._current_person: Optional[Person] = None
        self._exciters: List[Exciter] = []
        self._mode_data: Optional[np.ndarray] = None
        self._heatmap_data: Optional[np.ndarray] = None
        
        # Interaction state
        self._dragging = False
        self._drag_start: Tuple[int, int] = (0, 0)
        self._last_click_physical: Tuple[float, float] = (0.0, 0.0)
        
        # Callbacks
        self._on_click = on_click
        self._on_drag = on_drag
        
        # Bind events
        self._bind_events()
        
        # Initial draw
        self._draw_all()
    
    def _bind_events(self):
        """Bind mouse and keyboard events."""
        self.bind("<Button-1>", self._on_mouse_click)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/Linux
        self.bind("<Button-4>", self._on_mouse_wheel)    # Linux
        self.bind("<Button-5>", self._on_mouse_wheel)    # Linux
        self.bind("<Motion>", self._on_mouse_move)
        
        # Focus for keyboard events
        self.focus_set()
        self.bind("<Key>", self._on_key_press)
    
    def _physical_to_canvas(self, x_physical: float, y_physical: float) -> Tuple[int, int]:
        """Convert physical coordinates (meters) to canvas pixels."""
        scale = self._pixels_per_meter * self._zoom
        
        # Canvas center
        canvas_center_x = self.config.width // 2
        canvas_center_y = self.config.height // 2
        
        # Apply transformations
        canvas_x = canvas_center_x + (x_physical - self._pan_x) * scale
        canvas_y = canvas_center_y - (y_physical - self._pan_y) * scale  # Flip Y
        
        return int(canvas_x), int(canvas_y)
    
    def _canvas_to_physical(self, canvas_x: int, canvas_y: int) -> Tuple[float, float]:
        """Convert canvas pixels to physical coordinates (meters)."""
        scale = self._pixels_per_meter * self._zoom
        
        canvas_center_x = self.config.width // 2
        canvas_center_y = self.config.height // 2
        
        # Apply inverse transformations
        physical_x = self._pan_x + (canvas_x - canvas_center_x) / scale
        physical_y = self._pan_y - (canvas_y - canvas_center_y) / scale  # Flip Y
        
        return physical_x, physical_y
    
    def _snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to grid if enabled."""
        if not self.config.snap_to_grid:
            return x, y
        
        grid_size = self.config.grid_size
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size
        
        return snapped_x, snapped_y
    
    def _draw_all(self):
        """Redraw all canvas layers."""
        self.delete("all")
        
        # Draw in layer order
        if self.config.layer_visibility[LayerType.GRID] and self.config.show_grid:
            self._draw_grid()
        
        if self.config.layer_visibility[LayerType.RULERS] and self.config.show_rulers:
            self._draw_rulers()
        
        if self.config.layer_visibility[LayerType.PLATE] and self._current_plate:
            self._draw_plate_internal()
        
        if self.config.layer_visibility[LayerType.HUMAN] and self._current_person:
            self._draw_human_internal()
        
        if self.config.layer_visibility[LayerType.MODES] and self._mode_data is not None:
            self._draw_modes_internal()
        
        if self.config.layer_visibility[LayerType.EXCITERS] and self._exciters:
            self._draw_exciters_internal()
    
    def _draw_grid(self):
        """Draw coordinate grid."""
        grid_size = self.config.grid_size
        
        # Calculate grid bounds in physical coordinates
        top_left_phys = self._canvas_to_physical(0, 0)
        bottom_right_phys = self._canvas_to_physical(self.config.width, self.config.height)
        
        # Grid line spacing
        x_start = math.floor(top_left_phys[0] / grid_size) * grid_size
        x_end = math.ceil(bottom_right_phys[0] / grid_size) * grid_size
        y_start = math.floor(bottom_right_phys[1] / grid_size) * grid_size
        y_end = math.ceil(top_left_phys[1] / grid_size) * grid_size
        
        # Vertical lines
        x = x_start
        while x <= x_end:
            x1, y1 = self._physical_to_canvas(x, y_start)
            x2, y2 = self._physical_to_canvas(x, y_end)
            
            color = STYLE.GOLD if x == 0 else STYLE.BG_LIGHT
            width = 2 if x == 0 else 1
            
            self.create_line(
                x1, y1, x2, y2,
                fill=color, width=width,
                tags="grid"
            )
            x += grid_size
        
        # Horizontal lines
        y = y_start
        while y <= y_end:
            x1, y1 = self._physical_to_canvas(x_start, y)
            x2, y2 = self._physical_to_canvas(x_end, y)
            
            color = STYLE.GOLD if y == 0 else STYLE.BG_LIGHT
            width = 2 if y == 0 else 1
            
            self.create_line(
                x1, y1, x2, y2,
                fill=color, width=width,
                tags="grid"
            )
            y += grid_size
    
    def _draw_rulers(self):
        """Draw measurement rulers."""
        ruler_size = self.config.ruler_size
        
        # Top ruler (X-axis)
        self.create_rectangle(
            0, 0, self.config.width, ruler_size,
            fill=STYLE.BG_MEDIUM, outline=STYLE.BG_LIGHT,
            tags="ruler"
        )
        
        # Left ruler (Y-axis)
        self.create_rectangle(
            0, 0, ruler_size, self.config.height,
            fill=STYLE.BG_MEDIUM, outline=STYLE.BG_LIGHT,
            tags="ruler"
        )
        
        # Add tick marks and labels
        self._draw_ruler_ticks()
    
    def _draw_ruler_ticks(self):
        """Draw ruler tick marks and labels."""
        ruler_size = self.config.ruler_size
        
        # Calculate tick spacing
        scale = self._pixels_per_meter * self._zoom
        
        # Determine appropriate tick interval
        if scale > 200:  # Zoomed in
            tick_interval = 0.1  # 10 cm
        elif scale > 50:
            tick_interval = 0.5  # 50 cm
        else:  # Zoomed out
            tick_interval = 1.0  # 1 m
        
        # X-axis (top ruler)
        for x_canvas in range(ruler_size, self.config.width, max(1, int(scale * tick_interval))):
            x_physical, _ = self._canvas_to_physical(x_canvas, 0)
            
            # Tick mark
            self.create_line(
                x_canvas, ruler_size - 5, x_canvas, ruler_size,
                fill=STYLE.TEXT_SECONDARY, width=1,
                tags="ruler"
            )
            
            # Label
            if abs(x_physical) < 100:  # Reasonable range
                self.create_text(
                    x_canvas, ruler_size - 8,
                    text=f"{x_physical:.1f}",
                    fill=STYLE.TEXT_SECONDARY,
                    font=(STYLE.FONT_MONO, 8),
                    anchor="s",
                    tags="ruler"
                )
        
        # Y-axis (left ruler) - similar logic
        for y_canvas in range(ruler_size, self.config.height, max(1, int(scale * tick_interval))):
            _, y_physical = self._canvas_to_physical(0, y_canvas)
            
            # Tick mark
            self.create_line(
                ruler_size - 5, y_canvas, ruler_size, y_canvas,
                fill=STYLE.TEXT_SECONDARY, width=1,
                tags="ruler"
            )
            
            # Label
            if abs(y_physical) < 100:
                self.create_text(
                    ruler_size - 8, y_canvas,
                    text=f"{y_physical:.1f}",
                    fill=STYLE.TEXT_SECONDARY,
                    font=(STYLE.FONT_MONO, 8),
                    anchor="e",
                    tags="ruler"
                )
    
    def _draw_plate_internal(self):
        """Draw the current plate geometry."""
        if not self._current_plate:
            return
        
        plate = self._current_plate
        
        # Calculate plate corners
        half_length = plate.length / 2
        half_width = plate.width / 2
        
        # Convert to canvas coordinates
        corners = [
            self._physical_to_canvas(-half_width, -half_length),
            self._physical_to_canvas(half_width, -half_length),
            self._physical_to_canvas(half_width, half_length),
            self._physical_to_canvas(-half_width, half_length)
        ]
        
        # Draw based on contour type
        if plate.contour_type == ContourType.ELLIPSE:
            # Ellipse
            x1, y1 = self._physical_to_canvas(-half_width, -half_length)
            x2, y2 = self._physical_to_canvas(half_width, half_length)
            
            # Heatmap overlay if available
            if self._heatmap_data is not None:
                self._draw_heatmap(x1, y1, x2, y2)
            
            self.create_oval(
                x1, y1, x2, y2,
                fill=STYLE.PLATE_FILL if self._heatmap_data is None else "",
                outline=STYLE.PLATE_STROKE,
                width=3,
                tags="plate"
            )
        else:
            # Rectangle
            flat_corners = [coord for corner in corners for coord in corner]
            
            # Heatmap overlay if available
            if self._heatmap_data is not None:
                x1, y1 = corners[0]
                x2, y2 = corners[2]
                self._draw_heatmap(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            self.create_polygon(
                flat_corners,
                fill=STYLE.PLATE_FILL if self._heatmap_data is None else "",
                outline=STYLE.PLATE_STROKE,
                width=3,
                tags="plate"
            )
        
        # Add center point
        center_x, center_y = self._physical_to_canvas(0, 0)
        self.create_oval(
            center_x - 3, center_y - 3, center_x + 3, center_y + 3,
            fill=STYLE.GOLD, outline=STYLE.GOLD_LIGHT,
            tags="plate"
        )
    
    def _draw_heatmap(self, x1: int, y1: int, x2: int, y2: int):
        """Draw heatmap overlay on plate (simplified version)."""
        # This is a simplified heatmap - in practice you'd use PIL for better quality
        if self._heatmap_data is None:
            return
        
        # Create a simple gradient effect
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Sample heatmap at regular intervals
        resolution = 20
        for i in range(resolution):
            for j in range(resolution):
                # Interpolate heatmap value
                heat_value = self._heatmap_data[i * len(self._heatmap_data) // resolution,
                                               j * len(self._heatmap_data[0]) // resolution]
                
                # Convert to color (blue to red scale)
                normalized = max(0, min(1, heat_value))
                red = int(255 * normalized)
                blue = int(255 * (1 - normalized))
                color = f"#{red:02x}00{blue:02x}"
                
                # Draw small rectangle
                rect_x1 = x1 + i * width // resolution
                rect_y1 = y1 + j * height // resolution
                rect_x2 = rect_x1 + width // resolution
                rect_y2 = rect_y1 + height // resolution
                
                self.create_rectangle(
                    rect_x1, rect_y1, rect_x2, rect_y2,
                    fill=color, outline="",
                    tags="heatmap"
                )
    
    def _draw_human_internal(self):
        """Draw human body overlay."""
        if not self._current_person or not self._current_person.visible:
            return
        
        person = self._current_person
        
        # Human body center position
        body_x = person.position[0]
        body_y = person.position[1]
        
        # Simplified human outline (ellipse)
        half_width = person.width / 2
        half_height = person.height / 2
        
        x1, y1 = self._physical_to_canvas(body_x - half_width, body_y - half_height)
        x2, y2 = self._physical_to_canvas(body_x + half_width, body_y + half_height)
        
        self.create_oval(
            x1, y1, x2, y2,
            fill=STYLE.BODY_OVERLAY,
            outline=STYLE.TEXT_SECONDARY,
            width=2, dash=(5, 5),
            tags="human"
        )
        
        # Add head circle
        head_radius = 0.1  # meters
        head_y = body_y + half_height + head_radius
        
        hx1, hy1 = self._physical_to_canvas(body_x - head_radius, head_y - head_radius)
        hx2, hy2 = self._physical_to_canvas(body_x + head_radius, head_y + head_radius)
        
        self.create_oval(
            hx1, hy1, hx2, hy2,
            fill=STYLE.BODY_OVERLAY,
            outline=STYLE.TEXT_SECONDARY,
            width=2, dash=(5, 5),
            tags="human"
        )
    
    def _draw_modes_internal(self):
        """Draw mode shapes (simplified visualization)."""
        # This would typically show modal displacement patterns
        # For now, just indicate mode positions
        pass
    
    def _draw_exciters_internal(self):
        """Draw exciter/transducer positions."""
        for i, exciter in enumerate(self._exciters):
            x, y = self._physical_to_canvas(exciter.x, exciter.y)
            
            # Exciter circle
            radius_canvas = exciter.radius * self._pixels_per_meter * self._zoom
            
            color = STYLE.EXCITER_FILL if exciter.active else STYLE.TEXT_MUTED
            
            self.create_oval(
                x - radius_canvas, y - radius_canvas,
                x + radius_canvas, y + radius_canvas,
                fill=color,
                outline=STYLE.TEXT_PRIMARY,
                width=2,
                tags="exciter"
            )
            
            # Label
            if exciter.label:
                self.create_text(
                    x, y + radius_canvas + 10,
                    text=exciter.label,
                    fill=STYLE.TEXT_PRIMARY,
                    font=STYLE.font_small,
                    anchor="n",
                    tags="exciter"
                )
    
    def _on_mouse_click(self, event):
        """Handle mouse click."""
        self._drag_start = (event.x, event.y)
        physical_coords = self._canvas_to_physical(event.x, event.y)
        self._last_click_physical = physical_coords
        
        if self._on_click:
            snapped_coords = self._snap_to_grid(*physical_coords)
            self._on_click(*snapped_coords)
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag (pan)."""
        if not self._dragging:
            self._dragging = True
        
        # Calculate pan delta
        dx_canvas = event.x - self._drag_start[0]
        dy_canvas = event.y - self._drag_start[1]
        
        # Convert to physical coordinates
        scale = self._pixels_per_meter * self._zoom
        dx_physical = -dx_canvas / scale
        dy_physical = dy_canvas / scale  # Flip Y
        
        self._pan_x += dx_physical
        self._pan_y += dy_physical
        
        # Update drag start
        self._drag_start = (event.x, event.y)
        
        # Redraw
        self._draw_all()
        
        if self._on_drag:
            self._on_drag(dx_physical, dy_physical, self._pan_x, self._pan_y)
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        self._dragging = False
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel (zoom)."""
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Zoom in
            zoom_factor = 1.1
        else:  # Zoom out
            zoom_factor = 0.9
        
        # Calculate new zoom
        new_zoom = self._zoom * zoom_factor
        new_zoom = max(self.config.min_zoom, min(self.config.max_zoom, new_zoom))
        
        if new_zoom != self._zoom:
            # Zoom towards mouse cursor
            mouse_physical = self._canvas_to_physical(event.x, event.y)
            
            self._zoom = new_zoom
            
            # Adjust pan to keep mouse position constant
            new_mouse_physical = self._canvas_to_physical(event.x, event.y)
            self._pan_x += mouse_physical[0] - new_mouse_physical[0]
            self._pan_y += mouse_physical[1] - new_mouse_physical[1]
            
            self._draw_all()
    
    def _on_mouse_move(self, event):
        """Handle mouse movement (for cursor updates, etc.)."""
        # Could show coordinates in status bar
        pass
    
    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.char == 'r':  # Reset view
            self.reset_view()
        elif event.char == 'g':  # Toggle grid
            self.toggle_layer(LayerType.GRID)
        elif event.char == 'h':  # Toggle human overlay
            self.toggle_layer(LayerType.HUMAN)
    
    # Public API methods
    
    def draw_plate(self, genome: PlateGenome, heatmap: Optional[np.ndarray] = None):
        """Draw plate with optional heatmap overlay."""
        self._current_plate = genome
        self._heatmap_data = heatmap
        self._draw_all()
    
    def draw_human_overlay(self, person: Person, opacity: float = 0.3):
        """Draw human body overlay."""
        self._current_person = person
        self._draw_all()
    
    def draw_mode_shape(self, mode_idx: int, modes: np.ndarray):
        """Draw modal displacement pattern."""
        self._mode_data = modes
        self._draw_all()
    
    def add_exciter(self, x: float, y: float, radius: float = 0.05, label: str = ""):
        """Add exciter at physical coordinates."""
        exciter = Exciter(x=x, y=y, radius=radius, label=label)
        self._exciters.append(exciter)
        self._draw_all()
    
    def clear_exciters(self):
        """Remove all exciters."""
        self._exciters.clear()
        self._draw_all()
    
    def set_zoom(self, zoom: float):
        """Set zoom level."""
        self._zoom = max(self.config.min_zoom, min(self.config.max_zoom, zoom))
        self._draw_all()
    
    def set_pan(self, x: float, y: float):
        """Set pan offset in physical coordinates."""
        self._pan_x = x
        self._pan_y = y
        self._draw_all()
    
    def reset_view(self):
        """Reset zoom and pan to default."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._draw_all()
    
    def toggle_layer(self, layer: LayerType):
        """Toggle layer visibility."""
        self.config.layer_visibility[layer] = not self.config.layer_visibility[layer]
        self._draw_all()
    
    def fit_to_plate(self):
        """Adjust view to fit current plate."""
        if not self._current_plate:
            return
        
        # Calculate required zoom to fit plate
        plate = self._current_plate
        canvas_width = self.config.width - (self.config.ruler_size if self.config.show_rulers else 0)
        canvas_height = self.config.height - (self.config.ruler_size if self.config.show_rulers else 0)
        
        zoom_x = canvas_width * 0.8 / (plate.width * self._pixels_per_meter)
        zoom_y = canvas_height * 0.8 / (plate.length * self._pixels_per_meter)
        
        # Use smaller zoom to fit both dimensions
        self._zoom = min(zoom_x, zoom_y)
        self._zoom = max(self.config.min_zoom, min(self.config.max_zoom, self._zoom))
        
        # Center on plate
        self._pan_x = 0.0
        self._pan_y = 0.0
        
        self._draw_all()
    
    def get_view_bounds(self) -> Tuple[float, float, float, float]:
        """Get current view bounds in physical coordinates."""
        top_left = self._canvas_to_physical(0, 0)
        bottom_right = self._canvas_to_physical(self.config.width, self.config.height)
        
        return top_left[0], top_left[1], bottom_right[0], bottom_right[1]
