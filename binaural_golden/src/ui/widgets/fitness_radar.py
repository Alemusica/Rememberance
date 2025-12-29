"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           FITNESS RADAR CHART                               ║
║                                                                              ║
║   Radar/spider chart for visualizing fitness scores with smooth animation   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import math

from ..theme import PlateLabStyle

STYLE = PlateLabStyle()


@dataclass
class RadarConfig:
    """Configuration for FitnessRadar."""
    size: int = 200
    axes_labels: List[str] = None
    max_value: float = 1.0
    grid_levels: int = 5
    animated: bool = True
    show_grid: bool = True
    show_labels: bool = True
    show_values: bool = True
    
    def __post_init__(self):
        if self.axes_labels is None:
            self.axes_labels = ["Flatness", "Spine", "Mass", "Coupling", "Total"]


class FitnessRadar(tk.Canvas):
    """
    Radar/spider chart per visualizzare fitness scores.
    
    Features:
    - Visualizzazione Flatness, Spine, Mass, Coupling, Total
    - Animazione smooth su update valori
    - Tooltip su hover con dettagli
    - Grid personalizzabile
    - Supporto per confronti multipli
    
    Example:
        radar = FitnessRadar(
            parent,
            config=RadarConfig(size=250, animated=True)
        )
        radar.update_scores({
            "Flatness": 0.85,
            "Spine": 0.70,
            "Mass": 0.90,
            "Coupling": 0.65,
            "Total": 0.77
        })
    """
    
    def __init__(self, parent, *, config: Optional[RadarConfig] = None,
                 on_axis_click: Optional[Callable[[str], None]] = None):
        self.config = config or RadarConfig()
        
        super().__init__(
            parent,
            width=self.config.size,
            height=self.config.size,
            bg=STYLE.BG_DARK,
            highlightthickness=0
        )
        
        self._scores = {label: 0.0 for label in self.config.axes_labels}
        self._target_scores = self._scores.copy()
        self._animation_id = None
        self._on_axis_click = on_axis_click
        
        # Geometry calculations
        self.center = self.config.size // 2
        self.radius = self.center - 30  # Leave space for labels
        self.num_axes = len(self.config.axes_labels)
        self.angle_step = 2 * math.pi / self.num_axes
        
        # Tooltip
        self._tooltip = None
        self._hover_axis = None
        
        # Bind events
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Leave>", self._on_mouse_leave)
        
        # Initial draw
        self._draw()
    
    def _get_axis_position(self, axis_index: int, radius_ratio: float = 1.0) -> Tuple[float, float]:
        """Get x, y position for axis point."""
        angle = axis_index * self.angle_step - math.pi / 2  # Start at top
        x = self.center + radius_ratio * self.radius * math.cos(angle)
        y = self.center + radius_ratio * self.radius * math.sin(angle)
        return x, y
    
    def _draw(self):
        """Draw the complete radar chart."""
        self.delete("all")
        
        if self.config.show_grid:
            self._draw_grid()
        
        self._draw_data()
        
        if self.config.show_labels:
            self._draw_labels()
    
    def _draw_grid(self):
        """Draw radar grid lines and circles."""
        # Grid circles
        for level in range(1, self.config.grid_levels + 1):
            radius = self.radius * (level / self.config.grid_levels)
            
            self.create_oval(
                self.center - radius, self.center - radius,
                self.center + radius, self.center + radius,
                outline=STYLE.BG_LIGHT, width=1,
                tags="grid"
            )
            
            # Grid value labels
            value = self.config.max_value * (level / self.config.grid_levels)
            self.create_text(
                self.center + radius, self.center - 5,
                text=f"{value:.2f}",
                fill=STYLE.TEXT_MUTED,
                font=STYLE.font_small,
                anchor="w",
                tags="grid"
            )
        
        # Grid lines (axes)
        for i in range(self.num_axes):
            x, y = self._get_axis_position(i)
            
            self.create_line(
                self.center, self.center, x, y,
                fill=STYLE.BG_LIGHT, width=1,
                tags="grid"
            )
    
    def _draw_data(self):
        """Draw the data polygon and points."""
        if not any(self._scores.values()):
            return
        
        # Calculate points
        points = []
        for i, (label, score) in enumerate(self._scores.items()):
            ratio = score / self.config.max_value
            x, y = self._get_axis_position(i, ratio)
            points.extend([x, y])
        
        if len(points) >= 6:  # At least 3 points
            # Fill polygon
            self.create_polygon(
                points,
                fill=STYLE.GOLD + "40",  # Semi-transparent
                outline=STYLE.GOLD,
                width=2,
                tags="data"
            )
            
            # Draw points
            for i in range(0, len(points), 2):
                x, y = points[i], points[i + 1]
                
                self.create_oval(
                    x - 4, y - 4, x + 4, y + 4,
                    fill=STYLE.GOLD,
                    outline=STYLE.GOLD_LIGHT,
                    width=2,
                    tags="data_point"
                )
    
    def _draw_labels(self):
        """Draw axis labels."""
        for i, label in enumerate(self.config.axes_labels):
            # Label position (slightly outside radar)
            x, y = self._get_axis_position(i, 1.3)
            
            # Adjust anchor based on position
            angle = i * self.angle_step - math.pi / 2
            if -math.pi/4 <= angle <= math.pi/4:  # Top
                anchor = "s"
            elif 3*math.pi/4 <= angle <= 5*math.pi/4:  # Bottom
                anchor = "n"
            elif angle > 0:  # Right side
                anchor = "w"
            else:  # Left side
                anchor = "e"
            
            # Label text
            self.create_text(
                x, y,
                text=label,
                fill=STYLE.TEXT_PRIMARY,
                font=STYLE.font_small,
                anchor=anchor,
                tags=f"label_{i}"
            )
            
            # Value text if enabled
            if self.config.show_values:
                value_text = f"{self._scores[label]:.3f}"
                offset = 12 if anchor in ["n", "s"] else 15
                
                value_x = x + (offset if anchor == "w" else -offset if anchor == "e" else 0)
                value_y = y + (offset if anchor == "n" else -offset if anchor == "s" else 0)
                
                self.create_text(
                    value_x, value_y,
                    text=value_text,
                    fill=STYLE.TEXT_SECONDARY,
                    font=(STYLE.FONT_MONO, STYLE.FONT_SIZE_XS),
                    anchor=anchor,
                    tags=f"value_{i}"
                )
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for hover effects."""
        # Find nearest axis
        min_dist = float('inf')
        hover_axis = None
        
        for i, label in enumerate(self.config.axes_labels):
            x, y = self._get_axis_position(i, 1.3)
            dist = math.sqrt((event.x - x)**2 + (event.y - y)**2)
            
            if dist < 30 and dist < min_dist:  # Within hover range
                min_dist = dist
                hover_axis = i
        
        if hover_axis != self._hover_axis:
            self._hover_axis = hover_axis
            self._show_tooltip(event.x, event.y, hover_axis)
    
    def _show_tooltip(self, x: int, y: int, axis_index: Optional[int]):
        """Show/hide tooltip."""
        if self._tooltip:
            self.delete("tooltip")
        
        if axis_index is not None:
            label = self.config.axes_labels[axis_index]
            score = self._scores[label]
            percentage = (score / self.config.max_value) * 100
            
            tooltip_text = f"{label}\n{score:.3f} ({percentage:.1f}%)"
            
            # Tooltip background
            self.create_rectangle(
                x - 40, y - 25, x + 40, y + 5,
                fill=STYLE.BG_MEDIUM,
                outline=STYLE.GOLD,
                width=1,
                tags="tooltip"
            )
            
            # Tooltip text
            self.create_text(
                x, y - 10,
                text=tooltip_text,
                fill=STYLE.TEXT_PRIMARY,
                font=STYLE.font_small,
                justify="center",
                tags="tooltip"
            )
    
    def _on_click(self, event):
        """Handle click events."""
        if self._hover_axis is not None and self._on_axis_click:
            label = self.config.axes_labels[self._hover_axis]
            self._on_axis_click(label)
    
    def _on_mouse_leave(self, event):
        """Handle mouse leaving widget."""
        self._hover_axis = None
        if self._tooltip:
            self.delete("tooltip")
    
    def _animate_step(self):
        """Single animation step."""
        should_continue = False
        
        # Smooth transition for each score
        for label in self._scores:
            current = self._scores[label]
            target = self._target_scores[label]
            
            if abs(current - target) > 0.001:
                diff = target - current
                self._scores[label] += diff * 0.15  # Smooth easing
                should_continue = True
            else:
                self._scores[label] = target
        
        # Redraw
        self._draw()
        
        # Continue animation if needed
        if should_continue and self.config.animated:
            self._animation_id = self.after(50, self._animate_step)
        else:
            self._animation_id = None
    
    def update_scores(self, scores: Dict[str, float]):
        """Update radar chart with new scores."""
        # Validate and update target scores
        for label, score in scores.items():
            if label in self._target_scores:
                self._target_scores[label] = max(0, min(self.config.max_value, score))
        
        # Start animation if enabled
        if self.config.animated and not self._animation_id:
            self._animate_step()
        else:
            # Immediate update
            self._scores.update(self._target_scores)
            self._draw()
    
    def reset(self):
        """Reset all scores to zero."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        
        self._scores = {label: 0.0 for label in self.config.axes_labels}
        self._target_scores = self._scores.copy()
        self._draw()
    
    def destroy(self):
        """Clean up animation before destroying."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
        super().destroy()
