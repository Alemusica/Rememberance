"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  Radar Plot Widget for Optimization Parameters                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Interactive radar/spider chart for tuning:                                   ║
║  - Energy: Vibration intensity for body coupling                             ║
║  - Flatness: Frequency response uniformity                                   ║
║  - Spine: Coverage of spinal column zone                                     ║
║                                                                               ║
║  Draggable vertices for intuitive parameter adjustment.                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import math


class RadarWidget(tk.Canvas):
    """
    Interactive radar/spider chart widget.
    
    Features:
    - 3 draggable vertices (Energy, Flatness, Spine)
    - Visual feedback on hover
    - Smooth animations
    - Callback on value change
    """
    
    # Style constants
    COLORS = {
        'bg': '#1a1a2e',
        'grid': '#2d2d4a',
        'grid_accent': '#3d3d5c',
        'fill': '#4a90a4',
        'fill_alpha': '#4a90a480',
        'stroke': '#7fdbff',
        'text': '#e8e8e8',
        'vertex': '#ffd700',
        'vertex_hover': '#ff6b6b',
        'label_energy': '#ff6b6b',
        'label_flatness': '#4ecdc4',
        'label_spine': '#ffe66d'
    }
    
    PARAMETERS = ['Energy', 'Flatness', 'Spine']
    
    def __init__(
        self,
        parent,
        size: int = 180,
        on_change: Optional[Callable[[Dict[str, float]], None]] = None,
        **kwargs
    ):
        """
        Args:
            parent: Parent widget
            size: Widget size in pixels
            on_change: Callback when values change, receives dict of values [0-1]
        """
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=self.COLORS['bg'],
            highlightthickness=0,
            **kwargs
        )
        
        self.size = size
        self.center = size // 2
        self.radius = int(size * 0.38)
        self.on_change = on_change
        
        # Values [0-1] for each parameter
        self.values = {
            'Energy': 0.7,
            'Flatness': 0.5,
            'Spine': 0.8
        }
        
        # Dragging state
        self._dragging: Optional[str] = None
        self._hover: Optional[str] = None
        
        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Motion>', self._on_motion)
        self.bind('<Leave>', self._on_leave)
        
        # Initial draw
        self._draw()
    
    def _get_angles(self) -> List[float]:
        """Get angles for each axis (evenly spaced, starting from top)."""
        n = len(self.PARAMETERS)
        return [np.pi/2 + 2*np.pi*i/n for i in range(n)]
    
    def _value_to_point(self, param: str, value: float) -> Tuple[int, int]:
        """Convert value [0-1] to canvas coordinates."""
        idx = self.PARAMETERS.index(param)
        angle = self._get_angles()[idx]
        r = value * self.radius
        x = self.center + r * np.cos(angle)
        y = self.center - r * np.sin(angle)  # Y is inverted in canvas
        return int(x), int(y)
    
    def _point_to_value(self, x: int, y: int, param: str) -> float:
        """Convert canvas coordinates to value [0-1] for given parameter."""
        idx = self.PARAMETERS.index(param)
        angle = self._get_angles()[idx]
        
        # Calculate distance from center along the parameter axis
        dx = x - self.center
        dy = self.center - y  # Y is inverted
        
        # Project onto axis
        axis_x = np.cos(angle)
        axis_y = np.sin(angle)
        
        projection = dx * axis_x + dy * axis_y
        value = projection / self.radius
        
        return np.clip(value, 0.0, 1.0)
    
    def _get_vertex_at(self, x: int, y: int, threshold: int = 12) -> Optional[str]:
        """Get parameter name if point is near a vertex."""
        for param in self.PARAMETERS:
            vx, vy = self._value_to_point(param, self.values[param])
            dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            if dist < threshold:
                return param
        return None
    
    def _draw(self):
        """Redraw the entire radar chart."""
        self.delete('all')
        
        # Draw grid circles
        for level in [0.25, 0.5, 0.75, 1.0]:
            r = level * self.radius
            self.create_oval(
                self.center - r, self.center - r,
                self.center + r, self.center + r,
                outline=self.COLORS['grid'] if level < 1.0 else self.COLORS['grid_accent'],
                width=1
            )
        
        # Draw axis lines and labels
        angles = self._get_angles()
        label_colors = [
            self.COLORS['label_energy'],
            self.COLORS['label_flatness'],
            self.COLORS['label_spine']
        ]
        
        for i, (param, angle, color) in enumerate(zip(self.PARAMETERS, angles, label_colors)):
            # Axis line
            x_end = self.center + self.radius * np.cos(angle)
            y_end = self.center - self.radius * np.sin(angle)
            self.create_line(
                self.center, self.center, x_end, y_end,
                fill=self.COLORS['grid'], width=1
            )
            
            # Label
            label_r = self.radius + 18
            lx = self.center + label_r * np.cos(angle)
            ly = self.center - label_r * np.sin(angle)
            
            # Adjust anchor based on position
            if angle > np.pi/4 and angle < 3*np.pi/4:
                anchor = 's'
            elif angle > 3*np.pi/4 or angle < -3*np.pi/4:
                anchor = 'e'
            elif angle < -np.pi/4 or angle > 5*np.pi/4:
                anchor = 'w'
            else:
                anchor = 'n'
            
            self.create_text(
                lx, ly,
                text=param,
                fill=color,
                font=('SF Pro', 9, 'bold'),
                anchor=anchor
            )
            
            # Value label
            val_r = self.radius * self.values[param] + 12
            val_x = self.center + val_r * np.cos(angle)
            val_y = self.center - val_r * np.sin(angle)
            self.create_text(
                val_x, val_y,
                text=f"{int(self.values[param] * 100)}%",
                fill=self.COLORS['text'],
                font=('SF Pro', 7),
                anchor='center'
            )
        
        # Draw filled polygon
        points = []
        for param in self.PARAMETERS:
            x, y = self._value_to_point(param, self.values[param])
            points.extend([x, y])
        
        # Fill
        self.create_polygon(
            points,
            fill=self.COLORS['fill'],
            outline=self.COLORS['stroke'],
            width=2,
            stipple='gray50'  # Semi-transparent effect
        )
        
        # Draw vertices (draggable points)
        for param in self.PARAMETERS:
            x, y = self._value_to_point(param, self.values[param])
            
            is_hover = param == self._hover
            is_drag = param == self._dragging
            
            color = self.COLORS['vertex_hover'] if (is_hover or is_drag) else self.COLORS['vertex']
            size = 8 if (is_hover or is_drag) else 6
            
            self.create_oval(
                x - size, y - size,
                x + size, y + size,
                fill=color,
                outline='white',
                width=2 if (is_hover or is_drag) else 1
            )
        
        # Title
        self.create_text(
            self.center, self.size - 8,
            text="Optimization Priority",
            fill=self.COLORS['text'],
            font=('SF Pro', 8)
        )
    
    def _on_click(self, event):
        """Handle mouse click."""
        param = self._get_vertex_at(event.x, event.y)
        if param:
            self._dragging = param
    
    def _on_drag(self, event):
        """Handle mouse drag."""
        if self._dragging:
            new_value = self._point_to_value(event.x, event.y, self._dragging)
            self.values[self._dragging] = new_value
            self._draw()
            
            if self.on_change:
                self.on_change(self.values.copy())
    
    def _on_release(self, event):
        """Handle mouse release."""
        self._dragging = None
        self._draw()
    
    def _on_motion(self, event):
        """Handle mouse motion for hover effect."""
        new_hover = self._get_vertex_at(event.x, event.y)
        if new_hover != self._hover:
            self._hover = new_hover
            self._draw()
            
            # Change cursor
            self.config(cursor='hand2' if new_hover else '')
    
    def _on_leave(self, event):
        """Handle mouse leaving widget."""
        if self._hover:
            self._hover = None
            self._draw()
        self.config(cursor='')
    
    def get_values(self) -> Dict[str, float]:
        """Get current parameter values."""
        return self.values.copy()
    
    def set_values(self, values: Dict[str, float]):
        """Set parameter values and redraw."""
        for param, value in values.items():
            if param in self.values:
                self.values[param] = np.clip(value, 0.0, 1.0)
        self._draw()


class OptimizationRadarFrame(ttk.Frame):
    """
    Complete frame with radar widget and info labels.
    """
    
    def __init__(
        self,
        parent,
        on_change: Optional[Callable[[Dict[str, float]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.on_change = on_change
        
        # Radar widget
        self._radar = RadarWidget(
            self,
            size=180,
            on_change=self._on_radar_change
        )
        self._radar.pack(pady=5)
        
        # Info labels
        self._info_frame = ttk.Frame(self)
        self._info_frame.pack(fill='x', padx=5)
        
        self._info_labels: Dict[str, ttk.Label] = {}
        
        descriptions = {
            'Energy': 'Vibration intensity',
            'Flatness': 'Freq uniformity',
            'Spine': 'Spinal coverage'
        }
        
        for i, (param, desc) in enumerate(descriptions.items()):
            label = ttk.Label(
                self._info_frame,
                text=f"• {param}: {desc}",
                font=("SF Pro", 7),
                foreground='gray'
            )
            label.pack(anchor='w')
            self._info_labels[param] = label
    
    def _on_radar_change(self, values: Dict[str, float]):
        """Handle radar value changes."""
        if self.on_change:
            self.on_change(values)
    
    def get_values(self) -> Dict[str, float]:
        """Get current optimization parameters."""
        return self._radar.get_values()
    
    def set_values(self, values: Dict[str, float]):
        """Set optimization parameters."""
        self._radar.set_values(values)


# Quick test
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Radar Widget Test")
    root.configure(bg='#1a1a2e')
    
    def on_change(values):
        print(f"Values changed: {values}")
    
    frame = OptimizationRadarFrame(root, on_change=on_change)
    frame.pack(padx=20, pady=20)
    
    root.mainloop()
