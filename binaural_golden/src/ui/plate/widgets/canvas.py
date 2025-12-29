"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE CANVAS - Visualization Widget                       ║
║                                                                              ║
║   Renders:                                                                   ║
║   • Mode shape heatmap (red=antinode, blue=node)                            ║
║   • Human body overlay with body zones                                       ║
║   • Draggable exciters                                                       ║
║   • Nodal lines                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Tuple
import numpy as np

# Import state
from ..viewmodel import PlateState


# Colors
COLORS = {
    'bg': '#1a1a2e',
    'plate': '#2d2d44',
    'antinode': '#ff4444',  # Red = max displacement
    'node': '#4444ff',       # Blue = zero displacement
    'human': '#88888844',
    'exciter_active': '#ffaa00',
    'exciter_inactive': '#666666',
    'nodal_line': '#ffffff44',
    'text': '#ffffff',
}


class PlateCanvas(ttk.Frame):
    """
    Canvas widget for plate visualization.
    
    Features:
    - Mode shape heatmap with color gradient
    - Draggable exciter markers
    - Human body overlay
    - Responsive layout
    """
    
    def __init__(
        self,
        parent,
        on_exciter_moved: Callable[[int, float, float], None] = None,
        on_exciter_added: Callable[[float, float], None] = None,
    ):
        super().__init__(parent)
        
        # Callbacks
        self._on_exciter_moved = on_exciter_moved
        self._on_exciter_added = on_exciter_added
        
        # Canvas
        self._canvas = tk.Canvas(
            self,
            bg=COLORS['bg'],
            highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        
        # State
        self._state: Optional[PlateState] = None
        self._dragging_exciter: Optional[int] = None
        
        # Cached heatmap
        self._heatmap_image = None
        self._heatmap_photo = None
        
        # Bind events
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
    
    def update_state(self, state: PlateState):
        """Update visualization from state."""
        self._state = state
        self._redraw()
    
    def _redraw(self):
        """Redraw everything."""
        self._canvas.delete("all")
        
        if self._state is None:
            return
        
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            return
        
        # Calculate plate rectangle with margins
        margin = 40
        plate_w = cw - 2 * margin
        plate_h = ch - 2 * margin
        
        # Maintain aspect ratio
        plate_aspect = self._state.length / self._state.width
        canvas_aspect = plate_w / plate_h
        
        if canvas_aspect > plate_aspect:
            # Canvas is wider - fit height
            plate_w = plate_h * plate_aspect
        else:
            # Canvas is taller - fit width
            plate_h = plate_w / plate_aspect
        
        # Center plate
        x0 = (cw - plate_w) / 2
        y0 = (ch - plate_h) / 2
        x1 = x0 + plate_w
        y1 = y0 + plate_h
        
        # Store for coordinate conversion
        self._plate_rect = (x0, y0, x1, y1)
        
        # Draw plate background
        self._canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=COLORS['plate'],
            outline="#ffffff33",
            width=2
        )
        
        # Draw heatmap if we have mode data
        if self._state.selected_mode:
            self._draw_heatmap(x0, y0, plate_w, plate_h)
        
        # Draw human overlay
        if self._state.show_human_overlay:
            self._draw_human(x0, y0, plate_w, plate_h)
        
        # Draw nodal lines
        if self._state.show_nodal_lines and self._state.selected_mode:
            self._draw_nodal_lines(x0, y0, plate_w, plate_h)
        
        # Draw exciters
        self._draw_exciters(x0, y0, plate_w, plate_h)
        
        # Draw labels
        self._draw_labels(x0, y0, plate_w, plate_h)
    
    def _draw_heatmap(self, x0: float, y0: float, w: float, h: float):
        """Draw mode shape heatmap using PIL for performance."""
        mode = self._state.selected_mode
        if mode is None:
            return
        
        try:
            # Import PIL lazily
            from PIL import Image, ImageTk
            
            # Get mode shape grid (small for performance)
            X, Y, Z = mode.get_displacement_grid(nx=40, ny=25)
            
            # Normalize Z to [-1, 1]
            z_norm = Z / (np.max(np.abs(Z)) + 1e-10)
            
            # Create RGB image directly
            ny, nx = Z.shape
            
            # Map to colors: -1 = blue, 0 = dark, 1 = red
            r = np.clip((z_norm + 1) * 127, 0, 255).astype(np.uint8)
            g = np.zeros_like(r, dtype=np.uint8)
            b = np.clip((1 - z_norm) * 127, 0, 255).astype(np.uint8)
            
            # Stack to RGB
            rgb = np.stack([r, g, b], axis=-1)
            
            # Create PIL image and resize
            img = Image.fromarray(rgb, mode='RGB')
            img = img.resize((int(w), int(h)), Image.Resampling.NEAREST)
            
            # Keep reference to prevent garbage collection
            self._heatmap_photo = ImageTk.PhotoImage(img)
            
            # Draw single image
            self._canvas.create_image(
                x0, y0,
                image=self._heatmap_photo,
                anchor='nw'
            )
            
        except ImportError:
            # Fallback: draw sparse grid without PIL
            self._draw_heatmap_fallback(x0, y0, w, h)
        except Exception as e:
            print(f"Heatmap error: {e}")
    
    def _draw_heatmap_fallback(self, x0: float, y0: float, w: float, h: float):
        """Fallback heatmap without PIL (sparse rectangles)."""
        mode = self._state.selected_mode
        if mode is None:
            return
            
        try:
            X, Y, Z = mode.get_displacement_grid(nx=20, ny=12)
            z_norm = Z / (np.max(np.abs(Z)) + 1e-10)
            ny, nx = Z.shape
            dx = w / nx
            dy = h / ny
            
            for i in range(ny):
                for j in range(nx):
                    val = z_norm[i, j]
                    if val > 0:
                        color = f"#{int(val * 200):02x}0000"
                    else:
                        color = f"#0000{int(-val * 200):02x}"
                    
                    self._canvas.create_rectangle(
                        x0 + j * dx, y0 + i * dy,
                        x0 + (j + 1) * dx, y0 + (i + 1) * dy,
                        fill=color, outline=""
                    )
        except Exception as e:
            print(f"Heatmap fallback error: {e}")
    
    def _draw_human(self, x0: float, y0: float, w: float, h: float):
        """Draw human body silhouette."""
        # Simplified human outline (lying down, head on right)
        # Normalized proportions based on golden ratio
        
        color = "#ffffff22"
        
        # Body parts as ellipses (x_center, y_center, rx, ry) in normalized coords
        parts = [
            (0.92, 0.5, 0.06, 0.08),   # Head
            (0.78, 0.5, 0.08, 0.12),   # Torso upper
            (0.60, 0.5, 0.10, 0.14),   # Torso middle
            (0.42, 0.5, 0.08, 0.10),   # Pelvis
            (0.25, 0.35, 0.12, 0.06),  # Left thigh
            (0.25, 0.65, 0.12, 0.06),  # Right thigh
            (0.08, 0.30, 0.08, 0.05),  # Left foot
            (0.08, 0.70, 0.08, 0.05),  # Right foot
        ]
        
        for px, py, rx, ry in parts:
            cx = x0 + px * w
            cy = y0 + py * h
            self._canvas.create_oval(
                cx - rx * w, cy - ry * h,
                cx + rx * w, cy + ry * h,
                fill=color,
                outline="#ffffff44"
            )
        
        # Body zone labels
        zones = [
            (0.08, 0.5, "Feet"),
            (0.42, 0.5, "Pelvis"),
            (0.70, 0.5, "Heart"),
            (0.92, 0.5, "Head"),
        ]
        
        for zx, zy, label in zones:
            tx = x0 + zx * w
            ty = y0 + zy * h + 30
            self._canvas.create_text(
                tx, ty,
                text=label,
                fill="#ffffff66",
                font=("SF Pro", 9)
            )
    
    def _draw_nodal_lines(self, x0: float, y0: float, w: float, h: float):
        """Draw nodal lines (zero displacement contours)."""
        mode = self._state.selected_mode
        if mode is None:
            return
        
        # For modes (m, n), draw approximate nodal lines
        m, n = mode.m or 1, mode.n or 1
        
        color = COLORS['nodal_line']
        
        # Vertical nodal lines
        for i in range(1, m):
            x = x0 + (i / m) * w
            self._canvas.create_line(
                x, y0, x, y0 + h,
                fill=color,
                dash=(4, 4)
            )
        
        # Horizontal nodal lines
        for j in range(1, n):
            y = y0 + (j / n) * h
            self._canvas.create_line(
                x0, y, x0 + w, y,
                fill=color,
                dash=(4, 4)
            )
    
    def _draw_exciters(self, x0: float, y0: float, w: float, h: float):
        """Draw exciter markers."""
        if self._state is None:
            return
        
        radius = 12
        
        for i, exc in enumerate(self._state.exciters):
            # Convert normalized position to canvas coords
            cx = x0 + exc.x * w
            cy = y0 + exc.y * h
            
            # Color based on coupling
            if exc.coupling > 0.7:
                color = "#00ff00"  # Green = good coupling
            elif exc.coupling > 0.3:
                color = "#ffaa00"  # Orange = medium
            else:
                color = "#ff4444"  # Red = poor coupling
            
            # Outer ring
            self._canvas.create_oval(
                cx - radius, cy - radius,
                cx + radius, cy + radius,
                fill=exc.color,
                outline=color,
                width=3,
                tags=f"exciter_{i}"
            )
            
            # Label
            self._canvas.create_text(
                cx, cy,
                text=str(i + 1),
                fill="white",
                font=("SF Pro", 10, "bold"),
                tags=f"exciter_{i}"
            )
            
            # Phase indicator (small arrow)
            angle_rad = np.radians(exc.phase)
            ax = cx + radius * 0.7 * np.cos(angle_rad)
            ay = cy - radius * 0.7 * np.sin(angle_rad)
            self._canvas.create_line(
                cx, cy, ax, ay,
                fill="white",
                width=2,
                tags=f"exciter_{i}"
            )
    
    def _draw_labels(self, x0: float, y0: float, w: float, h: float):
        """Draw dimension labels."""
        # Length label (bottom)
        length_mm = self._state.length * 1000
        self._canvas.create_text(
            x0 + w / 2, y0 + h + 20,
            text=f"{length_mm:.0f} mm",
            fill=COLORS['text'],
            font=("SF Pro", 10)
        )
        
        # Width label (left)
        width_mm = self._state.width * 1000
        self._canvas.create_text(
            x0 - 20, y0 + h / 2,
            text=f"{width_mm:.0f} mm",
            fill=COLORS['text'],
            font=("SF Pro", 10),
            angle=90
        )
        
        # Orientation labels
        self._canvas.create_text(
            x0 - 5, y0 + h / 2,
            text="FEET",
            fill="#ffffff44",
            font=("SF Pro", 8),
            anchor="e"
        )
        self._canvas.create_text(
            x0 + w + 5, y0 + h / 2,
            text="HEAD",
            fill="#ffffff44",
            font=("SF Pro", 8),
            anchor="w"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_resize(self, event):
        """Canvas resized."""
        self._redraw()
    
    def _on_click(self, event):
        """Mouse click - check if on exciter."""
        if self._state is None or not hasattr(self, '_plate_rect'):
            return
        
        x0, y0, x1, y1 = self._plate_rect
        w = x1 - x0
        h = y1 - y0
        
        # Check each exciter
        for i, exc in enumerate(self._state.exciters):
            cx = x0 + exc.x * w
            cy = y0 + exc.y * h
            
            dist = np.sqrt((event.x - cx)**2 + (event.y - cy)**2)
            if dist < 15:
                self._dragging_exciter = i
                return
        
        self._dragging_exciter = None
    
    def _on_drag(self, event):
        """Mouse drag - move exciter."""
        if self._dragging_exciter is None:
            return
        
        if not hasattr(self, '_plate_rect'):
            return
        
        x0, y0, x1, y1 = self._plate_rect
        w = x1 - x0
        h = y1 - y0
        
        # Convert to normalized coords
        nx = (event.x - x0) / w
        ny = (event.y - y0) / h
        
        # Clamp to [0, 1]
        nx = max(0, min(1, nx))
        ny = max(0, min(1, ny))
        
        if self._on_exciter_moved:
            self._on_exciter_moved(self._dragging_exciter, nx, ny)
    
    def _on_release(self, event):
        """Mouse release."""
        self._dragging_exciter = None
    
    def _on_double_click(self, event):
        """Double-click to add exciter."""
        if not hasattr(self, '_plate_rect'):
            return
        
        x0, y0, x1, y1 = self._plate_rect
        w = x1 - x0
        h = y1 - y0
        
        # Check if inside plate
        if x0 <= event.x <= x1 and y0 <= event.y <= y1:
            nx = (event.x - x0) / w
            ny = (event.y - y0) / h
            
            if self._on_exciter_added:
                self._on_exciter_added(nx, ny)
