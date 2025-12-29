"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EVOLUTION CANVAS - Advanced Visualization                  ║
║                                                                              ║
║   High-quality canvas components for evolutionary plate optimization.        ║
║                                                                              ║
║   Components:                                                                 ║
║   • EvolutionCanvas - Main plate visualization with animations               ║
║   • GoldenProgressBar - Custom gradient progress bar                         ║
║   • FitnessRadarChart - Radar chart for fitness components                   ║
║   • FitnessLineChart - Line chart for fitness evolution                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import math
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass

# Core imports
from core.person import Person, SPINE_ZONES
from core.plate_genome import PlateGenome, ContourType
from core.fitness import FitnessResult

# Theme
from ui.theme import STYLE, hex_to_rgb, rgb_to_hex, blend_colors


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

# Animation settings
ANIMATION_FPS = 30
ANIMATION_DURATION_MS = 300

# Body zone colors (chakra-inspired)
BODY_ZONE_COLORS = {
    'feet': '#FF0000',
    'legs': '#FF8800',
    'pelvis': '#FFFF00',
    'torso': '#00FF00',
    'chest': '#00BFFF',
    'neck': '#4400FF',
    'head': '#FF00FF',
}


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION CANVAS
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionCanvas(tk.Canvas):
    """
    High-quality visualization canvas for plate evolution.
    
    Features:
    • Grid overlay with rulers
    • Smooth plate shape rendering
    • Human silhouette overlay
    • Spine zone visualization
    • Mode frequency labels
    • Dimension annotations
    • Animation support
    • Zoom & pan (future)
    """
    
    def __init__(
        self,
        parent,
        width: int = 700,
        height: int = 400,
        **kwargs
    ):
        # Default styling
        kwargs.setdefault('bg', STYLE.BG_DARK)
        kwargs.setdefault('highlightthickness', 0)
        
        super().__init__(parent, width=width, height=height, **kwargs)
        
        # State
        self._person: Optional[Person] = None
        self._genome: Optional[PlateGenome] = None
        self._fitness: Optional[FitnessResult] = None
        
        # Animation
        self._animation_progress: float = 1.0
        self._prev_genome: Optional[PlateGenome] = None
        self._animation_id: Optional[str] = None
        
        # Display options
        self._show_grid = True
        self._show_human = True
        self._show_spine = True
        self._show_dimensions = True
        self._show_modes = True
        
        # Colors from theme
        self._plate_fill = STYLE.PLATE_FILL
        self._plate_stroke = STYLE.GOLD
        self._grid_color = STYLE.BG_LIGHT
        self._text_color = STYLE.TEXT_SECONDARY
        self._accent_color = STYLE.GOLD
        
        # Bind events
        self.bind('<Configure>', self._on_resize)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_person(self, person: Optional[Person]):
        """Set person for body overlay."""
        self._person = person
        self.refresh()
    
    def set_plate(
        self,
        genome: Optional[PlateGenome],
        fitness: Optional[FitnessResult] = None,
        animate: bool = True
    ):
        """
        Update plate visualization.
        
        Args:
            genome: Plate genome to display
            fitness: Optional fitness result with mode frequencies
            animate: Animate transition from previous state
        """
        if animate and self._genome is not None:
            self._prev_genome = self._genome
            self._animation_progress = 0.0
            self._start_animation()
        else:
            self._animation_progress = 1.0
        
        self._genome = genome
        self._fitness = fitness
        self.refresh()
    
    def set_display_options(
        self,
        show_grid: bool = True,
        show_human: bool = True,
        show_spine: bool = True,
        show_dimensions: bool = True,
        show_modes: bool = True
    ):
        """Configure display options."""
        self._show_grid = show_grid
        self._show_human = show_human
        self._show_spine = show_spine
        self._show_dimensions = show_dimensions
        self._show_modes = show_modes
        self.refresh()
    
    def refresh(self):
        """Redraw canvas."""
        self._draw()
    
    def export_image(self, filename: str):
        """Export canvas as PostScript (can convert to PNG)."""
        self.postscript(file=filename, colormode='color')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Animation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _start_animation(self):
        """Start transition animation."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
        self._animate_step()
    
    def _animate_step(self):
        """Animation step."""
        if self._animation_progress < 1.0:
            # Ease-out cubic
            t = self._animation_progress
            eased = 1 - (1 - t) ** 3
            self._animation_progress = min(1.0, t + 1.0 / (ANIMATION_FPS * ANIMATION_DURATION_MS / 1000))
            
            self._draw()
            
            delay = int(1000 / ANIMATION_FPS)
            self._animation_id = self.after(delay, self._animate_step)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_resize(self, event):
        """Handle canvas resize."""
        self.refresh()
    
    def _draw(self):
        """Main draw function."""
        self.delete("all")
        
        cw = self.winfo_width()
        ch = self.winfo_height()
        
        if cw < 50 or ch < 50:
            return
        
        # Draw layers in order
        if self._show_grid:
            self._draw_grid(cw, ch)
        
        if self._genome:
            self._draw_plate(cw, ch)
            
            if self._show_human and self._person:
                self._draw_human_silhouette(cw, ch)
            
            if self._show_spine:
                self._draw_spine_zones(cw, ch)
            
            if self._show_dimensions:
                self._draw_dimensions(cw, ch)
            
            if self._show_modes and self._fitness:
                self._draw_mode_info(cw, ch)
        else:
            self._draw_placeholder(cw, ch)
    
    def _get_plate_coords(self, cw: int, ch: int) -> Tuple[float, float, float, float, float]:
        """
        Calculate plate coordinates on canvas.
        
        Returns (x0, y0, x1, y1, scale)
        """
        if self._genome is None:
            return 0, 0, cw, ch, 1.0
        
        margin = 50
        plate_length = self._genome.length
        plate_width = self._genome.width
        
        # Interpolate during animation
        if self._animation_progress < 1.0 and self._prev_genome:
            t = self._animation_progress
            plate_length = self._prev_genome.length + t * (plate_length - self._prev_genome.length)
            plate_width = self._prev_genome.width + t * (plate_width - self._prev_genome.width)
        
        # Scale to fit
        scale_x = (cw - 2 * margin) / plate_length
        scale_y = (ch - 2 * margin) / plate_width
        scale = min(scale_x, scale_y)
        
        # Center
        plate_w = plate_length * scale
        plate_h = plate_width * scale
        cx, cy = cw / 2, ch / 2
        
        x0 = cx - plate_w / 2
        y0 = cy - plate_h / 2
        x1 = cx + plate_w / 2
        y1 = cy + plate_h / 2
        
        return x0, y0, x1, y1, scale
    
    def _draw_grid(self, cw: int, ch: int):
        """Draw background grid."""
        grid_spacing = 40
        
        # Vertical lines
        for x in range(0, cw, grid_spacing):
            self.create_line(
                x, 0, x, ch,
                fill=self._grid_color, width=1, dash=(2, 4)
            )
        
        # Horizontal lines
        for y in range(0, ch, grid_spacing):
            self.create_line(
                0, y, cw, y,
                fill=self._grid_color, width=1, dash=(2, 4)
            )
    
    def _draw_placeholder(self, cw: int, ch: int):
        """Draw placeholder when no plate."""
        # Center text
        self.create_text(
            cw / 2, ch / 2,
            text="Configure person and\nstart evolution to see plate",
            fill=STYLE.TEXT_MUTED,
            font=(STYLE.FONT_MAIN, 14),
            justify='center'
        )
        
        # Golden spiral hint
        self._draw_golden_spiral(cw, ch, alpha=0.1)
    
    def _draw_golden_spiral(self, cw: int, ch: int, alpha: float = 0.3):
        """Draw decorative golden spiral."""
        cx, cy = cw / 2, ch / 2
        scale = min(cw, ch) * 0.3
        
        # Generate spiral points
        points = []
        for i in range(100):
            theta = i * 0.1
            r = scale * 0.05 * (PHI ** (theta / (2 * math.pi)))
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            points.append((x, y))
        
        # Draw as smooth line
        if len(points) > 2:
            flat_points = [coord for p in points for coord in p]
            color = blend_colors(STYLE.GOLD, STYLE.BG_DARK, 1 - alpha)
            self.create_line(
                flat_points, fill=color, width=2, smooth=True
            )
    
    def _draw_plate(self, cw: int, ch: int):
        """Draw plate shape."""
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        ct = self._genome.contour_type if self._genome else ContourType.RECTANGLE
        
        # Shadow (darker background color instead of alpha)
        shadow_offset = 4
        shadow_color = STYLE.BG_MEDIUM
        self._draw_plate_shape(
            x0 + shadow_offset, y0 + shadow_offset,
            x1 + shadow_offset, y1 + shadow_offset,
            ct, fill=shadow_color, outline=''
        )
        
        # Main plate
        self._draw_plate_shape(x0, y0, x1, y1, ct, 
                               fill=self._plate_fill, 
                               outline=self._plate_stroke,
                               width=2)
        
        # Inner glow
        self._draw_plate_shape(
            x0 + 3, y0 + 3, x1 - 3, y1 - 3, ct,
            fill='', outline=blend_colors(self._plate_stroke, self._plate_fill, 0.5),
            width=1
        )
        
        # Draw cutouts
        if self._genome and self._genome.cutouts:
            for cutout in self._genome.cutouts:
                self._draw_cutout(x0, y0, x1, y1, cutout)
    
    def _draw_plate_shape(
        self,
        x0: float, y0: float, x1: float, y1: float,
        contour_type: ContourType,
        fill: str = '',
        outline: str = '',
        width: int = 1
    ):
        """Draw plate shape based on contour type."""
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        rx = (x1 - x0) / 2
        ry = (y1 - y0) / 2
        
        if contour_type in [ContourType.RECTANGLE, ContourType.GOLDEN_RECT]:
            # Rounded rectangle
            self._draw_rounded_rect(x0, y0, x1, y1, radius=10, 
                                   fill=fill, outline=outline, width=width)
        
        elif contour_type == ContourType.ELLIPSE:
            self.create_oval(x0, y0, x1, y1, fill=fill, outline=outline, width=width)
        
        elif contour_type == ContourType.OVOID:
            # Ovoid shape - narrower at feet end
            points = self._generate_ovoid_points(cx, cy, rx, ry, n_points=60)
            if fill:
                self.create_polygon(points, fill=fill, outline=outline, 
                                   width=width, smooth=True)
            else:
                self.create_line(points + points[:2], fill=outline, 
                                width=width, smooth=True)
        
        elif contour_type == ContourType.VITRUVIAN:
            # Human-proportioned rounded shape
            points = self._generate_vitruvian_points(cx, cy, rx, ry, n_points=60)
            if fill:
                self.create_polygon(points, fill=fill, outline=outline,
                                   width=width, smooth=True)
            else:
                self.create_line(points + points[:2], fill=outline,
                                width=width, smooth=True)
        
        elif contour_type == ContourType.VESICA_PISCIS:
            # Two overlapping circles (sacred geometry)
            points = self._generate_vesica_points(cx, cy, rx, ry, n_points=60)
            if fill:
                self.create_polygon(points, fill=fill, outline=outline,
                                   width=width, smooth=True)
            else:
                self.create_line(points + points[:2], fill=outline,
                                width=width, smooth=True)
        
        else:
            # Default: rounded rectangle
            self._draw_rounded_rect(x0, y0, x1, y1, radius=10,
                                   fill=fill, outline=outline, width=width)
    
    def _draw_rounded_rect(
        self,
        x0: float, y0: float, x1: float, y1: float,
        radius: float = 10,
        **kwargs
    ):
        """Draw a rounded rectangle."""
        r = min(radius, (x1 - x0) / 4, (y1 - y0) / 4)
        
        points = [
            x0 + r, y0,
            x1 - r, y0,
            x1, y0,
            x1, y0 + r,
            x1, y1 - r,
            x1, y1,
            x1 - r, y1,
            x0 + r, y1,
            x0, y1,
            x0, y1 - r,
            x0, y0 + r,
            x0, y0,
            x0 + r, y0,
        ]
        
        self.create_polygon(points, smooth=True, **kwargs)
    
    def _generate_ovoid_points(
        self,
        cx: float, cy: float, rx: float, ry: float,
        n_points: int = 50
    ) -> List[float]:
        """Generate ovoid (egg-like) shape points."""
        points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            # Narrower at left (feet) end
            r_mod = 1 - 0.15 * math.cos(theta)
            x = cx + rx * math.cos(theta) * r_mod
            y = cy + ry * math.sin(theta)
            points.extend([x, y])
        return points
    
    def _generate_vitruvian_points(
        self,
        cx: float, cy: float, rx: float, ry: float,
        n_points: int = 50
    ) -> List[float]:
        """Generate Vitruvian (human-proportioned) shape."""
        points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            # Wider at torso (center-right), narrower at ends
            t = theta / (2 * math.pi)
            # Create smooth body-like curve
            width_mod = 1.0 + 0.1 * math.sin(2 * math.pi * (t - 0.25))
            height_mod = 1.0 - 0.05 * abs(math.cos(theta))
            
            x = cx + rx * math.cos(theta) * width_mod
            y = cy + ry * math.sin(theta) * height_mod
            points.extend([x, y])
        return points
    
    def _generate_vesica_points(
        self,
        cx: float, cy: float, rx: float, ry: float,
        n_points: int = 50
    ) -> List[float]:
        """Generate Vesica Piscis shape."""
        points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            # Vesica shape: intersection of two circles
            r_base = 1.0
            offset = 0.3
            
            if -math.pi/2 <= theta <= math.pi/2:
                r_mod = r_base * (1 + offset * math.cos(theta) ** 2)
            else:
                r_mod = r_base * (1 + offset * math.cos(theta) ** 2)
            
            x = cx + rx * math.cos(theta) * r_mod
            y = cy + ry * math.sin(theta)
            points.extend([x, y])
        return points
    
    def _draw_cutout(self, x0: float, y0: float, x1: float, y1: float, cutout):
        """Draw a cutout on the plate."""
        plate_w = x1 - x0
        plate_h = y1 - y0
        
        cut_cx = x0 + cutout.center_x * plate_w
        cut_cy = y0 + cutout.center_y * plate_h
        cut_rx = cutout.radius_x * plate_w / 2
        cut_ry = cutout.radius_y * plate_h / 2
        
        # Cutout with inner shadow
        self.create_oval(
            cut_cx - cut_rx, cut_cy - cut_ry,
            cut_cx + cut_rx, cut_cy + cut_ry,
            fill=STYLE.BG_DARK, outline=STYLE.BG_HIGHLIGHT, width=1
        )
    
    def _draw_human_silhouette(self, cw: int, ch: int):
        """Draw human body overlay, adapting to plate shape."""
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        plate_w = x1 - x0
        plate_h = y1 - y0
        
        # Body proportions (lying down, feet left, head right)
        # Use muted color for semi-transparent effect
        body_color = STYLE.TEXT_MUTED
        
        # === Adjust positions for non-rectangular shapes ===
        # For ellipse/ovoid, the effective width at y=center is full,
        # but we need margin from the actual edge
        is_elliptical = self._genome and self._genome.contour in [
            ContourType.ELLIPSE, ContourType.OVOID
        ]
        
        # Use 88% instead of 92% for head position (leaves ~12% margin)
        # This ensures head stays inside even for ellipses
        head_x_ratio = 0.88 if is_elliptical else 0.90
        feet_x_ratio = 0.08 if is_elliptical else 0.05
        
        # Head (ellipse at right)
        head_cx = x0 + plate_w * head_x_ratio
        head_cy = y0 + plate_h * 0.5
        head_rx = plate_h * 0.07  # Slightly smaller
        head_ry = plate_h * 0.09
        self.create_oval(
            head_cx - head_rx, head_cy - head_ry,
            head_cx + head_rx, head_cy + head_ry,
            outline=body_color, width=2
        )
        
        # Neck
        neck_x0 = x0 + plate_w * 0.82
        neck_x1 = head_cx - head_rx
        self.create_line(
            neck_x0, y0 + plate_h * 0.45,
            neck_x1, y0 + plate_h * 0.48,
            fill=body_color, width=2
        )
        self.create_line(
            neck_x0, y0 + plate_h * 0.55,
            neck_x1, y0 + plate_h * 0.52,
            fill=body_color, width=2
        )
        
        # Torso (tapered rectangle)
        torso_points = [
            x0 + plate_w * 0.45, y0 + plate_h * 0.34,  # Top left
            x0 + plate_w * 0.82, y0 + plate_h * 0.40,  # Top right
            x0 + plate_w * 0.82, y0 + plate_h * 0.60,  # Bottom right
            x0 + plate_w * 0.45, y0 + plate_h * 0.66,  # Bottom left
        ]
        self.create_polygon(torso_points, outline=body_color, fill='', width=2)
        
        # Legs - adjust width for elliptical shapes
        leg_width = plate_h * 0.10
        leg_y_inset = 0.32 if is_elliptical else 0.28  # More inset for ellipse
        
        # Left leg (upper in Y)
        self.create_rectangle(
            x0 + plate_w * feet_x_ratio, y0 + plate_h * leg_y_inset,
            x0 + plate_w * 0.45, y0 + plate_h * leg_y_inset + leg_width,
            outline=body_color, width=2
        )
        
        # Right leg (lower in Y)
        self.create_rectangle(
            x0 + plate_w * feet_x_ratio, y0 + plate_h * (1 - leg_y_inset) - leg_width,
            x0 + plate_w * 0.45, y0 + plate_h * (1 - leg_y_inset),
            outline=body_color, width=2
        )
        
        # Feet circles
        foot_r = plate_h * 0.035
        feet_cx = x0 + plate_w * (feet_x_ratio - 0.02)
        self.create_oval(
            feet_cx - foot_r, y0 + plate_h * (leg_y_inset + 0.02),
            feet_cx + foot_r, y0 + plate_h * (leg_y_inset + 0.06),
            outline=body_color, width=2
        )
        self.create_oval(
            feet_cx - foot_r, y0 + plate_h * (1 - leg_y_inset - 0.06),
            feet_cx + foot_r, y0 + plate_h * (1 - leg_y_inset - 0.02),
            outline=body_color, width=2
        )
        
        # Orientation labels - position inside plate
        self.create_text(
            x0 + plate_w * 0.12, y0 + plate_h / 2,
            text="FEET", fill=STYLE.TEXT_MUTED, 
            font=(STYLE.FONT_MAIN, 9, 'bold'), anchor='w'
        )
        self.create_text(
            x0 + plate_w * 0.88, y0 + plate_h / 2,
            text="HEAD", fill=STYLE.TEXT_MUTED,
            font=(STYLE.FONT_MAIN, 9, 'bold'), anchor='e'
        )
    
    def _draw_spine_zones(self, cw: int, ch: int):
        """Draw spine zone markers."""
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        plate_w = x1 - x0
        plate_h = y1 - y0
        
        spine_y = y0 + plate_h / 2
        
        zone_colors = {
            'lumbar': '#FF6B6B',
            'thoracic': '#4ECDC4',
            'cervical': '#95E1D3',
        }
        
        for zone_name, (start, end) in SPINE_ZONES.items():
            zone_x0 = x0 + plate_w * start
            zone_x1 = x0 + plate_w * end
            color = zone_colors.get(zone_name, '#FFFFFF')
            
            # Zone line
            self.create_line(
                zone_x0, spine_y - 3, zone_x1, spine_y - 3,
                fill=color, width=4, capstyle='round'
            )
            
            # Vertebrae dots
            n_dots = int((end - start) * 10)
            for i in range(n_dots):
                t = i / max(1, n_dots - 1)
                dot_x = zone_x0 + t * (zone_x1 - zone_x0)
                self.create_oval(
                    dot_x - 2, spine_y - 5, dot_x + 2, spine_y - 1,
                    fill=color, outline=''
                )
            
            # Label
            self.create_text(
                (zone_x0 + zone_x1) / 2, spine_y + 12,
                text=zone_name.upper(),
                fill=color, font=(STYLE.FONT_MAIN, 8, 'bold')
            )
    
    def _draw_dimensions(self, cw: int, ch: int):
        """Draw dimension annotations."""
        if self._genome is None:
            return
        
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        dim_color = STYLE.TEXT_MUTED
        
        # Length (bottom)
        ly = y1 + 20
        self.create_line(x0, ly, x1, ly, fill=dim_color, width=1)
        self.create_line(x0, ly - 5, x0, ly + 5, fill=dim_color, width=1)
        self.create_line(x1, ly - 5, x1, ly + 5, fill=dim_color, width=1)
        
        self.create_text(
            (x0 + x1) / 2, ly + 12,
            text=f"{self._genome.length:.2f} m",
            fill=dim_color, font=(STYLE.FONT_MONO, 10)
        )
        
        # Width (right)
        wx = x1 + 20
        self.create_line(wx, y0, wx, y1, fill=dim_color, width=1)
        self.create_line(wx - 5, y0, wx + 5, y0, fill=dim_color, width=1)
        self.create_line(wx - 5, y1, wx + 5, y1, fill=dim_color, width=1)
        
        # Rotated text for width
        self.create_text(
            wx + 15, (y0 + y1) / 2,
            text=f"{self._genome.width:.2f} m",
            fill=dim_color, font=(STYLE.FONT_MONO, 10), angle=90
        )
        
        # Thickness (top left info box)
        info_x = x0
        info_y = y0 - 15
        self.create_text(
            info_x, info_y,
            text=f"t = {self._genome.thickness_base * 1000:.1f} mm",
            fill=STYLE.GOLD_LIGHT, font=(STYLE.FONT_MONO, 9), anchor='sw'
        )
    
    def _draw_mode_info(self, cw: int, ch: int):
        """Draw mode frequency information."""
        if not self._fitness or not self._fitness.frequencies:
            return
        
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        
        # Info box on the right
        box_x = x1 + 50
        box_y = y0 + 10
        
        # Header
        self.create_text(
            box_x, box_y,
            text="🎵 Modes",
            fill=STYLE.GOLD, font=(STYLE.FONT_MAIN, 10, 'bold'), anchor='nw'
        )
        
        # Frequencies
        for i, freq in enumerate(self._fitness.frequencies[:8]):
            y = box_y + 20 + i * 16
            
            # Mode number
            self.create_text(
                box_x, y,
                text=f"f{i+1}:",
                fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MONO, 9), anchor='nw'
            )
            
            # Frequency value
            self.create_text(
                box_x + 25, y,
                text=f"{freq:.1f} Hz",
                fill=STYLE.TEXT_PRIMARY, font=(STYLE.FONT_MONO, 9), anchor='nw'
            )


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════

class GoldenProgressBar(tk.Canvas):
    """
    Custom progress bar with golden gradient.
    
    Features:
    • Smooth gradient fill
    • Animated shine effect
    • Percentage label
    • Pulsing glow when active
    """
    
    def __init__(
        self,
        parent,
        width: int = 200,
        height: int = 20,
        **kwargs
    ):
        super().__init__(
            parent, width=width, height=height,
            bg=STYLE.BG_DARK, highlightthickness=0, **kwargs
        )
        
        self._value = 0.0  # 0-100
        self._is_animating = False
        self._shine_position = 0.0
        self._shine_id: Optional[str] = None
        
        self._draw()
    
    def set_value(self, value: float, animate_shine: bool = True):
        """Set progress value (0-100)."""
        self._value = max(0, min(100, value))
        
        if animate_shine and not self._is_animating and value > 0:
            self._start_shine_animation()
        
        self._draw()
    
    def _start_shine_animation(self):
        """Start shine animation."""
        self._is_animating = True
        self._shine_position = 0.0
        self._animate_shine()
    
    def _animate_shine(self):
        """Animate shine effect."""
        if not self._is_animating:
            return
        
        self._shine_position += 0.05
        if self._shine_position > 1.5:
            self._is_animating = False
            self._shine_position = 0
        
        self._draw()
        
        if self._is_animating:
            self._shine_id = self.after(30, self._animate_shine)
    
    def _draw(self):
        """Draw progress bar."""
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w < 10 or h < 5:
            return
        
        radius = h / 2
        padding = 2
        
        # Background track
        self._draw_rounded_rect(
            padding, padding, w - padding, h - padding,
            radius - padding,
            fill=STYLE.BG_LIGHT, outline=STYLE.BG_HIGHLIGHT
        )
        
        # Progress fill
        fill_width = (w - 2 * padding) * (self._value / 100)
        if fill_width > 10:
            # Gradient segments
            n_segments = max(1, int(fill_width / 3))
            for i in range(n_segments):
                t = i / n_segments
                seg_x0 = padding + fill_width * t
                seg_x1 = padding + fill_width * (t + 1 / n_segments)
                
                # Golden gradient
                color = blend_colors(STYLE.GOLD_DARK, STYLE.GOLD_LIGHT, t)
                
                # Shine effect
                if self._is_animating:
                    shine_t = abs(t - self._shine_position)
                    if shine_t < 0.2:
                        color = blend_colors(color, '#FFFFFF', (0.2 - shine_t) * 2)
                
                self.create_rectangle(
                    seg_x0, padding + 1, seg_x1, h - padding - 1,
                    fill=color, outline=''
                )
            
            # Top highlight (use lighter gold)
            self.create_line(
                padding + 5, padding + 3,
                padding + fill_width - 5, padding + 3,
                fill=STYLE.GOLD_LIGHT, width=1
            )
        
        # Percentage text
        if self._value > 0:
            text_x = padding + fill_width / 2 if fill_width > 30 else w / 2
            self.create_text(
                text_x, h / 2,
                text=f"{self._value:.0f}%",
                fill=STYLE.TEXT_DARK if fill_width > 30 else STYLE.TEXT_PRIMARY,
                font=(STYLE.FONT_MAIN, 9, 'bold')
            )
    
    def _draw_rounded_rect(
        self,
        x0: float, y0: float, x1: float, y1: float,
        radius: float,
        **kwargs
    ):
        """Draw rounded rectangle."""
        r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
        points = [
            x0 + r, y0,
            x1 - r, y0,
            x1, y0, x1, y0 + r,
            x1, y1 - r,
            x1, y1, x1 - r, y1,
            x0 + r, y1,
            x0, y1, x0, y1 - r,
            x0, y0 + r,
            x0, y0, x0 + r, y0,
        ]
        self.create_polygon(points, smooth=True, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# FITNESS RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════

class FitnessRadarChart(tk.Canvas):
    """
    Radar chart showing fitness component scores.
    
    Displays: Flatness, Spine Coupling, Mass, Edge Support
    """
    
    def __init__(
        self,
        parent,
        size: int = 150,
        **kwargs
    ):
        super().__init__(
            parent, width=size, height=size,
            bg=STYLE.BG_DARK, highlightthickness=0, **kwargs
        )
        
        self._scores: Dict[str, float] = {
            'flatness': 0.0,
            'spine': 0.0,
            'mass': 0.0,
            'edge': 0.0,
        }
        
        self._labels = {
            'flatness': 'Flatness',
            'spine': 'Spine',
            'mass': 'Mass',
            'edge': 'Edge',
        }
        
        self._colors = {
            'flatness': '#FF6B6B',
            'spine': '#4ECDC4',
            'mass': '#FFE66D',
            'edge': '#95E1D3',
        }
        
        self._draw()
        self.bind('<Configure>', lambda e: self._draw())
    
    def set_scores(
        self,
        flatness: float = 0.0,
        spine: float = 0.0,
        mass: float = 0.0,
        edge: float = 0.0
    ):
        """Set fitness scores (0-1 range)."""
        self._scores = {
            'flatness': flatness,
            'spine': spine,
            'mass': mass,
            'edge': edge,
        }
        self._draw()
    
    def _draw(self):
        """Draw radar chart."""
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w < 50 or h < 50:
            return
        
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 25
        n_axes = len(self._scores)
        
        # Draw background grid
        for level in [0.25, 0.5, 0.75, 1.0]:
            points = []
            for i, key in enumerate(self._scores.keys()):
                angle = 2 * math.pi * i / n_axes - math.pi / 2
                r = radius * level
                points.extend([
                    cx + r * math.cos(angle),
                    cy + r * math.sin(angle)
                ])
            points.extend(points[:2])  # Close polygon
            
            self.create_polygon(
                points,
                fill='', outline=STYLE.BG_LIGHT, width=1
            )
        
        # Draw axes
        for i, key in enumerate(self._scores.keys()):
            angle = 2 * math.pi * i / n_axes - math.pi / 2
            
            # Axis line
            self.create_line(
                cx, cy,
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
                fill=STYLE.BG_HIGHLIGHT, width=1
            )
            
            # Label
            label_r = radius + 15
            self.create_text(
                cx + label_r * math.cos(angle),
                cy + label_r * math.sin(angle),
                text=self._labels[key],
                fill=self._colors[key],
                font=(STYLE.FONT_MAIN, 8, 'bold')
            )
        
        # Draw data polygon
        data_points = []
        for i, (key, value) in enumerate(self._scores.items()):
            angle = 2 * math.pi * i / n_axes - math.pi / 2
            r = radius * value
            data_points.extend([
                cx + r * math.cos(angle),
                cy + r * math.sin(angle)
            ])
        
        if len(data_points) >= 6:
            data_points.extend(data_points[:2])
            
            # Fill with blended color (simulating transparency)
            fill_color = blend_colors(STYLE.GOLD, STYLE.BG_DARK, 0.6)
            self.create_polygon(
                data_points,
                fill=fill_color,
                outline=STYLE.GOLD, width=2
            )
            
            # Dots at vertices
            for i in range(0, len(data_points) - 2, 2):
                self.create_oval(
                    data_points[i] - 4, data_points[i + 1] - 4,
                    data_points[i] + 4, data_points[i + 1] + 4,
                    fill=STYLE.GOLD, outline=''
                )


# ══════════════════════════════════════════════════════════════════════════════
# FITNESS LINE CHART
# ══════════════════════════════════════════════════════════════════════════════

class FitnessLineChart(tk.Canvas):
    """
    Line chart showing fitness evolution over generations.
    """
    
    def __init__(
        self,
        parent,
        width: int = 300,
        height: int = 150,
        **kwargs
    ):
        super().__init__(
            parent, width=width, height=height,
            bg=STYLE.BG_DARK, highlightthickness=0, **kwargs
        )
        
        self._data: Dict[str, List[float]] = {
            'total': [],
            'flatness': [],
            'spine': [],
            'mass': [],
        }
        
        self._colors = {
            'total': STYLE.GOLD,
            'flatness': '#FF6B6B',
            'spine': '#4ECDC4',
            'mass': '#FFE66D',
        }
        
        self._show_legend = True
        self._max_points = 100
        
        self._draw()
        self.bind('<Configure>', lambda e: self._draw())
    
    def add_point(
        self,
        total: float,
        flatness: float = 0.0,
        spine: float = 0.0,
        mass: float = 0.0
    ):
        """Add data point."""
        self._data['total'].append(total)
        self._data['flatness'].append(flatness)
        self._data['spine'].append(spine)
        self._data['mass'].append(mass)
        
        # Limit points
        for key in self._data:
            if len(self._data[key]) > self._max_points:
                self._data[key] = self._data[key][-self._max_points:]
        
        self._draw()
    
    def set_data(self, data: Dict[str, List[float]]):
        """Set all data at once."""
        self._data = data
        self._draw()
    
    def clear(self):
        """Clear all data."""
        self._data = {k: [] for k in self._data}
        self._draw()
    
    def _draw(self):
        """Draw line chart."""
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w < 50 or h < 30:
            return
        
        margin_left = 35
        margin_right = 10
        margin_top = 10
        margin_bottom = 25
        
        chart_w = w - margin_left - margin_right
        chart_h = h - margin_top - margin_bottom
        
        # Background
        self.create_rectangle(
            margin_left, margin_top,
            w - margin_right, h - margin_bottom,
            fill=STYLE.BG_MEDIUM, outline=STYLE.BG_HIGHLIGHT
        )
        
        # Grid lines
        for i in range(5):
            y = margin_top + chart_h * i / 4
            self.create_line(
                margin_left, y, w - margin_right, y,
                fill=STYLE.BG_HIGHLIGHT, dash=(2, 4)
            )
            
            # Y-axis labels
            value = 1.0 - i / 4
            self.create_text(
                margin_left - 5, y,
                text=f"{value:.1f}",
                fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MONO, 8), anchor='e'
            )
        
        # Draw lines
        n_points = len(self._data.get('total', []))
        if n_points < 2:
            # Placeholder text
            self.create_text(
                w / 2, h / 2,
                text="Start evolution\nto see progress",
                fill=STYLE.TEXT_MUTED,
                font=(STYLE.FONT_MAIN, 10),
                justify='center'
            )
            return
        
        # X-axis label
        self.create_text(
            w / 2, h - 5,
            text="Generation",
            fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MAIN, 8)
        )
        
        # Plot each series
        for key, values in self._data.items():
            if not values:
                continue
            
            color = self._colors.get(key, STYLE.TEXT_PRIMARY)
            points = []
            
            for i, v in enumerate(values):
                x = margin_left + chart_w * i / max(1, n_points - 1)
                y = margin_top + chart_h * (1 - v)
                points.extend([x, y])
            
            if len(points) >= 4:
                self.create_line(
                    points, fill=color, width=2 if key == 'total' else 1,
                    smooth=True
                )
        
        # Legend
        if self._show_legend:
            legend_y = margin_top + 5
            for i, (key, color) in enumerate(self._colors.items()):
                lx = margin_left + 5 + i * 55
                self.create_rectangle(
                    lx, legend_y, lx + 10, legend_y + 8,
                    fill=color, outline=''
                )
                self.create_text(
                    lx + 14, legend_y + 4,
                    text=key.capitalize()[:5],
                    fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MAIN, 7), anchor='w'
                )


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Evolution Canvas Test")
    root.configure(bg=STYLE.BG_DARK)
    
    # Main frame
    main_frame = tk.Frame(root, bg=STYLE.BG_DARK)
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Evolution canvas
    canvas = EvolutionCanvas(main_frame, width=600, height=350)
    canvas.pack(fill='both', expand=True, pady=5)
    
    # Test person and plate
    from core.person import Person
    person = Person(height_m=1.80, weight_kg=80)
    canvas.set_person(person)
    
    genome = PlateGenome(
        length=person.recommended_plate_length,
        width=person.recommended_plate_width,
        contour_type=ContourType.GOLDEN_RECT,
    )
    canvas.set_plate(genome)
    
    # Bottom widgets
    bottom_frame = tk.Frame(main_frame, bg=STYLE.BG_DARK)
    bottom_frame.pack(fill='x', pady=10)
    
    # Progress bar
    progress_label = tk.Label(bottom_frame, text="Progress:", 
                              bg=STYLE.BG_DARK, fg=STYLE.TEXT_PRIMARY)
    progress_label.pack(side='left', padx=5)
    
    progress = GoldenProgressBar(bottom_frame, width=200, height=20)
    progress.pack(side='left', padx=5)
    
    # Radar chart
    radar = FitnessRadarChart(bottom_frame, size=120)
    radar.pack(side='left', padx=20)
    radar.set_scores(flatness=0.8, spine=0.6, mass=0.9, edge=0.7)
    
    # Line chart
    line_chart = FitnessLineChart(bottom_frame, width=200, height=100)
    line_chart.pack(side='left', padx=10)
    
    # Simulate data
    import random
    for i in range(30):
        t = i / 30
        line_chart.add_point(
            total=0.3 + 0.5 * t + random.uniform(-0.05, 0.05),
            flatness=0.4 + 0.4 * t + random.uniform(-0.05, 0.05),
            spine=0.2 + 0.6 * t + random.uniform(-0.05, 0.05),
            mass=0.5 + 0.3 * t + random.uniform(-0.05, 0.05),
        )
    
    # Animate progress
    def animate_progress():
        val = progress._value + 2
        if val > 100:
            val = 0
        progress.set_value(val)
        root.after(100, animate_progress)
    
    animate_progress()
    
    root.mainloop()
