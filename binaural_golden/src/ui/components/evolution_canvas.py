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
║                                                                              ║
║   COORDINATE SYSTEM:                                                          ║
║   Genome coordinates:                                                         ║
║     X = lateral position (0=left, 1=right of body lying down)                ║
║     Y = longitudinal position (0=feet, 1=head)                               ║
║   Canvas coordinates:                                                         ║
║     X axis = plate LENGTH (feet on left, head on right)                      ║
║     Y axis = plate WIDTH (body left-right, screen Y inverted)                ║
║   Mapping: genome.y → canvas.x, genome.x → canvas.y (inverted)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import logging
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass

# Setup logging for canvas debugging
logger = logging.getLogger(__name__)

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
        self._material: Optional[str] = None  # Material name for display
        self._material_data: Optional[Dict] = None  # Material properties dict
        
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
        self._show_material = True  # NEW: Show material info panel
        
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
    
    def set_material(self, material_name: str, material_data: Optional[Dict] = None):
        """
        Set material for display in info panel.
        
        Args:
            material_name: Material key (e.g., 'spruce', 'birch_plywood')
            material_data: Optional dict with E_longitudinal, E_transverse, density, etc.
        """
        self._material = material_name
        self._material_data = material_data
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
                self._draw_zone_flatness_panel(cw, ch)  # NEW: Zone response visualization
                self._draw_structural_info(cw, ch)      # NEW: Deflection info
            
            if self._show_material and self._material:
                self._draw_material_info(cw, ch)
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
        
        # Draw cutouts (lutherie: f-holes for macro tuning)
        if self._genome and self._genome.cutouts:
            for cutout in self._genome.cutouts:
                self._draw_cutout(x0, y0, x1, y1, cutout)
        
        # Draw grooves (lutherie: thin slices for fine tuning)
        if self._genome and hasattr(self._genome, 'grooves') and self._genome.grooves:
            for groove in self._genome.grooves:
                self._draw_groove(x0, y0, x1, y1, groove)
        
        # Draw exciters (4× Dayton DAEX25)
        if self._genome and hasattr(self._genome, 'exciters') and self._genome.exciters:
            for exciter in self._genome.exciters:
                self._draw_exciter(x0, y0, x1, y1, exciter)
    
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
        """
        Draw a cutout on the plate (liuteria: tuning holes).
        
        COORDINATE MAPPING:
        - Genome X = lateral (left-right of body, 0=left, 1=right)
        - Genome Y = longitudinal (feet-head, 0=feet, 1=head)
        - Canvas plate_w = plate LENGTH (along X axis, feet left → head right)
        - Canvas plate_h = plate WIDTH (along Y axis, body left-right)
        
        Therefore: genome Y → canvas X, genome X → canvas Y (inverted)
        
        Supports multiple shapes:
        - ellipse: Standard f-hole style
        - rectangle: Rectangular slot
        - rounded_rect: Rectangle with rounded corners
        - f_hole: Stylized double-curve f-hole
        - slot: Long thin slot
        - crescent: Crescent moon shape
        - tear: Teardrop shape
        - diamond: Rotated square
        """
        plate_w = x1 - x0  # Canvas width = plate LENGTH (head-feet)
        plate_h = y1 - y0  # Canvas height = plate WIDTH (left-right body)
        
        # COORDINATE SWAP: genome (x,y) → canvas (y, 1-x)
        # - genome.y (0=feet, 1=head) → canvas X (0=left/feet, 1=right/head)
        # - genome.x (0=left, 1=right) → canvas Y (inverted: 0=top, 1=bottom)
        cut_cx = x0 + cutout.y * plate_w      # genome Y → canvas X
        cut_cy = y0 + (1 - cutout.x) * plate_h  # genome X → canvas Y (inverted)
        
        # Swap width/height too (genome width=lateral, height=longitudinal)
        cut_rx = cutout.height * plate_w / 2   # genome height (longitudinal) → canvas rx
        cut_ry = cutout.width * plate_h / 2    # genome width (lateral) → canvas ry
        
        # Get rotation (default 0)
        rotation = getattr(cutout, 'rotation', 0.0)
        shape = getattr(cutout, 'shape', 'ellipse')
        corner_radius = getattr(cutout, 'corner_radius', 0.3)
        
        fill_color = STYLE.BG_DARK
        outline_color = STYLE.BG_HIGHLIGHT
        
        if shape == 'ellipse':
            self._draw_rotated_oval(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'rectangle':
            self._draw_rotated_rect(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'rounded_rect':
            self._draw_cutout_rounded_rect(cut_cx, cut_cy, cut_rx, cut_ry, rotation, corner_radius, fill_color, outline_color)
            
        elif shape == 'f_hole':
            self._draw_f_hole(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'slot':
            # Slot: very elongated rectangle
            self._draw_rotated_rect(cut_cx, cut_cy, cut_rx * 2, cut_ry * 0.3, rotation, fill_color, outline_color)
            
        elif shape == 'stadium':
            # Stadium: slot with rounded ends (oblong/discorectangle)
            self._draw_stadium(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'crescent':
            self._draw_crescent(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'tear':
            self._draw_teardrop(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'diamond':
            self._draw_rotated_rect(cut_cx, cut_cy, cut_rx, cut_ry, rotation + np.pi/4, fill_color, outline_color)
            
        elif shape == 'circle':
            # Perfect circle (simple CNC path)
            r = min(cut_rx, cut_ry)
            self._draw_rotated_oval(cut_cx, cut_cy, r, r, 0, fill_color, outline_color)
            
        elif shape == 'arc':
            # Arc/sector shape (partial circle)
            self._draw_arc_cutout(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'kidney':
            # Kidney shape (sound port style)
            self._draw_kidney(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 's_curve':
            # S-curve shape (torsional mode tuning)
            self._draw_s_curve(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'hexagon':
            # Regular hexagon (force distribution)
            self._draw_hexagon(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
            
        elif shape == 'freeform':
            # Freeform polygon defined by control_points (fresa manuale)
            self._draw_freeform_cutout(cut_cx, cut_cy, cut_rx, cut_ry, rotation, cutout, fill_color, outline_color)
            
        else:
            # Fallback: ellipse
            self._draw_rotated_oval(cut_cx, cut_cy, cut_rx, cut_ry, rotation, fill_color, outline_color)
    
    def _draw_freeform_cutout(self, cx, cy, rx, ry, angle, cutout, fill, outline):
        """
        Draw freeform polygon cutout (for manual router / fresa manuale).
        
        Uses control_points from CutoutGene to define arbitrary shape.
        """
        control_points = getattr(cutout, 'control_points', None)
        if control_points is None or len(control_points) < 3:
            # Fallback to ellipse if no control points
            self._draw_rotated_oval(cx, cy, rx, ry, angle, fill, outline)
            return
        
        points = []
        for pt in control_points:
            # Scale by rx, ry and rotate
            px = pt[0] * rx
            py = pt[1] * ry
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_rotated_oval(self, cx, cy, rx, ry, angle, fill, outline):
        """Draw ellipse with rotation using polygon approximation."""
        points = []
        n_points = 24
        for i in range(n_points):
            theta = 2 * np.pi * i / n_points
            px = rx * np.cos(theta)
            py = ry * np.sin(theta)
            # Rotate
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_rotated_rect(self, cx, cy, rx, ry, angle, fill, outline):
        """Draw rectangle with rotation."""
        corners = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]
        points = []
        for px, py in corners:
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1)
    
    def _draw_cutout_rounded_rect(self, cx, cy, rx, ry, angle, corner_r, fill, outline):
        """Draw rounded rectangle cutout with rotation."""
        # Corner radius as fraction of smaller dimension
        cr = min(rx, ry) * corner_r
        points = []
        # Generate rounded corners
        for corner_x, corner_y, start_angle in [
            (rx - cr, -ry + cr, -np.pi/2),  # Top right
            (rx - cr, ry - cr, 0),           # Bottom right
            (-rx + cr, ry - cr, np.pi/2),    # Bottom left
            (-rx + cr, -ry + cr, np.pi),     # Top left
        ]:
            for i in range(6):
                theta = start_angle + (np.pi/2) * i / 5
                px = corner_x + cr * np.cos(theta)
                py = corner_y + cr * np.sin(theta)
                # Rotate
                rpx = px * np.cos(angle) - py * np.sin(angle)
                rpy = px * np.sin(angle) + py * np.cos(angle)
                points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_f_hole(self, cx, cy, rx, ry, angle, fill, outline):
        """Draw stylized f-hole (violin style double curve)."""
        # F-hole is like two connected S-curves
        points = []
        # Top bulge
        for t in np.linspace(0, np.pi, 10):
            px = rx * 0.6 * np.sin(t)
            py = -ry + ry * 0.3 * np.cos(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        # Middle narrow
        for t in np.linspace(0, np.pi, 8):
            px = rx * 0.2 * np.sin(t + np.pi)
            py = ry * 0.3 * (t / np.pi - 0.5)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        # Bottom bulge
        for t in np.linspace(0, np.pi, 10):
            px = rx * 0.6 * np.sin(t + np.pi)
            py = ry - ry * 0.3 * np.cos(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_crescent(self, cx, cy, rx, ry, angle, fill, outline):
        """Draw crescent moon shape."""
        points = []
        # Outer arc
        for t in np.linspace(-np.pi/2, np.pi/2, 16):
            px = rx * np.cos(t)
            py = ry * np.sin(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        # Inner arc (offset for crescent)
        for t in np.linspace(np.pi/2, -np.pi/2, 16):
            px = rx * 0.7 * np.cos(t) + rx * 0.3
            py = ry * 0.8 * np.sin(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_teardrop(self, cx, cy, rx, ry, angle, fill, outline):
        """Draw teardrop shape (bulb at bottom, point at top)."""
        points = []
        # Teardrop parametric curve
        for t in np.linspace(0, 2*np.pi, 24):
            # Modified cardioid for teardrop
            r = 1 - np.sin(t)
            px = rx * r * np.cos(t) * 0.5
            py = ry * (0.5 - r * np.sin(t) * 0.5)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW CNC-FRIENDLY CURVED CUTOUT SHAPES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _draw_stadium(self, cx, cy, rx, ry, angle, fill, outline):
        """
        Draw stadium/discorectangle shape (slot with semicircular ends).
        
        CNC-friendly: simple arcs + straight lines.
        Good for elongated cutouts that need rounded ends.
        """
        points = []
        # Make it elongated (2:1 ratio minimum)
        if rx < ry:
            rx, ry = ry * 1.5, rx * 0.6
        r_end = ry  # Radius of semicircular ends
        straight_len = rx - r_end
        
        # Right semicircle
        for t in np.linspace(-np.pi/2, np.pi/2, 12):
            px = straight_len + r_end * np.cos(t)
            py = r_end * np.sin(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        # Left semicircle
        for t in np.linspace(np.pi/2, 3*np.pi/2, 12):
            px = -straight_len + r_end * np.cos(t)
            py = r_end * np.sin(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_arc_cutout(self, cx, cy, rx, ry, angle, fill, outline):
        """
        Draw arc/sector shape (partial circle - like pizza slice).
        
        Useful for asymmetric mode tuning and directing vibration.
        """
        points = []
        arc_angle = np.pi * 0.7  # 126 degrees arc
        # Center point
        points.extend([cx, cy])
        # Arc
        for t in np.linspace(-arc_angle/2, arc_angle/2, 16):
            px = rx * np.cos(t)
            py = ry * np.sin(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_kidney(self, cx, cy, rx, ry, angle, fill, outline):
        """
        Draw kidney/bean shape (sound port style).
        
        Common in acoustic guitars as side sound ports.
        Good for redirecting sound radiation.
        """
        points = []
        # Kidney is like ellipse with indent on one side
        for t in np.linspace(0, 2*np.pi, 32):
            # Indentation factor based on angle
            indent = 0.3 * np.sin(t) ** 2 if np.sin(t) > 0 else 0
            r_factor = 1 - indent
            px = rx * r_factor * np.cos(t)
            py = ry * np.sin(t)
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_s_curve(self, cx, cy, rx, ry, angle, fill, outline):
        """
        Draw S-curve shape (sinuous slot).
        
        Excellent for torsional mode control - the S shape interrupts
        diagonal vibration patterns while maintaining structural integrity.
        """
        points = []
        # Width of the S-slot
        slot_width = ry * 0.3
        # Generate S-curve path (top edge)
        for t in np.linspace(-1, 1, 20):
            # S-curve: x proportional to t, y follows sigmoid
            px = rx * t
            py = ry * 0.6 * np.tanh(t * 2) + slot_width/2
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        # Return path (bottom edge)
        for t in np.linspace(1, -1, 20):
            px = rx * t
            py = ry * 0.6 * np.tanh(t * 2) - slot_width/2
            rpx = px * np.cos(angle) - py * np.sin(angle)
            rpy = px * np.sin(angle) + py * np.cos(angle)
            points.extend([cx + rpx, cy + rpy])
        self.create_polygon(points, fill=fill, outline=outline, width=1, smooth=True)
    
    def _draw_hexagon(self, cx, cy, rx, ry, angle, fill, outline):
        """
        Draw regular hexagon shape.
        
        Hexagons provide good structural stability and even force distribution.
        The 120° angles reduce stress concentration compared to rectangles.
        """
        points = []
        for i in range(6):
            t = angle + np.pi/6 + i * np.pi/3  # Start flat-side up
            px = rx * np.cos(t)
            py = ry * np.sin(t)
            points.extend([cx + px, cy + py])
        self.create_polygon(points, fill=fill, outline=outline, width=1)
    
    def _draw_groove(self, x0: float, y0: float, x1: float, y1: float, groove):
        """
        Draw a groove on the plate (liuteria: thin slices for fine tuning).
        
        COORDINATE MAPPING (same as cutouts):
        - Genome groove.x = lateral (left-right of body)
        - Genome groove.y = longitudinal (feet-head)
        - Canvas plate_w = plate LENGTH (X axis)
        - Canvas plate_h = plate WIDTH (Y axis)
        
        Grooves are rendered as thin lines with gradient to show depth.
        """
        plate_w = x1 - x0  # Canvas width = plate LENGTH
        plate_h = y1 - y0  # Canvas height = plate WIDTH
        
        # COORDINATE SWAP: genome (x,y) → canvas (y, 1-x)
        gcx = x0 + groove.y * plate_w       # genome Y → canvas X
        gcy = y0 + (1 - groove.x) * plate_h # genome X → canvas Y (inverted)
        half_len = groove.length * plate_w / 2  # length along plate length axis
        
        # Calculate endpoints based on angle
        cos_a = np.cos(groove.angle)
        sin_a = np.sin(groove.angle)
        
        x1_g = gcx - half_len * cos_a
        y1_g = gcy - half_len * sin_a
        x2_g = gcx + half_len * cos_a
        y2_g = gcy + half_len * sin_a
        
        # Width proportional to groove.width_mm (scaled)
        line_width = max(2, int(groove.width_mm * plate_w / 200))
        
        # Color: darker = deeper groove (more stiffness reduction)
        depth_intensity = int(255 * (1 - groove.depth * 0.7))
        groove_color = f'#{depth_intensity:02x}{depth_intensity//2:02x}00'  # Orange-brown
        
        # Draw groove as thick line
        self.create_line(
            x1_g, y1_g, x2_g, y2_g,
            fill=groove_color, width=line_width, capstyle='round'
        )
        
        # Thin highlight on edge (3D effect) - use lighter color instead of alpha
        highlight_color = '#DDDDDD'  # Light gray highlight
        self.create_line(
            x1_g + 1, y1_g + 1, x2_g + 1, y2_g + 1,
            fill=highlight_color, width=1, capstyle='round'
        )
    
    def _draw_exciter(self, x0: float, y0: float, x1: float, y1: float, exciter):
        """
        Draw an exciter on the plate (4× Dayton DAEX25 via JAB4).
        
        COORDINATE MAPPING:
        - Genome exciter.x = lateral (0=left, 1=right of body)
        - Genome exciter.y = longitudinal (0=feet, 1=head)
        - Canvas plate_w = plate LENGTH (X axis, feet→head)
        - Canvas plate_h = plate WIDTH (Y axis, body left-right)
        
        Exciters are rendered as circles with channel labels.
        CH1/CH2 = Head (stereo), CH3/CH4 = Feet (stereo)
        """
        plate_w = x1 - x0  # Canvas width = plate LENGTH
        plate_h = y1 - y0  # Canvas height = plate WIDTH
        
        # COORDINATE SWAP: genome (x,y) → canvas (y, 1-x)
        # - genome.y (0=feet, 1=head) → canvas X (left=feet, right=head)
        # - genome.x (0=left, 1=right) → canvas Y (inverted for screen coords)
        ex_cx = x0 + exciter.y * plate_w       # genome Y → canvas X
        ex_cy = y0 + (1 - exciter.x) * plate_h # genome X → canvas Y (inverted)
        
        # Size based on diameter (25mm for DAEX25)
        radius = max(8, exciter.diameter_mm * plate_w / 1000 * 1.5)  # Scaled for visibility
        
        # Color by channel zone
        if exciter.channel <= 2:  # Head channels
            fill_color = '#FF6B6B'  # Red-ish
            label_color = '#FFFFFF'
        else:  # Feet channels
            fill_color = '#6B6BFF'  # Blue-ish
            label_color = '#FFFFFF'
        
        # Outer ring (mounting plate simulation)
        self.create_oval(
            ex_cx - radius - 2, ex_cy - radius - 2,
            ex_cx + radius + 2, ex_cy + radius + 2,
            fill='#333333', outline='#555555', width=1
        )
        
        # Exciter body
        self.create_oval(
            ex_cx - radius, ex_cy - radius,
            ex_cx + radius, ex_cy + radius,
            fill=fill_color, outline='#FFFFFF', width=2
        )
        
        # Channel number label
        self.create_text(
            ex_cx, ex_cy,
            text=f'CH{exciter.channel}',
            fill=label_color, font=('SF Pro', 8, 'bold')
        )
        
        # Power indicator (small text below)
        self.create_text(
            ex_cx, ex_cy + radius + 8,
            text=f'{int(exciter.power_w)}W',
            fill=STYLE.TEXT_MUTED, font=('SF Pro', 7)
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
        is_elliptical = self._genome and self._genome.contour_type in [
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # ARMS (Shavasana position - from person.py BODY_SEGMENTS)
        # arms: position_start=0.50, position_end=0.85, width_fraction=1.20
        # Coordinate system: x=0 piedi, x=1 testa; y lateral
        # ═══════════════════════════════════════════════════════════════════════
        arm_width = plate_h * 0.03  # Thin arms
        
        # Arms extend laterally beyond shoulders (width_fraction=1.20)
        # Position along body: 0.50 (wrist/hand) to 0.85 (shoulder)
        shoulder_x = x0 + plate_w * 0.85  # Near shoulders in body coords
        elbow_x = x0 + plate_w * 0.68     # Mid-arm
        wrist_x = x0 + plate_w * 0.50     # At waist level (Shavasana)
        
        # Lateral offset - arms slightly beyond body width
        arm_y_offset = 0.20 if is_elliptical else 0.18
        
        # Left arm (upper side in canvas Y)
        arm_y_upper = y0 + plate_h * (0.5 - arm_y_offset)
        # Upper arm: shoulder to elbow
        self.create_line(
            shoulder_x, arm_y_upper + arm_width,        # At shoulder
            elbow_x, arm_y_upper - arm_width * 0.5,     # Slightly out at elbow
            fill=body_color, width=2
        )
        # Forearm: elbow to wrist
        self.create_line(
            elbow_x, arm_y_upper - arm_width * 0.5,     # Elbow
            wrist_x, arm_y_upper - arm_width * 1.5,     # Wrist further out
            fill=body_color, width=2
        )
        
        # Left hand (small oval)
        hand_r = plate_h * 0.022
        self.create_oval(
            wrist_x - hand_r * 1.2, arm_y_upper - arm_width * 1.5 - hand_r,
            wrist_x + hand_r * 1.2, arm_y_upper - arm_width * 1.5 + hand_r,
            outline=body_color, width=2
        )
        
        # Right arm (lower side in canvas Y - mirrored)
        arm_y_lower = y0 + plate_h * (0.5 + arm_y_offset)
        # Upper arm
        self.create_line(
            shoulder_x, arm_y_lower - arm_width,        # At shoulder
            elbow_x, arm_y_lower + arm_width * 0.5,     # Slightly out at elbow
            fill=body_color, width=2
        )
        # Forearm
        self.create_line(
            elbow_x, arm_y_lower + arm_width * 0.5,     # Elbow
            wrist_x, arm_y_lower + arm_width * 1.5,     # Wrist further out
            fill=body_color, width=2
        )
        
        # Right hand
        self.create_oval(
            wrist_x - hand_r * 1.2, arm_y_lower + arm_width * 1.5 - hand_r,
            wrist_x + hand_r * 1.2, arm_y_lower + arm_width * 1.5 + hand_r,
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
    
    def _draw_zone_flatness_panel(self, cw: int, ch: int):
        """
        Draw zone flatness visualization panel.
        
        Shows the frequency response balance between:
        - SPINE (tactile/vibration therapy) - controlled by GUI slider
        - HEAD (audio reproduction) - controlled by GUI slider
        
        This visual feedback helps the user understand how the zone_weights
        slider affects the optimization target:
        - 100% spine = pure tactile therapy optimization
        - 50/50 = balanced vibroacoustic experience  
        - 100% head = pure audio reproduction optimization
        
        Reference: Similar to AutoEQ frequency response visualization
        with separate bands for different optimization targets.
        """
        if not self._fitness:
            return
        
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        
        # Panel in bottom-right area
        panel_x = x1 + 15
        panel_y = y1 - 140  # Below mode info
        panel_w = 120
        panel_h = 130
        
        # Background
        self.create_rectangle(
            panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
            fill=STYLE.BG_DARK, outline=STYLE.GOLD, width=1
        )
        
        # Header
        self.create_text(
            panel_x + 8, panel_y + 6,
            text="📊 ZONE FLATNESS",
            fill=STYLE.GOLD, font=(STYLE.FONT_MAIN, 9, 'bold'), anchor='nw'
        )
        
        y_off = 26
        
        # Spine flatness score (70% weight by default)
        spine_score = self._fitness.spine_flatness_score
        spine_bar_w = int(80 * spine_score)
        
        self.create_text(
            panel_x + 8, panel_y + y_off,
            text="SPINE",
            fill='#00FF88', font=(STYLE.FONT_MONO, 8, 'bold'), anchor='nw'
        )
        self.create_text(
            panel_x + 45, panel_y + y_off,
            text=f"{spine_score*100:.0f}%",
            fill=STYLE.TEXT_PRIMARY, font=(STYLE.FONT_MONO, 8), anchor='nw'
        )
        
        # Spine bar (green gradient)
        y_off += 14
        self.create_rectangle(
            panel_x + 8, panel_y + y_off,
            panel_x + 8 + 80, panel_y + y_off + 10,
            fill=STYLE.BG_LIGHT, outline='', width=0
        )
        self.create_rectangle(
            panel_x + 8, panel_y + y_off,
            panel_x + 8 + spine_bar_w, panel_y + y_off + 10,
            fill='#00FF88', outline='', width=0
        )
        
        # Head flatness score (30% weight by default)
        y_off += 22
        head_score = self._fitness.head_flatness_score
        head_bar_w = int(80 * head_score)
        
        self.create_text(
            panel_x + 8, panel_y + y_off,
            text="HEAD",
            fill='#00BFFF', font=(STYLE.FONT_MONO, 8, 'bold'), anchor='nw'
        )
        self.create_text(
            panel_x + 45, panel_y + y_off,
            text=f"{head_score*100:.0f}%",
            fill=STYLE.TEXT_PRIMARY, font=(STYLE.FONT_MONO, 8), anchor='nw'
        )
        
        # Head bar (blue gradient)
        y_off += 14
        self.create_rectangle(
            panel_x + 8, panel_y + y_off,
            panel_x + 8 + 80, panel_y + y_off + 10,
            fill=STYLE.BG_LIGHT, outline='', width=0
        )
        self.create_rectangle(
            panel_x + 8, panel_y + y_off,
            panel_x + 8 + head_bar_w, panel_y + y_off + 10,
            fill='#00BFFF', outline='', width=0
        )
        
        # Combined score
        y_off += 20
        combined = self._fitness.flatness_score
        self.create_text(
            panel_x + 8, panel_y + y_off,
            text=f"Combined: {combined*100:.0f}%",
            fill=STYLE.GOLD, font=(STYLE.FONT_MONO, 9, 'bold'), anchor='nw'
        )
        
        # Usage hint
        y_off += 18
        self.create_text(
            panel_x + 8, panel_y + y_off,
            text="(Slider: Spine↔Head)",
            fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MONO, 7), anchor='nw'
        )
    
    def _draw_structural_info(self, cw: int, ch: int):
        """
        Draw structural integrity information panel.
        
        Shows:
        - Max deflection under person weight (must be < 10mm)
        - Safety factor (must be > 2.0)
        - Warning if cutouts compromise structural integrity
        
        CRITICAL: The plate must support the person safely!
        """
        if not self._fitness:
            return
        
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        
        # Panel in bottom area
        panel_x = x0
        panel_y = y1 + 15
        panel_w = 180
        panel_h = 50
        
        # Background
        is_safe = getattr(self._fitness, 'deflection_is_safe', True)
        bg_color = STYLE.BG_DARK if is_safe else '#331111'
        border_color = STYLE.GOLD if is_safe else '#FF4444'
        
        self.create_rectangle(
            panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
            fill=bg_color, outline=border_color, width=1
        )
        
        # Header with icon
        icon = "✅" if is_safe else "⚠️"
        self.create_text(
            panel_x + 8, panel_y + 6,
            text=f"{icon} STRUCTURAL",
            fill=STYLE.GOLD if is_safe else '#FF4444', 
            font=(STYLE.FONT_MAIN, 9, 'bold'), anchor='nw'
        )
        
        # Deflection
        defl_mm = getattr(self._fitness, 'max_deflection_mm', 0.0)
        defl_color = STYLE.TEXT_PRIMARY if defl_mm < 10 else '#FF4444'
        self.create_text(
            panel_x + 8, panel_y + 24,
            text=f"Deflection: {defl_mm:.1f}mm",
            fill=defl_color, font=(STYLE.FONT_MONO, 8), anchor='nw'
        )
        self.create_text(
            panel_x + 95, panel_y + 24,
            text=f"(max 10mm)",
            fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MONO, 7), anchor='nw'
        )
        
        # Safety factor
        sf = getattr(self._fitness, 'stress_safety_factor', 2.0)
        sf_color = STYLE.TEXT_PRIMARY if sf >= 2.0 else '#FF8800'
        self.create_text(
            panel_x + 8, panel_y + 36,
            text=f"Safety: {sf:.1f}×",
            fill=sf_color, font=(STYLE.FONT_MONO, 8), anchor='nw'
        )
        self.create_text(
            panel_x + 75, panel_y + 36,
            text=f"(min 2.0×)",
            fill=STYLE.TEXT_MUTED, font=(STYLE.FONT_MONO, 7), anchor='nw'
        )

    def _draw_material_info(self, cw: int, ch: int):
        """
        Draw material info panel with fiber direction indicator.
        
        LUTHERIE: Shows orthotropic wood properties and fiber direction
        so the DSP agent can understand physical constraints.
        """
        if not self._material:
            return
        
        x0, y0, x1, y1, scale = self._get_plate_coords(cw, ch)
        
        # Info panel in top-left corner
        panel_x = 10
        panel_y = 10
        panel_w = 160
        panel_h = 130
        
        # Semi-transparent background
        self.create_rectangle(
            panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
            fill=STYLE.BG_DARK, outline=STYLE.GOLD, width=1
        )
        
        # Header with wood icon
        self.create_text(
            panel_x + 8, panel_y + 8,
            text="🪵 MATERIAL",
            fill=STYLE.GOLD, font=(STYLE.FONT_MAIN, 10, 'bold'), anchor='nw'
        )
        
        # Material name
        material_display = self._material.replace('_', ' ').title()
        self.create_text(
            panel_x + 8, panel_y + 28,
            text=material_display,
            fill=STYLE.TEXT_PRIMARY, font=(STYLE.FONT_MONO, 9, 'bold'), anchor='nw'
        )
        
        y_offset = 48
        
        if self._material_data:
            # Young's modulus along grain (E_longitudinal)
            E_long = self._material_data.get('E_longitudinal', 0)
            E_long_GPa = E_long / 1e9 if E_long > 1e6 else E_long
            self.create_text(
                panel_x + 8, panel_y + y_offset,
                text=f"E∥ : {E_long_GPa:.1f} GPa",
                fill=STYLE.TEXT_SECONDARY, font=(STYLE.FONT_MONO, 8), anchor='nw'
            )
            y_offset += 14
            
            # Young's modulus across grain (E_transverse)
            E_trans = self._material_data.get('E_transverse', 0)
            E_trans_GPa = E_trans / 1e9 if E_trans > 1e6 else E_trans
            self.create_text(
                panel_x + 8, panel_y + y_offset,
                text=f"E⊥ : {E_trans_GPa:.1f} GPa",
                fill=STYLE.TEXT_SECONDARY, font=(STYLE.FONT_MONO, 8), anchor='nw'
            )
            y_offset += 14
            
            # Anisotropy ratio
            if E_trans > 0:
                ratio = E_long / E_trans
                aniso_char = "ORTHOTROPIC" if ratio > 2 else "ISOTROPIC"
                self.create_text(
                    panel_x + 8, panel_y + y_offset,
                    text=f"E∥/E⊥: {ratio:.1f}x ({aniso_char[:5]})",
                    fill=STYLE.GOLD_LIGHT if ratio > 2 else STYLE.TEXT_MUTED, 
                    font=(STYLE.FONT_MONO, 8), anchor='nw'
                )
                y_offset += 14
            
            # Density
            density = self._material_data.get('density', 0)
            self.create_text(
                panel_x + 8, panel_y + y_offset,
                text=f"ρ: {density:.0f} kg/m³",
                fill=STYLE.TEXT_SECONDARY, font=(STYLE.FONT_MONO, 8), anchor='nw'
            )
            y_offset += 14
            
            # Damping
            damping = self._material_data.get('damping_ratio', 0)
            damping_str = "Low" if damping < 0.015 else ("Med" if damping < 0.03 else "High")
            self.create_text(
                panel_x + 8, panel_y + y_offset,
                text=f"Damping: {damping:.3f} ({damping_str})",
                fill=STYLE.TEXT_SECONDARY, font=(STYLE.FONT_MONO, 8), anchor='nw'
            )
        
        # Draw fiber direction indicator on the plate
        self._draw_fiber_direction(x0, y0, x1, y1, scale)
    
    def _draw_fiber_direction(self, x0: float, y0: float, x1: float, y1: float, scale: float):
        """
        Draw fiber direction arrows on plate.
        
        LUTHERIE: For orthotropic wood, the grain direction (fiber direction)
        determines the stiffness axis. E_longitudinal is along the grain.
        
        Default: grain runs along the plate LENGTH (person's spine direction)
        """
        if not self._material_data:
            return
        
        E_long = self._material_data.get('E_longitudinal', 1)
        E_trans = self._material_data.get('E_transverse', 1)
        ratio = E_long / E_trans if E_trans > 0 else 1
        
        # Only show for orthotropic materials (ratio > 1.5)
        if ratio < 1.5:
            return
        
        # Draw arrows indicating grain direction (along Y axis = plate length)
        plate_cx = (x0 + x1) / 2
        plate_cy = (y0 + y1) / 2
        arrow_length = min(x1 - x0, y1 - y0) * 0.15
        
        # Fiber direction: vertical arrows (along spine)
        for offset_x in [-0.3, 0, 0.3]:
            ax = plate_cx + offset_x * (x1 - x0) * 0.4
            
            # Draw double-headed arrow
            self.create_line(
                ax, plate_cy - arrow_length,
                ax, plate_cy + arrow_length,
                fill=STYLE.GOLD_LIGHT, width=1, arrow='both', arrowshape=(6, 8, 3)
            )
        
        # Label
        self.create_text(
            plate_cx, y1 + 15,
            text="↕ GRAIN",
            fill=STYLE.GOLD_LIGHT, font=(STYLE.FONT_MONO, 8), anchor='n'
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
