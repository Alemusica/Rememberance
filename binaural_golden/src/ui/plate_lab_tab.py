"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE LAB v3 - FEM Modal Analysis                         ║
║                                                                              ║
║   Features:                                                                  ║
║   • Custom plate shape drawing (rectangle, ellipse, polygon)                 ║
║   • FEM-based modal analysis using scikit-fem                                ║
║   • Draggable exciters with real-time coupling calculation                   ║
║   • Heatmap visualization (red=antinode, blue=node)                          ║
║   • Multi-exciter support (up to 4)                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import threading
import time

# Audio
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Plate FEM module
try:
    from core.plate_fem import PlateAnalyzer, MATERIALS, PlateShape, FEMMode
    HAS_FEM = True
except ImportError:
    HAS_FEM = False
    MATERIALS = {
        "spruce": None,
        "birch_plywood": None,
        "marine_plywood": None,
        "mdf": None,
        "oak": None,
        "maple": None,
        "aluminum": None,
        "steel": None,
    }

# Sacred geometry & body proportions (refactored modules)
try:
    from core.sacred_geometry import (
        PHI, WATER_GEOMETRY, ANTAHKARANA,
        WaterMoleculeGeometry, AntahkaranaAxis, HumanBodyGolden,
        generate_butterfly_shape, generate_golden_ovoid
    )
    HAS_SACRED_GEOMETRY = True
except ImportError:
    HAS_SACRED_GEOMETRY = False

# UI Theme (refactored)
try:
    from ui.theme import STYLE, configure_ttk_style
except ImportError:
    STYLE = None

# UI Widgets (refactored)
try:
    from ui.plate_lab_widgets import ScrollableSidebar
    HAS_SCROLLABLE_SIDEBAR = True
except ImportError:
    HAS_SCROLLABLE_SIDEBAR = False


# ══════════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS (improved contrast)
# ══════════════════════════════════════════════════════════════════════════════

class Style:
    # Backgrounds (darker for better contrast)
    BG_DARK = "#1a1a2e"
    BG_PANEL = "#252544"
    BG_LIGHT = "#2d2d52"
    
    # Accents (brighter gold)
    ACCENT_GOLD = "#ffd700"
    ACCENT_BLUE = "#4a90d9"
    
    # Text (higher contrast)
    TEXT_LIGHT = "#f5f5f5"      # Brighter white
    TEXT_MUTED = "#a0a0b8"      # Lighter muted
    TEXT_DARK = "#1a1a2e"       # For light backgrounds
    
    # Status colors
    SUCCESS = "#4caf50"
    WARNING = "#ff9800"
    ERROR = "#f44336"
    
    # Fonts (larger for readability)
    FONT_LABEL = ("Segoe UI", 11)
    FONT_HEADER = ("Segoe UI", 13, "bold")
    FONT_MONO = ("Consolas", 11)
    FONT_SMALL = ("Segoe UI", 10)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

# Golden ratio (also available from core.sacred_geometry if imported)
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034

# ══════════════════════════════════════════════════════════════════════════════
# WATER MOLECULE GEOMETRY (H₂O)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WaterMoleculeGeometry:
    """
    Geometry of water molecule (H₂O) for vibroacoustic plate design.
    
    The human body is ~60-70% water, making these proportions deeply resonant.
    
    H₂O Properties:
    - H-O-H angle: 104.5° (tetrahedral-like due to lone pairs)
    - O-H bond length: 0.9584 Å (angstroms)
    - H-H distance: 1.5151 Å
    - Molecular symmetry: C2v (bent)
    
    Vibrational modes of water:
    - ν₁: Symmetric stretch ~3657 cm⁻¹
    - ν₂: Bending mode ~1595 cm⁻¹  
    - ν₃: Asymmetric stretch ~3756 cm⁻¹
    """
    
    # Fundamental constants
    BOND_ANGLE_DEG: float = 104.5  # H-O-H angle in degrees
    BOND_LENGTH_RATIO: float = 0.9584  # O-H bond normalized
    
    # Vibrational frequencies (cm⁻¹) - for reference
    NU1_SYMMETRIC: float = 3657.0
    NU2_BENDING: float = 1595.0
    NU3_ASYMMETRIC: float = 3756.0
    
    @property
    def bond_angle_rad(self) -> float:
        """Bond angle in radians."""
        return np.radians(self.BOND_ANGLE_DEG)
    
    @property
    def half_angle_rad(self) -> float:
        """Half of the bond angle."""
        return self.bond_angle_rad / 2
    
    @property
    def h_h_distance_ratio(self) -> float:
        """
        Distance between H atoms relative to O-H bond.
        Using law of cosines: H-H = 2 * O-H * sin(θ/2)
        """
        return 2 * self.BOND_LENGTH_RATIO * np.sin(self.half_angle_rad)
    
    def get_plate_dimensions(self, scale_m: float = 1.0) -> Tuple[float, float]:
        """
        Get plate dimensions based on water molecule geometry.
        
        Args:
            scale_m: Scale factor in meters (base dimension)
        
        Returns:
            (length, width) where the plate mimics H₂O geometry
        """
        # Length along H-H axis
        length = scale_m * self.h_h_distance_ratio
        # Width from O to H-H midpoint
        width = scale_m * self.BOND_LENGTH_RATIO * np.cos(self.half_angle_rad)
        return length, width
    
    def get_molecule_points(self, center_x: float, center_y: float, 
                           scale: float = 1.0) -> Dict[str, Tuple[float, float]]:
        """
        Get coordinates of H₂O atoms centered at (center_x, center_y).
        O is at center, H atoms spread according to bond angle.
        
        Returns:
            Dict with 'O', 'H1', 'H2' positions
        """
        # Oxygen at center
        o_pos = (center_x, center_y)
        
        # Hydrogen atoms at bond angle
        h1_angle = np.pi/2 + self.half_angle_rad  # Upper-left
        h2_angle = np.pi/2 - self.half_angle_rad  # Upper-right
        
        bond_len = scale * self.BOND_LENGTH_RATIO
        
        h1_pos = (
            center_x + bond_len * np.cos(h1_angle),
            center_y - bond_len * np.sin(h1_angle)  # Y inverted for canvas
        )
        h2_pos = (
            center_x + bond_len * np.cos(h2_angle),
            center_y - bond_len * np.sin(h2_angle)
        )
        
        return {'O': o_pos, 'H1': h1_pos, 'H2': h2_pos}
    
    def get_vibrational_ratios(self) -> Dict[str, float]:
        """
        Get ratios between vibrational modes.
        Useful for designing resonant structures.
        """
        base = self.NU2_BENDING  # Use bending mode as reference
        return {
            'ν₁/ν₂': self.NU1_SYMMETRIC / base,  # ~2.29
            'ν₃/ν₂': self.NU3_ASYMMETRIC / base,  # ~2.36
            'ν₃/ν₁': self.NU3_ASYMMETRIC / self.NU1_SYMMETRIC,  # ~1.03
        }


# Water molecule geometry singleton
WATER_GEOMETRY = WaterMoleculeGeometry()


# ══════════════════════════════════════════════════════════════════════════════
# ANTAHKARANA - The Rainbow Bridge (Sushumna Nadi / Central Channel)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AntahkaranaAxis:
    """
    Antahkarana (अन्तःकरण) - Sanskrit: "inner instrument" or "internal organ"
    
    The Antahkarana represents:
    1. The bridge between body and spirit
    2. The connection between lower mind (Manas) and higher mind (Buddhi)
    3. The pathway to enlightenment (Sushumna Nadi)
    
    Four Functions (Yogapedia/Vedanta):
    - Manas: Lower mind, sensory connection to external world
    - Buddhi: Intellect, discernment of truth
    - Chitta: Memory/consciousness, stored impressions
    - Ahamkara: Ego, the "I-maker"
    
    Sushumna Nadi:
    - Central energy channel running along the spine
    - When Ida (left) and Pingala (right) are balanced, prana flows
    - Passes through all 7 chakras from root to crown
    - Known as "Brahmanadi" - channel of the Absolute
    
    Three States of Consciousness:
    - Jagrat: Waking state
    - Svapna: Dream state
    - Susupti: Deep sleep state
    """
    
    # The 7 chakras along Sushumna with their properties
    # Positions as fraction of spine length (0 = base, 1 = crown)
    # Frequencies based on harmonic resonance theories
    CHAKRAS: Dict[str, Dict] = field(default_factory=lambda: {
        "Muladhara": {       # Root
            "position": 0.0,
            "frequency_hz": 256.0,   # C4 - Earth element
            "color": "#ff0000",      # Red
            "element": "Earth",
            "function": "Survival, grounding",
            "bija_mantra": "LAM",
        },
        "Svadhisthana": {    # Sacral
            "position": 0.15,
            "frequency_hz": 288.0,   # D4 - Water element
            "color": "#ff8800",      # Orange
            "element": "Water",
            "function": "Creativity, sexuality",
            "bija_mantra": "VAM",
        },
        "Manipura": {        # Solar Plexus
            "position": 0.35,
            "frequency_hz": 320.0,   # E4 - Fire element
            "color": "#ffff00",      # Yellow
            "element": "Fire",
            "function": "Will, power",
            "bija_mantra": "RAM",
        },
        "Anahata": {         # Heart - THE GOLDEN CENTER
            "position": 1 - 1/1.618034,  # ≈ 0.382 - φ point from crown!
            "frequency_hz": 341.3,   # F4 - Air element
            "color": "#00ff00",      # Green
            "element": "Air",
            "function": "Love, compassion",
            "bija_mantra": "YAM",
            "is_golden": True,       # Marks the φ chakra
        },
        "Vishuddha": {       # Throat
            "position": 0.65,
            "frequency_hz": 384.0,   # G4 - Ether/Space
            "color": "#00bfff",      # Light Blue
            "element": "Ether",
            "function": "Communication, truth",
            "bija_mantra": "HAM",
        },
        "Ajna": {            # Third Eye
            "position": 0.85,
            "frequency_hz": 426.7,   # A4 approx - Light
            "color": "#4400ff",      # Indigo
            "element": "Light",
            "function": "Intuition, wisdom",
            "bija_mantra": "OM",
        },
        "Sahasrara": {       # Crown
            "position": 1.0,
            "frequency_hz": 480.0,   # B4 approx - Thought/Spirit
            "color": "#ff00ff",      # Violet/White
            "element": "Thought",
            "function": "Connection to divine, enlightenment",
            "bija_mantra": "Silence",
        },
    })
    
    # The three main nadis (energy channels)
    NADIS: Dict[str, Dict] = field(default_factory=lambda: {
        "Sushumna": {
            "path": "central",
            "quality": "Balance, enlightenment",
            "color": "#ffd700",  # Gold
        },
        "Ida": {
            "path": "left",
            "quality": "Lunar, feminine, cooling",
            "color": "#c0c0c0",  # Silver
        },
        "Pingala": {
            "path": "right", 
            "quality": "Solar, masculine, heating",
            "color": "#ffa500",  # Orange/Gold
        },
    })
    
    def get_chakra_positions_on_body(self, body_height: float) -> Dict[str, float]:
        """
        Get actual positions of chakras along a body of given height.
        
        When lying down, positions are from feet (0) to head (height).
        The spine runs approximately from sacrum to skull base.
        
        Args:
            body_height: Height in meters
            
        Returns:
            Dict of chakra name -> position in meters from feet
        """
        # Spine runs from ~0.05H to ~0.85H when standing
        spine_start = body_height * 0.05  # Coccyx
        spine_end = body_height * 0.85    # Skull base
        spine_length = spine_end - spine_start
        
        positions = {}
        for name, chakra in self.CHAKRAS.items():
            pos_on_spine = chakra["position"] * spine_length
            pos_on_body = spine_start + pos_on_spine
            positions[name] = pos_on_body
            
        return positions
    
    def get_chakra_frequencies(self) -> List[float]:
        """Get all chakra frequencies in order from root to crown."""
        return [c["frequency_hz"] for c in self.CHAKRAS.values()]
    
    def get_golden_chakra(self) -> str:
        """
        Find the chakra closest to the golden ratio point.
        The heart chakra (Anahata) is traditionally at the φ point.
        """
        golden_pos = 1 / PHI  # ≈ 0.618 from crown, or ~0.382 from base
        
        closest = min(
            self.CHAKRAS.items(),
            key=lambda x: abs(x[1]["position"] - (1 - golden_pos))
        )
        return closest[0]
    
    def get_frequency_at_position(self, position: float) -> float:
        """
        Interpolate frequency at any position along the Antahkarana.
        
        Args:
            position: 0 = root, 1 = crown
            
        Returns:
            Interpolated frequency in Hz
        """
        chakras = list(self.CHAKRAS.values())
        positions = [c["position"] for c in chakras]
        frequencies = [c["frequency_hz"] for c in chakras]
        
        return np.interp(position, positions, frequencies)
    
    def get_resonant_ratios(self) -> Dict[str, float]:
        """
        Get harmonic ratios between chakra frequencies.
        These form a natural harmonic series.
        """
        freqs = self.get_chakra_frequencies()
        base = freqs[0]  # Root as fundamental
        
        return {
            name: chakra["frequency_hz"] / base
            for name, chakra in self.CHAKRAS.items()
        }


# Antahkarana singleton
ANTAHKARANA = AntahkaranaAxis()


@dataclass
class HumanBodyGolden:
    """
    Golden ratio proportions of the human body.
    Based on Leonardo da Vinci's Vitruvian Man and modern anthropometry.
    
    Reference: goldennumber.net/human-body/
    - Height to navel = φ
    - Navel to top / floor to navel = φ
    - Shoulder width = 1/4 height
    - Arm span = height
    """
    height_m: float = 1.75  # Total height in meters
    weight_kg: float = 70.0  # Weight in kg
    
    @property
    def navel_height(self) -> float:
        """Distance from floor to navel."""
        return self.height_m / PHI  # ≈ 1.08m for 1.75m person
    
    @property
    def shoulder_width(self) -> float:
        """Shoulder width."""
        return self.height_m / 4  # ≈ 0.44m
    
    @property
    def arm_span(self) -> float:
        """Arm span (equals height)."""
        return self.height_m
    
    @property
    def torso_length(self) -> float:
        """From shoulder to navel."""
        return self.height_m - self.navel_height - self.height_m / 8  # Head height
    
    @property
    def lying_length(self) -> float:
        """Length when lying down."""
        return self.height_m
    
    @property
    def lying_width(self) -> float:
        """Width when lying (shoulder width)."""
        return self.shoulder_width
    
    @property
    def mass_distribution(self) -> Dict[str, float]:
        """
        Mass distribution percentages for different body parts.
        When lying on a vibroacoustic plate.
        """
        return {
            "head": 0.08,      # 8% of body mass
            "torso": 0.50,     # 50% 
            "arms": 0.10,      # 10% (5% each)
            "legs": 0.32,      # 32% (16% each)
        }
    
    def get_pressure_at(self, x_norm: float, y_norm: float) -> float:
        """
        Get approximate pressure distribution at normalized position.
        Higher near torso center, lower at extremities.
        """
        # Gaussian centered on torso
        cx, cy = 0.5, 0.4  # Center of mass (slightly higher than geometric center)
        sigma = 0.3
        return np.exp(-((x_norm - cx)**2 + (y_norm - cy)**2) / (2 * sigma**2))


@dataclass
class Exciter:
    """Represents a vibroacoustic exciter."""
    x: float = 0.5  # Normalized [0, 1]
    y: float = 0.5
    phase: float = 0.0  # degrees
    amplitude: float = 1.0
    color: str = "#ff6b6b"
    canvas_id: Optional[int] = None
    label_id: Optional[int] = None
    coupling: float = 1.0  # Current coupling coefficient


class DrawingMode(Enum):
    SELECT = "select"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "freehand"


# ══════════════════════════════════════════════════════════════════════════════
# PLATE LAB TAB
# ══════════════════════════════════════════════════════════════════════════════

class PlateLabTab:
    """
    Plate Lab - FEM Modal Analysis for Vibroacoustic Plates
    """
    
    def __init__(self, parent: tk.Frame, audio_engine=None):
        self.parent = parent
        self.audio_engine = audio_engine  # Optional audio engine from main app
        
        # Create main frame (for compatibility with notebook.add)
        self.frame = tk.Frame(parent, bg=Style.BG_DARK)
        
        # Plate analyzer
        self.analyzer = PlateAnalyzer() if HAS_FEM else None
        
        # Exciters
        self.exciters: List[Exciter] = [
            Exciter(0.25, 0.5, 0, 1.0, "#ff6b6b"),
            Exciter(0.75, 0.5, 0, 1.0, "#6bff6b"),
        ]
        
        # State
        self.modes: List = []
        self.selected_mode_idx: int = 0
        self.is_playing: bool = False
        self.play_thread: Optional[threading.Thread] = None
        
        # Human body integration
        self.human_body = HumanBodyGolden(1.75, 70.0)
        self.show_human_overlay: bool = False
        self.mass_loading_enabled: bool = False
        self.show_antahkarana: bool = True  # Show Sushumna/chakra axis
        
        # Visualization options
        self.show_nodal_lines: bool = True
        self.n_modes_display: int = 12
        
        # Arbitrary frequency test
        self.test_frequency: float = 0.0
        
        # Drawing
        self.drawing_mode = DrawingMode.SELECT
        self.polygon_points: List[Tuple[float, float]] = []
        self.temp_polygon_ids: List[int] = []
        self.dragging_exciter: Optional[int] = None
        
        # Canvas dimensions
        self.canvas_width = 600
        self.canvas_height = 400
        
        # Build UI
        self._build_ui()
        
        # Initial calculation
        self.frame.after(100, self._run_analysis)
    
    # ──────────────────────────────────────────────────────────────────────────
    # UI BUILDING
    # ──────────────────────────────────────────────────────────────────────────
    
    def _build_ui(self):
        """Build the complete UI."""
        # Main container inside self.frame
        main_frame = tk.Frame(self.frame, bg=Style.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls (with scrollbar)
        if HAS_SCROLLABLE_SIDEBAR:
            self.sidebar = ScrollableSidebar(main_frame, width=320)
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            left_panel = self.sidebar.content
        else:
            left_panel = tk.Frame(main_frame, bg=Style.BG_PANEL, width=300)
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            left_panel.pack_propagate(False)
        
        self._build_left_panel(left_panel)
        
        # Right panel - Visualization
        right_panel = tk.Frame(main_frame, bg=Style.BG_PANEL)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._build_right_panel(right_panel)
    
    def _build_left_panel(self, parent):
        """Build left control panel."""
        
        # ── Shape Tools ────────────────────────────────────────────────────────
        shape_frame = tk.LabelFrame(
            parent, text="🔷 Forma Tavola",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        shape_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Shape buttons
        btn_frame = tk.Frame(shape_frame, bg=Style.BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        shapes = [
            ("▭ Rett", DrawingMode.RECTANGLE),
            ("⬭ Ellisse", DrawingMode.ELLIPSE),
            ("⬡ Poligono", DrawingMode.POLYGON),
        ]
        
        for i, (text, mode) in enumerate(shapes):
            btn = tk.Button(
                btn_frame, text=text,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT,
                activebackground=Style.ACCENT_GOLD,
                command=lambda m=mode: self._set_drawing_mode(m)
            )
            btn.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
        btn_frame.columnconfigure((0, 1, 2), weight=1)
        
        # Golden Ovoid button (special shape)
        tk.Button(
            shape_frame, text="🥚 Ovoide Aureo (φ)",
            bg=Style.ACCENT_GOLD, fg=Style.BG_DARK,
            activebackground="#e8c547",
            font=Style.FONT_LABEL,
            command=self._apply_golden_ovoid
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Water molecule button (H₂O geometry - 104.5°)
        tk.Button(
            shape_frame, text="💧 H₂O Acquatico (104.5°)",
            bg="#4488ff", fg="white",
            activebackground="#66aaff",
            font=Style.FONT_LABEL,
            command=self._apply_water_molecule_shape
        ).pack(fill=tk.X, padx=5, pady=2)
        
        # Butterfly shape button
        tk.Button(
            shape_frame, text="🦋 Farfalla",
            bg="#aa55cc", fg="white",
            activebackground="#cc77ee",
            font=Style.FONT_LABEL,
            command=self._apply_butterfly_shape
        ).pack(fill=tk.X, padx=5, pady=2)
        
        # Auto-optimizer button
        tk.Button(
            shape_frame, text="🔮 Auto-Ottimizza (AI)",
            bg="#228B22", fg="white",
            activebackground="#32CD32",
            font=Style.FONT_LABEL,
            command=self._run_auto_optimizer
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Shape parameters
        param_frame = tk.Frame(shape_frame, bg=Style.BG_PANEL)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Length
        tk.Label(param_frame, text="Lunghezza (m):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.length_var = tk.StringVar(value="1.95")
        tk.Entry(param_frame, textvariable=self.length_var, width=8,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT).grid(row=0, column=1, padx=5)
        
        # Width
        tk.Label(param_frame, text="Larghezza (m):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).grid(row=1, column=0, sticky="w")
        self.width_var = tk.StringVar(value="0.60")
        tk.Entry(param_frame, textvariable=self.width_var, width=8,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT).grid(row=1, column=1, padx=5)
        
        # Thickness
        tk.Label(param_frame, text="Spessore (mm):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).grid(row=2, column=0, sticky="w")
        self.thickness_var = tk.StringVar(value="10")
        tk.Entry(param_frame, textvariable=self.thickness_var, width=8,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT).grid(row=2, column=1, padx=5)
        
        # ── Material ───────────────────────────────────────────────────────────
        mat_frame = tk.LabelFrame(
            parent, text="🪵 Materiale",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        mat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.material_var = tk.StringVar(value="spruce")
        
        materials_list = list(MATERIALS.keys()) if MATERIALS else ["spruce", "birch_plywood", "mdf", "aluminum"]
        
        for mat in materials_list:
            rb = tk.Radiobutton(
                mat_frame, text=mat.replace("_", " ").title(),
                variable=self.material_var, value=mat,
                bg=Style.BG_PANEL, fg=Style.TEXT_LIGHT,
                selectcolor=Style.BG_DARK,
                activebackground=Style.BG_PANEL,
                activeforeground=Style.ACCENT_GOLD
            )
            rb.pack(anchor="w", padx=10)
        
        # ── Human Body Integration ─────────────────────────────────────────────
        human_frame = tk.LabelFrame(
            parent, text="👤 Corpo Umano (φ)",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        human_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Human dimensions
        human_dim_frame = tk.Frame(human_frame, bg=Style.BG_PANEL)
        human_dim_frame.pack(fill=tk.X, padx=5, pady=3)
        
        tk.Label(human_dim_frame, text="Altezza (m):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).grid(row=0, column=0, sticky="w")
        self.human_height_var = tk.StringVar(value="1.75")
        tk.Entry(human_dim_frame, textvariable=self.human_height_var, width=6,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT).grid(row=0, column=1, padx=3)
        
        tk.Label(human_dim_frame, text="Peso (kg):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).grid(row=1, column=0, sticky="w")
        self.human_weight_var = tk.StringVar(value="70")
        tk.Entry(human_dim_frame, textvariable=self.human_weight_var, width=6,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT).grid(row=1, column=1, padx=3)
        
        # Golden body button
        tk.Button(
            human_frame, text="🧍 Adatta Tavola al Corpo",
            bg="#8B4513", fg="white",
            font=Style.FONT_LABEL,
            command=self._apply_human_body_plate
        ).pack(fill=tk.X, padx=5, pady=3)
        
        # Checkboxes
        self.show_human_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            human_frame, text="Mostra silhouette umana",
            variable=self.show_human_var,
            bg=Style.BG_PANEL, fg=Style.TEXT_LIGHT,
            selectcolor=Style.BG_DARK,
            command=self._toggle_human_overlay
        ).pack(anchor="w", padx=10)
        
        self.mass_loading_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            human_frame, text="Simula carico massa (Δf)",
            variable=self.mass_loading_var,
            bg=Style.BG_PANEL, fg=Style.TEXT_LIGHT,
            selectcolor=Style.BG_DARK,
            command=self._toggle_mass_loading
        ).pack(anchor="w", padx=10)
        
        # Antahkarana axis checkbox
        self.show_antahkarana_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            human_frame, text="🌈 Antahkarana (Sushumna)",
            variable=self.show_antahkarana_var,
            bg=Style.BG_PANEL, fg="#ffd700",
            selectcolor=Style.BG_DARK,
            command=self._toggle_antahkarana
        ).pack(anchor="w", padx=10)
        
        # Info label
        self.human_info_label = tk.Label(
            human_frame, text="φ = 1.618 | Ombelico = H/φ",
            bg=Style.BG_PANEL, fg=Style.TEXT_MUTED,
            font=("Consolas", 8)
        )
        self.human_info_label.pack(anchor="w", padx=10, pady=3)
        
        # ── Exciters ───────────────────────────────────────────────────────────
        exc_frame = tk.LabelFrame(
            parent, text="🔊 Eccitatori",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        exc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add/Remove buttons
        exc_btn_frame = tk.Frame(exc_frame, bg=Style.BG_PANEL)
        exc_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(
            exc_btn_frame, text="➕ Aggiungi",
            bg=Style.SUCCESS, fg="white",
            command=self._add_exciter
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            exc_btn_frame, text="➖ Rimuovi",
            bg=Style.ERROR, fg="white",
            command=self._remove_exciter
        ).pack(side=tk.LEFT, padx=2)
        
        # Exciter list
        self.exciter_listbox = tk.Listbox(
            exc_frame, height=4,
            bg=Style.BG_DARK, fg=Style.TEXT_LIGHT,
            selectbackground=Style.ACCENT_GOLD,
            font=Style.FONT_MONO
        )
        self.exciter_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.exciter_listbox.bind("<<ListboxSelect>>", self._on_exciter_select)
        
        # Phase control for selected exciter
        phase_frame = tk.Frame(exc_frame, bg=Style.BG_PANEL)
        phase_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(phase_frame, text="Fase (°):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).pack(side=tk.LEFT)
        
        self.phase_var = tk.DoubleVar(value=0)
        self.phase_scale = tk.Scale(
            phase_frame, from_=0, to=360,
            variable=self.phase_var, orient=tk.HORIZONTAL,
            bg=Style.BG_PANEL, fg=Style.TEXT_LIGHT,
            troughcolor=Style.BG_DARK,
            highlightthickness=0,
            command=self._on_phase_change
        )
        self.phase_scale.pack(fill=tk.X, expand=True, padx=5)
        
        # ── Analysis ───────────────────────────────────────────────────────────
        ana_frame = tk.LabelFrame(
            parent, text="📊 Analisi Modale",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        ana_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Run analysis button
        tk.Button(
            ana_frame, text="▶️ ANALIZZA",
            bg=Style.ACCENT_GOLD, fg=Style.BG_DARK,
            font=Style.FONT_HEADER,
            command=self._run_analysis
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Mode listbox
        tk.Label(ana_frame, text="Modi trovati:", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).pack(anchor="w", padx=5)
        
        mode_list_frame = tk.Frame(ana_frame, bg=Style.BG_PANEL)
        mode_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(mode_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.mode_listbox = tk.Listbox(
            mode_list_frame, height=8,
            bg=Style.BG_DARK, fg=Style.TEXT_LIGHT,
            selectbackground=Style.ACCENT_GOLD,
            font=Style.FONT_MONO,
            yscrollcommand=scrollbar.set
        )
        self.mode_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.mode_listbox.yview)
        self.mode_listbox.bind("<<ListboxSelect>>", self._on_mode_select)
        
        # Play button
        self.play_btn = tk.Button(
            ana_frame, text="🔊 Riproduci Modo",
            bg=Style.ACCENT_BLUE, fg="white",
            font=Style.FONT_LABEL,
            command=self._toggle_play
        )
        self.play_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Coupling info
        self.coupling_label = tk.Label(
            ana_frame, text="Coupling: --",
            bg=Style.BG_PANEL, fg=Style.TEXT_MUTED,
            font=Style.FONT_MONO
        )
        self.coupling_label.pack(anchor="w", padx=5, pady=5)
        
        # ── Frequenza Arbitraria ───────────────────────────────────────────────
        freq_frame = tk.LabelFrame(
            parent, text="🎚️ Test Frequenza",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        freq_frame.pack(fill=tk.X, padx=5, pady=5)
        
        freq_input_frame = tk.Frame(freq_frame, bg=Style.BG_PANEL)
        freq_input_frame.pack(fill=tk.X, padx=5, pady=3)
        
        tk.Label(freq_input_frame, text="Freq (Hz):", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).pack(side=tk.LEFT)
        
        self.test_freq_var = tk.StringVar(value="40")
        tk.Entry(freq_input_frame, textvariable=self.test_freq_var, width=8,
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            freq_input_frame, text="▶",
            bg=Style.ACCENT_BLUE, fg="white",
            command=self._play_test_frequency
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            freq_input_frame, text="⏹",
            bg=Style.ERROR, fg="white",
            command=self._stop_play
        ).pack(side=tk.LEFT, padx=2)
        
        # Quick frequency buttons
        quick_freq_frame = tk.Frame(freq_frame, bg=Style.BG_PANEL)
        quick_freq_frame.pack(fill=tk.X, padx=5, pady=3)
        
        quick_freqs = [40, 60, 80, 100, 120, 432]
        for f in quick_freqs:
            tk.Button(
                quick_freq_frame, text=f"{f}",
                bg=Style.BG_DARK, fg=Style.TEXT_LIGHT,
                width=4,
                command=lambda freq=f: self._set_and_play_frequency(freq)
            ).pack(side=tk.LEFT, padx=1)
        
        # Response indicator
        self.freq_response_label = tk.Label(
            freq_frame, text="Risposta: --",
            bg=Style.BG_PANEL, fg=Style.TEXT_MUTED,
            font=Style.FONT_MONO
        )
        self.freq_response_label.pack(anchor="w", padx=5, pady=3)
        
        # ── Visualizzazione ────────────────────────────────────────────────────
        vis_frame = tk.LabelFrame(
            parent, text="👁️ Visualizzazione",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        )
        vis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_nodal_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            vis_frame, text="Mostra linee nodali",
            variable=self.show_nodal_var,
            bg=Style.BG_PANEL, fg=Style.TEXT_LIGHT,
            selectcolor=Style.BG_DARK,
            command=self._update_visualization
        ).pack(anchor="w", padx=10)
        
        # Number of modes slider
        modes_frame = tk.Frame(vis_frame, bg=Style.BG_PANEL)
        modes_frame.pack(fill=tk.X, padx=5, pady=3)
        
        tk.Label(modes_frame, text="N° Modi:", bg=Style.BG_PANEL,
                fg=Style.TEXT_LIGHT, font=Style.FONT_LABEL).pack(side=tk.LEFT)
        
        self.n_modes_var = tk.IntVar(value=12)
        tk.Scale(
            modes_frame, from_=4, to=24, orient=tk.HORIZONTAL,
            variable=self.n_modes_var,
            bg=Style.BG_PANEL, fg=Style.TEXT_LIGHT,
            troughcolor=Style.BG_DARK,
            highlightthickness=0,
            command=lambda v: setattr(self, 'n_modes_display', int(v))
        ).pack(fill=tk.X, expand=True, padx=5)
        
        # Update exciter list
        self._update_exciter_list()
    
    def _build_right_panel(self, parent):
        """Build right visualization panel."""
        
        # Header
        header = tk.Frame(parent, bg=Style.BG_PANEL)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            header, text="🔬 Visualizzazione Modale FEM",
            bg=Style.BG_PANEL, fg=Style.ACCENT_GOLD,
            font=Style.FONT_HEADER
        ).pack(side=tk.LEFT)
        
        # Status
        self.status_label = tk.Label(
            header, text="Pronto",
            bg=Style.BG_PANEL, fg=Style.TEXT_MUTED,
            font=Style.FONT_LABEL
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Canvas frame
        canvas_frame = tk.Frame(parent, bg=Style.BG_DARK)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#000020",
            highlightthickness=2,
            highlightbackground=Style.ACCENT_GOLD
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
        # Legend
        legend_frame = tk.Frame(parent, bg=Style.BG_PANEL)
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(legend_frame, text="🔴 Antinode (max)", bg=Style.BG_PANEL,
                fg="#ff4444", font=Style.FONT_LABEL).pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🔵 Node (min)", bg=Style.BG_PANEL,
                fg="#4444ff", font=Style.FONT_LABEL).pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="⚪ Trascina eccitatori | Doppio-click per chiudere poligono",
                bg=Style.BG_PANEL, fg=Style.TEXT_MUTED,
                font=Style.FONT_LABEL).pack(side=tk.RIGHT, padx=10)
    
    # ──────────────────────────────────────────────────────────────────────────
    # DRAWING MODES
    # ──────────────────────────────────────────────────────────────────────────
    
    def _set_drawing_mode(self, mode: DrawingMode):
        """Set current drawing mode."""
        self.drawing_mode = mode
        self.polygon_points = []
        self._clear_temp_polygon()
        
        if mode == DrawingMode.RECTANGLE:
            self._apply_rectangle()
        elif mode == DrawingMode.ELLIPSE:
            self._apply_ellipse()
        elif mode == DrawingMode.POLYGON:
            self.status_label.config(text="Clicca per aggiungere vertici, doppio-click per chiudere")
    
    def _apply_rectangle(self):
        """Apply rectangle shape."""
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
            if self.analyzer:
                self.analyzer.set_rectangle(L, W)
            self.status_label.config(text=f"Rettangolo {L}x{W}m")
            self._run_analysis()
        except ValueError:
            messagebox.showerror("Errore", "Dimensioni non valide")
    
    def _apply_ellipse(self):
        """Apply ellipse shape."""
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
            if self.analyzer:
                self.analyzer.set_ellipse(L/2, W/2)
            self.status_label.config(text=f"Ellisse {L}x{W}m")
            self._run_analysis()
        except ValueError:
            messagebox.showerror("Errore", "Dimensioni non valide")
    
    def _apply_golden_ovoid(self):
        """Apply Golden Ovoid shape based on φ (golden ratio)."""
        try:
            L = float(self.length_var.get())
        except:
            L = 1.95
        
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        
        # Golden ovoid proportions
        a = L / 2  # Semi-major axis
        b = a / phi  # Semi-minor axis (golden ratio)
        
        # Update width to match golden proportion
        self.width_var.set(f"{2*b:.2f}")
        
        # Generate ovoid points
        n_points = 36
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        # Golden ovoid parametric equation (egg-shaped)
        k = 0.08  # Asymmetry factor (subtle egg shape)
        
        points = []
        for t in theta:
            # Ellipse base
            r = (a * b) / np.sqrt((b * np.cos(t))**2 + (a * np.sin(t))**2)
            # Add golden asymmetry (narrower at HEAD end)
            r *= (1 + k * np.cos(t))
            
            x = r * np.cos(t) + a  # Shift to positive x
            y = r * np.sin(t) + b  # Shift to positive y
            points.append((x, y))
        
        if self.analyzer:
            self.analyzer.set_polygon(points)
        
        self.status_label.config(text=f"Ovoide Aureo φ={phi:.3f} ({L}x{2*b:.2f}m)")
        self._run_analysis()
    
    def _apply_water_molecule_shape(self):
        """
        Apply plate shape based on H₂O water molecule geometry.
        
        The human body is ~60-70% water, making this geometry
        potentially resonant with our biological structure.
        
        H₂O geometry:
        - Bond angle: 104.5° (due to sp³ hybridization and lone pairs)
        - Creates a bent/angular shape
        - The plate mimics the triangular molecular shape
        """
        try:
            base_scale = float(self.length_var.get())
        except:
            base_scale = 1.95
        
        # Get water molecule geometry
        water = WATER_GEOMETRY
        angle_rad = water.bond_angle_rad
        half_angle = water.half_angle_rad
        
        # Calculate plate dimensions from H₂O geometry
        # Scale the molecular shape to fit human body
        
        # The "length" is along the H-H axis
        # The "width" is the perpendicular distance from O to H-H line
        h_h_length = base_scale  # Full length for H-H distance
        o_height = (base_scale / 2) / np.tan(half_angle)  # Height to O from H-H midpoint
        
        # Create water molecule shaped plate
        # Smooth curved shape that follows the H₂O angular geometry
        n_points = 48
        
        # Center points for the three "atoms"
        cx = base_scale / 2
        cy = o_height / 2
        
        # H atoms at the base corners
        h1_x = 0
        h1_y = 0
        h2_x = base_scale
        h2_y = 0
        
        # O atom at the top (apex)
        o_x = cx
        o_y = o_height
        
        # Create smooth curved shape connecting the three points
        # Using quadratic bezier-like curves
        points = []
        
        # Generate smooth water-drop/bent shape
        for i in range(n_points):
            t = i / n_points * 2 * np.pi
            
            # Parametric shape that's wider at bottom (H-H) and pointed at top (O)
            # Uses the 104.5° angle to shape the curvature
            
            if t < np.pi:  # Upper half (toward O)
                # Curve from H1 through O to H2
                progress = t / np.pi
                
                # Bezier-like interpolation
                x = (1-progress)**2 * h1_x + 2*(1-progress)*progress * o_x + progress**2 * h2_x
                y = (1-progress)**2 * h1_y + 2*(1-progress)*progress * o_y + progress**2 * h2_y
                
                # Add some width (bulge) based on the bond angle
                bulge = np.sin(t) * 0.15 * base_scale
                y += bulge * np.cos(half_angle)
                
            else:  # Lower half (H-H base)
                progress = (t - np.pi) / np.pi
                
                # Simple arc along the bottom (H1 to H2)
                x = h2_x * (1 - progress) + h1_x * progress
                y = h2_y  # Flat bottom with slight curve
                
                # Subtle inward curve at bottom
                y -= np.sin(t - np.pi) * 0.08 * o_height
            
            points.append((x, y))
        
        # Update width var to reflect the geometry
        self.width_var.set(f"{o_height:.2f}")
        
        if self.analyzer:
            self.analyzer.set_polygon(points)
        
        # Display info
        self.status_label.config(
            text=f"💧 H₂O Molecola: {base_scale:.2f}×{o_height:.2f}m | ∠={water.BOND_ANGLE_DEG}°"
        )
        self._run_analysis()
    
    def _apply_butterfly_shape(self):
        """
        Apply butterfly/wing shape plate.
        
        Symmetric winged shape similar to water molecule but more elongated.
        The butterfly curve has special acoustic properties due to its
        symmetric lobes that can create complementary resonances.
        """
        try:
            base_scale = float(self.length_var.get())
        except:
            base_scale = 1.95
        
        n_points = 80
        points = []
        
        # Butterfly parametric curve (modified for plate shape)
        for i in range(n_points):
            t = 2 * np.pi * i / n_points
            
            # Butterfly curve formula
            # r = e^sin(t) - 2*cos(4t) + sin((2t-π)/24)^5
            r = np.exp(np.sin(t)) - 2 * np.cos(4 * t) + np.sin((2 * t - np.pi) / 24) ** 5
            r = abs(r) * base_scale / 8
            
            x = r * np.cos(t)
            y = r * np.sin(t)
            
            points.append((x, y))
        
        # Normalize to positive coordinates
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        points = [(p[0] - min_x, p[1] - min_y) for p in points]
        
        # Scale to fit length
        max_x = max(p[0] for p in points) or 1
        scale = base_scale / max_x
        points = [(p[0] * scale, p[1] * scale) for p in points]
        
        # Calculate width
        width = max(p[1] for p in points) - min(p[1] for p in points)
        self.width_var.set(f"{width:.2f}")
        
        if self.analyzer:
            self.analyzer.set_polygon(points)
        
        self.status_label.config(
            text=f"🦋 Farfalla: {base_scale:.2f}×{width:.2f}m"
        )
        self._run_analysis()
    
    def _run_auto_optimizer(self):
        """
        Run the physics-based plate optimizer.
        
        Tests multiple shapes and finds the one that best:
        - Matches chakra frequencies along the spine
        - Covers the musical frequency range
        - Provides uniform vibration distribution
        - Maintains golden ratio proportions
        """
        try:
            from core.plate_optimizer import PlateOptimizer, HumanBodyParams, auto_design_plate
        except ImportError:
            messagebox.showerror(
                "Errore",
                "Modulo plate_optimizer non trovato.\n"
                "Assicurati che core/plate_optimizer.py esista."
            )
            return
        
        # Get current parameters
        try:
            height = float(self.human_height_var.get())
            thickness_mm = float(self.thickness_var.get())
        except:
            height, thickness_mm = 1.75, 15.0
        
        material = self.material_var.get()
        
        self.status_label.config(text="🔮 Ottimizzazione in corso...")
        self.frame.update()
        
        try:
            # Run optimization
            result = auto_design_plate(
                height_m=height,
                material=material,
                thickness_mm=thickness_mm,
                template=None,  # Try all templates
                with_cutouts=False
            )
            
            if result:
                # Apply the best shape
                self.length_var.set(f"{result.length:.2f}")
                self.width_var.set(f"{result.width:.2f}")
                
                # Set polygon if available
                if result.vertices and self.analyzer:
                    self.analyzer.set_polygon(result.vertices)
                
                # Update modes from optimization
                if result.modes:
                    self.modes = result.modes
                    self._update_mode_list()
                    self._update_visualization()
                
                # Show result
                msg = (
                    f"✅ Forma ottimale: {result.template.value}\n"
                    f"Dimensioni: {result.length:.2f}m × {result.width:.2f}m\n"
                    f"Chakra Error: {result.chakra_error:.3f}\n"
                    f"Copertura freq: {result.frequency_coverage:.0%}\n"
                    f"Score totale: {result.total_score:.2f}"
                )
                messagebox.showinfo("Ottimizzazione Completata", msg)
                
                self.status_label.config(
                    text=f"🔮 Ottimo: {result.template.value} | Score: {result.total_score:.2f}"
                )
            else:
                self.status_label.config(text="❌ Ottimizzazione fallita")
                
        except Exception as e:
            self.status_label.config(text=f"❌ Errore: {str(e)[:50]}")
            messagebox.showerror("Errore Ottimizzazione", str(e))
    
    # ──────────────────────────────────────────────────────────────────────────
    # HUMAN BODY INTEGRATION
    # ──────────────────────────────────────────────────────────────────────────
    
    def _apply_human_body_plate(self):
        """
        Create plate shaped for human body using golden ratio proportions.
        Based on Leonardo's Vitruvian Man and anthropometry.
        """
        try:
            height = float(self.human_height_var.get())
            weight = float(self.human_weight_var.get())
        except:
            height, weight = 1.75, 70.0
        
        # Update human body model
        self.human_body = HumanBodyGolden(height, weight)
        
        # Plate dimensions based on lying human
        # Length = height + small margin
        plate_length = self.human_body.lying_length * 1.1  # 10% margin
        
        # Width = shoulder width × φ (golden proportion for comfort)
        plate_width = self.human_body.shoulder_width * PHI
        
        # Update dimension fields
        self.length_var.set(f"{plate_length:.2f}")
        self.width_var.set(f"{plate_width:.2f}")
        
        # Create golden ovoid shape for human body
        a = plate_length / 2
        b = plate_width / 2
        
        # Human body contour (more egg-shaped for head/feet)
        n_points = 48
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        # Asymmetric egg shape with golden ratio
        k = 0.12  # More pronounced asymmetry for head end
        
        points = []
        for t in theta:
            # Base ellipse
            r = (a * b) / np.sqrt((b * np.cos(t))**2 + (a * np.sin(t))**2)
            
            # Narrower at head (top), wider at torso
            # cos(t)=1 is head end, cos(t)=-1 is feet end
            asymmetry = 1 + k * np.cos(t) - 0.03 * np.cos(2*t)
            r *= asymmetry
            
            x = r * np.cos(t) + a
            y = r * np.sin(t) + b
            points.append((x, y))
        
        if self.analyzer:
            self.analyzer.set_polygon(points)
        
        # Update info label
        navel_h = self.human_body.navel_height
        self.human_info_label.config(
            text=f"H={height}m | Ombelico={navel_h:.2f}m | Massa={weight}kg"
        )
        
        self.status_label.config(
            text=f"Tavola Corpo Aureo: {plate_length:.2f}x{plate_width:.2f}m"
        )
        self._run_analysis()
    
    def _toggle_human_overlay(self):
        """Toggle human body silhouette overlay."""
        self.show_human_overlay = self.show_human_var.get()
        self._update_visualization()
    
    def _toggle_antahkarana(self):
        """Toggle Antahkarana (Sushumna/chakra axis) display."""
        self.show_antahkarana = self.show_antahkarana_var.get()
        self._update_visualization()
    
    def _toggle_mass_loading(self):
        """
        Toggle mass loading simulation.
        When enabled, modal frequencies shift down due to added mass.
        
        Formula (Rayleigh's method):
        f_loaded ≈ f_0 / √(1 + m_human/m_eff)
        
        where m_eff is the effective modal mass of the plate.
        """
        self.mass_loading_enabled = self.mass_loading_var.get()
        
        if self.mass_loading_enabled:
            self._apply_mass_loading()
        else:
            # Re-run analysis without mass loading
            self._run_analysis()
    
    def _apply_mass_loading(self):
        """
        Apply human mass loading to modal frequencies.
        
        Improved physics model:
        - Uses position-dependent mass distribution
        - Considers body contact area (~0.5 m²)
        - Mode-specific effective mass ratios
        - Damping increase from body contact
        
        Formula (extended Rayleigh):
        f_loaded = f_0 / √(1 + Σ(m_i * φ_i²) / m_eff)
        
        where φ_i is mode shape at contact point i
        """
        if not self.modes:
            return
        
        try:
            weight = float(self.human_weight_var.get())
            height = float(self.human_height_var.get())
        except:
            weight = 70.0
            height = 1.75
        
        # Calculate plate mass
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
            h = float(self.thickness_var.get()) / 1000.0  # mm to m
        except:
            L, W, h = 1.95, 0.6, 0.01
        
        # Get material density
        mat_key = self.material_var.get()
        density = 500  # Default for wood
        if mat_key in MATERIALS and MATERIALS[mat_key]:
            density = MATERIALS[mat_key].density
        
        plate_mass = L * W * h * density
        
        # Human body mass distribution along spine (from Antahkarana)
        # Position: fraction of plate length from head
        body_mass_distribution = {
            "head": (0.08, 0.05),      # 8% mass at 5% from head
            "upper_torso": (0.15, 0.20),
            "mid_torso": (0.25, 0.35), # Heaviest region
            "lower_torso": (0.20, 0.50),
            "upper_legs": (0.18, 0.70),
            "lower_legs": (0.14, 0.90),
        }
        
        # Contact area affects coupling
        contact_area = 0.5  # m² (approx body contact area lying down)
        coupling_factor = min(1.0, contact_area / (L * W))
        
        # Update mode frequencies
        self.mode_listbox.delete(0, tk.END)
        
        for i, mode in enumerate(self.modes):
            # Calculate weighted mass coupling for this mode
            total_weighted_mass = 0.0
            
            for part, (mass_frac, pos_frac) in body_mass_distribution.items():
                # Position on plate
                x = pos_frac * L
                y = W / 2  # Center of width
                
                # Get mode shape value at this position
                try:
                    if hasattr(mode, 'mode_shape') and len(mode.mode_shape) > 0:
                        # FEM mode - use interpolation
                        from scipy.interpolate import LinearNDInterpolator
                        interp = LinearNDInterpolator(mode.mesh_points, mode.mode_shape)
                        phi = float(interp(x, y))
                        if np.isnan(phi):
                            phi = 0.5
                    else:
                        # Analytical approximation
                        m_idx = (mode.index // 3) + 1
                        n_idx = (mode.index % 3) + 1
                        phi = np.cos(m_idx * np.pi * x / L) * np.cos(n_idx * np.pi * y / W)
                except:
                    phi = 0.5
                
                # Weighted mass contribution (φ² is key for kinetic energy)
                part_mass = weight * mass_frac
                total_weighted_mass += part_mass * phi**2
            
            # Effective modal mass of plate (typically ~25% of total mass for mode 1)
            modal_mass_factor = 0.25 / (1 + 0.5 * i)  # Decreases with mode number
            m_eff_plate = plate_mass * modal_mass_factor
            
            # Mass ratio with coupling
            m_ratio = (total_weighted_mass * coupling_factor) / m_eff_plate
            
            # Damping increase from body contact (empirical)
            damping_increase = 1 + 0.1 * coupling_factor
            
            # Frequency shift (Rayleigh quotient)
            f_original = mode.frequency
            f_loaded = f_original / np.sqrt(1 + m_ratio)
            
            # Apply damping effect (slight additional reduction)
            f_loaded = f_loaded / np.sqrt(damping_increase)
            
            delta_f = f_original - f_loaded
            delta_percent = (delta_f / f_original) * 100
            
            # Display with percentage shift
            self.mode_listbox.insert(
                tk.END,
                f"M{i+1}: {f_loaded:.1f} Hz (Δ-{delta_f:.1f} = -{delta_percent:.0f}%)"
            )
        
        if self.modes:
            self.mode_listbox.selection_set(0)
        
        # Update status with more info
        mass_ratio_overall = weight / plate_mass
        self.status_label.config(
            text=f"Carico: {weight}kg ({mass_ratio_overall:.1f}× massa tavola {plate_mass:.1f}kg)"
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # ARBITRARY FREQUENCY TEST
    # ──────────────────────────────────────────────────────────────────────────
    
    def _play_test_frequency(self):
        """Play arbitrary test frequency."""
        try:
            freq = float(self.test_freq_var.get())
        except:
            messagebox.showerror("Errore", "Frequenza non valida")
            return
        
        self.test_frequency = freq
        self._calculate_frequency_response(freq)
        self._play_frequency(freq)
    
    def _set_and_play_frequency(self, freq: float):
        """Set and play a frequency."""
        self.test_freq_var.set(str(freq))
        self._play_test_frequency()
    
    def _calculate_frequency_response(self, freq: float):
        """
        Calculate how the plate responds to an arbitrary frequency.
        Show which modes are excited and with what amplitude.
        """
        if not self.modes:
            self.freq_response_label.config(text="Nessun modo calcolato")
            return
        
        response_info = []
        total_response = 0
        
        for i, mode in enumerate(self.modes):
            # Modal response using single-DOF resonance
            f_mode = mode.frequency
            
            # Damping ratio (typical for wood)
            zeta = 0.02
            
            # Frequency ratio
            r = freq / f_mode if f_mode > 0 else 0
            
            # Magnification factor
            if r > 0:
                H = 1 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
            else:
                H = 1
            
            # Weight by exciter coupling
            avg_coupling = np.mean([exc.coupling for exc in self.exciters])
            response = H * avg_coupling
            total_response += response
            
            if H > 1.5:  # Significant response
                response_info.append(f"M{i+1}:{H:.1f}")
        
        # Find closest mode
        closest_mode = min(self.modes, key=lambda m: abs(m.frequency - freq))
        closest_idx = self.modes.index(closest_mode)
        
        # Update label
        if response_info:
            self.freq_response_label.config(
                text=f"{freq}Hz → {', '.join(response_info[:3])} | Vicino: M{closest_idx+1}"
            )
        else:
            self.freq_response_label.config(
                text=f"{freq}Hz → Risposta bassa | Vicino: M{closest_idx+1} ({closest_mode.frequency:.0f}Hz)"
            )
    
    def _play_frequency(self, freq: float):
        """Play a single frequency."""
        if not HAS_AUDIO:
            return
        
        self.is_playing = True
        self.play_btn.config(text="⏹ Stop", bg=Style.ERROR)
        
        def play():
            try:
                sample_rate = 44100
                duration = 3.0
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                audio = np.zeros_like(t)
                
                for exc in self.exciters:
                    phase_rad = np.radians(exc.phase)
                    amplitude = exc.amplitude * exc.coupling
                    audio += amplitude * np.sin(2 * np.pi * freq * t + phase_rad)
                
                # Normalize
                max_amp = np.max(np.abs(audio))
                if max_amp > 0:
                    audio = audio / max_amp * 0.7
                
                # Envelope
                attack = int(0.05 * sample_rate)
                release = int(0.2 * sample_rate)
                envelope = np.ones_like(audio)
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                audio *= envelope
                
                sd.play(audio.astype(np.float32), sample_rate)
                
                while self.is_playing and sd.get_stream().active:
                    time.sleep(0.1)
                
                sd.stop()
            except Exception as e:
                print(f"Audio error: {e}")
            finally:
                self.is_playing = False
                self.parent.after(0, lambda: self.play_btn.config(
                    text="🔊 Riproduci Modo", bg=Style.ACCENT_BLUE
                ))
        
        threading.Thread(target=play, daemon=True).start()
    
    def _clear_temp_polygon(self):
        """Clear temporary polygon drawing."""
        for item_id in self.temp_polygon_ids:
            self.canvas.delete(item_id)
        self.temp_polygon_ids = []
    
    def _finish_polygon(self):
        """Finish polygon drawing."""
        if len(self.polygon_points) < 3:
            self.status_label.config(text="Servono almeno 3 punti")
            return
        
        # Convert canvas coords to real coords
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
        except:
            L, W = 1.95, 0.6
        
        real_points = []
        for px, py in self.polygon_points:
            x = (px / self.canvas_width) * L
            y = (1 - py / self.canvas_height) * W  # Flip Y
            real_points.append((x, y))
        
        if self.analyzer:
            self.analyzer.set_polygon(real_points)
        
        self.status_label.config(text=f"Poligono con {len(real_points)} vertici")
        self._clear_temp_polygon()
        self.polygon_points = []
        self.drawing_mode = DrawingMode.SELECT
        self._run_analysis()
    
    # ──────────────────────────────────────────────────────────────────────────
    # EXCITER MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────────
    
    def _add_exciter(self):
        """Add a new exciter."""
        if len(self.exciters) >= 4:
            messagebox.showwarning("Limite", "Massimo 4 eccitatori")
            return
        
        colors = ["#ff6b6b", "#6bff6b", "#6b6bff", "#ffff6b"]
        color = colors[len(self.exciters) % len(colors)]
        
        # Place at random position
        x = 0.3 + 0.4 * np.random.random()
        y = 0.3 + 0.4 * np.random.random()
        
        self.exciters.append(Exciter(x, y, 0, 1.0, color))
        self._update_exciter_list()
        self._update_visualization()
    
    def _remove_exciter(self):
        """Remove selected exciter."""
        if len(self.exciters) <= 1:
            messagebox.showwarning("Limite", "Serve almeno 1 eccitatore")
            return
        
        sel = self.exciter_listbox.curselection()
        if sel:
            idx = sel[0]
            exc = self.exciters[idx]
            if exc.canvas_id:
                self.canvas.delete(exc.canvas_id)
            if exc.label_id:
                self.canvas.delete(exc.label_id)
            del self.exciters[idx]
        else:
            # Remove last
            exc = self.exciters.pop()
            if exc.canvas_id:
                self.canvas.delete(exc.canvas_id)
            if exc.label_id:
                self.canvas.delete(exc.label_id)
        
        self._update_exciter_list()
        self._update_visualization()
    
    def _update_exciter_list(self):
        """Update the exciter listbox."""
        self.exciter_listbox.delete(0, tk.END)
        for i, exc in enumerate(self.exciters):
            coupling_str = f"{exc.coupling:.2f}" if exc.coupling else "?"
            self.exciter_listbox.insert(
                tk.END,
                f"Exc {i+1}: ({exc.x:.2f}, {exc.y:.2f}) φ={exc.phase:.0f}° C={coupling_str}"
            )
    
    def _on_exciter_select(self, event):
        """Handle exciter selection."""
        sel = self.exciter_listbox.curselection()
        if sel:
            idx = sel[0]
            self.phase_var.set(self.exciters[idx].phase)
    
    def _on_phase_change(self, value):
        """Handle phase slider change."""
        sel = self.exciter_listbox.curselection()
        if sel:
            idx = sel[0]
            self.exciters[idx].phase = float(value)
            self._update_exciter_list()
            self._update_coupling_display()
    
    # ──────────────────────────────────────────────────────────────────────────
    # CANVAS EVENTS
    # ──────────────────────────────────────────────────────────────────────────
    
    def _on_canvas_click(self, event):
        """Handle canvas click."""
        x, y = event.x, event.y
        
        if self.drawing_mode == DrawingMode.POLYGON:
            # Add point to polygon
            self.polygon_points.append((x, y))
            
            # Draw point
            r = 5
            pt_id = self.canvas.create_oval(
                x-r, y-r, x+r, y+r,
                fill=Style.ACCENT_GOLD, outline="white"
            )
            self.temp_polygon_ids.append(pt_id)
            
            # Draw line to previous point
            if len(self.polygon_points) > 1:
                px, py = self.polygon_points[-2]
                line_id = self.canvas.create_line(
                    px, py, x, y,
                    fill=Style.ACCENT_GOLD, width=2
                )
                self.temp_polygon_ids.append(line_id)
            
            return
        
        # Check if clicking on exciter
        for i, exc in enumerate(self.exciters):
            cx = exc.x * self.canvas_width
            cy = (1 - exc.y) * self.canvas_height  # Flip Y
            
            if abs(x - cx) < 20 and abs(y - cy) < 20:
                self.dragging_exciter = i
                return
    
    def _on_canvas_drag(self, event):
        """Handle canvas drag."""
        if self.dragging_exciter is not None:
            margin = 20
            x = max(margin, min(event.x, self.canvas_width - margin))
            y = max(margin, min(event.y, self.canvas_height - margin))
            
            # Convert to normalized coords (within plate area)
            exc = self.exciters[self.dragging_exciter]
            exc.x = (x - margin) / (self.canvas_width - 2 * margin)
            exc.y = 1 - (y - margin) / (self.canvas_height - 2 * margin)  # Flip Y
            
            # Clamp to [0, 1]
            exc.x = max(0, min(1, exc.x))
            exc.y = max(0, min(1, exc.y))
            
            # Update coupling in REAL-TIME
            self._update_coupling_for_exciter(self.dragging_exciter)
            
            # Redraw entire visualization to show updated coupling
            self._update_visualization()
    
    def _on_canvas_release(self, event):
        """Handle canvas release."""
        if self.dragging_exciter is not None:
            self.dragging_exciter = None
            self._update_exciter_list()
            self._update_coupling_display()
    
    def _on_canvas_double_click(self, event):
        """Handle double click - finish polygon."""
        if self.drawing_mode == DrawingMode.POLYGON:
            self._finish_polygon()
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self.canvas_width = event.width
        self.canvas_height = event.height
        self._update_visualization()
    
    # ──────────────────────────────────────────────────────────────────────────
    # MODAL ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    
    def _run_analysis(self):
        """Run FEM modal analysis."""
        self.status_label.config(text="Analisi in corso...")
        self.parent.update()
        
        if not self.analyzer:
            self._run_analytical_fallback()
            return
        
        try:
            # Set parameters
            self.analyzer.set_material(self.material_var.get())
            
            thickness_mm = float(self.thickness_var.get())
            self.analyzer.set_thickness(thickness_mm / 1000.0)
            
            # Generate mesh
            self.analyzer.generate_mesh(resolution=20)
            
            # Run analysis
            self.modes = self.analyzer.analyze(n_modes=12)
            
            # Update mode list
            self.mode_listbox.delete(0, tk.END)
            for i, mode in enumerate(self.modes):
                self.mode_listbox.insert(
                    tk.END,
                    f"Modo {i+1}: {mode.frequency:.1f} Hz"
                )
            
            # Select first mode
            if self.modes:
                self.mode_listbox.selection_set(0)
                self.selected_mode_idx = 0
            
            self.status_label.config(text=f"Trovati {len(self.modes)} modi")
            
            # Update visualization
            self._update_visualization()
            self._update_all_couplings()
            
        except Exception as e:
            self.status_label.config(text=f"Errore: {e}")
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_analytical_fallback(self):
        """Fallback to analytical calculation."""
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
            h = float(self.thickness_var.get()) / 1000.0
            
            # Simple rectangular plate modes
            E = 12e9  # Approximate
            rho = 500
            nu = 0.3
            D = E * h**3 / (12 * (1 - nu**2))
            
            self.modes = []
            
            for m in range(1, 5):
                for n in range(1, 5):
                    freq = (np.pi / 2) * np.sqrt(D / (rho * h)) * ((m/L)**2 + (n/W)**2)
                    
                    # Create simple FEMMode-like object
                    mode = type('Mode', (), {
                        'index': len(self.modes),
                        'frequency': freq,
                        'm': m, 'n': n,
                        'mode_name': f"({m},{n})"
                    })()
                    self.modes.append(mode)
            
            self.modes.sort(key=lambda x: x.frequency)
            self.modes = self.modes[:12]
            
            self.mode_listbox.delete(0, tk.END)
            for i, mode in enumerate(self.modes):
                self.mode_listbox.insert(
                    tk.END,
                    f"Modo {i+1}: {mode.frequency:.1f} Hz"
                )
            
            if self.modes:
                self.mode_listbox.selection_set(0)
                self.selected_mode_idx = 0
            
            self.status_label.config(text=f"Analisi analitica - {len(self.modes)} modi")
            self._update_visualization()
            
        except Exception as e:
            self.status_label.config(text=f"Errore: {e}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    
    def _update_visualization(self):
        """Update the heatmap visualization."""
        self.canvas.delete("all")
        
        if not self.modes or self.selected_mode_idx >= len(self.modes):
            self._draw_empty_plate()
            return
        
        mode = self.modes[self.selected_mode_idx]
        
        # Draw heatmap
        if self.analyzer and hasattr(mode, 'mode_shape'):
            self._draw_fem_heatmap(mode)
        else:
            self._draw_analytical_heatmap(mode)
        
        # Draw exciters
        self._draw_exciters()
    
    def _draw_empty_plate(self):
        """Draw empty plate outline."""
        margin = 20
        
        if self.analyzer and self.analyzer.shape == PlateShape.ELLIPSE:
            self.canvas.create_oval(
                margin, margin,
                self.canvas_width - margin, self.canvas_height - margin,
                outline=Style.ACCENT_GOLD, width=2
            )
        else:
            self.canvas.create_rectangle(
                margin, margin,
                self.canvas_width - margin, self.canvas_height - margin,
                outline=Style.ACCENT_GOLD, width=2
            )
        
        self._draw_exciters()
    
    def _draw_fem_heatmap(self, mode):
        """Draw heatmap from FEM mode shape."""
        points = mode.mesh_points
        shape = mode.mode_shape
        
        if len(points) == 0 or len(shape) == 0:
            self._draw_empty_plate()
            return
        
        # Get bounds
        bbox = self.analyzer.get_bounding_box()
        min_x, min_y, max_x, max_y = bbox
        
        range_x = max_x - min_x if max_x > min_x else 1
        range_y = max_y - min_y if max_y > min_y else 1
        
        # Normalize shape values
        shape_norm = shape / (np.max(np.abs(shape)) + 1e-10)
        
        # Draw triangles with colors
        triangles = mode.mesh_triangles
        
        for tri in triangles:
            # Get triangle vertices
            pts = points[tri]
            vals = shape_norm[tri]
            
            # Average value for triangle color
            avg_val = np.mean(vals)
            
            # Map to color (red = positive, blue = negative)
            if avg_val >= 0:
                r = int(255 * avg_val)
                g = int(50 * (1 - avg_val))
                b = int(50 * (1 - avg_val))
            else:
                r = int(50 * (1 + avg_val))
                g = int(50 * (1 + avg_val))
                b = int(255 * (-avg_val))
            
            color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Convert to canvas coords
            canvas_pts = []
            for p in pts:
                cx = ((p[0] - min_x) / range_x) * (self.canvas_width - 40) + 20
                cy = (1 - (p[1] - min_y) / range_y) * (self.canvas_height - 40) + 20
                canvas_pts.extend([cx, cy])
            
            self.canvas.create_polygon(canvas_pts, fill=color, outline="")
        
        # Draw plate outline
        self._draw_plate_outline()
    
    def _draw_analytical_heatmap(self, mode):
        """Draw heatmap using analytical mode shape."""
        margin = 20
        
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
        except:
            L, W = 1.95, 0.6
        
        # Get mode indices
        if hasattr(mode, 'm'):
            m, n = mode.m, mode.n
        else:
            # Estimate from index
            idx = mode.index
            m = (idx // 3) + 1
            n = (idx % 3) + 1
        
        # Draw grid
        res = 30
        cell_w = (self.canvas_width - 2 * margin) / res
        cell_h = (self.canvas_height - 2 * margin) / res
        
        for i in range(res):
            for j in range(res):
                x = i / res
                y = j / res
                
                # Mode shape
                val = np.cos(m * np.pi * x) * np.cos(n * np.pi * y)
                
                # Color
                if val >= 0:
                    r = int(200 * val + 50)
                    g = int(50 * (1 - val))
                    b = int(50 * (1 - val))
                else:
                    r = int(50 * (1 + val))
                    g = int(50 * (1 + val))
                    b = int(200 * (-val) + 50)
                
                color = f"#{r:02x}{g:02x}{b:02x}"
                
                cx = margin + i * cell_w
                cy = margin + j * cell_h
                
                self.canvas.create_rectangle(
                    cx, cy, cx + cell_w, cy + cell_h,
                    fill=color, outline=""
                )
        
        # Plate outline
        self.canvas.create_rectangle(
            margin, margin,
            self.canvas_width - margin, self.canvas_height - margin,
            outline=Style.ACCENT_GOLD, width=2
        )
    
    def _draw_plate_outline(self):
        """Draw plate outline based on shape."""
        margin = 20
        
        if self.analyzer:
            if self.analyzer.shape == PlateShape.ELLIPSE:
                self.canvas.create_oval(
                    margin, margin,
                    self.canvas_width - margin, self.canvas_height - margin,
                    outline=Style.ACCENT_GOLD, width=2
                )
            elif self.analyzer.shape == PlateShape.POLYGON and self.analyzer.polygon_vertices:
                # Draw polygon outline
                bbox = self.analyzer.get_bounding_box()
                min_x, min_y, max_x, max_y = bbox
                range_x = max_x - min_x if max_x > min_x else 1
                range_y = max_y - min_y if max_y > min_y else 1
                
                canvas_pts = []
                for px, py in self.analyzer.polygon_vertices:
                    cx = ((px - min_x) / range_x) * (self.canvas_width - 40) + 20
                    cy = (1 - (py - min_y) / range_y) * (self.canvas_height - 40) + 20
                    canvas_pts.extend([cx, cy])
                
                self.canvas.create_polygon(
                    canvas_pts, fill="",
                    outline=Style.ACCENT_GOLD, width=2
                )
            else:
                self.canvas.create_rectangle(
                    margin, margin,
                    self.canvas_width - margin, self.canvas_height - margin,
                    outline=Style.ACCENT_GOLD, width=2
                )
        
        # Draw nodal lines if enabled
        if hasattr(self, 'show_nodal_var') and self.show_nodal_var.get():
            self._draw_nodal_lines()
        
        # Draw human overlay if enabled
        if hasattr(self, 'show_human_var') and self.show_human_var.get():
            self._draw_human_silhouette()
    
    def _draw_nodal_lines(self):
        """
        Draw nodal lines (where displacement = 0).
        These are the lines that separate antinodes.
        """
        if not self.modes or self.selected_mode_idx >= len(self.modes):
            return
        
        mode = self.modes[self.selected_mode_idx]
        margin = 20
        
        # Get mode indices
        if hasattr(mode, 'm'):
            m, n = mode.m, mode.n
        else:
            idx = mode.index
            m = (idx // 3) + 1
            n = (idx % 3) + 1
        
        # Draw vertical nodal lines
        # Nodes occur at x/L = (2k-1)/(2m) for k = 1, 2, ..., m
        for k in range(1, m + 1):
            x_node = (2 * k - 1) / (2 * m)
            cx = margin + x_node * (self.canvas_width - 2 * margin)
            
            self.canvas.create_line(
                cx, margin, cx, self.canvas_height - margin,
                fill="#ffffff", width=1, dash=(4, 4)
            )
            
            # Label
            self.canvas.create_text(
                cx, margin - 8,
                text=f"N{k}",
                fill="#aaaaaa",
                font=("Arial", 8)
            )
        
        # Draw horizontal nodal lines
        for k in range(1, n + 1):
            y_node = (2 * k - 1) / (2 * n)
            cy = margin + (1 - y_node) * (self.canvas_height - 2 * margin)
            
            self.canvas.create_line(
                margin, cy, self.canvas_width - margin, cy,
                fill="#ffffff", width=1, dash=(4, 4)
            )
        
        # Show mode number
        self.canvas.create_text(
            self.canvas_width - margin - 30, margin + 15,
            text=f"({m},{n})",
            fill=Style.ACCENT_GOLD,
            font=("Arial", 12, "bold")
        )
    
    def _draw_human_silhouette(self):
        """
        Draw golden-ratio human body silhouette overlay.
        Shows how a person would lie on the plate.
        """
        margin = 20
        
        try:
            plate_L = float(self.length_var.get())
            plate_W = float(self.width_var.get())
        except:
            plate_L, plate_W = 1.95, 0.6
        
        # Human dimensions relative to plate
        h_height = self.human_body.height_m
        h_shoulder = self.human_body.shoulder_width
        
        # Scale human to fit in plate
        scale_x = (self.canvas_width - 2 * margin) / plate_L
        scale_y = (self.canvas_height - 2 * margin) / plate_W
        
        # Center of plate
        cx = self.canvas_width / 2
        cy = self.canvas_height / 2
        
        # Draw simplified human outline (lying down, head to the left)
        # Using golden ratio proportions
        
        # Head
        head_r = h_height / 8 * scale_x / 2
        head_x = margin + 0.1 * (self.canvas_width - 2 * margin)
        self.canvas.create_oval(
            head_x - head_r, cy - head_r,
            head_x + head_r, cy + head_r,
            outline="#ffd700", width=2
        )
        
        # Torso (from shoulder to navel to hips)
        shoulder_x = head_x + head_r + 0.02 * self.canvas_width
        navel_x = margin + (h_height / PHI / plate_L) * (self.canvas_width - 2 * margin)
        hip_x = navel_x + 0.08 * self.canvas_width
        
        torso_w = h_shoulder * scale_y / 2 * 0.8
        
        # Shoulders
        self.canvas.create_line(
            shoulder_x, cy - torso_w, shoulder_x, cy + torso_w,
            fill="#ffd700", width=2
        )
        
        # Torso outline
        self.canvas.create_line(
            shoulder_x, cy - torso_w, hip_x, cy - torso_w * 0.9,
            fill="#ffd700", width=2
        )
        self.canvas.create_line(
            shoulder_x, cy + torso_w, hip_x, cy + torso_w * 0.9,
            fill="#ffd700", width=2
        )
        
        # Navel marker (golden point!)
        self.canvas.create_oval(
            navel_x - 4, cy - 4, navel_x + 4, cy + 4,
            fill="#ffd700", outline="#ff8800"
        )
        self.canvas.create_text(
            navel_x, cy - 15,
            text="φ",
            fill="#ffd700",
            font=("Arial", 10, "bold")
        )
        
        # Legs (simplified)
        foot_x = self.canvas_width - margin - 0.05 * self.canvas_width
        leg_w = torso_w * 0.4
        
        self.canvas.create_line(
            hip_x, cy - torso_w * 0.5, foot_x, cy - leg_w,
            fill="#ffd700", width=2
        )
        self.canvas.create_line(
            hip_x, cy + torso_w * 0.5, foot_x, cy + leg_w,
            fill="#ffd700", width=2
        )
        
        # Arms (outstretched)
        arm_y_top = cy - torso_w - 0.15 * self.canvas_height
        arm_y_bot = cy + torso_w + 0.15 * self.canvas_height
        
        self.canvas.create_line(
            shoulder_x, cy - torso_w, hip_x, arm_y_top,
            fill="#ffd700", width=1, dash=(3, 3)
        )
        self.canvas.create_line(
            shoulder_x, cy + torso_w, hip_x, arm_y_bot,
            fill="#ffd700", width=1, dash=(3, 3)
        )
        
        # Draw H₂O molecules representing body water content (60-70%)
        self._draw_water_molecules_overlay()
    
    def _draw_water_molecules_overlay(self):
        """
        Draw H₂O water molecule symbols on the body overlay.
        Represents that humans are ~60-70% water.
        """
        margin = 20
        
        # Get water geometry
        water = WATER_GEOMETRY
        angle = water.bond_angle_rad
        
        # Positions for water molecules (key body areas)
        positions = [
            (0.35, 0.5),  # Torso center
            (0.5, 0.4),   # Near navel (φ point)
            (0.5, 0.6),   # Lower torso
            (0.2, 0.5),   # Head area
            (0.7, 0.35),  # Upper leg
            (0.7, 0.65),  # Lower leg
        ]
        
        mol_size = 12  # Size of molecule symbol
        
        for px, py in positions:
            cx = margin + px * (self.canvas_width - 2 * margin)
            cy = margin + py * (self.canvas_height - 2 * margin)
            
            # Get atom positions using water geometry
            atoms = water.get_molecule_points(cx, cy, mol_size)
            
            # Draw O atom (larger, red/blue)
            ox, oy = atoms['O']
            self.canvas.create_oval(
                ox - 4, oy - 4, ox + 4, oy + 4,
                fill="#4488ff", outline="#2266dd", width=1
            )
            
            # Draw H atoms (smaller, white)
            for h_key in ['H1', 'H2']:
                hx, hy = atoms[h_key]
                self.canvas.create_oval(
                    hx - 2, hy - 2, hx + 2, hy + 2,
                    fill="white", outline="#aaaaaa", width=1
                )
                
                # Draw bond line
                self.canvas.create_line(
                    ox, oy, hx, hy,
                    fill="#88aacc", width=1
                )
        
        # Label with angle
        self.canvas.create_text(
            self.canvas_width - margin - 50, self.canvas_height - margin - 15,
            text=f"H₂O 104.5°",
            fill="#4488ff",
            font=("Arial", 9)
        )
        
        # Draw Antahkarana axis (Sushumna) with 7 chakras
        self._draw_antahkarana_axis()
    
    def _draw_antahkarana_axis(self):
        """
        Draw the Antahkarana (अन्तःकरण) - the central axis along the spine.
        
        The Sushumna Nadi runs from the root (Muladhara) to the crown (Sahasrara),
        passing through all 7 chakras. This is the "rainbow bridge" of consciousness.
        
        When Ida (lunar/left) and Pingala (solar/right) nadis are balanced,
        prana flows freely through Sushumna - the path to enlightenment.
        """
        # Check if we should display
        if not getattr(self, 'show_antahkarana', True):
            return
        if hasattr(self, 'show_antahkarana_var') and not self.show_antahkarana_var.get():
            return
            
        margin = 20
        
        try:
            plate_L = float(self.length_var.get())
        except:
            plate_L = 1.95
        
        h_height = self.human_body.height_m
        
        # Calculate spine position on canvas
        # When lying down, spine runs horizontally from sacrum to skull
        # Spine is ~80% of body height, positioned at center
        spine_start_frac = 0.05  # Coccyx at 5% from feet
        spine_end_frac = 0.85    # Skull base at 85%
        
        # Convert to canvas coordinates (person lying with head on left)
        spine_start_x = margin + spine_start_frac * (self.canvas_width - 2 * margin)
        spine_end_x = margin + spine_end_frac * (self.canvas_width - 2 * margin)
        spine_y = self.canvas_height / 2  # Center line
        
        # Draw the three nadis
        nadi_offset = 8  # Pixel offset for Ida/Pingala
        
        # Sushumna (central - gold) - the main axis
        self.canvas.create_line(
            spine_start_x, spine_y,
            spine_end_x, spine_y,
            fill="#ffd700", width=3,
            arrow=tk.LAST, arrowshape=(10, 12, 4)
        )
        
        # Ida (left/lunar - silver) - sinuous path
        ida_points = []
        pingala_points = []
        n_waves = 7  # One wave per chakra
        
        for i in range(50):
            t = i / 49
            x = spine_start_x + t * (spine_end_x - spine_start_x)
            wave = np.sin(t * n_waves * np.pi) * nadi_offset
            ida_points.extend([x, spine_y - wave])
            pingala_points.extend([x, spine_y + wave])
        
        # Draw Ida (silver, left channel)
        self.canvas.create_line(
            *ida_points,
            fill="#c0c0c0", width=1, smooth=True, dash=(3, 2)
        )
        
        # Draw Pingala (orange/gold, right channel)
        self.canvas.create_line(
            *pingala_points,
            fill="#ffa500", width=1, smooth=True, dash=(3, 2)
        )
        
        # Draw 7 chakras along Sushumna
        antahkarana = ANTAHKARANA
        chakra_radius = 10
        
        for name, chakra in antahkarana.CHAKRAS.items():
            pos = chakra["position"]
            color = chakra["color"]
            freq = chakra["frequency_hz"]
            
            # Position along spine (0=root at bottom/right, 1=crown at top/left)
            # Since person is lying with head on LEFT, we reverse
            x_frac = spine_start_frac + pos * (spine_end_frac - spine_start_frac)
            cx = margin + (1 - x_frac) * (self.canvas_width - 2 * margin)  # Reversed!
            cy = spine_y
            
            # Draw chakra circle
            self.canvas.create_oval(
                cx - chakra_radius, cy - chakra_radius,
                cx + chakra_radius, cy + chakra_radius,
                fill=color, outline="white", width=1
            )
            
            # Draw inner glow effect
            inner_r = chakra_radius * 0.5
            self.canvas.create_oval(
                cx - inner_r, cy - inner_r,
                cx + inner_r, cy + inner_r,
                fill="white", outline=""
            )
            
            # Label with frequency (alternating above/below to avoid overlap)
            label_y_offset = -18 if list(antahkarana.CHAKRAS.keys()).index(name) % 2 == 0 else 18
            
            # Short name for display
            short_names = {
                "Muladhara": "Root",
                "Svadhisthana": "Sacral",
                "Manipura": "Solar",
                "Anahata": "♡ Heart",
                "Vishuddha": "Throat",
                "Ajna": "Third Eye",
                "Sahasrara": "Crown",
            }
            
            self.canvas.create_text(
                cx, cy + label_y_offset,
                text=f"{short_names.get(name, name)}\n{freq:.0f}Hz",
                fill=color,
                font=("Arial", 7),
                justify="center"
            )
        
        # Mark the golden point (Heart chakra = Anahata at φ)
        golden_chakra = antahkarana.get_golden_chakra()
        gc_data = antahkarana.CHAKRAS[golden_chakra]
        gc_pos = gc_data["position"]
        gc_x_frac = spine_start_frac + gc_pos * (spine_end_frac - spine_start_frac)
        gc_cx = margin + (1 - gc_x_frac) * (self.canvas_width - 2 * margin)
        
        # Draw phi symbol at heart
        self.canvas.create_text(
            gc_cx, spine_y - 30,
            text="φ",
            fill="#00ff00",
            font=("Arial", 12, "bold")
        )
        
        # Title for Antahkarana
        self.canvas.create_text(
            (spine_start_x + spine_end_x) / 2, margin + 5,
            text="॥ Antahkarana ॥",
            fill="#ffd700",
            font=("Arial", 10, "italic")
        )
    
    def _draw_exciters(self):
        """Draw exciters on canvas."""
        margin = 20
        
        for i, exc in enumerate(self.exciters):
            # Convert normalized coords to canvas
            cx = margin + exc.x * (self.canvas_width - 2 * margin)
            cy = margin + (1 - exc.y) * (self.canvas_height - 2 * margin)
            
            r = 15
            
            # Draw exciter
            exc.canvas_id = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=exc.color, outline="white", width=2
            )
            
            # Draw label
            exc.label_id = self.canvas.create_text(
                cx, cy,
                text=str(i + 1),
                fill="white",
                font=("Arial", 10, "bold")
            )
            
            # Draw coupling indicator
            coupling_color = self._coupling_to_color(exc.coupling)
            self.canvas.create_arc(
                cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5,
                start=0, extent=360 * exc.coupling,
                fill="", outline=coupling_color, width=3, style=tk.ARC
            )
    
    def _coupling_to_color(self, coupling: float) -> str:
        """Convert coupling value to color."""
        # Green = high coupling, Red = low coupling
        r = int(255 * (1 - coupling))
        g = int(255 * coupling)
        return f"#{r:02x}{g:02x}44"
    
    def _update_exciter_positions(self):
        """Update exciter positions on canvas."""
        margin = 20
        
        for i, exc in enumerate(self.exciters):
            cx = margin + exc.x * (self.canvas_width - 2 * margin)
            cy = margin + (1 - exc.y) * (self.canvas_height - 2 * margin)
            
            r = 15
            
            if exc.canvas_id:
                self.canvas.coords(exc.canvas_id, cx - r, cy - r, cx + r, cy + r)
            if exc.label_id:
                self.canvas.coords(exc.label_id, cx, cy)
    
    # ──────────────────────────────────────────────────────────────────────────
    # COUPLING CALCULATION
    # ──────────────────────────────────────────────────────────────────────────
    
    def _update_all_couplings(self):
        """Update coupling for all exciters."""
        for i in range(len(self.exciters)):
            self._update_coupling_for_exciter(i)
        self._update_exciter_list()
        self._update_coupling_display()
    
    def _update_coupling_for_exciter(self, idx: int):
        """
        Update coupling coefficient for specific exciter.
        
        Based on Leissa (1969) and vibration theory:
        Coupling = |φ(x,y)| where φ is the mode shape at exciter position.
        
        For free-free rectangular plate mode (m,n):
        φ(x,y) = cos(λ_m * x) * cos(λ_n * y)
        where λ_m = (m + 0.5) * π / L, λ_n = (n + 0.5) * π / W
        """
        if not self.modes or self.selected_mode_idx >= len(self.modes):
            return
        
        exc = self.exciters[idx]
        mode = self.modes[self.selected_mode_idx]
        
        # Get plate dimensions
        try:
            L = float(self.length_var.get())
            W = float(self.width_var.get())
        except:
            L, W = 1.95, 0.6
        
        # Convert normalized coords to real coords
        real_x = exc.x * L
        real_y = exc.y * W
        
        if self.analyzer and hasattr(mode, 'get_displacement_at'):
            try:
                bbox = self.analyzer.get_bounding_box()
                min_x, min_y, max_x, max_y = bbox
                real_x = min_x + exc.x * (max_x - min_x)
                real_y = min_y + exc.y * (max_y - min_y)
                exc.coupling = abs(mode.get_displacement_at(real_x, real_y))
            except:
                exc.coupling = self._analytical_coupling(real_x, real_y, L, W, mode)
        else:
            exc.coupling = self._analytical_coupling(real_x, real_y, L, W, mode)
    
    def _analytical_coupling(self, x: float, y: float, L: float, W: float, mode) -> float:
        """
        Calculate coupling using analytical mode shape (Leissa formula).
        
        For free-free plate: φ(x,y) = cos(λ_m * x) * cos(λ_n * y)
        """
        # Get mode indices
        if hasattr(mode, 'm'):
            m, n = mode.m, mode.n
        else:
            # Estimate from mode index
            m = (mode.index // 3) + 1
            n = (mode.index % 3) + 1
        
        # Wave numbers for free-free boundary conditions
        lambda_m = (m + 0.5) * np.pi / L
        lambda_n = (n + 0.5) * np.pi / W
        
        # Mode shape value at exciter position
        phi = np.cos(lambda_m * x) * np.cos(lambda_n * y)
        
        return abs(phi)
    
    def _update_coupling_display(self):
        """Update coupling display label."""
        if not self.exciters:
            return
        
        lines = []
        total_coupling = 0
        
        for i, exc in enumerate(self.exciters):
            lines.append(f"E{i+1}: {exc.coupling:.2f}")
            total_coupling += exc.coupling * exc.amplitude
        
        avg_coupling = total_coupling / len(self.exciters) if self.exciters else 0
        
        self.coupling_label.config(
            text=f"Coupling: {' | '.join(lines)} | Media: {avg_coupling:.2f}"
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # MODE SELECTION & PLAYBACK
    # ──────────────────────────────────────────────────────────────────────────
    
    def _on_mode_select(self, event):
        """Handle mode selection."""
        sel = self.mode_listbox.curselection()
        if sel:
            self.selected_mode_idx = sel[0]
            self._update_visualization()
            self._update_all_couplings()
    
    def _toggle_play(self):
        """Toggle audio playback."""
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()
    
    def _start_play(self):
        """Start playing the selected mode."""
        if not HAS_AUDIO:
            messagebox.showerror("Errore", "sounddevice non installato")
            return
        
        if not self.modes or self.selected_mode_idx >= len(self.modes):
            return
        
        self.is_playing = True
        self.play_btn.config(text="⏹ Stop", bg=Style.ERROR)
        
        self.play_thread = threading.Thread(target=self._play_audio)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def _stop_play(self):
        """Stop playback."""
        self.is_playing = False
        self.play_btn.config(text="🔊 Riproduci Modo", bg=Style.ACCENT_BLUE)
    
    def _play_audio(self):
        """Play audio in background thread."""
        try:
            mode = self.modes[self.selected_mode_idx]
            freq = mode.frequency
            
            sample_rate = 44100
            duration = 5.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Generate audio with exciter coupling
            audio = np.zeros_like(t)
            
            for exc in self.exciters:
                # Phase in radians
                phase_rad = np.radians(exc.phase)
                
                # Generate sine with coupling amplitude
                amplitude = exc.amplitude * exc.coupling
                audio += amplitude * np.sin(2 * np.pi * freq * t + phase_rad)
            
            # Normalize
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp * 0.7
            
            # Apply envelope
            attack = int(0.05 * sample_rate)
            release = int(0.1 * sample_rate)
            
            envelope = np.ones_like(audio)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            audio *= envelope
            
            # Play
            sd.play(audio.astype(np.float32), sample_rate)
            
            while self.is_playing and sd.get_stream().active:
                time.sleep(0.1)
            
            sd.stop()
            
        except Exception as e:
            print(f"Audio error: {e}")
        
        finally:
            self.is_playing = False
            self.parent.after(0, lambda: self.play_btn.config(
                text="🔊 Riproduci Modo", bg=Style.ACCENT_BLUE
            ))


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Plate Lab - FEM Modal Analysis")
    root.geometry("1200x700")
    root.configure(bg=Style.BG_DARK)
    
    frame = tk.Frame(root, bg=Style.BG_DARK)
    frame.pack(fill=tk.BOTH, expand=True)
    
    app = PlateLabTab(frame)
    
    root.mainloop()
