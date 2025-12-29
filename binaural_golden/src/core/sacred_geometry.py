"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SACRED GEOMETRY - Golden Ratio & Body Proportions         ║
║                                                                              ║
║   Contains:                                                                  ║
║   • Golden ratio (φ) constants and calculations                              ║
║   • Water molecule geometry (H₂O - 104.5°)                                   ║
║   • Antahkarana axis (7 chakras along spine)                                 ║
║   • Human body golden proportions (Vitruvian)                                ║
║   • Shape generators (butterfly, vesica piscis, lemniscate, etc.)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN RATIO CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2          # Golden ratio ≈ 1.618034
PHI_INVERSE = 1 / PHI               # ≈ 0.618034
PHI_SQUARED = PHI ** 2              # ≈ 2.618034
GOLDEN_ANGLE_DEG = 137.5077640      # Golden angle in degrees
GOLDEN_ANGLE_RAD = np.radians(GOLDEN_ANGLE_DEG)

# Sacred ratios
SQRT_2 = np.sqrt(2)                 # √2 ≈ 1.414
SQRT_3 = np.sqrt(3)                 # √3 ≈ 1.732
SQRT_5 = np.sqrt(5)                 # √5 ≈ 2.236


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


# Singleton instance
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
            "is_golden": False,
        },
        "Svadhisthana": {    # Sacral
            "position": 0.15,
            "frequency_hz": 288.0,   # D4 - Water element
            "color": "#ff8800",      # Orange
            "element": "Water",
            "function": "Creativity, sexuality",
            "bija_mantra": "VAM",
            "is_golden": False,
        },
        "Manipura": {        # Solar Plexus
            "position": 0.35,
            "frequency_hz": 320.0,   # E4 - Fire element
            "color": "#ffff00",      # Yellow
            "element": "Fire",
            "function": "Will, power",
            "bija_mantra": "RAM",
            "is_golden": False,
        },
        "Anahata": {         # Heart - THE GOLDEN CENTER
            "position": 1 - PHI_INVERSE,  # ≈ 0.382 - Golden point!
            "frequency_hz": 341.3,   # F4 - Air element
            "color": "#00ff00",      # Green
            "element": "Air",
            "function": "Love, compassion",
            "bija_mantra": "YAM",
            "is_golden": True,       # At the golden ratio point!
        },
        "Vishuddha": {       # Throat
            "position": 0.65,
            "frequency_hz": 384.0,   # G4 - Ether/Space
            "color": "#00bfff",      # Light Blue
            "element": "Ether",
            "function": "Communication, truth",
            "bija_mantra": "HAM",
            "is_golden": False,
        },
        "Ajna": {            # Third Eye
            "position": 0.85,
            "frequency_hz": 426.7,   # A4 approx - Light
            "color": "#4400ff",      # Indigo
            "element": "Light",
            "function": "Intuition, wisdom",
            "bija_mantra": "OM",
            "is_golden": False,
        },
        "Sahasrara": {       # Crown
            "position": 1.0,
            "frequency_hz": 480.0,   # B4 approx - Thought/Spirit
            "color": "#ff00ff",      # Violet/White
            "element": "Thought",
            "function": "Connection to divine, enlightenment",
            "bija_mantra": "Silence",
            "is_golden": False,
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
        Find the chakra at the golden ratio point (Anahata/Heart).
        """
        for name, chakra in self.CHAKRAS.items():
            if chakra.get("is_golden", False):
                return name
        return "Anahata"
    
    def get_frequency_at_position(self, position: float) -> float:
        """
        Interpolate frequency at any position along the Antahkarana.
        
        Args:
            position: 0 = root, 1 = crown
        """
        chakras = list(self.CHAKRAS.values())
        positions = [c["position"] for c in chakras]
        frequencies = [c["frequency_hz"] for c in chakras]
        
        return np.interp(position, positions, frequencies)
    
    def get_resonant_ratios(self) -> Dict[str, float]:
        """
        Get harmonic ratios between chakra frequencies.
        """
        freqs = self.get_chakra_frequencies()
        base = freqs[0]  # Root as fundamental
        
        return {
            name: chakra["frequency_hz"] / base
            for name, chakra in self.CHAKRAS.items()
        }


# Singleton instance
ANTAHKARANA = AntahkaranaAxis()


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN BODY GOLDEN PROPORTIONS (Vitruvian Man)
# ══════════════════════════════════════════════════════════════════════════════

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
    def head_height(self) -> float:
        """Head height (1/8 of total)."""
        return self.height_m / 8
    
    @property
    def torso_length(self) -> float:
        """From shoulder to navel."""
        return self.height_m - self.navel_height - self.head_height
    
    @property
    def lying_length(self) -> float:
        """Length when lying down."""
        return self.height_m
    
    @property
    def lying_width(self) -> float:
        """Width when lying (shoulder width)."""
        return self.shoulder_width
    
    @property
    def hip_width(self) -> float:
        """Hip width (slightly less than shoulders)."""
        return self.shoulder_width * 0.9
    
    @property
    def mass_distribution(self) -> Dict[str, float]:
        """
        Mass distribution percentages for different body parts.
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
    
    def get_body_contour_points(self, n_points: int = 64) -> List[Tuple[float, float]]:
        """
        Generate body contour points for plate shape.
        Returns list of (x, y) tuples normalized to [0, height] × [0, width].
        """
        # Key widths at different heights (normalized)
        profile = [
            (0.00, 0.40),   # Feet
            (0.15, 0.35),   # Ankles
            (0.35, 0.45),   # Knees
            (0.50, 0.55),   # Hips (at navel)
            (0.60, 0.60),   # Waist
            (0.70, 0.70),   # Chest
            (0.80, 0.65),   # Shoulders
            (0.90, 0.40),   # Neck
            (0.95, 0.45),   # Head base
            (1.00, 0.35),   # Head top
        ]
        
        # Interpolate to n_points
        heights = [p[0] for p in profile]
        widths = [p[1] for p in profile]
        
        from scipy.interpolate import interp1d
        width_func = interp1d(heights, widths, kind='cubic', fill_value='extrapolate')
        
        points = []
        n_side = n_points // 2
        
        # One side
        for i in range(n_side):
            h = i / (n_side - 1)
            w = float(width_func(h))
            x = h * self.height_m
            y = w * self.shoulder_width / 2
            points.append((x, y))
        
        # Other side (mirrored)
        for i in range(n_side - 1, -1, -1):
            h = i / (n_side - 1)
            w = float(width_func(h))
            x = h * self.height_m
            y = -w * self.shoulder_width / 2
            points.append((x, y))
        
        # Shift to positive y
        min_y = min(p[1] for p in points)
        points = [(p[0], p[1] - min_y) for p in points]
        
        return points


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

class PlateShape(Enum):
    """Available plate shape templates."""
    RECTANGLE = "rectangle"
    GOLDEN_RECTANGLE = "golden_rectangle"
    ELLIPSE = "ellipse"
    GOLDEN_OVOID = "golden_ovoid"
    VITRUVIAN = "vitruvian"
    WATER_MOLECULE = "water_molecule"
    BUTTERFLY = "butterfly"
    LEMNISCATE = "lemniscate"
    VESICA_PISCIS = "vesica_piscis"
    FREEFORM = "freeform"


def generate_rectangle(length: float, width: float) -> List[Tuple[float, float]]:
    """Generate rectangle vertices."""
    return [
        (0, 0),
        (length, 0),
        (length, width),
        (0, width)
    ]


def generate_golden_rectangle(length: float) -> List[Tuple[float, float]]:
    """Generate golden rectangle vertices (L × L/φ)."""
    width = length / PHI
    return generate_rectangle(length, width)


def generate_ellipse(a: float, b: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """Generate ellipse points."""
    points = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        x = a * np.cos(theta) + a
        y = b * np.sin(theta) + b
        points.append((x, y))
    return points


def generate_golden_ovoid(length: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """
    Generate golden ovoid (egg-shaped with φ proportions).
    
    The golden ovoid has:
    - Major axis a = length/2
    - Minor axis b = a/φ
    - Slight asymmetry with pointed end
    """
    a = length / 2  # Semi-major axis
    b = a / PHI     # Semi-minor axis
    
    points = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        
        # Basic ellipse
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        
        # Add asymmetry for egg shape (pointed toward head)
        k = 0.15  # Asymmetry factor
        x = x * (1 + k * np.cos(theta))
        
        # Shift to positive coordinates
        points.append((x + a, y + b))
    
    return points


def generate_vitruvian_shape(height: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """
    Generate Vitruvian/Leonardo body-fitted shape.
    """
    body = HumanBodyGolden(height)
    return body.get_body_contour_points(n_points)


def generate_water_molecule_shape(length: float, n_points: int = 48) -> List[Tuple[float, float]]:
    """
    Generate shape based on H₂O molecule geometry (104.5°).
    """
    water = WATER_GEOMETRY
    half_angle = water.half_angle_rad
    
    # Calculate dimensions
    o_height = (length / 2) / np.tan(half_angle)
    
    points = []
    
    # H atoms at corners, O at apex
    h1_x, h1_y = 0, 0
    h2_x, h2_y = length, 0
    o_x, o_y = length / 2, o_height
    
    # Generate smooth curved shape
    for i in range(n_points):
        t = i / n_points * 2 * np.pi
        
        if t < np.pi:
            # Upper half - bezier-like curve through O
            progress = t / np.pi
            x = (1-progress)**2 * h1_x + 2*(1-progress)*progress * o_x + progress**2 * h2_x
            y = (1-progress)**2 * h1_y + 2*(1-progress)*progress * o_y + progress**2 * h2_y
            
            # Add bulge
            bulge = np.sin(t) * 0.15 * length
            y += bulge * np.cos(half_angle)
        else:
            # Lower half - slight inward curve
            progress = (t - np.pi) / np.pi
            x = h2_x * (1 - progress) + h1_x * progress
            y = h2_y - np.sin(t - np.pi) * 0.08 * o_height
        
        points.append((x, y))
    
    return points


def generate_butterfly_shape(length: float, n_points: int = 80) -> List[Tuple[float, float]]:
    """
    Generate butterfly/wing shape.
    """
    points = []
    
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        
        # Butterfly curve formula
        r = np.exp(np.sin(t)) - 2 * np.cos(4 * t) + np.sin((2 * t - np.pi) / 24) ** 5
        r = abs(r) * length / 8
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        points.append((x, y))
    
    # Normalize
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    points = [(p[0] - min_x, p[1] - min_y) for p in points]
    
    max_x = max(p[0] for p in points) or 1
    scale = length / max_x
    points = [(p[0] * scale, p[1] * scale) for p in points]
    
    return points


def generate_lemniscate_shape(length: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """
    Generate lemniscate (figure-8 / infinity) shape.
    """
    a = length / 4
    
    points = []
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        
        cos_2t = np.cos(2 * t)
        if cos_2t > 0:
            r = a * np.sqrt(cos_2t)
        else:
            r = a * np.sqrt(abs(cos_2t)) * 0.3
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        points.append((x, y))
    
    # Normalize
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    points = [(p[0] - min_x, p[1] - min_y) for p in points]
    
    max_x = max(p[0] for p in points) or 1
    scale = length / max_x
    points = [(p[0] * scale, p[1] * scale) for p in points]
    
    return points


def generate_vesica_piscis_shape(length: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """
    Generate Vesica Piscis shape (sacred geometry).
    Two intersecting circles where each passes through the other's center.
    """
    r = length / 2
    d = r
    
    points = []
    
    # First arc
    for i in range(n_points // 2):
        theta = -np.pi / 3 + (2 * np.pi / 3) * i / (n_points // 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x + d / 2, y + r))
    
    # Second arc
    for i in range(n_points // 2):
        theta = np.pi / 3 - (2 * np.pi / 3) * i / (n_points // 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x + d / 2 + d, y + r))
    
    # Normalize
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    points = [(p[0] - min_x, p[1] - min_y) for p in points]
    
    max_x = max(p[0] for p in points) or 1
    scale = length / max_x
    points = [(p[0] * scale, p[1] * scale) for p in points]
    
    return points


def generate_shape(shape: PlateShape, length: float, **kwargs) -> List[Tuple[float, float]]:
    """
    Generate shape points for any PlateShape type.
    """
    generators = {
        PlateShape.RECTANGLE: lambda: generate_rectangle(length, length / PHI),
        PlateShape.GOLDEN_RECTANGLE: lambda: generate_golden_rectangle(length),
        PlateShape.ELLIPSE: lambda: generate_ellipse(length / 2, length / (2 * PHI)),
        PlateShape.GOLDEN_OVOID: lambda: generate_golden_ovoid(length),
        PlateShape.VITRUVIAN: lambda: generate_vitruvian_shape(length),
        PlateShape.WATER_MOLECULE: lambda: generate_water_molecule_shape(length),
        PlateShape.BUTTERFLY: lambda: generate_butterfly_shape(length),
        PlateShape.LEMNISCATE: lambda: generate_lemniscate_shape(length),
        PlateShape.VESICA_PISCIS: lambda: generate_vesica_piscis_shape(length),
    }
    
    generator = generators.get(shape)
    if generator:
        return generator()
    else:
        return generate_golden_rectangle(length)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def golden_ratio_check(a: float, b: float, tolerance: float = 0.01) -> bool:
    """Check if a/b is approximately the golden ratio."""
    if b == 0:
        return False
    ratio = a / b
    return abs(ratio - PHI) < tolerance


def get_golden_subdivisions(length: float, n: int = 5) -> List[float]:
    """
    Get golden ratio subdivisions of a length.
    Each subdivision is 1/φ of the previous.
    """
    subdivisions = [length]
    for _ in range(n - 1):
        subdivisions.append(subdivisions[-1] / PHI)
    return subdivisions


def fibonacci_sequence(n: int) -> List[int]:
    """Generate first n Fibonacci numbers."""
    fib = [1, 1]
    for _ in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]


def fibonacci_ratios(n: int = 10) -> List[float]:
    """
    Get ratios of consecutive Fibonacci numbers.
    Converges to φ as n increases.
    """
    fib = fibonacci_sequence(n + 1)
    return [fib[i+1] / fib[i] for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SACRED GEOMETRY MODULE")
    print("=" * 60)
    
    print(f"\n📐 Golden Ratio (φ) = {PHI:.6f}")
    print(f"   1/φ = {PHI_INVERSE:.6f}")
    print(f"   φ² = {PHI_SQUARED:.6f}")
    print(f"   Golden Angle = {GOLDEN_ANGLE_DEG:.4f}°")
    
    print(f"\n💧 Water Molecule (H₂O):")
    print(f"   Bond angle: {WATER_GEOMETRY.BOND_ANGLE_DEG}°")
    print(f"   H-H ratio: {WATER_GEOMETRY.h_h_distance_ratio:.4f}")
    
    print(f"\n🌈 Antahkarana - 7 Chakras:")
    for name, chakra in ANTAHKARANA.CHAKRAS.items():
        golden = " ⭐φ" if chakra.get("is_golden") else ""
        print(f"   {name:15s}: {chakra['frequency_hz']:6.1f} Hz | {chakra['bija_mantra']:7s}{golden}")
    
    print(f"\n🧍 Human Body (1.75m):")
    body = HumanBodyGolden(1.75)
    print(f"   Navel at: {body.navel_height:.3f}m (H/φ)")
    print(f"   Shoulders: {body.shoulder_width:.3f}m")
