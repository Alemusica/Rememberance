"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PLATE OPTIMIZER - Physics-Based Shape Optimization                 ║
║                                                                              ║
║   Automatically generates optimal vibroacoustic plate shapes based on:       ║
║   • Golden ratio (φ) proportions                                             ║
║   • Human body dimensions (1.50m - 2.10m range)                              ║
║   • Zone-based multimodal analysis (configurable, not hardcoded!)            ║
║   • Material properties and thickness                                        ║
║   • Internal cuts/holes for resonance tuning                                 ║
║                                                                              ║
║   Optimization targets:                                                      ║
║   • Musical frequency transmission (20Hz - 500Hz)                            ║
║   • Body resonance coupling at anatomical zones                              ║
║   • Uniform vibration distribution                                           ║
║                                                                              ║
║   v2.0 - Zone-based architecture:                                            ║
║   • body_zones.py - Configurable anatomical zones                            ║
║   • coupled_system.py - Plate+body transfer function                         ║
║   • iterative_optimizer.py - SIMP/RAMP topology optimization                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from enum import Enum
import warnings

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618034

# Try imports
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Zone-based modules (v2.0)
try:
    from .body_zones import (
        BodyZone, ZoneType, ZoneAnalyzer, BodyResonance, BODY_RESONANCES,
        create_chakra_zones, create_vat_therapy_zones, create_body_resonance_zones
    )
    from .coupled_system import (
        CoupledSystem, PlatePhysics, HumanBody, ContactInterface,
        ZoneCoupledSystem
    )
    from .iterative_optimizer import (
        ZoneIterativeOptimizer, OptimizationConfig as IterativeConfig,
        InterpolationScheme, MSEObjective, OC_Optimizer
    )
    HAS_ZONE_MODULES = True
except ImportError as e:
    HAS_ZONE_MODULES = False
    warnings.warn(f"Zone modules not available: {e}")

try:
    from .plate_fem import (
        PlateAnalyzer, Material, MATERIALS, FEMMode,
        create_rectangle_mesh, create_ellipse_mesh, create_polygon_mesh,
        fem_modal_analysis, calculate_flexural_rigidity
    )
    HAS_FEM = True
except ImportError:
    HAS_FEM = False


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

class PlateTemplate(Enum):
    """Available plate shape templates."""
    RECTANGLE = "rectangle"
    GOLDEN_RECTANGLE = "golden_rectangle"  # L/W = φ
    ELLIPSE = "ellipse"
    GOLDEN_OVOID = "golden_ovoid"  # a/b = φ
    VITRUVIAN = "vitruvian"  # Human body proportions
    WATER_MOLECULE = "water_molecule"  # H₂O 104.5°
    BUTTERFLY = "butterfly"  # Symmetric winged shape
    LEMNISCATE = "lemniscate"  # Figure-8 / infinity
    VESICA_PISCIS = "vesica_piscis"  # Sacred geometry


@dataclass
class CutoutSpec:
    """Specification for internal cutout/hole."""
    shape: str  # 'circle', 'ellipse', 'rectangle', 'polygon'
    center: Tuple[float, float]  # (x, y) relative to plate center [0,1]
    size: Tuple[float, float]  # (w, h) or (radius,) depending on shape
    rotation: float = 0.0  # degrees
    purpose: str = ""  # e.g., "chakra_heart", "weight_reduction"


@dataclass
class ChakraTarget:
    """Target frequency and position for chakra resonance."""
    name: str
    position_spine: float  # 0 = root, 1 = crown
    frequency_hz: float
    color: str
    weight: float = 1.0  # Optimization weight


# The 7 chakras with their target frequencies
CHAKRA_TARGETS = [
    ChakraTarget("Muladhara", 0.00, 256.0, "#ff0000", 1.0),    # Root - C4
    ChakraTarget("Svadhisthana", 0.15, 288.0, "#ff8800", 1.0), # Sacral - D4
    ChakraTarget("Manipura", 0.35, 320.0, "#ffff00", 1.0),     # Solar - E4
    ChakraTarget("Anahata", 0.382, 341.3, "#00ff00", 1.5),     # Heart - F4 (φ point!)
    ChakraTarget("Vishuddha", 0.65, 384.0, "#00bfff", 1.0),    # Throat - G4
    ChakraTarget("Ajna", 0.85, 426.7, "#4400ff", 1.0),         # Third Eye - A4
    ChakraTarget("Sahasrara", 1.00, 480.0, "#ff00ff", 0.8),    # Crown - B4
]


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HumanBodyParams:
    """Human body parameters for plate sizing."""
    height_min: float = 1.50  # meters
    height_max: float = 2.10  # meters
    height_avg: float = 1.75  # meters
    weight_min: float = 45.0  # kg
    weight_max: float = 120.0  # kg
    weight_avg: float = 70.0  # kg
    
    # Body proportions (golden ratio based)
    shoulder_ratio: float = 0.25  # shoulder_width = height * ratio
    navel_ratio: float = 1 / PHI  # navel_height = height / φ
    
    def get_plate_length_range(self) -> Tuple[float, float]:
        """Get plate length range to accommodate all heights."""
        return (self.height_min, self.height_max + 0.10)
    
    def get_plate_width_range(self) -> Tuple[float, float]:
        """Get plate width range based on shoulder width."""
        w_min = self.height_min * self.shoulder_ratio
        w_max = self.height_max * self.shoulder_ratio + 0.15
        return (w_min, w_max)


@dataclass
class OptimizationConfig:
    """Configuration for plate shape optimization."""
    
    # Target frequencies (musical range)
    freq_min: float = 20.0   # Hz - low bass
    freq_max: float = 500.0  # Hz - upper musical range
    
    # How many modes to consider
    n_modes: int = 15
    
    # Optimization weights
    weight_chakra_match: float = 2.0     # Match chakra frequencies
    weight_freq_coverage: float = 1.0    # Cover musical frequency range
    weight_uniform_response: float = 0.5 # Uniform response across body
    weight_golden_ratio: float = 1.0     # Maintain φ proportions
    weight_mass: float = 0.3             # Lighter is better
    
    # Constraints
    thickness_range: Tuple[float, float] = (0.008, 0.025)  # 8mm - 25mm
    max_cutouts: int = 5
    min_material_fraction: float = 0.70  # At least 70% solid
    
    # Resolution
    mesh_resolution: int = 20
    optimization_iterations: int = 100


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_golden_rectangle(length: float) -> List[Tuple[float, float]]:
    """Generate golden rectangle vertices (L × L/φ)."""
    width = length / PHI
    return [
        (0, 0),
        (length, 0),
        (length, width),
        (0, width)
    ]


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
    
    Based on golden proportions of human body:
    - Total length = height
    - Width varies along body contour
    - Navel at height/φ
    """
    length = height
    
    # Key widths at different positions
    head_width = height / 8
    shoulder_width = height / 4
    waist_width = height / 5
    hip_width = height / 4 * 0.9
    leg_width = height / 8
    
    # Key positions (fraction of length)
    pos_head = 0.0
    pos_shoulder = height / 8 / height  # After head
    pos_waist = 1 / PHI  # Golden point (navel)
    pos_hip = 1 / PHI + 0.08
    pos_knee = 0.75
    pos_foot = 1.0
    
    # Generate smooth contour using spline
    x_control = np.array([pos_head, pos_shoulder, pos_waist, pos_hip, pos_knee, pos_foot])
    w_control = np.array([head_width, shoulder_width, waist_width, hip_width, leg_width, leg_width * 0.8])
    
    # Interpolate
    from scipy.interpolate import interp1d
    width_func = interp1d(x_control, w_control, kind='cubic', fill_value='extrapolate')
    
    # Generate outline
    points = []
    n_side = n_points // 2
    
    # Top contour (one side)
    for i in range(n_side):
        x = length * i / (n_side - 1)
        w = width_func(i / (n_side - 1))
        points.append((x, w / 2))
    
    # Bottom contour (other side, reversed)
    for i in range(n_side - 1, -1, -1):
        x = length * i / (n_side - 1)
        w = width_func(i / (n_side - 1))
        points.append((x, -w / 2))
    
    # Shift to positive y
    min_y = min(p[1] for p in points)
    points = [(p[0], p[1] - min_y) for p in points]
    
    return points


def generate_water_molecule_shape(length: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """
    Generate shape based on H₂O molecule geometry (104.5°).
    
    Two lobes (hydrogen positions) connected by center (oxygen).
    """
    angle_rad = np.radians(104.5)
    half_angle = angle_rad / 2
    
    # Lobe radius
    r_lobe = length / 4
    # Center connection width
    center_width = length / 6
    
    # Oxygen position (center)
    ox, oy = length / 2, length / (2 * PHI)
    
    # Hydrogen positions (lobes)
    h1_x = ox - length / 3 * np.sin(half_angle)
    h1_y = oy - length / 3 * np.cos(half_angle)
    h2_x = ox + length / 3 * np.sin(half_angle)
    h2_y = oy - length / 3 * np.cos(half_angle)
    
    points = []
    
    # Generate outline combining lobes
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        
        # Blend between lobes and center
        # This creates a smooth, bilobed shape
        r = r_lobe * (1 + 0.3 * np.cos(2 * theta) + 0.1 * np.cos(4 * theta))
        
        x = ox + r * np.cos(theta)
        y = oy + r * np.sin(theta) * 0.6  # Compress vertically
        
        points.append((x, y))
    
    # Normalize to fit in bounding box
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    points = [(p[0] - min_x, p[1] - min_y) for p in points]
    
    # Scale to length
    max_x = max(p[0] for p in points)
    scale = length / max_x if max_x > 0 else 1
    points = [(p[0] * scale, p[1] * scale) for p in points]
    
    return points


def generate_butterfly_shape(length: float, n_points: int = 80) -> List[Tuple[float, float]]:
    """
    Generate butterfly/wing shape.
    
    Symmetric wings with body in center.
    Similar to water molecule but more elongated wings.
    """
    points = []
    
    # Butterfly parametric curve
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        
        # Butterfly curve (modified)
        r = np.exp(np.sin(t)) - 2 * np.cos(4 * t) + np.sin((2 * t - np.pi) / 24) ** 5
        r = abs(r) * length / 8
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        points.append((x, y))
    
    # Normalize
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    points = [(p[0] - min_x, p[1] - min_y) for p in points]
    
    max_x = max(p[0] for p in points)
    scale = length / max_x if max_x > 0 else 1
    points = [(p[0] * scale, p[1] * scale) for p in points]
    
    return points


def generate_lemniscate_shape(length: float, n_points: int = 64) -> List[Tuple[float, float]]:
    """
    Generate lemniscate (figure-8 / infinity) shape.
    
    The lemniscate has special resonant properties.
    """
    a = length / 4  # Scale factor
    
    points = []
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        
        # Lemniscate of Bernoulli: r² = a² * cos(2θ)
        cos_2t = np.cos(2 * t)
        if cos_2t > 0:
            r = a * np.sqrt(cos_2t)
        else:
            r = a * np.sqrt(abs(cos_2t)) * 0.3  # Smoothed crossing
        
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
    The width/height ratio is √3 ≈ 1.732.
    """
    # Circle radius
    r = length / 2
    
    # Distance between centers
    d = r
    
    points = []
    
    # The vesica is bounded by two arcs
    for i in range(n_points // 2):
        # First arc (left circle)
        theta = -np.pi / 3 + (2 * np.pi / 3) * i / (n_points // 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x + d / 2, y + r))
    
    for i in range(n_points // 2):
        # Second arc (right circle)
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


# ══════════════════════════════════════════════════════════════════════════════
# CUTOUT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_chakra_cutouts(
    plate_length: float,
    plate_width: float,
    chakra_positions: List[float],
    cutout_radius_fraction: float = 0.03
) -> List[CutoutSpec]:
    """
    Generate circular cutouts at chakra positions along the spine.
    
    These cutouts can help tune resonances at specific frequencies.
    """
    cutouts = []
    
    for i, pos in enumerate(chakra_positions):
        # Position along spine (centered on plate width)
        x_rel = pos  # Position along length
        y_rel = 0.5  # Center of width
        
        # Size varies slightly by chakra importance
        radius = cutout_radius_fraction * plate_length
        if i == 3:  # Heart chakra (Anahata) - larger
            radius *= 1.5
        
        cutouts.append(CutoutSpec(
            shape='circle',
            center=(x_rel, y_rel),
            size=(radius,),
            purpose=f"chakra_{CHAKRA_TARGETS[i].name}"
        ))
    
    return cutouts


def generate_weight_reduction_cutouts(
    plate_length: float,
    plate_width: float,
    target_reduction: float = 0.15,  # Remove 15% of material
    n_cutouts: int = 4
) -> List[CutoutSpec]:
    """
    Generate cutouts for weight reduction while maintaining structural integrity.
    
    Places elliptical cutouts in low-stress areas.
    """
    cutouts = []
    
    # Low stress areas are typically away from center and edges
    positions = [
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    ]
    
    # Size each cutout to achieve target reduction
    area_total = plate_length * plate_width
    area_per_cutout = (target_reduction * area_total) / n_cutouts
    
    # Elliptical cutout with a/b = φ
    a = np.sqrt(area_per_cutout * PHI / np.pi)
    b = a / PHI
    
    for i, (x_rel, y_rel) in enumerate(positions[:n_cutouts]):
        cutouts.append(CutoutSpec(
            shape='ellipse',
            center=(x_rel, y_rel),
            size=(a, b),
            rotation=45 if i % 2 == 0 else -45,
            purpose="weight_reduction"
        ))
    
    return cutouts


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_chakra_frequency_error(
    modes: List['FEMMode'],
    chakra_targets: List[ChakraTarget]
) -> float:
    """
    Calculate error between plate modal frequencies and chakra target frequencies.
    
    Lower is better - means modes are well-aligned with chakras.
    """
    if not modes:
        return float('inf')
    
    mode_freqs = np.array([m.frequency for m in modes])
    
    total_error = 0.0
    for chakra in chakra_targets:
        # Find closest mode frequency
        min_diff = np.min(np.abs(mode_freqs - chakra.frequency_hz))
        # Normalized error (relative to target)
        rel_error = min_diff / chakra.frequency_hz
        # Weight by chakra importance
        total_error += rel_error * chakra.weight
    
    return total_error / len(chakra_targets)


def calculate_frequency_coverage(
    modes: List['FEMMode'],
    freq_min: float,
    freq_max: float,
    n_bins: int = 10
) -> float:
    """
    Calculate how well modes cover the target frequency range.
    
    Returns coverage fraction [0, 1] where 1 = perfect coverage.
    """
    if not modes:
        return 0.0
    
    mode_freqs = [m.frequency for m in modes if freq_min <= m.frequency <= freq_max]
    
    if not mode_freqs:
        return 0.0
    
    # Divide range into bins
    bins = np.linspace(freq_min, freq_max, n_bins + 1)
    covered = 0
    
    for i in range(n_bins):
        bin_min, bin_max = bins[i], bins[i + 1]
        if any(bin_min <= f <= bin_max for f in mode_freqs):
            covered += 1
    
    return covered / n_bins


def calculate_coupling_uniformity(
    modes: List['FEMMode'],
    test_positions: List[Tuple[float, float]]
) -> float:
    """
    Calculate uniformity of vibration response across test positions.
    
    Returns uniformity score [0, 1] where 1 = perfectly uniform.
    """
    if not modes:
        return 0.0
    
    # Calculate total response at each position
    responses = []
    for x, y in test_positions:
        total_response = 0
        for mode in modes:
            try:
                coupling = abs(mode.get_displacement_at(x, y))
                total_response += coupling
            except:
                total_response += 0.5
        responses.append(total_response)
    
    if not responses or max(responses) == 0:
        return 0.0
    
    # Uniformity = 1 - coefficient of variation
    mean_resp = np.mean(responses)
    std_resp = np.std(responses)
    
    if mean_resp == 0:
        return 0.0
    
    cv = std_resp / mean_resp
    uniformity = max(0, 1 - cv)
    
    return uniformity


def calculate_golden_ratio_score(length: float, width: float) -> float:
    """
    Score how close the L/W ratio is to golden ratio.
    
    Returns score [0, 1] where 1 = exactly φ.
    """
    ratio = length / width if width > 0 else 0
    error = abs(ratio - PHI) / PHI
    return max(0, 1 - error)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """Result of plate shape optimization."""
    template: PlateTemplate
    vertices: List[Tuple[float, float]]
    length: float
    width: float
    thickness: float
    material: str
    cutouts: List[CutoutSpec]
    
    # Performance metrics
    chakra_error: float
    frequency_coverage: float
    coupling_uniformity: float
    golden_ratio_score: float
    total_score: float
    
    # Modal analysis results
    modes: List['FEMMode']
    mode_frequencies: List[float]
    
    def __str__(self) -> str:
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║              OPTIMIZATION RESULT                              ║
╠═══════════════════════════════════════════════════════════════╣
║  Template: {self.template.value:<48} ║
║  Dimensions: {self.length:.3f}m × {self.width:.3f}m × {self.thickness*1000:.1f}mm  ║
║  Material: {self.material:<50} ║
║  Cutouts: {len(self.cutouts):<51} ║
╠═══════════════════════════════════════════════════════════════╣
║  SCORES:                                                      ║
║    Chakra Match:     {self.chakra_error:.3f} (lower is better)             ║
║    Freq Coverage:    {self.frequency_coverage:.1%}                             ║
║    Uniformity:       {self.coupling_uniformity:.1%}                             ║
║    Golden Ratio:     {self.golden_ratio_score:.1%}                             ║
║    ─────────────────────────────────────────────              ║
║    TOTAL SCORE:      {self.total_score:.2f}                                 ║
╠═══════════════════════════════════════════════════════════════╣
║  First 7 modes: {', '.join(f'{f:.0f}Hz' for f in self.mode_frequencies[:7]):<43} ║
╚═══════════════════════════════════════════════════════════════╝
"""


class PlateOptimizer:
    """
    Physics-based plate shape optimizer.
    
    Automatically generates optimal vibroacoustic plate designs
    based on material, thickness, and target human dimensions.
    """
    
    def __init__(
        self,
        material: str = "birch_plywood",
        thickness_m: float = 0.015,
        human_params: Optional[HumanBodyParams] = None,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            material: Material key from MATERIALS dict
            thickness_m: Plate thickness in meters
            human_params: Human body parameters (uses defaults if None)
            config: Optimization configuration (uses defaults if None)
        """
        self.material = material
        self.thickness = thickness_m
        self.human_params = human_params or HumanBodyParams()
        self.config = config or OptimizationConfig()
        
        # Get material properties
        self.mat_props = MATERIALS.get(material, MATERIALS["birch_plywood"])
    
    def optimize(
        self,
        template: Optional[PlateTemplate] = None,
        with_cutouts: bool = False,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run optimization to find best plate shape.
        
        Args:
            template: Specific template to optimize (None = try all)
            with_cutouts: Include internal cutouts in optimization
            verbose: Print progress
        
        Returns:
            OptimizationResult with best configuration
        """
        if template is not None:
            templates = [template]
        else:
            templates = list(PlateTemplate)
        
        best_result = None
        best_score = float('-inf')
        
        for tmpl in templates:
            if verbose:
                print(f"  Evaluating {tmpl.value}...")
            
            result = self._optimize_template(tmpl, with_cutouts)
            
            if result.total_score > best_score:
                best_score = result.total_score
                best_result = result
        
        if verbose and best_result:
            print(best_result)
        
        return best_result
    
    def _optimize_template(
        self,
        template: PlateTemplate,
        with_cutouts: bool
    ) -> OptimizationResult:
        """Optimize a single template."""
        
        # Get default dimensions
        length_range = self.human_params.get_plate_length_range()
        width_range = self.human_params.get_plate_width_range()
        
        # Start with average dimensions
        length = self.human_params.height_avg + 0.05
        
        # Generate shape
        if template == PlateTemplate.RECTANGLE:
            width = length / 3
            vertices = generate_golden_rectangle(length)
            
        elif template == PlateTemplate.GOLDEN_RECTANGLE:
            width = length / PHI
            vertices = generate_golden_rectangle(length)
            
        elif template == PlateTemplate.ELLIPSE:
            width = length / 3
            vertices = [(0, 0), (length, width)]  # Bounding box
            
        elif template == PlateTemplate.GOLDEN_OVOID:
            width = length / PHI / 2
            vertices = generate_golden_ovoid(length)
            
        elif template == PlateTemplate.VITRUVIAN:
            width = length / 4
            vertices = generate_vitruvian_shape(length)
            
        elif template == PlateTemplate.WATER_MOLECULE:
            width = length / 2
            vertices = generate_water_molecule_shape(length)
            
        elif template == PlateTemplate.BUTTERFLY:
            width = length / 2
            vertices = generate_butterfly_shape(length)
            
        elif template == PlateTemplate.LEMNISCATE:
            width = length / 3
            vertices = generate_lemniscate_shape(length)
            
        elif template == PlateTemplate.VESICA_PISCIS:
            width = length / np.sqrt(3)
            vertices = generate_vesica_piscis_shape(length)
        
        else:
            width = length / PHI
            vertices = generate_golden_rectangle(length)
        
        # Generate cutouts if requested
        cutouts = []
        if with_cutouts:
            chakra_positions = [c.position_spine for c in CHAKRA_TARGETS]
            cutouts = generate_chakra_cutouts(length, width, chakra_positions)
        
        # Run FEM analysis
        modes = self._analyze_shape(vertices, template)
        
        # Calculate scores
        chakra_error = calculate_chakra_frequency_error(modes, CHAKRA_TARGETS)
        freq_coverage = calculate_frequency_coverage(
            modes, self.config.freq_min, self.config.freq_max
        )
        
        # Test positions along spine
        test_positions = [(pos * length, width / 2) for pos in np.linspace(0.1, 0.9, 9)]
        uniformity = calculate_coupling_uniformity(modes, test_positions)
        
        golden_score = calculate_golden_ratio_score(length, width)
        
        # Total score (weighted sum)
        total_score = (
            (1 - chakra_error) * self.config.weight_chakra_match +
            freq_coverage * self.config.weight_freq_coverage +
            uniformity * self.config.weight_uniform_response +
            golden_score * self.config.weight_golden_ratio
        )
        
        return OptimizationResult(
            template=template,
            vertices=vertices,
            length=length,
            width=width,
            thickness=self.thickness,
            material=self.material,
            cutouts=cutouts,
            chakra_error=chakra_error,
            frequency_coverage=freq_coverage,
            coupling_uniformity=uniformity,
            golden_ratio_score=golden_score,
            total_score=total_score,
            modes=modes,
            mode_frequencies=[m.frequency for m in modes]
        )
    
    def _analyze_shape(
        self,
        vertices: List[Tuple[float, float]],
        template: PlateTemplate
    ) -> List['FEMMode']:
        """Run FEM modal analysis on shape."""
        
        if not HAS_FEM:
            return []
        
        try:
            # Get bounding box
            v_array = np.array(vertices)
            min_pt = v_array.min(axis=0)
            max_pt = v_array.max(axis=0)
            L = max_pt[0] - min_pt[0]
            W = max_pt[1] - min_pt[1]
            
            # Create mesh based on shape
            if template in [PlateTemplate.RECTANGLE, PlateTemplate.GOLDEN_RECTANGLE]:
                points, triangles = create_rectangle_mesh(
                    L, W, self.config.mesh_resolution
                )
            elif template in [PlateTemplate.ELLIPSE, PlateTemplate.GOLDEN_OVOID]:
                points, triangles = create_ellipse_mesh(
                    L / 2, W / 2, self.config.mesh_resolution
                )
            else:
                points, triangles = create_polygon_mesh(
                    vertices, self.config.mesh_resolution
                )
            
            # Run FEM
            modes = fem_modal_analysis(
                points, triangles,
                self.thickness,
                self.mat_props,
                self.config.n_modes
            )
            
            return modes
            
        except Exception as e:
            warnings.warn(f"FEM analysis failed: {e}")
            return []
    
    def suggest_thickness(
        self,
        target_fundamental: float = 40.0,
        length: float = 1.80
    ) -> float:
        """
        Suggest optimal thickness for target fundamental frequency.
        
        Args:
            target_fundamental: Target first mode frequency in Hz
            length: Plate length in meters
        
        Returns:
            Suggested thickness in meters
        """
        # Analytical approximation for rectangular plate
        # f₁ = (π/2) * √(D/(ρh)) * (1/L² + 1/W²)
        # where D = Eh³/(12(1-ν²))
        
        W = length / PHI
        E = self.mat_props.E_mean
        rho = self.mat_props.density
        nu = self.mat_props.poisson_ratio
        
        # Solve for h
        # f = C * h / L²  where C depends on material
        # h = f * L² / C
        
        C = np.sqrt(E / (12 * rho * (1 - nu**2))) * np.pi / 2
        
        geom_factor = 1 / length**2 + 1 / W**2
        h = target_fundamental / (C * np.sqrt(geom_factor))
        
        # Clamp to reasonable range
        h = np.clip(h, 0.005, 0.030)
        
        return h


# ══════════════════════════════════════════════════════════════════════════════
# QUICK API
# ══════════════════════════════════════════════════════════════════════════════

def auto_design_plate(
    height_m: float = 1.75,
    material: str = "birch_plywood",
    thickness_mm: float = 15.0,
    template: Optional[str] = None,
    with_cutouts: bool = False
) -> OptimizationResult:
    """
    Quickly design an optimized vibroacoustic plate.
    
    Args:
        height_m: Target user height (1.50 - 2.10m)
        material: Material name
        thickness_mm: Thickness in millimeters
        template: Template name (None for auto-select)
        with_cutouts: Include chakra cutouts
    
    Returns:
        OptimizationResult with optimal design
    
    Example:
        >>> result = auto_design_plate(1.80, "birch_plywood", 12)
        >>> print(f"Best shape: {result.template.value}")
        >>> print(f"Dimensions: {result.length:.2f}m × {result.width:.2f}m")
    """
    human_params = HumanBodyParams(
        height_avg=height_m,
        height_min=height_m - 0.10,
        height_max=height_m + 0.10
    )
    
    tmpl = None
    if template:
        try:
            tmpl = PlateTemplate(template)
        except ValueError:
            pass
    
    optimizer = PlateOptimizer(
        material=material,
        thickness_m=thickness_mm / 1000,
        human_params=human_params
    )
    
    return optimizer.optimize(template=tmpl, with_cutouts=with_cutouts)


# ══════════════════════════════════════════════════════════════════════════════
# ZONE-BASED API (v2.0)
# ══════════════════════════════════════════════════════════════════════════════

def zone_optimize_plate(
    zones: Optional[List['BodyZone']] = None,
    zone_preset: str = "chakra",
    height_m: float = 1.75,
    material: str = "birch_plywood",
    thickness_mm: float = 15.0,
    volume_fraction: float = 0.5,
    max_iterations: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Zone-based plate optimization using SIMP/RAMP topology optimization.
    
    NEW in v2.0: Uses configurable zones instead of hardcoded chakra frequencies!
    
    Args:
        zones: Custom BodyZone list (None to use preset)
        zone_preset: 'chakra', 'vat', 'body_resonance' (used if zones is None)
        height_m: Target user height [m]
        material: Material name
        thickness_mm: Plate thickness [mm]
        volume_fraction: Target material fraction (0-1)
        max_iterations: Maximum optimization iterations
        verbose: Print progress
    
    Returns:
        Dict with optimization results including density field and scores
    
    Example:
        >>> # Using preset
        >>> result = zone_optimize_plate(zone_preset='vat')
        >>> print(f"Coupling score: {result['coupling_score']:.2%}")
        >>> 
        >>> # Using custom zones
        >>> from body_zones import create_custom_zones
        >>> my_zones = create_custom_zones(3, [[40, 60], [80], [120]])
        >>> result = zone_optimize_plate(zones=my_zones)
    """
    if not HAS_ZONE_MODULES:
        raise ImportError(
            "Zone modules not available. "
            "Install body_zones, coupled_system, iterative_optimizer modules."
        )
    
    # Get zones
    if zones is None:
        if zone_preset == "chakra":
            zones = create_chakra_zones()
        elif zone_preset == "vat":
            zones = create_vat_therapy_zones()
        elif zone_preset == "body_resonance":
            zones = create_body_resonance_zones()
        else:
            zones = create_chakra_zones()
    
    if verbose:
        print("═" * 60)
        print(" ZONE-BASED PLATE OPTIMIZATION (v2.0)")
        print("═" * 60)
        print(f"\nZone preset: {zone_preset}")
        print(f"Zones configured:")
        for z in zones:
            print(f"  • {z.name}: {z.target_frequencies} Hz")
    
    # Create plate physics
    plate = PlatePhysics(
        length=height_m + 0.05,
        width=(height_m + 0.05) / PHI,
        thickness=thickness_mm / 1000
    )
    
    # Create coupled system with zones
    coupled = ZoneCoupledSystem(plate, zones)
    
    # Calculate coupling before optimization
    initial_coupling = coupled.total_coupling_score()
    
    if verbose:
        print(f"\nInitial coupling score: {initial_coupling:.2%}")
    
    # Create iterative optimizer
    config = IterativeConfig(
        nx=40,
        ny=24,
        interpolation=InterpolationScheme.SIMP,
        penalty=3.0,
        volume_fraction=volume_fraction,
        max_iterations=max_iterations,
        convergence_tol=1e-3,
        use_continuation=True
    )
    
    optimizer = ZoneIterativeOptimizer(
        config=config,
        zones=zones,
        objective=MSEObjective(),
        optimizer=OC_Optimizer()
    )
    
    # Initialize with zone-weighted density
    optimizer.zone_weighted_density()
    
    if verbose:
        print(f"\nRunning optimization...")
    
    # Simple FEM solver for testing
    from .iterative_optimizer import simple_plate_fem
    
    def fem_wrapper(density):
        return simple_plate_fem(
            density,
            length=plate.length,
            width=plate.width,
            thickness=plate.thickness,
            E_base=plate.E_longitudinal,
            rho_base=plate.density,
            n_modes=10
        )
    
    result = optimizer.optimize(fem_solver=fem_wrapper)
    
    # Calculate final coupling
    final_coupling = coupled.total_coupling_score()
    
    if verbose:
        print(f"\n{'═' * 60}")
        print(" RESULTS")
        print(f"{'═' * 60}")
        print(f"Converged: {result.converged}")
        print(f"Iterations: {result.iterations}")
        print(f"Initial coupling: {initial_coupling:.2%}")
        print(f"Final coupling: {final_coupling:.2%}")
        print(f"Frequencies: {result.frequencies[:5]} Hz")
    
    return {
        "density": result.density,
        "frequencies": result.frequencies,
        "zones": zones,
        "coupling_score": final_coupling,
        "initial_coupling": initial_coupling,
        "converged": result.converged,
        "iterations": result.iterations,
        "objective_history": result.objective_history,
        "plate_physics": plate,
        "coupled_system": coupled
    }


def analyze_plate_at_zones(
    mode_shapes: np.ndarray,
    mode_frequencies: np.ndarray,
    zones: Optional[List['BodyZone']] = None,
    zone_preset: str = "chakra",
    plate_length: float = 2.0,
    plate_width: float = 0.6
) -> Dict:
    """
    Analyze existing plate modes at configured body zones.
    
    Use this to evaluate how well an existing plate design
    couples to different anatomical zones.
    
    Args:
        mode_shapes: FEM mode shapes (n_modes, nx, ny)
        mode_frequencies: Modal frequencies [Hz]
        zones: BodyZone list (None to use preset)
        zone_preset: 'chakra', 'vat', 'body_resonance'
        plate_length, plate_width: Plate dimensions [m]
    
    Returns:
        Dict with zone analysis results
    
    Example:
        >>> # After FEM analysis
        >>> results = analyze_plate_at_zones(
        ...     mode_shapes=fem_modes,
        ...     mode_frequencies=fem_freqs,
        ...     zone_preset='vat'
        ... )
        >>> for zone_result in results['zone_results']:
        ...     print(f"{zone_result.zone.name}: {zone_result.coupling_efficiency:.1%}")
    """
    if not HAS_ZONE_MODULES:
        raise ImportError("Zone modules not available.")
    
    # Get zones
    if zones is None:
        if zone_preset == "chakra":
            zones = create_chakra_zones()
        elif zone_preset == "vat":
            zones = create_vat_therapy_zones()
        else:
            zones = create_body_resonance_zones()
    
    # Create analyzer
    analyzer = ZoneAnalyzer(zones)
    
    # Analyze all zones
    zone_results = analyzer.analyze_all_zones(
        mode_shapes=mode_shapes,
        mode_frequencies=mode_frequencies,
        plate_length=plate_length,
        plate_width=plate_width
    )
    
    # Calculate total score
    total_score = analyzer.total_score(zone_results)
    
    return {
        "zone_results": zone_results,
        "total_score": total_score,
        "zones": zones,
        "summary": {
            zone_r.zone.name: {
                "achieved_frequencies": zone_r.achieved_frequencies,
                "target_frequencies": zone_r.zone.target_frequencies,
                "frequency_error": zone_r.frequency_error,
                "coupling_efficiency": zone_r.coupling_efficiency
            }
            for zone_r in zone_results
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PLATE OPTIMIZER - Physics-Based Shape Optimization")
    print("=" * 70)
    
    # Test auto-design (legacy)
    result = auto_design_plate(
        height_m=1.80,
        material="birch_plywood",
        thickness_mm=15,
        with_cutouts=False
    )
    
    print("\nChakra target frequencies (legacy):")
    for c in CHAKRA_TARGETS:
        print(f"  {c.name}: {c.frequency_hz:.0f} Hz @ {c.position_spine:.0%} of spine")
    
    # Test zone-based optimization (v2.0)
    if HAS_ZONE_MODULES:
        print("\n" + "=" * 70)
        print("ZONE-BASED OPTIMIZATION (v2.0)")
        print("=" * 70)
        
        try:
            zone_result = zone_optimize_plate(
                zone_preset="vat",
                height_m=1.80,
                max_iterations=20,
                verbose=True
            )
        except Exception as e:
            print(f"Zone optimization test failed: {e}")

