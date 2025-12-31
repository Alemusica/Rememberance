"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PHYSICS CONSTANTS - Single Source of Truth                ║
║                                                                              ║
║   Centralized physical constants for vibroacoustic calculations.            ║
║   Import from here instead of hardcoding values across modules.             ║
║                                                                              ║
║   This module complements materials.py by providing:                         ║
║   • Universal physics constants (speed of sound, air properties)            ║
║   • Default plate dimensions and limits                                      ║
║   • Frequency bands for vibroacoustic therapy                               ║
║   • Structural safety thresholds                                             ║
║                                                                              ║
║   References:                                                                ║
║   • ANSI/ASA S1.1-2013 (Acoustical Terminology)                             ║
║   • ISO 226:2003 (Equal-Loudness Contours)                                  ║
║   • EN 1995-1-1 (Eurocode 5: Timber Structures)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL PHYSICS CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AirProperties:
    """Standard air properties at 20°C, 1 atm."""
    density: float = 1.204          # kg/m³
    speed_of_sound: float = 343.0   # m/s
    impedance: float = 413.0        # Pa·s/m (ρ·c)
    viscosity: float = 1.82e-5      # Pa·s (dynamic)
    
    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity ν = μ/ρ [m²/s]."""
        return self.viscosity / self.density


AIR = AirProperties()


# ══════════════════════════════════════════════════════════════════════════════
# FREQUENCY BANDS - Vibroacoustic Therapy
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FrequencyBands:
    """
    Frequency bands for vibroacoustic therapy optimization.
    
    References:
    - Skille (1989): VAT 20-120 Hz primary therapeutic range
    - Wigram (1996): Extended range to 200 Hz for music therapy
    - Boyd-Brewer (2004): 30-80 Hz for pain management
    """
    # Primary therapy band
    therapy_low: float = 20.0        # Hz - Infrasonic threshold
    therapy_high: float = 200.0      # Hz - Upper therapy limit
    
    # Sub-bands for specific effects
    relaxation: Tuple[float, float] = (30.0, 60.0)    # Deep relaxation
    pain_relief: Tuple[float, float] = (40.0, 80.0)   # Pain management
    circulation: Tuple[float, float] = (50.0, 100.0)  # Blood flow
    
    # Binaural audio band (head/ears)
    binaural_low: float = 20.0       # Hz
    binaural_high: float = 500.0     # Hz - Extended for music
    
    # Flatness targets (dB variation)
    spine_flatness_target: float = 10.0   # dB peak-to-peak max
    ear_flatness_target: float = 6.0      # dB (tighter for audio)
    lr_balance_target: float = 3.0        # dB L/R difference max


FREQ_BANDS = FrequencyBands()


# ══════════════════════════════════════════════════════════════════════════════
# PLATE GEOMETRY DEFAULTS & LIMITS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PlateGeometryLimits:
    """
    Geometric constraints for DML therapy plates.
    
    Based on:
    - Typical adult body dimensions (P5-P95 percentiles)
    - Manufacturing constraints (CNC router limits)
    - Modal density requirements (20+ modes in therapy band)
    """
    # Length (person lies on plate)
    length_min: float = 1.5          # m - Minimum for child
    length_max: float = 2.2          # m - Maximum for tall adult
    length_default: float = 2.0      # m - Average adult
    
    # Width (shoulder width + margin)
    width_min: float = 0.4           # m - Minimum practical
    width_max: float = 0.8           # m - Maximum practical
    width_default: float = 0.6       # m - Standard width
    
    # Thickness
    thickness_min: float = 0.008     # m (8mm) - Structural minimum
    thickness_max: float = 0.025     # m (25mm) - Modal frequency limit
    thickness_default: float = 0.015 # m (15mm) - Balanced choice
    
    # Aspect ratio L/W
    aspect_ratio_min: float = 2.0    # Avoid square plates (modal clustering)
    aspect_ratio_max: float = 5.0    # Avoid very narrow plates
    
    @property
    def default_dimensions(self) -> Tuple[float, float, float]:
        """Return (length, width, thickness) defaults."""
        return (self.length_default, self.width_default, self.thickness_default)


PLATE_LIMITS = PlateGeometryLimits()


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL SAFETY THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StructuralSafetyLimits:
    """
    Structural safety thresholds for person-on-plate loading.
    
    Based on:
    - Eurocode 5 timber design guidelines
    - Typical person weight (50-120 kg range)
    - User comfort requirements
    """
    # Deflection limits
    max_deflection_mm: float = 10.0         # mm - Maximum comfortable
    warning_deflection_mm: float = 7.0      # mm - Warning threshold
    
    # Safety factors
    min_safety_factor: float = 2.0          # Yield stress / max stress
    recommended_safety_factor: float = 3.0  # For production plates
    
    # Loading assumptions
    typical_person_weight_kg: float = 75.0  # kg - Average adult
    max_person_weight_kg: float = 120.0     # kg - Design maximum
    contact_efficiency: float = 0.6         # Fraction of weight in contact
    
    # Stress limits (fraction of yield)
    max_stress_fraction: float = 0.5        # 50% of yield stress


SAFETY_LIMITS = StructuralSafetyLimits()


# ══════════════════════════════════════════════════════════════════════════════
# MODAL ANALYSIS DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModalAnalysisConfig:
    """
    Default parameters for modal analysis.
    
    Based on:
    - Sufficient mode coverage in therapy band
    - Computational efficiency
    - Resolution requirements
    """
    # Number of modes
    n_modes_default: int = 15        # Sufficient for 20-200 Hz
    n_modes_detailed: int = 30       # For fine optimization
    n_modes_quick: int = 8           # For rapid preview
    
    # Frequency resolution
    n_freq_points_default: int = 50  # Points across band
    n_freq_points_detailed: int = 100
    
    # Mesh resolution (elements per dimension)
    mesh_nx_default: int = 40        # Elements in length
    mesh_ny_default: int = 24        # Elements in width
    mesh_coarse: Tuple[int, int] = (20, 12)
    mesh_fine: Tuple[int, int] = (60, 36)


MODAL_CONFIG = ModalAnalysisConfig()


# ══════════════════════════════════════════════════════════════════════════════
# EXCITER DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExciterDefaults:
    """
    Default parameters for DML exciters.
    
    Based on:
    - Commercial exciter specifications (Dayton, TEAX)
    - Optimal coupling research (Bai & Liu 2004)
    """
    # Diameter range
    diameter_min: float = 0.025      # m (25mm) - Small exciter
    diameter_max: float = 0.075      # m (75mm) - Large exciter
    diameter_default: float = 0.050  # m (50mm) - TEAX25C60-8
    
    # Count limits
    count_min: int = 1
    count_max: int = 6
    count_default: int = 2           # L/R stereo pair
    
    # Position constraints (normalized 0-1)
    margin: float = 0.1              # Distance from edges
    
    # Modal coupling threshold
    min_coupling_score: float = 0.3  # Minimum acceptable


EXCITER_DEFAULTS = ExciterDefaults()


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OptimizationDefaults:
    """
    Default parameters for genetic optimization.
    
    Based on:
    - NSGA-II guidelines (Deb 2002)
    - Exciter placement research (Bai & Liu 2004)
    """
    # Population
    population_size_default: int = 50
    population_size_quick: int = 20
    population_size_thorough: int = 100
    
    # Generations
    generations_default: int = 100
    generations_quick: int = 30
    generations_thorough: int = 200
    
    # Mutation/crossover rates
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    
    # Early stopping
    stall_generations: int = 15      # Stop if no improvement
    stall_threshold: float = 0.001   # Minimum improvement


OPTIM_DEFAULTS = OptimizationDefaults()


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_default_plate_config() -> dict:
    """
    Return default plate configuration dictionary.
    
    Compatible with JAX FEM and other modules.
    """
    return {
        'length': PLATE_LIMITS.length_default,
        'width': PLATE_LIMITS.width_default,
        'thickness': PLATE_LIMITS.thickness_default,
        'nx': MODAL_CONFIG.mesh_nx_default,
        'ny': MODAL_CONFIG.mesh_ny_default,
    }


def get_therapy_frequency_range() -> Tuple[float, float]:
    """Return standard therapy frequency range (Hz)."""
    return (FREQ_BANDS.therapy_low, FREQ_BANDS.therapy_high)


def is_deflection_safe(deflection_mm: float) -> bool:
    """Check if deflection is within safe limits."""
    return deflection_mm <= SAFETY_LIMITS.max_deflection_mm


def calculate_safety_factor(max_stress: float, yield_stress: float) -> float:
    """Calculate stress safety factor."""
    if max_stress <= 0:
        return SAFETY_LIMITS.recommended_safety_factor
    return yield_stress / max_stress
