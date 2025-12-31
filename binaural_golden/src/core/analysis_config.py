"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ANALYSIS CONFIG - Centralized Parameters                  ║
║                                                                              ║
║   Single source of truth for:                                                ║
║   • Modal analysis resolution (grid spacing)                                 ║
║   • Gene activation thresholds (seed→bloom→freeze)                          ║
║   • Physics constraints & tolerances                                         ║
║   • Evolutionary operator parameters                                         ║
║                                                                              ║
║   DESIGN PRINCIPLES:                                                         ║
║   • Protocol-based (PEP-544) for flexibility                                 ║
║   • Immutable defaults with runtime overrides                                ║
║   • Hierarchical: Global → Domain → Task specific                           ║
║                                                                              ║
║   REFERENCE:                                                                 ║
║   • Scattered params consolidated from fitness.py, plate_unified.py,         ║
║     plate_adapters.py (target_spacing_mm = 40.0)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Dict, Any, Optional, List, Tuple, runtime_checkable
from enum import Enum, auto
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS - Interface definitions (PEP-544)
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for config providers - allows dependency injection."""
    
    def get_modal_config(self) -> 'ModalAnalysisConfig':
        """Get modal analysis configuration."""
        ...
    
    def get_gene_activation_config(self) -> 'GeneActivationConfig':
        """Get gene activation thresholds."""
        ...
    
    def get_evolution_config(self) -> 'EvolutionConfig':
        """Get evolutionary algorithm parameters."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS - Activation phases
# ══════════════════════════════════════════════════════════════════════════════

class GenePhase(Enum):
    """
    Gene activation phases - like biological development.
    
    SEED:   Only position genes active (exploration)
    BLOOM:  Position + emission genes active (refinement)
    FREEZE: Position frozen, only emission evolvable (production)
    """
    SEED = auto()    # "Il seme non parla dei petali"
    BLOOM = auto()   # Emission genes activated
    FREEZE = auto()  # Position locked (CNC done)


class ActivationTrigger(Enum):
    """What triggers transition between phases."""
    POSITION_CONVERGED = auto()    # σ(x,y) < threshold for N generations
    FITNESS_PLATEAU = auto()       # Velocity < threshold for N generations
    USER_REQUEST = auto()          # Via PokayokeObserver
    PHYSICS_HINT = auto()          # Modal analysis suggests optimization
    GENERATION_COUNT = auto()      # After N generations (curriculum static)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASSES - Immutable parameter containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModalAnalysisConfig:
    """
    Configuration for modal/FEM analysis.
    
    CRITICAL: target_spacing_mm was scattered across 3 files!
    Now centralized here.
    """
    # Grid resolution
    target_spacing_mm: float = 40.0          # Max grid spacing (finer = more accurate)
    min_spacing_mm: float = 10.0             # Don't go finer than this (memory)
    max_spacing_mm: float = 50.0             # Don't go coarser than this (accuracy)
    
    # Adaptive resolution
    adaptive_resolution: bool = True         # Auto-compute based on plate size
    resolution_safety_factor: float = 1.2    # Multiply computed spacing by this
    
    # Frequency range
    freq_min_hz: float = 20.0                # Lowest frequency of interest
    freq_max_hz: float = 500.0               # Highest frequency (DML range)
    num_modes: int = 20                      # Number of modes to compute
    
    # Tolerances
    convergence_tol: float = 1e-6            # FEM solver convergence
    mass_matrix_lumped: bool = True          # Use lumped mass (faster)
    
    def compute_adaptive_spacing(self, plate_length_m: float, plate_width_m: float) -> float:
        """
        Compute optimal grid spacing based on plate dimensions.
        
        Rule: At least 15-20 points along smallest dimension.
        """
        min_dim = min(plate_length_m, plate_width_m)
        min_dim_mm = min_dim * 1000
        
        # Target: ~20 points along minimum dimension
        suggested_spacing = min_dim_mm / 20
        
        # Clamp to valid range
        spacing = np.clip(
            suggested_spacing * self.resolution_safety_factor,
            self.min_spacing_mm,
            self.target_spacing_mm  # Never exceed target
        )
        
        return float(spacing)


@dataclass(frozen=True)
class GeneActivationConfig:
    """
    Configuration for staged gene activation (seed→bloom→freeze).
    
    PHILOSOPHY: "Il seme non parla dei petali" - genes activate when needed.
    """
    # Initial phase
    initial_phase: GenePhase = GenePhase.SEED
    
    # SEED → BLOOM transition thresholds
    position_convergence_sigma: float = 0.05    # σ(x,y) threshold
    position_convergence_generations: int = 5   # Must hold for N gens
    
    fitness_plateau_velocity: float = 0.001     # Improvement rate threshold
    fitness_plateau_generations: int = 3        # Must hold for N gens
    
    # Curriculum learning (static schedule)
    curriculum_bloom_generation: int = 50       # Force bloom after N gens
    curriculum_enabled: bool = False            # Use static schedule?
    
    # BLOOM → FREEZE (user/external trigger only)
    allow_auto_freeze: bool = False             # Require explicit freeze
    
    # Emission bounds when active
    emission_bounds: 'EmissionBounds' = None    # Set in __post_init__
    
    def __post_init__(self):
        if self.emission_bounds is None:
            # Use default bounds - frozen dataclass workaround
            object.__setattr__(self, 'emission_bounds', EmissionBounds())
    
    def should_activate_emission(
        self,
        position_sigma: float,
        stable_generations: int,
        fitness_velocity: float,
        plateau_generations: int,
        current_generation: int
    ) -> Tuple[bool, ActivationTrigger]:
        """
        Check if emission genes should be activated.
        
        Returns:
            (should_activate, trigger_reason)
        """
        # Check position convergence
        if (position_sigma < self.position_convergence_sigma and 
            stable_generations >= self.position_convergence_generations):
            return True, ActivationTrigger.POSITION_CONVERGED
        
        # Check fitness plateau
        if (abs(fitness_velocity) < self.fitness_plateau_velocity and
            plateau_generations >= self.fitness_plateau_generations):
            return True, ActivationTrigger.FITNESS_PLATEAU
        
        # Check curriculum schedule
        if self.curriculum_enabled and current_generation >= self.curriculum_bloom_generation:
            return True, ActivationTrigger.GENERATION_COUNT
        
        return False, None


@dataclass(frozen=True)
class EmissionBounds:
    """
    Bounds for emission parameters when active.
    
    Based on hardware constraints (JAB4 + Dayton DAEX25).
    """
    # Phase [degrees]
    phase_min: float = 0.0
    phase_max: float = 360.0
    
    # Delay [samples @ 48kHz]
    # Max delay ~0.35ms for 1.9m plate at sound speed in spruce
    delay_min: int = 0
    delay_max: int = 17  # ~0.35ms @ 48kHz
    
    # Gain [dB relative to nominal]
    gain_min_db: float = -12.0
    gain_max_db: float = 6.0
    
    # Polarity (boolean, no bounds needed)


@dataclass(frozen=True)
class EvolutionConfig:
    """
    Configuration for evolutionary algorithm.
    """
    # Population
    population_size: int = 50
    elite_count: int = 5
    
    # Mutation rates (adaptive starting points)
    mutation_rate_position: float = 0.2
    mutation_rate_emission: float = 0.15     # Lower - finer tuning
    
    # Crossover
    crossover_rate: float = 0.8
    
    # Termination
    max_generations: int = 200
    fitness_target: float = 0.95
    stagnation_limit: int = 20
    
    # RDNN integration
    use_rdnn_guidance: bool = True
    rdnn_hidden_dim: int = 64
    
    # Physics rules integration
    use_physics_rules: bool = True
    physics_cache_size: int = 100


@dataclass(frozen=True)
class ObserverConfig:
    """
    Configuration for PokayokeObserver behavior.
    
    KEY: Pause and ask user, don't auto-adjust!
    """
    # When to alert user
    stagnation_threshold: int = 10           # Generations without improvement
    diversity_collapse_threshold: float = 0.1  # Population diversity
    regression_threshold: float = -0.05      # Fitness drop
    
    # User interaction mode
    auto_adjust: bool = False                # If True: auto-fix. If False: PAUSE + ASK
    timeout_seconds: float = 300.0           # Wait for user response
    default_action: str = "retry"            # If timeout: retry, skip, abort
    
    # Logging
    verbose: bool = True
    log_all_generations: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# MASTER CONFIG - Combines all sub-configs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisConfig:
    """
    Master configuration combining all sub-configs.
    
    USAGE:
        config = AnalysisConfig()  # Use defaults
        config = AnalysisConfig.for_quick_test()  # Preset for testing
        config = AnalysisConfig.for_production()  # Preset for production
        
        # Override specific values
        config = AnalysisConfig(
            modal=ModalAnalysisConfig(target_spacing_mm=30.0)
        )
    """
    modal: ModalAnalysisConfig = field(default_factory=ModalAnalysisConfig)
    gene_activation: GeneActivationConfig = field(default_factory=GeneActivationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    
    # Metadata
    name: str = "default"
    version: str = "3.0"
    
    # === Protocol implementation ===
    
    def get_modal_config(self) -> ModalAnalysisConfig:
        return self.modal
    
    def get_gene_activation_config(self) -> GeneActivationConfig:
        return self.gene_activation
    
    def get_evolution_config(self) -> EvolutionConfig:
        return self.evolution
    
    # === Factory methods for common presets ===
    
    @classmethod
    def for_quick_test(cls) -> 'AnalysisConfig':
        """Fast settings for unit tests."""
        return cls(
            modal=ModalAnalysisConfig(
                target_spacing_mm=50.0,  # Coarser
                num_modes=10
            ),
            evolution=EvolutionConfig(
                population_size=20,
                max_generations=50
            ),
            name="quick_test"
        )
    
    @classmethod
    def for_production(cls) -> 'AnalysisConfig':
        """High-quality settings for real optimization."""
        return cls(
            modal=ModalAnalysisConfig(
                target_spacing_mm=30.0,  # Finer
                num_modes=30,
                adaptive_resolution=True
            ),
            gene_activation=GeneActivationConfig(
                curriculum_enabled=True,
                curriculum_bloom_generation=30
            ),
            evolution=EvolutionConfig(
                population_size=100,
                max_generations=500,
                use_rdnn_guidance=True,
                use_physics_rules=True
            ),
            observer=ObserverConfig(
                auto_adjust=False  # PAUSE + ASK USER
            ),
            name="production"
        )
    
    @classmethod
    def for_exploration(cls) -> 'AnalysisConfig':
        """Settings for exploratory optimization (wide search)."""
        return cls(
            gene_activation=GeneActivationConfig(
                initial_phase=GenePhase.SEED,
                curriculum_enabled=False  # Let convergence trigger bloom
            ),
            evolution=EvolutionConfig(
                mutation_rate_position=0.3,  # Higher exploration
                population_size=80
            ),
            name="exploration"
        )
    
    @classmethod
    def for_refinement(cls) -> 'AnalysisConfig':
        """Settings for fine-tuning existing design."""
        return cls(
            gene_activation=GeneActivationConfig(
                initial_phase=GenePhase.BLOOM  # Start with emission active
            ),
            evolution=EvolutionConfig(
                mutation_rate_position=0.1,   # Lower - positions roughly ok
                mutation_rate_emission=0.25,  # Higher - tune DSP
                population_size=40
            ),
            name="refinement"
        )


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL DEFAULT - Can be overridden
# ══════════════════════════════════════════════════════════════════════════════

_default_config: AnalysisConfig = AnalysisConfig()


def get_default_config() -> AnalysisConfig:
    """Get the current default configuration."""
    return _default_config


def set_default_config(config: AnalysisConfig) -> None:
    """Set the default configuration."""
    global _default_config
    _default_config = config


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - For backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

def get_target_spacing_mm(
    plate_length_m: Optional[float] = None,
    plate_width_m: Optional[float] = None,
    config: Optional[AnalysisConfig] = None
) -> float:
    """
    Get target grid spacing in mm.
    
    BACKWARD COMPATIBILITY:
    This function replaces the scattered `target_spacing_mm = 40.0` 
    in fitness.py, plate_unified.py, plate_adapters.py.
    
    Args:
        plate_length_m: Plate length in meters (for adaptive resolution)
        plate_width_m: Plate width in meters (for adaptive resolution)
        config: Config to use (defaults to global)
    
    Returns:
        Grid spacing in mm
    """
    if config is None:
        config = get_default_config()
    
    modal_config = config.get_modal_config()
    
    if modal_config.adaptive_resolution and plate_length_m and plate_width_m:
        return modal_config.compute_adaptive_spacing(plate_length_m, plate_width_m)
    
    return modal_config.target_spacing_mm


# ══════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES - For cleaner type hints elsewhere
# ══════════════════════════════════════════════════════════════════════════════

ModalConfig = ModalAnalysisConfig
GeneConfig = GeneActivationConfig
EvoConfig = EvolutionConfig
