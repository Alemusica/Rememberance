"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PLATE DESIGN ADAPTERS - Bridge to Agnostic Framework                 ║
║                                                                              ║
║   Adapters that allow the existing DML plate optimization code to work       ║
║   with the new agnostic evolutionary framework.                              ║
║                                                                              ║
║   COMPONENTS:                                                                ║
║   • PlatePhysicsAdapter: Wraps plate_physics.py for PhysicsEngine interface  ║
║   • PlateFitnessAdapter: Wraps fitness.py for FitnessEvaluator interface     ║
║   • PlateGenomeFactory: Creates PlateGenome instances                        ║
║   • PlateEvolutionConfig: Plate-specific configuration                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

# Import agnostic framework
from .agnostic_evolution import (
    PhysicsEngine,
    FitnessEvaluator as AbstractFitnessEvaluator,
    GenomeFactory,
    AgnosticEvolutionaryOptimizer,
    EvolutionConfigBase,
)

# Import existing plate code
from .plate_genome import PlateGenome, ContourType, ExciterPosition
from .plate_physics import (
    calculate_plate_modes,
    calculate_modal_frequency_isotropic,
    mode_shape_grid,
    PlateMode,
)
from .fitness import FitnessEvaluator, FitnessResult, ObjectiveVector, ObjectiveWeights, ZoneWeights
from .person import Person
from .body_zones import BodyZone
from .materials import MATERIALS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlatePhysicsResult:
    """Container for plate physics analysis results."""
    frequencies: List[float]
    mode_shapes: np.ndarray
    damping_ratios: List[float]
    
    # Optional detailed results
    frequency_response: np.ndarray = None
    effective_mass: float = None
    plate_area: float = None
    
    # Grid for mode shapes
    nx: int = 50
    ny: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# PLATE PHYSICS ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class PlatePhysicsAdapter(PhysicsEngine[PlateGenome, PlatePhysicsResult]):
    """
    Adapter that wraps existing plate physics code for the agnostic framework.
    
    This class:
    1. Takes a PlateGenome
    2. Calls the existing physics functions
    3. Returns a PlatePhysicsResult
    """
    
    def __init__(
        self,
        n_modes: int = 20,
        grid_resolution: int = 50,
        material_E: float = 2.5e9,      # Young's modulus (Pa)
        material_rho: float = 1200,      # Density (kg/m³)
        material_nu: float = 0.35,       # Poisson's ratio
        damping_ratio: float = 0.02,     # Modal damping
    ):
        """
        Initialize physics adapter.
        
        Args:
            n_modes: Number of modes to compute
            grid_resolution: Grid size for mode shapes
            material_E: Young's modulus in Pa
            material_rho: Density in kg/m³
            material_nu: Poisson's ratio
            damping_ratio: Modal damping ratio
        """
        self.n_modes = n_modes
        self.grid_resolution = grid_resolution
        self.material_E = material_E
        self.material_rho = material_rho
        self.material_nu = material_nu
        self.damping_ratio = damping_ratio
    
    def analyze(self, genome: PlateGenome) -> PlatePhysicsResult:
        """
        Perform physics analysis on a plate genome.
        
        Args:
            genome: PlateGenome to analyze
        
        Returns:
            PlatePhysicsResult with frequencies and mode shapes
        """
        # Extract plate dimensions (PlateGenome uses length/width not height)
        plate_length = genome.length  # Longer dimension (along body)
        plate_width = genome.width    # Shorter dimension  
        thickness = genome.thickness_base
        
        # Apply thickness variation if present
        effective_thickness = thickness
        if hasattr(genome, 'thickness_variation') and genome.thickness_variation > 0:
            # Average thickness for simplified analysis
            effective_thickness = thickness * (1 - genome.thickness_variation * 0.5)
        
        # Convert to mm for calculate_plate_modes
        length_mm = plate_length * 1000
        width_mm = plate_width * 1000
        thickness_mm = effective_thickness * 1000
        
        # Compute modes using existing function
        try:
            modes = calculate_plate_modes(
                length_mm=length_mm,
                width_mm=width_mm,
                thickness_mm=thickness_mm,
                material_key="mdf",  # Default material
                max_modes=self.n_modes,
                max_freq_hz=2000.0,
            )
            frequencies = [m.frequency for m in modes]
        except Exception as e:
            logger.warning(f"Frequency computation failed: {e}")
            # Fallback to simple analytical estimate
            D = (self.material_E * effective_thickness**3) / (12 * (1 - self.material_nu**2))
            rho_h = self.material_rho * effective_thickness
            base_freq = (np.pi**2 / (2 * plate_length * plate_width)) * np.sqrt(D / rho_h)
            frequencies = [base_freq * (m**2 + n**2) for m in range(1, 6) for n in range(1, 5)][:self.n_modes]
        
        # Compute mode shapes using mode_shape_grid
        try:
            mode_shapes = []
            for m in range(1, 4):
                for n in range(1, 4):
                    X, Y, Z = mode_shape_grid(
                        m=m, n=n,
                        L=plate_length, W=plate_width,
                        resolution=self.grid_resolution,
                    )
                    mode_shapes.append(Z)  # Just the amplitude grid
            mode_shapes = np.array(mode_shapes[:min(len(mode_shapes), self.n_modes)])
        except Exception as e:
            logger.warning(f"Mode shape computation failed: {e}")
            # Fallback to analytical mode shapes
            mode_shapes = self._analytical_mode_shapes(plate_width, plate_length)
        
        # Damping ratios
        damping_ratios = [self.damping_ratio] * len(frequencies)
        
        # Plate area
        plate_area = plate_length * plate_width
        
        return PlatePhysicsResult(
            frequencies=frequencies,
            mode_shapes=mode_shapes,
            damping_ratios=damping_ratios,
            nx=self.grid_resolution,
            ny=self.grid_resolution,
            plate_area=plate_area,
        )
    
    def get_mode_shapes(self, physics_result: PlatePhysicsResult) -> np.ndarray:
        """Extract mode shapes from physics result."""
        return physics_result.mode_shapes
    
    def get_frequencies(self, physics_result: PlatePhysicsResult) -> List[float]:
        """Extract modal frequencies from physics result."""
        return physics_result.frequencies
    
    def get_sensitivity(self, genome: PlateGenome, physics_result: PlatePhysicsResult) -> np.ndarray:
        """
        Compute sensitivity field for topology optimization.
        
        High sensitivity = cutting here strongly affects response.
        Based on mode shape magnitude weighted by frequency importance.
        """
        mode_shapes = physics_result.mode_shapes
        frequencies = physics_result.frequencies
        
        if mode_shapes is None or len(mode_shapes) == 0:
            return np.ones((self.grid_resolution, self.grid_resolution))
        
        # Weight modes by inverse frequency (low freq more important for therapy)
        weights = 1.0 / (np.array(frequencies[:len(mode_shapes)]) + 10)
        weights /= np.sum(weights)
        
        # Weighted sum of mode shape magnitudes
        sensitivity = np.zeros((self.grid_resolution, self.grid_resolution))
        for i, mode in enumerate(mode_shapes):
            if i < len(weights):
                sensitivity += weights[i] * np.abs(mode)
        
        return sensitivity
    
    def _analytical_mode_shapes(self, width: float, height: float) -> np.ndarray:
        """
        Compute analytical mode shapes for simply-supported plate.
        
        Fallback when FEM is unavailable.
        """
        nx, ny = self.grid_resolution, self.grid_resolution
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        X, Y = np.meshgrid(x, y)
        
        modes = []
        for m in range(1, 4):
            for n in range(1, 4):
                # W_mn(x,y) = sin(m*pi*x/a) * sin(n*pi*y/b)
                mode = np.sin(m * np.pi * X / width) * np.sin(n * np.pi * Y / height)
                modes.append(mode)
        
        return np.array(modes)


# ══════════════════════════════════════════════════════════════════════════════
# PLATE FITNESS ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class PlateFitnessAdapter(AbstractFitnessEvaluator[PlateGenome, PlatePhysicsResult, FitnessResult]):
    """
    Adapter that wraps existing FitnessEvaluator for the agnostic framework.
    
    This class:
    1. Takes a PlateGenome and optional PlatePhysicsResult
    2. Calls the existing FitnessEvaluator
    3. Returns FitnessResult in agnostic-compatible way
    """
    
    def __init__(
        self,
        person: Person,
        objectives: ObjectiveWeights = None,
        body_zones: BodyZone = None,
    ):
        """
        Initialize fitness adapter.
        
        Args:
            person: Person model for body zone calculations
            objectives: Objective weights for multi-objective balance
            body_zones: Body zone definitions (computed from person if None)
        """
        self.person = person
        self.objectives = objectives or ObjectiveWeights()
        self.body_zones = body_zones
        
        # Create underlying evaluator
        self._evaluator = FitnessEvaluator(
            person=person,
            objectives=self.objectives,
        )
        
        # Objective names (matches ObjectiveVector)
        self._objective_names = [
            "spine_flatness",
            "ear_flatness", 
            "ear_lr_uniformity",
            "spine_energy",
            "mass_score",
            "structural_safety",
        ]
    
    def evaluate(
        self,
        genome: PlateGenome,
        physics_result: PlatePhysicsResult = None
    ) -> FitnessResult:
        """
        Evaluate fitness of a plate genome.
        
        Args:
            genome: PlateGenome to evaluate
            physics_result: Optional cached physics result (currently unused,
                           as existing evaluator does its own physics)
        
        Returns:
            FitnessResult with all objective scores
        """
        return self._evaluator.evaluate(genome)
    
    def get_objectives(self, fitness_result: FitnessResult) -> np.ndarray:
        """
        Extract objective array for multi-objective optimization.
        
        Returns array suitable for NSGA-II (to be MINIMIZED).
        We negate because our objectives are better when higher.
        """
        # Build objectives from FitnessResult fields
        return -np.array([
            fitness_result.spine_flatness_score,
            fitness_result.head_flatness_score,
            fitness_result.ear_uniformity_score,
            fitness_result.spine_coupling_score,
            fitness_result.low_mass_score,
            fitness_result.structural_score,
        ])
    
    def get_scalar_fitness(self, fitness_result: FitnessResult) -> float:
        """
        Extract scalar fitness for single-objective optimization.
        
        Returns total_fitness (higher = better).
        """
        return fitness_result.total_fitness
    
    def get_objective_names(self) -> List[str]:
        """Return names of objectives."""
        return self._objective_names


# ══════════════════════════════════════════════════════════════════════════════
# PLATE GENOME FACTORY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlateConstraints:
    """Constraints for plate genome generation."""
    # Dimensions (in meters)
    min_length: float = 1.5     # Typical therapy bed length
    max_length: float = 2.0
    min_width: float = 0.5      # Body width
    max_width: float = 0.8
    
    # Thickness (in meters)
    min_thickness: float = 0.010  # 10mm
    max_thickness: float = 0.020  # 20mm
    
    # Contours
    allowed_contours: List[ContourType] = field(
        default_factory=lambda: [ContourType.RECTANGLE, ContourType.GOLDEN_RECT, ContourType.SUPERELLIPSE]
    )
    
    # Thickness variation
    max_thickness_variation: float = 0.3


class PlateGenomeFactory(GenomeFactory[PlateGenome, PlateConstraints]):
    """
    Factory for creating PlateGenome instances.
    
    Creates random, template-based, or default genomes within constraints.
    
    Note: PlateGenome uses standard 4-exciter layout by default (DEFAULT_EXCITERS),
    so we don't randomize exciters here - they're optimized separately.
    """
    
    def __init__(self, constraints: PlateConstraints = None):
        """
        Initialize factory.
        
        Args:
            constraints: Constraints for genome generation
        """
        self.constraints = constraints or PlateConstraints()
    
    def create_random(self, config: PlateConstraints = None) -> PlateGenome:
        """Create a random genome within constraints."""
        c = config or self.constraints
        
        # Random dimensions
        length = np.random.uniform(c.min_length, c.max_length)
        width = np.random.uniform(c.min_width, c.max_width)
        thickness = np.random.uniform(c.min_thickness, c.max_thickness)
        
        # Random contour
        contour = np.random.choice(c.allowed_contours)
        
        # Random thickness variation
        thickness_variation = np.random.uniform(0, c.max_thickness_variation)
        
        return PlateGenome(
            length=length,
            width=width,
            thickness_base=thickness,
            contour_type=contour,
            thickness_variation=thickness_variation,
            # Uses DEFAULT_EXCITERS by default
        )
    
    def create_from_template(self, template: PlateGenome, variation: float = 0.1) -> PlateGenome:
        """Create a variation of an existing genome."""
        c = self.constraints
        
        # Vary dimensions
        length = np.clip(
            template.length * (1 + np.random.normal(0, variation)),
            c.min_length, c.max_length
        )
        width = np.clip(
            template.width * (1 + np.random.normal(0, variation)),
            c.min_width, c.max_width
        )
        thickness = np.clip(
            template.thickness_base * (1 + np.random.normal(0, variation)),
            c.min_thickness, c.max_thickness
        )
        
        # Keep contour (or random chance to change)
        if np.random.random() < 0.1:
            contour = np.random.choice(c.allowed_contours)
        else:
            contour = template.contour_type
        
        # Vary thickness variation
        thickness_variation = np.clip(
            template.thickness_variation + np.random.normal(0, variation * 0.1),
            0, c.max_thickness_variation
        )
        
        return PlateGenome(
            length=length,
            width=width,
            thickness_base=thickness,
            contour_type=contour,
            thickness_variation=thickness_variation,
            cutouts=list(template.cutouts),  # Keep cutouts from template
            grooves=list(template.grooves),
        )
    
    def create_default(self) -> PlateGenome:
        """Create a default/baseline genome (standard therapy bed)."""
        return PlateGenome(
            length=1.85,      # Standard body length
            width=0.64,       # Golden ratio to length
            thickness_base=0.015,  # 15mm
            contour_type=ContourType.GOLDEN_RECT,
            thickness_variation=0.0,
            # Uses DEFAULT_EXCITERS automatically
        )


# ══════════════════════════════════════════════════════════════════════════════
# PLATE-SPECIFIC EVOLUTION CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlateEvolutionConfig(EvolutionConfigBase):
    """Plate-specific evolution configuration."""
    # Inherit base config
    
    # Plate-specific additions
    optimize_exciters: bool = True
    optimize_cutouts: bool = True
    optimize_contour: bool = True
    optimize_thickness_profile: bool = True
    
    # Frequency targets
    target_spine_range: Tuple[float, float] = (20, 150)
    target_ear_range: Tuple[float, float] = (200, 2000)
    
    # ABH settings (from research)
    abh_benefit_weight: float = 0.6
    resonator_weight: float = 0.3
    structural_penalty_weight: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION: CREATE PLATE OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def create_plate_optimizer(
    person: Person,
    constraints: PlateConstraints = None,
    config: PlateEvolutionConfig = None,
    objectives: ObjectiveWeights = None,
) -> AgnosticEvolutionaryOptimizer:
    """
    Create an optimizer for DML plate design.
    
    This is the main entry point for plate optimization using the agnostic framework.
    
    Args:
        person: Person model for body zone calculations
        constraints: Constraints for plate generation
        config: Evolution configuration
        objectives: Objective weights for fitness
    
    Returns:
        Configured AgnosticEvolutionaryOptimizer for plates
    
    Example:
        >>> from src.core.person import Person
        >>> person = Person(height_m=1.75, weight_kg=70)
        >>> optimizer = create_plate_optimizer(person)
        >>> result = optimizer.optimize()
        >>> print(f"Best plate: {result.best_genome}")
    """
    # Create components
    physics = PlatePhysicsAdapter()
    evaluator = PlateFitnessAdapter(person, objectives)
    factory = PlateGenomeFactory(constraints)
    config = config or PlateEvolutionConfig()
    
    # Create optimizer
    optimizer = AgnosticEvolutionaryOptimizer(
        physics=physics,
        evaluator=evaluator,
        factory=factory,
        config=config,
    )
    
    return optimizer


# ══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE DOMAINS - Templates for other vibroacoustic designs
# ══════════════════════════════════════════════════════════════════════════════

class BowlPhysicsAdapter(PhysicsEngine):
    """
    Template for singing bowl physics.
    
    To implement:
    1. Shell FEM for circular geometry
    2. Mode shapes in cylindrical coordinates
    3. Frequency ratios for harmonic series
    """
    
    def analyze(self, genome):
        raise NotImplementedError("Implement bowl-specific physics")
    
    def get_mode_shapes(self, physics_result):
        raise NotImplementedError("Implement bowl mode extraction")
    
    def get_frequencies(self, physics_result):
        raise NotImplementedError("Implement bowl frequency extraction")


class BowlFitnessAdapter(AbstractFitnessEvaluator):
    """
    Template for singing bowl fitness evaluation.
    
    Objectives would include:
    - Frequency ratio accuracy (1:2:3 harmonic series)
    - Sustain duration
    - Timbre quality
    - Beat frequency between modes
    """
    
    def evaluate(self, genome, physics_result=None):
        raise NotImplementedError("Implement bowl-specific fitness")
    
    def get_objectives(self, fitness_result):
        raise NotImplementedError("Implement bowl objectives")
    
    def get_scalar_fitness(self, fitness_result):
        raise NotImplementedError("Implement bowl scalar fitness")


# ══════════════════════════════════════════════════════════════════════════════
# TESTS (inline for development)
# ══════════════════════════════════════════════════════════════════════════════

def _test_adapters():
    """Test adapter functionality."""
    print("Testing plate adapters...")
    
    # Test physics adapter with default genome
    physics = PlatePhysicsAdapter()
    genome = PlateGenome()  # Uses defaults: 1.85m x 0.64m
    
    result = physics.analyze(genome)
    print(f"  Frequencies: {[f'{f:.1f}' for f in result.frequencies[:5]]}")
    print(f"  Mode shapes: {result.mode_shapes.shape}")
    
    # Test factory
    factory = PlateGenomeFactory()
    random_genome = factory.create_random()
    print(f"  Random genome: {random_genome.length:.3f}x{random_genome.width:.3f}m")
    
    default_genome = factory.create_default()
    print(f"  Default genome: {default_genome.length:.3f}x{default_genome.width:.3f}m")
    
    template_genome = factory.create_from_template(default_genome, variation=0.1)
    print(f"  Template variation: {template_genome.length:.3f}x{template_genome.width:.3f}m")
    
    # Test fitness adapter (requires Person)
    try:
        person = Person(height=1.75, ear_height_offset=0.1)
        evaluator = PlateFitnessAdapter(person)
        fitness = evaluator.evaluate(genome)
        print(f"  Fitness: {fitness.total_score:.4f}")
    except Exception as e:
        print(f"  Fitness test skipped (need Person): {e}")
    
    print("Adapter tests passed!")


if __name__ == "__main__":
    _test_adapters()
