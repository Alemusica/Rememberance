"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PLATE UNIFIED - Bridge to Unified Optimization System              ║
║                                                                              ║
║   Connects the agnostic evolution framework (agnostic_evolution.py,          ║
║   plate_adapters.py) with the unified optimization system                    ║
║   (unified_optimizer.py).                                                    ║
║                                                                              ║
║   This is the HIGH-LEVEL ENTRY POINT for plate optimization.                ║
║                                                                              ║
║   Usage:                                                                     ║
║       from src.core.plate_unified import create_plate_optimization_system    ║
║       optimizer, config = create_plate_optimization_system(person)           ║
║       result = optimizer.optimize(config)                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Local imports - Unified system
from .unified_optimizer import (
    UnifiedOptimizer, UnifiedConfig, OptimizationStrategy,
    OptimizationResult, PhysicsEngineProtocol, FitnessEvaluatorProtocol,
    GenomeFactoryProtocol, StrategyRegistry
)

# Local imports - Plate-specific
from .person import Person
from .plate_genome import PlateGenome, ContourType, ExciterPosition, CutoutGene
from .plate_physics import calculate_plate_modes, mode_shape_grid
from .fitness import FitnessEvaluator, FitnessResult, ObjectiveWeights, ZoneWeights
from .analysis_config import get_target_spacing_mm, get_default_config

# Try to import adapters
try:
    from .plate_adapters import (
        PlatePhysicsAdapter, PlateFitnessAdapter, PlateGenomeFactory,
        PlatePhysicsResult
    )
    HAS_ADAPTERS = True
except ImportError:
    HAS_ADAPTERS = False
    logger.warning("plate_adapters not available, using direct implementations")


# ══════════════════════════════════════════════════════════════════════════════
# PLATE-SPECIFIC CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlateOptimizationConfig(UnifiedConfig):
    """
    Configuration for plate optimization with domain-specific parameters.
    
    Extends UnifiedConfig with plate-specific constraints and targets.
    """
    
    # Person parameters
    person_height: float = 1.75  # m
    person_weight: float = 70.0  # kg
    person_lying: bool = True
    
    # Plate dimension constraints
    min_length: float = 1.5  # m
    max_length: float = 2.4  # m
    min_width: float = 0.5   # m
    max_width: float = 1.0   # m
    min_thickness: float = 0.008  # m
    max_thickness: float = 0.020  # m
    
    # Exciter configuration
    n_exciters: int = 2
    exciter_min_diameter: float = 0.04  # m
    exciter_max_diameter: float = 0.08  # m
    
    # Material
    material: str = "birch_plywood"
    
    # Target frequencies for body resonances
    target_frequencies: List[float] = field(default_factory=lambda: [
        10.0,   # Spine resonance
        25.0,   # Chest cavity
        50.0,   # Skull vibration
        100.0,  # Ear perception
    ])
    
    # Contour options
    allowed_contours: List[ContourType] = field(default_factory=lambda: [
        ContourType.RECTANGLE,
        ContourType.GOLDEN_RECT,
        ContourType.PHI_ROUNDED,
        ContourType.ELLIPSE,
        ContourType.OVOID,
        ContourType.ERGONOMIC,
    ])
    fixed_contour: Optional[ContourType] = None
    
    # Optimization objectives
    optimize_spine: bool = True
    optimize_ears: bool = True
    optimize_head: bool = True
    ear_uniformity_weight: float = 2.0  # Higher weight for L/R balance
    
    # Cutouts and grooves
    max_cutouts: int = 4
    max_grooves: int = 2
    
    # Symmetry
    enforce_symmetry: bool = True
    
    # Decision variables count (for NSGA)
    n_decision_vars: int = 11  # length, width, thickness, 2x exciters (x,y,d) = 3 + 2*3 + ...
    
    # Objectives count
    n_objectives: int = 6  # From ObjectiveVector


# ══════════════════════════════════════════════════════════════════════════════
# PLATE PHYSICS ENGINE (implements protocol)
# ══════════════════════════════════════════════════════════════════════════════

class PlatePhysicsEngine:
    """
    Physics engine for plate modal analysis.
    
    Implements PhysicsEngineProtocol[PlateGenome, PlatePhysicsResult].
    """
    
    def __init__(self, n_modes: int = 20, freq_range: Tuple[float, float] = (10.0, 500.0)):
        self.n_modes = n_modes
        self.freq_range = freq_range
        self._cache: Dict[str, Any] = {}
    
    def analyze(self, genome: PlateGenome) -> Dict[str, Any]:
        """Run modal analysis on plate genome."""
        
        # Cache key based on geometry
        cache_key = f"{genome.length:.3f}_{genome.width:.3f}_{genome.thickness_base:.4f}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Convert to mm for calculate_plate_modes
            length_mm = genome.length * 1000
            width_mm = genome.width * 1000
            thickness_mm = genome.thickness_base * 1000
            
            # Use default material (birch plywood is good for DML)
            material_key = "birch_plywood"
            
            # Calculate modes
            modes = calculate_plate_modes(
                length_mm=length_mm,
                width_mm=width_mm,
                thickness_mm=thickness_mm,
                material_key=material_key,
                max_modes=self.n_modes,
                max_freq_hz=2000.0,
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # ADAPTIVE RESOLUTION for mode shapes
            # Uses centralized config from analysis_config.py
            # ═══════════════════════════════════════════════════════════════════
            target_spacing_mm = get_target_spacing_mm(genome.length, genome.width)
            adaptive_nx = max(21, int(np.ceil(length_mm / target_spacing_mm)))
            adaptive_ny = max(13, int(np.ceil(width_mm / target_spacing_mm)))
            # Ensure odd for symmetry
            adaptive_resolution = max(adaptive_nx, adaptive_ny)
            if adaptive_resolution % 2 == 0:
                adaptive_resolution += 1
            adaptive_resolution = min(adaptive_resolution, 101)  # Cap for performance
            
            # Generate mode shapes on adaptive grid
            shapes = []
            for mode in modes:
                X, Y, Z = mode_shape_grid(
                    m=mode.m, n=mode.n,
                    L=genome.length,
                    W=genome.width,
                    resolution=adaptive_resolution,
                )
                shapes.append(Z)
            
            result = {
                'modes': modes,
                'shapes': shapes,
                'frequencies': [m.frequency for m in modes],
                'n_modes': len(modes),
                'geometry': {
                    'length': genome.length,
                    'width': genome.width,
                    'thickness': genome.thickness_base,
                },
            }
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Physics analysis failed: {e}")
            return {
                'modes': [],
                'shapes': [],
                'frequencies': [],
                'n_modes': 0,
                'error': str(e),
            }
    
    def analyze_batch(self, genomes: List[PlateGenome]) -> List[Dict[str, Any]]:
        """Batch analysis for GPU acceleration."""
        return [self.analyze(g) for g in genomes]
    
    def clear_cache(self):
        """Clear physics cache."""
        self._cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# PLATE FITNESS EVALUATOR (implements protocol)
# ══════════════════════════════════════════════════════════════════════════════

class PlateFitnessEngine:
    """
    Fitness evaluation for plate optimization.
    
    Implements FitnessEvaluatorProtocol[PlateGenome, Dict, FitnessResult].
    """
    
    def __init__(self, person: Person, config: PlateOptimizationConfig):
        self.person = person
        self.config = config
        
        # Configure weights using actual ObjectiveWeights API
        # ObjectiveWeights has: flatness, spine_coupling, low_mass, manufacturability
        self.weights = ObjectiveWeights(
            flatness=1.0,
            spine_coupling=2.0 if config.optimize_spine else 0.5,
            low_mass=0.3,
            manufacturability=0.5,
        )
        
        # Create underlying evaluator
        self._evaluator = FitnessEvaluator(
            person=person,
            objectives=self.weights,
        )
    
    def evaluate(self, genome: PlateGenome, physics_result: Dict[str, Any]) -> FitnessResult:
        """Evaluate fitness given genome and physics result."""
        
        if physics_result.get('error') or not physics_result.get('modes'):
            # Return worst-case fitness for failed analysis
            return FitnessResult(
                total_fitness=0.0,
                flatness_score=0.0,
                spine_coupling_score=0.0,
                ear_uniformity_score=0.0,
                low_mass_score=0.0,
                manufacturability_score=0.0,
            )
        
        # FitnessEvaluator.evaluate() only takes genome, not physics_result
        # The evaluator does its own physics internally
        return self._evaluator.evaluate(genome)
    
    def get_objectives(self, result: FitnessResult) -> np.ndarray:
        """Extract 6D objective vector for multi-objective optimization."""
        return np.array([
            result.spine_flatness_score,      # Spine flatness
            result.head_flatness_score,       # Head/ear flatness (not ear_flatness)
            result.ear_uniformity_score,      # L/R balance
            result.spine_coupling_score,      # Spine energy coupling
            result.low_mass_score,            # Mass score
            result.structural_score,          # Structural safety
        ])


# ══════════════════════════════════════════════════════════════════════════════
# PLATE GENOME FACTORY (implements protocol)
# ══════════════════════════════════════════════════════════════════════════════

class PlateGenomeCreator:
    """
    Factory for creating and manipulating plate genomes.
    
    Implements GenomeFactoryProtocol[PlateGenome, PlateOptimizationConfig].
    """
    
    def random(self, config: PlateOptimizationConfig) -> PlateGenome:
        """Create random plate genome within constraints."""
        
        # Random dimensions within bounds
        length = np.random.uniform(config.min_length, config.max_length)
        width = np.random.uniform(config.min_width, config.max_width)
        thickness = np.random.uniform(config.min_thickness, config.max_thickness)
        
        # Random contour
        if config.fixed_contour:
            contour = config.fixed_contour
        else:
            contour = np.random.choice(config.allowed_contours)
        
        # Random exciters
        exciters = []
        for i in range(config.n_exciters):
            x = np.random.uniform(0.1, 0.9) * length
            y = np.random.uniform(0.1, 0.9) * width
            
            # Enforce symmetry if required
            if config.enforce_symmetry and i > 0:
                # Mirror previous exciter
                prev = exciters[-1]
                y = width - prev.y
            
            exciters.append(ExciterPosition(x=x, y=y, channel=i))
        
        # Random cutouts (30% chance of 1-2 cutouts)
        cutouts = []
        if np.random.random() < 0.3:
            n_cutouts = np.random.randint(1, 3)
            for _ in range(n_cutouts):
                # CutoutGene uses normalized coordinates [0,1]
                cx = np.random.uniform(0.2, 0.8)  # Normalized X
                cy = np.random.uniform(0.2, 0.8)  # Normalized Y
                size = np.random.uniform(0.03, 0.08)  # 3-8% of plate
                cutouts.append(CutoutGene(
                    x=cx,
                    y=cy,
                    width=size,
                    height=size * np.random.uniform(0.8, 1.2),  # Slight aspect variation
                    shape='ellipse',
                    rotation=np.random.uniform(0, np.pi) if np.random.random() < 0.3 else 0.0
                ))
        
        return PlateGenome(
            length=length,
            width=width,
            thickness_base=thickness,
            contour_type=contour,
            exciters=exciters,
            cutouts=cutouts,
        )
    
    def crossover(
        self, 
        parent1: PlateGenome, 
        parent2: PlateGenome, 
        config: PlateOptimizationConfig
    ) -> PlateGenome:
        """Create offspring from two parents using blend crossover."""
        
        alpha = np.random.uniform(0.3, 0.7)
        
        # Blend dimensions
        length = alpha * parent1.length + (1 - alpha) * parent2.length
        width = alpha * parent1.width + (1 - alpha) * parent2.width
        thickness = alpha * parent1.thickness_base + (1 - alpha) * parent2.thickness_base
        
        # Clamp to bounds
        length = np.clip(length, config.min_length, config.max_length)
        width = np.clip(width, config.min_width, config.max_width)
        thickness = np.clip(thickness, config.min_thickness, config.max_thickness)
        
        # Choose contour from one parent
        contour = parent1.contour_type if np.random.random() < 0.5 else parent2.contour_type
        
        # Blend exciter positions
        exciters = []
        n_ex = min(len(parent1.exciters), len(parent2.exciters), config.n_exciters)
        
        for i in range(n_ex):
            p1_ex = parent1.exciters[i] if i < len(parent1.exciters) else parent1.exciters[-1]
            p2_ex = parent2.exciters[i] if i < len(parent2.exciters) else parent2.exciters[-1]
            
            x = alpha * p1_ex.x + (1 - alpha) * p2_ex.x
            y = alpha * p1_ex.y + (1 - alpha) * p2_ex.y
            
            # Scale to new dimensions
            x = x * (length / parent1.length) if parent1.length > 0 else x
            y = y * (width / parent1.width) if parent1.width > 0 else y
            
            # Clamp to valid range
            x = np.clip(x, 0.05 * length, 0.95 * length)
            y = np.clip(y, 0.05 * width, 0.95 * width)
            
            exciters.append(ExciterPosition(x=x, y=y, channel=i))
        
        return PlateGenome(
            length=length,
            width=width,
            thickness_base=thickness,
            contour_type=contour,
            exciters=exciters,
        )
    
    def mutate(
        self, 
        genome: PlateGenome, 
        sigma: float, 
        config: PlateOptimizationConfig
    ) -> PlateGenome:
        """Mutate genome with given strength."""
        
        # Mutate dimensions with Gaussian noise
        length = genome.length + np.random.normal(0, sigma * (config.max_length - config.min_length))
        width = genome.width + np.random.normal(0, sigma * (config.max_width - config.min_width))
        thickness = genome.thickness_base + np.random.normal(0, sigma * (config.max_thickness - config.min_thickness))
        
        # Clamp
        length = np.clip(length, config.min_length, config.max_length)
        width = np.clip(width, config.min_width, config.max_width)
        thickness = np.clip(thickness, config.min_thickness, config.max_thickness)
        
        # Maybe mutate contour (10% chance)
        contour = genome.contour_type
        if not config.fixed_contour and np.random.random() < 0.1:
            contour = np.random.choice(config.allowed_contours)
        
        # Mutate exciters
        exciters = []
        for ex in genome.exciters:
            x = ex.x + np.random.normal(0, sigma * 0.1 * length)
            y = ex.y + np.random.normal(0, sigma * 0.1 * width)
            
            x = np.clip(x, 0.05 * length, 0.95 * length)
            y = np.clip(y, 0.05 * width, 0.95 * width)
            
            exciters.append(ExciterPosition(x=x, y=y, channel=ex.channel))
        
        # Enforce symmetry if needed
        if config.enforce_symmetry and len(exciters) >= 2:
            exciters[1] = ExciterPosition(
                x=exciters[0].x,
                y=width - exciters[0].y,
                channel=1
            )
        
        # Mutate cutouts (keep, modify, add, or remove)
        cutouts = []
        for cut in (genome.cutouts or []):
            if np.random.random() > 0.1:  # 90% keep (maybe modified)
                # Cutouts use normalized coordinates [0,1]
                cx = cut.x + np.random.normal(0, sigma * 0.05)
                cy = cut.y + np.random.normal(0, sigma * 0.05)
                cx = np.clip(cx, 0.1, 0.9)
                cy = np.clip(cy, 0.1, 0.9)
                cutouts.append(CutoutGene(
                    x=cx,
                    y=cy,
                    width=cut.width,
                    height=cut.height,
                    shape=cut.shape,
                    rotation=cut.rotation
                ))
        
        # 15% chance to add new cutout (max 3)
        if np.random.random() < 0.15 and len(cutouts) < 3:
            size = np.random.uniform(0.03, 0.08)
            cutouts.append(CutoutGene(
                x=np.random.uniform(0.2, 0.8),
                y=np.random.uniform(0.2, 0.8),
                width=size,
                height=size * np.random.uniform(0.8, 1.2),
                shape='ellipse',
                rotation=np.random.uniform(0, np.pi) if np.random.random() < 0.3 else 0.0
            ))
        
        return PlateGenome(
            length=length,
            width=width,
            thickness_base=thickness,
            contour_type=contour,
            exciters=exciters,
            cutouts=cutouts if cutouts else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION (Main Entry Point)
# ══════════════════════════════════════════════════════════════════════════════

def create_plate_optimization_system(
    person: Person,
    strategy: OptimizationStrategy = OptimizationStrategy.AUTO,
    use_memory: bool = True,
    memory_path: str = "./plate_optimization_memory",
) -> Tuple[UnifiedOptimizer, PlateOptimizationConfig]:
    """
    Create a complete plate optimization system.
    
    This is THE main entry point for plate optimization.
    
    Args:
        person: Person model (body dimensions)
        strategy: Which optimization strategy (AUTO selects best)
        use_memory: Enable evolution memory system
        memory_path: Path for memory storage
        
    Returns:
        Tuple of (UnifiedOptimizer, PlateOptimizationConfig)
        
    Example:
        >>> from src.core.person import Person
        >>> from src.core.plate_unified import create_plate_optimization_system
        >>> 
        >>> person = Person(height_m=1.80, weight_kg=75)
        >>> optimizer, config = create_plate_optimization_system(person)
        >>> 
        >>> # Add progress callback
        >>> optimizer.add_callback(lambda gen, genome, fit: print(f"Gen {gen}: {fit:.4f}"))
        >>> 
        >>> # Run optimization
        >>> result = optimizer.optimize(config)
        >>> 
        >>> print(f"Best fitness: {result.best_fitness:.4f}")
        >>> print(f"Best plate: {result.best_genome.length}m x {result.best_genome.width}m")
    """
    
    # Create config based on person
    config = PlateOptimizationConfig(
        person_height=person.height_m,
        person_weight=person.weight_kg,
        person_lying=True,
        
        # Adjust plate size based on person
        min_length=max(1.5, person.height_m - 0.2),
        max_length=min(2.4, person.height_m + 0.3),
        
        # Auto-select strategy hints
        n_objectives=6 if strategy in [OptimizationStrategy.NSGA2, OptimizationStrategy.NSGA3, OptimizationStrategy.AUTO] else 1,
        use_memory=use_memory,
        memory_path=memory_path,
    )
    
    # Create components
    physics_engine = PlatePhysicsEngine()
    fitness_engine = PlateFitnessEngine(person, config)
    genome_factory = PlateGenomeCreator()
    
    # Create unified optimizer
    optimizer = UnifiedOptimizer(
        physics_engine=physics_engine,
        fitness_evaluator=fitness_engine,
        genome_factory=genome_factory,
        memory_path=memory_path if use_memory else None,
    )
    
    return optimizer, config


def quick_optimize_plate(
    person: Person,
    n_generations: int = 50,
    population_size: int = 30,
    verbose: bool = True,
) -> OptimizationResult[PlateGenome]:
    """
    Quick plate optimization with sensible defaults.
    
    Use this for rapid prototyping and testing.
    
    Args:
        person: Person model
        n_generations: Number of generations
        population_size: Population size
        verbose: Print progress
        
    Returns:
        OptimizationResult with best plate design
    """
    
    optimizer, config = create_plate_optimization_system(
        person, 
        strategy=OptimizationStrategy.GENETIC,
        use_memory=False,
    )
    
    config.n_generations = n_generations
    config.population_size = population_size
    config.verbose = verbose
    
    if verbose:
        def progress_cb(gen, genome, fitness):
            if gen % 10 == 0:
                print(f"  Generation {gen}: fitness={fitness:.4f}")
        optimizer.add_callback(progress_cb)
    
    return optimizer.optimize(config)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'PlateOptimizationConfig',
    'PlatePhysicsEngine',
    'PlateFitnessEngine',
    'PlateGenomeCreator',
    'create_plate_optimization_system',
    'quick_optimize_plate',
]
