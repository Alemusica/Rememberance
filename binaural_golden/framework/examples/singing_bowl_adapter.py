"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SINGING BOWL ADAPTER - Template Example                         ║
║                                                                              ║
║   Template adapter for Tibetan singing bowl optimization.                   ║
║   This shows how to create a new domain adapter.                            ║
║                                                                              ║
║   To implement:                                                              ║
║   1. Define BowlGenome (geometry, wall thickness, alloy)                    ║
║   2. Implement BowlPhysics (3D FEM or analytical shell model)               ║
║   3. Define objectives (frequency ratios, sustain, timbre)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# GENOME - Bowl Design Representation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BowlGenome:
    """
    Genetic representation of a singing bowl.
    
    Parameters are based on traditional bowl making:
    - Diameter and height define basic size
    - Wall profile controls mode frequencies
    - Alloy composition affects damping and timbre
    """
    # Geometry
    diameter: float = 0.15          # Rim diameter [m]
    height: float = 0.08            # Bowl height [m]
    rim_thickness: float = 0.003    # Rim wall thickness [m]
    base_thickness: float = 0.005   # Base thickness [m]
    
    # Wall profile (controls mode frequencies)
    # Profile defined as (height_fraction, thickness_multiplier) pairs
    wall_profile: List[tuple] = field(default_factory=lambda: [
        (0.0, 1.2),   # Base - slightly thicker
        (0.3, 1.0),   # Lower wall
        (0.7, 0.9),   # Upper wall - slightly thinner
        (1.0, 1.0),   # Rim
    ])
    
    # Material (bronze alloy)
    tin_percentage: float = 0.20    # Tin content (20% = traditional)
    
    # Fitness (calculated by evaluator)
    fitness: float = 0.0
    
    def mutate(self, sigma: float = 0.05) -> 'BowlGenome':
        """Return mutated copy."""
        import copy
        mutated = copy.deepcopy(self)
        
        # Mutate dimensions
        mutated.diameter += np.random.normal(0, sigma * 0.02)
        mutated.height += np.random.normal(0, sigma * 0.02)
        mutated.rim_thickness += np.random.normal(0, sigma * 0.001)
        mutated.base_thickness += np.random.normal(0, sigma * 0.001)
        
        # Clamp to valid ranges
        mutated.diameter = np.clip(mutated.diameter, 0.08, 0.40)
        mutated.height = np.clip(mutated.height, 0.04, 0.25)
        mutated.rim_thickness = np.clip(mutated.rim_thickness, 0.001, 0.008)
        mutated.base_thickness = np.clip(mutated.base_thickness, 0.002, 0.010)
        
        return mutated
    
    def crossover(self, other: 'BowlGenome') -> 'BowlGenome':
        """Return offspring from crossover."""
        import copy
        child = copy.deepcopy(self)
        
        # Uniform crossover
        if np.random.random() < 0.5:
            child.diameter = other.diameter
        if np.random.random() < 0.5:
            child.height = other.height
        if np.random.random() < 0.5:
            child.rim_thickness = other.rim_thickness
        if np.random.random() < 0.5:
            child.base_thickness = other.base_thickness
        
        return child
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'diameter': self.diameter,
            'height': self.height,
            'rim_thickness': self.rim_thickness,
            'base_thickness': self.base_thickness,
            'wall_profile': self.wall_profile,
            'tin_percentage': self.tin_percentage,
            'fitness': self.fitness,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BowlGenome':
        """Deserialize from dictionary."""
        return cls(**data)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS - Bowl Modal Analysis (Placeholder)
# ══════════════════════════════════════════════════════════════════════════════

class BowlPhysics:
    """
    Physics engine for singing bowl modal analysis.
    
    TODO: Implement one of:
    - Analytical shell theory (Rossing, Fletcher)
    - 3D FEM with shell elements
    - Rayleigh-Ritz approximation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Material properties (bronze)
        self.density = 8900  # kg/m³
        self.youngs_modulus = 110e9  # Pa
        self.poisson_ratio = 0.34
    
    def analyze(self, genome: BowlGenome) -> Dict[str, Any]:
        """
        Perform modal analysis on bowl geometry.
        
        Returns:
            Dict with modal_frequencies, mode_shapes, damping, etc.
        """
        # PLACEHOLDER: Return estimated frequencies based on scaling laws
        # Real implementation would use FEM or analytical model
        
        # Approximate fundamental frequency (empirical)
        # f ∝ t / D² (thickness / diameter squared)
        t_avg = (genome.rim_thickness + genome.base_thickness) / 2
        f_fundamental = 1000 * t_avg / (genome.diameter ** 1.8)
        
        # Approximate overtones (typical bowl ratios)
        # Traditional bowls have specific harmonic structure
        overtone_ratios = [2.0, 2.9, 3.8, 4.7]  # Approximate
        
        modal_frequencies = [f_fundamental] + [
            f_fundamental * r for r in overtone_ratios
        ]
        
        return {
            'modal_frequencies': modal_frequencies,
            'fundamental': f_fundamental,
            'harmonic_ratios': [f / f_fundamental for f in modal_frequencies],
            'quality_factor': 2000,  # Typical for bronze bowl
        }


# ══════════════════════════════════════════════════════════════════════════════
# FITNESS EVALUATOR - Bowl Quality Assessment
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BowlFitnessResult:
    """Fitness result for singing bowl."""
    harmonic_accuracy: float      # How close ratios are to ideal
    fundamental_match: float      # How close fundamental is to target
    sustain_estimate: float       # Estimated sustain quality
    manufacturability: float      # Can it be made with traditional methods?
    
    def to_minimize_array(self) -> np.ndarray:
        """Convert to NSGA-II minimization array."""
        return np.array([
            -self.harmonic_accuracy,    # Maximize
            -self.fundamental_match,    # Maximize
            -self.sustain_estimate,     # Maximize
            -self.manufacturability,    # Maximize
        ])
    
    def to_labeled_dict(self) -> Dict[str, float]:
        return {
            'harmonic_accuracy': self.harmonic_accuracy,
            'fundamental_match': self.fundamental_match,
            'sustain_estimate': self.sustain_estimate,
            'manufacturability': self.manufacturability,
        }


class BowlEvaluator:
    """Fitness evaluator for singing bowls."""
    
    def __init__(
        self,
        target_fundamental: float = 200.0,
        target_ratios: List[float] = None,
    ):
        self.target_fundamental = target_fundamental
        self.target_ratios = target_ratios or [2.0, 3.0, 4.0]
    
    def evaluate(
        self, 
        genome: BowlGenome, 
        physics_result: Dict[str, Any]
    ) -> BowlFitnessResult:
        """Evaluate bowl fitness."""
        
        freqs = physics_result['modal_frequencies']
        fundamental = physics_result['fundamental']
        ratios = physics_result['harmonic_ratios']
        
        # 1. Harmonic accuracy (how close to ideal ratios)
        ratio_errors = [
            abs(r - t) for r, t in zip(ratios[1:], self.target_ratios)
        ]
        harmonic_accuracy = 1.0 - np.mean(ratio_errors) / 0.5
        harmonic_accuracy = np.clip(harmonic_accuracy, 0, 1)
        
        # 2. Fundamental match
        fundamental_error = abs(fundamental - self.target_fundamental)
        fundamental_match = np.exp(-fundamental_error / 50)
        
        # 3. Sustain estimate (based on Q factor and geometry)
        # Thinner walls = more sustain but less volume
        sustain_estimate = physics_result.get('quality_factor', 1000) / 3000
        sustain_estimate = np.clip(sustain_estimate, 0, 1)
        
        # 4. Manufacturability (thickness ratios, size constraints)
        thickness_ratio = genome.rim_thickness / genome.base_thickness
        # Ideal ratio is around 0.6-0.8
        manuf_score = 1.0 - abs(thickness_ratio - 0.7) / 0.3
        manufacturability = np.clip(manuf_score, 0, 1)
        
        return BowlFitnessResult(
            harmonic_accuracy=harmonic_accuracy,
            fundamental_match=fundamental_match,
            sustain_estimate=sustain_estimate,
            manufacturability=manufacturability,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER - Framework Integration
# ══════════════════════════════════════════════════════════════════════════════

class SingingBowlAdapter:
    """Adapter for singing bowl domain."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.physics = BowlPhysics(config)
        self.evaluator = BowlEvaluator(
            target_fundamental=config.get('fundamental', 200),
            target_ratios=config.get('harmonic_ratios', [2.0, 3.0, 4.0]),
        )
    
    def create_genome(self, **kwargs) -> BowlGenome:
        return BowlGenome(
            diameter=self.config.get('diameter', 0.15),
            height=self.config.get('height', 0.08),
            **kwargs
        )
    
    def analyze(self, genome: BowlGenome) -> Dict[str, Any]:
        return self.physics.analyze(genome)
    
    def evaluate(self, genome: BowlGenome, physics_result: Dict = None) -> BowlFitnessResult:
        if physics_result is None:
            physics_result = self.analyze(genome)
        return self.evaluator.evaluate(genome, physics_result)


def create_adapter(config: Dict[str, Any]) -> SingingBowlAdapter:
    """Factory function."""
    return SingingBowlAdapter(config)


def get_genome_class():
    return BowlGenome


def get_objective_names() -> List[str]:
    return ['harmonic_accuracy', 'fundamental_match', 'sustain_estimate', 'manufacturability']
