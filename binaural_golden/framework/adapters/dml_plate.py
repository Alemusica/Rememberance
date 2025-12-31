"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DOMAIN ADAPTER - DML PLATE                                ║
║                                                                              ║
║   Adapter connecting the agnostic evolution framework to DML plate domain.  ║
║   This is the reference implementation for creating new domain adapters.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Dict, List, Optional, Any
import numpy as np

# Import from agnostic framework
from ...core.agnostic_evolution import (
    PhysicsEngine,
    FitnessEvaluator, 
    GenomeFactory,
    ObjectiveResult,
)

# Import domain-specific implementations
from src.core.plate_genome import PlateGenome
from src.core.plate_physics import PlatePhysics as PlatePhysicsImpl
from src.core.fitness import FitnessEvaluator as FitnessEvaluatorImpl
from src.core.person import Person


class DMLPlateAdapter:
    """
    Adapter class that bridges the agnostic framework to DML plate domain.
    
    This class provides:
    1. GenomeFactory for creating PlateGenome instances
    2. PhysicsEngine wrapping PlatePhysics
    3. FitnessEvaluator wrapping the existing fitness module
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with domain configuration.
        
        Args:
            config: Domain configuration from domain_config.yaml
        """
        self.config = config
        
        # Initialize person model for zone definitions
        self.person = Person(
            height=self.config.get('person_height', 1.75),
            ear_distance=self.config.get('ear_distance', 0.15),
        )
        
        # Initialize physics engine
        self.physics = PlatePhysicsImpl(
            plate_length=config.get('length', 1.85),
            plate_width=config.get('width', 0.64),
        )
        
        # Initialize fitness evaluator
        self.evaluator = FitnessEvaluatorImpl(
            person=self.person,
            weights=self._get_weights_from_config(),
        )
    
    def _get_weights_from_config(self) -> Dict[str, float]:
        """Extract fitness weights from configuration."""
        objectives = self.config.get('objectives', {})
        weights = {}
        
        for obj in objectives.get('maximize', []):
            weights[obj['name']] = obj.get('weight', 1.0)
        for obj in objectives.get('minimize', []):
            weights[obj['name']] = obj.get('weight', 1.0)
        
        return weights
    
    def create_genome(self, **kwargs) -> PlateGenome:
        """
        Create a new PlateGenome instance.
        
        This is the GenomeFactory interface implementation.
        """
        return PlateGenome(
            length=self.config.get('length', 1.85),
            width=self.config.get('width', 0.64),
            thickness_base=self.config.get('thickness', 0.015),
            **kwargs
        )
    
    def analyze(self, genome: PlateGenome) -> Dict[str, Any]:
        """
        Perform physics analysis on a genome.
        
        This is the PhysicsEngine interface implementation.
        
        Returns:
            Dict containing modal_frequencies, mode_shapes, etc.
        """
        return self.physics.analyze(genome)
    
    def evaluate(self, genome: PlateGenome, physics_result: Optional[Dict] = None) -> 'DMLFitnessResult':
        """
        Evaluate fitness of a genome.
        
        This is the FitnessEvaluator interface implementation.
        """
        if physics_result is None:
            physics_result = self.analyze(genome)
        
        result = self.evaluator.evaluate(genome, physics_result)
        
        return DMLFitnessResult(
            ear_uniformity=result.ear_uniformity,
            spine_coupling=result.spine_coupling,
            response_flatness=result.response_flatness,
            abh_benefit=result.get('abh_benefit', 0.0),
            scalar_fitness=result.total_fitness,
        )


class DMLFitnessResult:
    """
    Fitness result container for DML plate domain.
    
    Implements ObjectiveResult protocol for multi-objective optimization.
    """
    
    def __init__(
        self,
        ear_uniformity: float,
        spine_coupling: float,
        response_flatness: float,
        abh_benefit: float = 0.0,
        scalar_fitness: float = 0.0,
    ):
        self.ear_uniformity = ear_uniformity
        self.spine_coupling = spine_coupling
        self.response_flatness = response_flatness
        self.abh_benefit = abh_benefit
        self.scalar_fitness = scalar_fitness
    
    def to_minimize_array(self) -> np.ndarray:
        """
        Convert to minimization array for NSGA-II.
        
        Note: We NEGATE values that should be maximized.
        """
        return np.array([
            -self.ear_uniformity,      # Maximize → negate
            -self.spine_coupling,      # Maximize → negate
            self.response_flatness,    # Minimize (lower variation is better)
            -self.abh_benefit,         # Maximize → negate
        ])
    
    def to_labeled_dict(self) -> Dict[str, float]:
        """Return labeled dict of objective values."""
        return {
            'ear_uniformity': self.ear_uniformity,
            'spine_coupling': self.spine_coupling,
            'response_flatness': self.response_flatness,
            'abh_benefit': self.abh_benefit,
            'scalar_fitness': self.scalar_fitness,
        }


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS for framework integration
# ══════════════════════════════════════════════════════════════════════════════

def create_adapter(config: Dict[str, Any]) -> DMLPlateAdapter:
    """Factory function to create adapter from config."""
    return DMLPlateAdapter(config)


def get_genome_class():
    """Return the genome class for this domain."""
    return PlateGenome


def get_objective_names() -> List[str]:
    """Return names of objectives for this domain."""
    return ['ear_uniformity', 'spine_coupling', 'response_flatness', 'abh_benefit']
