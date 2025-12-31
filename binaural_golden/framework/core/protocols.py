"""
══════════════════════════════════════════════════════════════════════════════
AGNOSTIC CORE - Protocol Definitions
══════════════════════════════════════════════════════════════════════════════

This module defines the abstract protocols/interfaces that domain adapters
must implement. These are the core contracts that make the framework
domain-agnostic.

Based on research:
- Python Protocol classes (PEP 544) for structural subtyping
- ABC for inheritance-based contracts
- Inspired by EvoTorch Problem interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, TypeVar, Generic, Protocol, runtime_checkable
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

G = TypeVar('G')  # Genome type
R = TypeVar('R')  # Result type
P = TypeVar('P')  # Physics result type


# ══════════════════════════════════════════════════════════════════════════════
# GENOME PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class GenomeProtocol(Protocol):
    """
    Protocol defining the interface for domain genomes.
    
    A genome represents a candidate solution in the search space.
    It must support:
    - Mutation (variation operator)
    - Crossover (recombination operator)
    - Serialization (for logging/checkpointing)
    """
    
    fitness: float
    
    def mutate(self, sigma: float = 0.1) -> 'GenomeProtocol':
        """
        Return mutated copy of genome.
        
        Args:
            sigma: Mutation strength (interpretation is domain-specific)
            
        Returns:
            New genome instance with mutations applied
        """
        ...
    
    def crossover(self, other: 'GenomeProtocol') -> 'GenomeProtocol':
        """
        Return offspring from crossover with another genome.
        
        Args:
            other: Other parent genome
            
        Returns:
            New genome instance from recombination
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenomeProtocol':
        """Deserialize genome from dictionary."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE RESULT PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ObjectiveResultProtocol(Protocol):
    """
    Protocol for multi-objective fitness results.
    
    Used by NSGA-II for Pareto ranking and crowding distance.
    """
    
    def to_minimize_array(self) -> np.ndarray:
        """
        Convert objectives to array for NSGA-II minimization.
        
        Note: For maximization objectives, return negative values.
        
        Returns:
            1D numpy array of objective values (to minimize)
        """
        ...
    
    def to_labeled_dict(self) -> Dict[str, float]:
        """
        Get objectives as labeled dictionary for logging.
        
        Returns:
            Dict mapping objective names to values
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE ABC
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsEngineABC(ABC, Generic[G, P]):
    """
    Abstract base class for domain physics/analysis engines.
    
    The physics engine performs the computationally expensive analysis
    of a genome to produce physical quantities needed for fitness evaluation.
    
    Examples:
    - DML plate: FEM modal analysis, frequency response
    - Singing bowl: Shell vibration modes
    - Speaker box: Acoustic FEM, port tuning
    """
    
    @abstractmethod
    def analyze(self, genome: G) -> P:
        """
        Perform physics analysis on genome.
        
        Args:
            genome: Domain genome to analyze
            
        Returns:
            Physics result (domain-specific structure)
        """
        pass
    
    def batch_analyze(self, genomes: List[G]) -> List[P]:
        """
        Analyze multiple genomes (can be overridden for vectorization).
        
        Default implementation calls analyze() sequentially.
        Override for GPU/vectorized implementations.
        
        Args:
            genomes: List of genomes to analyze
            
        Returns:
            List of physics results in same order
        """
        return [self.analyze(g) for g in genomes]


# ══════════════════════════════════════════════════════════════════════════════
# FITNESS EVALUATOR ABC
# ══════════════════════════════════════════════════════════════════════════════

class FitnessEvaluatorABC(ABC, Generic[G, P, R]):
    """
    Abstract base class for fitness evaluators.
    
    The evaluator takes a genome and physics results to compute
    multi-objective fitness scores.
    
    Curriculum learning support:
    - Evaluator can adjust behavior based on current phase
    - Phase determines which objectives are active/weighted
    """
    
    @abstractmethod
    def evaluate(self, genome: G, physics_result: P) -> R:
        """
        Evaluate genome fitness.
        
        Args:
            genome: Domain genome
            physics_result: Results from physics engine
            
        Returns:
            Fitness result implementing ObjectiveResultProtocol
        """
        pass
    
    def set_phase(self, phase: str) -> None:
        """
        Set curriculum learning phase.
        
        Override to implement phase-dependent evaluation.
        
        Args:
            phase: Phase identifier (e.g., 'SEED', 'BLOOM', 'FREEZE')
        """
        pass
    
    def get_objective_names(self) -> List[str]:
        """
        Get list of objective names for logging.
        
        Returns:
            List of objective name strings
        """
        return []


# ══════════════════════════════════════════════════════════════════════════════
# GENOME FACTORY ABC
# ══════════════════════════════════════════════════════════════════════════════

class GenomeFactoryABC(ABC, Generic[G]):
    """
    Abstract factory for creating domain genomes.
    
    Handles:
    - Random initialization
    - Seeded initialization (from known good designs)
    - Constraint validation
    """
    
    @abstractmethod
    def create_random(self) -> G:
        """Create random genome within valid bounds."""
        pass
    
    def create_seeded(self, seed_data: Dict[str, Any]) -> G:
        """
        Create genome from seed data.
        
        Default: falls back to create_random()
        """
        return self.create_random()
    
    def is_valid(self, genome: G) -> bool:
        """
        Check if genome satisfies constraints.
        
        Default: always True
        """
        return True
    
    def repair(self, genome: G) -> G:
        """
        Repair invalid genome to satisfy constraints.
        
        Default: return genome unchanged
        """
        return genome


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN ADAPTER ABC
# ══════════════════════════════════════════════════════════════════════════════

class DomainAdapterABC(ABC, Generic[G, P, R]):
    """
    Abstract base class for domain adapters.
    
    A domain adapter ties together:
    - Genome factory
    - Physics engine
    - Fitness evaluator
    
    This is the main interface the evolutionary optimizer uses.
    """
    
    @abstractmethod
    def get_factory(self) -> GenomeFactoryABC[G]:
        """Get genome factory."""
        pass
    
    @abstractmethod
    def get_physics_engine(self) -> PhysicsEngineABC[G, P]:
        """Get physics/analysis engine."""
        pass
    
    @abstractmethod
    def get_evaluator(self) -> FitnessEvaluatorABC[G, P, R]:
        """Get fitness evaluator."""
        pass
    
    def create_genome(self, **kwargs) -> G:
        """Convenience: create genome via factory."""
        return self.get_factory().create_random()
    
    def analyze(self, genome: G) -> P:
        """Convenience: analyze via physics engine."""
        return self.get_physics_engine().analyze(genome)
    
    def evaluate(self, genome: G, physics_result: P = None) -> R:
        """Convenience: evaluate genome."""
        if physics_result is None:
            physics_result = self.analyze(genome)
        return self.get_evaluator().evaluate(genome, physics_result)


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY PROTOCOLS
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for evolutionary memory systems."""
    
    def store(self, genome: GenomeProtocol, fitness_result: ObjectiveResultProtocol) -> None:
        """Store genome and fitness in memory."""
        ...
    
    def recall(self, n: int = 10) -> List[tuple]:
        """Recall top N entries from memory."""
        ...


@runtime_checkable  
class DistillerProtocol(Protocol):
    """Protocol for long-term memory distillers."""
    
    def distill(self, memory: MemoryProtocol) -> Dict[str, Any]:
        """Distill patterns from memory."""
        ...
    
    def get_insights(self) -> List[str]:
        """Get human-readable insights."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVER PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ObserverProtocol(Protocol):
    """
    Protocol for evolution observers (logging, UI, etc.)
    
    Implements Observer pattern for loose coupling.
    """
    
    def on_generation_start(self, generation: int) -> None:
        """Called at start of generation."""
        ...
    
    def on_generation_end(self, generation: int, stats: Dict[str, Any]) -> None:
        """Called at end of generation with statistics."""
        ...
    
    def on_new_best(self, genome: GenomeProtocol, fitness: float) -> None:
        """Called when new best individual found."""
        ...
    
    def on_evolution_end(self, results: Dict[str, Any]) -> None:
        """Called when evolution completes."""
        ...
