"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AGNOSTIC EVOLUTION FRAMEWORK - Goal-Independent Optimization         ║
║                                                                              ║
║   Abstract base classes for evolutionary optimization that can be applied    ║
║   to ANY vibroacoustic design problem:                                       ║
║   • DML plates for therapy beds                                              ║
║   • Tibetan singing bowls                                                    ║
║   • Wine glasses / bottles                                                   ║
║   • Percussion instruments                                                   ║
║   • Loudspeaker enclosures                                                   ║
║   • Any vibrating structure!                                                 ║
║                                                                              ║
║   DESIGN PRINCIPLES:                                                         ║
║   1. SEPARATION: Physics, Evaluation, Evolution are decoupled               ║
║   2. INTERFACE: Abstract methods define the contract                        ║
║   3. PLUGGABLE: Swap implementations without changing optimizer             ║
║   4. MEMORY: Short-term and long-term learning are optional plugins         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
from typing import (
    List, Dict, Any, Optional, Tuple, TypeVar, Generic,
    Callable, Iterator, Protocol, runtime_checkable
)
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES for Generic Framework
# ══════════════════════════════════════════════════════════════════════════════

G = TypeVar('G')           # Genome type (PlateGenome, BowlGenome, etc.)
F = TypeVar('F')           # Fitness result type
P = TypeVar('P')           # Physics result type (modal analysis, etc.)
C = TypeVar('C')           # Configuration type


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS - Define interfaces for pluggable components
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Genome(Protocol):
    """
    Protocol for any genome type.
    
    A genome must be able to:
    - Mutate itself
    - Crossover with another genome
    - Report its fitness-relevant parameters
    """
    
    def mutate(self, sigma: float) -> 'Genome':
        """Return a mutated copy of this genome."""
        ...
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """Return offspring from crossover with another genome."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Deserialize genome from dictionary."""
        ...


@runtime_checkable
class ObjectiveResult(Protocol):
    """
    Protocol for multi-objective fitness results.
    
    Must provide a way to convert to minimization array for NSGA-II.
    """
    
    def to_minimize_array(self) -> np.ndarray:
        """Return array of objectives to MINIMIZE (negate if maximizing)."""
        ...
    
    def to_labeled_dict(self) -> Dict[str, float]:
        """Return labeled dict of objective values."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsEngine(ABC, Generic[G, P]):
    """
    Abstract base class for physics simulation.
    
    Different design problems have different physics:
    - DML plates: 2D modal analysis, FEM
    - Bowls: 3D modal analysis, shell elements
    - Strings: 1D wave equation
    
    Implement this class for your specific domain.
    """
    
    @abstractmethod
    def analyze(self, genome: G) -> P:
        """
        Perform physics analysis on a genome.
        
        Args:
            genome: The design to analyze
        
        Returns:
            Physics result (modal frequencies, mode shapes, etc.)
        """
        pass
    
    @abstractmethod
    def get_mode_shapes(self, physics_result: P) -> np.ndarray:
        """Extract mode shapes from physics result."""
        pass
    
    @abstractmethod
    def get_frequencies(self, physics_result: P) -> List[float]:
        """Extract modal frequencies from physics result."""
        pass
    
    def get_sensitivity(self, genome: G, physics_result: P) -> np.ndarray:
        """
        Compute sensitivity field for topology optimization.
        
        Default implementation returns mode shape magnitude.
        Override for more sophisticated sensitivity analysis.
        """
        mode_shapes = self.get_mode_shapes(physics_result)
        if mode_shapes is None or len(mode_shapes) == 0:
            return np.ones((10, 10))  # Fallback
        
        # Sum absolute mode shapes (where cutting has most effect)
        return np.sum(np.abs(mode_shapes), axis=0)


class FitnessEvaluator(ABC, Generic[G, P, F]):
    """
    Abstract base class for fitness evaluation.
    
    Different goals have different fitness functions:
    - Therapy bed: Spine coupling, ear uniformity
    - Singing bowl: Specific frequency ratios
    - Speaker: Flat frequency response
    
    Implement this class for your specific goals.
    """
    
    @abstractmethod
    def evaluate(self, genome: G, physics_result: P = None) -> F:
        """
        Evaluate fitness of a genome.
        
        Args:
            genome: The design to evaluate
            physics_result: Optional cached physics result
        
        Returns:
            Fitness result with objective scores
        """
        pass
    
    @abstractmethod
    def get_objectives(self, fitness_result: F) -> np.ndarray:
        """
        Extract objective array for multi-objective optimization.
        
        Returns array suitable for NSGA-II (to be MINIMIZED).
        """
        pass
    
    @abstractmethod
    def get_scalar_fitness(self, fitness_result: F) -> float:
        """
        Extract scalar fitness for single-objective optimization.
        
        Returns single value (higher = better).
        """
        pass
    
    def get_objective_names(self) -> List[str]:
        """Return names of objectives (for logging/UI)."""
        return ["objective_1", "objective_2"]


class GenomeFactory(ABC, Generic[G, C]):
    """
    Abstract factory for creating genomes.
    
    Different problems have different genome structures:
    - Plates: Length, width, cutouts, exciters
    - Bowls: Diameter, wall thickness, material
    - Strings: Length, tension, mass
    """
    
    @abstractmethod
    def create_random(self, config: C = None) -> G:
        """Create a random genome within constraints."""
        pass
    
    @abstractmethod
    def create_from_template(self, template: G, variation: float = 0.1) -> G:
        """Create a variation of an existing genome."""
        pass
    
    @abstractmethod
    def create_default(self) -> G:
        """Create a default/baseline genome."""
        pass


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC EVOLUTIONARY OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvolutionConfigBase:
    """
    Base configuration for evolutionary optimization.
    
    Domain-specific configs can extend this with additional parameters.
    """
    # Population
    population_size: int = 50
    n_generations: int = 100
    elite_count: int = 3
    
    # Selection
    tournament_size: int = 4
    
    # Operators
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.05
    
    # Adaptive mutation
    adaptive_mutation: bool = True
    mutation_decay: float = 0.97
    min_mutation_sigma: float = 0.008
    
    # Stopping
    convergence_threshold: float = 0.00005
    patience: int = 30
    min_generations: int = 50


@dataclass
class EvolutionResult(Generic[G]):
    """Result of an evolutionary optimization run."""
    best_genome: G
    best_fitness: float
    best_objectives: Dict[str, float]
    
    # History
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    
    # Stats
    total_generations: int = 0
    convergence_generation: int = 0
    total_evaluations: int = 0
    elapsed_time: float = 0.0
    
    # Final population (for Pareto front)
    final_population: List[G] = field(default_factory=list)
    pareto_front: List[Tuple[G, Dict[str, float]]] = field(default_factory=list)


class AgnosticEvolutionaryOptimizer(Generic[G, P, F, C]):
    """
    Goal-agnostic evolutionary optimizer.
    
    This optimizer works with ANY:
    - Genome type (plates, bowls, strings, etc.)
    - Physics engine (FEM, analytical, surrogate)
    - Fitness evaluator (therapy goals, music goals, etc.)
    
    USAGE:
        # Define your domain
        physics = MyPlatePhysics()
        evaluator = MyTherapyFitness(target_person)
        factory = MyPlateFactory(constraints)
        
        # Create optimizer
        optimizer = AgnosticEvolutionaryOptimizer(
            physics=physics,
            evaluator=evaluator,
            factory=factory,
            config=MyConfig(),
        )
        
        # Optional: Add memory
        from evolution_memory import EvolutionMemory
        optimizer.set_memory(EvolutionMemory("./memory"))
        
        # Run optimization
        result = optimizer.optimize(callbacks=[my_callback])
        
        print(f"Best fitness: {result.best_fitness}")
        print(f"Best genome: {result.best_genome}")
    """
    
    def __init__(
        self,
        physics: PhysicsEngine[G, P],
        evaluator: FitnessEvaluator[G, P, F],
        factory: GenomeFactory[G, C],
        config: EvolutionConfigBase = None,
    ):
        """
        Initialize the optimizer.
        
        Args:
            physics: Physics simulation engine
            evaluator: Fitness evaluator
            factory: Genome factory
            config: Evolution configuration
        """
        self.physics = physics
        self.evaluator = evaluator
        self.factory = factory
        self.config = config or EvolutionConfigBase()
        
        # State
        self.population: List[G] = []
        self.fitnesses: List[F] = []
        self.physics_cache: Dict[str, P] = {}  # Cache physics results
        
        # Memory (optional)
        self.memory = None
        
        # Current generation
        self.generation = 0
        self.current_sigma = self.config.mutation_sigma
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        logger.info(f"AgnosticEvolutionaryOptimizer initialized with "
                   f"pop_size={self.config.population_size}")
    
    def set_memory(self, memory):
        """
        Attach memory system for learning.
        
        Args:
            memory: EvolutionMemory instance (or compatible)
        """
        self.memory = memory
        logger.info("Memory system attached to optimizer")
    
    def optimize(
        self,
        initial_population: List[G] = None,
        callbacks: List[Callable] = None,
    ) -> EvolutionResult[G]:
        """
        Run evolutionary optimization.
        
        Args:
            initial_population: Optional starting population
            callbacks: List of callback functions called each generation
                      Signature: callback(generation, population, fitnesses, best)
        
        Returns:
            EvolutionResult with best solution and history
        """
        import time
        start_time = time.time()
        
        if callbacks:
            self.callbacks = callbacks
        
        # Initialize population
        self._initialize_population(initial_population)
        
        # Evaluate initial population
        self._evaluate_population()
        
        # Track history
        fitness_history = []
        diversity_history = []
        best_fitness = float('-inf')
        best_genome = None
        best_objectives = {}
        convergence_gen = 0
        generations_without_improvement = 0
        
        # Main evolution loop
        for gen in range(self.config.n_generations):
            self.generation = gen
            
            # Get current best
            idx_best = np.argmax([self._get_scalar(f) for f in self.fitnesses])
            current_best = self._get_scalar(self.fitnesses[idx_best])
            current_best_genome = self.population[idx_best]
            
            # Update best ever
            if current_best > best_fitness:
                best_fitness = current_best
                best_genome = current_best_genome
                best_objectives = self._get_objective_dict(self.fitnesses[idx_best])
                convergence_gen = gen
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Track history
            fitness_history.append(current_best)
            diversity_history.append(self._compute_diversity())
            
            # Memory recording
            if self.memory:
                obj_vectors = [
                    self._get_objective_dict(f) for f in self.fitnesses
                ]
                self.memory.record_generation(
                    generation=gen,
                    population_fitnesses=np.array([self._get_scalar(f) for f in self.fitnesses]),
                    objective_vectors=obj_vectors,
                    best_genome=current_best_genome,
                )
            
            # Callbacks
            for callback in self.callbacks:
                callback(gen, self.population, self.fitnesses, current_best_genome)
            
            # Early stopping check
            if gen >= self.config.min_generations:
                if generations_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping at generation {gen} (no improvement)")
                    break
            
            # Create next generation
            self._evolve_generation()
            
            # Adaptive mutation
            if self.config.adaptive_mutation:
                self._adapt_mutation()
            
            # Log progress
            if gen % 10 == 0:
                logger.info(f"Gen {gen}: best={current_best:.4f}, "
                           f"div={diversity_history[-1]:.3f}, sigma={self.current_sigma:.4f}")
        
        elapsed = time.time() - start_time
        
        # Build result
        result = EvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            best_objectives=best_objectives,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            total_generations=self.generation + 1,
            convergence_generation=convergence_gen,
            total_evaluations=(self.generation + 1) * self.config.population_size,
            elapsed_time=elapsed,
            final_population=list(self.population),
        )
        
        # Finalize memory
        if self.memory:
            self.memory.finalize_run(
                final_fitness=best_fitness,
                final_objectives=best_objectives,
                best_genome_summary=best_genome.to_dict() if hasattr(best_genome, 'to_dict') else {},
                outcome="success" if best_fitness > 0.5 else "partial",
            )
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _initialize_population(self, initial: List[G] = None):
        """Initialize or seed the population."""
        if initial:
            self.population = initial[:self.config.population_size]
        else:
            self.population = []
        
        # Fill remaining with random genomes
        while len(self.population) < self.config.population_size:
            genome = self.factory.create_random()
            self.population.append(genome)
        
        # Get suggestions from memory if available
        if self.memory:
            suggestions = self.memory.get_pattern_suggestions()
            # Could warm-start some individuals based on suggestions
            # (Implementation depends on domain)
    
    def _evaluate_population(self):
        """Evaluate fitness of all individuals."""
        self.fitnesses = []
        
        for genome in self.population:
            # Check cache
            genome_key = self._genome_key(genome)
            
            if genome_key in self.physics_cache:
                physics_result = self.physics_cache[genome_key]
            else:
                physics_result = self.physics.analyze(genome)
                self.physics_cache[genome_key] = physics_result
            
            fitness = self.evaluator.evaluate(genome, physics_result)
            self.fitnesses.append(fitness)
        
        # Limit cache size
        if len(self.physics_cache) > 1000:
            # Keep recent entries (simple LRU approximation)
            keys = list(self.physics_cache.keys())
            for key in keys[:500]:
                del self.physics_cache[key]
    
    def _evolve_generation(self):
        """Create next generation through selection, crossover, mutation."""
        scalar_fitnesses = np.array([self._get_scalar(f) for f in self.fitnesses])
        
        # Elitism: Keep best individuals
        elite_indices = np.argsort(scalar_fitnesses)[-self.config.elite_count:]
        new_population = [self.population[i] for i in elite_indices]
        
        # Fill rest through tournament selection + crossover + mutation
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = self._tournament_select(scalar_fitnesses)
            parent2 = self._tournament_select(scalar_fitnesses)
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child = child.mutate(self.current_sigma)
            
            new_population.append(child)
        
        self.population = new_population
        
        # Re-evaluate
        self._evaluate_population()
    
    def _tournament_select(self, fitnesses: np.ndarray) -> G:
        """Select individual via tournament selection."""
        indices = np.random.choice(
            len(self.population),
            size=min(self.config.tournament_size, len(self.population)),
            replace=False
        )
        winner_idx = indices[np.argmax(fitnesses[indices])]
        return self.population[winner_idx]
    
    def _adapt_mutation(self):
        """Adapt mutation rate based on progress."""
        self.current_sigma = max(
            self.current_sigma * self.config.mutation_decay,
            self.config.min_mutation_sigma
        )
        
        # Boost if stagnating (if memory available)
        if self.memory:
            analysis = self.memory.get_trajectory_analysis()
            if hasattr(analysis, 'is_stagnating') and analysis.is_stagnating:
                self.current_sigma = min(
                    self.current_sigma * 1.5,
                    self.config.mutation_sigma
                )
    
    def _compute_diversity(self) -> float:
        """Compute population diversity."""
        fitnesses = np.array([self._get_scalar(f) for f in self.fitnesses])
        if len(fitnesses) < 2:
            return 0.0
        
        mean = np.mean(fitnesses)
        if mean == 0:
            return 0.0
        
        return np.std(fitnesses) / (abs(mean) + 1e-8)
    
    def _get_scalar(self, fitness: F) -> float:
        """Extract scalar fitness from result."""
        return self.evaluator.get_scalar_fitness(fitness)
    
    def _get_objective_dict(self, fitness: F) -> Dict[str, float]:
        """Extract objective dict from fitness result."""
        objectives = self.evaluator.get_objectives(fitness)
        names = self.evaluator.get_objective_names()
        
        result = {}
        for i, val in enumerate(objectives):
            name = names[i] if i < len(names) else f"obj_{i}"
            result[name] = float(-val)  # Negate back from minimization
        
        return result
    
    def _genome_key(self, genome: G) -> str:
        """Generate cache key for genome."""
        if hasattr(genome, 'fingerprint'):
            return genome.fingerprint()
        elif hasattr(genome, 'to_dict'):
            import hashlib
            import json
            data = json.dumps(genome.to_dict(), sort_keys=True, default=str)
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return str(id(genome))


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: SINGING BOWL GENOME (to show framework flexibility)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SingingBowlGenome:
    """
    Example genome for singing bowl optimization.
    
    Shows how the framework extends to different domains.
    """
    # Geometry
    diameter: float = 0.15      # meters
    height: float = 0.08        # meters
    wall_thickness: float = 0.003  # meters
    
    # Material
    material: str = "bronze"    # bronze, crystal, steel
    
    # Shape profile (radius at different heights)
    # Normalized heights [0, 1] -> radius multipliers
    profile_points: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.9],   # Base: 90% of max radius
        [0.3, 1.0],   # Lower curve: max radius
        [0.7, 0.85],  # Upper curve: 85%
        [1.0, 0.7],   # Rim: 70%
    ]))
    
    def mutate(self, sigma: float = 0.05) -> 'SingingBowlGenome':
        """Mutate bowl parameters."""
        new_profile = self.profile_points.copy()
        new_profile[:, 1] += np.random.normal(0, sigma * 0.5, len(new_profile))
        new_profile[:, 1] = np.clip(new_profile[:, 1], 0.5, 1.2)
        
        return SingingBowlGenome(
            diameter=np.clip(self.diameter + np.random.normal(0, sigma * 0.02), 0.08, 0.30),
            height=np.clip(self.height + np.random.normal(0, sigma * 0.01), 0.04, 0.15),
            wall_thickness=np.clip(self.wall_thickness + np.random.normal(0, sigma * 0.001), 0.002, 0.008),
            material=self.material,
            profile_points=new_profile,
        )
    
    def crossover(self, other: 'SingingBowlGenome') -> 'SingingBowlGenome':
        """Crossover with another bowl genome."""
        alpha = np.random.uniform(0.3, 0.7)
        
        new_profile = alpha * self.profile_points + (1 - alpha) * other.profile_points
        
        return SingingBowlGenome(
            diameter=alpha * self.diameter + (1 - alpha) * other.diameter,
            height=alpha * self.height + (1 - alpha) * other.height,
            wall_thickness=alpha * self.wall_thickness + (1 - alpha) * other.wall_thickness,
            material=np.random.choice([self.material, other.material]),
            profile_points=new_profile,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'diameter': self.diameter,
            'height': self.height,
            'wall_thickness': self.wall_thickness,
            'material': self.material,
            'profile_points': self.profile_points.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SingingBowlGenome':
        data['profile_points'] = np.array(data['profile_points'])
        return cls(**data)


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTERS: Bridge to existing plate-specific code
# ══════════════════════════════════════════════════════════════════════════════

def create_plate_optimizer():
    """
    Factory function to create optimizer for DML plate design.
    
    This bridges the agnostic framework to the existing plate-specific code.
    
    Returns:
        Configured AgnosticEvolutionaryOptimizer for plates
    """
    # Import existing implementations
    from .plate_physics import PlatePhysicsEngine  # You'd create this
    from .fitness import FitnessEvaluator as PlateFitness
    from .plate_genome import PlateGenome
    
    # These would need adapter classes - left as exercise
    # physics = PlatePhysicsEngineAdapter()
    # evaluator = PlateFitnessAdapter(person, weights)
    # factory = PlateGenomeFactory(constraints)
    
    # return AgnosticEvolutionaryOptimizer(physics, evaluator, factory)
    
    raise NotImplementedError(
        "Create adapters for PlatePhysics, FitnessEvaluator, PlateGenomeFactory "
        "to use the agnostic framework with existing plate code."
    )
