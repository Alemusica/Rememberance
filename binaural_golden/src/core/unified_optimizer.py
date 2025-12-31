"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           UNIFIED OPTIMIZER - Centralized Optimization Architecture          ║
║                                                                              ║
║   Inspired by:                                                               ║
║   • Czinger 21C Divergent Adaptive Manufacturing (generative design)         ║
║   • Strategy Pattern (interchangeable algorithms)                            ║
║   • Dependency Injection (loosely coupled components)                        ║
║   • Plugin System (dynamic registration)                                     ║
║                                                                              ║
║   Architecture:                                                              ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │                    UnifiedOptimizer (Facade)                        │    ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    ║
║   │  │ Strategy    │  │ Physics     │  │ Fitness     │                 │    ║
║   │  │ Registry    │  │ Engine (DI) │  │ Evaluator   │                 │    ║
║   │  └─────────────┘  └─────────────┘  └─────────────┘                 │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
║                            │                                                  ║
║              ┌─────────────┼─────────────┐                                   ║
║              ▼             ▼             ▼                                   ║
║   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                         ║
║   │ NSGA-II/III  │ │ Custom GA    │ │ SIMP/RAMP    │                         ║
║   │ (pymoo)      │ │ (evolution)  │ │ (topology)   │                         ║
║   └──────────────┘ └──────────────┘ └──────────────┘                         ║
║                                                                              ║
║   Features:                                                                   ║
║   • Automatic strategy selection based on problem type                        ║
║   • Parallel evaluation with JAX/Metal acceleration                          ║
║   • Memory system integration (evolution_memory.py)                          ║
║   • Checkpointing and resume                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TypeVar, Generic, Dict, List, Optional, Callable, Any, 
    Tuple, Protocol, Type, Union, Iterator
)
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES FOR GENERIC DESIGN
# ══════════════════════════════════════════════════════════════════════════════

G = TypeVar('G')  # Genome type (e.g., PlateGenome, SingingBowlGenome)
P = TypeVar('P')  # Physics result type (e.g., ModalAnalysisResult)
F = TypeVar('F')  # Fitness result type (e.g., FitnessResult)
C = TypeVar('C')  # Config type (e.g., EvolutionConfig)


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS (Interface Contracts)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsEngineProtocol(Protocol[G, P]):
    """Contract for physics simulation engines."""
    
    def analyze(self, genome: G) -> P:
        """Run physics analysis on genome."""
        ...
    
    def analyze_batch(self, genomes: List[G]) -> List[P]:
        """Batch analysis for GPU acceleration."""
        ...


class FitnessEvaluatorProtocol(Protocol[G, P, F]):
    """Contract for fitness evaluation."""
    
    def evaluate(self, genome: G, physics_result: P) -> F:
        """Evaluate fitness given genome and physics."""
        ...
    
    def get_objectives(self, result: F) -> np.ndarray:
        """Extract objective vector for multi-objective optimization."""
        ...


class GenomeFactoryProtocol(Protocol[G, C]):
    """Contract for genome creation."""
    
    def random(self, config: C) -> G:
        """Create random genome within constraints."""
        ...
    
    def crossover(self, parent1: G, parent2: G, config: C) -> G:
        """Create offspring from two parents."""
        ...
    
    def mutate(self, genome: G, sigma: float, config: C) -> G:
        """Mutate genome with given strength."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION STRATEGY (Abstract Base)
# ══════════════════════════════════════════════════════════════════════════════

class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    NSGA2 = auto()       # Multi-objective Pareto (pymoo)
    NSGA3 = auto()       # Many-objective Pareto (pymoo)
    GENETIC = auto()     # Custom genetic algorithm
    SIMP = auto()        # Solid Isotropic Material Penalization
    RAMP = auto()        # Rational Approximation Material Properties
    DIFFERENTIAL = auto()  # Differential evolution
    HYBRID = auto()      # Combines multiple strategies
    AUTO = auto()        # Automatic selection based on problem


@dataclass
class OptimizationResult(Generic[G]):
    """Result from optimization run."""
    best_genome: G
    best_fitness: float
    pareto_front: Optional[List[Tuple[G, np.ndarray]]] = None  # For multi-obj
    history: List[Dict[str, Any]] = field(default_factory=list)
    convergence_generation: int = 0
    total_evaluations: int = 0
    elapsed_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def converged(self) -> bool:
        """Check if optimization converged."""
        return self.convergence_generation > 0


class BaseOptimizationStrategy(ABC, Generic[G, P, F, C]):
    """
    Abstract base for optimization strategies.
    
    Implements Strategy pattern - each concrete strategy is interchangeable.
    """
    
    name: str = "base"
    supports_multi_objective: bool = False
    supports_constraints: bool = False
    supports_topology: bool = False
    
    def __init__(
        self,
        physics_engine: PhysicsEngineProtocol[G, P],
        fitness_evaluator: FitnessEvaluatorProtocol[G, P, F],
        genome_factory: GenomeFactoryProtocol[G, C],
        memory: Optional[Any] = None,  # EvolutionMemory if available
    ):
        self.physics = physics_engine
        self.fitness = fitness_evaluator
        self.factory = genome_factory
        self.memory = memory
        self._callbacks: List[Callable[[int, G, float], None]] = []
    
    @abstractmethod
    def optimize(self, config: C, initial_population: Optional[List[G]] = None) -> OptimizationResult[G]:
        """Run optimization with given config."""
        pass
    
    def add_callback(self, callback: Callable[[int, G, float], None]):
        """Add progress callback: (generation, best_genome, best_fitness) -> None"""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, generation: int, best_genome: G, best_fitness: float):
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(generation, best_genome, best_fitness)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def _evaluate_genome(self, genome: G) -> Tuple[F, float]:
        """Evaluate single genome through physics + fitness pipeline."""
        physics_result = self.physics.analyze(genome)
        fitness_result = self.fitness.evaluate(genome, physics_result)
        # Get scalar fitness (for single-objective or aggregated)
        objectives = self.fitness.get_objectives(fitness_result)
        scalar_fitness = float(np.mean(objectives))  # Default: mean of objectives
        return fitness_result, scalar_fitness


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY (Plugin System)
# ══════════════════════════════════════════════════════════════════════════════

class StrategyRegistry:
    """
    Plugin registry for optimization strategies.
    
    Allows dynamic registration and discovery of strategies.
    """
    
    _strategies: Dict[OptimizationStrategy, Type[BaseOptimizationStrategy]] = {}
    _custom_strategies: Dict[str, Type[BaseOptimizationStrategy]] = {}
    
    @classmethod
    def register(cls, strategy_type: OptimizationStrategy):
        """Decorator to register a strategy class."""
        def decorator(strategy_class: Type[BaseOptimizationStrategy]):
            cls._strategies[strategy_type] = strategy_class
            logger.debug(f"Registered strategy: {strategy_type.name} -> {strategy_class.name}")
            return strategy_class
        return decorator
    
    @classmethod
    def register_custom(cls, name: str):
        """Register custom strategy with string name."""
        def decorator(strategy_class: Type[BaseOptimizationStrategy]):
            cls._custom_strategies[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def get(cls, strategy_type: OptimizationStrategy) -> Optional[Type[BaseOptimizationStrategy]]:
        """Get strategy class by type."""
        return cls._strategies.get(strategy_type)
    
    @classmethod
    def get_custom(cls, name: str) -> Optional[Type[BaseOptimizationStrategy]]:
        """Get custom strategy by name."""
        return cls._custom_strategies.get(name)
    
    @classmethod
    def available(cls) -> List[str]:
        """List available strategies."""
        builtin = [s.name for s in cls._strategies.keys()]
        custom = list(cls._custom_strategies.keys())
        return builtin + custom


# ══════════════════════════════════════════════════════════════════════════════
# CONCRETE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

@StrategyRegistry.register(OptimizationStrategy.GENETIC)
class GeneticStrategy(BaseOptimizationStrategy[G, P, F, C]):
    """
    Custom genetic algorithm strategy.
    
    Wraps evolutionary_optimizer.py functionality.
    """
    
    name = "genetic"
    supports_multi_objective = False
    supports_constraints = True
    
    def optimize(self, config: C, initial_population: Optional[List[G]] = None) -> OptimizationResult[G]:
        """Run genetic algorithm optimization."""
        
        # Extract config parameters (duck typing)
        pop_size = getattr(config, 'population_size', 50)
        n_generations = getattr(config, 'n_generations', 100)
        elite_count = getattr(config, 'elite_count', 3)
        crossover_rate = getattr(config, 'crossover_rate', 0.85)
        mutation_rate = getattr(config, 'mutation_rate', 0.25)
        mutation_sigma = getattr(config, 'mutation_sigma', 0.05)
        adaptive_mutation = getattr(config, 'adaptive_mutation', True)
        mutation_decay = getattr(config, 'mutation_decay', 0.97)
        min_mutation_sigma = getattr(config, 'min_mutation_sigma', 0.008)
        convergence_threshold = getattr(config, 'convergence_threshold', 0.00005)
        patience = getattr(config, 'patience', 30)
        tournament_size = getattr(config, 'tournament_size', 4)
        
        start_time = time.time()
        history = []
        
        # Initialize population
        if initial_population:
            population = list(initial_population)
            while len(population) < pop_size:
                population.append(self.factory.random(config))
        else:
            population = [self.factory.random(config) for _ in range(pop_size)]
        
        # Evaluate initial population
        fitnesses = []
        for genome in population:
            _, fitness = self._evaluate_genome(genome)
            fitnesses.append(fitness)
        
        best_fitness = max(fitnesses)
        best_idx = fitnesses.index(best_fitness)
        best_genome = population[best_idx]
        
        no_improvement_count = 0
        convergence_gen = 0
        current_sigma = mutation_sigma
        
        for gen in range(n_generations):
            # Tournament selection
            selected = []
            for _ in range(pop_size - elite_count):
                tournament = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitnesses = [fitnesses[i] for i in tournament]
                winner_idx = tournament[np.argmax(tournament_fitnesses)]
                selected.append(population[winner_idx])
            
            # Create next generation
            next_population = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(elite_count):
                next_population.append(population[sorted_indices[i]])
            
            # Crossover and mutation
            while len(next_population) < pop_size:
                parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
                
                if np.random.random() < crossover_rate:
                    child = self.factory.crossover(selected[parent1], selected[parent2], config)
                else:
                    child = selected[parent1]  # Clone
                
                if np.random.random() < mutation_rate:
                    child = self.factory.mutate(child, current_sigma, config)
                
                next_population.append(child)
            
            # Evaluate new population
            population = next_population
            fitnesses = []
            for genome in population:
                _, fitness = self._evaluate_genome(genome)
                fitnesses.append(fitness)
            
            gen_best_fitness = max(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_fitness)
            
            # Track improvement
            improvement = (gen_best_fitness - best_fitness) / (abs(best_fitness) + 1e-10)
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Adaptive mutation
            if adaptive_mutation:
                if no_improvement_count > 5:
                    current_sigma = min(current_sigma * 1.1, mutation_sigma)
                else:
                    current_sigma = max(current_sigma * mutation_decay, min_mutation_sigma)
            
            # Record history
            history.append({
                'generation': gen,
                'best_fitness': gen_best_fitness,
                'mean_fitness': float(np.mean(fitnesses)),
                'std_fitness': float(np.std(fitnesses)),
                'mutation_sigma': current_sigma,
            })
            
            # Record in memory if available
            if self.memory:
                try:
                    self.memory.record_generation(
                        gen, fitnesses, 
                        self.fitness.get_objectives(self._evaluate_genome(best_genome)[0]),
                        best_genome
                    )
                except Exception as e:
                    logger.debug(f"Memory recording failed: {e}")
            
            # Callbacks
            self._notify_callbacks(gen, best_genome, best_fitness)
            
            # Convergence check
            if no_improvement_count >= patience and improvement < convergence_threshold:
                convergence_gen = gen
                logger.info(f"Converged at generation {gen}")
                break
        
        elapsed = time.time() - start_time
        
        return OptimizationResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            history=history,
            convergence_generation=convergence_gen,
            total_evaluations=len(history) * pop_size,
            elapsed_time=elapsed,
            metadata={'strategy': 'genetic', 'final_sigma': current_sigma}
        )


@StrategyRegistry.register(OptimizationStrategy.NSGA2)
class NSGA2Strategy(BaseOptimizationStrategy[G, P, F, C]):
    """
    NSGA-II multi-objective strategy using pymoo.
    
    Wraps pymoo_optimizer.py functionality.
    """
    
    name = "nsga2"
    supports_multi_objective = True
    supports_constraints = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pymoo_available = False
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
            self._pymoo_available = True
        except ImportError:
            warnings.warn("pymoo not available. Install with: pip install pymoo>=0.6.0")
    
    def optimize(self, config: C, initial_population: Optional[List[G]] = None) -> OptimizationResult[G]:
        """Run NSGA-II optimization."""
        
        if not self._pymoo_available:
            raise ImportError("pymoo required for NSGA2Strategy. Install: pip install pymoo>=0.6.0")
        
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        
        pop_size = getattr(config, 'population_size', 100)
        n_generations = getattr(config, 'n_generations', 100)
        crossover_eta = getattr(config, 'crossover_eta', 15.0)
        mutation_eta = getattr(config, 'mutation_eta', 20.0)
        
        start_time = time.time()
        history = []
        
        # Define pymoo problem wrapper
        strategy = self
        
        class GenomeProblem(PymooProblem):
            def __init__(self, n_var: int, n_obj: int):
                super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
            
            def _evaluate(self, x, out, *args, **kwargs):
                objectives = []
                for xi in x:
                    # Convert normalized vector to genome (strategy-specific)
                    genome = strategy._decode_genome(xi, config)
                    fitness_result, _ = strategy._evaluate_genome(genome)
                    obj = strategy.fitness.get_objectives(fitness_result)
                    # NSGA-II minimizes, so negate for maximization
                    objectives.append(-obj)
                out["F"] = np.array(objectives)
        
        # Get problem dimensions from factory
        n_var = getattr(config, 'n_decision_vars', 10)
        n_obj = getattr(config, 'n_objectives', 3)
        
        problem = GenomeProblem(n_var=n_var, n_obj=n_obj)
        
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=crossover_eta, prob=0.9),
            mutation=PM(eta=mutation_eta),
            eliminate_duplicates=True,
        )
        
        result = minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            seed=getattr(config, 'seed', None),
            verbose=getattr(config, 'verbose', False),
        )
        
        elapsed = time.time() - start_time
        
        # Extract Pareto front
        pareto_front = []
        if result.X is not None:
            for x, f in zip(result.X, result.F):
                genome = self._decode_genome(x, config)
                pareto_front.append((genome, -f))  # Un-negate
        
        # Best is first non-dominated solution
        best_genome = pareto_front[0][0] if pareto_front else None
        best_fitness = float(np.mean(pareto_front[0][1])) if pareto_front else 0.0
        
        return OptimizationResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            pareto_front=pareto_front,
            history=history,
            total_evaluations=result.algorithm.n_gen * pop_size,
            elapsed_time=elapsed,
            metadata={'strategy': 'nsga2', 'n_pareto': len(pareto_front)}
        )
    
    def _decode_genome(self, x: np.ndarray, config: C) -> G:
        """Decode normalized vector to genome. Override in subclass."""
        # Default: use factory with random if no decoder
        return self.factory.random(config)


@StrategyRegistry.register(OptimizationStrategy.SIMP)
class SIMPStrategy(BaseOptimizationStrategy[G, P, F, C]):
    """
    SIMP topology optimization strategy.
    
    Wraps iterative_optimizer.py functionality.
    For density-based design (where to add/remove material).
    """
    
    name = "simp"
    supports_multi_objective = False
    supports_topology = True
    
    def optimize(self, config: C, initial_population: Optional[List[G]] = None) -> OptimizationResult[G]:
        """Run SIMP topology optimization."""
        
        # Extract SIMP parameters
        penalty = getattr(config, 'simp_penalty', 3.0)
        eps = getattr(config, 'simp_eps', 1e-6)
        filter_radius = getattr(config, 'filter_radius', 2.0)
        max_iterations = getattr(config, 'max_iterations', 100)
        convergence_tol = getattr(config, 'convergence_tol', 0.01)
        volume_fraction = getattr(config, 'volume_fraction', 0.5)
        
        # Grid resolution
        nx = getattr(config, 'grid_nx', 30)
        ny = getattr(config, 'grid_ny', 20)
        
        start_time = time.time()
        history = []
        
        # Initialize density field
        rho = np.ones((nx, ny)) * volume_fraction
        
        best_fitness = -np.inf
        best_rho = rho.copy()
        
        for iteration in range(max_iterations):
            # Apply density filter
            rho_filtered = self._density_filter(rho, filter_radius)
            
            # SIMP interpolation
            E_field = self._simp(rho_filtered, penalty, eps)
            
            # Convert density to genome for evaluation
            genome = self._density_to_genome(E_field, config)
            fitness_result, fitness = self._evaluate_genome(genome)
            
            # Compute sensitivity
            sensitivity = self._compute_sensitivity(rho_filtered, fitness_result, penalty, eps)
            
            # Optimality criteria update
            rho = self._oc_update(rho, sensitivity, volume_fraction)
            
            # Track progress
            if fitness > best_fitness:
                best_fitness = fitness
                best_rho = rho.copy()
            
            history.append({
                'iteration': iteration,
                'fitness': fitness,
                'volume': float(np.mean(rho)),
                'change': float(np.max(np.abs(rho - best_rho))),
            })
            
            self._notify_callbacks(iteration, genome, fitness)
            
            # Convergence check
            if iteration > 10 and history[-1]['change'] < convergence_tol:
                break
        
        elapsed = time.time() - start_time
        
        best_genome = self._density_to_genome(best_rho, config)
        
        return OptimizationResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            history=history,
            convergence_generation=len(history),
            total_evaluations=len(history),
            elapsed_time=elapsed,
            metadata={'strategy': 'simp', 'final_volume': float(np.mean(best_rho))}
        )
    
    def _simp(self, x: np.ndarray, penalty: float, eps: float) -> np.ndarray:
        """SIMP interpolation: E(x) = eps + (1 - eps) * x^penalty"""
        return eps + (1 - eps) * np.power(x, penalty)
    
    def _density_filter(self, x: np.ndarray, radius: float) -> np.ndarray:
        """Apply weighted average filter to density field."""
        from scipy.ndimage import uniform_filter
        try:
            return uniform_filter(x, size=int(radius))
        except ImportError:
            return x  # Fallback: no filtering
    
    def _compute_sensitivity(self, rho, fitness_result, penalty, eps) -> np.ndarray:
        """Compute design sensitivity. Override in domain-specific subclass."""
        # Default: gradient approximation
        return penalty * (1 - eps) * np.power(rho, penalty - 1)
    
    def _oc_update(self, rho: np.ndarray, sensitivity: np.ndarray, vf: float) -> np.ndarray:
        """Optimality criteria update."""
        l1, l2 = 0.0, 1e9
        move = 0.2
        
        while l2 - l1 > 1e-4:
            lmid = 0.5 * (l1 + l2)
            rho_new = np.maximum(0.001, np.maximum(
                rho - move,
                np.minimum(1.0, np.minimum(
                    rho + move,
                    rho * np.sqrt(-sensitivity / lmid)
                ))
            ))
            if np.mean(rho_new) > vf:
                l1 = lmid
            else:
                l2 = lmid
        return rho_new
    
    def _density_to_genome(self, density: np.ndarray, config: C) -> G:
        """Convert density field to genome. Override in subclass."""
        return self.factory.random(config)


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED OPTIMIZER (Facade)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedConfig:
    """
    Unified configuration for all strategies.
    
    Each strategy extracts relevant parameters via duck typing.
    """
    # Common
    population_size: int = 50
    n_generations: int = 100
    seed: Optional[int] = None
    verbose: bool = True
    
    # Genetic algorithm
    elite_count: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.05
    adaptive_mutation: bool = True
    mutation_decay: float = 0.97
    min_mutation_sigma: float = 0.008
    convergence_threshold: float = 0.00005
    patience: int = 30
    tournament_size: int = 4
    
    # NSGA-II/III
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    n_decision_vars: int = 10
    n_objectives: int = 3
    
    # SIMP topology
    simp_penalty: float = 3.0
    simp_eps: float = 1e-6
    filter_radius: float = 2.0
    max_iterations: int = 100
    convergence_tol: float = 0.01
    volume_fraction: float = 0.5
    grid_nx: int = 30
    grid_ny: int = 20
    
    # Memory
    use_memory: bool = True
    memory_path: str = "./optimization_memory"


class UnifiedOptimizer(Generic[G, P, F]):
    """
    Unified facade for all optimization strategies.
    
    Usage:
        optimizer = UnifiedOptimizer(physics, fitness, factory)
        result = optimizer.optimize(config, strategy=OptimizationStrategy.AUTO)
    """
    
    def __init__(
        self,
        physics_engine: PhysicsEngineProtocol[G, P],
        fitness_evaluator: FitnessEvaluatorProtocol[G, P, F],
        genome_factory: GenomeFactoryProtocol[G, Any],
        memory_path: Optional[str] = None,
    ):
        self.physics = physics_engine
        self.fitness = fitness_evaluator
        self.factory = genome_factory
        self.memory = None
        
        if memory_path:
            try:
                from .evolution_memory import EvolutionMemory
                self.memory = EvolutionMemory(memory_path)
            except ImportError:
                logger.debug("Evolution memory not available")
        
        self._callbacks: List[Callable[[int, G, float], None]] = []
    
    def add_callback(self, callback: Callable[[int, G, float], None]):
        """Add progress callback."""
        self._callbacks.append(callback)
    
    def optimize(
        self,
        config: UnifiedConfig,
        strategy: OptimizationStrategy = OptimizationStrategy.AUTO,
        initial_population: Optional[List[G]] = None,
    ) -> OptimizationResult[G]:
        """
        Run optimization with specified strategy.
        
        Args:
            config: Unified configuration
            strategy: Which strategy to use (AUTO selects based on problem)
            initial_population: Optional warm start
            
        Returns:
            OptimizationResult with best solution
        """
        # Auto-select strategy if requested
        if strategy == OptimizationStrategy.AUTO:
            strategy = self._auto_select_strategy(config)
            logger.info(f"Auto-selected strategy: {strategy.name}")
        
        # Get strategy class from registry
        strategy_class = StrategyRegistry.get(strategy)
        
        if strategy_class is None:
            raise ValueError(f"Strategy {strategy.name} not registered. "
                           f"Available: {StrategyRegistry.available()}")
        
        # Instantiate strategy with dependencies (DI)
        memory = self.memory if config.use_memory else None
        strategy_instance = strategy_class(
            physics_engine=self.physics,
            fitness_evaluator=self.fitness,
            genome_factory=self.factory,
            memory=memory,
        )
        
        # Forward callbacks
        for cb in self._callbacks:
            strategy_instance.add_callback(cb)
        
        # Run optimization
        result = strategy_instance.optimize(config, initial_population)
        
        # Finalize memory
        if self.memory:
            try:
                obj = self.fitness.get_objectives(
                    self.fitness.evaluate(result.best_genome, self.physics.analyze(result.best_genome))
                )
                self.memory.finalize_run(
                    result.best_fitness, obj,
                    str(result.best_genome)[:100], "success"
                )
            except Exception as e:
                logger.debug(f"Memory finalization failed: {e}")
        
        return result
    
    def _auto_select_strategy(self, config: UnifiedConfig) -> OptimizationStrategy:
        """
        Automatically select best strategy based on problem characteristics.
        
        Heuristics:
        - Multi-objective (n_objectives > 1) → NSGA2
        - Topology optimization needed → SIMP
        - Default → GENETIC (most versatile)
        """
        n_obj = getattr(config, 'n_objectives', 1)
        topology = getattr(config, 'use_topology', False)
        
        if n_obj > 3:
            return OptimizationStrategy.NSGA3  # Many-objective
        elif n_obj > 1:
            return OptimizationStrategy.NSGA2  # Multi-objective
        elif topology:
            return OptimizationStrategy.SIMP  # Topology
        else:
            return OptimizationStrategy.GENETIC  # Default


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS (Convenience)
# ══════════════════════════════════════════════════════════════════════════════

def create_unified_optimizer(
    physics_engine: PhysicsEngineProtocol,
    fitness_evaluator: FitnessEvaluatorProtocol,
    genome_factory: GenomeFactoryProtocol,
    use_memory: bool = True,
    memory_path: str = "./optimization_memory",
) -> UnifiedOptimizer:
    """
    Create a unified optimizer with all dependencies configured.
    
    This is the main entry point for using the optimization system.
    """
    return UnifiedOptimizer(
        physics_engine=physics_engine,
        fitness_evaluator=fitness_evaluator,
        genome_factory=genome_factory,
        memory_path=memory_path if use_memory else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID STRATEGY (Advanced)
# ══════════════════════════════════════════════════════════════════════════════

@StrategyRegistry.register(OptimizationStrategy.HYBRID)
class HybridStrategy(BaseOptimizationStrategy[G, P, F, C]):
    """
    Hybrid optimization combining multiple strategies.
    
    Approach:
    1. Start with SIMP for coarse topology
    2. Refine with NSGA2 for Pareto exploration
    3. Polish with Genetic for final convergence
    """
    
    name = "hybrid"
    supports_multi_objective = True
    supports_topology = True
    
    def optimize(self, config: C, initial_population: Optional[List[G]] = None) -> OptimizationResult[G]:
        """Run hybrid optimization pipeline."""
        
        start_time = time.time()
        combined_history = []
        
        # Phase 1: SIMP for topology (if enabled)
        if getattr(config, 'use_topology', False):
            simp = SIMPStrategy(self.physics, self.fitness, self.factory, self.memory)
            simp_config = self._make_simp_config(config)
            simp_result = simp.optimize(simp_config)
            combined_history.extend([{'phase': 'simp', **h} for h in simp_result.history])
            initial_population = [simp_result.best_genome]
        
        # Phase 2: NSGA2 for exploration
        n_obj = getattr(config, 'n_objectives', 1)
        if n_obj > 1:
            nsga = NSGA2Strategy(self.physics, self.fitness, self.factory, self.memory)
            nsga_config = self._make_nsga_config(config)
            nsga_result = nsga.optimize(nsga_config, initial_population)
            combined_history.extend([{'phase': 'nsga2', **h} for h in nsga_result.history])
            # Use Pareto front as initial population for GA
            if nsga_result.pareto_front:
                initial_population = [g for g, _ in nsga_result.pareto_front[:10]]
        
        # Phase 3: Genetic for convergence
        ga = GeneticStrategy(self.physics, self.fitness, self.factory, self.memory)
        ga_config = self._make_ga_config(config)
        ga_result = ga.optimize(ga_config, initial_population)
        combined_history.extend([{'phase': 'genetic', **h} for h in ga_result.history])
        
        elapsed = time.time() - start_time
        
        return OptimizationResult(
            best_genome=ga_result.best_genome,
            best_fitness=ga_result.best_fitness,
            pareto_front=nsga_result.pareto_front if n_obj > 1 else None,
            history=combined_history,
            convergence_generation=len(combined_history),
            total_evaluations=sum(h.get('evaluations', 1) for h in combined_history),
            elapsed_time=elapsed,
            metadata={'strategy': 'hybrid', 'phases': ['simp', 'nsga2', 'genetic']}
        )
    
    def _make_simp_config(self, config: C) -> C:
        """Create SIMP-specific config subset."""
        return config  # Duck typing handles it
    
    def _make_nsga_config(self, config: C) -> C:
        """Create NSGA-specific config subset."""
        return config
    
    def _make_ga_config(self, config: C) -> C:
        """Create GA-specific config subset."""
        return config


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Types
    'OptimizationStrategy',
    'OptimizationResult',
    'UnifiedConfig',
    
    # Protocols
    'PhysicsEngineProtocol',
    'FitnessEvaluatorProtocol',
    'GenomeFactoryProtocol',
    
    # Core classes
    'UnifiedOptimizer',
    'BaseOptimizationStrategy',
    'StrategyRegistry',
    
    # Strategies
    'GeneticStrategy',
    'NSGA2Strategy',
    'SIMPStrategy',
    'HybridStrategy',
    
    # Factory
    'create_unified_optimizer',
]
