"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PYMOO MULTI-OBJECTIVE OPTIMIZER - NSGA-II for DML Plates           ║
║                                                                              ║
║   True Pareto-based multi-objective optimization using pymoo.                ║
║   Separates conflicting objectives for proper trade-off analysis:            ║
║                                                                              ║
║   Objectives (minimize):                                                     ║
║   1. -ear_uniformity: L/R balance (want to MAXIMIZE → minimize negative)     ║
║   2. -spine_flatness: Response flatness at spine (maximize → min negative)   ║
║   3. -ear_flatness: Response flatness at ears (maximize → min negative)      ║
║                                                                              ║
║   Reference: Deb et al. 2002 "A Fast and Elitist Multi-objective GA"        ║
║   Applied: Bai & Liu 2004 for exciter placement optimization                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import copy
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Try to import pymoo - graceful fallback if not available
try:
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    PymooProblem = object  # Dummy base class when pymoo not available
    logger.warning("pymoo not installed. Run: pip install pymoo>=0.6.0")

# Local imports
from .person import Person
from .plate_genome import PlateGenome, ContourType, ExciterPosition, CutoutGene
from .fitness import FitnessEvaluator, FitnessResult, ObjectiveWeights, ZoneWeights, ObjectiveVector


@dataclass
class PymooConfig:
    """Configuration for pymoo multi-objective optimization."""
    
    # Algorithm
    algorithm: str = "nsga2"  # "nsga2" or "nsga3"
    population_size: int = 100
    n_generations: int = 100
    
    # Operators
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0  # Distribution index for SBX
    mutation_prob: float = None  # Auto-set to 1/n_var if None
    mutation_eta: float = 20.0   # Distribution index for PM
    
    # Decision variables bounds
    # Plate dimensions
    min_length: float = 1.5   # m
    max_length: float = 2.4   # m
    min_width: float = 0.5    # m
    max_width: float = 1.0    # m
    
    # Thickness variation
    min_thickness_var: float = 0.0
    max_thickness_var: float = 0.3
    
    # Exciter positions (normalized 0-1)
    n_exciters: int = 2
    min_exciter_diameter: float = 0.04  # m
    max_exciter_diameter: float = 0.08  # m
    
    # Convergence
    seed: Optional[int] = None
    verbose: bool = True


class PlateOptimizationProblem(PymooProblem):
    """
    Pymoo Problem definition for DML plate optimization.
    
    Decision Variables (example with 2 exciters):
    - x[0]: plate_length (normalized)
    - x[1]: plate_width (normalized)
    - x[2]: thickness_variation (0-1)
    - x[3]: exciter1_x (0-1)
    - x[4]: exciter1_y (0-1)
    - x[5]: exciter1_diameter (normalized)
    - x[6]: exciter2_x (0-1)
    - x[7]: exciter2_y (0-1)
    - x[8]: exciter2_diameter (normalized)
    
    Objectives (all minimized) - 6D ObjectiveVector:
    - f[0]: -spine_flatness (want high → minimize negative)
    - f[1]: -ear_flatness (want high → minimize negative)
    - f[2]: -ear_lr_uniformity (L/R balance, want high → minimize negative)
    - f[3]: -spine_energy (want high → minimize negative)
    - f[4]: -mass_score (lighter is better → minimize negative)
    - f[5]: -structural_safety (deflection constraint → minimize negative)
    
    Constraints:
    - g[0]: deflection_mm - 10.0 <= 0 (structural safety)
    """
    
    # Number of objectives in ObjectiveVector
    N_OBJECTIVES = 6
    
    # Objective names for Pareto front analysis
    OBJECTIVE_NAMES = [
        "spine_flatness", "ear_flatness", "ear_lr_uniformity",
        "spine_energy", "mass_score", "structural_safety"
    ]
    
    def __init__(
        self,
        person: Person,
        evaluator: FitnessEvaluator,
        config: PymooConfig,
        contour_type: ContourType = ContourType.RECTANGLE,
    ):
        self.person = person
        self.evaluator = evaluator
        self.config = config
        self.contour_type = contour_type
        
        # Calculate number of decision variables
        # 3 base (length, width, thickness_var) + 3 per exciter (x, y, diameter)
        n_var = 3 + config.n_exciters * 3
        
        # Bounds
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        
        # Initialize pymoo Problem with 6 objectives from ObjectiveVector
        super().__init__(
            n_var=n_var,
            n_obj=self.N_OBJECTIVES,  # 6D multi-objective from ObjectiveVector
            n_ieq_constr=1,           # structural deflection constraint
            xl=xl,
            xu=xu,
        )
        
        # Cache for evaluation results
        self._eval_cache: Dict[str, Tuple[np.ndarray, np.ndarray, FitnessResult, ObjectiveVector]] = {}
    
    def _decode_genome(self, x: np.ndarray) -> PlateGenome:
        """
        Decode decision vector to PlateGenome.
        
        Args:
            x: Normalized decision vector [0, 1]
        
        Returns:
            PlateGenome instance
        """
        cfg = self.config
        
        # Plate dimensions
        length = cfg.min_length + x[0] * (cfg.max_length - cfg.min_length)
        width = cfg.min_width + x[1] * (cfg.max_width - cfg.min_width)
        thickness_var = cfg.min_thickness_var + x[2] * (cfg.max_thickness_var - cfg.min_thickness_var)
        
        # Exciters
        exciters = []
        for i in range(cfg.n_exciters):
            base_idx = 3 + i * 3
            ex_x = x[base_idx]      # Normalized position
            ex_y = x[base_idx + 1]  # Normalized position
            ex_d = cfg.min_exciter_diameter + x[base_idx + 2] * (
                cfg.max_exciter_diameter - cfg.min_exciter_diameter
            )
            exciters.append(ExciterPosition(x=ex_x, y=ex_y, diameter=ex_d))
        
        return PlateGenome(
            length=length,
            width=width,
            contour=self.contour_type,
            exciters=exciters,
            thickness_variation=thickness_var,
            cutouts=[],  # Start without cutouts for simpler optimization
        )
    
    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate population of solutions using 6D ObjectiveVector.
        
        Args:
            x: Population matrix (pop_size x n_var)
            out: Output dictionary for objectives and constraints
        """
        pop_size = x.shape[0]
        
        # 6D Objectives (minimize) from ObjectiveVector
        F = np.zeros((pop_size, self.N_OBJECTIVES))
        
        # Constraints (g <= 0 is feasible)
        G = np.zeros((pop_size, 1))
        
        for i in range(pop_size):
            genome = self._decode_genome(x[i])
            
            # ═══════════════════════════════════════════════════════════════════
            # EVALUATE WITH MULTI-OBJECTIVE (returns FitnessResult + ObjectiveVector)
            # ═══════════════════════════════════════════════════════════════════
            result, obj_vector = self.evaluator.evaluate_multi(genome)
            
            # ═══════════════════════════════════════════════════════════════════
            # 6D OBJECTIVE VECTOR (all negated for minimization)
            # ═══════════════════════════════════════════════════════════════════
            F[i, :] = obj_vector.to_minimize_array()
            
            # ═══════════════════════════════════════════════════════════════════
            # CONSTRAINT: Structural Deflection < 10mm
            # ═══════════════════════════════════════════════════════════════════
            G[i, 0] = result.max_deflection_mm - 10.0  # g <= 0 means feasible
        
        out["F"] = F
        out["G"] = G
    
    def _compute_ear_uniformity(self, result: FitnessResult) -> float:
        """
        Compute L/R ear uniformity from head_response.
        
        This is the KEY metric we need to optimize!
        
        Returns:
            Uniformity score [0, 1] where 1 = perfect L/R balance
        """
        if result.head_response is None:
            return 0.0
        
        head_resp = result.head_response
        
        # Head response is (n_positions, n_frequencies)
        # First position is left ear, second is right ear
        if len(head_resp) < 2:
            return 0.5  # Can't compute without both ears
        
        # Get left and right ear responses (first two positions)
        left_response = head_resp[0] if len(head_resp.shape) == 2 else head_resp[0:len(head_resp)//2]
        right_response = head_resp[1] if len(head_resp.shape) == 2 else head_resp[len(head_resp)//2:]
        
        # Ensure both are 1D arrays
        if len(left_response.shape) > 1:
            left_response = np.mean(left_response, axis=0)
        if len(right_response.shape) > 1:
            right_response = np.mean(right_response, axis=0)
        
        # Compute uniformity as 1 - normalized difference
        # Use RMS of difference normalized by average level
        left_rms = np.sqrt(np.mean(left_response**2))
        right_rms = np.sqrt(np.mean(right_response**2))
        
        if left_rms + right_rms < 1e-10:
            return 0.0
        
        # L/R balance: ratio of smaller to larger
        min_rms = min(left_rms, right_rms)
        max_rms = max(left_rms, right_rms)
        
        if max_rms < 1e-10:
            return 0.0
        
        level_balance = min_rms / max_rms  # 1.0 = perfect balance
        
        # Also check frequency-by-frequency correlation
        if len(left_response) == len(right_response) and len(left_response) > 1:
            correlation = np.corrcoef(left_response, right_response)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            correlation = max(0.0, correlation)  # Clamp negative correlations
        else:
            correlation = 0.5
        
        # Combined uniformity: 60% level balance + 40% correlation
        uniformity = 0.6 * level_balance + 0.4 * correlation
        
        return float(np.clip(uniformity, 0.0, 1.0))


class PymooOptimizer:
    """
    Multi-objective optimizer using pymoo NSGA-II/NSGA-III.
    
    Provides Pareto-optimal solutions for:
    - Ear L/R uniformity
    - Spine response flatness
    - Ear response flatness
    
    Usage:
        optimizer = PymooOptimizer(person)
        result = optimizer.run()
        
        # Get best balanced solution
        best = result.get_best_by_preference(ear_weight=0.5, spine_weight=0.3)
        
        # Or get Pareto front
        pareto_front = result.pareto_genomes
    """
    
    def __init__(
        self,
        person: Person,
        config: Optional[PymooConfig] = None,
        objectives: Optional[ObjectiveWeights] = None,
        zone_weights: Optional[ZoneWeights] = None,
        material: str = "birch_plywood",
        contour_type: ContourType = ContourType.RECTANGLE,
    ):
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "pymoo is required for PymooOptimizer. "
                "Install with: pip install pymoo>=0.6.0"
            )
        
        self.person = person
        self.config = config or PymooConfig()
        self.material = material
        self.contour_type = contour_type
        
        # Create fitness evaluator
        self.evaluator = FitnessEvaluator(
            person=person,
            objectives=objectives,
            zone_weights=zone_weights,
            material=material,
        )
        
        # Create problem
        self.problem = PlateOptimizationProblem(
            person=person,
            evaluator=self.evaluator,
            config=self.config,
            contour_type=contour_type,
        )
        
        # Results
        self._result = None
        self._pareto_genomes: List[PlateGenome] = []
        self._pareto_fitness: List[FitnessResult] = []
    
    def run(
        self,
        callback: Optional[Callable] = None,
        verbose: bool = True,
    ) -> 'PymooResult':
        """
        Run multi-objective optimization.
        
        Returns:
            PymooResult with Pareto front and utilities
        """
        cfg = self.config
        
        # Set seed
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        
        # Mutation probability
        mutation_prob = cfg.mutation_prob
        if mutation_prob is None:
            mutation_prob = 1.0 / self.problem.n_var
        
        # Number of objectives (6D from ObjectiveVector)
        n_obj = PlateOptimizationProblem.N_OBJECTIVES
        
        # Create algorithm
        if cfg.algorithm == "nsga3":
            # NSGA-III needs reference directions - more partitions for 6D
            # For 6 objectives, use layered approach to avoid explosion
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
            algorithm = NSGA3(
                pop_size=cfg.population_size,
                ref_dirs=ref_dirs,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=cfg.crossover_prob, eta=cfg.crossover_eta),
                mutation=PM(prob=mutation_prob, eta=cfg.mutation_eta),
            )
        else:
            # Default: NSGA-II
            algorithm = NSGA2(
                pop_size=cfg.population_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=cfg.crossover_prob, eta=cfg.crossover_eta),
                mutation=PM(prob=mutation_prob, eta=cfg.mutation_eta),
                eliminate_duplicates=True,
            )
        
        if verbose:
            print(f"Starting {cfg.algorithm.upper()} 6D optimization:")
            print(f"  Population: {cfg.population_size}")
            print(f"  Generations: {cfg.n_generations}")
            print(f"  Decision vars: {self.problem.n_var}")
            print(f"  Objectives (6D): {', '.join(PlateOptimizationProblem.OBJECTIVE_NAMES)}")
        
        start_time = time.time()
        
        # Run optimization
        self._result = minimize(
            self.problem,
            algorithm,
            ('n_gen', cfg.n_generations),
            seed=cfg.seed,
            verbose=verbose,
        )
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nOptimization completed in {elapsed:.1f}s")
            print(f"  Pareto solutions: {len(self._result.X)}")
        
        # Decode Pareto front
        self._decode_pareto_front()
        
        return PymooResult(
            pareto_genomes=self._pareto_genomes,
            pareto_fitness=self._pareto_fitness,
            pareto_objectives=-self._result.F,  # Negate back to maximization
            elapsed_time=elapsed,
            n_generations=cfg.n_generations,
            algorithm=cfg.algorithm,
        )
    
    def _decode_pareto_front(self):
        """Decode Pareto optimal solutions to PlateGenome."""
        self._pareto_genomes = []
        self._pareto_fitness = []
        
        if self._result is None or self._result.X is None:
            return
        
        for x in self._result.X:
            genome = self.problem._decode_genome(x)
            fitness = self.evaluator.evaluate(genome)
            
            self._pareto_genomes.append(genome)
            self._pareto_fitness.append(fitness)


@dataclass
class PymooResult:
    """
    Result from multi-objective optimization.
    
    Contains 6D Pareto front from ObjectiveVector:
    - spine_flatness, ear_flatness, ear_lr_uniformity
    - spine_energy, mass_score, structural_safety
    """
    pareto_genomes: List[PlateGenome]
    pareto_fitness: List[FitnessResult]
    pareto_objectives: np.ndarray  # (n_solutions, 6) from ObjectiveVector
    elapsed_time: float
    n_generations: int
    algorithm: str
    
    # 6D objective names matching ObjectiveVector order
    OBJECTIVE_NAMES = PlateOptimizationProblem.OBJECTIVE_NAMES
    
    def get_best_by_preference(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[PlateGenome, FitnessResult]:
        """
        Select best solution based on preference weights for 6D objectives.
        
        Args:
            weights: Dict mapping objective names to weights, e.g.:
                {"spine_flatness": 0.2, "ear_lr_uniformity": 0.5, "spine_energy": 0.3}
                Missing objectives default to 0.
        
        Returns:
            (best_genome, best_fitness)
        """
        if len(self.pareto_genomes) == 0:
            raise ValueError("No Pareto solutions available")
        
        # Default weights if not provided (prioritize ear balance)
        if weights is None:
            weights = {
                "spine_flatness": 0.15,
                "ear_flatness": 0.15,
                "ear_lr_uniformity": 0.40,  # Highest priority!
                "spine_energy": 0.15,
                "mass_score": 0.05,
                "structural_safety": 0.10,
            }
        
        # Build weight vector
        w = np.array([weights.get(name, 0.0) for name in self.OBJECTIVE_NAMES])
        
        # Normalize
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(self.OBJECTIVE_NAMES)) / len(self.OBJECTIVE_NAMES)
        
        # Weighted sum (objectives are already positive after negation in run())
        scores = self.pareto_objectives @ w
        
        best_idx = np.argmax(scores)
        
        return self.pareto_genomes[best_idx], self.pareto_fitness[best_idx]
    
    def get_best_for_objective(self, objective_name: str) -> Tuple[PlateGenome, FitnessResult]:
        """
        Get solution with best value for a specific objective.
        
        Args:
            objective_name: One of OBJECTIVE_NAMES
        
        Returns:
            (best_genome, best_fitness)
        """
        if objective_name not in self.OBJECTIVE_NAMES:
            raise ValueError(f"Unknown objective: {objective_name}. Valid: {self.OBJECTIVE_NAMES}")
        
        idx = self.OBJECTIVE_NAMES.index(objective_name)
        best_idx = np.argmax(self.pareto_objectives[:, idx])
        return self.pareto_genomes[best_idx], self.pareto_fitness[best_idx]
    
    def get_best_ear_uniformity(self) -> Tuple[PlateGenome, FitnessResult]:
        """Get solution with best ear L/R uniformity."""
        return self.get_best_for_objective("ear_lr_uniformity")
    
    def get_best_spine_flatness(self) -> Tuple[PlateGenome, FitnessResult]:
        """Get solution with best spine flatness."""
        return self.get_best_for_objective("spine_flatness")
    
    def get_best_ear_flatness(self) -> Tuple[PlateGenome, FitnessResult]:
        """Get solution with best ear flatness."""
        return self.get_best_for_objective("ear_flatness")
    
    def get_knee_point(self) -> Tuple[PlateGenome, FitnessResult]:
        """
        Find knee point of 6D Pareto front (best trade-off).
        
        Uses perpendicular distance from utopia-nadir line in normalized space.
        """
        if len(self.pareto_objectives) <= 1:
            return self.pareto_genomes[0], self.pareto_fitness[0]
        
        n_obj = self.pareto_objectives.shape[1]  # Should be 6
        
        # Normalize objectives
        obj = self.pareto_objectives
        obj_min = obj.min(axis=0)
        obj_max = obj.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0
        obj_norm = (obj - obj_min) / obj_range
        
        # Utopia point (max of each objective) in n_obj dimensions
        utopia = np.ones(n_obj)
        # Nadir point (min of each objective)
        nadir = np.zeros(n_obj)
        
        # Line direction
        line_dir = utopia - nadir
        line_dir = line_dir / np.linalg.norm(line_dir)
        
        # Find point with max perpendicular distance
        max_dist = -1
        knee_idx = 0
        
        for i, point in enumerate(obj_norm):
            # Vector from nadir to point
            v = point - nadir
            # Projection onto line
            proj = np.dot(v, line_dir) * line_dir
            # Perpendicular distance
            perp = v - proj
            dist = np.linalg.norm(perp)
            
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return self.pareto_genomes[knee_idx], self.pareto_fitness[knee_idx]
    
    def get_best_balanced_index(self) -> int:
        """
        Get index of best balanced solution (knee point) in 6D space.
        
        Returns:
            Index into pareto_genomes/pareto_fitness arrays
        """
        if len(self.pareto_objectives) <= 1:
            return 0
        
        n_obj = self.pareto_objectives.shape[1]  # Should be 6
        
        # Use same knee-point algorithm for 6D
        obj = self.pareto_objectives
        obj_min = obj.min(axis=0)
        obj_max = obj.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0
        obj_norm = (obj - obj_min) / obj_range
        
        utopia = np.ones(n_obj)
        nadir = np.zeros(n_obj)
        line_dir = utopia - nadir
        line_dir = line_dir / np.linalg.norm(line_dir)
        
        max_dist = -1
        knee_idx = 0
        
        for i, point in enumerate(obj_norm):
            v = point - nadir
            proj = np.dot(v, line_dir) * line_dir
            perp = v - proj
            dist = np.linalg.norm(perp)
            
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return knee_idx
    
    def get_labeled_objectives(self, idx: int) -> Dict[str, float]:
        """
        Get labeled objective values for a specific Pareto solution.
        
        Args:
            idx: Index into pareto_objectives
        
        Returns:
            Dict with objective names and values, e.g.:
            {"spine_flatness": 0.85, "ear_lr_uniformity": 0.92, ...}
        """
        if idx < 0 or idx >= len(self.pareto_objectives):
            raise ValueError(f"Index {idx} out of range [0, {len(self.pareto_objectives)})")
        
        return {name: self.pareto_objectives[idx, i] 
                for i, name in enumerate(self.OBJECTIVE_NAMES)}
    
    def summary(self) -> str:
        """Generate summary of 6D optimization results."""
        lines = [
            f"═══════════════════════════════════════════════════════════════",
            f"  PYMOO {self.algorithm.upper()} 6D Optimization Results",
            f"═══════════════════════════════════════════════════════════════",
            f"  Generations: {self.n_generations}",
            f"  Time: {self.elapsed_time:.1f}s",
            f"  Pareto solutions: {len(self.pareto_genomes)}",
            "",
        ]
        
        if len(self.pareto_objectives) > 0:
            lines.append("  Objective Ranges (6D Pareto Front):")
            for i, name in enumerate(self.OBJECTIVE_NAMES):
                obj_min = self.pareto_objectives[:, i].min()
                obj_max = self.pareto_objectives[:, i].max()
                lines.append(f"    {name:20s}: [{obj_min:.3f} - {obj_max:.3f}]")
            lines.append("")
            
            # Best solutions
            best_ear, _ = self.get_best_ear_uniformity()
            best_spine, _ = self.get_best_spine_flatness()
            knee_genome, knee_fit = self.get_knee_point()
            knee_idx = self.get_best_balanced_index()
            knee_objectives = self.get_labeled_objectives(knee_idx)
            
            lines.extend([
                "  Best Ear Uniformity Solution:",
                f"    Plate: {best_ear.length:.2f}m x {best_ear.width:.2f}m",
                f"    Exciters: {len(best_ear.exciters)}",
                "",
                "  Knee Point (Best 6D Trade-off):",
                f"    Plate: {knee_genome.length:.2f}m x {knee_genome.width:.2f}m",
                f"    Total Fitness: {knee_fit.total_fitness:.3f}",
                "    Objectives:",
            ])
            for name, value in knee_objectives.items():
                lines.append(f"      {name:20s}: {value:.3f}")
        
        lines.append("═══════════════════════════════════════════════════════════════")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Add ear_uniformity to standard FitnessResult
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ear_uniformity_from_result(result: FitnessResult) -> float:
    """
    Compute ear L/R uniformity from a FitnessResult.
    
    Can be called externally to add ear_uniformity to existing results.
    
    Args:
        result: FitnessResult with head_response
    
    Returns:
        Uniformity score [0, 1]
    """
    if result.head_response is None:
        return 0.0
    
    head_resp = result.head_response
    
    if len(head_resp) < 2:
        return 0.5
    
    # Get left and right ear responses
    left_response = head_resp[0] if len(head_resp.shape) == 2 else head_resp[0:len(head_resp)//2]
    right_response = head_resp[1] if len(head_resp.shape) == 2 else head_resp[len(head_resp)//2:]
    
    if len(left_response.shape) > 1:
        left_response = np.mean(left_response, axis=0)
    if len(right_response.shape) > 1:
        right_response = np.mean(right_response, axis=0)
    
    # Compute uniformity
    left_rms = np.sqrt(np.mean(left_response**2))
    right_rms = np.sqrt(np.mean(right_response**2))
    
    if left_rms + right_rms < 1e-10:
        return 0.0
    
    min_rms = min(left_rms, right_rms)
    max_rms = max(left_rms, right_rms)
    
    if max_rms < 1e-10:
        return 0.0
    
    level_balance = min_rms / max_rms
    
    # Correlation
    if len(left_response) == len(right_response) and len(left_response) > 1:
        correlation = np.corrcoef(left_response, right_response)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        correlation = max(0.0, correlation)
    else:
        correlation = 0.5
    
    uniformity = 0.6 * level_balance + 0.4 * correlation
    
    return float(np.clip(uniformity, 0.0, 1.0))
