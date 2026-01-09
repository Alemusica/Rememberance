"""
PyMoo Multi-Objective Optimization Tools.

Wrapper per algoritmi evolutivi multi-obiettivo da PyMoo:
- NSGA-II: Non-dominated Sorting Genetic Algorithm II
- NSGA-III: NSGA-II con reference directions per many-objective
- MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition
"""

from typing import Callable, Dict, List, Any
import time
from ..registry import tool

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    import numpy as np
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


class PyMooProblem(Problem):
    """
    Wrapper per rendere una funzione Python compatibile con PyMoo.
    """
    
    def __init__(self, n_var: int, n_obj: int, xl: List[float], xu: List[float], 
                 evaluate_fn: Callable):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=np.array(xl),
            xu=np.array(xu),
        )
        self.evaluate_fn = evaluate_fn
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Valuta batch di soluzioni.
        """
        objectives = []
        for individual in x:
            obj = self.evaluate_fn(individual.tolist())
            objectives.append(obj)
        out["F"] = np.array(objectives)


@tool(
    name="pymoo_nsga2",
    description="Multi-objective NSGA-II optimization with non-dominated sorting",
    domain="evolution",
    cost="high",
    input_schema={
        "n_var": {"type": "int", "description": "Number of decision variables"},
        "n_obj": {"type": "int", "description": "Number of objectives"},
        "xl": {"type": "List[float]", "description": "Lower bounds"},
        "xu": {"type": "List[float]", "description": "Upper bounds"},
        "evaluate_fn": {"type": "Callable", "description": "Fitness function"},
        "pop_size": {"type": "int", "description": "Population size", "default": 50},
        "n_gen": {"type": "int", "description": "Number of generations", "default": 100},
    },
    output_schema={
        "pareto_F": "List[List[float]]",
        "pareto_X": "List[List[float]]",
        "n_evals": "int",
        "time_seconds": "float",
    }
)
def pymoo_nsga2_tool(
    n_var: int,
    n_obj: int,
    xl: List[float],
    xu: List[float],
    evaluate_fn: Callable,
    pop_size: int = 50,
    n_gen: int = 100,
) -> Dict[str, Any]:
    """
    Execute NSGA-II multi-objective optimization.
    
    NSGA-II uses non-dominated sorting and crowding distance to maintain
    diversity along the Pareto front.
    
    Args:
        n_var: Number of decision variables
        n_obj: Number of objectives to optimize
        xl: Lower bounds for each variable
        xu: Upper bounds for each variable
        evaluate_fn: Function that takes a list of variables and returns
                     a list of objective values
        pop_size: Population size
        n_gen: Number of generations
    
    Returns:
        Dictionary with Pareto front solutions and statistics
    """
    if not PYMOO_AVAILABLE:
        return {
            "pareto_F": [],
            "pareto_X": [],
            "n_evals": 0,
            "time_seconds": 0.0,
            "error": "PyMoo not installed. Install with: pip install pymoo"
        }
    
    start_time = time.time()
    
    problem = PyMooProblem(n_var, n_obj, xl, xu, evaluate_fn)
    
    algorithm = NSGA2(pop_size=pop_size)
    
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=False,
        seed=None,
    )
    
    elapsed = time.time() - start_time
    
    pareto_F = result.F.tolist() if result.F is not None else []
    pareto_X = result.X.tolist() if result.X is not None else []
    
    if len(pareto_F) > 0 and not isinstance(pareto_F[0], list):
        pareto_F = [[f] for f in pareto_F]
    if len(pareto_X) > 0 and not isinstance(pareto_X[0], list):
        pareto_X = [[x] for x in pareto_X]
    
    return {
        "pareto_F": pareto_F,
        "pareto_X": pareto_X,
        "n_evals": result.algorithm.evaluator.n_eval,
        "time_seconds": elapsed,
    }


@tool(
    name="pymoo_nsga3",
    description="Multi-objective NSGA-III optimization with reference directions for many-objective problems",
    domain="evolution",
    cost="high",
    input_schema={
        "n_var": {"type": "int", "description": "Number of decision variables"},
        "n_obj": {"type": "int", "description": "Number of objectives"},
        "xl": {"type": "List[float]", "description": "Lower bounds"},
        "xu": {"type": "List[float]", "description": "Upper bounds"},
        "evaluate_fn": {"type": "Callable", "description": "Fitness function"},
        "pop_size": {"type": "int", "description": "Population size", "default": 50},
        "n_gen": {"type": "int", "description": "Number of generations", "default": 100},
    },
    output_schema={
        "pareto_F": "List[List[float]]",
        "pareto_X": "List[List[float]]",
        "n_evals": "int",
        "time_seconds": "float",
    }
)
def pymoo_nsga3_tool(
    n_var: int,
    n_obj: int,
    xl: List[float],
    xu: List[float],
    evaluate_fn: Callable,
    pop_size: int = 50,
    n_gen: int = 100,
) -> Dict[str, Any]:
    """
    Execute NSGA-III multi-objective optimization.
    
    NSGA-III extends NSGA-II for many-objective problems using reference
    directions to maintain diversity.
    
    Args:
        n_var: Number of decision variables
        n_obj: Number of objectives to optimize
        xl: Lower bounds for each variable
        xu: Upper bounds for each variable
        evaluate_fn: Function that takes a list of variables and returns
                     a list of objective values
        pop_size: Population size
        n_gen: Number of generations
    
    Returns:
        Dictionary with Pareto front solutions and statistics
    """
    if not PYMOO_AVAILABLE:
        return {
            "pareto_F": [],
            "pareto_X": [],
            "n_evals": 0,
            "time_seconds": 0.0,
            "error": "PyMoo not installed. Install with: pip install pymoo"
        }
    
    start_time = time.time()
    
    problem = PyMooProblem(n_var, n_obj, xl, xu, evaluate_fn)
    
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=False,
        seed=None,
    )
    
    elapsed = time.time() - start_time
    
    pareto_F = result.F.tolist() if result.F is not None else []
    pareto_X = result.X.tolist() if result.X is not None else []
    
    if len(pareto_F) > 0 and not isinstance(pareto_F[0], list):
        pareto_F = [[f] for f in pareto_F]
    if len(pareto_X) > 0 and not isinstance(pareto_X[0], list):
        pareto_X = [[x] for x in pareto_X]
    
    return {
        "pareto_F": pareto_F,
        "pareto_X": pareto_X,
        "n_evals": result.algorithm.evaluator.n_eval,
        "time_seconds": elapsed,
    }


@tool(
    name="pymoo_moead",
    description="Multi-objective MOEA/D optimization using decomposition approach",
    domain="evolution",
    cost="high",
    input_schema={
        "n_var": {"type": "int", "description": "Number of decision variables"},
        "n_obj": {"type": "int", "description": "Number of objectives"},
        "xl": {"type": "List[float]", "description": "Lower bounds"},
        "xu": {"type": "List[float]", "description": "Upper bounds"},
        "evaluate_fn": {"type": "Callable", "description": "Fitness function"},
        "pop_size": {"type": "int", "description": "Population size", "default": 50},
        "n_gen": {"type": "int", "description": "Number of generations", "default": 100},
    },
    output_schema={
        "pareto_F": "List[List[float]]",
        "pareto_X": "List[List[float]]",
        "n_evals": "int",
        "time_seconds": "float",
    }
)
def pymoo_moead_tool(
    n_var: int,
    n_obj: int,
    xl: List[float],
    xu: List[float],
    evaluate_fn: Callable,
    pop_size: int = 50,
    n_gen: int = 100,
) -> Dict[str, Any]:
    """
    Execute MOEA/D multi-objective optimization.
    
    MOEA/D decomposes the multi-objective problem into scalar subproblems
    and optimizes them simultaneously using neighborhood information.
    
    Args:
        n_var: Number of decision variables
        n_obj: Number of objectives to optimize
        xl: Lower bounds for each variable
        xu: Upper bounds for each variable
        evaluate_fn: Function that takes a list of variables and returns
                     a list of objective values
        pop_size: Population size
        n_gen: Number of generations
    
    Returns:
        Dictionary with Pareto front solutions and statistics
    """
    if not PYMOO_AVAILABLE:
        return {
            "pareto_F": [],
            "pareto_X": [],
            "n_evals": 0,
            "time_seconds": 0.0,
            "error": "PyMoo not installed. Install with: pip install pymoo"
        }
    
    start_time = time.time()
    
    problem = PyMooProblem(n_var, n_obj, xl, xu, evaluate_fn)
    
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=False,
        seed=None,
    )
    
    elapsed = time.time() - start_time
    
    pareto_F = result.F.tolist() if result.F is not None else []
    pareto_X = result.X.tolist() if result.X is not None else []
    
    if len(pareto_F) > 0 and not isinstance(pareto_F[0], list):
        pareto_F = [[f] for f in pareto_F]
    if len(pareto_X) > 0 and not isinstance(pareto_X[0], list):
        pareto_X = [[x] for x in pareto_X]
    
    return {
        "pareto_F": pareto_F,
        "pareto_X": pareto_X,
        "n_evals": result.algorithm.evaluator.n_eval,
        "time_seconds": elapsed,
    }
