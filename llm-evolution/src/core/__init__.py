"""
LLM-Evolution Core - Agnostic Evolutionary Optimization

This module provides domain-agnostic evolutionary algorithms
supervised by LLM agents.
"""

from .genome import (
    Genome,
    GeneSpec,
    GenomeSchema,
    DictGenome,
    random_genome,
    crossover_uniform,
    crossover_blend,
    mutate_gaussian,
)
from .coordinator import (
    EvolutionState,
    EvolutionResult,
    EvolutionCoordinator,
)

__all__ = [
    # Genome
    "Genome",
    "GeneSpec", 
    "GenomeSchema",
    "DictGenome",
    "random_genome",
    "crossover_uniform",
    "crossover_blend",
    "mutate_gaussian",
    # Coordinator
    "EvolutionState",
    "EvolutionResult",
    "EvolutionCoordinator",
]
