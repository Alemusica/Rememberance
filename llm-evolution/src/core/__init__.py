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
)

__all__ = [
    "Genome",
    "GeneSpec", 
    "GenomeSchema",
    "DictGenome",
    "random_genome",
]
