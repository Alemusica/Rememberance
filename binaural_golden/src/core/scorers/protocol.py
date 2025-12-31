"""
Scorer Protocol - Interface for all fitness scorers.

All scorers must implement this protocol to be used by FitnessEvaluator.
"""

from typing import Protocol, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ScorerResult:
    """Result from a scorer evaluation."""
    score: float  # Main score [0, 1] where 1 is best
    name: str  # Scorer name for logging
    details: Dict[str, Any] = field(default_factory=dict)  # Additional metrics
    weight: float = 1.0  # Weight in total fitness
    
    def weighted_score(self) -> float:
        """Return score multiplied by weight."""
        return self.score * self.weight


class Scorer(Protocol):
    """
    Protocol for fitness scorers.
    
    Each scorer evaluates ONE aspect of a genome's fitness.
    Scorers are composable - FitnessEvaluator uses a list of scorers.
    
    Example implementation:
        class MyScorer:
            name = "my_scorer"
            weight = 0.5
            
            def score(self, genome, context) -> ScorerResult:
                # Calculate score
                return ScorerResult(score=0.8, name=self.name, weight=self.weight)
    """
    
    name: str
    weight: float
    
    def score(
        self,
        genome: Any,  # PlateGenome
        context: Dict[str, Any]  # Shared computation results
    ) -> ScorerResult:
        """
        Evaluate genome and return score.
        
        Args:
            genome: PlateGenome to evaluate
            context: Shared context with pre-computed values:
                - 'frequencies': Modal frequencies
                - 'mode_shapes': Mode shape arrays
                - 'spine_response': Frequency response at spine
                - 'head_response': Frequency response at head
                - 'person': Person model
                - 'material': Material properties
                
        Returns:
            ScorerResult with score and optional details
        """
        ...


class ScorerBase:
    """
    Base class for scorers with common utilities.
    
    Provides helper methods used by multiple scorers.
    """
    
    name: str = "base"
    weight: float = 1.0
    
    def __init__(self, weight: Optional[float] = None):
        if weight is not None:
            self.weight = weight
    
    def _safe_normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range safely."""
        if max_val <= min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def _coefficient_of_variation(self, data: np.ndarray) -> float:
        """Calculate coefficient of variation (std/mean)."""
        if len(data) == 0:
            return 1.0
        mean = np.mean(data)
        if mean == 0:
            return 1.0
        return np.std(data) / mean
    
    def _db_to_linear(self, db: float) -> float:
        """Convert decibels to linear scale."""
        return 10 ** (db / 20)
    
    def _linear_to_db(self, linear: float) -> float:
        """Convert linear to decibels."""
        if linear <= 0:
            return -100.0
        return 20 * np.log10(linear)
