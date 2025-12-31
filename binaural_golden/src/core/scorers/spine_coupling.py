"""
Spine Coupling Scorer - Vibroacoustic coupling at spine evaluation.

Evaluates how well the plate couples vibrational energy to the spine zone.
Higher coupling = better therapeutic effect for VAT (Vibroacoustic Therapy).

Target: Uniform, high-level response across all spine points.
"""

import numpy as np
from typing import Dict, Any

from .protocol import ScorerBase, ScorerResult


class SpineCouplingScorer(ScorerBase):
    """
    Score vibroacoustic coupling at spine.
    
    For effective VAT:
    - High average response level (energy transfer)
    - Low variation across spine length (uniformity)
    
    Combined metric: 60% level + 40% uniformity
    
    Reference: Clinical VAT studies require uniform spine stimulation
    """
    
    name = "spine_coupling"
    weight = 2.0  # High priority for VAT applications
    
    def __init__(self, weight: float = 2.0):
        """
        Initialize spine coupling scorer.
        
        Args:
            weight: Scorer weight in total fitness
        """
        super().__init__(weight=weight)
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate spine coupling.
        
        Args:
            genome: PlateGenome (not used directly)
            context: Must contain:
                - 'spine_response': Array (n_points, n_freq) with spine response
        
        Returns:
            ScorerResult with coupling score
        """
        spine_response = context.get('spine_response')
        
        if spine_response is None or spine_response.size == 0:
            return ScorerResult(
                score=0.0,
                name=self.name,
                weight=self.weight,
                details={'error': 'No spine response data'}
            )
        
        # Average response across all points and frequencies
        mean_response = np.mean(spine_response)
        
        # Coefficient of variation (lower = more uniform)
        cv = np.std(spine_response) / (mean_response + 1e-10)
        uniformity = np.clip(1 - cv, 0, 1)
        
        # Level score (arbitrary scaling to [0, 1])
        level_score = np.clip(mean_response * 2, 0, 1)
        
        # Combined score
        coupling_score = 0.6 * level_score + 0.4 * uniformity
        
        return ScorerResult(
            score=float(coupling_score),
            name=self.name,
            weight=self.weight,
            details={
                'mean_response': float(mean_response),
                'uniformity': float(uniformity),
                'level_score': float(level_score),
                'coefficient_of_variation': float(cv),
            }
        )
