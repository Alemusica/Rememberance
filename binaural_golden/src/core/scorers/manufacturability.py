"""
Manufacturability Scorer - CNC/production feasibility evaluation.

Evaluates how easy/practical it is to manufacture the plate design.

Penalizes:
- Complex shapes (freeform contours)
- Too many cutouts
- Extreme thicknesses
- Extreme aspect ratios
- Plates too short for the person
"""

import numpy as np
from typing import Dict, Any

from .protocol import ScorerBase, ScorerResult


class ManufacturabilityScorer(ScorerBase):
    """
    Score manufacturability for CNC production.
    
    A manufacturable design:
    - Simple, standard shape
    - Reasonable number of cutouts
    - Standard thickness range (10-25mm)
    - Sensible aspect ratio (2:1 to 4:1)
    - Sufficient length for person
    """
    
    name = "manufacturability"
    weight = 0.5
    
    # Physical constraints
    MIN_THICKNESS_M = 0.010
    MAX_THICKNESS_M = 0.025
    MIN_ASPECT_RATIO = 2.0
    MAX_ASPECT_RATIO = 4.0
    MAX_CUTOUTS = 5
    
    def __init__(self, weight: float = 0.5):
        """Initialize manufacturability scorer."""
        super().__init__(weight=weight)
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate manufacturability.
        
        Args:
            genome: PlateGenome to evaluate
            context: Must contain:
                - 'person': Person model with recommended_plate_length
        
        Returns:
            ScorerResult with manufacturability score
        """
        score = 1.0
        penalties = {}
        
        person = context.get('person')
        
        # === CRITICAL: Plate too short for person ===
        if person and hasattr(person, 'recommended_plate_length'):
            min_length = person.recommended_plate_length
            if genome.length < min_length:
                deficit = min_length - genome.length
                deficit_ratio = deficit / min_length
                penalty = min(0.8, deficit_ratio * 2.0)
                score -= penalty
                penalties['length_deficit'] = penalty
        
        # Cutout penalty
        n_cuts = len(genome.cutouts) if hasattr(genome, 'cutouts') and genome.cutouts else 0
        if n_cuts > 0:
            cutout_penalty = min(0.5, 0.1 * n_cuts)
            score -= cutout_penalty
            penalties['cutouts'] = cutout_penalty
        
        # Freeform shape penalty
        if hasattr(genome, 'contour_type'):
            from ..plate_genome import ContourType
            if genome.contour_type == ContourType.FREEFORM:
                score -= 0.2
                penalties['freeform'] = 0.2
        
        # Thickness penalty
        h = genome.thickness_base
        if h < self.MIN_THICKNESS_M or h > self.MAX_THICKNESS_M:
            score -= 0.15
            penalties['thickness'] = 0.15
        
        # Aspect ratio penalty
        aspect = genome.length / genome.width if genome.width > 0 else 999
        if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
            score -= 0.1
            penalties['aspect_ratio'] = 0.1
        
        return ScorerResult(
            score=float(max(0, min(1, score))),
            name=self.name,
            weight=self.weight,
            details={
                'n_cutouts': n_cuts,
                'thickness_m': float(h),
                'aspect_ratio': float(aspect),
                'penalties': penalties,
            }
        )
