"""
Exciter Scorer - Exciter placement modal coupling evaluation.

Evaluates how well exciter positions couple to the plate's vibration modes.

Optimal placement:
- Exciters at mode antinodes maximize energy transfer
- Avoid nodal lines (zero coupling)
- Lower modes are more important (weighted by 1/(mode_idx+1))

Hardware: 4× Dayton DAEX25 (25mm, 40W, 8Ω) via JAB4 WONDOM
Reference: Bai & Liu 2004 - Genetic algorithm for exciter placement
"""

import numpy as np
from typing import Dict, Any, List

from .protocol import ScorerBase, ScorerResult


class ExciterScorer(ScorerBase):
    """
    Score exciter placement quality.
    
    Good placement = exciters at antinodes of important modes.
    
    Metrics:
    - Modal coupling (amplitude at exciter position)
    - Mode importance weighting (1/(mode_idx+1))
    
    Reference: Lu 2012 - Multi-exciter DML optimization
    """
    
    name = "exciter_placement"
    weight = 0.1  # Bonus score
    
    def __init__(self, weight: float = 0.1):
        """Initialize exciter scorer."""
        super().__init__(weight=weight)
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate exciter placement.
        
        Args:
            genome: PlateGenome with exciters
            context: Must contain:
                - 'frequencies': Modal frequencies
                - 'mode_shapes': Mode shape arrays (n_modes, nx, ny)
        
        Returns:
            ScorerResult with exciter coupling score
        """
        if not hasattr(genome, 'exciters') or not genome.exciters:
            return ScorerResult(
                score=0.5,
                name=self.name,
                weight=self.weight,
                details={'error': 'No exciters defined'}
            )
        
        frequencies = context.get('frequencies', [])
        mode_shapes = context.get('mode_shapes')
        
        if mode_shapes is None or len(frequencies) == 0:
            return ScorerResult(
                score=0.5,
                name=self.name,
                weight=self.weight,
                details={'error': 'No modal data'}
            )
        
        total_coupling = 0.0
        n_modes = min(len(frequencies), len(mode_shapes))
        per_exciter_coupling = []
        
        for mode_idx in range(n_modes):
            mode_shape = mode_shapes[mode_idx]
            nx, ny = mode_shape.shape
            
            mode_coupling = 0.0
            for exciter in genome.exciters:
                # Convert exciter position to grid indices
                ix = int(np.clip(exciter.x * nx, 0, nx - 1))
                iy = int(np.clip(exciter.y * ny, 0, ny - 1))
                
                # Mode amplitude at exciter position
                amplitude = np.abs(mode_shape[ix, iy])
                
                # Weight by mode importance (lower modes more important)
                mode_weight = 1.0 / (mode_idx + 1)
                
                mode_coupling += amplitude * mode_weight
            
            total_coupling += mode_coupling / len(genome.exciters)
        
        # Normalize to 0-1
        max_possible = sum(1.0 / (i + 1) for i in range(n_modes))
        score = total_coupling / max_possible if max_possible > 0 else 0.5
        
        return ScorerResult(
            score=float(np.clip(score, 0, 1)),
            name=self.name,
            weight=self.weight,
            details={
                'n_exciters': len(genome.exciters),
                'n_modes': n_modes,
                'total_coupling': float(total_coupling),
                'max_possible': float(max_possible),
            }
        )
