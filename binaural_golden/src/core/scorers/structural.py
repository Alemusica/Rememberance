"""
Structural Scorer - Deflection and safety evaluation.

CRITICAL SAFETY CHECK: The plate MUST support the person without
excessive deflection or structural failure.

Limits:
- Max deflection: 10mm (comfort + stability)
- Min safety factor: 2.0 (engineering standard)

Also includes peninsula detection for ABH (Acoustic Black Hole) benefit.
Reference: Krylov 2014, Deng 2019, Zhao 2014
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from .protocol import ScorerBase, ScorerResult

logger = logging.getLogger(__name__)


class StructuralScorer(ScorerBase):
    """
    Score structural integrity under person weight.
    
    Uses StructuralAnalyzer to calculate:
    - Max deflection under distributed body load
    - Stress concentration at cutout edges
    - Safety factor vs yield stress
    - Peninsula ABH benefit (DON'T PENALIZE beneficial peninsulas!)
    
    Formula: score = 0.7 * deflection_score + 0.2 * safety_score + 0.1 * peninsula_net
    """
    
    name = "structural"
    weight = 1.0
    
    # Physical limits
    MAX_DEFLECTION_MM = 10.0
    MIN_SAFETY_FACTOR = 2.0
    
    def __init__(
        self,
        max_deflection_mm: float = 10.0,
        min_safety_factor: float = 2.0,
        weight: float = 1.0
    ):
        """
        Initialize structural scorer.
        
        Args:
            max_deflection_mm: Maximum acceptable deflection
            min_safety_factor: Minimum stress safety factor
            weight: Scorer weight in total fitness
        """
        super().__init__(weight=weight)
        self.MAX_DEFLECTION_MM = max_deflection_mm
        self.MIN_SAFETY_FACTOR = min_safety_factor
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate structural integrity.
        
        Args:
            genome: PlateGenome to evaluate
            context: Must contain:
                - 'person': Person model with weight_kg
                - 'material': Material properties
                - Optional 'structural_result': Pre-computed deflection
        
        Returns:
            ScorerResult with structural safety score
        """
        person = context.get('person')
        material = context.get('material')
        
        # Try to use pre-computed result if available
        pre_computed = context.get('structural_result')
        if pre_computed:
            return self._score_from_precomputed(pre_computed)
        
        # Otherwise compute deflection
        try:
            from ..structural_analysis import StructuralAnalyzer
            
            analyzer = StructuralAnalyzer(
                length=genome.length,
                width=genome.width,
                thickness=genome.thickness_base,
                material=material.name if hasattr(material, 'name') else "birch_plywood",
                E_modulus=material.E_longitudinal if hasattr(material, 'E_longitudinal') else 13e9,
                poisson=0.33,
            )
            
            # Set grooves if present
            if hasattr(genome, 'grooves') and genome.grooves:
                analyzer.set_grooves(genome.grooves)
            
            # Prepare cutouts
            cutouts_for_fem = self._prepare_cutouts(genome)
            
            # Get person weight
            person_weight = 80.0  # Default
            if person and hasattr(person, 'weight_kg'):
                person_weight = person.weight_kg
            
            # Calculate deflection
            defl_result = analyzer.calculate_deflection(
                person_weight_kg=person_weight,
                cutouts=cutouts_for_fem if cutouts_for_fem else None,
                resolution=30,
                use_fem=False,
            )
            
            # Calculate scores
            defl_score = self._deflection_score(defl_result.max_deflection_mm)
            safety_score = 1.0  # Assume safe unless calculated
            peninsula_net = 0.0
            
            # Calculate stress and peninsula if cutouts present
            if cutouts_for_fem:
                try:
                    stress_result = analyzer.calculate_stress(
                        person_weight_kg=person_weight,
                        cutouts=cutouts_for_fem,
                        resolution=30,
                    )
                    safety_score = self._safety_factor_score(stress_result.safety_factor)
                except Exception as e:
                    logger.debug(f"Stress calculation failed: {e}")
                
                # Peninsula detection
                peninsula_net = self._detect_peninsula_benefit(genome, analyzer)
            
            # Combined score
            final_score = (
                0.7 * defl_score +
                0.2 * safety_score +
                0.1 * (peninsula_net + 1) / 2  # Normalize peninsula_net to [0, 1]
            )
            
            return ScorerResult(
                score=float(np.clip(final_score, 0, 1)),
                name=self.name,
                weight=self.weight,
                details={
                    'max_deflection_mm': float(defl_result.max_deflection_mm),
                    'deflection_is_safe': defl_result.is_acceptable,
                    'deflection_score': float(defl_score),
                    'safety_factor_score': float(safety_score),
                    'peninsula_net_score': float(peninsula_net),
                }
            )
            
        except ImportError:
            logger.warning("StructuralAnalyzer not available, using default score")
            return ScorerResult(
                score=0.5,
                name=self.name,
                weight=self.weight,
                details={'error': 'StructuralAnalyzer not available'}
            )
        except Exception as e:
            logger.error(f"Structural analysis failed: {e}")
            return ScorerResult(
                score=0.5,
                name=self.name,
                weight=self.weight,
                details={'error': str(e)}
            )
    
    def _deflection_score(self, deflection_mm: float) -> float:
        """Score from deflection (1.0 = no deflection, 0.0 = dangerous)."""
        if deflection_mm <= self.MAX_DEFLECTION_MM:
            return 1.0 - (deflection_mm / self.MAX_DEFLECTION_MM) * 0.3
        else:
            excess = deflection_mm - self.MAX_DEFLECTION_MM
            return max(0.0, 0.7 - min(excess / 5.0, 0.7))
    
    def _safety_factor_score(self, safety_factor: float) -> float:
        """Score from safety factor (1.0 = safe, 0.0 = at yield)."""
        if safety_factor >= self.MIN_SAFETY_FACTOR:
            return 1.0
        return safety_factor / self.MIN_SAFETY_FACTOR
    
    def _prepare_cutouts(self, genome: Any) -> list:
        """Convert genome cutouts to FEM format."""
        if not hasattr(genome, 'cutouts') or not genome.cutouts:
            return []
        
        cutouts = []
        for cut in genome.cutouts:
            cx = cut.x * genome.length
            cy = cut.y * genome.width
            r_equiv = (cut.width * genome.length + cut.height * genome.width) / 4
            cutouts.append((cx, cy, r_equiv))
        return cutouts
    
    def _detect_peninsula_benefit(self, genome: Any, analyzer: Any) -> float:
        """
        Detect if peninsulas can act as ABH (Acoustic Black Holes).
        
        DON'T PENALIZE beneficial peninsulas!
        
        Formula: abh_benefit * 0.6 + resonator_potential * 0.3 - structural_penalty * 0.5
        
        Returns:
            Net score [-1, 1] where positive = beneficial peninsula
        """
        try:
            if hasattr(analyzer, 'detect_peninsulas'):
                peninsulas = analyzer.detect_peninsulas()
                if not peninsulas:
                    return 0.0
                
                total_net = 0.0
                for pen in peninsulas:
                    abh_benefit = pen.get('abh_benefit', 0.0)
                    resonator = pen.get('resonator_potential', 0.0)
                    structural = pen.get('structural_penalty', 0.0)
                    
                    net = abh_benefit * 0.6 + resonator * 0.3 - structural * 0.5
                    total_net += net
                
                return float(np.clip(total_net / len(peninsulas), -1, 1))
        except Exception:
            pass
        return 0.0
    
    def _score_from_precomputed(self, result: Any) -> ScorerResult:
        """Create score from pre-computed structural result."""
        defl_score = self._deflection_score(result.max_deflection_mm)
        safety_score = self._safety_factor_score(
            getattr(result, 'stress_safety_factor', 2.0)
        )
        
        final_score = 0.8 * defl_score + 0.2 * safety_score
        
        return ScorerResult(
            score=float(np.clip(final_score, 0, 1)),
            name=self.name,
            weight=self.weight,
            details={
                'max_deflection_mm': float(result.max_deflection_mm),
                'deflection_is_safe': result.is_acceptable,
                'precomputed': True,
            }
        )
