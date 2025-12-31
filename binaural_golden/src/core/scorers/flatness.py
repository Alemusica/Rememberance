"""
Zone Flatness Scorer - Frequency response flatness evaluation.

Evaluates how flat the frequency response is in each zone (spine, head).
Target: < 6dB variation for high-quality vibroacoustic therapy.
"""

import numpy as np
from typing import Dict, Any

from .protocol import ScorerBase, ScorerResult


class ZoneFlatnessScorer(ScorerBase):
    """
    Score frequency response flatness per zone.
    
    A flat response ensures consistent energy delivery across the
    therapeutic frequency range (20-200 Hz for spine, 200-2000 Hz for head).
    
    Metrics:
    - Peak-to-peak variation in dB (target: < 6 dB)
    - Spatial uniformity (all points respond equally)
    
    Reference: Pueo 2009 - Target ±3dB for high-quality binaural
    """
    
    name = "zone_flatness"
    weight = 1.0
    
    def __init__(
        self,
        target_variation_db: float = 6.0,
        weight: float = 1.0
    ):
        """
        Initialize flatness scorer.
        
        Args:
            target_variation_db: Target max peak-to-peak variation (default 6 dB)
            weight: Scorer weight in total fitness
        """
        super().__init__(weight=weight)
        self.target_variation_db = target_variation_db
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate frequency response flatness.
        
        Args:
            genome: PlateGenome (not used directly)
            context: Must contain:
                - 'spine_response': Array (n_points, n_freq)
                - 'head_response': Array (n_points, n_freq)
                - 'zone_weights': ZoneWeights (spine, head weights)
        
        Returns:
            ScorerResult with combined flatness score
        """
        spine_response = context.get('spine_response')
        head_response = context.get('head_response')
        zone_weights = context.get('zone_weights', {'spine': 0.7, 'head': 0.3})
        
        # Handle ZoneWeights dataclass or dict
        if hasattr(zone_weights, 'spine'):
            spine_weight = zone_weights.spine
            head_weight = zone_weights.head
        else:
            spine_weight = zone_weights.get('spine', 0.7)
            head_weight = zone_weights.get('head', 0.3)
        
        spine_score = self._score_zone(spine_response)
        head_score = self._score_zone(head_response)
        
        # Weighted combination
        combined_score = spine_weight * spine_score + head_weight * head_score
        
        return ScorerResult(
            score=combined_score,
            name=self.name,
            weight=self.weight,
            details={
                'spine_flatness': spine_score,
                'head_flatness': head_score,
                'spine_weight': spine_weight,
                'head_weight': head_weight,
            }
        )
    
    def _score_zone(self, zone_response: np.ndarray) -> float:
        """
        Score flatness for a single zone.
        
        Args:
            zone_response: Array (n_points, n_freq) with frequency response
            
        Returns:
            Score [0, 1] where 1.0 = perfectly flat
        """
        if zone_response is None or zone_response.size == 0:
            return 0.0
        
        # Convert to dB (with floor to avoid -inf)
        response_db = 20 * np.log10(np.maximum(zone_response, 1e-10))
        
        # Average response across all points in zone
        mean_response_db = np.mean(response_db, axis=0)
        
        # Peak-to-peak variation
        peak_to_peak = np.max(mean_response_db) - np.min(mean_response_db)
        
        # Base score from variation (< target = 1.0)
        score = np.clip(1 - peak_to_peak / (2 * self.target_variation_db), 0, 1)
        
        # Bonus for spatial uniformity (all points respond equally)
        spatial_std = np.std(response_db, axis=0).mean()
        uniformity_bonus = np.clip(1 - spatial_std / 10, 0, 0.2)
        
        return float(min(score + uniformity_bonus, 1.0))
