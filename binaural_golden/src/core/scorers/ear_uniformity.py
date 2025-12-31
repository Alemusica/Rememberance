"""
Ear Uniformity Scorer - Left/Right binaural balance evaluation.

CRITICAL metric for proper binaural audio reproduction.
L/R imbalance causes localization errors and reduces therapy effectiveness.

Target: > 90% uniformity for therapeutic applications.
"""

import numpy as np
from typing import Dict, Any

from .protocol import ScorerBase, ScorerResult


class EarUniformityScorer(ScorerBase):
    """
    Score Left/Right ear uniformity for binaural balance.
    
    Metrics (weighted):
    - RMS Level Balance (50%): L and R have same overall energy
    - Frequency Correlation (25%): L and R have same spectral shape
    - Spectral Match (25%): Per-frequency difference < 3 dB
    
    Reference: Clinical requirement > 90% for binaural therapy
    """
    
    name = "ear_uniformity"
    weight = 0.4  # High weight - critical for binaural!
    
    def __init__(
        self,
        target_diff_db: float = 3.0,
        weight: float = 0.4
    ):
        """
        Initialize ear uniformity scorer.
        
        Args:
            target_diff_db: Target max L/R difference per frequency (default 3 dB)
            weight: Scorer weight in total fitness
        """
        super().__init__(weight=weight)
        self.target_diff_db = target_diff_db
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate L/R ear uniformity.
        
        Args:
            genome: PlateGenome (not used directly)
            context: Must contain:
                - 'head_response': Array (n_positions, n_freq) where
                  first 2 positions are left and right ears
        
        Returns:
            ScorerResult with uniformity score and breakdown
        """
        head_response = context.get('head_response')
        
        if head_response is None or len(head_response) < 2:
            return ScorerResult(
                score=0.0,
                name=self.name,
                weight=self.weight,
                details={'error': 'Insufficient head response data'}
            )
        
        # Get left and right ear responses (first two positions)
        left_ear = head_response[0]
        right_ear = head_response[1]
        
        # Ensure 1D arrays
        if len(left_ear.shape) > 1:
            left_ear = np.mean(left_ear, axis=0)
        if len(right_ear.shape) > 1:
            right_ear = np.mean(right_ear, axis=0)
        
        # Calculate metrics
        level_balance = self._calculate_level_balance(left_ear, right_ear)
        correlation = self._calculate_correlation(left_ear, right_ear)
        spectral_match = self._calculate_spectral_match(left_ear, right_ear)
        
        # Combined score: 50% level + 25% correlation + 25% spectral
        uniformity = 0.50 * level_balance + 0.25 * correlation + 0.25 * spectral_match
        
        return ScorerResult(
            score=float(np.clip(uniformity, 0.0, 1.0)),
            name=self.name,
            weight=self.weight,
            details={
                'level_balance': level_balance,
                'correlation': correlation,
                'spectral_match': spectral_match,
                'left_rms': float(np.sqrt(np.mean(left_ear**2))),
                'right_rms': float(np.sqrt(np.mean(right_ear**2))),
            }
        )
    
    def _calculate_level_balance(
        self,
        left_ear: np.ndarray,
        right_ear: np.ndarray
    ) -> float:
        """Calculate RMS level balance (1.0 = perfect)."""
        left_rms = np.sqrt(np.mean(left_ear**2))
        right_rms = np.sqrt(np.mean(right_ear**2))
        
        if left_rms + right_rms < 1e-10:
            return 0.0
        
        min_rms = min(left_rms, right_rms)
        max_rms = max(left_rms, right_rms)
        
        if max_rms < 1e-10:
            return 0.0
        
        return float(min_rms / max_rms)
    
    def _calculate_correlation(
        self,
        left_ear: np.ndarray,
        right_ear: np.ndarray
    ) -> float:
        """Calculate Pearson correlation (1.0 = identical shape)."""
        if len(left_ear) != len(right_ear) or len(left_ear) < 3:
            return 0.5
        
        try:
            correlation = np.corrcoef(left_ear, right_ear)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return float(max(0.0, correlation))
        except Exception:
            return 0.5
    
    def _calculate_spectral_match(
        self,
        left_ear: np.ndarray,
        right_ear: np.ndarray
    ) -> float:
        """Calculate frequency-by-frequency match (1.0 = < target_diff_db everywhere)."""
        if len(left_ear) != len(right_ear) or len(left_ear) < 2:
            return 0.5
        
        # Calculate dB difference per frequency
        diff_db = 20 * np.log10(
            (np.abs(left_ear - right_ear) + 1e-10) /
            (np.maximum(left_ear, right_ear) + 1e-10)
        )
        mean_diff_db = np.mean(np.abs(diff_db))
        
        # Score: 1.0 if mean diff < target, decreasing linearly
        return float(np.clip(1 - mean_diff_db / (2 * self.target_diff_db), 0, 1))
