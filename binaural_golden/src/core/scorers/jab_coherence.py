"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     JAB (Jitter-Aware Binaural) Phase Coherence Scorer                       ║
║                                                                              ║
║     Evaluates phase coherence between L/R exciter pairs for optimal          ║
║     binaural reproduction. Critical for ear_uniformity > 90%.                ║
║                                                                              ║
║     JAB4 Hardware Channel Mapping:                                           ║
║     ┌─────────┬─────────┐                                                    ║
║     │  CH1 L  │  CH2 R  │  ← HEAD (ears)                                     ║
║     │  (y>0.6)│  (y>0.6)│                                                    ║
║     ├─────────┼─────────┤                                                    ║
║     │  CH3 L  │  CH4 R  │  ← FEET (spine base)                               ║
║     │  (y<0.4)│  (y<0.4)│                                                    ║
║     └─────────┴─────────┘                                                    ║
║                                                                              ║
║     Phase Coherence Rules:                                                   ║
║     1. SAME-SIDE COHERENCE: CH1↔CH3 (L) and CH2↔CH4 (R) should have         ║
║        similar phases for constructive interference on same ear              ║
║     2. L/R BALANCE: CH1+CH3 vs CH2+CH4 phase difference controls            ║
║        stereo imaging and binaural beat generation                           ║
║     3. DELAY ALIGNMENT: Same-side channels should have minimal delay         ║
║        difference for coherent wavefront arrival at ear                      ║
║                                                                              ║
║     Created: 2025-01-04                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List

from .protocol import ScorerBase, ScorerResult

# Logger per debug - configurabile via logging.getLogger(__name__).setLevel()
logger = logging.getLogger(__name__)


class JABCoherenceScorer(ScorerBase):
    """
    Score phase coherence for Jitter-Aware Binaural optimization.
    
    Metrics (weighted):
    - Same-Side Phase Coherence (40%): CH1↔CH3 and CH2↔CH4 aligned
    - L/R Phase Balance (30%): Symmetric or intentional binaural offset
    - Delay Alignment (20%): Same-side delay matching
    - Gain Balance (10%): L/R gain matching for stereo image
    
    Higher score = better binaural reproduction quality.
    """
    
    name = "jab_coherence"
    weight = 0.25  # Important but not dominant
    
    def __init__(
        self,
        target_phase_diff_deg: float = 30.0,
        target_delay_diff_ms: float = 5.0,
        binaural_mode: str = "coherent",  # or "binaural_beat"
        weight: float = 0.25
    ):
        """
        Initialize JAB coherence scorer.
        
        Args:
            target_phase_diff_deg: Max acceptable phase diff between same-side channels
            target_delay_diff_ms: Max acceptable delay diff between same-side channels
            binaural_mode: 
                - "coherent": L/R should be in-phase (mono-compatible)
                - "binaural_beat": L/R should have controlled offset for beats
            weight: Scorer weight in total fitness
        """
        super().__init__(weight=weight)
        self.target_phase_diff_deg = target_phase_diff_deg
        self.target_delay_diff_ms = target_delay_diff_ms
        self.binaural_mode = binaural_mode
    
    def score(
        self,
        genome: Any,
        context: Dict[str, Any]
    ) -> ScorerResult:
        """
        Evaluate JAB phase coherence.
        
        Args:
            genome: PlateGenome with exciters
            context: Additional context (optional frequency info)
        
        Returns:
            ScorerResult with JAB coherence score and detailed breakdown
        """
        if not hasattr(genome, 'exciters') or len(genome.exciters) < 2:
            logger.debug("JAB: insufficient exciters (%d), returning neutral",
                        len(getattr(genome, 'exciters', [])))
            return ScorerResult(
                score=0.5,  # Neutral if no exciters
                name=self.name,
                weight=self.weight,
                details={'error': 'Insufficient exciters for JAB scoring'}
            )
        
        # Group exciters by channel
        exciters = genome.exciters
        ch_map = self._group_by_channel(exciters)
        logger.debug("JAB: grouped %d exciters into %d channels", 
                    len(exciters), len([k for k in ch_map if ch_map[k]]))
        
        # Calculate each metric
        same_side_coherence = self._score_same_side_coherence(ch_map)
        lr_balance = self._score_lr_balance(ch_map)
        delay_alignment = self._score_delay_alignment(ch_map)
        gain_balance = self._score_gain_balance(ch_map)
        
        logger.debug("JAB metrics: same_side=%.3f, lr=%.3f, delay=%.3f, gain=%.3f",
                    same_side_coherence, lr_balance, delay_alignment, gain_balance)
        
        # Combined score: 40% same-side + 30% L/R + 20% delay + 10% gain
        total_score = (
            0.40 * same_side_coherence +
            0.30 * lr_balance +
            0.20 * delay_alignment +
            0.10 * gain_balance
        )
        
        logger.info("JAB coherence score: %.3f (mode=%s)", total_score, self.binaural_mode)
        
        return ScorerResult(
            score=float(np.clip(total_score, 0.0, 1.0)),
            name=self.name,
            weight=self.weight,
            details={
                'same_side_coherence': same_side_coherence,
                'lr_balance': lr_balance,
                'delay_alignment': delay_alignment,
                'gain_balance': gain_balance,
                'binaural_mode': self.binaural_mode,
                'channel_phases': {
                    f'CH{ch}': self._get_channel_phase(ch_map, ch)
                    for ch in [1, 2, 3, 4]
                },
            }
        )
    
    def _group_by_channel(self, exciters: List[Any]) -> Dict[int, Any]:
        """Group exciters by JAB4 channel (1-4)."""
        ch_map = {}
        for exc in exciters:
            ch = getattr(exc, 'channel', 1)
            if ch not in ch_map:
                ch_map[ch] = exc
        return ch_map
    
    def _get_channel_phase(self, ch_map: Dict[int, Any], channel: int) -> float:
        """Get phase of a channel (0 if missing)."""
        if channel not in ch_map:
            return 0.0
        return getattr(ch_map[channel], 'phase_deg', 0.0)
    
    def _get_channel_delay(self, ch_map: Dict[int, Any], channel: int) -> float:
        """Get delay of a channel (0 if missing)."""
        if channel not in ch_map:
            return 0.0
        return getattr(ch_map[channel], 'delay_ms', 0.0)
    
    def _get_channel_gain(self, ch_map: Dict[int, Any], channel: int) -> float:
        """Get gain of a channel (0 dB if missing)."""
        if channel not in ch_map:
            return 0.0
        return getattr(ch_map[channel], 'gain_db', 0.0)
    
    def _score_same_side_coherence(self, ch_map: Dict[int, Any]) -> float:
        """
        Score phase coherence between same-side channels.
        
        CH1 ↔ CH3 (Left side): should be in-phase for coherent L ear signal
        CH2 ↔ CH4 (Right side): should be in-phase for coherent R ear signal
        
        Returns:
            Score [0, 1] where 1.0 = perfect same-side coherence
        """
        # Left side: CH1 ↔ CH3
        phase_1 = self._get_channel_phase(ch_map, 1)
        phase_3 = self._get_channel_phase(ch_map, 3)
        left_diff = self._phase_difference(phase_1, phase_3)
        
        # Right side: CH2 ↔ CH4
        phase_2 = self._get_channel_phase(ch_map, 2)
        phase_4 = self._get_channel_phase(ch_map, 4)
        right_diff = self._phase_difference(phase_2, phase_4)
        
        # Score: 1.0 if diff < target, decreases linearly to 0 at 180°
        left_score = 1.0 - min(left_diff / 180.0, 1.0)
        right_score = 1.0 - min(right_diff / 180.0, 1.0)
        
        # Bonus for being within target
        if left_diff < self.target_phase_diff_deg:
            left_score = min(left_score + 0.2, 1.0)
        if right_diff < self.target_phase_diff_deg:
            right_score = min(right_score + 0.2, 1.0)
        
        return (left_score + right_score) / 2.0
    
    def _score_lr_balance(self, ch_map: Dict[int, Any]) -> float:
        """
        Score L/R phase balance based on binaural mode.
        
        "coherent" mode: L and R should be in-phase (stereo imaging)
        "binaural_beat" mode: L/R can have controlled offset
        
        Returns:
            Score [0, 1] where 1.0 = optimal L/R relationship
        """
        # Effective L phase = average of CH1 + CH3
        phase_L = (self._get_channel_phase(ch_map, 1) + 
                   self._get_channel_phase(ch_map, 3)) / 2.0
        
        # Effective R phase = average of CH2 + CH4
        phase_R = (self._get_channel_phase(ch_map, 2) + 
                   self._get_channel_phase(ch_map, 4)) / 2.0
        
        lr_diff = self._phase_difference(phase_L, phase_R)
        
        if self.binaural_mode == "coherent":
            # In coherent mode, L/R should be in-phase
            # Score decreases with phase difference
            score = 1.0 - min(lr_diff / 90.0, 1.0)
        else:
            # In binaural_beat mode, small controlled offset is OK
            # Penalize only large differences (>90°) or very small (<5°)
            if 5.0 < lr_diff < 90.0:
                score = 1.0  # Good binaural range
            elif lr_diff <= 5.0:
                score = 0.7  # Too in-phase for binaural
            else:
                score = 1.0 - (lr_diff - 90.0) / 90.0  # Too out of phase
        
        return max(0.0, min(1.0, score))
    
    def _score_delay_alignment(self, ch_map: Dict[int, Any]) -> float:
        """
        Score delay alignment between same-side channels.
        
        Same-side channels should have similar delays for coherent wavefront.
        
        Returns:
            Score [0, 1] where 1.0 = perfect delay alignment
        """
        # Left side: CH1 ↔ CH3
        delay_1 = self._get_channel_delay(ch_map, 1)
        delay_3 = self._get_channel_delay(ch_map, 3)
        left_diff = abs(delay_1 - delay_3)
        
        # Right side: CH2 ↔ CH4
        delay_2 = self._get_channel_delay(ch_map, 2)
        delay_4 = self._get_channel_delay(ch_map, 4)
        right_diff = abs(delay_2 - delay_4)
        
        # Score: 1.0 if diff < target, decreases to 0 at 50ms
        max_delay = 50.0  # ms
        left_score = 1.0 - min(left_diff / max_delay, 1.0)
        right_score = 1.0 - min(right_diff / max_delay, 1.0)
        
        # Bonus for being within target
        if left_diff < self.target_delay_diff_ms:
            left_score = min(left_score + 0.2, 1.0)
        if right_diff < self.target_delay_diff_ms:
            right_score = min(right_score + 0.2, 1.0)
        
        return (left_score + right_score) / 2.0
    
    def _score_gain_balance(self, ch_map: Dict[int, Any]) -> float:
        """
        Score gain balance between L/R sides.
        
        L/R should have similar total gain for stereo image centering.
        
        Returns:
            Score [0, 1] where 1.0 = perfect L/R gain balance
        """
        # Left side total gain (linear sum)
        gain_1 = self._get_channel_gain(ch_map, 1)
        gain_3 = self._get_channel_gain(ch_map, 3)
        left_linear = 10**(gain_1/20) + 10**(gain_3/20)
        
        # Right side total gain
        gain_2 = self._get_channel_gain(ch_map, 2)
        gain_4 = self._get_channel_gain(ch_map, 4)
        right_linear = 10**(gain_2/20) + 10**(gain_4/20)
        
        # Ratio (should be close to 1.0)
        if left_linear < 1e-10 or right_linear < 1e-10:
            return 0.5  # Neutral if no gain
        
        ratio = min(left_linear, right_linear) / max(left_linear, right_linear)
        
        # Score: ratio directly (1.0 = perfect balance)
        # Bonus for being very close (within 1 dB)
        if ratio > 0.89:  # ~1 dB difference
            ratio = min(ratio + 0.1, 1.0)
        
        return float(ratio)
    
    @staticmethod
    def _phase_difference(phase1: float, phase2: float) -> float:
        """
        Calculate shortest phase difference (0-180°).
        
        Handles wraparound: e.g., 350° and 10° differ by 20°, not 340°.
        """
        diff = abs(phase1 - phase2) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        return diff


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def create_jab_scorer(
    mode: str = "coherent",
    weight: float = 0.25
) -> JABCoherenceScorer:
    """
    Create JAB scorer with preset configuration.
    
    Args:
        mode: "coherent" (stereo) or "binaural_beat" (therapeutic)
        weight: Scorer weight
    
    Returns:
        Configured JABCoherenceScorer
    """
    return JABCoherenceScorer(
        target_phase_diff_deg=30.0,
        target_delay_diff_ms=5.0,
        binaural_mode=mode,
        weight=weight
    )
