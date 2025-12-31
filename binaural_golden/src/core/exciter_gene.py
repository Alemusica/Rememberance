"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXCITER GENE - Staged Gene Activation                      ║
║                                                                              ║
║   "Il seme non parla dei petali" - Emission genes activate when needed       ║
║                                                                              ║
║   GENE PHASES:                                                               ║
║   • SEED: Position genes only (x, y) - emission dormant                      ║
║   • BLOOM: Position + Emission genes (phase, delay, gain, polarity)          ║
║   • FREEZE: Position locked (CNC), only DSP/emission evolvable               ║
║                                                                              ║
║   This implements the "recessivo/dominante" pattern for gene expression:     ║
║   - Position genes are always "dominant" (expressed)                         ║
║   - Emission genes start "recessive" (dormant) and activate contextually     ║
║                                                                              ║
║   COMPATIBILITY:                                                             ║
║   - ExciterGene IS-A ExciterPosition (inheritance)                          ║
║   - to_position() converts to legacy ExciterPosition                         ║
║   - from_position() upgrades legacy data                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any, Union
from enum import Enum, auto
import numpy as np
import copy

from .analysis_config import (
    GenePhase, EmissionBounds, GeneActivationConfig,
    get_default_config
)

# Type alias for backwards compatibility
ExciterPosition = 'ExciterGene'  # Forward reference


# ══════════════════════════════════════════════════════════════════════════════
# EMISSION GENES - DSP parameters (dormant until BLOOM)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmissionGenes:
    """
    DSP emission parameters for an exciter.
    
    These are "recessive" genes - dormant in SEED phase, expressed in BLOOM.
    
    Hardware: JAB4 WONDOM DSP @ 48kHz
    - Per-channel phase shift, delay, gain, polarity
    """
    # Phase rotation [degrees] - for beam steering
    phase_deg: float = 0.0
    
    # Sample delay @ 48kHz - for time alignment
    delay_samples: int = 0
    
    # Gain relative to nominal [dB]
    gain_db: float = 0.0
    
    # Polarity inversion
    polarity_inverted: bool = False
    
    # Lock flags (for partial FREEZE)
    phase_locked: bool = False
    delay_locked: bool = False
    gain_locked: bool = False
    polarity_locked: bool = False
    
    def to_dsp_dict(self) -> Dict[str, Any]:
        """Export for DSP processor configuration."""
        return {
            "phase_deg": self.phase_deg,
            "delay_samples": self.delay_samples,
            "delay_ms": self.delay_samples / 48.0,  # @ 48kHz
            "gain_db": self.gain_db,
            "gain_linear": 10 ** (self.gain_db / 20),
            "polarity": -1 if self.polarity_inverted else 1,
        }
    
    def is_neutral(self) -> bool:
        """Check if emission is at neutral (no effect)."""
        return (
            abs(self.phase_deg) < 0.1 and
            self.delay_samples == 0 and
            abs(self.gain_db) < 0.01 and
            not self.polarity_inverted
        )
    
    def clone(self) -> 'EmissionGenes':
        """Deep copy."""
        return EmissionGenes(
            phase_deg=self.phase_deg,
            delay_samples=self.delay_samples,
            gain_db=self.gain_db,
            polarity_inverted=self.polarity_inverted,
            phase_locked=self.phase_locked,
            delay_locked=self.delay_locked,
            gain_locked=self.gain_locked,
            polarity_locked=self.polarity_locked,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXCITER GENE - Unified position + emission with phase awareness
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExciterGene:
    """
    Complete exciter gene with position and emission.
    
    INHERITS BEHAVIOR FROM ExciterPosition:
    - x, y: Normalized position (0-1)
    - channel: JAB4 channel (1-4)
    - exciter_model, diameter_mm, power_w, impedance_ohm
    
    ADDS EMISSION GENES:
    - emission: EmissionGenes (phase, delay, gain, polarity)
    
    PHASE-AWARE BEHAVIOR:
    - In SEED phase: emission is dormant (ignored in fitness, not mutated)
    - In BLOOM phase: emission is active (contributes to fitness, mutated)
    - In FREEZE phase: position locked, only emission evolvable
    
    USAGE:
        # Create in SEED phase (position only matters)
        gene = ExciterGene(x=0.3, y=0.85, channel=1)
        gene.activate_emission()  # → BLOOM
        gene.freeze_position()    # → FREEZE (CNC'd, only DSP tunable)
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION GENES (from ExciterPosition - always active)
    # ═══════════════════════════════════════════════════════════════════════════
    
    x: float = 0.5              # Lateral position (0=left, 1=right)
    y: float = 0.5              # Longitudinal position (0=feet, 1=head)
    channel: int = 1            # JAB4 channel (1-4)
    
    # Hardware model
    exciter_model: str = "dayton_daex25"
    diameter_mm: float = 25.0
    power_w: float = 10.0       # RMS
    impedance_ohm: float = 4.0  # 4Ω version
    
    # Position lock (for FREEZE phase)
    position_locked: bool = False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMISSION GENES (dormant in SEED, active in BLOOM)
    # ═══════════════════════════════════════════════════════════════════════════
    
    emission: EmissionGenes = field(default_factory=EmissionGenes)
    
    # Current gene phase
    _phase: GenePhase = GenePhase.SEED
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE TRANSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @property
    def phase(self) -> GenePhase:
        """Current gene activation phase."""
        return self._phase
    
    def activate_emission(self) -> 'ExciterGene':
        """
        Transition SEED → BLOOM: Activate emission genes.
        
        "Il seme sboccia" - The seed blooms.
        """
        if self._phase == GenePhase.SEED:
            self._phase = GenePhase.BLOOM
        return self
    
    def freeze_position(self) -> 'ExciterGene':
        """
        Transition to FREEZE: Lock position (CNC'd), only emission tunable.
        
        Physical position is now fixed. Only DSP can evolve.
        """
        self._phase = GenePhase.FREEZE
        self.position_locked = True
        return self
    
    def is_emission_active(self) -> bool:
        """Check if emission genes are currently expressed."""
        return self._phase in (GenePhase.BLOOM, GenePhase.FREEZE)
    
    def is_position_evolvable(self) -> bool:
        """Check if position genes can be mutated."""
        return not self.position_locked and self._phase != GenePhase.FREEZE
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MUTATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def mutate_position(
        self,
        sigma_x: float = 0.05,
        sigma_y: float = 0.05,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> 'ExciterGene':
        """
        Mutate position genes (if not frozen).
        
        Args:
            sigma_x, sigma_y: Standard deviation of Gaussian noise
            bounds: (x_min, x_max, y_min, y_max) - defaults to (0.1, 0.9, 0.1, 0.9)
        """
        if not self.is_position_evolvable():
            return self  # Position locked
        
        bounds = bounds or (0.1, 0.9, 0.1, 0.9)
        
        # Gaussian mutation
        self.x = np.clip(self.x + np.random.normal(0, sigma_x), bounds[0], bounds[1])
        self.y = np.clip(self.y + np.random.normal(0, sigma_y), bounds[2], bounds[3])
        
        return self
    
    def mutate_emission(
        self,
        config: Optional[EmissionBounds] = None,
        mutation_rates: Optional[Dict[str, float]] = None,
    ) -> 'ExciterGene':
        """
        Mutate emission genes (if active and not locked).
        
        Args:
            config: Bounds for emission parameters
            mutation_rates: Per-parameter mutation probabilities
        """
        if not self.is_emission_active():
            return self  # Emission dormant
        
        config = config or get_default_config().gene_activation.emission_bounds
        rates = mutation_rates or {
            "phase": 0.3,
            "delay": 0.2,
            "gain": 0.3,
            "polarity": 0.1,  # Less frequent
        }
        
        em = self.emission
        
        # Phase mutation
        if not em.phase_locked and np.random.random() < rates["phase"]:
            # Gaussian mutation with wrap-around
            delta = np.random.normal(0, 30)  # ~30° std
            em.phase_deg = (em.phase_deg + delta) % 360.0
        
        # Delay mutation
        if not em.delay_locked and np.random.random() < rates["delay"]:
            delta = np.random.choice([-2, -1, 0, 1, 2])
            em.delay_samples = int(np.clip(
                em.delay_samples + delta,
                config.delay_min,
                config.delay_max
            ))
        
        # Gain mutation
        if not em.gain_locked and np.random.random() < rates["gain"]:
            delta = np.random.normal(0, 1.5)  # ~1.5dB std
            em.gain_db = np.clip(
                em.gain_db + delta,
                config.gain_min_db,
                config.gain_max_db
            )
        
        # Polarity mutation
        if not em.polarity_locked and np.random.random() < rates["polarity"]:
            em.polarity_inverted = not em.polarity_inverted
        
        return self
    
    def mutate(
        self,
        position_sigma: float = 0.05,
        emission_config: Optional[EmissionBounds] = None,
    ) -> 'ExciterGene':
        """
        Mutate both position and emission (respecting phase).
        """
        self.mutate_position(sigma_x=position_sigma, sigma_y=position_sigma)
        self.mutate_emission(config=emission_config)
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CROSSOVER
    # ═══════════════════════════════════════════════════════════════════════════
    
    def crossover_with(
        self,
        other: 'ExciterGene',
        position_blend: float = 0.5,
        emission_from_other: bool = False,
    ) -> 'ExciterGene':
        """
        Create child gene via crossover.
        
        Args:
            other: Other parent gene
            position_blend: Blend factor for position (0=self, 1=other)
            emission_from_other: If True, take emission from other parent
        """
        child = self.clone()
        
        # Position blending (if evolvable)
        if child.is_position_evolvable():
            child.x = self.x * (1 - position_blend) + other.x * position_blend
            child.y = self.y * (1 - position_blend) + other.y * position_blend
        
        # Emission crossover (if active)
        if child.is_emission_active():
            if emission_from_other:
                child.emission = other.emission.clone()
            else:
                # Blend or random selection per parameter
                em1, em2 = self.emission, other.emission
                child.emission = EmissionGenes(
                    phase_deg=(em1.phase_deg + em2.phase_deg) / 2,
                    delay_samples=np.random.choice([em1.delay_samples, em2.delay_samples]),
                    gain_db=(em1.gain_db + em2.gain_db) / 2,
                    polarity_inverted=np.random.choice([em1.polarity_inverted, em2.polarity_inverted]),
                )
        
        return child
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVERSION & COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """
        Convert to absolute coordinates [m].
        
        Compatible with legacy ExciterPosition.to_absolute()
        """
        return {
            "center": (self.y * plate_length, self.x * plate_width),
            "diameter": self.diameter_mm / 1000,
            "channel": self.channel,
            "emission": self.emission.to_dsp_dict() if self.is_emission_active() else None,
        }
    
    def to_position_dict(self) -> Dict[str, Any]:
        """Export position genes only (for legacy compatibility)."""
        return {
            "x": self.x,
            "y": self.y,
            "channel": self.channel,
            "exciter_model": self.exciter_model,
            "diameter_mm": self.diameter_mm,
            "power_w": self.power_w,
            "impedance_ohm": self.impedance_ohm,
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        """Export all genes (position + emission if active)."""
        result = self.to_position_dict()
        result["phase"] = self._phase.name
        result["position_locked"] = self.position_locked
        
        if self.is_emission_active():
            result["emission"] = self.emission.to_dsp_dict()
        
        return result
    
    def clone(self) -> 'ExciterGene':
        """Deep copy of gene."""
        return ExciterGene(
            x=self.x,
            y=self.y,
            channel=self.channel,
            exciter_model=self.exciter_model,
            diameter_mm=self.diameter_mm,
            power_w=self.power_w,
            impedance_ohm=self.impedance_ohm,
            position_locked=self.position_locked,
            emission=self.emission.clone(),
            _phase=self._phase,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PROPERTIES (compatibility with ExciterPosition)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @property
    def is_head_zone(self) -> bool:
        """True if exciter is in head zone (y > 0.7)."""
        return self.y > 0.7
    
    @property
    def is_feet_zone(self) -> bool:
        """True if exciter is in feet zone (y < 0.3)."""
        return self.y < 0.3
    
    @property
    def zone(self) -> str:
        """Get zone name."""
        if self.is_head_zone:
            return "head"
        elif self.is_feet_zone:
            return "feet"
        else:
            return "torso"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLASS METHODS - Factory & conversion
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def from_legacy_position(
        cls,
        pos: Any,  # ExciterPosition or dict
        phase: GenePhase = GenePhase.SEED
    ) -> 'ExciterGene':
        """
        Create ExciterGene from legacy ExciterPosition.
        
        Upgrades old data to new format with dormant emission.
        """
        if isinstance(pos, dict):
            return cls(
                x=pos.get("x", 0.5),
                y=pos.get("y", 0.5),
                channel=pos.get("channel", 1),
                exciter_model=pos.get("exciter_model", "dayton_daex25"),
                diameter_mm=pos.get("diameter_mm", 25.0),
                power_w=pos.get("power_w", 10.0),
                impedance_ohm=pos.get("impedance_ohm", 4.0),
                _phase=phase,
            )
        
        # Assume dataclass-like with attributes
        return cls(
            x=getattr(pos, 'x', 0.5),
            y=getattr(pos, 'y', 0.5),
            channel=getattr(pos, 'channel', 1),
            exciter_model=getattr(pos, 'exciter_model', "dayton_daex25"),
            diameter_mm=getattr(pos, 'diameter_mm', 25.0),
            power_w=getattr(pos, 'power_w', 10.0),
            impedance_ohm=getattr(pos, 'impedance_ohm', 4.0),
            _phase=phase,
        )
    
    @classmethod
    def create_head_stereo(
        cls,
        left_x: float = 0.3,
        right_x: float = 0.7,
        y: float = 0.85,
        phase: GenePhase = GenePhase.SEED,
    ) -> Tuple['ExciterGene', 'ExciterGene']:
        """Create stereo pair for head zone."""
        left = cls(x=left_x, y=y, channel=1, _phase=phase)
        right = cls(x=right_x, y=y, channel=2, _phase=phase)
        return left, right
    
    @classmethod
    def create_feet_stereo(
        cls,
        left_x: float = 0.3,
        right_x: float = 0.7,
        y: float = 0.15,
        phase: GenePhase = GenePhase.SEED,
    ) -> Tuple['ExciterGene', 'ExciterGene']:
        """Create stereo pair for feet zone."""
        left = cls(x=left_x, y=y, channel=3, _phase=phase)
        right = cls(x=right_x, y=y, channel=4, _phase=phase)
        return left, right
    
    @classmethod
    def create_default_layout(
        cls,
        phase: GenePhase = GenePhase.SEED
    ) -> List['ExciterGene']:
        """
        Create default 4-exciter layout.
        
        Standard JAB4 configuration:
        - CH1: Head Left
        - CH2: Head Right
        - CH3: Feet Left
        - CH4: Feet Right
        """
        head_l, head_r = cls.create_head_stereo(phase=phase)
        feet_l, feet_r = cls.create_feet_stereo(phase=phase)
        return [head_l, head_r, feet_l, feet_r]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def upgrade_exciters_to_genes(
    exciters: List[Any],
    phase: GenePhase = GenePhase.SEED
) -> List[ExciterGene]:
    """
    Upgrade list of legacy ExciterPosition to ExciterGene.
    
    Safe to call on already-converted data.
    """
    result = []
    for exc in exciters:
        if isinstance(exc, ExciterGene):
            result.append(exc)
        else:
            result.append(ExciterGene.from_legacy_position(exc, phase=phase))
    return result


def activate_all_emission(genes: List[ExciterGene]) -> List[ExciterGene]:
    """Activate emission on all genes (SEED → BLOOM)."""
    for gene in genes:
        gene.activate_emission()
    return genes


def freeze_all_positions(genes: List[ExciterGene]) -> List[ExciterGene]:
    """Freeze positions on all genes (→ FREEZE)."""
    for gene in genes:
        gene.freeze_position()
    return genes


def calculate_position_sigma(genes: List[ExciterGene]) -> float:
    """
    Calculate standard deviation of exciter positions.
    
    Used for detecting position convergence (SEED → BLOOM trigger).
    """
    if len(genes) < 2:
        return 0.0
    
    xs = [g.x for g in genes]
    ys = [g.y for g in genes]
    
    # Combined position variance
    var_x = np.var(xs)
    var_y = np.var(ys)
    
    return np.sqrt(var_x + var_y)


def get_emission_summary(genes: List[ExciterGene]) -> Dict[str, Any]:
    """
    Get summary of emission parameters across all exciters.
    """
    active_count = sum(1 for g in genes if g.is_emission_active())
    
    phases = [g.emission.phase_deg for g in genes if g.is_emission_active()]
    delays = [g.emission.delay_samples for g in genes if g.is_emission_active()]
    gains = [g.emission.gain_db for g in genes if g.is_emission_active()]
    
    return {
        "active_count": active_count,
        "total_count": len(genes),
        "phases_deg": phases,
        "delays_samples": delays,
        "gains_db": gains,
        "phase_spread": max(phases) - min(phases) if phases else 0,
        "delay_spread": max(delays) - min(delays) if delays else 0,
        "gain_spread": max(gains) - min(gains) if gains else 0,
    }
