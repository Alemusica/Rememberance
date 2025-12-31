"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PHYSICS-BASED CUTOUT PLACEMENT for DML Plate Optimization           ║
║                                                                              ║
║   Intelligent cutout placement based on modal analysis and ABH theory.       ║
║   Replaces random placement with physics-driven suggestions.                 ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Schleske 2002: Cutouts at antinodes shift frequencies maximally         ║
║   • Krylov 2014, Deng 2019: ABH (Acoustic Black Holes) focus energy         ║
║   • Zhao 2025: ABH + cutouts synergy for enhanced focusing                  ║
║   • Bai & Liu 2004: GA-based exciter placement (similar principles)         ║
║                                                                              ║
║   KEY PRINCIPLES:                                                            ║
║   1. Cutout at mode ANTINODE → maximum frequency shift                       ║
║   2. Cutout at mode NODE → minimal effect                                    ║
║   3. Edge cutouts → ABH-like energy focusing                                 ║
║   4. SYMMETRIC cutouts for SYMMETRIC modes (reduce coupling)                 ║
║   5. ASYMMETRIC cutouts for mode DEGENERACY breaking                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

# Local imports
from .plate_genome import CutoutGene, PlateGenome, CUTOUT_SHAPES

logger = logging.getLogger(__name__)


class CutoutPurpose(Enum):
    """Purpose/intent of a cutout for physics-based placement."""
    FREQUENCY_SHIFT = "freq_shift"      # Shift specific modal frequency
    ABH_FOCUS = "abh_focus"             # Acoustic black hole energy focusing
    MODE_BREAK = "mode_break"           # Break mode degeneracy
    SPINE_AVOID = "spine_avoid"         # Avoid spine load zone
    EAR_ENHANCE = "ear_enhance"         # Enhance ear zone response
    LUTHERIE = "lutherie"               # Traditional f-hole style (aesthetics + acoustics)


class SymmetryType(Enum):
    """Symmetry classification for cutout placement."""
    SYMMETRIC = "symmetric"       # Mirror about centerline (y=0.5)
    ASYMMETRIC = "asymmetric"     # Single cutout, no mirror
    DIAGONAL = "diagonal"         # Mirror about diagonal (corners)
    QUAD = "quad"                 # Four-fold symmetry (all corners)


@dataclass
class CutoutSuggestion:
    """
    Physics-based suggestion for cutout placement.
    
    Includes position, recommended shape, and reasoning.
    """
    x: float                          # Normalized X position (lateral, 0=left)
    y: float                          # Normalized Y position (longitudinal, 0=feet)
    purpose: CutoutPurpose            # Why place here
    symmetry: SymmetryType            # Symmetry recommendation
    recommended_shape: str = "ellipse"  # Suggested shape from CUTOUT_SHAPES
    recommended_width: float = 0.05   # Suggested width (normalized)
    recommended_height: float = 0.05  # Suggested height (normalized)
    target_mode: Optional[int] = None # Mode index this affects (if freq_shift)
    confidence: float = 0.8           # Confidence in suggestion [0-1]
    reasoning: str = ""               # Human-readable explanation
    
    def to_cutout_gene(self, rotation: float = 0.0) -> CutoutGene:
        """Convert suggestion to CutoutGene for genome."""
        return CutoutGene(
            x=self.x,
            y=self.y,
            width=self.recommended_width,
            height=self.recommended_height,
            rotation=rotation,
            shape=self.recommended_shape,
        )


class CutoutPlacementOptimizer:
    """
    Physics-based cutout placement optimizer.
    
    Analyzes modal shapes to suggest optimal cutout positions
    for specific acoustic goals (frequency tuning, energy focusing, etc.).
    
    USAGE:
        optimizer = CutoutPlacementOptimizer()
        suggestions = optimizer.suggest_for_frequency_tuning(
            mode_shapes, current_frequencies, target_frequencies
        )
        # Apply suggestions to genome
        for suggestion in suggestions:
            genome.cutouts.append(suggestion.to_cutout_gene())
    """
    
    # Avoid these zones for cutouts (normalized coords)
    SPINE_ZONE = (0.35, 0.65, 0.35, 0.65)  # x_min, x_max, y_min, y_max (central spine)
    EDGE_THRESHOLD = 0.15  # Distance from edge for ABH effect
    
    def __init__(self, plate_length: float = 2.0, plate_width: float = 0.7):
        """
        Initialize optimizer.
        
        Args:
            plate_length: Plate length in meters
            plate_width: Plate width in meters
        """
        self.plate_length = plate_length
        self.plate_width = plate_width
    
    def suggest_for_frequency_tuning(
        self,
        mode_shapes: np.ndarray,
        current_frequencies: List[float],
        target_frequencies: List[float],
        max_suggestions: int = 4,
    ) -> List[CutoutSuggestion]:
        """
        Suggest cutout positions to tune frequencies toward targets.
        
        PHYSICS: Cutouts at antinodes reduce stiffness locally,
        lowering that mode's frequency. Larger cutouts = larger shift.
        
        Args:
            mode_shapes: Array of mode shapes (n_modes, nx, ny)
            current_frequencies: Current modal frequencies [Hz]
            target_frequencies: Desired frequencies [Hz]
            max_suggestions: Maximum number of suggestions
        
        Returns:
            List of CutoutSuggestion with positions and reasoning
        """
        suggestions = []
        
        n_modes = min(len(current_frequencies), len(target_frequencies), len(mode_shapes))
        
        # Find modes needing adjustment (>5% deviation from target)
        modes_to_adjust = []
        for i in range(n_modes):
            if target_frequencies[i] > 0:
                deviation = abs(current_frequencies[i] - target_frequencies[i]) / target_frequencies[i]
                if deviation > 0.05:
                    modes_to_adjust.append((i, deviation, current_frequencies[i] > target_frequencies[i]))
        
        # Sort by deviation (largest first)
        modes_to_adjust.sort(key=lambda x: -x[1])
        
        for mode_idx, deviation, need_lower in modes_to_adjust[:max_suggestions]:
            mode_shape = mode_shapes[mode_idx]
            
            # Find antinode (maximum amplitude point)
            antinode_pos = self._find_antinode(mode_shape, avoid_spine=True)
            
            if antinode_pos is None:
                continue
            
            ax, ay = antinode_pos
            
            # Determine symmetry based on mode shape
            symmetry = self._analyze_mode_symmetry(mode_shape)
            
            # Size based on needed frequency shift
            # Larger deviation → larger cutout needed
            base_size = 0.03 + 0.05 * min(deviation, 0.3)  # 3-8% of plate
            
            # Determine if frequency should go up or down
            if need_lower:
                # Need to lower frequency → cutout at antinode
                reasoning = (
                    f"Mode {mode_idx+1} at {current_frequencies[mode_idx]:.1f}Hz is too high "
                    f"(target: {target_frequencies[mode_idx]:.1f}Hz). "
                    f"Cutout at antinode ({ax:.2f}, {ay:.2f}) will lower frequency."
                )
            else:
                # Need to raise frequency → cutout near node
                # (actually, cutouts generally lower freq; suggest different approach)
                ax, ay = self._find_node(mode_shape)
                reasoning = (
                    f"Mode {mode_idx+1} at {current_frequencies[mode_idx]:.1f}Hz is too low "
                    f"(target: {target_frequencies[mode_idx]:.1f}Hz). "
                    f"Small cutout near node ({ax:.2f}, {ay:.2f}) for minimal impact while "
                    f"allowing adjacent mode coupling to shift frequency."
                )
                base_size *= 0.5  # Smaller cutout near nodes
            
            suggestions.append(CutoutSuggestion(
                x=ax,
                y=ay,
                purpose=CutoutPurpose.FREQUENCY_SHIFT,
                symmetry=symmetry,
                recommended_shape="ellipse",
                recommended_width=base_size * 1.2,  # Slightly wider than tall
                recommended_height=base_size,
                target_mode=mode_idx,
                confidence=0.7 + 0.2 * (1 - deviation),  # Higher confidence for smaller deviations
                reasoning=reasoning,
            ))
        
        return suggestions
    
    def suggest_for_abh_focusing(
        self,
        target_zone: str = "ear",  # "ear", "spine", "both"
        use_spirals: bool = True,
    ) -> List[CutoutSuggestion]:
        """
        Suggest cutout positions for ABH (Acoustic Black Hole) energy focusing.
        
        ABH PHYSICS (Krylov 2014, Deng 2019):
        - Tapered thickness profiles trap and focus vibrational energy
        - Edge/corner cutouts create ABH-like effect
        - Energy accumulates at ABH tip → enhanced response in target zone
        
        Args:
            target_zone: Where to focus energy ("ear", "spine", "both")
            use_spirals: Whether to use spiral shapes (golden ratio)
        
        Returns:
            List of suggestions for ABH-style cutouts
        """
        suggestions = []
        shape = "spiral" if use_spirals else "ellipse"
        
        if target_zone in ["ear", "both"]:
            # Ear zone: corners near head end (y > 0.85)
            # Diagonal symmetry for even energy distribution to both ears
            suggestions.append(CutoutSuggestion(
                x=0.12, y=0.88,  # Near head, left corner
                purpose=CutoutPurpose.ABH_FOCUS,
                symmetry=SymmetryType.SYMMETRIC,  # Mirror to right side
                recommended_shape=shape,
                recommended_width=0.06,
                recommended_height=0.06,
                confidence=0.85,
                reasoning=(
                    "ABH-style cutout at head corner focuses vibrational energy "
                    "toward ear zone. Krylov 2014 shows edge ABH enhances energy trapping."
                ),
            ))
            # Add symmetric partner
            suggestions.append(CutoutSuggestion(
                x=0.88, y=0.88,  # Near head, right corner
                purpose=CutoutPurpose.ABH_FOCUS,
                symmetry=SymmetryType.SYMMETRIC,
                recommended_shape=shape,
                recommended_width=0.06,
                recommended_height=0.06,
                confidence=0.85,
                reasoning="Symmetric partner for L/R balanced ABH focusing.",
            ))
        
        if target_zone in ["spine", "both"]:
            # Spine zone: lateral edges near center
            # Creates ABH corridor along spine
            suggestions.append(CutoutSuggestion(
                x=0.05, y=0.50,  # Left edge, center height
                purpose=CutoutPurpose.ABH_FOCUS,
                symmetry=SymmetryType.SYMMETRIC,
                recommended_shape="crescent",  # Crescent follows edge better
                recommended_width=0.04,
                recommended_height=0.10,
                confidence=0.75,
                reasoning=(
                    "Lateral ABH cutout creates energy focusing along spine axis. "
                    "Deng 2019 ring-ABH principle applied linearly."
                ),
            ))
            suggestions.append(CutoutSuggestion(
                x=0.95, y=0.50,  # Right edge, center height
                purpose=CutoutPurpose.ABH_FOCUS,
                symmetry=SymmetryType.SYMMETRIC,
                recommended_shape="crescent",
                recommended_width=0.04,
                recommended_height=0.10,
                confidence=0.75,
                reasoning="Symmetric partner for balanced spine ABH.",
            ))
        
        return suggestions
    
    def suggest_symmetric_pair(
        self,
        base_position: Tuple[float, float],
        shape: str = "ellipse",
        size: float = 0.05,
        symmetry: SymmetryType = SymmetryType.SYMMETRIC,
    ) -> List[CutoutSuggestion]:
        """
        Generate symmetric cutout pair for balanced vibration.
        
        PHYSICS: Symmetric cutouts affect symmetric modes evenly,
        maintaining L/R balance crucial for binaural applications.
        
        Args:
            base_position: (x, y) of first cutout (normalized)
            shape: Cutout shape from CUTOUT_SHAPES
            size: Base size (normalized)
            symmetry: Type of symmetry to apply
        
        Returns:
            List of 2+ suggestions forming symmetric arrangement
        """
        x, y = base_position
        suggestions = []
        
        # Base cutout
        suggestions.append(CutoutSuggestion(
            x=x, y=y,
            purpose=CutoutPurpose.MODE_BREAK,
            symmetry=symmetry,
            recommended_shape=shape,
            recommended_width=size,
            recommended_height=size,
            confidence=0.80,
            reasoning="Base cutout for symmetric pair.",
        ))
        
        if symmetry == SymmetryType.SYMMETRIC:
            # Mirror about y=0.5 (lateral centerline)
            mirror_x = 1.0 - x
            suggestions.append(CutoutSuggestion(
                x=mirror_x, y=y,
                purpose=CutoutPurpose.MODE_BREAK,
                symmetry=symmetry,
                recommended_shape=shape,
                recommended_width=size,
                recommended_height=size,
                confidence=0.80,
                reasoning="Symmetric partner (mirrored about centerline).",
            ))
        
        elif symmetry == SymmetryType.DIAGONAL:
            # Mirror about diagonal (swap corners)
            suggestions.append(CutoutSuggestion(
                x=1.0 - x, y=1.0 - y,
                purpose=CutoutPurpose.MODE_BREAK,
                symmetry=symmetry,
                recommended_shape=shape,
                recommended_width=size,
                recommended_height=size,
                confidence=0.80,
                reasoning="Diagonal partner (opposite corner).",
            ))
        
        elif symmetry == SymmetryType.QUAD:
            # Four-fold symmetry (all corners)
            corners = [
                (x, y),
                (1.0 - x, y),
                (x, 1.0 - y),
                (1.0 - x, 1.0 - y),
            ]
            for i, (cx, cy) in enumerate(corners[1:], 1):
                suggestions.append(CutoutSuggestion(
                    x=cx, y=cy,
                    purpose=CutoutPurpose.MODE_BREAK,
                    symmetry=symmetry,
                    recommended_shape=shape,
                    recommended_width=size,
                    recommended_height=size,
                    confidence=0.80,
                    reasoning=f"Quad symmetry partner #{i+1}.",
                ))
        
        return suggestions
    
    def _find_antinode(
        self,
        mode_shape: np.ndarray,
        avoid_spine: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """
        Find antinode (maximum amplitude) position in mode shape.
        
        Args:
            mode_shape: 2D array of mode amplitudes
            avoid_spine: If True, avoid central spine zone
        
        Returns:
            (x_norm, y_norm) of antinode, or None if not found
        """
        if mode_shape is None or mode_shape.size == 0:
            return None
        
        ny, nx = mode_shape.shape
        abs_shape = np.abs(mode_shape)
        
        if avoid_spine:
            # Mask out spine zone
            x_min = int(self.SPINE_ZONE[0] * nx)
            x_max = int(self.SPINE_ZONE[1] * nx)
            y_min = int(self.SPINE_ZONE[2] * ny)
            y_max = int(self.SPINE_ZONE[3] * ny)
            abs_shape[y_min:y_max, x_min:x_max] = 0
        
        # Find maximum
        max_idx = np.unravel_index(np.argmax(abs_shape), abs_shape.shape)
        
        if abs_shape[max_idx] < 0.1:  # No significant antinode outside spine
            return None
        
        y_idx, x_idx = max_idx
        return (x_idx / nx, y_idx / ny)
    
    def _find_node(self, mode_shape: np.ndarray) -> Tuple[float, float]:
        """
        Find node (minimum amplitude) position in mode shape.
        
        Nodes are where cutouts have minimal acoustic effect.
        """
        if mode_shape is None or mode_shape.size == 0:
            return (0.5, 0.5)
        
        ny, nx = mode_shape.shape
        abs_shape = np.abs(mode_shape)
        
        # Find minimum (but not exact zero at boundaries)
        # Mask boundaries
        abs_shape[0, :] = np.inf
        abs_shape[-1, :] = np.inf
        abs_shape[:, 0] = np.inf
        abs_shape[:, -1] = np.inf
        
        min_idx = np.unravel_index(np.argmin(abs_shape), abs_shape.shape)
        y_idx, x_idx = min_idx
        return (x_idx / nx, y_idx / ny)
    
    def _analyze_mode_symmetry(self, mode_shape: np.ndarray) -> SymmetryType:
        """
        Analyze mode shape to determine symmetry type.
        
        Symmetric modes benefit from symmetric cutout pairs.
        """
        if mode_shape is None:
            return SymmetryType.ASYMMETRIC
        
        ny, nx = mode_shape.shape
        
        # Check lateral symmetry (mirror about y=0.5)
        left_half = mode_shape[:, :nx//2]
        right_half = np.flip(mode_shape[:, nx//2:], axis=1)
        
        # Ensure same shape
        min_cols = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_cols]
        right_half = right_half[:, :min_cols]
        
        symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        
        if symmetry_score > 0.9:
            return SymmetryType.SYMMETRIC
        elif symmetry_score < -0.5:
            # Anti-symmetric → diagonal placement better
            return SymmetryType.DIAGONAL
        else:
            return SymmetryType.ASYMMETRIC


def apply_cutout_suggestions(
    genome: PlateGenome,
    suggestions: List[CutoutSuggestion],
    replace: bool = False,
) -> PlateGenome:
    """
    Apply cutout suggestions to a genome.
    
    Args:
        genome: PlateGenome to modify
        suggestions: List of CutoutSuggestion to apply
        replace: If True, replace existing cutouts; if False, append
    
    Returns:
        Modified genome (same instance if mutable)
    """
    if replace:
        genome.cutouts = []
    
    for suggestion in suggestions:
        if suggestion.confidence >= 0.5:  # Only apply confident suggestions
            cutout = suggestion.to_cutout_gene()
            genome.cutouts.append(cutout)
            logger.info(
                f"Applied cutout: {suggestion.recommended_shape} at "
                f"({suggestion.x:.2f}, {suggestion.y:.2f}) for {suggestion.purpose.value}"
            )
    
    return genome
