"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 MODAL GUIDANCE - Physics-Based Cutout Placement              ║
║                                                                              ║
║   Uses modal analysis to guide WHERE to place cutouts for targeted           ║
║   frequency response modification.                                           ║
║                                                                              ║
║   PHYSICS PRINCIPLE (Schleske 2002, Fletcher & Rossing 1998):                ║
║   - Cutout at ANTINODE → maximum frequency shift (mass + stiffness)          ║
║   - Cutout at NODE → minimal effect on that mode                             ║
║   - F-holes in violins are positioned to tune Helmholtz and corpus modes     ║
║                                                                              ║
║   RAYLEIGH QUOTIENT for frequency shift:                                     ║
║   Δω²/ω² ≈ -(1/M)ΔM + (1/K)ΔK                                               ║
║   where ΔM depends on mode amplitude at cutout, ΔK on local curvature        ║
║                                                                              ║
║   References:                                                                ║
║   - Schleske, M. (2002). Empirical Tools in Contemporary Violin Making       ║
║   - Fletcher & Rossing (1998). The Physics of Musical Instruments            ║
║   - Bai & Liu (2004). Genetic algorithm for exciter placement                ║
║   - Deng et al. (2019). ABH for vibration isolation                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Evolution logging (physics-driven)
from .evolution_logger import (
    log_cutout_placement,
    log_physics_decision,
)
evo_logger = logging.getLogger("golden_studio.evolution")

# Physical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


class CutoutPurpose(Enum):
    """Purpose of cutout placement - guides WHERE to cut."""
    LOWER_MODE = "lower_mode"          # Cut at antinode to lower specific mode frequency
    PRESERVE_MODE = "preserve_mode"    # Cut at node to NOT affect specific mode
    FLATTEN_RESPONSE = "flatten_response"  # Cut to reduce peak (antinode of problematic mode)
    ENHANCE_COUPLING = "enhance_coupling"  # Cut to improve body contact vibration transfer
    ABH_FOCUS = "abh_focus"            # Edge taper for acoustic black hole energy focusing


@dataclass
class ModeInfo:
    """Information about a single vibrational mode."""
    m: int                    # Mode number in X direction
    n: int                    # Mode number in Y direction  
    frequency_hz: float       # Natural frequency
    mode_shape: np.ndarray    # 2D array of mode amplitudes
    antinodes: List[Tuple[float, float]]  # (x, y) positions of antinodes (|φ| > 0.7)
    nodes: List[Tuple[float, float]]       # (x, y) positions near nodal lines (|φ| < 0.2)
    
    @property
    def mode_id(self) -> str:
        return f"({self.m},{self.n})"


@dataclass
class CutoutSuggestion:
    """Physics-guided suggestion for cutout placement."""
    x: float                  # Suggested X position (normalized 0-1)
    y: float                  # Suggested Y position (normalized 0-1)
    purpose: CutoutPurpose    # Why this location was chosen
    target_mode: Optional[ModeInfo]  # Which mode this affects
    expected_freq_shift: float  # Estimated frequency change (Hz)
    confidence: float         # 0-1, how certain this is a good location
    shape_suggestion: str     # Recommended shape based on local mode curvature
    size_suggestion: Tuple[float, float]  # (width, height) normalized


class ModalAnalyzer:
    """
    Computes mode shapes and identifies optimal cutout locations.
    
    Based on analytical Navier solution for simply-supported rectangular plate:
    φ_mn(x,y) = sin(mπx/L) * sin(nπy/W)
    f_mn = (π/2) * sqrt(D/(ρh)) * ((m/L)² + (n/W)²)
    """
    
    def __init__(
        self,
        length: float,           # Plate length [m]
        width: float,            # Plate width [m]
        thickness: float,        # Plate thickness [m]
        density: float = 450.0,  # kg/m³ (birch plywood)
        E_modulus: float = 12e9, # Pa (birch plywood)
        poisson: float = 0.35,   # Poisson's ratio
        resolution: Tuple[int, int] = (40, 24),  # Higher resolution for antinode detection
    ):
        self.L = length
        self.W = width
        self.h = thickness
        self.rho = density
        self.E = E_modulus
        self.nu = poisson
        self.nx, self.ny = resolution
        
        # Flexural rigidity
        self.D = E_modulus * thickness**3 / (12 * (1 - poisson**2))
        
        # Create evaluation grid
        self.x_norm = np.linspace(0, 1, self.nx)
        self.y_norm = np.linspace(0, 1, self.ny)
        self.X, self.Y = np.meshgrid(self.x_norm, self.y_norm, indexing='ij')
        
        # Cache computed modes
        self._modes: List[ModeInfo] = []
    
    def compute_modes(self, n_modes: int = 15) -> List[ModeInfo]:
        """
        Compute first n_modes natural frequencies and mode shapes.
        
        Returns sorted by frequency (lowest first).
        """
        modes = []
        
        # Compute all combinations up to reasonable m, n
        max_mn = int(np.sqrt(n_modes)) + 3
        
        for m in range(1, max_mn + 1):
            for n in range(1, max_mn + 1):
                # Natural frequency (Navier solution)
                freq = (np.pi / 2) * np.sqrt(self.D / (self.rho * self.h)) * (
                    (m / self.L)**2 + (n / self.W)**2
                )
                
                # Mode shape: sin(mπx/L) * sin(nπy/W)
                shape = np.sin(m * np.pi * self.X) * np.sin(n * np.pi * self.Y)
                
                # Find antinodes (|φ| > 0.7)
                antinodes = self._find_extrema(shape, threshold=0.7)
                
                # Find nodes (|φ| < 0.2)
                nodes = self._find_nodes(shape, threshold=0.2)
                
                modes.append(ModeInfo(
                    m=m, n=n,
                    frequency_hz=freq,
                    mode_shape=shape,
                    antinodes=antinodes,
                    nodes=nodes,
                ))
        
        # Sort by frequency and take first n_modes
        modes.sort(key=lambda m: m.frequency_hz)
        self._modes = modes[:n_modes]
        
        return self._modes
    
    def _find_extrema(self, shape: np.ndarray, threshold: float = 0.7) -> List[Tuple[float, float]]:
        """Find positions where |φ| > threshold (antinodes)."""
        extrema = []
        abs_shape = np.abs(shape)
        max_val = np.max(abs_shape)
        
        if max_val < 1e-10:
            return extrema
        
        # Find local maxima above threshold
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                val = abs_shape[i, j]
                if val > threshold * max_val:
                    # Check if local maximum
                    neighbors = [
                        abs_shape[i-1, j], abs_shape[i+1, j],
                        abs_shape[i, j-1], abs_shape[i, j+1]
                    ]
                    if val >= max(neighbors):
                        extrema.append((self.x_norm[i], self.y_norm[j]))
        
        return extrema
    
    def _find_nodes(self, shape: np.ndarray, threshold: float = 0.2) -> List[Tuple[float, float]]:
        """Find positions where |φ| < threshold (nodal regions)."""
        nodes = []
        abs_shape = np.abs(shape)
        max_val = np.max(abs_shape)
        
        if max_val < 1e-10:
            return nodes
        
        # Sample nodal regions
        for i in range(0, self.nx, 3):  # Sparse sampling
            for j in range(0, self.ny, 3):
                if abs_shape[i, j] < threshold * max_val:
                    nodes.append((self.x_norm[i], self.y_norm[j]))
        
        return nodes
    
    def rayleigh_ritz_frequency_shift(
        self,
        mode: ModeInfo,
        cutout_x: float,
        cutout_y: float,
        cutout_size: float,
        cutout_shape: str = "circle",
    ) -> Dict:
        """
        Calculate frequency shift using Rayleigh-Ritz quotient.
        
        PHYSICS (Fletcher & Rossing 1998, Schleske 2002):
        
        Δω²/ω² = -(1/M)ΔM + (1/K)ΔK
        
        where:
        - ΔM = mass removed = ρ * h * cutout_area * φ²(x,y)
          (weighted by mode amplitude squared at cutout location)
        - ΔK = stiffness change ≈ D * cutout_area * (∇²φ)²
          (weighted by local curvature squared)
        
        For a cutout at an ANTINODE:
        - High φ² → large mass effect → frequency DROPS
        - High (∇²φ)² → large stiffness effect → frequency can RISE
        - Net effect depends on which dominates
        
        For a cutout at a NODE:
        - Low φ² → minimal mass effect  
        - Stiffness effect can still occur
        - Mode relatively unaffected
        
        Args:
            mode: Target vibrational mode
            cutout_x, cutout_y: Position (normalized 0-1)
            cutout_size: Characteristic size (normalized, ~0.02-0.1)
            cutout_shape: Shape affects effective area and stiffness
            
        Returns:
            Dict with delta_f_hz, mass_effect, stiffness_effect, confidence
        """
        # Convert to grid indices
        ix = int(cutout_x * (self.nx - 1))
        iy = int(cutout_y * (self.ny - 1))
        ix = np.clip(ix, 1, self.nx - 2)
        iy = np.clip(iy, 1, self.ny - 2)
        
        # Get mode amplitude at cutout location
        phi = mode.mode_shape[ix, iy]
        phi_sq = phi ** 2
        
        # Compute local curvature (Laplacian approximation)
        # ∇²φ ≈ (φ[i+1,j] - 2φ[i,j] + φ[i-1,j])/dx² + same for y
        dx = self.L / (self.nx - 1)
        dy = self.W / (self.ny - 1)
        
        d2phi_dx2 = (
            mode.mode_shape[ix+1, iy] - 2*mode.mode_shape[ix, iy] + mode.mode_shape[ix-1, iy]
        ) / (dx ** 2)
        d2phi_dy2 = (
            mode.mode_shape[ix, iy+1] - 2*mode.mode_shape[ix, iy] + mode.mode_shape[ix, iy-1]
        ) / (dy ** 2)
        
        laplacian = d2phi_dx2 + d2phi_dy2
        laplacian_sq = laplacian ** 2
        
        # Cutout area (convert normalized size to physical)
        cutout_physical_size = cutout_size * min(self.L, self.W)
        
        # Shape factor (affects effective area and stiffness)
        shape_factors = {
            "circle": (1.0, 1.0),      # Area factor, stiffness factor
            "ellipse": (1.0, 0.9),
            "f_hole": (0.7, 0.6),      # F-hole removes less stiffness
            "kidney": (0.9, 0.75),
            "crescent": (0.6, 0.5),
            "tear": (0.8, 0.65),
            "leaf": (0.85, 0.7),
            "vesica": (0.8, 0.7),
            "s_curve": (0.7, 0.55),
        }
        area_factor, stiffness_factor = shape_factors.get(cutout_shape, (1.0, 1.0))
        
        cutout_area = np.pi * (cutout_physical_size / 2) ** 2 * area_factor
        
        # ════════════════════════════════════════════════════════════════════
        # RAYLEIGH-RITZ FREQUENCY SHIFT
        # ════════════════════════════════════════════════════════════════════
        
        # Total plate mass and stiffness (for normalization)
        plate_area = self.L * self.W
        M_total = self.rho * self.h * plate_area
        
        # Modal mass contribution at cutout (simplified)
        # In a mode, mass contribution is weighted by φ²
        delta_M = self.rho * self.h * cutout_area * phi_sq
        
        # Modal stiffness contribution 
        # Stiffness is related to curvature energy: D * (∇²φ)² integrated
        # For local cutout, we approximate this locally
        delta_K = self.D * cutout_area * laplacian_sq * stiffness_factor
        
        # Effective modal mass and stiffness (rough estimates)
        # For a (m,n) mode, M_eff ≈ M/4, K_eff = ω² * M_eff
        M_eff = M_total / 4
        omega = 2 * np.pi * mode.frequency_hz
        K_eff = omega ** 2 * M_eff
        
        # Frequency shift ratio
        # Δω²/ω² = -ΔM/M_eff + ΔK/K_eff
        mass_effect = -delta_M / M_eff
        stiffness_effect = delta_K / K_eff
        
        delta_omega_sq_ratio = mass_effect + stiffness_effect
        
        # Convert to frequency shift
        # Δf ≈ f * (Δω²/ω²) / 2  (first-order approximation)
        delta_f = mode.frequency_hz * delta_omega_sq_ratio / 2
        
        # Confidence based on grid resolution and cutout size
        grid_coverage = cutout_size * max(self.nx, self.ny)
        confidence = np.clip(grid_coverage / 2, 0.3, 1.0)
        
        return {
            "delta_f_hz": delta_f,
            "mass_effect": mass_effect,
            "stiffness_effect": stiffness_effect,
            "phi_squared": phi_sq,
            "laplacian_squared": laplacian_sq,
            "cutout_area_m2": cutout_area,
            "confidence": confidence,
            "net_effect": "lower" if delta_f < 0 else "raise",
        }
    
    def suggest_cutout_for_mode(
        self,
        target_mode_index: int,
        purpose: CutoutPurpose = CutoutPurpose.LOWER_MODE,
        avoid_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[CutoutSuggestion]:
        """
        Suggest cutout position to affect a specific mode.
        
        Args:
            target_mode_index: Index of mode to affect (0 = fundamental)
            purpose: What we want to achieve
            avoid_positions: Existing cutouts to avoid (min distance constraint)
        
        Returns:
            CutoutSuggestion or None if no good position found
        """
        if not self._modes:
            self.compute_modes()
        
        if target_mode_index >= len(self._modes):
            logger.warning(f"Mode index {target_mode_index} out of range")
            return None
        
        mode = self._modes[target_mode_index]
        avoid = avoid_positions or []
        
        if purpose == CutoutPurpose.LOWER_MODE:
            # Cut at antinode for maximum effect
            candidates = mode.antinodes
        elif purpose == CutoutPurpose.PRESERVE_MODE:
            # Cut at node to preserve this mode
            candidates = mode.nodes
        else:
            candidates = mode.antinodes
        
        if not candidates:
            logger.warning(f"No candidate positions for mode {mode.mode_id}")
            return None
        
        # Filter out positions too close to existing cutouts
        min_distance = 0.15  # Minimum normalized distance
        valid_candidates = []
        
        for cx, cy in candidates:
            # Don't place too close to edges
            if cx < 0.1 or cx > 0.9 or cy < 0.1 or cy > 0.9:
                continue
            
            too_close = False
            for ax, ay in avoid:
                dist = np.sqrt((cx - ax)**2 + (cy - ay)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                valid_candidates.append((cx, cy))
        
        if not valid_candidates:
            logger.info(f"All candidate positions blocked by existing cutouts")
            return None
        
        # Choose best candidate (highest mode amplitude = most effect)
        best_pos = None
        best_amplitude = 0.0
        
        for cx, cy in valid_candidates:
            # Get mode amplitude at this position
            ix = min(int(cx * (self.nx - 1)), self.nx - 1)
            iy = min(int(cy * (self.ny - 1)), self.ny - 1)
            amplitude = np.abs(mode.mode_shape[ix, iy])
            
            if amplitude > best_amplitude:
                best_amplitude = amplitude
                best_pos = (cx, cy)
        
        if best_pos is None:
            return None
        
        # Suggest shape based on mode pattern
        shape = self._suggest_shape_for_mode(mode, best_pos)
        
        # Suggest size (smaller for high frequency modes)
        base_size = 0.05 + 0.05 / (mode.m + mode.n)
        size = (base_size, base_size * 1.2)  # Slightly elongated
        
        # ════════════════════════════════════════════════════════════════════
        # RAYLEIGH-RITZ FREQUENCY SHIFT (replaces empirical formula)
        # Δω²/ω² = -(1/M)ΔM + (1/K)ΔK
        # ════════════════════════════════════════════════════════════════════
        rr_result = self.rayleigh_ritz_frequency_shift(
            mode=mode,
            cutout_x=best_pos[0],
            cutout_y=best_pos[1],
            cutout_size=base_size,
            cutout_shape=shape,
        )
        freq_shift = rr_result["delta_f_hz"]
        confidence = rr_result["confidence"] * best_amplitude
        
        # ════════════════════════════════════════════════════════════════════
        # LOG PHYSICS-GUIDED CUTOUT PLACEMENT
        # ════════════════════════════════════════════════════════════════════
        log_cutout_placement(
            cutout_index=len(avoid) + 1,  # Index based on existing cutouts
            x=best_pos[0],
            y=best_pos[1],
            shape=shape,
            purpose=purpose.value,
            target_mode=mode.mode_id,
            expected_freq_shift=freq_shift,
            confidence=confidence,
            physics_guided=True,
            logger=evo_logger
        )
        
        log_physics_decision(
            decision=f"Place cutout at ({best_pos[0]:.2f}, {best_pos[1]:.2f})",
            reason=f"Mode {mode.mode_id} antinode with amplitude {best_amplitude:.2f}. "
                   f"Rayleigh-Ritz predicts Δf={freq_shift:+.1f}Hz",
            parameters={
                "mode": mode.mode_id,
                "amplitude": best_amplitude,
                "delta_mass_effect": rr_result.get("mass_effect", 0),
                "delta_stiffness_effect": rr_result.get("stiffness_effect", 0),
            },
            reference="Schleske 2002, Fletcher & Rossing 1998",
            logger=evo_logger
        )
        
        return CutoutSuggestion(
            x=best_pos[0],
            y=best_pos[1],
            purpose=purpose,
            target_mode=mode,
            expected_freq_shift=freq_shift,
            confidence=confidence,  # Higher amplitude = more predictable effect
            shape_suggestion=shape,
            size_suggestion=size,
        )
    
    def _suggest_shape_for_mode(self, mode: ModeInfo, position: Tuple[float, float]) -> str:
        """
        Suggest cutout shape based on local mode curvature.
        
        - High curvature (high m,n) → smaller, rounder shapes
        - Low modes → can use elongated shapes
        - Near edges → organic shapes for ABH effect
        """
        x, y = position
        
        # Edge proximity → organic shapes (ABH benefit)
        edge_dist = min(x, 1-x, y, 1-y)
        if edge_dist < 0.15:
            return np.random.choice(["crescent", "tear", "leaf"])
        
        # High-order modes → round shapes
        if mode.m + mode.n > 5:
            return np.random.choice(["circle", "ellipse"])
        
        # Low modes with asymmetry → f-hole style
        if mode.m != mode.n:
            return np.random.choice(["f_hole", "s_curve", "kidney"])
        
        # Default: organic ellipse
        return np.random.choice(["ellipse", "vesica", "leaf"])
    
    def suggest_cutouts_for_flatness(
        self,
        freq_range: Tuple[float, float] = (20.0, 200.0),
        target_variation_db: float = 6.0,
        max_cutouts: int = 4,
        existing_cutouts: Optional[List[Tuple[float, float]]] = None,
    ) -> List[CutoutSuggestion]:
        """
        Suggest cutouts to flatten frequency response in target range.
        
        Strategy:
        1. Find modes with frequencies in target range
        2. Identify modes that would cause peaks (high coupling to body)
        3. Suggest cutouts at those modes' antinodes to shift/dampen them
        
        Args:
            freq_range: Target frequency range [Hz]
            target_variation_db: Acceptable response variation
            max_cutouts: Maximum number of cutouts to suggest
            existing_cutouts: Positions of existing cutouts
        
        Returns:
            List of CutoutSuggestion objects
        """
        if not self._modes:
            self.compute_modes()
        
        suggestions = []
        avoid = list(existing_cutouts) if existing_cutouts else []
        
        # Find modes in target range
        target_modes = [
            (i, m) for i, m in enumerate(self._modes)
            if freq_range[0] <= m.frequency_hz <= freq_range[1]
        ]
        
        if not target_modes:
            logger.warning(f"No modes found in range {freq_range} Hz")
            return suggestions
        
        # Sort by frequency (target low modes first for bigger impact)
        target_modes.sort(key=lambda x: x[1].frequency_hz)
        
        for mode_idx, mode in target_modes[:max_cutouts]:
            suggestion = self.suggest_cutout_for_mode(
                target_mode_index=mode_idx,
                purpose=CutoutPurpose.FLATTEN_RESPONSE,
                avoid_positions=avoid,
            )
            
            if suggestion:
                suggestions.append(suggestion)
                avoid.append((suggestion.x, suggestion.y))
        
        return suggestions
    
    def get_mode_amplitude_at(self, x: float, y: float, mode_index: int) -> float:
        """Get mode amplitude at a specific position."""
        if not self._modes or mode_index >= len(self._modes):
            return 0.0
        
        mode = self._modes[mode_index]
        ix = min(int(x * (self.nx - 1)), self.nx - 1)
        iy = min(int(y * (self.ny - 1)), self.ny - 1)
        
        return mode.mode_shape[ix, iy]
    
    def score_cutout_position(
        self,
        x: float, y: float,
        target_freq_range: Tuple[float, float] = (20.0, 200.0),
    ) -> float:
        """
        Score a potential cutout position (0-1, higher = better).
        
        Good positions are at antinodes of modes in the target frequency range.
        """
        if not self._modes:
            self.compute_modes()
        
        score = 0.0
        count = 0
        
        for mode in self._modes:
            if target_freq_range[0] <= mode.frequency_hz <= target_freq_range[1]:
                amplitude = abs(self.get_mode_amplitude_at(x, y, self._modes.index(mode)))
                score += amplitude
                count += 1
        
        if count == 0:
            return 0.5  # Neutral score if no modes in range
        
        return min(1.0, score / count)


def create_physics_guided_cutout(
    genome,  # PlateGenome (imported in function to avoid circular import)
    analyzer: ModalAnalyzer,
    purpose: CutoutPurpose = CutoutPurpose.FLATTEN_RESPONSE,
    existing_cutouts: Optional[List] = None,
    n_cutout: int = 1,  # Which cutout number (for variety)
) -> Optional[dict]:
    """
    Create a physics-guided cutout with INTELLIGENT VARIATION.
    
    PHYSICS-BASED DECISION MAKING:
    
    1. POSITION varies based on:
       - Target mode antinodes (different modes = different positions)
       - Existing cutouts (spread them out, avoid clustering)
       - Body zones (spine vs head vs feet)
    
    2. SIZE varies based on:
       - Mode wavelength at that position (λ/4 to λ/2 effective)
       - Desired frequency shift magnitude
       - Structural constraints (not too big near edges)
    
    3. SHAPE varies based on:
       - Local mode curvature direction (ellipse aligned with curvature)
       - Frequency shift direction (leaf/vesica for asymmetric tuning)
       - Zone purpose (f_hole for resonance, slot for bass)
    
    4. ROTATION varies based on:
       - Mode nodal line direction (perpendicular for max effect)
       - Symmetry requirements (mirrored pairs)
    
    Args:
        genome: PlateGenome to add cutout to
        analyzer: ModalAnalyzer with computed modes
        purpose: Goal of the cutout
        existing_cutouts: List of existing CutoutGene objects
        n_cutout: Which cutout number (1, 2, 3...) for variety
    
    Returns:
        Dict with cutout parameters, or None if no good position found
    """
    # Get existing cutout positions
    avoid = []
    if existing_cutouts:
        for c in existing_cutouts:
            avoid.append((c.x, c.y))
    
    # Ensure modes are computed
    if not analyzer._modes:
        analyzer.compute_modes(n_modes=15)
    
    modes = analyzer._modes
    if not modes:
        return None
    
    # ════════════════════════════════════════════════════════════════════════
    # 1. SELECT TARGET MODE based on cutout number (VARIETY!)
    # First cutout targets mode 1, second targets mode 2, etc.
    # ════════════════════════════════════════════════════════════════════════
    target_mode_idx = (n_cutout - 1) % len(modes)
    target_mode = modes[target_mode_idx]
    
    # ════════════════════════════════════════════════════════════════════════
    # 2. FIND BEST ANTINODE for this mode (avoiding existing cutouts)
    # ════════════════════════════════════════════════════════════════════════
    antinodes = target_mode.antinodes
    
    if not antinodes:
        # Fallback: find max amplitude location
        shape = target_mode.mode_shape
        max_idx = np.unravel_index(np.argmax(np.abs(shape)), shape.shape)
        x = analyzer.x_norm[max_idx[0]]
        y = analyzer.y_norm[max_idx[1]]
        antinodes = [(x, y)]
    
    # Score antinodes - prefer ones away from existing cutouts
    best_antinode = None
    best_score = -np.inf
    
    for ax, ay in antinodes:
        # Distance to nearest existing cutout
        min_dist = 1.0
        for ex, ey in avoid:
            dist = np.sqrt((ax - ex)**2 + (ay - ey)**2)
            min_dist = min(min_dist, dist)
        
        # Distance from edge (prefer not too close)
        edge_dist = min(ax, 1-ax, ay, 1-ay)
        
        # Mode amplitude at this point
        amp = abs(analyzer.get_mode_amplitude_at(ax, ay, target_mode_idx))
        
        # Combined score
        score = amp * min_dist * edge_dist
        
        if score > best_score:
            best_score = score
            best_antinode = (ax, ay)
    
    if best_antinode is None:
        return None
    
    x, y = best_antinode
    
    # ════════════════════════════════════════════════════════════════════════
    # 3. CALCULATE SIZE based on mode wavelength and frequency shift goal
    # ════════════════════════════════════════════════════════════════════════
    # Mode wavelength: λ = 2L/m (in X) or 2W/n (in Y)
    lambda_x = 2.0 / target_mode.m  # Normalized
    lambda_y = 2.0 / target_mode.n  # Normalized
    
    # Cutout should be λ/8 to λ/4 for effective tuning
    base_width = lambda_x * np.random.uniform(0.08, 0.20)
    base_height = lambda_y * np.random.uniform(0.08, 0.20)
    
    # Clamp to reasonable range
    width = np.clip(base_width, 0.02, 0.12)
    height = np.clip(base_height, 0.02, 0.12)
    
    # ════════════════════════════════════════════════════════════════════════
    # 4. CHOOSE SHAPE based on zone and mode character
    # ════════════════════════════════════════════════════════════════════════
    # Spine zone (y=0.3-0.7): slots/ellipse for bass reinforcement
    # Head zone (y>0.7): f_hole/vesica for detail
    # Feet zone (y<0.3): crescent/kidney for vibration
    
    if y > 0.65:  # Head zone
        shapes = ["f_hole", "vesica", "ellipse", "leaf", "tear"]
    elif y < 0.35:  # Feet zone  
        shapes = ["crescent", "kidney", "ellipse", "wave"]
    else:  # Spine zone
        shapes = ["ellipse", "slot", "s_curve", "leaf", "spiral"]
    
    # Higher modes (more complex) → more complex shapes
    if target_mode.m + target_mode.n > 4:
        shapes = ["spiral", "wave", "freeform", "s_curve"] + shapes[:2]
    
    shape = np.random.choice(shapes)
    
    # ════════════════════════════════════════════════════════════════════════
    # 5. CALCULATE ROTATION based on local mode curvature
    # ════════════════════════════════════════════════════════════════════════
    # Get local gradient direction
    ix = int(x * (analyzer.nx - 1))
    iy = int(y * (analyzer.ny - 1))
    ix = np.clip(ix, 1, analyzer.nx - 2)
    iy = np.clip(iy, 1, analyzer.ny - 2)
    
    shape_arr = target_mode.mode_shape
    grad_x = (shape_arr[min(ix+1, analyzer.nx-1), iy] - shape_arr[max(ix-1, 0), iy]) / 2
    grad_y = (shape_arr[ix, min(iy+1, analyzer.ny-1)] - shape_arr[ix, max(iy-1, 0)]) / 2
    
    # Rotation perpendicular to gradient (along nodal line) for max effect
    if abs(grad_x) + abs(grad_y) > 0.01:
        rotation = np.arctan2(grad_y, grad_x) + np.pi/2
    else:
        rotation = np.random.uniform(0, 2*np.pi)
    
    # Add some randomness
    rotation += np.random.normal(0, 0.2)
    
    # ════════════════════════════════════════════════════════════════════════
    # 6. CALCULATE EXPECTED FREQUENCY SHIFT (for logging)
    # ════════════════════════════════════════════════════════════════════════
    rr = analyzer.rayleigh_ritz_frequency_shift(
        target_mode, x, y, 
        cutout_size=(width + height) / 2,
        cutout_shape=shape
    )
    delta_f = rr.get("delta_f_hz", 0.0) if isinstance(rr, dict) else 0.0
    
    # Log the physics decision
    log_cutout_placement(
        x=x, y=y,
        shape=shape,
        purpose=purpose.value,
        target_mode=target_mode.mode_id,
        expected_freq_shift=delta_f,
        confidence=best_score,
        logger=evo_logger
    )
    
    log_physics_decision(
        decision=f"Place cutout at ({x:.2f}, {y:.2f})",
        reason=f"Mode {target_mode.mode_id} antinode with amplitude {analyzer.get_mode_amplitude_at(x, y, target_mode_idx):.2f}. "
               f"Rayleigh-Ritz predicts Δf={delta_f:.1f}Hz",
        reference="Schleske 2002, Fletcher & Rossing 1998",
        logger=evo_logger
    )
    
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": rotation,
        "shape": shape,
        "corner_radius": np.random.uniform(0.1, 0.5),
        "aspect_bias": np.random.uniform(0.7, 1.5),
    }


def create_physics_guided_groove(
    genome,
    analyzer: ModalAnalyzer,
    existing_grooves: Optional[List] = None,
    n_groove: int = 1,
) -> Optional[dict]:
    """
    Create a physics-guided groove with INTELLIGENT PARAMETERS.
    
    GROOVES vs CUTOUTS - When to use each:
    
    CUTOUTS (fori passanti):
    - Large frequency shifts (> 5Hz)
    - Major mode restructuring
    - Helmholtz resonance (like f-holes)
    - Mass removal dominates
    
    GROOVES (scanalature):
    - Fine frequency tuning (< 5Hz)
    - Stiffness reduction without mass removal
    - Graduation like violin plates
    - Local thickness thinning
    
    PHYSICS-BASED GROOVE PARAMETERS:
    
    1. DEPTH varies based on:
       - Required stiffness change (deeper = more effect)
       - I ∝ h³, so small depth changes → big stiffness changes
       - Typical: 20-40% of thickness
    
    2. LENGTH varies based on:
       - Mode wavelength (longer groove = wider frequency effect)
       - Zone coverage (longer in spine for broad bass)
    
    3. WIDTH varies based on:
       - Fresa tool size (3mm, 5mm, 6mm, 8mm standard)
       - Precision needed (narrower for fine tuning)
    
    4. ANGLE varies based on:
       - Cross grain: perpendicular to mode nodal lines
       - Diagonal: for torsional mode tuning
       - Parallel: for bending mode tuning
    
    Args:
        genome: PlateGenome
        analyzer: ModalAnalyzer with computed modes
        existing_grooves: List of existing GrooveGene objects
        n_groove: Which groove number (1, 2, 3...) for variety
    
    Returns:
        Dict with groove parameters, or None
    """
    # Ensure modes computed
    if not analyzer._modes:
        analyzer.compute_modes(n_modes=15)
    
    modes = analyzer._modes
    if not modes:
        return None
    
    # Get existing positions
    avoid = []
    if existing_grooves:
        for g in existing_grooves:
            avoid.append((g.x, g.y))
    
    # ════════════════════════════════════════════════════════════════════════
    # 1. SELECT TARGET MODE (different from cutouts - target HIGHER modes)
    # Grooves are for fine tuning, so target modes 3-8 typically
    # ════════════════════════════════════════════════════════════════════════
    target_mode_idx = min(2 + n_groove, len(modes) - 1)  # Start from mode 3
    target_mode = modes[target_mode_idx]
    
    # ════════════════════════════════════════════════════════════════════════
    # 2. POSITION: Grooves go along NODAL LINES for mode isolation
    # Or across antinodes for maximum stiffness change
    # ════════════════════════════════════════════════════════════════════════
    # Strategy: alternate between nodal and antinode placement
    if n_groove % 2 == 0:
        # Place near nodal line (minimal effect on this mode, affects others)
        candidates = target_mode.nodes
    else:
        # Place at antinode (maximum effect on this mode)
        candidates = target_mode.antinodes
    
    if not candidates:
        # Fallback: random in spine zone
        x = np.random.uniform(0.25, 0.75)
        y = np.random.uniform(0.35, 0.65)
    else:
        # Pick one away from existing grooves
        best_candidate = None
        best_dist = -1
        for cx, cy in candidates:
            min_dist = 1.0
            for ex, ey in avoid:
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                min_dist = min(min_dist, dist)
            if min_dist > best_dist:
                best_dist = min_dist
                best_candidate = (cx, cy)
        
        if best_candidate:
            x, y = best_candidate
        else:
            x, y = candidates[0]
    
    # Keep in valid range
    x = np.clip(x, 0.15, 0.85)
    y = np.clip(y, 0.20, 0.80)
    
    # ════════════════════════════════════════════════════════════════════════
    # 3. LENGTH based on mode wavelength
    # ════════════════════════════════════════════════════════════════════════
    lambda_x = 2.0 / target_mode.m
    lambda_y = 2.0 / target_mode.n
    
    # Groove length: λ/4 to λ/2
    length = np.random.uniform(0.5, 1.0) * min(lambda_x, lambda_y) / 2
    length = np.clip(length, 0.05, 0.20)
    
    # ════════════════════════════════════════════════════════════════════════
    # 4. ANGLE based on mode character
    # ════════════════════════════════════════════════════════════════════════
    # For bending modes (low m or n): perpendicular to dominant direction
    # For torsional modes (m≈n): diagonal
    
    if target_mode.m > target_mode.n:
        # Dominant X mode: groove perpendicular (vertical)
        base_angle = np.pi / 2
    elif target_mode.n > target_mode.m:
        # Dominant Y mode: groove horizontal
        base_angle = 0.0
    else:
        # Torsional: diagonal
        base_angle = np.pi / 4 * np.random.choice([-1, 1])
    
    angle = base_angle + np.random.normal(0, 0.2)
    
    # ════════════════════════════════════════════════════════════════════════
    # 5. DEPTH and WIDTH based on required tuning magnitude
    # ════════════════════════════════════════════════════════════════════════
    # Higher mode number → needs less depth (more sensitive)
    mode_order = target_mode.m + target_mode.n
    depth = np.random.uniform(0.20, 0.45) / (1 + 0.1 * mode_order)
    depth = np.clip(depth, 0.15, 0.40)
    
    # Standard fresa widths
    width_options = [3.0, 4.0, 5.0, 6.0, 8.0]
    # Narrower for higher modes (more precision)
    if mode_order > 5:
        width_mm = np.random.choice(width_options[:3])
    else:
        width_mm = np.random.choice(width_options)
    
    return {
        "x": x,
        "y": y,
        "length": length,
        "angle": angle,
        "depth": depth,
        "width_mm": width_mm,
    }


def decide_cutout_vs_groove(
    genome,
    analyzer: ModalAnalyzer,
    target_freq_shift: float,
    max_cutouts: int,
    max_grooves: int,
    existing_cutouts: Optional[List] = None,
    existing_grooves: Optional[List] = None,
) -> Tuple[str, Optional[dict]]:
    """
    Intelligently decide whether to add a CUTOUT or GROOVE.
    
    DECISION LOGIC (based on lutherie principles):
    
    | Δf needed | Zone     | Recommendation |
    |-----------|----------|----------------|
    | > 10 Hz   | Any      | CUTOUT (large mass effect) |
    | 5-10 Hz   | Spine    | GROOVE (fine tune bass) |
    | 5-10 Hz   | Head     | CUTOUT (detail clarity) |
    | < 5 Hz    | Any      | GROOVE (precise adjustment) |
    | Any       | Edge     | CUTOUT (ABH potential) |
    
    Args:
        genome: PlateGenome
        analyzer: ModalAnalyzer
        target_freq_shift: Desired frequency change magnitude (Hz)
        max_cutouts, max_grooves: Limits from config
        existing_cutouts, existing_grooves: Current features
        
    Returns:
        ("cutout", params) or ("groove", params) or ("none", None)
    """
    n_existing_cutouts = len(existing_cutouts) if existing_cutouts else 0
    n_existing_grooves = len(existing_grooves) if existing_grooves else 0
    
    can_add_cutout = max_cutouts > 0 and n_existing_cutouts < max_cutouts
    can_add_groove = max_grooves > 0 and n_existing_grooves < max_grooves
    
    if not can_add_cutout and not can_add_groove:
        return ("none", None)
    
    # ════════════════════════════════════════════════════════════════════════
    # DECISION BASED ON FREQUENCY SHIFT MAGNITUDE
    # ════════════════════════════════════════════════════════════════════════
    
    if abs(target_freq_shift) > 10.0:
        # Large shift needed → CUTOUT (mass removal)
        if can_add_cutout:
            params = create_physics_guided_cutout(
                genome, analyzer,
                existing_cutouts=existing_cutouts,
                n_cutout=n_existing_cutouts + 1
            )
            return ("cutout", params)
    
    elif abs(target_freq_shift) < 5.0:
        # Small shift → GROOVE (stiffness fine-tune)
        if can_add_groove:
            params = create_physics_guided_groove(
                genome, analyzer,
                existing_grooves=existing_grooves,
                n_groove=n_existing_grooves + 1
            )
            return ("groove", params)
    
    # Medium shift: depends on what's available
    if can_add_cutout and (not can_add_groove or np.random.random() < 0.6):
        params = create_physics_guided_cutout(
            genome, analyzer,
            existing_cutouts=existing_cutouts,
            n_cutout=n_existing_cutouts + 1
        )
        return ("cutout", params)
    elif can_add_groove:
        params = create_physics_guided_groove(
            genome, analyzer,
            existing_grooves=existing_grooves,
            n_groove=n_existing_grooves + 1
        )
        return ("groove", params)
    
    return ("none", None)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-EXCITER MODAL COUPLING OPTIMIZER
# Based on: Bai & Liu 2004 "Genetic algorithm for exciter placement"
#           Lu 2012, Shen 2016 "Multi-exciter flat panel speakers"
#           Sum & Pan 2000 "Modal cross-coupling"
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExciterCouplingScore:
    """Result of multi-exciter modal coupling analysis."""
    position: Tuple[float, float]     # (x, y) normalized
    broadband_score: float            # Coupling across frequency range [0-1]
    antinode_score: float             # Proximity to modal antinodes [0-1]
    uniformity_score: float           # Contribution to uniform coverage [0-1]
    cross_coupling_penalty: float     # Penalty for mode cross-talk [0-1]
    modal_breakdown: Dict[str, float] # Per-mode coupling strength
    net_score: float                  # Combined score [0-1]


class MultiExciterOptimizer:
    """
    Optimize multi-exciter placement for broadband modal coupling.
    
    PHYSICS (Bai & Liu 2004):
    - Exciter at antinode of mode → maximum coupling to that mode
    - Multiple exciters can "fill in" frequency response gaps
    - Optimal placement minimizes cross-mode interference
    - Target: flat response across 20-200 Hz therapeutic range
    
    GENETIC ALGORITHM APPROACH:
    - Chromosome: [x1, y1, x2, y2, ..., xn, yn] exciter positions
    - Fitness: broadband coupling + uniformity - cross-coupling penalty
    - Constraints: symmetric pairs for stereo, minimum distance
    """
    
    def __init__(
        self,
        analyzer: ModalAnalyzer,
        n_exciters: int = 4,
        symmetric: bool = True,  # Force L/R symmetry for stereo
        freq_range: Tuple[float, float] = (20.0, 200.0),
    ):
        self.analyzer = analyzer
        self.n_exciters = n_exciters
        self.symmetric = symmetric
        self.freq_range = freq_range
        
        # Ensure modes are computed
        if not analyzer._modes:
            analyzer.compute_modes(n_modes=15)
        
        # Filter modes in frequency range
        self.target_modes = [
            m for m in analyzer._modes
            if freq_range[0] <= m.frequency_hz <= freq_range[1]
        ]
        
        if not self.target_modes:
            logger.warning(f"No modes in frequency range {freq_range} Hz")
    
    def score_exciter_position(
        self,
        x: float,
        y: float,
        other_exciters: Optional[List[Tuple[float, float]]] = None,
    ) -> ExciterCouplingScore:
        """
        Score a single exciter position for modal coupling.
        
        Args:
            x, y: Position (normalized 0-1)
            other_exciters: Existing exciter positions
            
        Returns:
            ExciterCouplingScore with detailed breakdown
        """
        others = other_exciters or []
        modal_breakdown = {}
        
        # ════════════════════════════════════════════════════════════════════
        # 1. ANTINODE PROXIMITY SCORE
        # Higher mode amplitude at exciter → better coupling
        # ════════════════════════════════════════════════════════════════════
        antinode_scores = []
        for mode in self.target_modes:
            amp = abs(self.analyzer.get_mode_amplitude_at(
                x, y, self.analyzer._modes.index(mode)
            ))
            modal_breakdown[mode.mode_id] = amp
            antinode_scores.append(amp)
        
        antinode_score = np.mean(antinode_scores) if antinode_scores else 0.0
        
        # ════════════════════════════════════════════════════════════════════
        # 2. BROADBAND COVERAGE SCORE
        # Exciter should couple to modes across the frequency range
        # Penalize if only coupling to low OR high frequencies
        # ════════════════════════════════════════════════════════════════════
        if self.target_modes:
            freqs = np.array([m.frequency_hz for m in self.target_modes])
            amps = np.array(antinode_scores)
            
            # Weight by frequency spread
            freq_range = freqs.max() - freqs.min()
            if freq_range > 0:
                # Ideal: high coupling at both low and high frequencies
                low_freq_coupling = np.mean(amps[freqs < np.median(freqs)])
                high_freq_coupling = np.mean(amps[freqs >= np.median(freqs)])
                broadband_score = np.sqrt(low_freq_coupling * high_freq_coupling)
            else:
                broadband_score = antinode_score
        else:
            broadband_score = 0.5
        
        # ════════════════════════════════════════════════════════════════════
        # 3. UNIFORMITY SCORE (spatial coverage with other exciters)
        # Exciters should cover different regions of the plate
        # ════════════════════════════════════════════════════════════════════
        if others:
            min_distance = float('inf')
            for ox, oy in others:
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                min_distance = min(min_distance, dist)
            
            # Ideal min distance: ~0.3-0.5 normalized
            if min_distance < 0.1:
                uniformity_score = 0.2  # Too close
            elif min_distance < 0.2:
                uniformity_score = 0.5
            elif min_distance < 0.4:
                uniformity_score = 1.0  # Optimal
            else:
                uniformity_score = 0.8  # Good but maybe too spread
        else:
            # First exciter - prefer not at exact center
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            uniformity_score = min(1.0, center_dist * 2 + 0.3)
        
        # ════════════════════════════════════════════════════════════════════
        # 4. CROSS-COUPLING PENALTY (Sum & Pan 2000)
        # Multiple exciters at same mode antinode → phase interference
        # ════════════════════════════════════════════════════════════════════
        cross_coupling = 0.0
        if others:
            for ox, oy in others:
                # Check if both at same antinode
                for mode in self.target_modes:
                    amp_this = abs(self.analyzer.get_mode_amplitude_at(
                        x, y, self.analyzer._modes.index(mode)
                    ))
                    amp_other = abs(self.analyzer.get_mode_amplitude_at(
                        ox, oy, self.analyzer._modes.index(mode)
                    ))
                    
                    # Both high amplitude at same mode → interference risk
                    if amp_this > 0.7 and amp_other > 0.7:
                        cross_coupling += 0.1
            
            cross_coupling = min(1.0, cross_coupling)
        
        # ════════════════════════════════════════════════════════════════════
        # 5. COMBINED NET SCORE
        # ════════════════════════════════════════════════════════════════════
        net_score = (
            antinode_score * 0.30 +
            broadband_score * 0.35 +
            uniformity_score * 0.25 -
            cross_coupling * 0.10
        )
        
        return ExciterCouplingScore(
            position=(x, y),
            broadband_score=broadband_score,
            antinode_score=antinode_score,
            uniformity_score=uniformity_score,
            cross_coupling_penalty=cross_coupling,
            modal_breakdown=modal_breakdown,
            net_score=np.clip(net_score, 0, 1),
        )
    
    def optimize_positions(
        self,
        n_generations: int = 50,
        population_size: int = 30,
        mutation_rate: float = 0.15,
    ) -> List[Tuple[float, float]]:
        """
        Genetic algorithm optimization for exciter positions.
        
        Based on Bai & Liu 2004:
        - Chromosome: flattened list of (x, y) positions
        - Selection: tournament
        - Crossover: single-point
        - Mutation: Gaussian perturbation
        
        Args:
            n_generations: Number of evolution iterations
            population_size: Number of individuals
            mutation_rate: Probability of mutation per gene
            
        Returns:
            List of optimized (x, y) positions
        """
        n_pos = self.n_exciters if not self.symmetric else self.n_exciters // 2
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = []
            for _ in range(n_pos):
                x = np.random.uniform(0.15, 0.85)
                y = np.random.uniform(0.15, 0.85)
                individual.append((x, y))
            population.append(individual)
        
        def evaluate_individual(positions):
            """Evaluate fitness of exciter configuration."""
            if self.symmetric:
                # Expand to symmetric pairs
                full_positions = []
                for x, y in positions:
                    full_positions.append((x, y))        # Left
                    full_positions.append((1-x, y))      # Right mirror
            else:
                full_positions = positions
            
            total_score = 0.0
            for i, (x, y) in enumerate(full_positions):
                others = full_positions[:i] + full_positions[i+1:]
                score = self.score_exciter_position(x, y, others)
                total_score += score.net_score
            
            return total_score / len(full_positions)
        
        # Evolution loop
        best_fitness = 0.0
        best_individual = population[0]
        
        for gen in range(n_generations):
            # Evaluate
            fitnesses = [evaluate_individual(ind) for ind in population]
            
            # Track best
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = population[max_idx].copy()
            
            # Selection (tournament, k=3)
            new_population = []
            for _ in range(population_size):
                candidates = np.random.choice(len(population), size=3, replace=False)
                winner = candidates[np.argmax([fitnesses[c] for c in candidates])]
                new_population.append(population[winner].copy())
            
            # Crossover (single-point)
            for i in range(0, len(new_population) - 1, 2):
                if np.random.random() < 0.7:
                    point = np.random.randint(1, n_pos)
                    p1, p2 = new_population[i], new_population[i+1]
                    new_population[i] = p1[:point] + p2[point:]
                    new_population[i+1] = p2[:point] + p1[point:]
            
            # Mutation (Gaussian perturbation)
            for ind in new_population:
                for j in range(len(ind)):
                    if np.random.random() < mutation_rate:
                        x, y = ind[j]
                        x = np.clip(x + np.random.normal(0, 0.1), 0.1, 0.9)
                        y = np.clip(y + np.random.normal(0, 0.1), 0.1, 0.9)
                        ind[j] = (x, y)
            
            # Elitism: keep best
            new_population[0] = best_individual.copy()
            population = new_population
        
        # Expand best to full positions
        if self.symmetric:
            result = []
            for x, y in best_individual:
                result.append((x, y))
                result.append((1-x, y))
            return result
        else:
            return best_individual
    
    def suggest_optimal_positions(
        self,
        quick: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Suggest optimal exciter positions.
        
        Args:
            quick: If True, use fewer generations for faster result
            
        Returns:
            List of (x, y) positions for exciters
        """
        if quick:
            return self.optimize_positions(n_generations=20, population_size=15)
        else:
            return self.optimize_positions(n_generations=100, population_size=50)
    
    def evaluate_configuration(
        self,
        positions: List[Tuple[float, float]],
    ) -> Dict:
        """
        Evaluate a given exciter configuration.
        
        Returns detailed metrics for the configuration.
        """
        scores = []
        for i, (x, y) in enumerate(positions):
            others = positions[:i] + positions[i+1:]
            score = self.score_exciter_position(x, y, others)
            scores.append(score)
        
        avg_broadband = np.mean([s.broadband_score for s in scores])
        avg_antinode = np.mean([s.antinode_score for s in scores])
        avg_uniformity = np.mean([s.uniformity_score for s in scores])
        total_cross = sum([s.cross_coupling_penalty for s in scores])
        overall = np.mean([s.net_score for s in scores])
        
        return {
            "positions": positions,
            "overall_score": overall,
            "broadband_coupling": avg_broadband,
            "antinode_coupling": avg_antinode,
            "spatial_uniformity": avg_uniformity,
            "cross_coupling_penalty": total_cross,
            "per_exciter_scores": scores,
            "n_target_modes": len(self.target_modes),
            "freq_range": self.freq_range,
        }
