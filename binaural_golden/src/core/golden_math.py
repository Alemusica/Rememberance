"""
Golden Ratio Mathematical Functions
====================================

Pure math functions based on φ (phi) for use throughout the application.
No UI dependencies - just math.
"""

import numpy as np
from typing import Callable

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHI: float = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_CONJUGATE: float = PHI - 1      # 0.618033988749895
GOLDEN_ANGLE_DEG: float = 360.0 / (PHI * PHI)  # 137.5077640500378546°
GOLDEN_ANGLE_RAD: float = np.radians(GOLDEN_ANGLE_DEG)


# ══════════════════════════════════════════════════════════════════════════════
# FADE CURVES
# ══════════════════════════════════════════════════════════════════════════════

def golden_fade(t: float, fade_in: bool = True) -> float:
    """
    Golden ratio based fade curve using φ exponent.
    
    Creates a smooth S-curve that follows golden proportions:
    - Slow start (breathing in)
    - Natural acceleration through middle  
    - Slow end (settling)
    
    Args:
        t: Progress 0.0 to 1.0
        fade_in: True for fade in, False for fade out
        
    Returns:
        Amplitude multiplier 0.0 to 1.0
    """
    t = max(0.0, min(1.0, t))  # Clamp to 0-1
    
    if fade_in:
        # Raised cosine with golden exponent for smooth start/end
        base = (1 - np.cos(t * np.pi)) / 2.0
        return float(base ** PHI_CONJUGATE)
    else:
        # Fade out: mirror of fade in
        base = (1 - np.cos((1 - t) * np.pi)) / 2.0
        return 1.0 - float(base ** PHI_CONJUGATE)


def golden_ease(t: float, ease_type: str = "in_out") -> float:
    """
    Golden ratio easing functions for animations.
    
    Args:
        t: Progress 0.0 to 1.0
        ease_type: "in", "out", or "in_out"
        
    Returns:
        Eased value 0.0 to 1.0
    """
    t = max(0.0, min(1.0, t))
    
    if ease_type == "in":
        return float(t ** PHI)
    elif ease_type == "out":
        return 1.0 - float((1 - t) ** PHI)
    else:  # in_out
        if t < 0.5:
            return float((2 * t) ** PHI) / 2
        else:
            return 1.0 - float((2 * (1 - t)) ** PHI) / 2


def golden_interpolate(start: float, end: float, t: float, 
                       ease: str = "in_out") -> float:
    """
    Interpolate between two values using golden easing.
    
    Args:
        start: Starting value
        end: Ending value
        t: Progress 0.0 to 1.0
        ease: Easing type
        
    Returns:
        Interpolated value
    """
    eased_t = golden_ease(t, ease)
    return start + (end - start) * eased_t


# ══════════════════════════════════════════════════════════════════════════════
# PHASE BOUNDARIES (Golden proportions for multi-phase sequences)
# ══════════════════════════════════════════════════════════════════════════════

def golden_phase_boundaries(num_phases: int) -> list[float]:
    """
    Calculate phase boundaries based on golden proportions.
    
    For 5 phases, uses symmetric pattern 1:φ:φ:φ:1
    For other numbers, distributes using φ-based ratios.
    
    Args:
        num_phases: Number of phases (2-10)
        
    Returns:
        List of boundary values [0, b1, b2, ..., 1.0]
    """
    if num_phases == 5:
        # Special case: symmetric 1:φ:φ:φ:1
        unit = 1.0 / (2.0 + 3.0 * PHI)
        phi_unit = PHI * unit
        return [
            0.0,
            unit,                          # ~0.146
            unit + phi_unit,               # ~0.382
            unit + 2 * phi_unit,           # ~0.618
            unit + 3 * phi_unit,           # ~0.854
            1.0
        ]
    else:
        # General case: equal φ-weighted distribution
        weights = [PHI_CONJUGATE ** (i % 3) for i in range(num_phases)]
        total = sum(weights)
        normalized = [w / total for w in weights]
        
        boundaries = [0.0]
        cumsum = 0.0
        for w in normalized:
            cumsum += w
            boundaries.append(cumsum)
        boundaries[-1] = 1.0  # Ensure exact 1.0
        return boundaries


# ══════════════════════════════════════════════════════════════════════════════
# FIBONACCI / GOLDEN SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


def fibonacci_harmonics(fundamental: float, num_harmonics: int = 5) -> list[float]:
    """
    Generate harmonic frequencies based on Fibonacci ratios.
    
    Args:
        fundamental: Base frequency in Hz
        num_harmonics: Number of harmonics to generate
        
    Returns:
        List of frequencies [f, 2f, 3f, 5f, 8f, ...]
    """
    harmonics = []
    for i in range(min(num_harmonics, len(FIBONACCI))):
        harmonics.append(fundamental * FIBONACCI[i])
    return harmonics


def phi_amplitude_decay(n: int) -> float:
    """
    Calculate amplitude for nth harmonic using φ decay.
    
    Args:
        n: Harmonic number (0-based)
        
    Returns:
        Amplitude multiplier (1.0 for fundamental, decreasing for harmonics)
    """
    return PHI_CONJUGATE ** n


def golden_phases(num_phases: int) -> list[float]:
    """
    Generate phases rotated by golden angle.
    
    Args:
        num_phases: Number of phase values needed
        
    Returns:
        List of phases in radians [0, φ_angle, 2*φ_angle, ...]
    """
    return [GOLDEN_ANGLE_RAD * i for i in range(num_phases)]
