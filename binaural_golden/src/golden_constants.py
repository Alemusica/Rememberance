#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GOLDEN CONSTANTS - CENTRALIZED MODULE                     ║
║                                                                              ║
║   Single source of truth for all sacred constants, ratios, and sequences    ║
║   used throughout the Golden Sound Studio application.                       ║
║                                                                              ║
║   "One module to rule them all - no more duplication"                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import List, Callable

# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN RATIO CONSTANTS (Maximum Precision)
# ══════════════════════════════════════════════════════════════════════════════

# Golden Ratio φ = (1 + √5) / 2
PHI: float = (1 + np.sqrt(5)) / 2  # 1.618033988749895

# Golden Ratio Conjugate = φ - 1 = 1/φ
PHI_CONJUGATE: float = PHI - 1  # 0.618033988749895

# √5 for golden calculations
SQRT_5: float = np.sqrt(5)  # 2.2360679774997896

# Golden Angle = 360° / φ² (phyllotaxis in nature)
GOLDEN_ANGLE_DEG: float = 360.0 / (PHI * PHI)  # 137.5077640500378546°
GOLDEN_ANGLE_RAD: float = np.radians(GOLDEN_ANGLE_DEG)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Standard sample rate (CD quality)
SAMPLE_RATE: int = 44100

# High precision sample rate (golden relationship)
SAMPLE_RATE_HI: int = 96000

# Sacred base frequency (Verdi tuning)
SACRED_FREQUENCY: float = 432.0

# Audio frequency ranges
AUDIO_FREQ_MIN: float = 50.0     # Lowest audible
AUDIO_FREQ_MAX: float = 4000.0   # Highest comfortable

# Brainwave frequency ranges
DELTA_RANGE = (0.5, 4.0)    # Deep sleep
THETA_RANGE = (4.0, 8.0)    # Meditation, creativity
ALPHA_RANGE = (8.0, 14.0)   # Relaxed focus
BETA_RANGE = (14.0, 30.0)   # Active thinking
GAMMA_RANGE = (30.0, 100.0) # High cognition


# ══════════════════════════════════════════════════════════════════════════════
# SACRED ANGLES (Degrees)
# ══════════════════════════════════════════════════════════════════════════════

SACRED_ANGLES = {
    # Golden / Fibonacci related
    "Golden Angle (360°/φ²)": GOLDEN_ANGLE_DEG,
    
    # Physics constants
    "Fine Structure (α⁻¹)": 137.035999084,
    
    # Biology / DNA
    "DNA Helix (per base)": 34.3,
    
    # Geometry
    "Pentagon Internal": 108.0,
    "Pyramid Giza Slope": 51.8392,
    "Hexagon Internal": 120.0,
    
    # Phase relationships
    "Cancellation (180°)": 180.0,
    "Quadrature (90°)": 90.0,
    
    # Molecular bond angles
    "Water H-O-H": 104.5,
    "Methane (tetrahedral)": 109.5,
    "Ammonia H-N-H": 107.3,
    "CO2 (linear)": 180.0,
    "Ozone O-O-O": 116.8,
}


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Speed of light (m/s)
C: float = 299792458.0

# Planck constant (J·s)
PLANCK: float = 6.62607e-34

# Rydberg constant (m⁻¹)
RYDBERG: float = 1.097373e7

# Fine structure constant inverse α⁻¹
FINE_STRUCTURE_INVERSE: float = 137.035999084


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate Fibonacci sequence of length n.
    Ratio of consecutive terms converges to φ.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]
    
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib


def fibonacci_harmonic_ratios(n: int = 6) -> List[int]:
    """
    Return first n Fibonacci numbers > 1 for harmonic ratios.
    [2, 3, 5, 8, 13, 21, ...]
    """
    fib = fibonacci_sequence(n + 2)
    return fib[2:]  # Skip first two 1s


# Pre-computed Fibonacci sequence (first 30 terms)
FIBONACCI: List[int] = fibonacci_sequence(30)

# First 20 prime numbers
PRIMES: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 
                     31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN WAVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def golden_spiral_interpolation(t: float) -> float:
    """
    Golden spiral easing function.
    Maps [0,1] → [0,1] following divine golden curve (NOT linear!)
    
    Used for smooth transitions that follow natural growth patterns.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    
    # Golden spiral easing
    theta = t * np.pi * PHI
    golden_ease = (1.0 - np.cos(theta * PHI_CONJUGATE)) / 2.0
    
    # Golden sigmoid for additional smoothing
    x = (t - 0.5) * 4.0
    golden_sigmoid = 1.0 / (1.0 + np.exp(-x * PHI))
    
    # Blend with golden weights
    result = golden_ease * PHI_CONJUGATE + golden_sigmoid * (1.0 - PHI_CONJUGATE)
    
    return np.clip(result, 0.0, 1.0)


def golden_transition(start: float, end: float, t: float) -> float:
    """
    Transition between two values using golden spiral interpolation.
    """
    golden_t = golden_spiral_interpolation(t)
    return start + (end - start) * golden_t


def golden_ease(t: float) -> float:
    """
    Simple golden easing curve for attack/decay envelopes.
    """
    theta = t * np.pi * PHI
    return (1.0 - np.cos(theta * PHI_CONJUGATE)) / 2.0


def apply_golden_envelope(signal: np.ndarray, 
                          attack_ratio: float = None,
                          release_ratio: float = None) -> np.ndarray:
    """
    Apply amplitude envelope using golden ratio proportions.
    
    Args:
        signal: Input audio signal
        attack_ratio: Attack time as fraction of length (default: φ⁻² × 0.2)
        release_ratio: Release time as fraction of length (default: φ⁻¹ × 0.3)
    
    Returns:
        Signal with golden envelope applied
    """
    length = len(signal)
    
    if attack_ratio is None:
        attack_ratio = PHI_CONJUGATE * PHI_CONJUGATE * 0.2
    if release_ratio is None:
        release_ratio = PHI_CONJUGATE * 0.3
    
    attack_len = int(length * attack_ratio)
    release_len = int(length * release_ratio)
    
    envelope = np.ones(length)
    
    # Attack phase with golden easing
    for i in range(min(attack_len, length)):
        t = i / attack_len
        envelope[i] = golden_ease(t)
    
    # Release phase with golden easing
    for i in range(min(release_len, length)):
        t = i / release_len
        envelope[length - 1 - i] = golden_ease(t)
    
    return signal * envelope


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN WAVE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def golden_wave_sample(phase: float, reversed: bool = True) -> float:
    """
    Generate single golden wave sample.
    
    A waveform based on φ where:
    - Rise time = φ⁻¹ (0.618...) of period
    - Fall time = φ⁻² (0.382...) of period
    
    This creates a naturally pleasing asymmetric waveform.
    """
    theta = phase % (2 * np.pi)
    if reversed:
        theta = 2 * np.pi - theta
    
    t = theta / (2 * np.pi)
    rise = PHI_CONJUGATE
    
    if t < rise:
        return np.sin(np.pi * t / rise / 2)
    else:
        return np.cos(np.pi * (t - rise) / (1 - rise) / 2)


def golden_wave(phase: np.ndarray, reversed: bool = True) -> np.ndarray:
    """
    Vectorized golden wave generator.
    
    Args:
        phase: Array of phase values (radians)
        reversed: If True, use reversed golden wave
    
    Returns:
        Array of waveform samples
    """
    theta = phase % (2 * np.pi)
    if reversed:
        theta = 2 * np.pi - theta
    
    t = theta / (2 * np.pi)
    rise = PHI_CONJUGATE
    
    wave = np.zeros_like(t)
    rising = t < rise
    wave = np.where(rising, np.sin(np.pi * t / rise / 2), 0)
    wave = np.where(~rising, np.cos(np.pi * (t - rise) / (1 - rise) / 2), wave)
    
    return wave


# ══════════════════════════════════════════════════════════════════════════════
# PHASE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_golden_phases(n: int) -> np.ndarray:
    """
    Generate n phase values using golden angle distribution.
    Each phase is offset by golden angle from the previous.
    This is the pattern found in sunflower seeds, pinecones, etc.
    """
    return np.array([(i * GOLDEN_ANGLE_RAD) % (2 * np.pi) for i in range(n)])


def generate_fibonacci_phases(n: int) -> np.ndarray:
    """
    Generate phases from Fibonacci sequence normalized to [0, 2π].
    """
    fib = fibonacci_sequence(max(n, 2))[:n]
    fib_max = max(fib)
    return np.array([2 * np.pi * f / fib_max for f in fib])


def generate_coherent_phases(n: int) -> np.ndarray:
    """
    Generate coherent (all zero) phases.
    """
    return np.zeros(n)


def generate_incoherent_phases(n: int) -> np.ndarray:
    """
    Generate random (quantum-realistic) phases.
    """
    return np.random.uniform(0, 2 * np.pi, n)


# ══════════════════════════════════════════════════════════════════════════════
# HARMONIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def fibonacci_harmonics(fundamental: float, n: int = 5) -> List[float]:
    """
    Generate harmonics at Fibonacci ratios of fundamental.
    
    Args:
        fundamental: Base frequency in Hz
        n: Number of harmonics (default 5: 2f, 3f, 5f, 8f, 13f)
    
    Returns:
        List of harmonic frequencies
    """
    ratios = fibonacci_harmonic_ratios(n)
    return [fundamental * r for r in ratios]


def phi_amplitude_decay(n: int) -> np.ndarray:
    """
    Generate amplitude values that decay by φ⁻¹ for each harmonic.
    
    harmonic 1: 1.0
    harmonic 2: φ⁻¹ ≈ 0.618
    harmonic 3: φ⁻² ≈ 0.382
    ...
    """
    return np.array([PHI_CONJUGATE ** i for i in range(n)])


def inverse_k_amplitude(n: int) -> np.ndarray:
    """
    Classic 1/k amplitude rolloff for harmonics.
    """
    return np.array([1.0 / (i + 1) for i in range(n)])


def inverse_sqrt_amplitude(n: int) -> np.ndarray:
    """
    1/√k amplitude rolloff (gentler than 1/k).
    """
    return np.array([1.0 / np.sqrt(i + 1) for i in range(n)])


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / np.pi


def normalize_audio(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio signal to target peak level.
    """
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal * (target_peak / max_val)
    return signal


def stereo_pan(pan: float) -> tuple:
    """
    Calculate left/right gains from pan position.
    
    Args:
        pan: -1.0 (full left) to 1.0 (full right), 0.0 = center
    
    Returns:
        (left_gain, right_gain) using equal power panning
    """
    # Convert [-1, 1] to [0, 1]
    pan_normalized = (pan + 1.0) / 2.0
    # Equal power panning
    left_gain = np.cos(pan_normalized * np.pi / 2)
    right_gain = np.sin(pan_normalized * np.pi / 2)
    return left_gain, right_gain


# ══════════════════════════════════════════════════════════════════════════════
# MODULE INFO
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Golden ratio constants
    'PHI', 'PHI_CONJUGATE', 'SQRT_5', 'GOLDEN_ANGLE_DEG', 'GOLDEN_ANGLE_RAD',
    
    # Audio constants
    'SAMPLE_RATE', 'SAMPLE_RATE_HI', 'SACRED_FREQUENCY',
    'AUDIO_FREQ_MIN', 'AUDIO_FREQ_MAX',
    'DELTA_RANGE', 'THETA_RANGE', 'ALPHA_RANGE', 'BETA_RANGE', 'GAMMA_RANGE',
    
    # Sacred angles
    'SACRED_ANGLES',
    
    # Physical constants
    'C', 'PLANCK', 'RYDBERG', 'FINE_STRUCTURE_INVERSE',
    
    # Sequences
    'FIBONACCI', 'PRIMES',
    'fibonacci_sequence', 'fibonacci_harmonic_ratios',
    
    # Golden functions
    'golden_spiral_interpolation', 'golden_transition', 'golden_ease',
    'apply_golden_envelope', 'golden_wave_sample', 'golden_wave',
    
    # Phase generation
    'generate_golden_phases', 'generate_fibonacci_phases',
    'generate_coherent_phases', 'generate_incoherent_phases',
    
    # Harmonic generation
    'fibonacci_harmonics', 'phi_amplitude_decay',
    'inverse_k_amplitude', 'inverse_sqrt_amplitude',
    
    # Utilities
    'deg_to_rad', 'rad_to_deg', 'normalize_audio', 'stereo_pan',
]


if __name__ == "__main__":
    # Print module info
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GOLDEN CONSTANTS MODULE                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  φ (Golden Ratio)     = {phi:.15f}                      ║
║  φ⁻¹ (Conjugate)      = {phi_conj:.15f}                      ║
║  Golden Angle         = {angle:.10f}°                           ║
║  Sample Rate          = {sr} Hz                                      ║
║  Sacred Frequency     = {sacred} Hz                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """.format(
        phi=PHI,
        phi_conj=PHI_CONJUGATE,
        angle=GOLDEN_ANGLE_DEG,
        sr=SAMPLE_RATE,
        sacred=SACRED_FREQUENCY
    ))
    
    print("Fibonacci sequence (first 10):", FIBONACCI[:10])
    print("Fibonacci harmonic ratios:", fibonacci_harmonic_ratios(6))
    print("Golden phases (5):", [f"{p:.3f}" for p in generate_golden_phases(5)])
    print("φ amplitude decay (5):", [f"{a:.3f}" for a in phi_amplitude_decay(5)])
