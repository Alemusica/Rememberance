"""
Core module - Audio engine, physics, and sacred geometry
"""
try:
    from .audio_engine import AudioEngine
    from .golden_math import golden_fade, golden_ease
except ImportError:
    pass

try:
    from .sacred_geometry import (
        PHI, WATER_GEOMETRY, ANTAHKARANA,
        WaterMoleculeGeometry, AntahkaranaAxis, HumanBodyGolden
    )
except ImportError:
    pass

try:
    from .exciter import EXCITERS, Exciter, find_exciters_for_frequency
except ImportError:
    pass

__all__ = [
    'AudioEngine', 'golden_fade', 'golden_ease',
    'PHI', 'WATER_GEOMETRY', 'ANTAHKARANA',
    'WaterMoleculeGeometry', 'AntahkaranaAxis', 'HumanBodyGolden',
    'EXCITERS', 'Exciter', 'find_exciters_for_frequency',
]
