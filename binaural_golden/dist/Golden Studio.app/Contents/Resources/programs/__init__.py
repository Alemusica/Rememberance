"""
Programs module - Sequence/step management and generators

Usage:
    from programs import orchestra_tuning, chakra_journey, binaural_sweep
    
    # Generate a program with one line
    program = orchestra_tuning(432.0, duration_min=4)
    program.save('orchestra_432.json')
"""
try:
    from .step import Step, FrequencyConfig, PositionConfig, FadeCurve, BODY_POSITIONS
    from .program import Program
    from .generators import (
        orchestra_tuning,
        harmonic_meditation,
        binaural_sweep,
        chakra_journey,
        quick_program,
        harmonic_series,
        octave_series,
        ORCHESTRA_SECTIONS,
        CHAKRA_FREQUENCIES,
    )
except ImportError:
    pass

__all__ = [
    # Core classes
    'Step', 'FrequencyConfig', 'PositionConfig', 'FadeCurve', 
    'BODY_POSITIONS', 'Program',
    # Generators
    'orchestra_tuning',
    'harmonic_meditation',
    'binaural_sweep',
    'chakra_journey',
    'quick_program',
    # Utilities
    'harmonic_series',
    'octave_series',
    'ORCHESTRA_SECTIONS',
    'CHAKRA_FREQUENCIES',
]
