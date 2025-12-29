"""
Core module - Audio engine and physics
"""
try:
    from .audio_engine import AudioEngine
    from .golden_math import golden_fade, golden_ease
except ImportError:
    pass

__all__ = ['AudioEngine', 'golden_fade', 'golden_ease']
