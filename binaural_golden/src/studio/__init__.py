"""
Studio module - Core application components for Golden Sound Studio.

This module provides re-exports for backwards compatibility.
The actual AudioEngine is in core/audio_engine.py.

For Pi5 deployment, this module structure allows lazy loading and
modular initialization of heavy components.
"""

# Re-export AudioEngine from core for backwards compatibility
from core.audio_engine import AudioEngine

__all__ = ['AudioEngine']
