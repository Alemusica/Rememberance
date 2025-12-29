"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE MODULE - Unified Plate Designer                     ║
║                                                                              ║
║   Single entry point for all plate-related functionality.                    ║
║   Combines modal analysis (Plate Lab) + optimization (Plate Designer).       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .plate_tab import PlateTab
from .viewmodel import PlateViewModel, PlateState, PlateMode

__all__ = [
    'PlateTab',
    'PlateViewModel',
    'PlateState',
    'PlateMode',
]
