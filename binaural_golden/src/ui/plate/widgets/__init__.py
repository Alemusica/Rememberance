"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE WIDGETS - Reusable UI Components                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .canvas import PlateCanvas
from .controls import ControlPanel
from .modes_list import ModesListPanel
from .evolution_panel import EvolutionPanel

__all__ = [
    'PlateCanvas',
    'ControlPanel',
    'ModesListPanel',
    'EvolutionPanel',
]
