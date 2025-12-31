"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCORERS - Modular Fitness Evaluation                      ║
║                                                                              ║
║   Extracted from fitness.py (1556 lines → modular components)                ║
║                                                                              ║
║   Each scorer is responsible for ONE aspect of fitness evaluation:           ║
║   • ZoneFlatnessScorer: Frequency response flatness per zone                 ║
║   • EarUniformityScorer: L/R balance for binaural audio                      ║
║   • SpineCouplingScorer: Vibroacoustic coupling at spine                     ║
║   • StructuralScorer: Deflection and peninsula analysis                      ║
║   • ManufacturabilityScorer: CNC/production feasibility                      ║
║   • ExciterScorer: Exciter placement modal coupling                          ║
║                                                                              ║
║   SOLID Principles applied:                                                  ║
║   • SRP: Each scorer has single responsibility                               ║
║   • OCP: New scorers can be added without modifying FitnessEvaluator        ║
║   • LSP: All scorers implement Scorer Protocol                               ║
║   • ISP: Minimal interface (score method + name)                             ║
║   • DIP: FitnessEvaluator depends on Protocol, not concrete classes         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .protocol import Scorer, ScorerResult, ScorerBase
from .flatness import ZoneFlatnessScorer
from .ear_uniformity import EarUniformityScorer
from .spine_coupling import SpineCouplingScorer
from .structural import StructuralScorer
from .manufacturability import ManufacturabilityScorer
from .exciter import ExciterScorer

__all__ = [
    # Protocol and Base
    'Scorer',
    'ScorerResult',
    'ScorerBase',
    # Implementations
    'ZoneFlatnessScorer',
    'EarUniformityScorer',
    'SpineCouplingScorer',
    'StructuralScorer',
    'ManufacturabilityScorer',
    'ExciterScorer',
]
