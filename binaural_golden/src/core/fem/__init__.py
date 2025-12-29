"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FEM MODULE - Unified Interface                            ║
║                                                                              ║
║   Provides consistent interface to different FEM backends:                   ║
║   • analytical  - Fast closed-form solutions (rectangular only)              ║
║   • skfem       - scikit-fem for arbitrary shapes                           ║
║   • jax         - Differentiable FEM for optimization (optional)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .interface import (
    # Core interface
    FEMSolver,
    FEMMode,
    FEMResult,
    MeshData,
    ShapeType,
    
    # Factory function
    get_solver,
    
    # Utilities
    create_mesh,
    calculate_flexural_rigidity,
)

from .analytical import AnalyticalSolver
from .mesh import create_rectangle_mesh, create_ellipse_mesh, create_polygon_mesh

# Check available backends
try:
    from .skfem_solver import SkfemSolver
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False
    SkfemSolver = None

# JAX solver is optional and requires separate implementation
HAS_JAX = False
JaxSolver = None

__all__ = [
    # Interface
    'FEMSolver',
    'FEMMode', 
    'FEMResult',
    'MeshData',
    
    # Factory
    'get_solver',
    
    # Solvers
    'AnalyticalSolver',
    
    # Utilities
    'create_mesh',
    'create_rectangle_mesh',
    'create_ellipse_mesh',
    'create_polygon_mesh',
    'calculate_flexural_rigidity',
    
    # Backend availability
    'HAS_SKFEM',
    'HAS_JAX',
]
