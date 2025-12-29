"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FEM INTERFACE - Abstract Base Classes                     ║
║                                                                              ║
║   Defines the common interface for all FEM solvers.                          ║
║   Implement FEMSolver ABC to add new backends.                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

# Import from single source
from ..materials import Material, MATERIALS, DEFAULT_MATERIAL


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class ShapeType(Enum):
    """Supported plate shapes."""
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    CUSTOM = "custom"


@dataclass
class MeshData:
    """Container for mesh geometry."""
    points: np.ndarray          # (N, 2) node coordinates in meters
    triangles: np.ndarray       # (M, 3) triangle connectivity
    shape_type: ShapeType
    bounding_box: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    
    @property
    def n_nodes(self) -> int:
        return len(self.points)
    
    @property
    def n_elements(self) -> int:
        return len(self.triangles)
    
    @property
    def length(self) -> float:
        """Bounding box length (x-direction)."""
        return self.bounding_box[2] - self.bounding_box[0]
    
    @property
    def width(self) -> float:
        """Bounding box width (y-direction)."""
        return self.bounding_box[3] - self.bounding_box[1]


@dataclass
class FEMMode:
    """Represents a single vibrational mode from FEM analysis."""
    index: int
    frequency: float            # Hz
    eigenvalue: float           # raw eigenvalue (omega^2 * rho * h)
    mode_shape: np.ndarray      # displacement at each mesh node
    mesh: MeshData              # reference to mesh
    
    # Mode identification (estimated from shape analysis)
    m: int = 0                  # mode number in x-direction
    n: int = 0                  # mode number in y-direction
    
    @property
    def mode_name(self) -> str:
        if self.m > 0 and self.n > 0:
            return f"({self.m},{self.n})"
        return f"Mode {self.index + 1}"
    
    @property
    def angular_frequency(self) -> float:
        """Angular frequency omega = 2*pi*f."""
        return 2 * np.pi * self.frequency
    
    def get_displacement_at(self, x: float, y: float) -> float:
        """
        Interpolate mode shape at arbitrary point.
        
        Args:
            x, y: Position in meters (within plate bounds)
            
        Returns:
            Normalized displacement [-1, 1]
        """
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(self.mesh.points, self.mode_shape)
        result = interp(x, y)
        return float(result) if not np.isnan(result) else 0.0
    
    def get_displacement_grid(
        self, 
        nx: int = 50, 
        ny: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mode shape on regular grid for visualization.
        
        Returns:
            (X, Y, Z) meshgrid arrays
        """
        from scipy.interpolate import griddata
        
        x_min, y_min, x_max, y_max = self.mesh.bounding_box
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        Z = griddata(
            self.mesh.points, 
            self.mode_shape, 
            (X, Y), 
            method='linear',
            fill_value=0.0
        )
        
        return X, Y, Z


@dataclass
class FEMResult:
    """Complete result from modal analysis."""
    modes: List[FEMMode]
    mesh: MeshData
    material: Material
    thickness: float            # meters
    solve_time: float           # seconds
    solver_name: str
    converged: bool = True
    warnings: List[str] = field(default_factory=list)
    
    @property
    def frequencies(self) -> np.ndarray:
        """Array of all mode frequencies."""
        return np.array([m.frequency for m in self.modes])
    
    def get_mode_at_frequency(self, target_freq: float) -> Optional[FEMMode]:
        """Find mode closest to target frequency."""
        if not self.modes:
            return None
        diffs = np.abs(self.frequencies - target_freq)
        idx = np.argmin(diffs)
        return self.modes[idx]
    
    def get_modes_in_range(self, f_min: float, f_max: float) -> List[FEMMode]:
        """Get all modes within frequency range."""
        return [m for m in self.modes if f_min <= m.frequency <= f_max]


# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FEMSolver(ABC):
    """
    Abstract base class for FEM solvers.
    
    Implement this interface to add new backends (e.g., FEniCS, deal.II).
    """
    
    name: str = "AbstractSolver"
    
    @abstractmethod
    def solve(
        self,
        mesh: MeshData,
        thickness: float,
        material: Material,
        n_modes: int = 10,
        **kwargs
    ) -> FEMResult:
        """
        Solve the eigenvalue problem for plate vibration.
        
        Args:
            mesh: Triangular mesh data
            thickness: Plate thickness in meters
            material: Material properties
            n_modes: Number of modes to compute
            **kwargs: Solver-specific options
            
        Returns:
            FEMResult with computed modes
        """
        pass
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this solver's dependencies are installed."""
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_flexural_rigidity(E: float, h: float, nu: float) -> float:
    """
    Calculate plate flexural rigidity D.
    
    D = E * h³ / (12 * (1 - ν²))
    
    Args:
        E: Young's modulus (Pa)
        h: Thickness (m)
        nu: Poisson's ratio
        
    Returns:
        Flexural rigidity D (N·m)
    """
    return (E * h**3) / (12 * (1 - nu**2))


def create_mesh(
    shape_type: ShapeType,
    length: float,
    width: float,
    resolution: int = 20,
    vertices: Optional[List[Tuple[float, float]]] = None
) -> MeshData:
    """
    Factory function to create mesh for given shape.
    
    Args:
        shape_type: Type of shape
        length: Length in meters (x-direction)
        width: Width in meters (y-direction)
        resolution: Mesh resolution (approximate elements per dimension)
        vertices: For POLYGON type, list of (x, y) vertices
        
    Returns:
        MeshData object
    """
    from .mesh import create_rectangle_mesh, create_ellipse_mesh, create_polygon_mesh
    
    if shape_type == ShapeType.RECTANGLE:
        points, triangles = create_rectangle_mesh(length, width, resolution)
        bbox = (0, 0, length, width)
        
    elif shape_type == ShapeType.ELLIPSE:
        points, triangles = create_ellipse_mesh(length/2, width/2, resolution)
        bbox = (-length/2, -width/2, length/2, width/2)
        
    elif shape_type == ShapeType.POLYGON:
        if vertices is None:
            raise ValueError("vertices required for POLYGON shape")
        points, triangles = create_polygon_mesh(vertices, resolution)
        verts = np.array(vertices)
        bbox = (verts[:, 0].min(), verts[:, 1].min(), 
                verts[:, 0].max(), verts[:, 1].max())
        
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")
    
    return MeshData(
        points=points,
        triangles=triangles,
        shape_type=shape_type,
        bounding_box=bbox
    )


def get_solver(backend: str = "auto") -> FEMSolver:
    """
    Get FEM solver instance.
    
    Args:
        backend: One of "auto", "analytical", "skfem", "jax"
        
    Returns:
        FEMSolver instance
    """
    from .analytical import AnalyticalSolver
    
    if backend == "analytical":
        return AnalyticalSolver()
    
    if backend == "skfem":
        try:
            from .skfem_solver import SkfemSolver
            return SkfemSolver()
        except ImportError:
            raise ImportError("scikit-fem not installed. Install with: pip install scikit-fem")
    
    if backend == "jax":
        try:
            from .jax_solver import JaxSolver
            return JaxSolver()
        except ImportError:
            raise ImportError("JAX not installed. Install with: pip install jax jaxlib")
    
    if backend == "auto":
        # Try in order of preference
        try:
            from .skfem_solver import SkfemSolver
            return SkfemSolver()
        except ImportError:
            pass
        
        # Fallback to analytical
        return AnalyticalSolver()
    
    raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'analytical', 'skfem', or 'jax'")
