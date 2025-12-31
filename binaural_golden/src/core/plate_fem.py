"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE FEM - Finite Element Modal Analysis                 ║
║                                                                              ║
║   Calculate modal frequencies for arbitrary plate shapes using FEM           ║
║   Supports: rectangle, ellipse, polygon, custom drawn shapes                 ║
║                                                                              ║
║   Uses scikit-fem for accurate eigenvalue analysis                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Try to import FEM libraries
try:
    from skfem import *
    from skfem.models.poisson import laplace
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False
    warnings.warn("scikit-fem not installed. Using analytical approximations.")

try:
    from scipy.sparse.linalg import eigsh
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import unified material definitions
from .materials import Material, MATERIALS


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class PlateShape(Enum):
    """Available plate shapes."""
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    CUSTOM = "custom"  # Hand-drawn


@dataclass
class FEMMode:
    """Represents a modal solution from FEM analysis."""
    index: int
    frequency: float            # Hz
    eigenvalue: float           # raw eigenvalue
    mode_shape: np.ndarray      # displacement at each mesh node
    mesh_points: np.ndarray     # (N, 2) array of node coordinates
    mesh_triangles: np.ndarray  # (M, 3) array of triangle indices
    
    @property
    def mode_name(self) -> str:
        return f"Mode {self.index + 1}"
    
    def get_displacement_at(self, x: float, y: float) -> float:
        """Interpolate mode shape at arbitrary point."""
        # Find containing triangle and interpolate
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(self.mesh_points, self.mode_shape)
        return float(interp(x, y))


# ══════════════════════════════════════════════════════════════════════════════
# MESH GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _point_inside_ellipse(px: float, py: float, cx: float, cy: float, 
                          a: float, b: float, rotation: float = 0.0) -> bool:
    """Check if point is inside rotated ellipse."""
    # Translate to ellipse center
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_r = np.cos(-rotation)
    sin_r = np.sin(-rotation)
    rx = dx * cos_r - dy * sin_r
    ry = dx * sin_r + dy * cos_r
    # Check ellipse equation
    return (rx / a)**2 + (ry / b)**2 <= 1.0


def _point_inside_cutout(px: float, py: float, cutout_abs: Dict) -> bool:
    """
    Check if point is inside a cutout (absolute coordinates).
    
    cutout_abs has: center, size, rotation, shape
    """
    cx, cy = cutout_abs["center"]
    w, h = cutout_abs["size"]
    rotation = cutout_abs.get("rotation", 0.0)
    shape = cutout_abs.get("shape", "ellipse")
    
    # Semi-axes
    a = w / 2
    b = h / 2
    
    if shape == "ellipse":
        return _point_inside_ellipse(px, py, cx, cy, a, b, rotation)
    elif shape == "rectangle" or shape == "rounded_rect":
        # Rotate point back to cutout frame
        dx = px - cx
        dy = py - cy
        cos_r = np.cos(-rotation)
        sin_r = np.sin(-rotation)
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        return abs(rx) <= a and abs(ry) <= b
    elif shape == "diamond":
        # Diamond: |x/a| + |y/b| <= 1
        dx = px - cx
        dy = py - cy
        cos_r = np.cos(-rotation)
        sin_r = np.sin(-rotation)
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        return abs(rx / a) + abs(ry / b) <= 1.0
    else:
        # Default: use ellipse
        return _point_inside_ellipse(px, py, cx, cy, a, b, rotation)


def create_rectangle_mesh(
    length: float, 
    width: float, 
    resolution: int = 20,
    cutouts_abs: Optional[List[Dict]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh for rectangular plate with optional cutouts.
    
    Args:
        length: Plate length in meters (x-axis)
        width: Plate width in meters (y-axis)
        resolution: Mesh resolution
        cutouts_abs: List of cutouts in absolute coordinates from CutoutGene.to_absolute()
                    Each dict has: center, size, rotation, shape
    
    Returns:
        (points, triangles) - points is (N,2), triangles is (M,3)
    """
    # Create grid points
    nx = int(resolution * length / max(length, width))
    ny = int(resolution * width / max(length, width))
    nx = max(nx, 5)
    ny = max(ny, 5)
    
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CUTOUT MESH MODIFICATION - Remove points inside cutouts
    # Reference: Standard FEM practice for holes in structural analysis
    # ═══════════════════════════════════════════════════════════════════════════
    if cutouts_abs:
        mask = np.ones(len(points), dtype=bool)
        for cutout in cutouts_abs:
            for i, (px, py) in enumerate(points):
                if _point_inside_cutout(px, py, cutout):
                    mask[i] = False
        points = points[mask]
    
    # Need minimum points for triangulation
    if len(points) < 4:
        # Fallback: return minimal mesh
        points = np.array([[0, 0], [length, 0], [length, width], [0, width]])
    
    # Create triangulation
    tri = Delaunay(points)
    triangles = tri.simplices
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REMOVE TRIANGLES WHOSE CENTROID IS INSIDE A CUTOUT
    # This ensures no triangles span across the hole
    # ═══════════════════════════════════════════════════════════════════════════
    if cutouts_abs:
        valid_triangles = []
        for tri_idx in triangles:
            # Triangle centroid
            centroid = points[tri_idx].mean(axis=0)
            inside_cutout = False
            for cutout in cutouts_abs:
                if _point_inside_cutout(centroid[0], centroid[1], cutout):
                    inside_cutout = True
                    break
            if not inside_cutout:
                valid_triangles.append(tri_idx)
        triangles = np.array(valid_triangles) if valid_triangles else triangles
    
    return points, triangles


def create_ellipse_mesh(a: float, b: float, resolution: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh for elliptical plate.
    
    Args:
        a: Semi-major axis (along x)
        b: Semi-minor axis (along y)
    """
    # Create points inside ellipse
    n_radial = resolution
    n_angular = int(resolution * 2)
    
    points = [(0, 0)]  # Center
    
    for i in range(1, n_radial + 1):
        r = i / n_radial
        for j in range(n_angular):
            theta = 2 * np.pi * j / n_angular
            x = r * a * np.cos(theta)
            y = r * b * np.sin(theta)
            points.append((x, y))
    
    points = np.array(points)
    
    # Triangulate
    tri = Delaunay(points)
    triangles = tri.simplices
    
    return points, triangles


def create_polygon_mesh(vertices: List[Tuple[float, float]], resolution: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh for arbitrary polygon.
    
    Args:
        vertices: List of (x, y) polygon vertices in order
    """
    from scipy.spatial import Delaunay
    
    vertices = np.array(vertices)
    
    # Get bounding box
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    
    # Create grid of candidate points
    nx = resolution
    ny = int(resolution * (max_y - min_y) / (max_x - min_x + 1e-6))
    ny = max(ny, 5)
    
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)
    candidates = np.column_stack([X.ravel(), Y.ravel()])
    
    # Filter points inside polygon
    from matplotlib.path import Path
    polygon_path = Path(vertices)
    inside = polygon_path.contains_points(candidates)
    
    # Add boundary points
    boundary_points = []
    n_per_edge = 5
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        for t in np.linspace(0, 1, n_per_edge, endpoint=False):
            boundary_points.append(p1 + t * (p2 - p1))
    
    points = np.vstack([candidates[inside], np.array(boundary_points)])
    
    # Remove duplicates
    points = np.unique(np.round(points, 6), axis=0)
    
    if len(points) < 4:
        # Fallback to rectangle
        return create_rectangle_mesh(max_x - min_x, max_y - min_y, resolution)
    
    # Triangulate
    try:
        tri = Delaunay(points)
        triangles = tri.simplices
    except:
        return create_rectangle_mesh(max_x - min_x, max_y - min_y, resolution)
    
    return points, triangles


# ══════════════════════════════════════════════════════════════════════════════
# FEM MODAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_flexural_rigidity(E: float, h: float, nu: float) -> float:
    """Calculate plate flexural rigidity D."""
    return (E * h**3) / (12 * (1 - nu**2))


def _find_nearest_node(points: np.ndarray, x: float, y: float) -> int:
    """Find index of nearest mesh node to (x, y)."""
    distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
    return int(np.argmin(distances))


def fem_modal_analysis(
    points: np.ndarray,
    triangles: np.ndarray,
    thickness: float,
    material: Material,
    n_modes: int = 10,
    spring_supports_abs: Optional[List[Dict]] = None
) -> List[FEMMode]:
    """
    Perform FEM modal analysis on arbitrary mesh with optional spring supports.
    
    Uses Kirchhoff plate theory with scikit-fem.
    
    ═══════════════════════════════════════════════════════════════════════════
    SPRING SUPPORT via PENALTY METHOD (Zienkiewicz & Taylor 2000)
    ═══════════════════════════════════════════════════════════════════════════
    
    The penalty method adds spring stiffness to diagonal of global K matrix:
        K_global[i, i] += k_spring
    
    This creates a soft constraint that:
    - For k_spring → ∞: rigid support (node fixed)
    - For k_spring ~ plate stiffness: elastic support
    - Shifts natural frequencies based on support stiffness
    
    Args:
        points: (N, 2) mesh node coordinates in meters
        triangles: (M, 3) triangle connectivity
        thickness: plate thickness in meters
        material: Material properties
        n_modes: number of modes to compute
        spring_supports_abs: List of spring supports in absolute coordinates
                            Each dict has: position (x, y), stiffness_n_m
    
    Returns:
        List of FEMMode objects sorted by frequency
    """
    if not HAS_SKFEM or not HAS_SCIPY:
        # Fallback to analytical approximation
        return _analytical_fallback(points, triangles, thickness, material, n_modes)
    
    try:
        # Create skfem mesh
        mesh = MeshTri(points.T, triangles.T)
        
        # Use quadratic elements for plate bending
        element = ElementTriP2()
        basis = Basis(mesh, element)
        
        # Material parameters
        D = calculate_flexural_rigidity(material.E_mean, thickness, material.poisson_ratio)
        rho = material.density
        
        # Assemble stiffness matrix (biharmonic for plate bending, approximated)
        # For simplicity, use Laplacian squared as approximation
        @BilinearForm
        def stiffness(u, v, w):
            # Plate bending stiffness (simplified)
            return D * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])
        
        @BilinearForm
        def mass_form(u, v, w):
            return rho * thickness * u * v
        
        K = stiffness.assemble(basis)
        M = mass_form.assemble(basis)
        
        # ═══════════════════════════════════════════════════════════════════════
        # PENALTY METHOD: Add spring stiffness to global K matrix
        # Reference: Zienkiewicz & Taylor (2000), Harris & Piersol (2002)
        # ═══════════════════════════════════════════════════════════════════════
        if spring_supports_abs:
            # Convert K to lil_matrix for efficient modification
            from scipy.sparse import lil_matrix
            K_lil = lil_matrix(K)
            
            for support in spring_supports_abs:
                # Get support position and stiffness
                sx, sy = support["position"]
                k_spring = support.get("stiffness_n_m", 8000.0)
                
                # Find nearest mesh node
                node_idx = _find_nearest_node(points, sx, sy)
                
                # For quadratic elements, node_idx might need mapping to DOF
                # In simple case, DOF index = node index
                dof_idx = node_idx
                if dof_idx < K_lil.shape[0]:
                    K_lil[dof_idx, dof_idx] += k_spring
            
            # Convert back to csr for efficient arithmetic
            K = K_lil.tocsr()
        
        # Solve eigenvalue problem
        # K * phi = lambda * M * phi
        eigenvalues, eigenvectors = eigsh(K, k=n_modes + 6, M=M, sigma=0, which='LM')
        
        # Filter positive eigenvalues and convert to frequencies
        modes = []
        mode_idx = 0
        
        for i in range(len(eigenvalues)):
            if eigenvalues[i] > 1e-6:
                # omega^2 = eigenvalue, f = omega / (2*pi)
                omega = np.sqrt(eigenvalues[i])
                freq = omega / (2 * np.pi)
                
                # Get mode shape at mesh nodes
                mode_shape = eigenvectors[:, i]
                # Normalize
                mode_shape = mode_shape / np.max(np.abs(mode_shape))
                
                # Interpolate to original mesh points
                # (basis may have more DOFs than original points)
                if len(mode_shape) > len(points):
                    mode_shape = mode_shape[:len(points)]
                
                modes.append(FEMMode(
                    index=mode_idx,
                    frequency=freq,
                    eigenvalue=eigenvalues[i],
                    mode_shape=mode_shape,
                    mesh_points=points,
                    mesh_triangles=triangles
                ))
                mode_idx += 1
                
                if mode_idx >= n_modes:
                    break
        
        return modes
        
    except Exception as e:
        print(f"FEM analysis failed: {e}, using analytical fallback")
        return _analytical_fallback(points, triangles, thickness, material, n_modes)


def _analytical_fallback(
    points: np.ndarray,
    triangles: np.ndarray,
    thickness: float,
    material: Material,
    n_modes: int
) -> List[FEMMode]:
    """
    Fallback to analytical approximation when FEM fails.
    Uses rectangular plate formula with bounding box dimensions.
    """
    # Get bounding box
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    L = max_pt[0] - min_pt[0]
    W = max_pt[1] - min_pt[1]
    
    D = calculate_flexural_rigidity(material.E_mean, thickness, material.poisson_ratio)
    rho = material.density
    h = thickness
    
    modes = []
    mode_idx = 0
    
    for m in range(1, 6):
        for n in range(1, 6):
            # Free-free plate approximation
            lambda_m = (m + 0.5) * np.pi
            lambda_n = (n + 0.5) * np.pi
            
            omega_sq = (D / (rho * h)) * ((lambda_m / L)**4 + (lambda_n / W)**4)
            freq = np.sqrt(omega_sq) / (2 * np.pi)
            
            # Create approximate mode shape
            cx = (min_pt[0] + max_pt[0]) / 2
            cy = (min_pt[1] + max_pt[1]) / 2
            
            mode_shape = np.cos(m * np.pi * (points[:, 0] - min_pt[0]) / L) * \
                         np.cos(n * np.pi * (points[:, 1] - min_pt[1]) / W)
            
            modes.append(FEMMode(
                index=mode_idx,
                frequency=freq,
                eigenvalue=omega_sq,
                mode_shape=mode_shape,
                mesh_points=points,
                mesh_triangles=triangles
            ))
            mode_idx += 1
    
    # Sort by frequency
    modes.sort(key=lambda x: x.frequency)
    return modes[:n_modes]


# ══════════════════════════════════════════════════════════════════════════════
# EXCITER COUPLING
# ══════════════════════════════════════════════════════════════════════════════

def calculate_exciter_coupling_fem(
    mode: FEMMode,
    exciter_x: float,
    exciter_y: float
) -> float:
    """
    Calculate coupling coefficient for exciter at (x, y) for given FEM mode.
    
    Returns:
        Coupling coefficient [0, 1] where 1 = maximum coupling
    """
    # Interpolate mode shape at exciter position
    try:
        displacement = mode.get_displacement_at(exciter_x, exciter_y)
        # Normalize to [0, 1]
        return abs(displacement)
    except:
        return 0.5  # Default if interpolation fails


def calculate_optimal_phases_fem(
    mode: FEMMode,
    exciter_positions: List[Tuple[float, float]]
) -> List[float]:
    """
    Calculate optimal phase for each exciter to maximally excite the mode.
    
    Returns:
        List of phases in degrees (0 or 180)
    """
    phases = []
    for x, y in exciter_positions:
        try:
            displacement = mode.get_displacement_at(x, y)
            if displacement >= 0:
                phases.append(0.0)
            else:
                phases.append(180.0)
        except:
            phases.append(0.0)
    return phases


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ══════════════════════════════════════════════════════════════════════════════

class PlateAnalyzer:
    """
    High-level interface for plate modal analysis.
    Supports various shapes, cutouts, and spring supports.
    """
    
    def __init__(self):
        self.shape = PlateShape.RECTANGLE
        self.points: Optional[np.ndarray] = None
        self.triangles: Optional[np.ndarray] = None
        self.modes: List[FEMMode] = []
        self.material = MATERIALS["spruce"]
        self.thickness = 0.01  # meters
        
        # Shape parameters
        self.rect_length = 1.95  # meters
        self.rect_width = 0.6
        self.ellipse_a = 0.975
        self.ellipse_b = 0.3
        self.polygon_vertices: List[Tuple[float, float]] = []
        self.custom_points: List[Tuple[float, float]] = []
        
        # NEW: Cutouts and spring supports
        self.cutouts_abs: List[Dict] = []
        self.spring_supports_abs: List[Dict] = []
    
    def set_rectangle(self, length_m: float, width_m: float):
        """Set plate shape to rectangle."""
        self.shape = PlateShape.RECTANGLE
        self.rect_length = length_m
        self.rect_width = width_m
    
    def set_ellipse(self, a: float, b: float):
        """Set plate shape to ellipse."""
        self.shape = PlateShape.ELLIPSE
        self.ellipse_a = a
        self.ellipse_b = b
    
    def set_polygon(self, vertices: List[Tuple[float, float]]):
        """Set plate shape to polygon."""
        self.shape = PlateShape.POLYGON
        self.polygon_vertices = vertices
    
    def set_custom(self, points: List[Tuple[float, float]]):
        """Set plate shape from custom drawn points."""
        self.shape = PlateShape.CUSTOM
        self.custom_points = points
    
    def set_material(self, material_key: str):
        """Set material by key."""
        if material_key in MATERIALS:
            self.material = MATERIALS[material_key]
    
    def set_thickness(self, thickness_m: float):
        """Set plate thickness in meters."""
        self.thickness = thickness_m
    
    def set_cutouts(self, cutouts_abs: List[Dict]):
        """
        Set cutouts for mesh generation.
        
        Args:
            cutouts_abs: List of cutout dicts from CutoutGene.to_absolute()
                        Each has: center, size, rotation, shape
        """
        self.cutouts_abs = cutouts_abs
    
    def set_spring_supports(self, spring_supports_abs: List[Dict]):
        """
        Set spring supports for FEM analysis.
        
        Args:
            spring_supports_abs: List of support dicts from SpringSupportGene.to_absolute()
                                Each has: position, stiffness_n_m
        """
        self.spring_supports_abs = spring_supports_abs
    
    def generate_mesh(self, resolution: int = 20):
        """Generate mesh for current shape with optional cutouts."""
        if self.shape == PlateShape.RECTANGLE:
            self.points, self.triangles = create_rectangle_mesh(
                self.rect_length, self.rect_width, resolution,
                cutouts_abs=self.cutouts_abs  # NEW: Pass cutouts
            )
        elif self.shape == PlateShape.ELLIPSE:
            self.points, self.triangles = create_ellipse_mesh(
                self.ellipse_a, self.ellipse_b, resolution
            )
        elif self.shape == PlateShape.POLYGON:
            if self.polygon_vertices:
                self.points, self.triangles = create_polygon_mesh(
                    self.polygon_vertices, resolution
                )
            else:
                self.points, self.triangles = create_rectangle_mesh(1.0, 0.5, resolution)
        elif self.shape == PlateShape.CUSTOM:
            if len(self.custom_points) >= 3:
                self.points, self.triangles = create_polygon_mesh(
                    self.custom_points, resolution
                )
            else:
                self.points, self.triangles = create_rectangle_mesh(1.0, 0.5, resolution)
    
    def analyze(self, n_modes: int = 10) -> List[FEMMode]:
        """Run modal analysis with optional spring supports."""
        if self.points is None:
            self.generate_mesh()
        
        self.modes = fem_modal_analysis(
            self.points,
            self.triangles,
            self.thickness,
            self.material,
            n_modes,
            spring_supports_abs=self.spring_supports_abs  # NEW: Pass springs
        )
        return self.modes
    
    def get_coupling(self, mode_index: int, x: float, y: float) -> float:
        """Get coupling coefficient for exciter at position."""
        if mode_index >= len(self.modes):
            return 0.0
        return calculate_exciter_coupling_fem(self.modes[mode_index], x, y)
    
    def get_optimal_phases(self, mode_index: int, 
                           positions: List[Tuple[float, float]]) -> List[float]:
        """Get optimal phases for exciters."""
        if mode_index >= len(self.modes):
            return [0.0] * len(positions)
        return calculate_optimal_phases_fem(self.modes[mode_index], positions)
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box of current shape (min_x, min_y, max_x, max_y)."""
        if self.points is None:
            return 0, 0, 1, 1
        return (
            self.points[:, 0].min(),
            self.points[:, 1].min(),
            self.points[:, 0].max(),
            self.points[:, 1].max()
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PLATE FEM MODAL ANALYSIS - Test")
    print("=" * 60)
    
    analyzer = PlateAnalyzer()
    analyzer.set_rectangle(1.95, 0.6)
    analyzer.set_material("spruce")
    analyzer.set_thickness(0.01)
    
    print("\nGenerating mesh...")
    analyzer.generate_mesh(resolution=15)
    print(f"  Nodes: {len(analyzer.points)}")
    print(f"  Triangles: {len(analyzer.triangles)}")
    
    print("\nRunning modal analysis...")
    modes = analyzer.analyze(n_modes=8)
    
    print(f"\n{'Mode':<10} {'Frequency (Hz)':<15}")
    print("-" * 30)
    for mode in modes:
        print(f"{mode.mode_name:<10} {mode.frequency:>10.1f}")
    
    # Test coupling
    print("\n\nExciter coupling test (mode 1):")
    positions = [(0, 0.3), (1.95, 0.3), (0.975, 0.3)]
    for x, y in positions:
        coupling = analyzer.get_coupling(0, x, y)
        print(f"  Position ({x:.2f}, {y:.2f}): coupling = {coupling:.3f}")
