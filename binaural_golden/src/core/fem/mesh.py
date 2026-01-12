"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FEM MESH - Mesh Generation Utilities                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import List, Tuple

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def create_rectangle_mesh(
    length: float, 
    width: float, 
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh for rectangular plate.
    
    Args:
        length: Plate length in meters (x-direction)
        width: Plate width in meters (y-direction)
        resolution: Base resolution (adjusted for aspect ratio)
        
    Returns:
        (points, triangles) - points is (N, 2), triangles is (M, 3)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for mesh generation")
    
    # Adjust resolution for aspect ratio
    aspect = length / width if width > 0 else 1.0
    nx = max(5, int(resolution * min(aspect, 2.0)))
    ny = max(5, int(resolution / max(aspect, 0.5)))
    
    # Create grid points
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create triangulation
    tri = Delaunay(points)
    triangles = tri.simplices
    
    return points, triangles


def create_ellipse_mesh(
    a: float, 
    b: float, 
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh for elliptical plate.
    
    Args:
        a: Semi-major axis in meters (x-direction)
        b: Semi-minor axis in meters (y-direction)
        resolution: Radial resolution
        
    Returns:
        (points, triangles) - centered at origin
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for mesh generation")
    
    n_radial = max(5, resolution)
    n_angular = max(12, int(resolution * 2))
    
    points = [(0, 0)]  # Center point
    
    # Generate points in concentric rings
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


def create_polygon_mesh(
    vertices: List[Tuple[float, float]], 
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh for arbitrary convex polygon.
    
    Args:
        vertices: List of (x, y) vertices in order (clockwise or counter-clockwise)
        resolution: Approximate points per dimension
        
    Returns:
        (points, triangles)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for mesh generation")
    
    try:
        from matplotlib.path import Path
    except ImportError:
        # Fallback: use convex hull
        vertices = np.array(vertices)
        bbox = (vertices[:, 0].min(), vertices[:, 1].min(),
                vertices[:, 0].max(), vertices[:, 1].max())
        return create_rectangle_mesh(
            bbox[2] - bbox[0], bbox[3] - bbox[1], resolution
        )
    
    vertices = np.array(vertices)
    
    # Bounding box
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    
    # Create candidate grid
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    nx = max(5, resolution)
    ny = max(5, int(resolution * span_y / (span_x + 1e-6)))
    
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)
    candidates = np.column_stack([X.ravel(), Y.ravel()])
    
    # Filter points inside polygon
    polygon_path = Path(vertices)
    inside = polygon_path.contains_points(candidates)
    
    # Add boundary points
    boundary_points = []
    n_per_edge = max(3, resolution // len(vertices))
    
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        for t in np.linspace(0, 1, n_per_edge, endpoint=False):
            boundary_points.append(p1 + t * (p2 - p1))
    
    # Combine interior and boundary
    interior_points = candidates[inside]
    boundary_arr = np.array(boundary_points)
    
    if len(interior_points) == 0:
        points = boundary_arr
    else:
        points = np.vstack([interior_points, boundary_arr])
    
    # Remove near-duplicates
    points = np.unique(np.round(points, 6), axis=0)
    
    if len(points) < 4:
        # Fallback to bounding rectangle
        return create_rectangle_mesh(span_x, span_y, resolution)
    
    # Triangulate
    try:
        tri = Delaunay(points)
        triangles = tri.simplices
    except Exception:
        return create_rectangle_mesh(span_x, span_y, resolution)
    
    return points, triangles


def create_golden_rectangle_mesh(
    length: float,
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mesh for golden ratio rectangle.
    
    Width = Length / PHI
    """
    PHI = 1.618033988749895
    width = length / PHI
    return create_rectangle_mesh(length, width, resolution)


def refine_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    refinement_level: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine mesh by adding midpoints (simple subdivision).
    
    Each triangle becomes 4 triangles.
    """
    if refinement_level <= 0:
        return points, triangles
    
    for _ in range(refinement_level):
        new_points = list(points)
        new_triangles = []
        edge_midpoints = {}
        
        def get_midpoint(i, j):
            """Get or create midpoint between vertices i and j."""
            key = (min(i, j), max(i, j))
            if key not in edge_midpoints:
                mid = (points[i] + points[j]) / 2
                edge_midpoints[key] = len(new_points)
                new_points.append(mid)
            return edge_midpoints[key]
        
        for tri in triangles:
            i, j, k = tri
            
            # Get midpoints
            ij = get_midpoint(i, j)
            jk = get_midpoint(j, k)
            ki = get_midpoint(k, i)
            
            # Create 4 new triangles
            new_triangles.append([i, ij, ki])
            new_triangles.append([j, jk, ij])
            new_triangles.append([k, ki, jk])
            new_triangles.append([ij, jk, ki])
        
        points = np.array(new_points)
        triangles = np.array(new_triangles)
    
    return points, triangles
