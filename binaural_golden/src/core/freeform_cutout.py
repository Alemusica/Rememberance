"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          FREEFORM CUTOUT GENERATOR - Topology-Style Shape Optimization       ║
║                                                                              ║
║   Generate arbitrary cutout shapes without predefined constraints.           ║
║   Inspired by topology optimization (SIMP method) and CNC freedom.           ║
║                                                                              ║
║   KEY CONCEPTS:                                                              ║
║   1. DENSITY FIELD: ρ(x,y) ∈ [0,1] where 0=void, 1=material                 ║
║   2. BEZIER/NURBS: Smooth parametric curves for CNC-friendly shapes          ║
║   3. EVOLVING BOUNDARIES: Contour evolves through generations                ║
║   4. PHYSICS-GUIDED INITIALIZATION: Start near antinodes/ABH zones          ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Bendsøe & Kikuchi 1988: Topology optimization foundations               ║
║   • SIMP method: Solid Isotropic Material with Penalization                 ║
║   • Czinger/Divergent: Generative design for 3D printed structures          ║
║   • Level-set methods for shape optimization                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.interpolate import splprep, splev, interp1d
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DENSITY FIELD - Topology Optimization Style
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DensityField:
    """
    Continuous density field for topology-style cutout generation.
    
    ρ(x,y) ∈ [0,1] where:
    - 0 = void (cutout)
    - 1 = material (solid)
    - intermediate = transition (smoothed for manufacturability)
    
    The field is evolved through generations, with physics guiding
    where material should be removed.
    
    SIMP-INSPIRED:
    - Intermediate densities are penalized (pushed to 0 or 1)
    - Filtering ensures smooth, manufacturable shapes
    """
    resolution: Tuple[int, int] = (100, 60)  # (nx, ny) grid
    field: np.ndarray = None
    
    # Manufacturing constraints
    min_feature_size: float = 0.03   # Minimum hole/island size (normalized)
    smoothing_sigma: float = 2.0     # Gaussian smoothing radius
    
    # Evolution parameters
    penalization_power: float = 3.0  # SIMP exponent (higher = sharper boundaries)
    threshold: float = 0.5           # Density threshold for void/solid
    
    def __post_init__(self):
        """Initialize with full material (no cutouts)."""
        if self.field is None:
            self.field = np.ones(self.resolution)
    
    @classmethod
    def from_physics_guidance(
        cls,
        mode_shapes: np.ndarray,
        target_frequencies: List[float],
        current_frequencies: List[float],
        resolution: Tuple[int, int] = (100, 60),
        initial_void_ratio: float = 0.1,
    ) -> 'DensityField':
        """
        Initialize density field based on modal physics.
        
        Places initial voids near antinodes of modes that need
        frequency adjustment (as suggested by physics).
        
        Args:
            mode_shapes: Array of mode shapes (n_modes, ny, nx)
            target_frequencies: Desired frequencies [Hz]
            current_frequencies: Current frequencies [Hz]
            resolution: Grid resolution
            initial_void_ratio: Fraction of plate to start as void
        
        Returns:
            DensityField initialized with physics-guided voids
        """
        density = cls(resolution=resolution)
        
        if mode_shapes is None or len(mode_shapes) == 0:
            return density
        
        nx, ny = resolution
        
        # Combine mode shapes weighted by frequency deviation
        combined_sensitivity = np.zeros((ny, nx))
        
        n_modes = min(len(mode_shapes), len(target_frequencies), len(current_frequencies))
        
        for i in range(n_modes):
            if target_frequencies[i] <= 0:
                continue
            
            deviation = abs(current_frequencies[i] - target_frequencies[i]) / target_frequencies[i]
            
            # Resize mode shape to grid resolution
            mode = mode_shapes[i]
            if mode.shape != (ny, nx):
                from scipy.ndimage import zoom
                scale_y = ny / mode.shape[0]
                scale_x = nx / mode.shape[1]
                mode = zoom(mode, (scale_y, scale_x), order=1)
            
            # Modes with high deviation contribute more
            # High amplitude regions = good cutout locations
            combined_sensitivity += deviation * np.abs(mode)
        
        # Normalize
        if combined_sensitivity.max() > 0:
            combined_sensitivity /= combined_sensitivity.max()
        
        # Create initial void pattern based on sensitivity
        # Higher sensitivity = lower density (more likely to be cut)
        initial_density = 1.0 - initial_void_ratio * combined_sensitivity
        
        # Apply smoothing
        initial_density = gaussian_filter(initial_density, sigma=2.0)
        initial_density = np.clip(initial_density, 0.0, 1.0)
        
        density.field = initial_density
        
        return density
    
    def evolve(
        self,
        sensitivity_field: np.ndarray,
        learning_rate: float = 0.1,
        preserve_zones: List[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Evolve the density field based on sensitivity analysis.
        
        This is the core topology optimization update step:
        - High sensitivity regions become voids (cutouts)
        - Low sensitivity regions remain material
        
        Args:
            sensitivity_field: Physics-based sensitivity (higher = better for cutout)
            learning_rate: How much to update per step
            preserve_zones: List of (x_min, x_max, y_min, y_max) zones to keep solid
        """
        # Normalize sensitivity
        sens = np.copy(sensitivity_field)
        if sens.max() > 0:
            sens = sens / sens.max()
        
        # Resize if needed
        if sens.shape != self.field.shape:
            from scipy.ndimage import zoom
            scale = (self.field.shape[0] / sens.shape[0], 
                    self.field.shape[1] / sens.shape[1])
            sens = zoom(sens, scale, order=1)
        
        # Update: high sensitivity → lower density
        delta = -learning_rate * sens
        self.field = np.clip(self.field + delta, 0.0, 1.0)
        
        # Preserve zones (spine, edges, etc.)
        if preserve_zones:
            ny, nx = self.field.shape
            for x_min, x_max, y_min, y_max in preserve_zones:
                ix_min, ix_max = int(x_min * nx), int(x_max * nx)
                iy_min, iy_max = int(y_min * ny), int(y_max * ny)
                self.field[iy_min:iy_max, ix_min:ix_max] = 1.0  # Keep solid
        
        # Apply SIMP penalization (push intermediate values to 0 or 1)
        self.field = np.power(self.field, self.penalization_power)
        
        # Apply smoothing for manufacturability
        self.field = gaussian_filter(self.field, sigma=self.smoothing_sigma)
        self.field = np.clip(self.field, 0.0, 1.0)
        
        # Remove too-small features
        self._enforce_minimum_feature_size()
    
    def _enforce_minimum_feature_size(self):
        """Remove features smaller than minimum size."""
        ny, nx = self.field.shape
        min_pixels = int(self.min_feature_size * min(nx, ny))
        
        if min_pixels < 2:
            return
        
        # Binary version
        binary = self.field < self.threshold
        
        # Morphological opening removes small holes
        struct = np.ones((min_pixels, min_pixels))
        cleaned = binary_erosion(binary, structure=struct)
        cleaned = binary_dilation(cleaned, structure=struct)
        
        # Apply back to density (soft, not hard)
        smooth_binary = gaussian_filter(cleaned.astype(float), sigma=1.0)
        
        # Blend: keep original but push small features toward solid
        mask = (binary != cleaned)
        if mask.any():
            self.field[mask] = 0.5 * self.field[mask] + 0.5 * (1.0 - smooth_binary[mask])
    
    def extract_contours(self, level: float = None) -> List[np.ndarray]:
        """
        Extract boundary contours at given density level.
        
        Returns list of contours, each as (N, 2) array of (x, y) points.
        Uses marching squares algorithm (or simple threshold if skimage unavailable).
        """
        if level is None:
            level = self.threshold
        
        try:
            from skimage.measure import find_contours
            contours = find_contours(self.field, level)
        except ImportError:
            # Fallback: simple threshold-based contour extraction
            contours = self._simple_contours(level)
        
        # Normalize to [0, 1]
        ny, nx = self.field.shape
        normalized = []
        for contour in contours:
            if len(contour) < 4:
                continue
            # find_contours returns (row, col) = (y, x)
            norm_contour = np.column_stack([
                contour[:, 1] / nx,  # x
                contour[:, 0] / ny,  # y
            ])
            normalized.append(norm_contour)
        
        return normalized
    
    def _simple_contours(self, level: float) -> List[np.ndarray]:
        """
        Simple contour extraction without skimage.
        
        Uses edge detection on binary threshold.
        """
        binary = self.field < level
        
        # Find edges using simple gradient
        edges_x = np.diff(binary.astype(float), axis=1)
        edges_y = np.diff(binary.astype(float), axis=0)
        
        # Combine edges
        edge_map = np.zeros_like(self.field, dtype=bool)
        edge_map[:-1, :-1] = (np.abs(edges_x[:-1, :]) > 0) | (np.abs(edges_y[:, :-1]) > 0)
        
        # Find connected edge components (simple flood fill)
        contours = []
        visited = np.zeros_like(edge_map, dtype=bool)
        
        for i in range(edge_map.shape[0]):
            for j in range(edge_map.shape[1]):
                if edge_map[i, j] and not visited[i, j]:
                    contour = self._trace_contour(edge_map, visited, i, j)
                    if len(contour) > 10:  # Minimum contour length
                        contours.append(np.array(contour))
        
        return contours
    
    def _trace_contour(self, edge_map: np.ndarray, visited: np.ndarray, 
                       start_i: int, start_j: int) -> List[Tuple[int, int]]:
        """Trace a contour starting from given point."""
        contour = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if visited[i, j]:
                continue
            
            visited[i, j] = True
            contour.append((i, j))
            
            # Check 8-connected neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if (0 <= ni < edge_map.shape[0] and 
                        0 <= nj < edge_map.shape[1] and
                        edge_map[ni, nj] and not visited[ni, nj]):
                        stack.append((ni, nj))
        
        return contour
    
    def to_cutout_genes(self, min_area: float = 0.001) -> List['FreeformCutout']:
        """
        Convert density field voids to freeform cutout genes.
        
        Args:
            min_area: Minimum cutout area (normalized)
        
        Returns:
            List of FreeformCutout objects
        """
        contours = self.extract_contours()
        cutouts = []
        
        for contour in contours:
            # Filter by area
            area = self._polygon_area(contour)
            if area < min_area:
                continue
            
            # Skip if touches edge (not a proper internal cutout)
            if contour[:, 0].min() < 0.02 or contour[:, 0].max() > 0.98:
                continue
            if contour[:, 1].min() < 0.02 or contour[:, 1].max() > 0.98:
                continue
            
            cutout = FreeformCutout.from_contour(contour)
            cutouts.append(cutout)
        
        return cutouts
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ══════════════════════════════════════════════════════════════════════════════
# FREEFORM CUTOUT - Bezier/NURBS Parametric Shape
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FreeformCutout:
    """
    Freeform cutout with smooth parametric boundary.
    
    The shape is defined by control points that generate
    a smooth Bezier/B-spline curve, suitable for CNC milling.
    
    Unlike predefined shapes (ellipse, crescent, etc.), this
    can represent ANY smooth closed curve.
    """
    # Center position (normalized)
    center_x: float = 0.5
    center_y: float = 0.5
    
    # Control points relative to center (N, 2) array
    # Points are connected by smooth spline
    control_points: np.ndarray = None
    
    # Scale factors
    scale_x: float = 0.05
    scale_y: float = 0.05
    
    # Rotation
    rotation: float = 0.0  # radians
    
    # Spline parameters
    smoothness: int = 3     # B-spline degree (3 = cubic)
    n_samples: int = 50     # Points for final curve
    
    def __post_init__(self):
        """Initialize default control points if not provided."""
        if self.control_points is None:
            # Default: ellipse-like with 8 control points
            n = 8
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            self.control_points = np.column_stack([
                np.cos(angles),
                np.sin(angles),
            ])
    
    @classmethod
    def from_contour(cls, contour: np.ndarray, n_control: int = 12) -> 'FreeformCutout':
        """
        Create freeform cutout from an extracted contour.
        
        Simplifies the contour to a manageable number of control points
        while preserving the essential shape.
        """
        if len(contour) < 4:
            return None
        
        # Calculate center
        center_x = contour[:, 0].mean()
        center_y = contour[:, 1].mean()
        
        # Calculate scale from bounding box
        scale_x = (contour[:, 0].max() - contour[:, 0].min()) / 2
        scale_y = (contour[:, 1].max() - contour[:, 1].min()) / 2
        
        # Normalize to unit circle (relative to center)
        relative = contour - np.array([center_x, center_y])
        relative[:, 0] /= (scale_x + 1e-8)
        relative[:, 1] /= (scale_y + 1e-8)
        
        # Simplify to n_control points using angular sampling
        # (preserves shape better than uniform subsampling)
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        radii = np.linalg.norm(relative, axis=1)
        
        # Sort by angle and interpolate
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        radii_sorted = radii[sort_idx]
        
        # Sample at uniform angles
        target_angles = np.linspace(-np.pi, np.pi, n_control, endpoint=False)
        
        # Handle wrap-around interpolation
        angles_extended = np.concatenate([
            angles_sorted - 2 * np.pi,
            angles_sorted,
            angles_sorted + 2 * np.pi,
        ])
        radii_extended = np.concatenate([radii_sorted] * 3)
        
        interp_func = interp1d(angles_extended, radii_extended, kind='linear', bounds_error=False, fill_value='extrapolate')
        target_radii = interp_func(target_angles)
        target_radii = np.clip(target_radii, 0.1, 2.0)  # Reasonable bounds
        
        # Convert back to cartesian
        control_points = np.column_stack([
            target_radii * np.cos(target_angles),
            target_radii * np.sin(target_angles),
        ])
        
        return cls(
            center_x=center_x,
            center_y=center_y,
            control_points=control_points,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    
    @classmethod
    def random(
        cls,
        center_x: float = None,
        center_y: float = None,
        scale: float = None,
        n_control: int = None,
        roughness: float = 0.3,
    ) -> 'FreeformCutout':
        """
        Generate random freeform cutout.
        
        Args:
            center_x: Center X (random if None)
            center_y: Center Y (random if None)
            scale: Base scale (random if None)
            n_control: Number of control points (random 6-12 if None)
            roughness: How irregular the shape is (0=circle, 1=very rough)
        
        Returns:
            Random FreeformCutout
        """
        if center_x is None:
            center_x = np.random.uniform(0.15, 0.85)
        if center_y is None:
            center_y = np.random.uniform(0.15, 0.85)
        if scale is None:
            scale = np.random.uniform(0.03, 0.12)
        if n_control is None:
            n_control = np.random.randint(6, 13)
        
        # Generate control points with varying radii
        angles = np.linspace(0, 2 * np.pi, n_control, endpoint=False)
        
        # Add angular noise for organic shapes
        angles += np.random.normal(0, 0.1, n_control)
        angles = np.sort(angles)  # Keep sorted for proper polygon
        
        # Radii with smooth variation
        base_radii = np.ones(n_control)
        noise = np.random.normal(0, roughness, n_control)
        noise = gaussian_filter(np.hstack([noise, noise, noise]), sigma=1.0)[n_control:2*n_control]
        radii = np.clip(base_radii + noise, 0.3, 1.5)
        
        control_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles),
        ])
        
        # Random aspect ratio
        aspect = np.random.uniform(0.7, 1.3)
        scale_x = scale * aspect
        scale_y = scale / aspect
        
        return cls(
            center_x=center_x,
            center_y=center_y,
            control_points=control_points,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=np.random.uniform(0, 2 * np.pi),
        )
    
    def get_boundary(self, n_points: int = None) -> np.ndarray:
        """
        Get smooth boundary curve using B-spline interpolation.
        
        Returns:
            Array of (n_points, 2) with (x, y) coordinates (normalized)
        """
        if n_points is None:
            n_points = self.n_samples
        
        if self.control_points is None or len(self.control_points) < 4:
            return np.array([])
        
        # Close the curve
        closed_points = np.vstack([self.control_points, self.control_points[0]])
        
        try:
            # Fit B-spline
            tck, u = splprep([closed_points[:, 0], closed_points[:, 1]], 
                            s=0, k=min(self.smoothness, len(closed_points) - 1), per=True)
            
            # Evaluate at uniform parameters
            u_new = np.linspace(0, 1, n_points, endpoint=False)
            x, y = splev(u_new, tck)
            
            # Apply scale and rotation
            points = np.column_stack([x, y])
            
            # Rotation
            c, s = np.cos(self.rotation), np.sin(self.rotation)
            rot_matrix = np.array([[c, -s], [s, c]])
            points = points @ rot_matrix.T
            
            # Scale and translate
            points[:, 0] = points[:, 0] * self.scale_x + self.center_x
            points[:, 1] = points[:, 1] * self.scale_y + self.center_y
            
            return points
        
        except Exception as e:
            logger.warning(f"B-spline fitting failed: {e}")
            # Fallback to simple polygon
            points = self.control_points.copy()
            points[:, 0] = points[:, 0] * self.scale_x + self.center_x
            points[:, 1] = points[:, 1] * self.scale_y + self.center_y
            return points
    
    def mutate(self, sigma: float = 0.05) -> 'FreeformCutout':
        """
        Mutate the cutout shape.
        
        Mutations can affect:
        - Position (center)
        - Scale (size)
        - Rotation
        - Control points (shape deformation)
        """
        new_points = self.control_points.copy()
        
        # Deform control points
        if np.random.random() < 0.7:
            # Smooth deformation (affects neighboring points similarly)
            noise = np.random.normal(0, sigma * 0.5, new_points.shape)
            # Smooth the noise
            noise[:, 0] = gaussian_filter(np.hstack([noise[:, 0]] * 3), sigma=1.0)[len(noise):2*len(noise)]
            noise[:, 1] = gaussian_filter(np.hstack([noise[:, 1]] * 3), sigma=1.0)[len(noise):2*len(noise)]
            new_points += noise
        
        # Add/remove control point
        if np.random.random() < 0.1 and len(new_points) > 5:
            # Remove random point
            idx = np.random.randint(len(new_points))
            new_points = np.delete(new_points, idx, axis=0)
        elif np.random.random() < 0.1 and len(new_points) < 15:
            # Add point between two existing
            idx = np.random.randint(len(new_points))
            next_idx = (idx + 1) % len(new_points)
            new_pt = (new_points[idx] + new_points[next_idx]) / 2
            new_pt += np.random.normal(0, 0.1, 2)
            new_points = np.insert(new_points, next_idx, new_pt, axis=0)
        
        # Ensure points stay in reasonable range
        new_points = np.clip(new_points, -1.5, 1.5)
        
        return FreeformCutout(
            center_x=np.clip(self.center_x + np.random.normal(0, sigma), 0.1, 0.9),
            center_y=np.clip(self.center_y + np.random.normal(0, sigma), 0.1, 0.9),
            control_points=new_points,
            scale_x=np.clip(self.scale_x * np.exp(np.random.normal(0, sigma * 0.3)), 0.02, 0.2),
            scale_y=np.clip(self.scale_y * np.exp(np.random.normal(0, sigma * 0.3)), 0.02, 0.2),
            rotation=(self.rotation + np.random.normal(0, 0.2)) % (2 * np.pi),
            smoothness=self.smoothness,
        )
    
    def crossover(self, other: 'FreeformCutout') -> 'FreeformCutout':
        """
        Crossover with another freeform cutout.
        
        Creates a child that blends features of both parents.
        """
        # Blend centers
        alpha = np.random.uniform(0.3, 0.7)
        new_center_x = alpha * self.center_x + (1 - alpha) * other.center_x
        new_center_y = alpha * self.center_y + (1 - alpha) * other.center_y
        
        # Blend scales
        new_scale_x = np.sqrt(self.scale_x * other.scale_x)
        new_scale_y = np.sqrt(self.scale_y * other.scale_y)
        
        # Blend control points (needs matching count)
        if len(self.control_points) == len(other.control_points):
            new_points = alpha * self.control_points + (1 - alpha) * other.control_points
        else:
            # Use the one with more detail
            if len(self.control_points) > len(other.control_points):
                new_points = self.control_points.copy()
            else:
                new_points = other.control_points.copy()
        
        return FreeformCutout(
            center_x=new_center_x,
            center_y=new_center_y,
            control_points=new_points,
            scale_x=new_scale_x,
            scale_y=new_scale_y,
            rotation=np.random.choice([self.rotation, other.rotation]),
        )
    
    def to_cutout_gene(self) -> Dict[str, Any]:
        """Convert to CutoutGene-compatible dict for integration."""
        return {
            'x': self.center_x,
            'y': self.center_y,
            'width': self.scale_x * 2,  # Convert radius to diameter
            'height': self.scale_y * 2,
            'rotation': self.rotation,
            'shape': 'freeform',
            'control_points': self.control_points.copy(),
        }
    
    def area(self) -> float:
        """Calculate approximate area of the cutout."""
        boundary = self.get_boundary()
        if len(boundary) < 3:
            return 0.0
        
        x, y = boundary[:, 0], boundary[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ══════════════════════════════════════════════════════════════════════════════
# FREEFORM CUTOUT OPTIMIZER - Combines Topology + Parametric
# ══════════════════════════════════════════════════════════════════════════════

class FreeformCutoutOptimizer:
    """
    Optimizer for freeform cutout shapes.
    
    Combines two approaches:
    1. TOPOLOGY: Density field evolution (SIMP-style)
    2. PARAMETRIC: Direct control point optimization (Bezier)
    
    The topology approach is used for initial shape discovery,
    then parametric optimization refines the shapes.
    
    USAGE:
        optimizer = FreeformCutoutOptimizer()
        
        # Initialize from physics
        optimizer.initialize_from_physics(mode_shapes, frequencies)
        
        # Evolve shapes
        for generation in range(100):
            sensitivity = compute_sensitivity(...)  # From FEM or surrogate
            optimizer.evolve(sensitivity)
        
        # Get final cutouts
        cutouts = optimizer.get_cutouts()
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (100, 60),
        max_cutouts: int = 8,
        preserve_spine: bool = True,
        spine_zone: Tuple[float, float, float, float] = (0.35, 0.65, 0.35, 0.65),
    ):
        """
        Initialize optimizer.
        
        Args:
            resolution: Grid resolution for density field
            max_cutouts: Maximum number of cutouts to generate
            preserve_spine: Whether to preserve the spine load zone
            spine_zone: (x_min, x_max, y_min, y_max) of spine zone
        """
        self.resolution = resolution
        self.max_cutouts = max_cutouts
        self.preserve_spine = preserve_spine
        self.spine_zone = spine_zone
        
        # Density field for topology optimization
        self.density = DensityField(resolution=resolution)
        
        # Parametric cutouts (extracted from density field)
        self.cutouts: List[FreeformCutout] = []
        
        # Evolution tracking
        self.generation = 0
        self.best_fitness = 0.0
        self.fitness_history: List[float] = []
    
    def initialize_from_physics(
        self,
        mode_shapes: np.ndarray,
        current_frequencies: List[float],
        target_frequencies: List[float],
        initial_void_ratio: float = 0.1,
    ):
        """
        Initialize density field based on modal physics.
        
        This gives the optimizer a physics-informed starting point
        rather than random initialization.
        """
        self.density = DensityField.from_physics_guidance(
            mode_shapes=mode_shapes,
            target_frequencies=target_frequencies,
            current_frequencies=current_frequencies,
            resolution=self.resolution,
            initial_void_ratio=initial_void_ratio,
        )
        
        # Extract initial cutouts
        self._extract_cutouts()
    
    def evolve(
        self,
        sensitivity_field: np.ndarray,
        fitness_value: float,
        learning_rate: float = 0.1,
    ):
        """
        Evolve cutout shapes based on sensitivity analysis.
        
        Args:
            sensitivity_field: Physics-based sensitivity (where cutting helps)
            fitness_value: Current fitness (for tracking)
            learning_rate: Evolution step size
        """
        self.generation += 1
        self.fitness_history.append(fitness_value)
        
        if fitness_value > self.best_fitness:
            self.best_fitness = fitness_value
        
        # Determine preserve zones
        preserve = []
        if self.preserve_spine:
            preserve.append(self.spine_zone)
        
        # Evolve density field
        self.density.evolve(
            sensitivity_field=sensitivity_field,
            learning_rate=learning_rate,
            preserve_zones=preserve,
        )
        
        # Extract parametric cutouts
        self._extract_cutouts()
        
        # Refine via parametric mutation (every 5 generations)
        if self.generation % 5 == 0:
            self._refine_cutouts()
    
    def _extract_cutouts(self):
        """Extract parametric cutouts from density field."""
        self.cutouts = self.density.to_cutout_genes(min_area=0.002)
        
        # Limit to max_cutouts (keep largest)
        if len(self.cutouts) > self.max_cutouts:
            self.cutouts.sort(key=lambda c: c.area(), reverse=True)
            self.cutouts = self.cutouts[:self.max_cutouts]
    
    def _refine_cutouts(self, sigma: float = 0.02):
        """Apply small mutations to refine cutout shapes."""
        for i, cutout in enumerate(self.cutouts):
            if np.random.random() < 0.5:
                self.cutouts[i] = cutout.mutate(sigma=sigma)
    
    def get_cutouts(self) -> List[FreeformCutout]:
        """Get current freeform cutouts."""
        return self.cutouts
    
    def get_cutout_genes(self) -> List[Dict[str, Any]]:
        """Get cutouts as CutoutGene-compatible dicts."""
        return [c.to_cutout_gene() for c in self.cutouts]
    
    def visualize(self) -> np.ndarray:
        """
        Create visualization of density field and cutouts.
        
        Returns:
            RGB image array (ny, nx, 3) for display
        """
        ny, nx = self.density.resolution
        img = np.zeros((ny, nx, 3))
        
        # Background: density field (grayscale → blue channel)
        img[:, :, 0] = self.density.field * 0.3  # R
        img[:, :, 1] = self.density.field * 0.3  # G
        img[:, :, 2] = self.density.field * 0.8  # B (stronger for density)
        
        # Draw cutout boundaries in yellow
        for cutout in self.cutouts:
            boundary = cutout.get_boundary()
            for i in range(len(boundary) - 1):
                x1, y1 = boundary[i]
                x2, y2 = boundary[i + 1]
                self._draw_line(img, 
                               int(x1 * nx), int(y1 * ny),
                               int(x2 * nx), int(y2 * ny),
                               color=(1.0, 1.0, 0.0))
        
        # Mark spine zone in green (faint)
        if self.preserve_spine:
            x1, x2, y1, y2 = self.spine_zone
            ix1, ix2 = int(x1 * nx), int(x2 * nx)
            iy1, iy2 = int(y1 * ny), int(y2 * ny)
            img[iy1:iy2, ix1:ix2, 1] += 0.2
        
        return np.clip(img, 0, 1)
    
    def _draw_line(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[float, float, float]):
        """Draw a line on the image using Bresenham's algorithm."""
        ny, nx = img.shape[:2]
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if 0 <= x1 < nx and 0 <= y1 < ny:
                img[y1, x1] = color
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_random_freeform_population(
    n_cutouts: int = 4,
    n_individuals: int = 20,
    center_bounds: Tuple[float, float, float, float] = (0.15, 0.85, 0.15, 0.85),
) -> List[List[FreeformCutout]]:
    """
    Generate population of random freeform cutout configurations.
    
    Useful for initializing genetic algorithm population.
    
    Args:
        n_cutouts: Number of cutouts per individual
        n_individuals: Population size
        center_bounds: (x_min, x_max, y_min, y_max) for cutout centers
    
    Returns:
        List of n_individuals, each with n_cutouts FreeformCutouts
    """
    population = []
    
    for _ in range(n_individuals):
        individual = []
        for _ in range(n_cutouts):
            cutout = FreeformCutout.random(
                center_x=np.random.uniform(center_bounds[0], center_bounds[1]),
                center_y=np.random.uniform(center_bounds[2], center_bounds[3]),
            )
            individual.append(cutout)
        population.append(individual)
    
    return population
