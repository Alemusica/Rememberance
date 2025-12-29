"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FEM ANALYTICAL - Closed-Form Solution                     ║
║                                                                              ║
║   Fast analytical approximation for rectangular plates.                      ║
║   Uses Kirchhoff-Love thin plate theory with free-free boundaries.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import time
from typing import List, Optional

from .interface import (
    FEMSolver, FEMResult, FEMMode, MeshData, ShapeType,
    calculate_flexural_rigidity
)
from ..materials import Material


class AnalyticalSolver(FEMSolver):
    """
    Analytical solver using closed-form solutions.
    
    Fast but limited to rectangular plates with uniform properties.
    Good for quick estimates and validation.
    """
    
    name = "Analytical (Kirchhoff)"
    
    def solve(
        self,
        mesh: MeshData,
        thickness: float,
        material: Material,
        n_modes: int = 10,
        **kwargs
    ) -> FEMResult:
        """
        Compute modes using analytical formulas.
        
        For non-rectangular shapes, uses bounding box approximation.
        """
        start_time = time.time()
        warnings = []
        
        # Get dimensions
        L = mesh.length
        W = mesh.width
        h = thickness
        
        # Warn if not rectangular
        if mesh.shape_type != ShapeType.RECTANGLE:
            warnings.append(
                f"Analytical solver assumes rectangle. "
                f"Using bounding box for {mesh.shape_type.value} shape."
            )
        
        # Material properties
        rho = material.density
        nu = material.poisson_ratio
        
        # Check if orthotropic
        is_orthotropic = not material.is_isotropic
        
        if is_orthotropic:
            E_L = material.E_longitudinal
            E_W = material.E_transverse
        else:
            E_L = E_W = material.E_mean
        
        # Compute modes
        modes = []
        mode_idx = 0
        
        # Search through mode number combinations
        for m in range(1, 10):
            for n in range(1, 10):
                if mode_idx >= n_modes:
                    break
                
                # Calculate frequency
                if is_orthotropic:
                    freq = self._freq_orthotropic(m, n, L, W, h, E_L, E_W, rho, nu)
                else:
                    freq = self._freq_isotropic(m, n, L, W, h, E_L, rho, nu)
                
                # Generate mode shape on mesh points
                mode_shape = self._compute_mode_shape(
                    mesh.points, m, n, mesh.bounding_box
                )
                
                modes.append(FEMMode(
                    index=mode_idx,
                    frequency=freq,
                    eigenvalue=(2 * np.pi * freq) ** 2 * rho * h,
                    mode_shape=mode_shape,
                    mesh=mesh,
                    m=m,
                    n=n
                ))
                
                mode_idx += 1
            
            if mode_idx >= n_modes:
                break
        
        # Sort by frequency
        modes.sort(key=lambda m: m.frequency)
        
        # Re-index
        for i, mode in enumerate(modes):
            mode.index = i
        
        solve_time = time.time() - start_time
        
        return FEMResult(
            modes=modes[:n_modes],
            mesh=mesh,
            material=material,
            thickness=thickness,
            solve_time=solve_time,
            solver_name=self.name,
            converged=True,
            warnings=warnings
        )
    
    def _freq_isotropic(
        self, 
        m: int, n: int,
        L: float, W: float, h: float,
        E: float, rho: float, nu: float
    ) -> float:
        """
        Modal frequency for isotropic rectangular plate (free-free).
        
        Uses beam mode shape eigenvalues for free-free boundaries.
        """
        D = calculate_flexural_rigidity(E, h, nu)
        
        # Free-free eigenvalues: λ ≈ (m + 0.5)π for m ≥ 1
        lambda_m = (m + 0.5) * np.pi
        lambda_n = (n + 0.5) * np.pi
        
        # Frequency formula
        f = (1 / (2 * np.pi)) * np.sqrt(D / (rho * h)) * (
            (lambda_m / L)**2 + (lambda_n / W)**2
        )
        
        return f
    
    def _freq_orthotropic(
        self,
        m: int, n: int,
        L: float, W: float, h: float,
        E_L: float, E_W: float, rho: float, nu: float
    ) -> float:
        """
        Modal frequency for orthotropic rectangular plate.
        
        Wood has different stiffness along/across grain.
        """
        D_L = calculate_flexural_rigidity(E_L, h, nu)
        D_W = calculate_flexural_rigidity(E_W, h, nu)
        D_LW = np.sqrt(D_L * D_W) * nu  # Cross-coupling term
        
        # Free-free eigenvalues
        lambda_m = (m + 0.5) * np.pi
        lambda_n = (n + 0.5) * np.pi
        
        # Orthotropic plate formula
        term1 = D_L * (lambda_m / L)**4
        term2 = D_W * (lambda_n / W)**4
        term3 = 2 * D_LW * (lambda_m / L)**2 * (lambda_n / W)**2
        
        f = (1 / (2 * np.pi)) * np.sqrt((term1 + term2 + term3) / (rho * h))
        
        return f
    
    def _compute_mode_shape(
        self,
        points: np.ndarray,
        m: int, n: int,
        bbox: tuple
    ) -> np.ndarray:
        """
        Compute mode shape at mesh points.
        
        Free-free mode shapes use cosine functions.
        """
        x_min, y_min, x_max, y_max = bbox
        L = x_max - x_min
        W = y_max - y_min
        
        # Normalize coordinates to [0, 1]
        x_norm = (points[:, 0] - x_min) / L
        y_norm = (points[:, 1] - y_min) / W
        
        # Mode shape: cos(m*π*x/L) * cos(n*π*y/W)
        shape = np.cos(m * np.pi * x_norm) * np.cos(n * np.pi * y_norm)
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(shape))
        if max_val > 0:
            shape = shape / max_val
        
        return shape
    
    @classmethod
    def is_available(cls) -> bool:
        """Always available - no external dependencies."""
        return True
