"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               ITERATIVE OPTIMIZER - Zone-Based Plate Design                  ║
║                                                                              ║
║   Topology optimization with SIMP/RAMP material interpolation.               ║
║                                                                              ║
║   Based on research from:                                                     ║
║   • stefanhiemer/topoptlab - SIMP, RAMP, bound interpolation                 ║
║   • Sageblatt/plate_inverse_problem - JAX-differentiable FEM                 ║
║   • aatmdelissen/pyMOTO - Modular topology optimization                      ║
║                                                                              ║
║   Optimization Goal:                                                          ║
║   Find density distribution ρ(x,y) that maximizes coupling to body zones     ║
║   at specified target frequencies.                                            ║
║                                                                              ║
║   Physics:                                                                    ║
║   E(ρ) = E_min + (E_max - E_min) * ρ^p    [SIMP]                            ║
║   E(ρ) = E_min + (E_max - E_min) * ρ/(1 + p*(1-ρ))    [RAMP]               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod
import json

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL INTERPOLATION SCHEMES
# From stefanhiemer/topoptlab
# ══════════════════════════════════════════════════════════════════════════════

class InterpolationScheme(Enum):
    """Material interpolation methods for topology optimization."""
    SIMP = "simp"          # Solid Isotropic Material with Penalization
    RAMP = "ramp"          # Rational Approximation of Material Properties
    BOUND = "bound"        # Hashin-Shtrikman bounds
    LINEAR = "linear"      # Simple linear interpolation


def simp(x: np.ndarray, penalty: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """
    SIMP interpolation: E(x) = eps + (1 - eps) * x^penalty
    
    Args:
        x: Density field [0, 1]
        penalty: Penalization factor (typically 3)
        eps: Minimum stiffness ratio (avoid singularity)
    
    Returns:
        Stiffness scaling factor
    """
    return eps + (1 - eps) * np.power(x, penalty)


def simp_dx(x: np.ndarray, penalty: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """
    SIMP derivative: dE/dx = penalty * (1 - eps) * x^(penalty-1)
    
    Used for sensitivity analysis in gradient-based optimization.
    """
    return penalty * (1 - eps) * np.power(x, penalty - 1)


def ramp(x: np.ndarray, penalty: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """
    RAMP interpolation: E(x) = eps + (1 - eps) * x / (1 + penalty*(1-x))
    
    More gradual penalization than SIMP, better for intermediate densities.
    """
    return eps + (1 - eps) * x / (1 + penalty * (1 - x))


def ramp_dx(x: np.ndarray, penalty: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """
    RAMP derivative: dE/dx = (1 - eps) * (1 + penalty) / (1 + penalty*(1-x))^2
    """
    denom = 1 + penalty * (1 - x)
    return (1 - eps) * (1 + penalty) / (denom * denom)


def bound_interpolation(
    x: np.ndarray, 
    penalty: float = 3.0, 
    eps: float = 1e-6
) -> np.ndarray:
    """
    Hashin-Shtrikman bound interpolation.
    
    Provides physical lower bounds on effective properties.
    """
    # Lower HS bound for 2-phase material
    nu = 0.3  # Poisson's ratio (typical)
    G = x / (3 - x)  # Shear modulus bound
    K = x / (3 * (1 - x) + 3 * x * (1 + nu) / (3 * (1 - 2*nu)))  # Bulk bound
    return eps + (1 - eps) * G


# ══════════════════════════════════════════════════════════════════════════════
# DENSITY FILTER (for manufacturable designs)
# ══════════════════════════════════════════════════════════════════════════════

def density_filter(
    x: np.ndarray,
    radius: float,
    dx: float = 1.0,
    dy: float = 1.0
) -> np.ndarray:
    """
    Apply density filter to avoid checkerboard patterns.
    
    Weighted average within radius:
    x_filtered[i,j] = Σ w[k,l] * x[k,l] / Σ w[k,l]
    
    where w = max(0, radius - distance)
    
    Args:
        x: Density field (nx, ny)
        radius: Filter radius
        dx, dy: Element sizes
    
    Returns:
        Filtered density field
    """
    nx, ny = x.shape
    x_filtered = np.zeros_like(x)
    
    # Precompute filter kernel
    r_cells = int(np.ceil(radius / min(dx, dy)))
    
    for i in range(nx):
        for j in range(ny):
            total_weight = 0.0
            weighted_sum = 0.0
            
            for di in range(-r_cells, r_cells + 1):
                for dj in range(-r_cells, r_cells + 1):
                    ii = i + di
                    jj = j + dj
                    
                    if 0 <= ii < nx and 0 <= jj < ny:
                        dist = np.sqrt((di * dx)**2 + (dj * dy)**2)
                        weight = max(0, radius - dist)
                        total_weight += weight
                        weighted_sum += weight * x[ii, jj]
            
            x_filtered[i, j] = weighted_sum / total_weight if total_weight > 0 else x[i, j]
    
    return x_filtered


def convolution_filter(x: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Fast convolution-based filter using numpy.
    
    Faster than element-by-element filter for large grids.
    """
    from scipy.ndimage import uniform_filter
    return uniform_filter(x, size=kernel_size, mode='reflect')


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION PROBLEM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationConfig:
    """Configuration for iterative optimization."""
    # Grid resolution
    nx: int = 50
    ny: int = 30
    
    # Material interpolation
    interpolation: InterpolationScheme = InterpolationScheme.SIMP
    penalty: float = 3.0
    eps: float = 1e-6
    
    # Density filter
    filter_radius: float = 1.5  # In element sizes
    
    # Volume constraint
    volume_fraction: float = 0.5  # Target material usage
    
    # DENSITY LIMITS - Evita buchi!
    min_density: float = 0.3  # Minimo 30% - niente buchi
    max_density: float = 1.0  # Massimo 100%
    
    # Optimization parameters
    max_iterations: int = 100
    convergence_tol: float = 1e-4
    move_limit: float = 0.15  # Max density change per iteration (ridotto)
    
    # Learning rate for gradient descent
    learning_rate: float = 0.5
    
    # Continuation (gradually increase penalty)
    use_continuation: bool = True
    penalty_start: float = 1.0
    penalty_increment: float = 0.5


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    density: np.ndarray
    frequencies: np.ndarray
    objective_history: List[float]
    constraint_history: List[float]
    converged: bool
    iterations: int
    final_penalty: float
    
    def save(self, filepath: str):
        """Save result to JSON."""
        data = {
            "density": self.density.tolist(),
            "frequencies": self.frequencies.tolist(),
            "objective_history": self.objective_history,
            "constraint_history": self.constraint_history,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_penalty": self.final_penalty,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS (from Sageblatt/plate_inverse_problem)
# ══════════════════════════════════════════════════════════════════════════════

class ObjectiveFunction(ABC):
    """Base class for optimization objectives."""
    
    @abstractmethod
    def evaluate(
        self, 
        achieved_frequencies: np.ndarray, 
        target_frequencies: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Evaluate objective (lower is better)."""
        pass
    
    @abstractmethod
    def gradient(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        df_dx: np.ndarray,  # Sensitivity of frequencies to density
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate gradient of objective w.r.t. density."""
        pass


class MSEObjective(ObjectiveFunction):
    """Mean Squared Error: Σ w_i * (f_achieved - f_target)²"""
    
    def evaluate(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        if weights is None:
            weights = np.ones(len(target_frequencies))
        
        # Match achieved to target (closest frequency)
        error = 0.0
        for i, f_target in enumerate(target_frequencies):
            if len(achieved_frequencies) > 0:
                closest_idx = np.argmin(np.abs(achieved_frequencies - f_target))
                f_achieved = achieved_frequencies[closest_idx]
                error += weights[i] * (f_achieved - f_target)**2
            else:
                error += weights[i] * f_target**2  # Penalize if no frequencies
        
        return error / np.sum(weights)
    
    def gradient(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        df_dx: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(target_frequencies))
        
        grad = np.zeros(df_dx.shape[1:])  # (nx, ny)
        
        for i, f_target in enumerate(target_frequencies):
            if len(achieved_frequencies) > 0:
                closest_idx = np.argmin(np.abs(achieved_frequencies - f_target))
                f_achieved = achieved_frequencies[closest_idx]
                # d(MSE)/dx = 2 * w * (f - f_target) * df/dx
                grad += 2 * weights[i] * (f_achieved - f_target) * df_dx[closest_idx]
        
        return grad / np.sum(weights)


class RMSEObjective(ObjectiveFunction):
    """Root Mean Squared Error: sqrt(MSE)"""
    
    def __init__(self):
        self.mse = MSEObjective()
    
    def evaluate(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        return np.sqrt(self.mse.evaluate(achieved_frequencies, target_frequencies, weights))
    
    def gradient(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        df_dx: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        mse_val = self.mse.evaluate(achieved_frequencies, target_frequencies, weights)
        if mse_val < 1e-10:
            return np.zeros(df_dx.shape[1:])
        
        mse_grad = self.mse.gradient(achieved_frequencies, target_frequencies, df_dx, weights)
        return mse_grad / (2 * np.sqrt(mse_val))


class LogMSEObjective(ObjectiveFunction):
    """Log MSE for relative frequency matching: Σ w * log(f/f_target)²"""
    
    def evaluate(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        if weights is None:
            weights = np.ones(len(target_frequencies))
        
        error = 0.0
        for i, f_target in enumerate(target_frequencies):
            if len(achieved_frequencies) > 0 and f_target > 0:
                closest_idx = np.argmin(np.abs(achieved_frequencies - f_target))
                f_achieved = max(achieved_frequencies[closest_idx], 1e-6)
                error += weights[i] * np.log(f_achieved / f_target)**2
            else:
                error += weights[i] * 10.0  # Large penalty
        
        return error / np.sum(weights)
    
    def gradient(
        self,
        achieved_frequencies: np.ndarray,
        target_frequencies: np.ndarray,
        df_dx: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(target_frequencies))
        
        grad = np.zeros(df_dx.shape[1:])
        
        for i, f_target in enumerate(target_frequencies):
            if len(achieved_frequencies) > 0 and f_target > 0:
                closest_idx = np.argmin(np.abs(achieved_frequencies - f_target))
                f_achieved = max(achieved_frequencies[closest_idx], 1e-6)
                # d/dx[log(f/ft)²] = 2*log(f/ft) * (1/f) * df/dx
                grad += 2 * weights[i] * np.log(f_achieved / f_target) / f_achieved * df_dx[closest_idx]
        
        return grad / np.sum(weights)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZERS (from Sageblatt/plate_inverse_problem)
# ══════════════════════════════════════════════════════════════════════════════

class Optimizer(ABC):
    """Base class for optimization algorithms."""
    
    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        gradient: np.ndarray,
        config: OptimizationConfig
    ) -> np.ndarray:
        """Take one optimization step."""
        pass


class GradientDescentOptimizer(Optimizer):
    """Simple gradient descent with move limits."""
    
    def step(
        self,
        x: np.ndarray,
        gradient: np.ndarray,
        config: OptimizationConfig
    ) -> np.ndarray:
        # Scale gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-10:
            gradient = gradient / grad_norm
        
        # Update
        x_new = x - config.learning_rate * gradient
        
        # Apply move limit
        change = x_new - x
        change = np.clip(change, -config.move_limit, config.move_limit)
        x_new = x + change
        
        # Clip to [0, 1]
        x_new = np.clip(x_new, 0.0, 1.0)
        
        return x_new


class MomentumOptimizer(Optimizer):
    """Gradient descent with momentum."""
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.velocity = None
    
    def step(
        self,
        x: np.ndarray,
        gradient: np.ndarray,
        config: OptimizationConfig
    ) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        
        # Update velocity
        self.velocity = self.momentum * self.velocity - config.learning_rate * gradient
        
        # Apply move limit to velocity
        self.velocity = np.clip(self.velocity, -config.move_limit, config.move_limit)
        
        # Update position
        x_new = x + self.velocity
        x_new = np.clip(x_new, 0.0, 1.0)
        
        return x_new


class OC_Optimizer(Optimizer):
    """
    Optimality Criteria method.
    
    Classic topology optimization update scheme:
    x_new = x * sqrt(-dC/dx / (λ * dV/dx))
    
    where λ is Lagrange multiplier for volume constraint.
    """
    
    def step(
        self,
        x: np.ndarray,
        gradient: np.ndarray,  # dObjective/dx
        config: OptimizationConfig,
        volume_grad: Optional[np.ndarray] = None  # dVolume/dx
    ) -> np.ndarray:
        if volume_grad is None:
            volume_grad = np.ones_like(x)  # d(Σx)/dx = 1 for each element
        
        # Current volume
        vol_current = np.mean(x)
        vol_target = config.volume_fraction
        
        # Bisection to find Lagrange multiplier
        lambda_low = 1e-10
        lambda_high = 1e10
        
        # Get density limits from config
        rho_min = getattr(config, 'min_density', 0.3)
        rho_max = getattr(config, 'max_density', 1.0)
        
        for _ in range(50):  # Bisection iterations
            lambda_mid = 0.5 * (lambda_low + lambda_high)
            
            # OC update formula
            B = -gradient / (lambda_mid * volume_grad + 1e-10)
            B = np.maximum(B, 1e-10)  # Ensure positive
            
            x_new = x * np.sqrt(B)
            
            # Apply move limits
            x_new = np.minimum(x_new, x + config.move_limit)
            x_new = np.maximum(x_new, x - config.move_limit)
            
            # VINCOLO CRITICO: clip tra min_density e max_density
            # Questo evita buchi (density=0) nella tavola!
            x_new = np.clip(x_new, rho_min, rho_max)
            
            # Check volume
            vol_new = np.mean(x_new)
            
            if vol_new > vol_target:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
            
            if abs(vol_new - vol_target) < 1e-6:
                break
        
        return x_new


# ══════════════════════════════════════════════════════════════════════════════
# ZONE-AWARE ITERATIVE OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class ZoneIterativeOptimizer:
    """
    Zone-based iterative plate optimizer.
    
    Optimizes density distribution to match zone-specific frequency targets.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        zones: List['BodyZone'],
        objective: ObjectiveFunction = None,
        optimizer: Optimizer = None
    ):
        self.config = config
        self.zones = zones
        self.objective = objective or MSEObjective()
        self.optimizer = optimizer or OC_Optimizer()
        
        # Initialize density field
        self.x = np.ones((config.nx, config.ny)) * config.volume_fraction
        
        # History
        self.history = []
    
    def _get_zone_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract all target frequencies and weights from zones."""
        all_targets = []
        all_weights = []
        
        for zone in self.zones:
            for i, f in enumerate(zone.target_frequencies):
                all_targets.append(f)
                w = zone.frequency_weights[i] if i < len(zone.frequency_weights) else 1.0
                all_weights.append(w * zone.optimization_weight)
        
        return np.array(all_targets), np.array(all_weights)
    
    def _compute_frequencies(
        self,
        x: np.ndarray,
        fem_solver: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenfrequencies and their sensitivities.
        
        Args:
            x: Density field
            fem_solver: Function that takes density and returns (frequencies, df/dx)
        
        Returns:
            (frequencies, sensitivity df/dx of shape (n_modes, nx, ny))
        """
        return fem_solver(x)
    
    def optimize(
        self,
        fem_solver: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> OptimizationResult:
        """
        Run iterative optimization.
        
        Args:
            fem_solver: FEM solver returning (frequencies, df/dx)
            callback: Optional callback(iteration, density, objective)
        
        Returns:
            OptimizationResult
        """
        targets, weights = self._get_zone_targets()
        penalty = self.config.penalty_start if self.config.use_continuation else self.config.penalty
        
        objective_history = []
        constraint_history = []
        
        x = self.x.copy()
        x_old = x.copy()
        
        for iteration in range(self.config.max_iterations):
            # Apply interpolation
            if self.config.interpolation == InterpolationScheme.SIMP:
                x_phys = simp(x, penalty, self.config.eps)
                dx_scale = simp_dx(x, penalty, self.config.eps)
            elif self.config.interpolation == InterpolationScheme.RAMP:
                x_phys = ramp(x, penalty, self.config.eps)
                dx_scale = ramp_dx(x, penalty, self.config.eps)
            else:
                x_phys = x
                dx_scale = np.ones_like(x)
            
            # Apply filter
            x_filtered = density_filter(x_phys, self.config.filter_radius)
            
            # Compute frequencies
            frequencies, df_dx = fem_solver(x_filtered)
            
            # Compute objective
            obj_value = self.objective.evaluate(frequencies, targets, weights)
            objective_history.append(obj_value)
            
            # Volume constraint
            vol = np.mean(x)
            vol_constraint = abs(vol - self.config.volume_fraction)
            constraint_history.append(vol_constraint)
            
            # Callback
            if callback:
                callback(iteration, x, obj_value)
            
            # Compute gradient
            obj_grad = self.objective.gradient(frequencies, targets, df_dx, weights)
            
            # Scale by interpolation derivative
            obj_grad = obj_grad * dx_scale
            
            # Optimizer step
            if isinstance(self.optimizer, OC_Optimizer):
                x_new = self.optimizer.step(x, obj_grad, self.config)
            else:
                x_new = self.optimizer.step(x, obj_grad, self.config)
            
            # Check convergence
            change = np.max(np.abs(x_new - x))
            if change < self.config.convergence_tol:
                # Continuation: increase penalty
                if self.config.use_continuation and penalty < self.config.penalty:
                    penalty += self.config.penalty_increment
                    penalty = min(penalty, self.config.penalty)
                    print(f"Iteration {iteration}: Increasing penalty to {penalty:.1f}")
                else:
                    print(f"✓ Converged at iteration {iteration}, change = {change:.6f}")
                    self.x = x_new
                    return OptimizationResult(
                        density=x_new,
                        frequencies=frequencies,
                        objective_history=objective_history,
                        constraint_history=constraint_history,
                        converged=True,
                        iterations=iteration + 1,
                        final_penalty=penalty
                    )
            
            x_old = x.copy()
            x = x_new
            
            # Progress print every 10 iterations
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d}: Obj={obj_value:.4f}, Vol={vol:.2%}, Change={change:.4f}")
        
        self.x = x
        return OptimizationResult(
            density=x,
            frequencies=frequencies,
            objective_history=objective_history,
            constraint_history=constraint_history,
            converged=False,
            iterations=self.config.max_iterations,
            final_penalty=penalty
        )
    
    def zone_weighted_density(self) -> np.ndarray:
        """
        Create density field with zone-specific weights.
        
        Initializes higher density in important zones.
        """
        x = np.ones((self.config.nx, self.config.ny)) * 0.3
        
        for zone in self.zones:
            # Zone bounds in grid
            ix_start = int(zone.position_start * self.config.nx)
            ix_end = int(zone.position_end * self.config.nx)
            iy_center = self.config.ny // 2
            iy_extent = int(zone.lateral_extent * self.config.ny)
            
            # Set higher density in important zones
            importance = zone.optimization_weight
            density = min(1.0, 0.3 + 0.2 * importance)
            
            x[ix_start:ix_end, iy_center-iy_extent:iy_center+iy_extent] = density
        
        # Normalize to target volume fraction
        current_vol = np.mean(x)
        if current_vol > 0:
            x = x * (self.config.volume_fraction / current_vol)
            x = np.clip(x, 0.0, 1.0)
        
        self.x = x
        return x


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLIFIED FEM SOLVER (for testing)
# ══════════════════════════════════════════════════════════════════════════════

def simple_plate_fem(
    density: np.ndarray,
    length: float = 2.0,
    width: float = 0.6,
    thickness: float = 0.02,
    E_base: float = 12e9,
    rho_base: float = 400.0,
    n_modes: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified plate FEM using Rayleigh-Ritz for testing.
    
    This is a simplified approximation - use create_jax_fem_solver() 
    for real physics-based optimization!
    
    Args:
        density: Density field (nx, ny) in [0, 1]
        length, width, thickness: Plate dimensions
        E_base, rho_base: Base material properties
        n_modes: Number of modes to compute
    
    Returns:
        (frequencies, df/dx sensitivity)
    """
    nx, ny = density.shape
    dx = length / nx
    dy = width / ny
    
    # Effective properties
    E_eff = E_base * density
    rho_eff = rho_base * density
    
    # Average properties for each mode
    E_avg = np.mean(E_eff)
    rho_avg = np.mean(rho_eff)
    
    # Flexural rigidity
    D = E_avg * thickness**3 / (12 * (1 - 0.35**2))
    
    # Approximate eigenfrequencies for simply supported plate
    # f_mn = (π/2) * sqrt(D/(ρ*h)) * ((m/L)² + (n/W)²)
    frequencies = []
    for m in range(1, n_modes + 1):
        for n in range(1, 3):  # Low n modes
            f_mn = (np.pi / 2) * np.sqrt(D / (rho_avg * thickness)) * (
                (m / length)**2 + (n / width)**2
            )
            frequencies.append(f_mn)
    
    frequencies = np.sort(np.array(frequencies))[:n_modes]
    
    # Simplified sensitivity (proportional to local stiffness contribution)
    df_dx = np.zeros((n_modes, nx, ny))
    for i in range(n_modes):
        # Higher modes more sensitive to local stiffness changes
        df_dx[i] = frequencies[i] * density / (2 * np.mean(density) + 1e-10)
    
    return frequencies, df_dx


# ══════════════════════════════════════════════════════════════════════════════
# JAX FEM SOLVER (Real Physics!)
# ══════════════════════════════════════════════════════════════════════════════

def create_jax_fem_solver(
    length: float = 2.0,
    width: float = 0.6,
    thickness: float = 0.015,
    E: float = 12e9,
    rho: float = 450.0,
    n_modes: int = 10
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Create a JAX-based FEM solver for real topology optimization.
    
    Uses automatic differentiation for exact sensitivity computation!
    
    Args:
        length, width, thickness: Plate dimensions [m]
        E: Young's modulus [Pa]
        rho: Density [kg/m³]
        n_modes: Number of modes to compute
    
    Returns:
        Function(density) -> (frequencies, df_dx)
    
    Example:
        >>> solver = create_jax_fem_solver(length=2.0, width=0.6)
        >>> freqs, sensitivity = solver(density_field)
    """
    try:
        from .jax_plate_fem import JAXPlateFEM
        
        # Create FEM instance
        fem = JAXPlateFEM(
            length=length,
            width=width,
            thickness=thickness,
            E=E,
            rho=rho,
            nx=int(20 * length),
            ny=int(20 * width)
        )
        
        def solver(density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """JAX FEM solver."""
            # Resize if needed
            nx, ny = density.shape
            fem.config.nx = nx
            fem.config.ny = ny
            
            # Compute frequencies
            frequencies = fem.compute_frequencies(density, n_modes)
            
            # Compute sensitivities via JAX autodiff
            df_dx = np.zeros((len(frequencies), nx, ny))
            for i in range(len(frequencies)):
                df_dx[i] = fem.compute_sensitivity(density, i)
            
            return frequencies, df_dx
        
        print("✓ JAX FEM solver created (differentiable!)")
        return solver
        
    except ImportError as e:
        print(f"⚠️ JAX FEM non disponibile: {e}")
        print("  Usando solver semplificato (Rayleigh-Ritz)")
        
        # Return simplified solver
        def fallback_solver(density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return simple_plate_fem(
                density, length, width, thickness, E, rho, n_modes
            )
        
        return fallback_solver


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimization_result(
    result: OptimizationResult,
    zones: List['BodyZone'],
    save_path: Optional[str] = None
):
    """Plot optimization results."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Density distribution
        ax1 = axes[0, 0]
        im = ax1.imshow(result.density.T, origin='lower', cmap='bone_r', 
                       aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax1, label='Density')
        
        # Overlay zones
        nx, ny = result.density.shape
        for zone in zones:
            rect = Rectangle(
                (zone.position_start * nx, (0.5 - zone.lateral_extent) * ny),
                (zone.position_end - zone.position_start) * nx,
                2 * zone.lateral_extent * ny,
                fill=False, edgecolor=zone.color, linewidth=2, linestyle='--'
            )
            ax1.add_patch(rect)
            ax1.text(
                (zone.position_start + zone.position_end) / 2 * nx,
                (0.5 + zone.lateral_extent) * ny + 1,
                zone.name[:10],
                ha='center', va='bottom', fontsize=8, color=zone.color
            )
        
        ax1.set_title('Optimized Density Distribution')
        ax1.set_xlabel('Length direction')
        ax1.set_ylabel('Width direction')
        
        # Convergence history
        ax2 = axes[0, 1]
        ax2.semilogy(result.objective_history, 'b-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective')
        ax2.set_title('Convergence History')
        ax2.grid(True, alpha=0.3)
        
        # Frequencies achieved vs targets
        ax3 = axes[1, 0]
        targets = []
        for zone in zones:
            targets.extend(zone.target_frequencies)
        targets = sorted(set(targets))
        
        achieved = result.frequencies[:len(targets)]
        
        x_pos = np.arange(len(targets))
        width = 0.35
        
        ax3.bar(x_pos - width/2, targets, width, label='Target', color='blue', alpha=0.7)
        ax3.bar(x_pos + width/2, achieved[:len(targets)], width, label='Achieved', color='green', alpha=0.7)
        ax3.set_xlabel('Mode')
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_title('Target vs Achieved Frequencies')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Volume history
        ax4 = axes[1, 1]
        vol_history = [np.mean(result.density)] * len(result.constraint_history)  # Simplified
        ax4.plot(result.constraint_history, 'r-', linewidth=2)
        ax4.axhline(0, color='g', linestyle='--', label='Target')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Volume Constraint Violation')
        ax4.set_title('Volume Constraint History')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from body_zones import create_chakra_zones, create_vat_therapy_zones
    
    print("═" * 60)
    print(" ITERATIVE ZONE OPTIMIZER TEST")
    print("═" * 60)
    
    # Create zones
    zones = create_vat_therapy_zones()
    
    print("\nZones configured:")
    for zone in zones:
        print(f"  {zone.name}: {zone.target_frequencies} Hz")
    
    # Create optimizer
    config = OptimizationConfig(
        nx=40,
        ny=24,
        interpolation=InterpolationScheme.SIMP,
        penalty=3.0,
        volume_fraction=0.5,
        max_iterations=50,
        convergence_tol=1e-3,
        use_continuation=True,
        penalty_start=1.0
    )
    
    optimizer = ZoneIterativeOptimizer(
        config=config,
        zones=zones,
        objective=MSEObjective(),
        optimizer=OC_Optimizer()
    )
    
    # Initialize with zone-weighted density
    optimizer.zone_weighted_density()
    
    print(f"\nInitial density: mean={np.mean(optimizer.x):.2%}")
    print(f"Running optimization (max {config.max_iterations} iterations)...\n")
    
    # Run optimization
    result = optimizer.optimize(
        fem_solver=simple_plate_fem,
        callback=None
    )
    
    print(f"\n{'═' * 60}")
    print(f" RESULTS")
    print(f"{'═' * 60}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final penalty: {result.final_penalty}")
    print(f"Final objective: {result.objective_history[-1]:.4f}")
    print(f"Frequencies: {result.frequencies[:5]} Hz")
    
    # Plot
    fig = plot_optimization_result(result, zones, '/tmp/zone_optimization_result.png')
