"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PLATE PHYSICS - Modal Frequency Calculator               ║
║                                                                              ║
║   Calculate resonant frequencies and mode shapes for rectangular plates      ║
║   Used for vibroacoustic table design and exciter optimization               ║
║                                                                              ║
║   Physics: Kirchhoff-Love plate theory for thin orthotropic plates           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Import unified material definitions
from .materials import Material, MATERIALS

# ══════════════════════════════════════════════════════════════════════════════
# RE-EXPORT FOR BACKWARD COMPATIBILITY
# Other modules can still do: from .plate_physics import Material, MATERIALS
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# MODE SHAPE DATA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlateMode:
    """Represents a single vibrational mode of a plate"""
    m: int                      # Mode number in length direction (1, 2, 3...)
    n: int                      # Mode number in width direction (1, 2, 3...)
    frequency: float            # Hz
    wavelength_x: float         # m (along length)
    wavelength_y: float         # m (along width)
    
    @property
    def mode_name(self) -> str:
        return f"({self.m},{self.n})"
    
    @property
    def description(self) -> str:
        """Human-readable mode description"""
        if self.m == 1 and self.n == 1:
            return "Fundamental - whole plate flexes"
        elif self.m == 2 and self.n == 1:
            return "1 nodal line across length"
        elif self.m == 1 and self.n == 2:
            return "1 nodal line across width"
        elif self.m == 2 and self.n == 2:
            return "Cross pattern - 2 nodal lines"
        else:
            nodes_x = self.m - 1
            nodes_y = self.n - 1
            return f"{nodes_x}×{nodes_y} nodal grid"


# ══════════════════════════════════════════════════════════════════════════════
# MODAL FREQUENCY CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_flexural_rigidity(E: float, h: float, nu: float) -> float:
    """
    Calculate flexural rigidity D for a plate.
    
    D = E * h³ / (12 * (1 - ν²))
    
    Args:
        E: Young's modulus (Pa)
        h: Thickness (m)
        nu: Poisson's ratio
    
    Returns:
        Flexural rigidity D (N·m)
    """
    return (E * h**3) / (12 * (1 - nu**2))


def calculate_modal_frequency_isotropic(
    m: int, n: int,
    L: float, W: float, h: float,
    E: float, rho: float, nu: float
) -> float:
    """
    Calculate modal frequency for isotropic rectangular plate (free-free).
    
    Uses the classical plate equation for simply-supported boundaries,
    adjusted for free-free with empirical correction.
    
    f_mn = (π/2) * √(D/(ρh)) * [(m/L)² + (n/W)²]
    
    For free-free boundaries, multiply by ~0.5 for fundamental mode.
    
    Args:
        m, n: Mode numbers (1, 2, 3...)
        L: Length (m)
        W: Width (m)
        h: Thickness (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m³)
        nu: Poisson's ratio
    
    Returns:
        Frequency in Hz
    """
    D = calculate_flexural_rigidity(E, h, nu)
    
    # Free-free eigenvalues (approximation using beam mode shapes)
    # λ_m for free-free beam: 4.73, 7.85, 11.0, 14.14, ...
    # For m=1,2,3,4: use (m + 0.5)π approximation
    lambda_m = (m + 0.5) * np.pi
    lambda_n = (n + 0.5) * np.pi
    
    # Frequency formula for free-free plate
    f = (1 / (2 * np.pi)) * np.sqrt(D / (rho * h)) * (
        (lambda_m / L)**2 + (lambda_n / W)**2
    )
    
    return f


def calculate_modal_frequency_orthotropic(
    m: int, n: int,
    L: float, W: float, h: float,
    E_L: float, E_W: float, rho: float, nu: float
) -> float:
    """
    Calculate modal frequency for orthotropic rectangular plate (like wood).
    
    Wood has different stiffness along grain (L) vs across grain (W).
    
    Args:
        m, n: Mode numbers
        L: Length along grain (m)
        W: Width across grain (m)
        h: Thickness (m)
        E_L: Young's modulus along length/grain (Pa)
        E_W: Young's modulus along width (Pa)
        rho: Density (kg/m³)
        nu: Poisson's ratio
    
    Returns:
        Frequency in Hz
    """
    D_L = calculate_flexural_rigidity(E_L, h, nu)
    D_W = calculate_flexural_rigidity(E_W, h, nu)
    
    # Geometric mean for cross-coupling
    D_LW = np.sqrt(D_L * D_W) * nu
    
    # Free-free eigenvalues
    lambda_m = (m + 0.5) * np.pi
    lambda_n = (n + 0.5) * np.pi
    
    # Orthotropic plate frequency (simplified)
    term1 = D_L * (lambda_m / L)**4
    term2 = D_W * (lambda_n / W)**4
    term3 = 2 * D_LW * (lambda_m / L)**2 * (lambda_n / W)**2
    
    f = (1 / (2 * np.pi)) * np.sqrt((term1 + term2 + term3) / (rho * h))
    
    return f


def calculate_plate_modes(
    length_mm: float,
    width_mm: float,
    thickness_mm: float,
    material_key: str,
    max_modes: int = 10,
    max_freq_hz: float = 500.0
) -> List[PlateMode]:
    """
    Calculate all modal frequencies for a plate up to a maximum frequency.
    
    Args:
        length_mm: Plate length in mm
        width_mm: Plate width in mm
        thickness_mm: Plate thickness in mm
        material_key: Key into MATERIALS dict
        max_modes: Maximum number of modes to return
        max_freq_hz: Maximum frequency to consider
    
    Returns:
        List of PlateMode objects, sorted by frequency
    """
    # Convert to meters
    L = length_mm / 1000.0
    W = width_mm / 1000.0
    h = thickness_mm / 1000.0
    
    # Get material properties
    mat = MATERIALS.get(material_key, MATERIALS["spruce"])
    
    modes = []
    
    # Search through mode combinations
    for m in range(1, 8):
        for n in range(1, 8):
            # Use orthotropic formula if material is anisotropic
            if abs(mat.E_longitudinal - mat.E_transverse) / mat.E_longitudinal > 0.1:
                freq = calculate_modal_frequency_orthotropic(
                    m, n, L, W, h,
                    mat.E_longitudinal, mat.E_transverse,
                    mat.density, mat.poisson_ratio
                )
            else:
                # Use simpler isotropic formula
                E_avg = (mat.E_longitudinal + mat.E_transverse) / 2
                freq = calculate_modal_frequency_isotropic(
                    m, n, L, W, h,
                    E_avg, mat.density, mat.poisson_ratio
                )
            
            if freq <= max_freq_hz:
                mode = PlateMode(
                    m=m, n=n,
                    frequency=freq,
                    wavelength_x=2 * L / m,
                    wavelength_y=2 * W / n
                )
                modes.append(mode)
    
    # Sort by frequency and limit
    modes.sort(key=lambda x: x.frequency)
    return modes[:max_modes]


# ══════════════════════════════════════════════════════════════════════════════
# MODE SHAPE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def mode_shape(
    x: float, y: float,
    m: int, n: int,
    L: float, W: float
) -> float:
    """
    Calculate the normalized mode shape amplitude at position (x, y).
    
    For free-free plate, mode shape is approximately:
    Z(x,y) = cos(m*π*x/L) * cos(n*π*y/W)
    
    With corrections at edges for free boundary conditions.
    
    Args:
        x: Position along length [0, L]
        y: Position along width [0, W]
        m, n: Mode numbers
        L, W: Plate dimensions
    
    Returns:
        Normalized amplitude [-1, 1] where:
        - 0 = nodal line (no motion)
        - ±1 = antinode (maximum motion)
    """
    # Free-free mode shapes use cos functions
    # (fixed-fixed would use sin)
    shape_x = np.cos(m * np.pi * x / L)
    shape_y = np.cos(n * np.pi * y / W)
    
    return shape_x * shape_y


def mode_shape_grid(
    m: int, n: int,
    L: float, W: float,
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D grid of mode shape amplitudes for visualization.
    
    Args:
        m, n: Mode numbers
        L, W: Plate dimensions (any units, just for aspect ratio)
        resolution: Grid points per dimension
    
    Returns:
        (X, Y, Z) meshgrid arrays where Z is normalized amplitude [-1, 1]
    """
    x = np.linspace(0, L, resolution)
    y = np.linspace(0, W, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = mode_shape(X, Y, m, n, L, W)
    
    return X, Y, Z


# ══════════════════════════════════════════════════════════════════════════════
# EXCITER COUPLING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_exciter_coupling(
    exciter_x: float, exciter_y: float,
    m: int, n: int,
    L: float, W: float
) -> float:
    """
    Calculate how well an exciter at (x, y) couples to mode (m, n).
    
    Coupling is the absolute mode shape amplitude at exciter position.
    - 0.0 = exciter at nodal line, no coupling
    - 1.0 = exciter at antinode, maximum coupling
    
    Args:
        exciter_x, exciter_y: Exciter position
        m, n: Mode numbers
        L, W: Plate dimensions
    
    Returns:
        Coupling coefficient [0, 1]
    """
    return abs(mode_shape(exciter_x, exciter_y, m, n, L, W))


def analyze_exciter_positions(
    exciter_positions: List[Tuple[float, float]],
    modes: List[PlateMode],
    L: float, W: float
) -> Dict[str, List[float]]:
    """
    Analyze how well a set of exciters couple to each mode.
    
    Args:
        exciter_positions: List of (x, y) positions
        modes: List of PlateMode objects
        L, W: Plate dimensions
    
    Returns:
        Dict mapping mode name to list of coupling coefficients per exciter
    """
    result = {}
    
    for mode in modes:
        couplings = []
        for (ex, ey) in exciter_positions:
            coupling = calculate_exciter_coupling(ex, ey, mode.m, mode.n, L, W)
            couplings.append(coupling)
        result[mode.mode_name] = couplings
    
    return result


def optimal_phase_for_mode(
    exciter_positions: List[Tuple[float, float]],
    m: int, n: int,
    L: float, W: float
) -> List[float]:
    """
    Calculate optimal phase angles for each exciter to maximally excite a mode.
    
    If mode shape is positive at exciter → phase = 0°
    If mode shape is negative at exciter → phase = 180°
    
    Args:
        exciter_positions: List of (x, y) positions
        m, n: Target mode numbers
        L, W: Plate dimensions
    
    Returns:
        List of phase angles in degrees [0 or 180]
    """
    phases = []
    for (ex, ey) in exciter_positions:
        shape_val = mode_shape(ex, ey, m, n, L, W)
        if shape_val >= 0:
            phases.append(0.0)
        else:
            phases.append(180.0)
    return phases


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_material_list() -> List[str]:
    """Return list of available material keys."""
    return list(MATERIALS.keys())


def get_material_info(key: str) -> Optional[Material]:
    """Get material properties by key."""
    return MATERIALS.get(key)


def estimate_decay_time(freq: float, damping_ratio: float) -> float:
    """
    Estimate the decay time (T60) for a mode.
    
    T60 ≈ 6.9 / (damping_ratio * 2π * freq)
    
    Args:
        freq: Modal frequency in Hz
        damping_ratio: Material damping ratio
    
    Returns:
        Decay time in seconds (time to decay 60dB)
    """
    if freq <= 0 or damping_ratio <= 0:
        return float('inf')
    return 6.9 / (damping_ratio * 2 * np.pi * freq)


# ══════════════════════════════════════════════════════════════════════════════
# TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PLATE MODAL ANALYSIS - Demo")
    print("=" * 60)
    
    # Test with default soundboard dimensions
    L_mm = 1950.0
    W_mm = 600.0
    T_mm = 10.0
    
    print(f"\nPlate: {L_mm} × {W_mm} × {T_mm} mm Spruce")
    print("-" * 40)
    
    modes = calculate_plate_modes(L_mm, W_mm, T_mm, "spruce", max_modes=10)
    
    print(f"\n{'Mode':<10} {'Freq (Hz)':<12} {'Description'}")
    print("-" * 50)
    for mode in modes:
        print(f"{mode.mode_name:<10} {mode.frequency:>8.1f} Hz   {mode.description}")
    
    # Test exciter coupling
    print("\n" + "=" * 60)
    print("EXCITER COUPLING ANALYSIS")
    print("=" * 60)
    
    # 2-exciter setup: HEAD (0mm) and FEET (1950mm)
    exciters = [(0, W_mm/2), (L_mm, W_mm/2)]  # Center of each end
    
    print(f"\nExciter positions: HEAD={exciters[0]}, FEET={exciters[1]}")
    print("-" * 40)
    
    couplings = analyze_exciter_positions(
        [(e[0]/1000, e[1]/1000) for e in exciters],
        modes,
        L_mm/1000, W_mm/1000
    )
    
    print(f"\n{'Mode':<10} {'HEAD':<10} {'FEET':<10} {'Optimal Phase'}")
    print("-" * 50)
    for mode in modes[:6]:
        c = couplings[mode.mode_name]
        phases = optimal_phase_for_mode(
            [(e[0]/1000, e[1]/1000) for e in exciters],
            mode.m, mode.n, L_mm/1000, W_mm/1000
        )
        print(f"{mode.mode_name:<10} {c[0]:>6.2f}    {c[1]:>6.2f}    {phases[0]:.0f}° / {phases[1]:.0f}°")
