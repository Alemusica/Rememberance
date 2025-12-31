"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MATERIALS - Single Source of Truth                        ║
║                                                                              ║
║   Unified material definitions for all plate-related modules.               ║
║   Import from here instead of plate_fem.py or plate_physics.py              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Material:
    """
    Material properties for plate modal analysis.
    
    Immutable (frozen) to prevent accidental modification.
    All values in SI units.
    
    For orthotropic materials (wood):
    - L = Longitudinal (along grain)
    - R = Radial (across rings)
    - T = Tangential (tangent to rings)
    
    References:
    - Forest Products Laboratory, "Wood Handbook" (FPL-GTR-190)
    - Bucur, "Acoustics of Wood" (Springer, 2006)
    - Ross, "Wood Handbook: Wood as an Engineering Material" (USDA, 2010)
    """
    name: str
    density: float              # kg/m³
    E_longitudinal: float       # Pa (Young's modulus along grain/length) - E_L
    E_transverse: float         # Pa (Young's modulus across grain/width) - E_R or E_T
    poisson_ratio: float        # dimensionless (ν_LR typically 0.25-0.40)
    damping_ratio: float        # dimensionless (typical 0.001-0.05)
    description: str = ""
    
    # === EXTENDED ORTHOTROPIC PROPERTIES (optional, for accurate FEM) ===
    E_radial: float = None      # Pa - E_R (radial direction)
    E_tangential: float = None  # Pa - E_T (tangential direction)
    G_LR: float = None          # Pa - Shear modulus L-R plane
    G_LT: float = None          # Pa - Shear modulus L-T plane
    G_RT: float = None          # Pa - Shear modulus R-T plane
    nu_LR: float = None         # Poisson ratio L-R
    nu_LT: float = None         # Poisson ratio L-T
    nu_RT: float = None         # Poisson ratio R-T
    
    # Acoustic properties
    sound_velocity_L: float = None  # m/s - Speed of sound along grain
    radiation_ratio: float = None   # R = c_L / ρ (higher = better radiator)
    
    @property
    def E_mean(self) -> float:
        """Average Young's modulus for isotropic approximation."""
        return (self.E_longitudinal + self.E_transverse) / 2
    
    @property
    def is_isotropic(self) -> bool:
        """Check if material is approximately isotropic."""
        ratio = self.E_longitudinal / self.E_transverse
        return 0.9 <= ratio <= 1.1
    
    @property
    def Q_factor(self) -> float:
        """Quality factor (inverse of 2x damping ratio)."""
        return 1.0 / (2.0 * self.damping_ratio)
    
    @property
    def orthotropy_ratio(self) -> float:
        """E_L / E_T ratio. Higher = more orthotropic. Wood typically 10-20."""
        return self.E_longitudinal / self.E_transverse
    
    @property
    def acoustic_constant(self) -> float:
        """
        Acoustic constant K = √(E_L/ρ³). 
        Higher = better sound radiation. 
        Reference: Bucur (2006) "Acoustics of Wood"
        """
        import math
        return math.sqrt(self.E_longitudinal / (self.density ** 3))
    
    def get_stiffness_matrix_2d(self) -> 'np.ndarray':
        """
        Get 2D plane stress stiffness matrix [D] for FEM.
        
        For orthotropic plate in x-y plane:
        D = [D11  D12   0  ]
            [D12  D22   0  ]
            [ 0    0   D66]
        
        Returns:
            3x3 numpy array
        """
        import numpy as np
        
        E_L = self.E_longitudinal
        E_T = self.E_transverse if self.E_transverse else E_L
        nu = self.poisson_ratio
        
        # Use extended properties if available
        if self.G_LT:
            G = self.G_LT
        else:
            G = E_L / (2 * (1 + nu))  # Isotropic approximation
        
        # Compliance matrix coefficients
        denom = 1 - nu * nu * E_T / E_L
        
        D11 = E_L / denom
        D22 = E_T / denom
        D12 = nu * E_T / denom
        D66 = G
        
        return np.array([
            [D11, D12, 0],
            [D12, D22, 0],
            [0,   0,  D66]
        ])


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

class MaterialCategory(Enum):
    """Material categories for filtering."""
    SOFTWOOD = "softwood"
    HARDWOOD = "hardwood"
    PLYWOOD = "plywood"
    COMPOSITE = "composite"
    METAL = "metal"


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL DATABASE - SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════════════════

MATERIALS: Dict[str, Material] = {
    # ══════════════════════════════════════════════════════════════════════════
    # TONEWOODS - Scientific data from:
    # - Forest Products Laboratory "Wood Handbook" FPL-GTR-190 (2010)
    # - Bucur "Acoustics of Wood" Springer (2006)
    # - Wegst "Wood for Sound" Am. J. Botany 93(10):1439-1448 (2006)
    # ══════════════════════════════════════════════════════════════════════════
    
    # === SPRUCE (Picea) - The gold standard for soundboards ===
    "spruce_sitka": Material(
        name="Sitka Spruce (Picea sitchensis)",
        density=425.0,              # kg/m³ (air dry, 12% MC)
        E_longitudinal=11.9e9,      # 11.9 GPa (USDA Wood Handbook)
        E_transverse=0.62e9,        # 0.62 GPa (E_R)
        poisson_ratio=0.372,        # ν_LR
        damping_ratio=0.008,        # Low damping = excellent sustain
        E_radial=0.62e9,            # E_R
        E_tangential=0.46e9,        # E_T  
        G_LR=0.75e9,                # Shear modulus L-R
        G_LT=0.72e9,                # Shear modulus L-T
        G_RT=0.037e9,               # Shear modulus R-T (very low!)
        nu_LR=0.372,                # Poisson ratio
        nu_LT=0.467,
        nu_RT=0.435,
        sound_velocity_L=5300.0,    # m/s along grain
        radiation_ratio=12.5,       # Excellent radiator
        description="Premium soundboard wood. Sitka has highest radiation ratio. "
                   "Used in guitar tops, piano soundboards. E_L/E_T ≈ 19."
    ),
    "spruce_engelmann": Material(
        name="Engelmann Spruce (Picea engelmannii)",
        density=385.0,              # Lighter than Sitka
        E_longitudinal=10.3e9,
        E_transverse=0.54e9,
        poisson_ratio=0.39,
        damping_ratio=0.007,        # Even lower damping
        E_radial=0.54e9,
        E_tangential=0.40e9,
        G_LR=0.65e9,
        G_LT=0.62e9,
        G_RT=0.032e9,
        nu_LR=0.39,
        nu_LT=0.49,
        nu_RT=0.44,
        sound_velocity_L=5170.0,
        radiation_ratio=13.4,       # Highest radiation ratio!
        description="Premium classical guitar tops. Lighter, warmer than Sitka. "
                   "Preferred by luthiers for fingerstyle."
    ),
    "spruce_european": Material(
        name="European Spruce (Picea abies)",
        density=450.0,
        E_longitudinal=12.3e9,      # Slightly stiffer than Sitka
        E_transverse=0.65e9,
        poisson_ratio=0.35,
        damping_ratio=0.009,
        E_radial=0.65e9,
        E_tangential=0.48e9,
        G_LR=0.80e9,
        G_LT=0.76e9,
        G_RT=0.040e9,
        nu_LR=0.35,
        nu_LT=0.45,
        nu_RT=0.42,
        sound_velocity_L=5230.0,
        radiation_ratio=11.6,
        description="Traditional violin top wood (German/Alpine spruce). "
                   "Stradivari used this! Tight grain, excellent Q factor."
    ),
    # Backward compat alias
    "spruce": Material(
        name="Spruce (Sitka)",
        density=425.0,
        E_longitudinal=11.9e9,
        E_transverse=0.62e9,
        poisson_ratio=0.37,
        damping_ratio=0.008,
        description="Alias for spruce_sitka - traditional soundboard wood."
    ),
    
    # === CEDAR ===
    "cedar": Material(
        name="Western Red Cedar",
        density=380.0,
        E_longitudinal=9.0e9,
        E_transverse=0.6e9,
        poisson_ratio=0.35,
        damping_ratio=0.012,
        E_radial=0.6e9,
        E_tangential=0.45e9,
        G_LR=0.55e9,
        G_LT=0.52e9,
        G_RT=0.030e9,
        nu_LR=0.35,
        sound_velocity_L=4870.0,
        radiation_ratio=12.8,
        description="Lighter than spruce, warmer/darker tone. Classical guitar tops."
    ),
    
    # === HARDWOODS ===
    "oak": Material(
        name="White Oak",
        density=700.0,
        E_longitudinal=12.0e9,
        E_transverse=1.0e9,
        poisson_ratio=0.35,
        damping_ratio=0.015,
        description="Dense hardwood, warm tone, good sustain."
    ),
    "maple": Material(
        name="Hard Maple",
        density=650.0,
        E_longitudinal=13.0e9,
        E_transverse=1.1e9,
        poisson_ratio=0.35,
        damping_ratio=0.012,
        description="Bright, articulate. Guitar backs, drum shells."
    ),
    "walnut": Material(
        name="Black Walnut",
        density=600.0,
        E_longitudinal=11.5e9,
        E_transverse=0.9e9,
        poisson_ratio=0.35,
        damping_ratio=0.018,
        description="Balanced tone, beautiful grain. Guitar bodies."
    ),
    "ash": Material(
        name="White Ash",
        density=670.0,
        E_longitudinal=12.0e9,
        E_transverse=1.0e9,
        poisson_ratio=0.33,
        damping_ratio=0.014,
        description="Punchy, clear midrange. Electric guitar bodies."
    ),
    
    # === PLYWOOD (quasi-isotropic) ===
    "birch_plywood": Material(
        name="Baltic Birch Plywood",
        density=680.0,
        E_longitudinal=13.0e9,
        E_transverse=13.0e9,        # Plywood is quasi-isotropic
        poisson_ratio=0.33,
        damping_ratio=0.025,
        description="Strong, stable, uniform. Best for vibroacoustic tables."
    ),
    "marine_plywood": Material(
        name="Marine Plywood",
        density=700.0,
        E_longitudinal=12.5e9,
        E_transverse=12.5e9,
        poisson_ratio=0.30,
        damping_ratio=0.02,
        description="Water resistant, stable, good damping."
    ),
    "bamboo_plywood": Material(
        name="Bamboo Plywood",
        density=700.0,
        E_longitudinal=14.0e9,
        E_transverse=10.0e9,
        poisson_ratio=0.30,
        damping_ratio=0.018,
        description="Sustainable, stiff, bright tone."
    ),
    
    # === COMPOSITES ===
    "mdf": Material(
        name="MDF (Medium Density Fiberboard)",
        density=750.0,
        E_longitudinal=4.0e9,
        E_transverse=4.0e9,         # Perfectly isotropic
        poisson_ratio=0.25,
        damping_ratio=0.04,         # Higher damping
        description="Uniform, affordable, higher damping. Speaker cabinets."
    ),
    "hdf": Material(
        name="HDF (High Density Fiberboard)",
        density=850.0,
        E_longitudinal=5.0e9,
        E_transverse=5.0e9,
        poisson_ratio=0.25,
        damping_ratio=0.035,
        description="Denser than MDF, more rigid."
    ),
    "carbon_fiber": Material(
        name="Carbon Fiber Composite",
        density=1600.0,
        E_longitudinal=70.0e9,      # Along fibers
        E_transverse=7.0e9,         # Across fibers
        poisson_ratio=0.30,
        damping_ratio=0.005,
        description="Extremely stiff, lightweight. High-end instruments."
    ),
    
    # === METALS ===
    "aluminum": Material(
        name="Aluminum 6061-T6",
        density=2700.0,
        E_longitudinal=69.0e9,
        E_transverse=69.0e9,        # Isotropic
        poisson_ratio=0.33,
        damping_ratio=0.002,        # Very low - rings long
        description="Lightweight metal, long sustain, bright. Bell-like tone."
    ),
    "steel": Material(
        name="Mild Steel",
        density=7850.0,
        E_longitudinal=200.0e9,
        E_transverse=200.0e9,
        poisson_ratio=0.30,
        damping_ratio=0.001,
        description="Very stiff, long sustain. Steel drums, bells."
    ),
    "brass": Material(
        name="Brass (70/30)",
        density=8500.0,
        E_longitudinal=110.0e9,
        E_transverse=110.0e9,
        poisson_ratio=0.34,
        damping_ratio=0.003,
        description="Warm, rich overtones. Cymbals, bells."
    ),
    "titanium": Material(
        name="Titanium Grade 5",
        density=4430.0,
        E_longitudinal=114.0e9,
        E_transverse=114.0e9,
        poisson_ratio=0.34,
        damping_ratio=0.001,
        description="Strong, lightweight, excellent sustain."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL CATEGORY MAPPING
# ══════════════════════════════════════════════════════════════════════════════

MATERIAL_CATEGORIES: Dict[str, MaterialCategory] = {
    "spruce": MaterialCategory.SOFTWOOD,
    "cedar": MaterialCategory.SOFTWOOD,
    "oak": MaterialCategory.HARDWOOD,
    "maple": MaterialCategory.HARDWOOD,
    "walnut": MaterialCategory.HARDWOOD,
    "ash": MaterialCategory.HARDWOOD,
    "birch_plywood": MaterialCategory.PLYWOOD,
    "marine_plywood": MaterialCategory.PLYWOOD,
    "bamboo_plywood": MaterialCategory.PLYWOOD,
    "mdf": MaterialCategory.COMPOSITE,
    "hdf": MaterialCategory.COMPOSITE,
    "carbon_fiber": MaterialCategory.COMPOSITE,
    "aluminum": MaterialCategory.METAL,
    "steel": MaterialCategory.METAL,
    "brass": MaterialCategory.METAL,
    "titanium": MaterialCategory.METAL,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_material(name: str) -> Material:
    """Get material by name. Raises KeyError if not found."""
    if name not in MATERIALS:
        available = ", ".join(sorted(MATERIALS.keys()))
        raise KeyError(f"Unknown material '{name}'. Available: {available}")
    return MATERIALS[name]


def get_materials_by_category(category: MaterialCategory) -> Dict[str, Material]:
    """Get all materials in a category."""
    return {
        name: MATERIALS[name]
        for name, cat in MATERIAL_CATEGORIES.items()
        if cat == category
    }


def get_best_for_vibroacoustic() -> list[str]:
    """
    Get materials recommended for vibroacoustic tables.
    
    Criteria:
    - Low damping (good sustain)
    - Moderate density (comfortable weight)
    - Good stiffness-to-weight ratio
    """
    return [
        "birch_plywood",    # Best overall: stable, uniform, affordable
        "marine_plywood",   # Good alternative: water resistant
        "spruce",           # Best acoustics but harder to work with
        "maple",            # Premium option: bright, articulate
        "bamboo_plywood",   # Sustainable option
    ]


def list_materials() -> None:
    """Print formatted list of all materials."""
    print("\n" + "=" * 70)
    print("AVAILABLE MATERIALS")
    print("=" * 70)
    
    for category in MaterialCategory:
        mats = get_materials_by_category(category)
        if mats:
            print(f"\n{category.value.upper()}")
            print("-" * 40)
            for name, mat in mats.items():
                print(f"  {name:20s} ρ={mat.density:5.0f} kg/m³  "
                      f"E={mat.E_mean/1e9:5.1f} GPa  ζ={mat.damping_ratio:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY - Aliases for migration
# ══════════════════════════════════════════════════════════════════════════════

# These allow gradual migration from plate_fem.py and plate_physics.py
MaterialData = Material  # Alias if someone used different name

# Default material for new plates
DEFAULT_MATERIAL = "birch_plywood"


if __name__ == "__main__":
    list_materials()
    print(f"\nRecommended for vibroacoustic: {get_best_for_vibroacoustic()}")
