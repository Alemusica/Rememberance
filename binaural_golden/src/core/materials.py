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
    """
    name: str
    density: float              # kg/m³
    E_longitudinal: float       # Pa (Young's modulus along grain/length)
    E_transverse: float         # Pa (Young's modulus across grain/width)
    poisson_ratio: float        # dimensionless (typically 0.25-0.40)
    damping_ratio: float        # dimensionless (typical 0.001-0.05)
    description: str = ""
    
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
    # === SOFTWOODS (best for soundboards) ===
    "spruce": Material(
        name="Spruce (Sitka)",
        density=450.0,
        E_longitudinal=12.0e9,      # 12 GPa along grain
        E_transverse=0.8e9,         # 0.8 GPa across grain (highly orthotropic)
        poisson_ratio=0.37,
        damping_ratio=0.01,         # Very low - excellent resonance
        description="Traditional soundboard wood, excellent acoustic properties. "
                   "High stiffness-to-weight ratio. Used in guitars, pianos, violins."
    ),
    "cedar": Material(
        name="Western Red Cedar",
        density=380.0,
        E_longitudinal=9.0e9,
        E_transverse=0.6e9,
        poisson_ratio=0.35,
        damping_ratio=0.012,
        description="Lighter than spruce, warmer tone. Classical guitar tops."
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
