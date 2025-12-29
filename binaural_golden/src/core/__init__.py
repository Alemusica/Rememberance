"""
Core module - Audio engine, physics, and sacred geometry
"""
try:
    from .audio_engine import AudioEngine
    from .golden_math import golden_fade, golden_ease
except ImportError:
    pass

try:
    from .sacred_geometry import (
        PHI, WATER_GEOMETRY, ANTAHKARANA,
        WaterMoleculeGeometry, AntahkaranaAxis, HumanBodyGolden
    )
except ImportError:
    pass

try:
    from .exciter import EXCITERS, Exciter, find_exciters_for_frequency
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# PLATE OPTIMIZATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
try:
    from .body_zones import (
        ZoneType, BodyZone, BodyZoneModel,
        create_chakra_zones, create_vat_therapy_zones, create_body_resonance_zones
    )
except ImportError:
    pass

try:
    from .coupled_system import (
        CoupledSystem, CoupledResult, ZoneCoupledSystem
    )
except ImportError:
    pass

try:
    from .iterative_optimizer import (
        ZoneIterativeOptimizer, OptimizationResult,
        simp, ramp, density_filter,
        create_jax_fem_solver, simple_plate_fem
    )
except ImportError:
    pass

try:
    from .jax_plate_fem import JAXPlateFEM, create_jax_fem_solver as create_jax_solver
except ImportError:
    pass

try:
    from .plate_optimizer import zone_optimize_plate
except ImportError:
    pass

__all__ = [
    # Audio
    'AudioEngine', 'golden_fade', 'golden_ease',
    # Sacred Geometry
    'PHI', 'WATER_GEOMETRY', 'ANTAHKARANA',
    'WaterMoleculeGeometry', 'AntahkaranaAxis', 'HumanBodyGolden',
    # Exciters
    'EXCITERS', 'Exciter', 'find_exciters_for_frequency',
    # Zone Model
    'ZoneType', 'BodyZone', 'BodyZoneModel',
    'create_chakra_zones', 'create_vat_therapy_zones', 'create_body_resonance_zones',
    # Coupled System
    'CoupledSystem', 'CoupledResult', 'ZoneCoupledSystem',
    # Optimization
    'ZoneIterativeOptimizer', 'OptimizationResult',
    'simp', 'ramp', 'density_filter',
    'create_jax_fem_solver', 'simple_plate_fem',
    # JAX FEM
    'JAXPlateFEM',
    # High-Level API
    'zone_optimize_plate',
]
