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

try:
    from .dsp_export import (
        DSPExportResult, ExciterData, ModalResonance, TransferFunctionData,
        MaterialData, ZoneData, PlateGeometryData, export_for_dsp
    )
except ImportError:
    pass

try:
    from .stl_export import PlateSTLExporter, export_plate_for_cnc
except ImportError:
    pass

try:
    from .dml_frequency_model import (
        DMLFrequencyModel, DMLResponse, ModeShape, ExciterCoupling,
        ExciterStrategy, create_dml_model_for_genome, analyze_exciter_placement
    )
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS CONFIG - Centralized parameters (Action Plan 3.0 Phase 0)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from .analysis_config import (
        AnalysisConfig, ModalAnalysisConfig, GeneActivationConfig,
        EvolutionConfig, ObserverConfig, EmissionBounds,
        GenePhase, ActivationTrigger,
        get_default_config, set_default_config, get_target_spacing_mm,
        ConfigProvider
    )
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# EXCITER GENE - Staged activation (Action Plan 3.0 Phase 2)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from .exciter_gene import (
        ExciterGene, EmissionGenes,
        upgrade_exciters_to_genes, activate_all_emission, freeze_all_positions,
        calculate_position_sigma, get_emission_summary
    )
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS RULES ENGINE - Hybrid rules (Action Plan 3.0 Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from .physics_rules import (
        PhysicsRulesEngine, PhysicsRule, LearnedRule,
        RuleCategory, RuleDomain, RuleCondition, RuleSuggestion, RuleEvaluationResult,
        ExciterAtAntinodeRule, ExciterAvoidNodeRule, PhaseSteeringRule,
        EdgeDistanceRule, CutoutAntinodeTuningRule,
        create_rule_context, create_physics_engine
    )
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# POKAYOKE OBSERVER - Intelligent monitoring with PAUSE + ASK USER
# ══════════════════════════════════════════════════════════════════════════════
try:
    from .pokayoke_observer import (
        PokayokeObserver, AnomalyType, UserAction, AnomalyContext, UserDecision,
        ObserverState, UserInteractionHandler, HeadlessHandler, CLIHandler,
        create_observer
    )
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# PYMOO MULTI-OBJECTIVE OPTIMIZER (NSGA-II/NSGA-III)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from .pymoo_optimizer import (
        PymooOptimizer, PymooConfig, PymooResult,
        PlateOptimizationProblem, compute_ear_uniformity_from_result,
        PYMOO_AVAILABLE
    )
except ImportError:
    PYMOO_AVAILABLE = False

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
    # DSP Export
    'DSPExportResult', 'ExciterData', 'ModalResonance', 'TransferFunctionData',
    'MaterialData', 'ZoneData', 'PlateGeometryData', 'export_for_dsp',
    # CNC Export (Issue #7)
    'PlateSTLExporter', 'export_plate_for_cnc',
    # DML Frequency Model (from Harris 2010, Aures 2001, Bank 2010)
    'DMLFrequencyModel', 'DMLResponse', 'ModeShape', 'ExciterCoupling',
    'ExciterStrategy', 'create_dml_model_for_genome', 'analyze_exciter_placement',
    # Analysis Config (Action Plan 3.0 - centralized parameters)
    'AnalysisConfig', 'ModalAnalysisConfig', 'GeneActivationConfig',
    'EvolutionConfig', 'ObserverConfig', 'EmissionBounds',
    'GenePhase', 'ActivationTrigger',
    'get_default_config', 'set_default_config', 'get_target_spacing_mm',
    'ConfigProvider',
    # ExciterGene (Action Plan 3.0 Phase 2 - staged gene activation)
    'ExciterGene', 'EmissionGenes',
    'upgrade_exciters_to_genes', 'activate_all_emission', 'freeze_all_positions',
    'calculate_position_sigma', 'get_emission_summary',
    # Physics Rules Engine (Action Plan 3.0 Phase 3 - hybrid rules)
    'PhysicsRulesEngine', 'PhysicsRule', 'LearnedRule',
    'RuleCategory', 'RuleDomain', 'RuleCondition', 'RuleSuggestion', 'RuleEvaluationResult',
    'create_rule_context', 'create_physics_engine',
    # PokayokeObserver (Action Plan 3.0 Phase 1 - PAUSE + ASK USER)
    'PokayokeObserver', 'AnomalyType', 'UserAction', 'AnomalyContext', 'UserDecision',
    'ObserverState', 'UserInteractionHandler', 'HeadlessHandler', 'CLIHandler',
    'create_observer',
    # Pymoo Multi-Objective Optimizer (NSGA-II/III for ear L/R balance)
    'PymooOptimizer', 'PymooConfig', 'PymooResult', 'PlateOptimizationProblem',
    'compute_ear_uniformity_from_result', 'PYMOO_AVAILABLE',
]
