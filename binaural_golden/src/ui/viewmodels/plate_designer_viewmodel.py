"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE DESIGNER VIEWMODEL                                   ║
║                                                                              ║
║   MVVM ViewModel for evolutionary plate optimization.                        ║
║   Separates business logic from UI concerns.                                 ║
║                                                                              ║
║   Features:                                                                   ║
║   • Observable state pattern                                                  ║
║   • Thread-safe evolution management                                          ║
║   • Fitness history tracking                                                  ║
║   • Export capabilities                                                       ║
║   • PHYSICS-DRIVEN EVOLUTION LOGGING                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from enum import Enum
import threading
import queue
import time
import copy
import logging

# Core imports
from core.person import Person, PERSON_PRESETS
from core.plate_genome import PlateGenome, ContourType
from core.fitness import FitnessEvaluator, FitnessResult, ObjectiveWeights, ZoneWeights
from core.evolutionary_optimizer import (
    EvolutionaryOptimizer,
    EvolutionConfig,
    EvolutionState,
    SelectionMethod,
)
# EvolutionPipeline with RDNN, LTM, Pokayoke integration
from core.evolution_pipeline import (
    EvolutionPipeline,
    PipelineConfig,
    PipelineMode,
    PipelineResult,
)

# Pymoo NSGA-II (multi-objective)
try:
    from core.pymoo_optimizer import PymooOptimizer, PymooConfig, PymooResult, PYMOO_AVAILABLE
except ImportError:
    PYMOO_AVAILABLE = False
    PymooOptimizer = None
    PymooConfig = None

# Evolution logging (physics-driven)
from core.evolution_logger import (
    setup_evolution_logging,
    get_evolution_handler,
    get_formatted_logs,
    generate_evolution_summary,
    log_comparison_with_target,
)

# Setup evolution logging
evo_logger = setup_evolution_logging(logging.INFO)


class EvolutionPhase(Enum):
    """Evolution progress phase."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    EVOLVING = "evolving"
    CONVERGED = "converged"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class FitnessSnapshot:
    """
    Snapshot of fitness scores at a generation.
    
    Used for plotting fitness evolution graph.
    """
    generation: int
    total: float
    flatness: float
    spine_coupling: float
    low_mass: float
    edge_support: float = 0.0
    diversity: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for charts."""
        return {
            "generation": self.generation,
            "total": self.total,
            "flatness": self.flatness,
            "spine_coupling": self.spine_coupling,
            "low_mass": self.low_mass,
        }


@dataclass
class PlateDesignerState:
    """
    Observable state container for PlateDesigner UI.
    
    All UI updates should be driven by changes to this state.
    """
    # Person configuration
    person: Optional[Person] = None
    preset_name: str = "average_male"
    
    # Evolution status
    phase: EvolutionPhase = EvolutionPhase.IDLE
    is_running: bool = False
    
    # Current generation
    generation: int = 0
    max_generations: int = 50
    
    # Best results
    best_genome: Optional[PlateGenome] = None
    best_fitness: Optional[FitnessResult] = None
    best_fitness_value: float = 0.0
    
    # History for charts
    fitness_history: List[FitnessSnapshot] = field(default_factory=list)
    
    # Population diversity
    diversity: float = 0.0
    
    # Timing
    start_time: float = 0.0
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0
    
    # Evolution config
    population_size: int = 30
    mutation_rate: float = 0.3
    max_cutouts: int = 4  # Lutherie standard: 4 cutouts (like f-holes, 2 per side)
    max_grooves: int = 0  # Default OFF - expensive computationally
    max_attached_masses: int = 0  # Attached masses for modal tuning (Shen 2016)
    
    # Spring support configuration (physics-based)
    spring_count: int = 5  # Number of spring supports (3-8)
    spring_stiffness_kn_m: float = 10.0  # Default stiffness in kN/m
    spring_damping_ratio: float = 0.10  # ζ = damping ratio (0.02-0.30)
    spring_clearance_mm: float = 70.0  # Min clearance under plate for hardware
    
    # Algorithm selection: GA custom vs pymoo NSGA-II/III
    use_nsga2: bool = False  # False = GA custom, True = pymoo NSGA-II (Pareto front)
    use_pipeline: bool = True  # True = EvolutionPipeline with RDNN/LTM memory (DEFAULT)
    
    # Zone weights for frequency response optimization (spine vs head)
    spine_weight: float = 0.70  # 70% priority on spine flatness
    head_weight: float = 0.30   # 30% priority on head/ear flatness
    
    # Contour type for plate shape
    contour_type: str = "ORGANIC"  # Default: smooth organic curves
    
    # Early stopping control
    force_all_generations: bool = True  # True = run ALL generations (no early stopping)
    
    # Advanced optimization weights (from radar widget)
    energy_weight: float = 0.7   # Vibration intensity
    flatness_weight: float = 0.5  # Frequency response uniformity
    
    # Error handling
    error_message: Optional[str] = None
    
    # Evolution logs (physics-driven debugging)
    evolution_logs: List[str] = field(default_factory=list)  # Recent log messages
    log_count: int = 0  # Total log entries for UI refresh trigger
    
    @property
    def progress_percent(self) -> float:
        """Progress as percentage 0-100."""
        if self.max_generations <= 0:
            return 0.0
        return min(100.0, (self.generation / self.max_generations) * 100)
    
    @property
    def status_text(self) -> str:
        """Human-readable status."""
        if self.phase == EvolutionPhase.IDLE:
            return "Ready to start"
        elif self.phase == EvolutionPhase.INITIALIZING:
            return "Initializing population..."
        elif self.phase == EvolutionPhase.EVOLVING:
            return f"Generation {self.generation}/{self.max_generations}"
        elif self.phase == EvolutionPhase.CONVERGED:
            return "✓ Converged!"
        elif self.phase == EvolutionPhase.STOPPED:
            return "⏹ Stopped"
        elif self.phase == EvolutionPhase.ERROR:
            return f"⚠ Error: {self.error_message}"
        return ""
    
    @property
    def plate_info_text(self) -> str:
        """Plate info for display."""
        if self.best_genome is None:
            return "No plate generated"
        g = self.best_genome
        return (
            f"{g.contour_type.value} {g.length:.2f}×{g.width:.2f}m, "
            f"{g.thickness_base * 1000:.1f}mm"
        )


# Type alias for state observer callback
StateObserver = Callable[[PlateDesignerState], None]


class PlateDesignerViewModel:
    """
    ViewModel for PlateDesigner - orchestrates business logic.
    
    Implements Observer pattern for reactive UI updates.
    Thread-safe for background evolution.
    
    Usage:
        vm = PlateDesignerViewModel()
        vm.add_observer(lambda state: update_ui(state))
        vm.set_person_preset("tall_male")
        vm.start_evolution()
    """
    
    def __init__(self):
        # State
        self._state = PlateDesignerState()
        
        # Observers
        self._observers: List[StateObserver] = []
        self._lock = threading.Lock()
        
        # Evolution components
        self._optimizer: Optional[EvolutionaryOptimizer] = None
        self._pipeline: Optional[EvolutionPipeline] = None  # Pipeline with RDNN/LTM
        self._evolution_thread: Optional[threading.Thread] = None
        self._update_queue: queue.Queue = queue.Queue()
        
        # Initialize with default person
        self._init_default_person()
    
    def _init_default_person(self):
        """Initialize with default person preset."""
        preset_name = self._state.preset_name
        if preset_name in PERSON_PRESETS:
            preset = PERSON_PRESETS[preset_name]
            self._state.person = Person(
                height_m=preset.height_m,
                weight_kg=preset.weight_kg
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Observer Pattern
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_observer(self, callback: StateObserver):
        """Add state change observer."""
        with self._lock:
            self._observers.append(callback)
    
    def remove_observer(self, callback: StateObserver):
        """Remove state change observer."""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)
    
    def _notify_observers(self):
        """Notify all observers of state change."""
        with self._lock:
            state_copy = copy.copy(self._state)
            observers_copy = self._observers.copy()
        
        for observer in observers_copy:
            try:
                observer(state_copy)
            except Exception as e:
                print(f"Observer error: {e}")
    
    @property
    def state(self) -> PlateDesignerState:
        """Current state (read-only copy)."""
        with self._lock:
            return copy.copy(self._state)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Person Configuration
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_person_preset(self, preset_name: str):
        """Set person from preset."""
        if preset_name not in PERSON_PRESETS:
            return
        
        preset = PERSON_PRESETS[preset_name]
        self.set_person(preset.height_m, preset.weight_kg)
        self._state.preset_name = preset_name
    
    def set_person(self, height_m: float, weight_kg: float):
        """Set custom person dimensions."""
        with self._lock:
            self._state.person = Person(
                height_m=height_m,
                weight_kg=weight_kg
            )
        self._notify_observers()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evolution Configuration
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_evolution_config(
        self,
        population_size: int = 30,
        max_generations: int = 50,
        mutation_rate: float = 0.3,
        max_cutouts: int = 4,  # Default 4 like lutherie f-holes
        max_grooves: int = 0,  # Default OFF - expensive computationally
        max_attached_masses: int = 0,  # Attached masses (Shen 2016)
        use_nsga2: bool = False,  # True = pymoo NSGA-II (Pareto front)
        # Spring support configuration (physics-based: f_n = √(k/m)/2π)
        spring_count: int = 5,
        spring_stiffness_kn_m: float = 10.0,
        spring_damping_ratio: float = 0.10,
        spring_clearance_mm: float = 70.0,
        # Early stopping control
        force_all_generations: bool = True  # True = disable early stopping
    ):
        """Configure evolution parameters including spring supports.
        
        Spring supports use real vibration isolation physics:
        - Natural frequency: f_n = √(k/m) / (2π)
        - Isolation starts above f_n × √2
        - Transmissibility formula: T(ω) = √[(1+(2ζr)²)/((1-r²)²+(2ζr)²)]
        """
        with self._lock:
            self._state.population_size = population_size
            self._state.max_generations = max_generations
            self._state.mutation_rate = mutation_rate
            self._state.max_cutouts = max_cutouts
            self._state.max_grooves = max_grooves
            self._state.max_attached_masses = max_attached_masses
            self._state.use_nsga2 = use_nsga2
            # Spring configuration
            self._state.spring_count = spring_count
            self._state.spring_stiffness_kn_m = spring_stiffness_kn_m
            self._state.spring_damping_ratio = spring_damping_ratio
            self._state.spring_clearance_mm = spring_clearance_mm
            # Early stopping
            self._state.force_all_generations = force_all_generations
        self._notify_observers()
    
    def set_algorithm(self, use_nsga2: bool):
        """
        Switch between GA custom and pymoo NSGA-II.
        
        Args:
            use_nsga2: True for pymoo NSGA-II (Pareto front),
                      False for custom GA (single objective)
        """
        with self._lock:
            self._state.use_nsga2 = use_nsga2
        self._notify_observers()
    
    def set_contour_type(self, contour_name: str):
        """
        Set the plate contour type for evolution.
        
        Args:
            contour_name: One of RECTANGLE, GOLDEN_RECT, ELLIPSE, OVOID,
                         SUPERELLIPSE, ORGANIC, ERGONOMIC, FREEFORM, AUTO
        """
        with self._lock:
            self._state.contour_type = contour_name
        self._notify_observers()
    
    def set_zone_weights(self, spine_pct: int):
        """
        Set zone weights for frequency response optimization.
        
        Args:
            spine_pct: Percentage (0-100) of weight for spine flatness.
                       Head weight is automatically calculated as (100 - spine_pct)%.
        
        The person is the "string to be tuned" - we optimize:
        - 70% default for spine (tactile vibration feeling)
        - 30% default for head (binaural listening at ears)
        """
        with self._lock:
            # Convert percentage to normalized weights
            spine_pct = max(0, min(100, spine_pct))  # Clamp 0-100
            self._state.spine_weight = spine_pct / 100.0
            self._state.head_weight = (100 - spine_pct) / 100.0
        self._notify_observers()
    
    def set_optimization_weights(
        self,
        energy: float = 0.7,
        flatness: float = 0.5,
        spine: float = 0.8
    ):
        """
        Set advanced optimization weights from radar widget.
        
        Args:
            energy: Weight for vibration intensity (0-1)
            flatness: Weight for uniform frequency response (0-1)
            spine: Weight for spine zone coverage (0-1)
        
        These map to fitness function components:
        - Energy → modal_energy_weight
        - Flatness → frequency_response_weight
        - Spine → spine_zone_weight (vs head/limbs)
        """
        with self._lock:
            # Store advanced weights
            self._state.energy_weight = max(0.0, min(1.0, energy))
            self._state.flatness_weight = max(0.0, min(1.0, flatness))
            self._state.spine_weight = max(0.0, min(1.0, spine))
            
            # Derive head weight from remaining
            total = energy + flatness + spine
            if total > 0:
                self._state.head_weight = max(0.0, 1.0 - spine)
        
        self._notify_observers()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evolution Control
    # ─────────────────────────────────────────────────────────────────────────
    
    def start_evolution(self):
        """Start evolutionary optimization."""
        if self._state.is_running:
            return
        
        if self._state.person is None:
            self._set_error("No person configured")
            return
        
        # Check if NSGA-II requested but not available
        if self._state.use_nsga2 and not PYMOO_AVAILABLE:
            self._set_error("pymoo not installed. Run: pip install pymoo>=0.6.0")
            return
        
        # Update state
        with self._lock:
            self._state.is_running = True
            self._state.phase = EvolutionPhase.INITIALIZING
            self._state.generation = 0
            self._state.fitness_history = []
            self._state.start_time = time.time()
            self._state.error_message = None
        
        self._notify_observers()
        
        # Determine fixed contour from user selection
        fixed_contour = None
        if self._state.contour_type != "AUTO":
            # Map string to ContourType enum
            contour_map = {
                "RECTANGLE": ContourType.RECTANGLE,
                "GOLDEN_RECT": ContourType.GOLDEN_RECT,
                "ELLIPSE": ContourType.ELLIPSE,
                "OVOID": ContourType.OVOID,
                "VESICA_PISCIS": ContourType.VESICA_PISCIS,
                "SUPERELLIPSE": ContourType.SUPERELLIPSE,
                "ORGANIC": ContourType.ORGANIC,
                "ERGONOMIC": ContourType.ERGONOMIC,
                "FREEFORM": ContourType.FREEFORM,
            }
            fixed_contour = contour_map.get(self._state.contour_type)
        
        # Create zone weights for frequency response optimization
        zone_weights = ZoneWeights(
            spine=self._state.spine_weight,
            head=self._state.head_weight
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # CREATE OBJECTIVE WEIGHTS FROM RADAR/UI SETTINGS
        # Maps user-facing parameters to fitness objective weights
        # ═══════════════════════════════════════════════════════════════════════
        objectives = ObjectiveWeights(
            flatness=self._state.flatness_weight,        # From radar widget
            spine_coupling=self._state.energy_weight * 2,  # Energy → coupling (scaled)
            low_mass=0.3,                                 # Keep plate light
            manufacturability=0.5,                        # CNC-friendly shapes
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # ALGORITHM SELECTION: Pipeline (RDNN/LTM) vs NSGA-II vs GA
        # ═══════════════════════════════════════════════════════════════════════
        
        if self._state.use_pipeline:
            # Use EvolutionPipeline with full RDNN/LTM/Pokayoke integration
            # This is the "intelligent" optimizer with cross-run memory
            
            # Determine contour string for pipeline (None = AUTO)
            contour_str = None
            if self._state.contour_type != "AUTO":
                contour_str = self._state.contour_type
            
            # If force_all_generations is True, set very high stall threshold
            # to effectively disable early stopping
            stall_threshold = 9999 if self._state.force_all_generations else 10
            
            pipeline_config = PipelineConfig(
                mode=PipelineMode.HEADLESS,  # No user interaction during evolution
                enable_pokayoke=True,   # Phase 1: Anomaly detection
                enable_physics_rules=True,  # Phase 3: Hybrid physics rules
                enable_rdnn=True,       # Phase 4: PyTorch recurrent memory
                enable_ltm=True,        # Phase 5: Long-term knowledge distillation
                enable_templates=True,  # Phase 6: Scoring templates
                template_name="VAT Therapy",  # Vibroacoustic therapy template
                population_size=self._state.population_size,
                n_generations=self._state.max_generations,
                mutation_rate=self._state.mutation_rate,
                # Pass UI constraints to pipeline
                max_cutouts=self._state.max_cutouts,
                max_grooves=self._state.max_grooves,
                fixed_contour=contour_str,  # RECTANGLE, ELLIPSE, etc. or None
                log_every_n_generations=5,
                fitness_stall_threshold=stall_threshold,  # Force all gens if user wants
            )
            
            self._pipeline = EvolutionPipeline(
                config=pipeline_config,
                person=self._state.person,
            )
            
            # Start pipeline thread
            self._evolution_thread = threading.Thread(
                target=self._run_pipeline_evolution,
                daemon=True
            )
            
        elif self._state.use_nsga2 and PYMOO_AVAILABLE:
            # Use pymoo NSGA-II for Pareto front multi-objective optimization
            pymoo_config = PymooConfig(
                algorithm="nsga2",
                population_size=self._state.population_size,
                n_generations=self._state.max_generations,
                n_exciters=2,  # Start with 2 exciters for simplicity
            )
            
            self._pymoo_optimizer = PymooOptimizer(
                person=self._state.person,
                config=pymoo_config,
                zone_weights=zone_weights,
                contour_type=fixed_contour or ContourType.RECTANGLE,
            )
            
            # Start NSGA-II thread
            self._evolution_thread = threading.Thread(
                target=self._run_nsga2_evolution,
                daemon=True
            )
        else:
            # Use custom GA (single weighted objective) - LEGACY
            config = EvolutionConfig(
                population_size=self._state.population_size,
                n_generations=self._state.max_generations,
                mutation_rate=self._state.mutation_rate,
                max_cutouts=self._state.max_cutouts,
                max_grooves=self._state.max_grooves,
                fixed_contour=fixed_contour,
                # Spring support configuration (physics-based)
                spring_count=self._state.spring_count,
                spring_stiffness_kn_m=self._state.spring_stiffness_kn_m,
                spring_damping_ratio=self._state.spring_damping_ratio,
                spring_clearance_mm=self._state.spring_clearance_mm,
            )
            
            self._optimizer = EvolutionaryOptimizer(
                person=self._state.person,
                config=config,
                zone_weights=zone_weights,
                objectives=objectives,  # Pass radar weights to fitness evaluator
            )
            
            # Start GA thread
            self._evolution_thread = threading.Thread(
                target=self._run_evolution,
                daemon=True
            )
        
        self._evolution_thread.start()
    
    def stop_evolution(self):
        """Stop evolution."""
        # Request pipeline to stop (if using pipeline)
        if self._pipeline is not None:
            self._pipeline.request_stop()
        
        with self._lock:
            self._state.is_running = False
            self._state.phase = EvolutionPhase.STOPPED
        self._notify_observers()
    
    def reset(self):
        """Reset to initial state."""
        self.stop_evolution()
        
        with self._lock:
            self._state.generation = 0
            self._state.best_genome = None
            self._state.best_fitness = None
            self._state.best_fitness_value = 0.0
            self._state.fitness_history = []
            self._state.elapsed_time = 0.0
            self._state.phase = EvolutionPhase.IDLE
            self._state.error_message = None
        
        self._notify_observers()
    
    def _set_error(self, message: str):
        """Set error state."""
        with self._lock:
            self._state.phase = EvolutionPhase.ERROR
            self._state.error_message = message
            self._state.is_running = False
        self._notify_observers()
    
    def _run_evolution(self):
        """Run evolution in background thread."""
        
        def evolution_callback(state: EvolutionState) -> bool:
            """Called by optimizer each generation."""
            # Update our state
            with self._lock:
                self._state.phase = EvolutionPhase.EVOLVING
                self._state.generation = state.generation
                self._state.best_genome = state.best_genome
                self._state.best_fitness = state.best_fitness
                self._state.best_fitness_value = state.best_fitness.total_fitness
                self._state.diversity = state.diversity
                self._state.elapsed_time = time.time() - self._state.start_time
                
                # Calculate ETA
                if state.generation > 0:
                    time_per_gen = self._state.elapsed_time / state.generation
                    remaining_gens = self._state.max_generations - state.generation
                    self._state.eta_seconds = time_per_gen * remaining_gens
                
                # Record fitness snapshot
                snapshot = FitnessSnapshot(
                    generation=state.generation,
                    total=state.best_fitness.total_fitness,
                    flatness=state.best_fitness.flatness_score,
                    spine_coupling=state.best_fitness.spine_coupling_score,
                    low_mass=state.best_fitness.low_mass_score,
                    diversity=state.diversity,
                    timestamp=time.time(),
                )
                self._state.fitness_history.append(snapshot)
            
            # Notify observers (will be called from main thread via polling)
            self._update_queue.put(True)
            
            return self._state.is_running
        
        try:
            best = self._optimizer.run(callback=evolution_callback, verbose=False)
            
            with self._lock:
                if self._state.is_running:
                    self._state.phase = EvolutionPhase.CONVERGED
                self._state.is_running = False
            
        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[Evolution Error] {error_details}")  # Debug logging
            self._set_error(str(e))
        
        self._update_queue.put(True)
    
    def _run_nsga2_evolution(self):
        """Run pymoo NSGA-II optimization in background thread."""
        try:
            with self._lock:
                self._state.phase = EvolutionPhase.EVOLVING
            
            # Run NSGA-II (blocking call)
            result: PymooResult = self._pymoo_optimizer.run(verbose=True)
            
            # Get best balanced solution from Pareto front
            if result.pareto_genomes:
                # Use preference-based selection (balance all objectives)
                best_idx = result.get_best_balanced_index()
                best_genome = result.pareto_genomes[best_idx]
                best_fitness = result.pareto_fitness[best_idx]
                
                with self._lock:
                    self._state.best_genome = best_genome
                    self._state.best_fitness = best_fitness
                    self._state.best_fitness_value = best_fitness.total_fitness
                    self._state.phase = EvolutionPhase.CONVERGED
                    self._state.elapsed_time = result.elapsed_time
                    self._state.generation = result.n_generations
                    
                    # Store Pareto front info
                    self._pareto_result = result
                    
                    # Log Pareto front
                    n_pareto = len(result.pareto_genomes)
                    self._state.evolution_logs.append(
                        f"✅ NSGA-II: {n_pareto} Pareto-optimal solutions found"
                    )
                    self._state.log_count += 1
            
            self._state.is_running = False
            
        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[NSGA-II Error] {error_details}")
            self._set_error(str(e))
        
        self._update_queue.put(True)
    
    def _run_pipeline_evolution(self):
        """
        Run EvolutionPipeline with full RDNN/LTM/Pokayoke integration.
        
        This is the "intelligent" evolution that learns from previous runs
        and applies physics-based rules for better convergence.
        
        Integrates:
        - Phase 1: PokayokeObserver (anomaly detection)
        - Phase 2: ExciterGene (staged activation)
        - Phase 3: PhysicsRulesEngine (hybrid physics + learned rules)
        - Phase 4: RDNNMemory (PyTorch recurrent with warm start)
        - Phase 5: LTMDistiller (long-term knowledge extraction)
        - Phase 6: ScoringTemplates (VAT therapy fitness)
        """
        try:
            with self._lock:
                self._state.phase = EvolutionPhase.EVOLVING
                self._state.evolution_logs.append(
                    "🧠 Pipeline: Initializing RDNN/LTM memory system..."
                )
                self._state.log_count += 1
            
            # Initialize pipeline components
            self._pipeline.initialize()
            
            # Log active components
            active_components = []
            if self._pipeline.observer:
                active_components.append("Pokayoke")
            if self._pipeline.physics_engine:
                active_components.append("PhysicsRules")
            if self._pipeline.rdnn:
                active_components.append("RDNN")
            if self._pipeline.ltm_distiller:
                active_components.append("LTM")
            if self._pipeline.template:
                active_components.append("Templates")
            
            with self._lock:
                self._state.evolution_logs.append(
                    f"✅ Active: {', '.join(active_components)}"
                )
                self._state.log_count += 1
            
            self._notify_observers()
            
            # Define callback for per-generation updates
            def pipeline_callback(gen: int, best_fitness: float, best_genome):
                with self._lock:
                    self._state.generation = gen
                    self._state.best_genome = best_genome
                    self._state.best_fitness_value = best_fitness
                    self._state.elapsed_time = time.time() - self._state.start_time
                    
                    # Calculate ETA
                    if gen > 0:
                        time_per_gen = self._state.elapsed_time / gen
                        remaining_gens = self._state.max_generations - gen
                        self._state.eta_seconds = time_per_gen * remaining_gens
                    
                    # Create fitness snapshot for charts
                    # Note: PipelineResult doesn't have component scores yet
                    snapshot = FitnessSnapshot(
                        generation=gen,
                        total=best_fitness,
                        flatness=0.0,  # Pipeline doesn't expose these yet
                        spine_coupling=0.0,
                        low_mass=0.0,
                        diversity=0.0,
                        timestamp=time.time(),
                    )
                    self._state.fitness_history.append(snapshot)
                
                self._update_queue.put(True)
            
            # Add callback to pipeline
            self._pipeline.add_generation_callback(pipeline_callback)
            
            # Run evolution with pipeline
            result: PipelineResult = self._pipeline.run()
            
            # Update final state
            with self._lock:
                if self._state.is_running:
                    self._state.phase = EvolutionPhase.CONVERGED
                self._state.is_running = False
                self._state.best_genome = result.best_genome
                self._state.best_fitness_value = result.best_fitness
                self._state.generation = result.total_generations
                self._state.elapsed_time = result.runtime_seconds
                
                # Store pipeline result for access to RDNN state, distilled knowledge
                self._pipeline_result = result
                
                # Log completion with memory info
                memory_info = ""
                if result.rdnn_state is not None:
                    memory_info = " (RDNN state saved for next run)"
                if result.distilled_knowledge is not None:
                    memory_info += " (LTM patterns distilled)"
                
                self._state.evolution_logs.append(
                    f"✅ Pipeline complete: fitness={result.best_fitness:.4f}, "
                    f"evals={result.total_evaluations}, "
                    f"time={result.runtime_seconds:.1f}s{memory_info}"
                )
                self._state.log_count += 1
                
                # Log anomalies if any
                if result.anomalies:
                    self._state.evolution_logs.append(
                        f"⚠️ {len(result.anomalies)} anomalies detected during evolution"
                    )
                    self._state.log_count += 1
            
        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[Pipeline Error] {error_details}")
            self._set_error(str(e))
        
        self._update_queue.put(True)
    
    def get_pareto_front(self) -> Optional[List[PlateGenome]]:
        """Get all Pareto-optimal solutions (only after NSGA-II run)."""
        if hasattr(self, '_pareto_result') and self._pareto_result is not None:
            return self._pareto_result.pareto_genomes
        return None
    
    def poll_updates(self) -> bool:
        """
        Poll for updates from evolution thread.
        
        Returns True if there was an update.
        Call this from main thread periodically.
        """
        had_update = False
        try:
            while True:
                self._update_queue.get_nowait()
                had_update = True
        except queue.Empty:
            pass
        
        if had_update:
            self._notify_observers()
        
        return had_update
    
    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────
    
    def export_genome_json(self) -> Optional[Dict[str, Any]]:
        """Export best genome as JSON-serializable dict."""
        if self._state.best_genome is None:
            return None
        
        g = self._state.best_genome
        return {
            "contour_type": g.contour_type.value,
            "length": g.length,
            "width": g.width,
            "thickness_base": g.thickness_base,
            "thickness_profile": g.thickness_profile,
            "material_type": g.material_type,
            "cutouts": [
                {
                    "center_x": c.center_x,
                    "center_y": c.center_y,
                    "radius_x": c.radius_x,
                    "radius_y": c.radius_y,
                }
                for c in g.cutouts
            ],
            "fitness": {
                "total": self._state.best_fitness_value,
                "flatness": self._state.best_fitness.flatness_score if self._state.best_fitness else 0,
                "spine_coupling": self._state.best_fitness.spine_coupling_score if self._state.best_fitness else 0,
                "low_mass": self._state.best_fitness.low_mass_score if self._state.best_fitness else 0,
            },
            "person": {
                "height_m": self._state.person.height_m if self._state.person else 0,
                "weight_kg": self._state.person.weight_kg if self._state.person else 0,
            },
        }
    
    def export_for_dsp(self) -> Optional[Dict[str, Any]]:
        """
        Export simulation results for DSP agent processing.
        
        This exports comprehensive data that the DSP agent can use to:
        - Compensate for material limitations via EQ/filtering
        - Take advantage of plate resonances
        - Optimize per-channel audio processing
        
        Returns:
            DSPExportResult as dict, or None if no simulation data
        """
        if self._state.best_genome is None:
            print("[DSP Export] ERROR: best_genome is None")
            return None
        if self._state.best_fitness is None:
            print("[DSP Export] ERROR: best_fitness is None")
            return None
        
        print(f"[DSP Export] Starting export with genome type: {type(self._state.best_genome)}")
        print(f"[DSP Export] Fitness: {self._state.best_fitness}")
        
        try:
            from core.dsp_export import export_for_dsp as dsp_export_fn
            from core.plate_physics import MATERIALS
            
            # Get material
            material_name = getattr(self._state.best_genome, 'material_type', 'birch_plywood')
            print(f"[DSP Export] Material: {material_name}")
            if material_name not in MATERIALS:
                material_name = 'birch_plywood'
            
            # Get person weight
            person_weight = self._state.person.weight_kg if self._state.person else 70.0
            print(f"[DSP Export] Person weight: {person_weight}kg")
            
            # Get zone weights
            zone_weights = {
                'spine': self._state.spine_weight,
                'head': self._state.head_weight,
            }
            print(f"[DSP Export] Zone weights: {zone_weights}")
            
            # Export
            result = dsp_export_fn(
                genome=self._state.best_genome,
                fitness=self._state.best_fitness,
                material_name=material_name,
                person_weight_kg=person_weight,
                zone_weights=zone_weights,
            )
            
            print(f"[DSP Export] Success! Result type: {type(result)}")
            return result.to_dict()
        except ImportError as e:
            print(f"[DSP Export] ImportError: {e}")
            import traceback
            traceback.print_exc()
            # Return basic genome data instead of None on import error
            return self.export_genome_json()
        except Exception as e:
            print(f"[DSP Export] Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_fitness_history_arrays(self) -> Dict[str, List[float]]:
        """Get fitness history as arrays for plotting."""
        history = self._state.fitness_history
        return {
            "generations": [s.generation for s in history],
            "total": [s.total for s in history],
            "flatness": [s.flatness for s in history],
            "spine_coupling": [s.spine_coupling for s in history],
            "low_mass": [s.low_mass for s in history],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # EVOLUTION LOGGING - Physics-driven debugging
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_evolution_logs(self, n: int = 30, category: Optional[str] = None) -> List[str]:
        """
        Get recent evolution logs for UI display.
        
        Args:
            n: Number of recent logs to return
            category: Filter by category (zone_priority, modal_analysis, 
                      cutout_placement, fitness, generation, physics_decision)
        
        Returns:
            List of formatted log strings with timestamps
        """
        return get_formatted_logs(n=n, category=category)
    
    def get_evolution_summary(self) -> str:
        """Get a human-readable summary of the evolution process."""
        return generate_evolution_summary()
    
    def clear_evolution_logs(self):
        """Clear all evolution logs (for new optimization run)."""
        handler = get_evolution_handler()
        handler.clear()
        with self._lock:
            self._state.evolution_logs = []
            self._state.log_count = 0
    
    def log_target_comparison(self):
        """
        Log comparison between achieved and target metrics.
        
        Targets (from research):
        - Spine: < 10dB variation (Sum & Pan 2000)
        - Head: < 6dB variation (Lu 2012)  
        - Ear L/R: > 90% uniformity
        """
        if self._state.best_fitness is None:
            return
        
        f = self._state.best_fitness
        
        # Calculate achieved values (approximate from scores)
        # Score 1.0 = 0dB variation, Score 0.5 = target variation
        spine_achieved = (1.0 - f.spine_flatness_score) * 20.0  # Rough dB estimate
        head_achieved = (1.0 - f.head_flatness_score) * 12.0
        ear_achieved = f.ear_uniformity_score
        
        log_comparison_with_target(
            spine_target_db=10.0,
            spine_achieved_db=spine_achieved,
            head_target_db=6.0,
            head_achieved_db=head_achieved,
            ear_target_uniformity=0.90,
            ear_achieved_uniformity=ear_achieved,
            logger=evo_logger
        )
    
    def register_log_callback(self, callback: Callable[[Dict], None]):
        """
        Register callback to be notified on new log entries.
        
        Useful for real-time log display in UI.
        """
        handler = get_evolution_handler()
        handler.add_callback(callback)
    
    def unregister_log_callback(self, callback: Callable[[Dict], None]):
        """Unregister log callback."""
        handler = get_evolution_handler()
        handler.remove_callback(callback)


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PLATE DESIGNER VIEWMODEL TEST")
    print("=" * 60)
    
    # Test observer pattern
    def on_state_change(state: PlateDesignerState):
        print(f"State: {state.status_text}, Gen: {state.generation}, "
              f"Fitness: {state.best_fitness_value:.4f}")
    
    vm = PlateDesignerViewModel()
    vm.add_observer(on_state_change)
    
    print("\n1. Setting person preset...")
    vm.set_person_preset("tall_male")
    
    print("\n2. Configuring evolution...")
    vm.set_evolution_config(
        population_size=20,
        max_generations=10,
        mutation_rate=0.3
    )
    
    print("\n3. Starting evolution...")
    vm.start_evolution()
    
    # Poll for updates
    import time
    while vm.state.is_running:
        vm.poll_updates()
        time.sleep(0.1)
    
    print("\n4. Final state:")
    state = vm.state
    print(f"   Phase: {state.phase.value}")
    print(f"   Best fitness: {state.best_fitness_value:.4f}")
    print(f"   Plate: {state.plate_info_text}")
    
    print("\n5. Fitness history:")
    for s in state.fitness_history[-5:]:
        print(f"   Gen {s.generation}: {s.total:.4f}")
