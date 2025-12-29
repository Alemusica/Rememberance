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
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from enum import Enum
import threading
import queue
import time
import copy

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
    max_cutouts: int = 0  # Internal cutouts for weight reduction
    
    # Zone weights for frequency response optimization (spine vs head)
    spine_weight: float = 0.70  # 70% priority on spine flatness
    head_weight: float = 0.30   # 30% priority on head/ear flatness
    
    # Contour type for plate shape
    contour_type: str = "ORGANIC"  # Default: smooth organic curves
    
    # Advanced optimization weights (from radar widget)
    energy_weight: float = 0.7   # Vibration intensity
    flatness_weight: float = 0.5  # Frequency response uniformity
    
    # Error handling
    error_message: Optional[str] = None
    
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
        max_cutouts: int = 0
    ):
        """Configure evolution parameters."""
        with self._lock:
            self._state.population_size = population_size
            self._state.max_generations = max_generations
            self._state.mutation_rate = mutation_rate
            self._state.max_cutouts = max_cutouts
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
                "SUPERELLIPSE": ContourType.SUPERELLIPSE,
                "ORGANIC": ContourType.ORGANIC,
                "ERGONOMIC": ContourType.ERGONOMIC,
                "FREEFORM": ContourType.FREEFORM,
            }
            fixed_contour = contour_map.get(self._state.contour_type)
        
        # Create config
        config = EvolutionConfig(
            population_size=self._state.population_size,
            n_generations=self._state.max_generations,
            mutation_rate=self._state.mutation_rate,
            max_cutouts=self._state.max_cutouts,
            fixed_contour=fixed_contour,  # None means AUTO (evolve contour type)
        )
        
        # Create zone weights for frequency response optimization
        zone_weights = ZoneWeights(
            spine=self._state.spine_weight,
            head=self._state.head_weight
        )
        
        # Create optimizer with zone weights
        self._optimizer = EvolutionaryOptimizer(
            person=self._state.person,
            config=config,
            zone_weights=zone_weights,
        )
        
        # Start evolution thread
        self._evolution_thread = threading.Thread(
            target=self._run_evolution,
            daemon=True
        )
        self._evolution_thread.start()
    
    def stop_evolution(self):
        """Stop evolution."""
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
