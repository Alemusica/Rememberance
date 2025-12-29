"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE VIEWMODEL - Business Logic Layer                    ║
║                                                                              ║
║   Unified ViewModel combining:                                               ║
║   • Modal analysis (from Plate Lab)                                          ║
║   • Evolutionary optimization (from Plate Designer)                          ║
║   • Audio playback control                                                   ║
║                                                                              ║
║   Thread-safe, observable state pattern for reactive UI.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any, Tuple
from enum import Enum
import threading
import queue
import time
import copy
import numpy as np

# Core imports
from core.materials import Material, MATERIALS, DEFAULT_MATERIAL, get_material
from core.fem import FEMSolver, FEMResult, FEMMode as FEMModalMode, MeshData, ShapeType, get_solver, create_mesh

# Try evolutionary imports (optional)
try:
    from core.person import Person, PERSON_PRESETS
    from core.plate_genome import PlateGenome, ContourType
    from core.fitness import FitnessEvaluator, FitnessResult
    from core.evolutionary_optimizer import EvolutionaryOptimizer, EvolutionConfig, EvolutionState
    HAS_EVOLUTION = True
except ImportError:
    HAS_EVOLUTION = False
    Person = None
    PERSON_PRESETS = {}

# Try body zones (optional)
try:
    from core.body_zones import BodyZoneModel, ZONE_PRESETS
    HAS_BODY_ZONES = True
except ImportError:
    HAS_BODY_ZONES = False

# Try audio (optional)
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class PlateMode(Enum):
    """Operating modes for the plate tab."""
    DESIGN = "design"           # Manual shape design + modal analysis
    OPTIMIZE = "optimize"       # Evolutionary/SIMP optimization
    PREVIEW = "preview"         # Audio preview + visualization


class EvolutionPhase(Enum):
    """Evolution progress phase."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    EVOLVING = "evolving"
    CONVERGED = "converged"
    STOPPED = "stopped"
    ERROR = "error"


# ══════════════════════════════════════════════════════════════════════════════
# EXCITER DATA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Exciter:
    """Exciter/transducer position and settings."""
    x: float = 0.5              # Normalized position [0, 1]
    y: float = 0.5
    phase: float = 0.0          # Degrees
    amplitude: float = 1.0
    enabled: bool = True
    color: str = "#ff6b6b"
    
    # Computed
    coupling: float = 1.0       # Coupling coefficient for current mode


# ══════════════════════════════════════════════════════════════════════════════
# STATE CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlateState:
    """
    Complete state for Plate Tab.
    
    All UI updates driven by changes to this state.
    """
    # === Operating Mode ===
    mode: PlateMode = PlateMode.DESIGN
    
    # === Plate Geometry ===
    shape_type: ShapeType = ShapeType.RECTANGLE
    length: float = 1.95        # meters
    width: float = 0.60         # meters
    thickness: float = 0.018    # meters
    polygon_vertices: List[Tuple[float, float]] = field(default_factory=list)
    
    # === Material ===
    material_key: str = DEFAULT_MATERIAL
    
    # === Mesh & FEM Results ===
    mesh: Optional[MeshData] = None
    fem_result: Optional[FEMResult] = None
    selected_mode_idx: int = 0
    n_modes_to_compute: int = 12
    
    # === Exciters ===
    exciters: List[Exciter] = field(default_factory=lambda: [
        Exciter(0.25, 0.5, 0, 1.0, True, "#ff6b6b"),
        Exciter(0.75, 0.5, 0, 1.0, True, "#6bff6b"),
    ])
    
    # === Person (for optimization) ===
    person_height: float = 1.75
    person_weight: float = 70.0
    
    # === Evolution State ===
    evolution_phase: EvolutionPhase = EvolutionPhase.IDLE
    evolution_generation: int = 0
    evolution_max_generations: int = 50
    evolution_population_size: int = 30
    evolution_best_fitness: float = 0.0
    evolution_history: List[Dict[str, float]] = field(default_factory=list)
    
    # === Audio State ===
    is_playing: bool = False
    play_frequency: float = 0.0     # 0 = use mode frequency
    play_volume: float = 0.7
    
    # === Visualization Options ===
    show_human_overlay: bool = True
    show_nodal_lines: bool = True
    show_antahkarana: bool = False
    heatmap_resolution: int = 50
    
    # === Status ===
    is_computing: bool = False
    status_message: str = "Ready"
    error_message: Optional[str] = None
    
    # === Computed Properties ===
    @property
    def material(self) -> Material:
        """Get current material."""
        return get_material(self.material_key)
    
    @property
    def selected_mode(self) -> Optional[FEMModalMode]:
        """Get currently selected mode."""
        if self.fem_result and 0 <= self.selected_mode_idx < len(self.fem_result.modes):
            return self.fem_result.modes[self.selected_mode_idx]
        return None
    
    @property
    def mode_frequencies(self) -> List[float]:
        """List of all mode frequencies."""
        if self.fem_result:
            return [m.frequency for m in self.fem_result.modes]
        return []
    
    @property
    def plate_dimensions_mm(self) -> Tuple[float, float, float]:
        """Plate dimensions in mm (length, width, thickness)."""
        return (
            self.length * 1000,
            self.width * 1000,
            self.thickness * 1000
        )


# Type alias for state observer
StateObserver = Callable[[PlateState], None]


# ══════════════════════════════════════════════════════════════════════════════
# VIEWMODEL
# ══════════════════════════════════════════════════════════════════════════════

class PlateViewModel:
    """
    ViewModel for unified Plate Tab.
    
    Combines modal analysis + evolution + audio into single interface.
    Thread-safe with observable state pattern.
    
    Usage:
        vm = PlateViewModel()
        vm.add_observer(lambda s: update_ui(s))
        vm.set_dimensions(2.0, 0.6, 0.018)
        vm.run_analysis()
    """
    
    def __init__(self):
        # State
        self._state = PlateState()
        
        # Thread safety
        self._lock = threading.Lock()
        self._observers: List[StateObserver] = []
        
        # FEM solver
        self._solver = get_solver("auto")
        
        # Background threads
        self._analysis_thread: Optional[threading.Thread] = None
        self._evolution_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        
        # Evolution (optional)
        self._optimizer: Optional[Any] = None  # EvolutionaryOptimizer
        
        # Audio stream
        self._audio_stream: Optional[Any] = None
        
    # ─────────────────────────────────────────────────────────────────────────
    # Observer Pattern
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_observer(self, callback: StateObserver):
        """Add state change observer."""
        with self._lock:
            self._observers.append(callback)
    
    def remove_observer(self, callback: StateObserver):
        """Remove observer."""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)
    
    def _notify(self):
        """Notify all observers."""
        with self._lock:
            state_copy = copy.copy(self._state)
            observers = self._observers.copy()
        
        for obs in observers:
            try:
                obs(state_copy)
            except Exception as e:
                print(f"Observer error: {e}")
    
    @property
    def state(self) -> PlateState:
        """Get current state (copy)."""
        with self._lock:
            return copy.copy(self._state)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Mode Control
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_mode(self, mode: PlateMode):
        """Switch operating mode."""
        with self._lock:
            self._state.mode = mode
        self._notify()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plate Configuration
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_dimensions(self, length: float, width: float, thickness: float):
        """Set plate dimensions in meters."""
        with self._lock:
            self._state.length = length
            self._state.width = width
            self._state.thickness = thickness
            self._state.mesh = None  # Invalidate mesh
            self._state.fem_result = None
        self._notify()
    
    def set_shape(self, shape_type: ShapeType, vertices: List[Tuple[float, float]] = None):
        """Set plate shape."""
        with self._lock:
            self._state.shape_type = shape_type
            if vertices:
                self._state.polygon_vertices = vertices
            self._state.mesh = None
            self._state.fem_result = None
        self._notify()
    
    def set_material(self, material_key: str):
        """Set material by key."""
        if material_key not in MATERIALS:
            return
        with self._lock:
            self._state.material_key = material_key
            self._state.fem_result = None  # Invalidate
        self._notify()
    
    def set_thickness(self, thickness_m: float):
        """Set thickness in meters."""
        with self._lock:
            self._state.thickness = thickness_m
            self._state.fem_result = None
        self._notify()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Exciter Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_exciter(self, x: float = 0.5, y: float = 0.5) -> int:
        """Add exciter at position, return index."""
        colors = ["#ff6b6b", "#6bff6b", "#6b6bff", "#ffff6b"]
        with self._lock:
            idx = len(self._state.exciters)
            color = colors[idx % len(colors)]
            self._state.exciters.append(Exciter(x, y, 0, 1.0, True, color))
        self._notify()
        return idx
    
    def remove_exciter(self, index: int):
        """Remove exciter by index."""
        with self._lock:
            if 0 <= index < len(self._state.exciters):
                self._state.exciters.pop(index)
        self._notify()
    
    def move_exciter(self, index: int, x: float, y: float):
        """Move exciter to new position."""
        with self._lock:
            if 0 <= index < len(self._state.exciters):
                self._state.exciters[index].x = max(0, min(1, x))
                self._state.exciters[index].y = max(0, min(1, y))
                # Update coupling
                self._update_exciter_couplings()
        self._notify()
    
    def set_exciter_phase(self, index: int, phase: float):
        """Set exciter phase in degrees."""
        with self._lock:
            if 0 <= index < len(self._state.exciters):
                self._state.exciters[index].phase = phase % 360
        self._notify()
    
    def _update_exciter_couplings(self):
        """Update coupling coefficients for all exciters."""
        mode = self._state.selected_mode
        if mode is None:
            return
        
        for exc in self._state.exciters:
            # Convert normalized position to meters
            x_m = exc.x * self._state.length
            y_m = exc.y * self._state.width
            
            try:
                disp = mode.get_displacement_at(x_m, y_m)
                exc.coupling = abs(disp)
            except:
                exc.coupling = 0.5
    
    # ─────────────────────────────────────────────────────────────────────────
    # Modal Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def run_analysis(self, n_modes: int = None):
        """Run FEM modal analysis (async)."""
        if self._state.is_computing:
            return
        
        if n_modes:
            self._state.n_modes_to_compute = n_modes
        
        with self._lock:
            self._state.is_computing = True
            self._state.status_message = "Computing modes..."
        self._notify()
        
        # Run in background
        self._analysis_thread = threading.Thread(
            target=self._run_analysis_bg,
            daemon=True
        )
        self._analysis_thread.start()
    
    def _run_analysis_bg(self):
        """Background analysis worker."""
        try:
            # Generate mesh if needed
            if self._state.mesh is None:
                mesh = create_mesh(
                    self._state.shape_type,
                    self._state.length,
                    self._state.width,
                    resolution=25,
                    vertices=self._state.polygon_vertices or None
                )
                with self._lock:
                    self._state.mesh = mesh
            
            # Run FEM
            result = self._solver.solve(
                self._state.mesh,
                self._state.thickness,
                self._state.material,
                n_modes=self._state.n_modes_to_compute
            )
            
            with self._lock:
                self._state.fem_result = result
                self._state.is_computing = False
                self._state.status_message = f"Found {len(result.modes)} modes"
                self._state.error_message = None
                self._update_exciter_couplings()
            
        except Exception as e:
            with self._lock:
                self._state.is_computing = False
                self._state.error_message = str(e)
                self._state.status_message = "Analysis failed"
        
        self._notify()
    
    def select_mode(self, index: int):
        """Select mode by index."""
        with self._lock:
            if self._state.fem_result and 0 <= index < len(self._state.fem_result.modes):
                self._state.selected_mode_idx = index
                self._update_exciter_couplings()
        self._notify()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Audio Playback
    # ─────────────────────────────────────────────────────────────────────────
    
    def play_mode(self, frequency: float = None):
        """Start playing current mode frequency."""
        if not HAS_AUDIO:
            with self._lock:
                self._state.error_message = "sounddevice not installed"
            self._notify()
            return
        
        mode = self._state.selected_mode
        freq = frequency or (mode.frequency if mode else 100.0)
        
        with self._lock:
            self._state.is_playing = True
            self._state.play_frequency = freq
        self._notify()
        
        self._start_audio_stream(freq)
    
    def stop_playback(self):
        """Stop audio playback."""
        self._stop_audio_stream()
        
        with self._lock:
            self._state.is_playing = False
        self._notify()
    
    def _start_audio_stream(self, frequency: float):
        """Start audio stream."""
        if not HAS_AUDIO:
            return
        
        self._stop_audio_stream()
        
        sample_rate = 44100
        phase = 0.0
        
        def callback(outdata, frames, time_info, status):
            nonlocal phase
            t = (np.arange(frames) + phase) / sample_rate
            # Binaural with small offset for spatial effect
            left = np.sin(2 * np.pi * frequency * t)
            right = np.sin(2 * np.pi * (frequency + 0.5) * t)  # Slight detuning
            
            outdata[:, 0] = left * self._state.play_volume
            outdata[:, 1] = right * self._state.play_volume
            phase += frames
        
        try:
            self._audio_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=2,
                callback=callback
            )
            self._audio_stream.start()
        except Exception as e:
            print(f"Audio error: {e}")
    
    def _stop_audio_stream(self):
        """Stop audio stream."""
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evolution (Optional)
    # ─────────────────────────────────────────────────────────────────────────
    
    def start_evolution(self):
        """Start evolutionary optimization."""
        if not HAS_EVOLUTION:
            with self._lock:
                self._state.error_message = "Evolution modules not available"
            self._notify()
            return
        
        if self._state.evolution_phase == EvolutionPhase.EVOLVING:
            return
        
        with self._lock:
            self._state.evolution_phase = EvolutionPhase.INITIALIZING
            self._state.evolution_generation = 0
            self._state.evolution_history = []
        self._notify()
        
        self._evolution_thread = threading.Thread(
            target=self._run_evolution_bg,
            daemon=True
        )
        self._evolution_thread.start()
    
    def stop_evolution(self):
        """Stop evolution."""
        with self._lock:
            self._state.evolution_phase = EvolutionPhase.STOPPED
        self._notify()
    
    def _run_evolution_bg(self):
        """Background evolution worker."""
        try:
            # Create person
            person = Person(
                height_m=self._state.person_height,
                weight_kg=self._state.person_weight
            )
            
            # Create optimizer
            config = EvolutionConfig(
                population_size=self._state.evolution_population_size,
                n_generations=self._state.evolution_max_generations,
            )
            
            self._optimizer = EvolutionaryOptimizer(
                person=person,
                config=config
            )
            
            # Run evolution
            def callback(evo_state: EvolutionState) -> bool:
                with self._lock:
                    self._state.evolution_phase = EvolutionPhase.EVOLVING
                    self._state.evolution_generation = evo_state.generation
                    if evo_state.best_fitness:
                        self._state.evolution_best_fitness = evo_state.best_fitness.total
                        self._state.evolution_history.append({
                            "gen": evo_state.generation,
                            "fitness": evo_state.best_fitness.total
                        })
                
                self._notify()
                
                # Continue if not stopped
                return self._state.evolution_phase == EvolutionPhase.EVOLVING
            
            result = self._optimizer.run(callback=callback)
            
            with self._lock:
                self._state.evolution_phase = EvolutionPhase.CONVERGED
                if result:
                    # Update plate dimensions from best genome
                    self._state.length = result.best_genome.length
                    self._state.width = result.best_genome.width
                    self._state.thickness = result.best_genome.thickness_base
                    self._state.mesh = None  # Invalidate
                    self._state.fem_result = None
            
        except Exception as e:
            with self._lock:
                self._state.evolution_phase = EvolutionPhase.ERROR
                self._state.error_message = str(e)
        
        self._notify()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Visualization Options
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_show_human(self, show: bool):
        """Toggle human overlay."""
        with self._lock:
            self._state.show_human_overlay = show
        self._notify()
    
    def set_show_nodal_lines(self, show: bool):
        """Toggle nodal lines."""
        with self._lock:
            self._state.show_nodal_lines = show
        self._notify()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────
    
    def export_dxf(self, filepath: str) -> bool:
        """Export plate outline to DXF."""
        # TODO: Implement DXF export
        return False
    
    def export_preset(self, filepath: str) -> bool:
        """Export current configuration to JSON."""
        import json
        
        try:
            data = {
                "length_m": self._state.length,
                "width_m": self._state.width,
                "thickness_m": self._state.thickness,
                "material": self._state.material_key,
                "shape": self._state.shape_type.value,
                "exciters": [
                    {"x": e.x, "y": e.y, "phase": e.phase}
                    for e in self._state.exciters
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────
    
    def cleanup(self):
        """Clean up resources."""
        self._stop_audio_stream()
        self.stop_evolution()
