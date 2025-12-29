"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE TAB - Unified View Layer                            ║
║                                                                              ║
║   Single tab combining Design + Optimize + Preview modes.                    ║
║   Clean MVVM architecture with separated concerns.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from .viewmodel import PlateViewModel, PlateState, PlateMode, EvolutionPhase
from .widgets.canvas import PlateCanvas
from .widgets.controls import ControlPanel
from .widgets.modes_list import ModesListPanel
from .widgets.evolution_panel import EvolutionPanel

# Theme
try:
    from ui.theme import STYLE, PlateLabStyle
    HAS_THEME = True
except ImportError:
    HAS_THEME = False


class PlateTab(ttk.Frame):
    """
    Unified Plate Tab combining modal analysis and optimization.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  [MODE TABS: Design | Optimize | Preview]                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌───────────────┐  ┌───────────────────────────────────────────────┐  │
    │  │               │  │                                                │  │
    │  │   CONTROLS    │  │              PLATE CANVAS                      │  │
    │  │   (Left)      │  │         (Heatmap + Human + Exciters)          │  │
    │  │               │  │                                                │  │
    │  │  • Dimensions │  │                                                │  │
    │  │  • Material   │  └───────────────────────────────────────────────┘  │
    │  │  • Exciters   │  ┌───────────────────────────────────────────────┐  │
    │  │               │  │  MODE LIST / EVOLUTION PROGRESS               │  │
    │  └───────────────┘  └───────────────────────────────────────────────┘  │
    │                                                                          │
    │  [STATUS BAR: Mode info | Coupling | Play controls]                     │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, parent, audio_engine=None):
        super().__init__(parent)
        
        # ViewModel
        self._vm = PlateViewModel()
        self._vm.add_observer(self._on_state_changed)
        
        # Audio engine (optional external)
        self._audio_engine = audio_engine
        
        # Build UI
        self._create_widgets()
        self._layout_widgets()
        self._bind_events()
        
        # Initial state (no auto-analysis to avoid lag)
        self._update_ui(self._vm.state)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Widget Creation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_widgets(self):
        """Create all widgets."""
        
        # === Mode Selection Tabs ===
        self._mode_frame = ttk.Frame(self)
        
        self._mode_var = tk.StringVar(value=PlateMode.DESIGN.value)
        
        modes = [
            ("🎨 Design", PlateMode.DESIGN),
            ("⚡ Optimize", PlateMode.OPTIMIZE),
            ("🔊 Preview", PlateMode.PREVIEW),
        ]
        
        for text, mode in modes:
            rb = ttk.Radiobutton(
                self._mode_frame,
                text=text,
                variable=self._mode_var,
                value=mode.value,
                command=self._on_mode_changed
            )
            rb.pack(side=tk.LEFT, padx=10, pady=5)
        
        # === Main Content Area ===
        self._content_frame = ttk.Frame(self)
        
        # Left Panel - Controls
        self._left_panel = ttk.Frame(self._content_frame, width=280)
        self._left_panel.pack_propagate(False)
        
        self._controls = ControlPanel(
            self._left_panel,
            on_dimension_change=self._on_dimension_change,
            on_material_change=self._on_material_change,
            on_analyze=self._on_analyze_clicked,
        )
        
        # Right Panel - Visualization
        self._right_panel = ttk.Frame(self._content_frame)
        
        # Canvas
        self._canvas = PlateCanvas(
            self._right_panel,
            on_exciter_moved=self._on_exciter_moved,
            on_exciter_added=self._on_exciter_added,
        )
        
        # Bottom panel - Modes list or Evolution
        self._bottom_panel = ttk.Frame(self._right_panel)
        
        self._modes_list = ModesListPanel(
            self._bottom_panel,
            on_mode_selected=self._on_mode_selected,
            on_play_clicked=self._on_play_clicked,
        )
        
        self._evolution_panel = EvolutionPanel(
            self._bottom_panel,
            on_start=self._on_evolution_start,
            on_stop=self._on_evolution_stop,
        )
        
        # === Status Bar ===
        self._status_frame = ttk.Frame(self)
        
        self._status_label = ttk.Label(
            self._status_frame,
            text="Ready",
            anchor="w"
        )
        
        self._coupling_label = ttk.Label(
            self._status_frame,
            text="Coupling: --",
            anchor="center"
        )
        
        self._play_btn = ttk.Button(
            self._status_frame,
            text="▶ Play",
            command=self._on_play_stop_clicked
        )
    
    def _layout_widgets(self):
        """Layout widgets."""
        
        # Mode tabs at top
        self._mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Content area
        self._content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self._controls.pack(fill=tk.BOTH, expand=True)
        
        self._right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._bottom_panel.pack(fill=tk.X, pady=(5, 0))
        
        # Initially show modes list
        self._modes_list.pack(fill=tk.BOTH, expand=True)
        
        # Status bar at bottom
        self._status_frame.pack(fill=tk.X, padx=5, pady=5)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._coupling_label.pack(side=tk.LEFT, padx=20)
        self._play_btn.pack(side=tk.RIGHT, padx=5)
    
    def _bind_events(self):
        """Bind events."""
        # Keyboard shortcuts
        self.bind_all("<space>", lambda e: self._on_play_stop_clicked())
        self.bind_all("<Escape>", lambda e: self._vm.stop_playback())
    
    # ─────────────────────────────────────────────────────────────────────────
    # State Observer
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_state_changed(self, state: PlateState):
        """Called when ViewModel state changes."""
        # Schedule UI update on main thread
        self.after(0, lambda: self._update_ui(state))
    
    def _update_ui(self, state: PlateState):
        """Update all UI from state."""
        
        # Update controls
        self._controls.update_state(state)
        
        # Update canvas
        self._canvas.update_state(state)
        
        # Update bottom panel based on mode
        if state.mode == PlateMode.OPTIMIZE:
            self._modes_list.pack_forget()
            self._evolution_panel.pack(fill=tk.BOTH, expand=True)
            self._evolution_panel.update_state(state)
        else:
            self._evolution_panel.pack_forget()
            self._modes_list.pack(fill=tk.BOTH, expand=True)
            self._modes_list.update_state(state)
        
        # Update status bar
        self._status_label.config(text=state.status_message)
        
        if state.selected_mode:
            mode = state.selected_mode
            coupling = sum(e.coupling for e in state.exciters) / max(1, len(state.exciters))
            self._coupling_label.config(
                text=f"Mode {mode.index + 1}: {mode.frequency:.1f} Hz | Coupling: {coupling:.2f}"
            )
        
        # Update play button
        if state.is_playing:
            self._play_btn.config(text="⏹ Stop")
        else:
            self._play_btn.config(text="▶ Play")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_mode_changed(self):
        """Mode tab changed."""
        mode_str = self._mode_var.get()
        mode = PlateMode(mode_str)
        self._vm.set_mode(mode)
    
    def _on_dimension_change(self, length: float, width: float, thickness: float):
        """Dimensions changed."""
        self._vm.set_dimensions(length, width, thickness)
    
    def _on_material_change(self, material_key: str):
        """Material changed."""
        self._vm.set_material(material_key)
    
    def _on_analyze_clicked(self):
        """Analyze button clicked."""
        self._vm.run_analysis()
    
    def _on_exciter_moved(self, index: int, x: float, y: float):
        """Exciter dragged to new position."""
        self._vm.move_exciter(index, x, y)
    
    def _on_exciter_added(self, x: float, y: float):
        """Double-click to add exciter."""
        self._vm.add_exciter(x, y)
    
    def _on_mode_selected(self, index: int):
        """Mode selected from list."""
        self._vm.select_mode(index)
    
    def _on_play_clicked(self, index: int):
        """Play specific mode."""
        self._vm.select_mode(index)
        state = self._vm.state
        if state.selected_mode:
            self._vm.play_mode(state.selected_mode.frequency)
    
    def _on_play_stop_clicked(self):
        """Play/stop toggle."""
        state = self._vm.state
        if state.is_playing:
            self._vm.stop_playback()
        else:
            if state.selected_mode:
                self._vm.play_mode(state.selected_mode.frequency)
    
    def _on_evolution_start(self):
        """Start evolution."""
        self._vm.start_evolution()
    
    def _on_evolution_stop(self):
        """Stop evolution."""
        self._vm.stop_evolution()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────
    
    def destroy(self):
        """Clean up on destroy."""
        self._vm.cleanup()
        super().destroy()
