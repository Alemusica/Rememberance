"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EVOLUTION PANEL - Optimization Control                    ║
║                                                                              ║
║   Shows evolution progress with:                                              ║
║   • Progress bar                                                              ║
║   • Generation counter                                                        ║
║   • Fitness history chart                                                     ║
║   • Start/Stop controls                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..viewmodel import PlateState, EvolutionPhase


class EvolutionPanel(ttk.Frame):
    """
    Panel for evolutionary optimization control.
    """
    
    def __init__(
        self,
        parent,
        on_start: Callable[[], None] = None,
        on_stop: Callable[[], None] = None,
    ):
        super().__init__(parent)
        
        # Callbacks
        self._on_start = on_start
        self._on_stop = on_stop
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        """Create evolution control widgets."""
        
        # Header
        ttk.Label(
            self,
            text="⚡ Evolutionary Optimization",
            font=("SF Pro", 11, "bold")
        ).pack(anchor="w", padx=5, pady=5)
        
        # Config frame
        config_frame = ttk.Frame(self)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Population size
        ttk.Label(config_frame, text="Population:").grid(
            row=0, column=0, sticky="w", padx=2
        )
        self._pop_var = tk.IntVar(value=30)
        ttk.Spinbox(
            config_frame,
            from_=10, to=100, increment=5,
            textvariable=self._pop_var,
            width=6
        ).grid(row=0, column=1, padx=2)
        
        # Generations
        ttk.Label(config_frame, text="Generations:").grid(
            row=0, column=2, sticky="w", padx=2
        )
        self._gen_var = tk.IntVar(value=50)
        ttk.Spinbox(
            config_frame,
            from_=10, to=200, increment=10,
            textvariable=self._gen_var,
            width=6
        ).grid(row=0, column=3, padx=2)
        
        # Progress frame
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self._progress_var,
            maximum=100
        )
        self._progress_bar.pack(fill=tk.X, pady=2)
        
        # Status label
        self._status_label = ttk.Label(
            progress_frame,
            text="Ready to start",
            font=("SF Pro", 9)
        )
        self._status_label.pack(anchor="w")
        
        # Fitness label
        self._fitness_label = ttk.Label(
            progress_frame,
            text="Best fitness: --",
            font=("SF Pro", 10, "bold")
        )
        self._fitness_label.pack(anchor="w", pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self._start_btn = ttk.Button(
            btn_frame,
            text="▶ Start",
            command=self._on_start_clicked
        )
        self._start_btn.pack(side=tk.LEFT, padx=2)
        
        self._stop_btn = ttk.Button(
            btn_frame,
            text="⏹ Stop",
            command=self._on_stop_clicked,
            state="disabled"
        )
        self._stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Mini fitness chart (simple canvas)
        self._chart_canvas = tk.Canvas(
            self,
            height=60,
            bg="#1a1a2e",
            highlightthickness=0
        )
        self._chart_canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # History for chart
        self._fitness_history = []
    
    def update_state(self, state: PlateState):
        """Update panel from state."""
        
        # Update progress
        if state.evolution_max_generations > 0:
            progress = (state.evolution_generation / state.evolution_max_generations) * 100
            self._progress_var.set(progress)
        
        # Update status
        phase = state.evolution_phase
        
        if phase == EvolutionPhase.IDLE:
            self._status_label.config(text="Ready to start")
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            
        elif phase == EvolutionPhase.INITIALIZING:
            self._status_label.config(text="Initializing population...")
            self._start_btn.config(state="disabled")
            self._stop_btn.config(state="normal")
            
        elif phase == EvolutionPhase.EVOLVING:
            self._status_label.config(
                text=f"Generation {state.evolution_generation}/{state.evolution_max_generations}"
            )
            self._start_btn.config(state="disabled")
            self._stop_btn.config(state="normal")
            
        elif phase == EvolutionPhase.CONVERGED:
            self._status_label.config(text="✓ Optimization complete!")
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            
        elif phase == EvolutionPhase.STOPPED:
            self._status_label.config(text="⏹ Stopped")
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            
        elif phase == EvolutionPhase.ERROR:
            self._status_label.config(text=f"⚠ Error: {state.error_message}")
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
        
        # Update fitness
        if state.evolution_best_fitness > 0:
            self._fitness_label.config(
                text=f"Best fitness: {state.evolution_best_fitness:.3f}"
            )
        
        # Update chart
        self._fitness_history = [h.get("fitness", 0) for h in state.evolution_history]
        self._draw_chart()
    
    def _draw_chart(self):
        """Draw mini fitness history chart."""
        self._chart_canvas.delete("all")
        
        if len(self._fitness_history) < 2:
            return
        
        cw = self._chart_canvas.winfo_width()
        ch = self._chart_canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            return
        
        # Normalize data
        data = self._fitness_history[-50:]  # Last 50 points
        if not data:
            return
        
        min_val = min(data)
        max_val = max(data)
        val_range = max_val - min_val + 1e-6
        
        # Draw line
        points = []
        for i, val in enumerate(data):
            x = (i / (len(data) - 1)) * (cw - 10) + 5
            y = ch - 5 - ((val - min_val) / val_range) * (ch - 10)
            points.extend([x, y])
        
        if len(points) >= 4:
            self._chart_canvas.create_line(
                points,
                fill="#00ff88",
                width=2,
                smooth=True
            )
    
    def _on_start_clicked(self):
        """Start button clicked."""
        if self._on_start:
            self._on_start()
    
    def _on_stop_clicked(self):
        """Stop button clicked."""
        if self._on_stop:
            self._on_stop()
