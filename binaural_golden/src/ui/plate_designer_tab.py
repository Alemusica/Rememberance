"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PLATE DESIGNER TAB - MVVM Architecture                      ║
║                                                                              ║
║   Modern GUI for evolutionary plate optimization.                            ║
║   Uses MVVM pattern with separated ViewModel and reusable components.       ║
║                                                                              ║
║   Architecture:                                                               ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │  PlateDesignerTab (View)                                             │    ║
║   │  ├── PlateDesignerViewModel (Business Logic)                         │    ║
║   │  ├── EvolutionCanvas (Plate Visualization)                           │    ║
║   │  ├── GoldenProgressBar (Progress Indicator)                          │    ║
║   │  ├── FitnessRadarChart (Score Breakdown)                             │    ║
║   │  └── FitnessLineChart (Evolution History)                            │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from typing import Optional

# Core imports
from core.person import Person, PERSON_PRESETS
from core.plate_genome import PlateGenome, ContourType

# ViewModel
from ui.viewmodels.plate_designer_viewmodel import (
    PlateDesignerViewModel,
    PlateDesignerState,
    EvolutionPhase,
)

# Components
from ui.components.evolution_canvas import (
    EvolutionCanvas,
    GoldenProgressBar,
    FitnessRadarChart,
    FitnessLineChart,
)

# Theme
from ui.theme import STYLE, configure_ttk_style


class PlateDesignerTab(ttk.Frame):
    """
    Tab per design evolutivo tavola vibroacustica.
    
    Implements View layer of MVVM pattern.
    All state and business logic delegated to PlateDesignerViewModel.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  [PERSON CONFIG]      [EVOLUTION CONFIG]      [CONTROLS]               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                    EVOLUTION CANVAS                                      │
    │                 (plate + human overlay)                                  │
    │                                                                          │
    ├──────────────────────────────┬──────────────────────────────────────────┤
    │  [PROGRESS]                  │  [RADAR]  │  [LINE CHART]                │
    │  ████████████░░░ 75%         │           │                              │
    │  Gen: 38/50  Best: 0.847     │           │                              │
    └──────────────────────────────┴──────────────────────────────────────────┘
    """
    
    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        
        # ViewModel
        self._viewmodel = PlateDesignerViewModel()
        self._viewmodel.add_observer(self._on_state_changed)
        
        # Build UI
        self._create_widgets()
        self._layout_widgets()
        self._bind_events()
        
        # Initialize display
        self._update_from_state(self._viewmodel.state)
        
        # Start polling for background updates
        self._poll_viewmodel()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Widget Creation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_widgets(self):
        """Create all UI widgets."""
        
        # === TOP BAR: Configuration & Controls ===
        self._top_frame = ttk.Frame(self, style="TFrame")
        
        # --- Person Config Card ---
        self._person_frame = self._create_card_frame(self._top_frame, "👤 Person")
        
        # Height
        self._create_label(self._person_frame, "Height (m):").grid(
            row=0, column=0, padx=5, pady=3, sticky='e')
        self._height_var = tk.DoubleVar(value=1.75)
        self._height_spin = ttk.Spinbox(
            self._person_frame, from_=1.40, to=2.10, increment=0.01,
            textvariable=self._height_var, width=8, style="TSpinbox"
        )
        self._height_spin.grid(row=0, column=1, padx=5, pady=3)
        
        # Weight
        self._create_label(self._person_frame, "Weight (kg):").grid(
            row=1, column=0, padx=5, pady=3, sticky='e')
        self._weight_var = tk.DoubleVar(value=75.0)
        self._weight_spin = ttk.Spinbox(
            self._person_frame, from_=40.0, to=150.0, increment=1.0,
            textvariable=self._weight_var, width=8, style="TSpinbox"
        )
        self._weight_spin.grid(row=1, column=1, padx=5, pady=3)
        
        # Preset
        self._create_label(self._person_frame, "Preset:").grid(
            row=2, column=0, padx=5, pady=3, sticky='e')
        self._preset_var = tk.StringVar(value="average_male")
        self._preset_combo = ttk.Combobox(
            self._person_frame,
            textvariable=self._preset_var,
            values=list(PERSON_PRESETS.keys()),
            state="readonly",
            width=15
        )
        self._preset_combo.grid(row=2, column=1, padx=5, pady=3)
        
        # --- Evolution Config Card ---
        self._config_frame = self._create_card_frame(self._top_frame, "⚙️ Evolution")
        
        # Population
        self._create_label(self._config_frame, "Population:").grid(
            row=0, column=0, padx=5, pady=3, sticky='e')
        self._pop_var = tk.IntVar(value=30)
        self._pop_spin = ttk.Spinbox(
            self._config_frame, from_=10, to=100, increment=5,
            textvariable=self._pop_var, width=6
        )
        self._pop_spin.grid(row=0, column=1, padx=5, pady=3)
        
        # Generations
        self._create_label(self._config_frame, "Generations:").grid(
            row=1, column=0, padx=5, pady=3, sticky='e')
        self._gen_var = tk.IntVar(value=50)
        self._gen_spin = ttk.Spinbox(
            self._config_frame, from_=10, to=200, increment=10,
            textvariable=self._gen_var, width=6
        )
        self._gen_spin.grid(row=1, column=1, padx=5, pady=3)
        
        # Mutation Rate
        self._create_label(self._config_frame, "Mutation:").grid(
            row=2, column=0, padx=5, pady=3, sticky='e')
        self._mutation_var = tk.DoubleVar(value=0.3)
        self._mutation_spin = ttk.Spinbox(
            self._config_frame, from_=0.1, to=0.8, increment=0.05,
            textvariable=self._mutation_var, width=6
        )
        self._mutation_spin.grid(row=2, column=1, padx=5, pady=3)
        
        # Cutouts (LUTHERIE: fori per accordatura modi, come f-holes violino)
        # Reference: Schleske (2002) "Empirical Tools in Contemporary Violin Making"
        # NOTA: Nessun limite arbitrario! L'analisi strutturale durante evoluzione
        # valuta automaticamente se la configurazione cutouts è sicura (stress, deflection).
        self._cutouts_enabled_var = tk.BooleanVar(value=True)  # Default ON (liuteria)
        self._cutouts_var = tk.IntVar(value=4)  # Default 4 (2 per lato, come violino)
        self._cutouts_check = ttk.Checkbutton(
            self._config_frame, text="Cutouts (tuning):",
            variable=self._cutouts_enabled_var,
            command=self._on_cutouts_toggle
        )
        self._cutouts_check.grid(row=3, column=0, padx=5, pady=3, sticky='e')
        
        # Max cutouts: no arbitrary limit! Structural analysis validates during evolution.
        # "to=20" is practical maximum - real limit comes from structural integrity.
        self._cutouts_spin = ttk.Spinbox(
            self._config_frame, from_=1, to=20, increment=1,
            textvariable=self._cutouts_var, width=6, state="normal"  # Enabled by default
        )
        self._cutouts_spin.grid(row=3, column=1, padx=5, pady=3)
        
        # Tooltip explaining structural validation
        self._cutouts_info = ttk.Label(
            self._config_frame, 
            text="(structural check during evolution)",
            font=("SF Pro", 7), foreground="gray"
        )
        self._cutouts_info.grid(row=3, column=2, padx=2, pady=3, sticky='w')
        
        # Zone Weights Slider (Spine vs Head priority) - LEGACY: kept for compatibility
        self._create_label(self._config_frame, "Zone Priority:").grid(
            row=4, column=0, padx=5, pady=3, sticky='e')
        
        self._zone_frame = ttk.Frame(self._config_frame)
        self._zone_frame.grid(row=4, column=1, padx=5, pady=3, sticky='w')
        
        # Spine label
        ttk.Label(self._zone_frame, text="Spine", font=("SF Pro", 8)).pack(side='left')
        
        # Slider: 0 = 100% head, 100 = 100% spine (default 70)
        self._zone_weight_var = tk.IntVar(value=70)
        self._zone_slider = ttk.Scale(
            self._zone_frame, from_=0, to=100,
            variable=self._zone_weight_var,
            orient='horizontal', length=80,
            command=self._on_zone_weight_changed
        )
        self._zone_slider.pack(side='left', padx=2)
        
        # Head label
        ttk.Label(self._zone_frame, text="Head", font=("SF Pro", 8)).pack(side='left')
        
        # Percentage display
        self._zone_display = ttk.Label(
            self._zone_frame, text="70/30", font=("SF Pro", 8, "bold")
        )
        self._zone_display.pack(side='left', padx=5)
        
        # === CONTOUR TYPE SELECTOR ===
        # Row 5: Plate shape selection
        self._create_label(self._config_frame, "Contour:").grid(
            row=5, column=0, padx=5, pady=3, sticky='e')
        
        self._contour_var = tk.StringVar(value="ORGANIC")  # Default: organic curves
        self._contour_options = [
            "RECTANGLE",      # Fixed rectangle
            "GOLDEN_RECT",    # Golden ratio rectangle
            "ELLIPSE",        # Smooth ellipse
            "OVOID",          # Egg shape (narrower at one end)
            "SUPERELLIPSE",   # Squircle (rounded rectangle)
            "ORGANIC",        # Fourier-based blob (guitar-like)
            "ERGONOMIC",      # Body-conforming shape
            "FREEFORM",       # Fully evolvable spline
            "AUTO"            # Let evolution choose
        ]
        self._contour_combo = ttk.Combobox(
            self._config_frame, textvariable=self._contour_var,
            values=self._contour_options, state="readonly", width=12
        )
        self._contour_combo.grid(row=5, column=1, padx=5, pady=3)
        self._contour_combo.bind("<<ComboboxSelected>>", self._on_contour_changed)
        
        # Contour description
        self._contour_info = ttk.Label(
            self._config_frame, 
            text="smooth curves, CNC-friendly",
            font=("SF Pro", 7), foreground="gray"
        )
        self._contour_info.grid(row=5, column=2, padx=2, pady=3, sticky='w')
        
        # === ADVANCED: Radar Plot for multi-parameter optimization ===
        # Row 6: Radar plot toggle
        self._radar_enabled_var = tk.BooleanVar(value=False)
        self._radar_check = ttk.Checkbutton(
            self._config_frame, text="Advanced Tuning (Radar):",
            variable=self._radar_enabled_var,
            command=self._on_radar_toggle
        )
        self._radar_check.grid(row=6, column=0, padx=5, pady=3, sticky='e')
        
        # Radar widget (initially hidden)
        self._radar_frame = ttk.Frame(self._config_frame)
        self._radar_frame.grid(row=6, column=1, columnspan=2, padx=5, pady=3, sticky='w')
        
        try:
            from ui.widgets.radar_widget import RadarWidget
            self._radar_widget = RadarWidget(
                self._radar_frame,
                size=140,
                on_change=self._on_radar_changed
            )
            self._radar_widget.pack()
            self._radar_widget.pack_forget()  # Initially hidden
        except ImportError:
            self._radar_widget = None
            self._radar_check.config(state='disabled')
            ttk.Label(self._radar_frame, text="(Radar unavailable)", 
                      foreground='gray').pack()
        
        # --- Control Buttons ---
        self._control_frame = ttk.Frame(self._top_frame, style="TFrame")
        
        self._start_btn = ttk.Button(
            self._control_frame, text="▶ Start", 
            command=self._on_start, style="Primary.TButton"
        )
        self._stop_btn = ttk.Button(
            self._control_frame, text="⏹ Stop",
            command=self._on_stop, state="disabled"
        )
        self._reset_btn = ttk.Button(
            self._control_frame, text="↺ Reset",
            command=self._on_reset
        )
        self._export_btn = ttk.Button(
            self._control_frame, text="💾 Export",
            command=self._on_export
        )
        
        # === MAIN CANVAS ===
        self._canvas_frame = self._create_card_frame(self, "🔬 Plate Evolution")
        
        self._evolution_canvas = EvolutionCanvas(
            self._canvas_frame,
            width=700,
            height=380
        )
        
        # === BOTTOM BAR: Status & Charts ===
        self._bottom_frame = ttk.Frame(self, style="TFrame")
        
        # --- Progress Card ---
        self._progress_card = self._create_card_frame(self._bottom_frame, "📈 Progress")
        
        # Status label
        self._status_label = ttk.Label(
            self._progress_card, text="Ready to start",
            style="TLabel", font=STYLE.font_bold
        )
        
        # Golden progress bar
        self._progress_bar = GoldenProgressBar(
            self._progress_card, width=220, height=22
        )
        
        # Generation info
        self._gen_label = ttk.Label(
            self._progress_card, text="Generation: 0/0",
            style="Muted.TLabel"
        )
        
        # Best fitness
        self._best_label = ttk.Label(
            self._progress_card, text="Best: —",
            style="Value.TLabel"
        )
        
        # Time info
        self._time_label = ttk.Label(
            self._progress_card, text="Time: 0.0s",
            style="Muted.TLabel"
        )
        
        # Plate info
        self._plate_label = ttk.Label(
            self._progress_card, text="Plate: —",
            style="Muted.TLabel", wraplength=200
        )
        
        # --- Fitness Radar ---
        self._radar_frame = self._create_card_frame(self._bottom_frame, "🎯 Fitness")
        
        self._fitness_radar = FitnessRadarChart(
            self._radar_frame, size=140
        )
        
        # Fitness values
        self._fitness_values_frame = ttk.Frame(self._radar_frame, style="TFrame")
        
        self._flatness_label = self._create_fitness_row(
            self._fitness_values_frame, "Flatness", "#FF6B6B", 0
        )
        self._spine_label = self._create_fitness_row(
            self._fitness_values_frame, "Spine", "#4ECDC4", 1
        )
        self._mass_label = self._create_fitness_row(
            self._fitness_values_frame, "Mass", "#FFE66D", 2
        )
        self._total_label = self._create_fitness_row(
            self._fitness_values_frame, "TOTAL", STYLE.GOLD, 3, bold=True
        )
        
        # --- Evolution Chart ---
        self._chart_frame = self._create_card_frame(self._bottom_frame, "📊 Evolution")
        
        self._fitness_chart = FitnessLineChart(
            self._chart_frame, width=280, height=130
        )
    
    def _create_card_frame(self, parent, title: str) -> ttk.LabelFrame:
        """Create a styled card frame."""
        frame = ttk.LabelFrame(parent, text=title, style="TLabelframe")
        return frame
    
    def _create_label(self, parent, text: str) -> ttk.Label:
        """Create a styled label."""
        return ttk.Label(parent, text=text, style="TLabel")
    
    def _create_fitness_row(
        self, parent, name: str, color: str, row: int, bold: bool = False
    ) -> ttk.Label:
        """Create a fitness score row."""
        # Color indicator
        indicator = tk.Canvas(
            parent, width=10, height=10,
            bg=STYLE.BG_DARK, highlightthickness=0
        )
        indicator.create_oval(1, 1, 9, 9, fill=color, outline='')
        indicator.grid(row=row, column=0, padx=3, pady=2)
        
        # Name
        ttk.Label(
            parent, text=f"{name}:",
            style="Muted.TLabel"
        ).grid(row=row, column=1, padx=3, pady=2, sticky='w')
        
        # Value
        style = "Value.TLabel" if not bold else "Heading.TLabel"
        value_label = ttk.Label(parent, text="—", style=style)
        value_label.grid(row=row, column=2, padx=3, pady=2, sticky='e')
        
        return value_label
    
    def _layout_widgets(self):
        """Layout all widgets."""
        
        # Top frame
        self._top_frame.pack(fill='x', padx=10, pady=5)
        
        self._person_frame.pack(side='left', padx=5, pady=5)
        self._config_frame.pack(side='left', padx=10, pady=5)
        self._control_frame.pack(side='left', padx=15, pady=5)
        
        # Control buttons
        self._start_btn.pack(side='left', padx=3, pady=3)
        self._stop_btn.pack(side='left', padx=3, pady=3)
        self._reset_btn.pack(side='left', padx=3, pady=3)
        self._export_btn.pack(side='left', padx=3, pady=3)
        
        # Canvas frame
        self._canvas_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self._evolution_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bottom frame
        self._bottom_frame.pack(fill='x', padx=10, pady=5)
        
        # Progress card
        self._progress_card.pack(side='left', padx=5, pady=5, fill='y')
        
        self._status_label.pack(anchor='w', padx=10, pady=3)
        self._progress_bar.pack(padx=10, pady=5)
        self._gen_label.pack(anchor='w', padx=10, pady=2)
        self._best_label.pack(anchor='w', padx=10, pady=2)
        self._time_label.pack(anchor='w', padx=10, pady=2)
        self._plate_label.pack(anchor='w', padx=10, pady=2)
        
        # Radar frame
        self._radar_frame.pack(side='left', padx=10, pady=5, fill='y')
        self._fitness_radar.pack(padx=5, pady=5)
        self._fitness_values_frame.pack(padx=5, pady=5)
        
        # Chart frame
        self._chart_frame.pack(side='left', padx=10, pady=5, fill='both', expand=True)
        self._fitness_chart.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _bind_events(self):
        """Bind event handlers."""
        self._preset_combo.bind('<<ComboboxSelected>>', self._on_preset_changed)
        self._height_spin.bind('<FocusOut>', self._on_person_changed)
        self._weight_spin.bind('<FocusOut>', self._on_person_changed)
        
        # Config changes
        self._pop_spin.bind('<FocusOut>', self._on_config_changed)
        self._gen_spin.bind('<FocusOut>', self._on_config_changed)
        self._mutation_spin.bind('<FocusOut>', self._on_config_changed)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_zone_weight_changed(self, value=None):
        """Handle zone weight slider change."""
        spine_pct = self._zone_weight_var.get()
        head_pct = 100 - spine_pct
        
        # Update display label
        self._zone_display.config(text=f"{spine_pct}/{head_pct}")
        
        # Update viewmodel (accepts spine percentage only)
        self._viewmodel.set_zone_weights(spine_pct)
    
    def _on_contour_changed(self, event=None):
        """Handle contour type selection change."""
        contour_name = self._contour_var.get()
        
        # Update info label with description
        descriptions = {
            "RECTANGLE": "fixed rectangular shape",
            "GOLDEN_RECT": "φ ratio rectangle (1:1.618)",
            "ELLIPSE": "smooth elliptical shape",
            "OVOID": "egg shape, narrow at top",
            "SUPERELLIPSE": "squircle (rounded corners)",
            "ORGANIC": "smooth curves, CNC-friendly",
            "ERGONOMIC": "body-conforming shape",
            "FREEFORM": "fully evolvable spline",
            "AUTO": "let evolution choose shape"
        }
        self._contour_info.config(text=descriptions.get(contour_name, ""))
        
        # Update viewmodel
        self._viewmodel.set_contour_type(contour_name)
    
    def _on_radar_toggle(self):
        """Toggle radar widget visibility."""
        if self._radar_widget is None:
            return
        
        if self._radar_enabled_var.get():
            # Show radar, hide simple slider
            self._zone_frame.grid_remove()
            self._radar_widget.pack()
        else:
            # Hide radar, show simple slider
            self._radar_widget.pack_forget()
            self._zone_frame.grid()
    
    def _on_radar_changed(self, values: dict):
        """
        Handle radar parameter changes.
        
        Maps radar values to optimization weights:
        - Energy: Weight for vibration intensity
        - Flatness: Weight for uniform frequency response
        - Spine: Weight for spine coverage
        """
        # Update viewmodel with multi-parameter weights
        if hasattr(self._viewmodel, 'set_optimization_weights'):
            self._viewmodel.set_optimization_weights(
                energy=values.get('Energy', 0.7),
                flatness=values.get('Flatness', 0.5),
                spine=values.get('Spine', 0.8)
            )
        else:
            # Fallback: Map to single spine weight
            # Combined weight: higher spine = more spine focus
            combined = int(values.get('Spine', 0.8) * 100)
            self._zone_weight_var.set(combined)
            self._viewmodel.set_zone_weights(combined)
    
    def _on_preset_changed(self, event=None):
        """Handle preset selection."""
        preset_name = self._preset_var.get()
        if preset_name in PERSON_PRESETS:
            preset = PERSON_PRESETS[preset_name]
            self._height_var.set(preset.height_m)
            self._weight_var.set(preset.weight_kg)
            self._viewmodel.set_person_preset(preset_name)
    
    def _on_person_changed(self, event=None):
        """Handle person dimension change."""
        try:
            height = self._height_var.get()
            weight = self._weight_var.get()
            self._viewmodel.set_person(height, weight)
        except tk.TclError:
            pass  # Invalid value
    
    def _on_config_changed(self, event=None):
        """Handle config change."""
        try:
            max_cutouts = self._cutouts_var.get() if self._cutouts_enabled_var.get() else 0
            self._viewmodel.set_evolution_config(
                population_size=self._pop_var.get(),
                max_generations=self._gen_var.get(),
                mutation_rate=self._mutation_var.get(),
                max_cutouts=max_cutouts
            )
        except tk.TclError:
            pass
    
    def _on_cutouts_toggle(self):
        """Toggle cutouts enable/disable (liuteria mode)."""
        enabled = self._cutouts_enabled_var.get()
        self._cutouts_spin.config(state="normal" if enabled else "disabled")
        if enabled and self._cutouts_var.get() < 1:
            self._cutouts_var.set(4)  # Default 4 cutouts (like f-holes: 2 per side)
        self._on_config_changed()
    
    def _on_start(self):
        """Start evolution."""
        # Update config from UI
        self._on_config_changed()
        self._on_person_changed()
        
        # Clear chart
        self._fitness_chart.clear()
        
        # Start
        self._viewmodel.start_evolution()
        
        # Update button states
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_inputs_enabled(False)
    
    def _on_stop(self):
        """Stop evolution."""
        self._viewmodel.stop_evolution()
        
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._set_inputs_enabled(True)
    
    def _on_reset(self):
        """Reset state."""
        self._viewmodel.reset()
        self._fitness_chart.clear()
        
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._set_inputs_enabled(True)
    
    def _on_export(self):
        """Export options: Genome JSON, DSP Data, or STL/CNC files."""
        from tkinter import simpledialog
        
        # Create export dialog with multiple options
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Export Options")
        dialog.geometry("300x200")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        
        result = {"choice": None}
        
        ttk.Label(dialog, text="Select Export Type:", font=("SF Pro", 12, "bold")).pack(pady=10)
        
        def select(choice):
            result["choice"] = choice
            dialog.destroy()
        
        ttk.Button(dialog, text="📊 DSP Data (for audio processing)", 
                   command=lambda: select("dsp")).pack(pady=5, padx=20, fill='x')
        ttk.Button(dialog, text="📐 STL/OBJ/DXF (for CNC fabrication)", 
                   command=lambda: select("cnc")).pack(pady=5, padx=20, fill='x')
        ttk.Button(dialog, text="📋 Genome JSON (basic design)", 
                   command=lambda: select("json")).pack(pady=5, padx=20, fill='x')
        ttk.Button(dialog, text="Cancel", 
                   command=dialog.destroy).pack(pady=10)
        
        dialog.wait_window()
        
        if result["choice"] == "dsp":
            self._export_dsp_data()
        elif result["choice"] == "cnc":
            self._export_cnc_files()
        elif result["choice"] == "json":
            self._export_genome_json()
    
    def _export_cnc_files(self):
        """Export STL/OBJ/DXF files for CNC fabrication."""
        genome = self._viewmodel.state.best_genome
        if genome is None:
            messagebox.showwarning("Export", "No plate design to export.\nRun evolution first.")
            return
        
        # Ask for base filename
        filename = filedialog.asksaveasfilename(
            defaultextension="",
            filetypes=[("All files", "*.*")],
            initialfile="plate_cnc_export",
            title="Export CNC Files (will create .stl, .obj, .dxf)"
        )
        
        if not filename:
            return
        
        # Remove any extension
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        
        try:
            from core.stl_export import export_plate_for_cnc
            exports = export_plate_for_cnc(genome, filename)
            
            # Show summary
            files_list = "\n".join([f"  • {fmt.upper()}: {path}" for fmt, path in exports.items()])
            messagebox.showinfo(
                "CNC Export Complete",
                f"Exported {len(exports)} files:\n\n{files_list}\n\n"
                f"Manufacturing notes included for CNC shop."
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CNC files:\n{str(e)}")
    
    def _export_genome_json(self):
        """Export basic genome JSON."""
        data = self._viewmodel.export_genome_json()
        if data is None:
            messagebox.showwarning("Export", "No plate design to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Plate Design"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Export", f"Design saved to {filename}")
    
    def _export_dsp_data(self):
        """
        Export comprehensive DSP data for DSP agent.
        
        Includes:
        - Modal frequencies and resonances
        - Transfer functions
        - Material properties (orthotropic)
        - Exciter data per channel
        - Zone-specific recommendations
        - Suggested EQ curve
        """
        data = self._viewmodel.export_for_dsp()
        if data is None:
            # Debug info
            state = self._viewmodel.state
            debug_info = (
                f"genome: {'Yes' if state.best_genome else 'No'}, "
                f"fitness: {'Yes' if state.best_fitness else 'No'}, "
                f"phase: {state.phase.value}"
            )
            messagebox.showwarning("Export", f"No simulation data to export.\nRun evolution first.\n\nDebug: {debug_info}")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("DSP JSON files", "*.dsp.json"), ("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="plate_dsp_export.dsp.json",
            title="Export DSP Data for Audio Agent"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Show summary
            summary = self._format_dsp_summary(data)
            messagebox.showinfo(
                "DSP Export Complete", 
                f"DSP data saved to:\n{filename}\n\n{summary}"
            )
    
    def _format_dsp_summary(self, data: dict) -> str:
        """Format DSP export summary for display."""
        lines = []
        
        # Material
        if 'material' in data and data['material']:
            mat = data['material']
            lines.append(f"Material: {mat.get('name', 'Unknown')}")
            if mat.get('resonance_character'):
                lines.append(f"  Character: {mat['resonance_character']}")
        
        # Modes
        if 'mode_frequencies_hz' in data:
            modes = data['mode_frequencies_hz'][:3]
            modes_str = ", ".join([f"{f:.1f}Hz" for f in modes])
            lines.append(f"First modes: {modes_str}")
        
        # Recommendations
        if 'global_recommendations' in data:
            n_recs = len(data['global_recommendations'])
            lines.append(f"DSP recommendations: {n_recs}")
        
        return "\n".join(lines) if lines else "Data exported successfully"
    
    def _set_inputs_enabled(self, enabled: bool):
        """Enable/disable input widgets during evolution."""
        state = "normal" if enabled else "disabled"
        self._height_spin.config(state=state)
        self._weight_spin.config(state=state)
        self._preset_combo.config(state="readonly" if enabled else "disabled")
        self._pop_spin.config(state=state)
        self._gen_spin.config(state=state)
        self._mutation_spin.config(state=state)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ViewModel Observer
    # ─────────────────────────────────────────────────────────────────────────
    
    def _poll_viewmodel(self):
        """Poll ViewModel for background updates."""
        self._viewmodel.poll_updates()
        
        # Check if evolution finished
        state = self._viewmodel.state
        if not state.is_running and state.phase in [
            EvolutionPhase.CONVERGED, EvolutionPhase.STOPPED, EvolutionPhase.ERROR
        ]:
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._set_inputs_enabled(True)
        
        # Schedule next poll
        self.after(50, self._poll_viewmodel)
    
    def _on_state_changed(self, state: PlateDesignerState):
        """Called when ViewModel state changes."""
        # Schedule UI update on main thread
        self.after(0, lambda: self._update_from_state(state))
    
    def _update_from_state(self, state: PlateDesignerState):
        """Update all UI from state."""
        
        # Status
        self._status_label.config(text=state.status_text)
        
        # Progress
        self._progress_bar.set_value(state.progress_percent)
        self._gen_label.config(text=f"Generation: {state.generation}/{state.max_generations}")
        self._best_label.config(text=f"Best: {state.best_fitness_value:.4f}")
        self._time_label.config(text=f"Time: {state.elapsed_time:.1f}s")
        self._plate_label.config(text=f"Plate: {state.plate_info_text}")
        
        # Fitness scores
        if state.best_fitness:
            f = state.best_fitness
            self._flatness_label.config(text=f"{f.flatness_score * 100:.0f}%")
            self._spine_label.config(text=f"{f.spine_coupling_score * 100:.0f}%")
            self._mass_label.config(text=f"{f.low_mass_score * 100:.0f}%")
            self._total_label.config(text=f"{f.total_fitness:.3f}")
            
            # Radar chart
            self._fitness_radar.set_scores(
                flatness=f.flatness_score,
                spine=f.spine_coupling_score,
                mass=f.low_mass_score,
                edge=getattr(f, 'edge_support_score', 0.5)
            )
        else:
            self._flatness_label.config(text="—")
            self._spine_label.config(text="—")
            self._mass_label.config(text="—")
            self._total_label.config(text="—")
            self._fitness_radar.set_scores()
        
        # Evolution canvas
        self._evolution_canvas.set_person(state.person)
        if state.best_genome:
            self._evolution_canvas.set_plate(
                state.best_genome, 
                state.best_fitness,
                animate=True
            )
        else:
            # Show default plate
            if state.person:
                default_genome = PlateGenome(
                    length=state.person.recommended_plate_length,
                    width=state.person.recommended_plate_width,
                    contour_type=ContourType.GOLDEN_RECT,
                )
                self._evolution_canvas.set_plate(default_genome, animate=False)
        
        # Line chart - add new point if we have history
        if state.fitness_history:
            latest = state.fitness_history[-1]
            # Only add if it's a new generation
            if len(self._fitness_chart._data.get('total', [])) < len(state.fitness_history):
                self._fitness_chart.add_point(
                    total=latest.total,
                    flatness=latest.flatness,
                    spine=latest.spine_coupling,
                    mass=latest.low_mass
                )


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Plate Designer - MVVM Architecture")
    root.geometry("1100x750")
    root.configure(bg=STYLE.BG_DARK)
    
    # Configure ttk styles
    style = ttk.Style()
    configure_ttk_style(style)
    
    # Create tab
    tab = PlateDesignerTab(root)
    tab.pack(fill='both', expand=True)
    
    root.mainloop()
