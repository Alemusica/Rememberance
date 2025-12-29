"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PLATE DESIGNER TAB - Evolutionary Optimization UI          ║
║                                                                              ║
║   GUI per ottimizzazione evolutiva forma tavola vibroacustica.               ║
║   Mostra evoluzione in tempo reale della forma della tavola.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
from typing import Optional, Callable
import queue

# Core imports
from core.person import Person, PERSON_PRESETS, SPINE_ZONES
from core.plate_genome import PlateGenome, ContourType
from core.fitness import FitnessEvaluator, FitnessResult, ObjectiveWeights
from core.evolutionary_optimizer import (
    EvolutionaryOptimizer, 
    EvolutionConfig, 
    EvolutionState,
    SelectionMethod,
)

# Theme
from ui.golden_theme import Colors


class PlateDesignerTab(ttk.Frame):
    """
    Tab per design evolutivo tavola vibroacustica.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │  PERSON INPUT          │  EVOLUTION CONFIG                     │
    │  [Height] [Weight]     │  [Population] [Generations]           │
    │  [Preset ▼]            │  [Start] [Stop] [Reset]               │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │              PLATE VISUALIZATION CANVAS                         │
    │           (shape evolving in real-time)                         │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  FITNESS SCORES        │  PROGRESS                              │
    │  Flatness: ████░░ 67%  │  Gen: 12/50                           │
    │  Spine:    █████░ 85%  │  Best: 0.723                          │
    │  Mass:     ██████ 100% │  [Progress Bar]                       │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # State
        self._person: Optional[Person] = None
        self._optimizer: Optional[EvolutionaryOptimizer] = None
        self._is_running: bool = False
        self._evolution_thread: Optional[threading.Thread] = None
        self._update_queue: queue.Queue = queue.Queue()
        self._best_genome: Optional[PlateGenome] = None
        self._best_fitness: Optional[FitnessResult] = None
        
        # Build UI
        self._create_widgets()
        self._layout_widgets()
        self._bind_events()
        
        # Initial state
        self._on_preset_selected(None)
        
        # Start UI update loop
        self._poll_updates()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Widget Creation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_widgets(self):
        """Crea tutti i widget."""
        
        # === TOP FRAME: Input ===
        self.top_frame = ttk.Frame(self)
        
        # Person frame
        self.person_frame = ttk.LabelFrame(self.top_frame, text="👤 Person")
        
        # Height
        ttk.Label(self.person_frame, text="Height (m):").grid(row=0, column=0, padx=5, pady=2)
        self.height_var = tk.DoubleVar(value=1.75)
        self.height_spin = ttk.Spinbox(
            self.person_frame, from_=1.40, to=2.10, increment=0.01,
            textvariable=self.height_var, width=8
        )
        self.height_spin.grid(row=0, column=1, padx=5, pady=2)
        
        # Weight
        ttk.Label(self.person_frame, text="Weight (kg):").grid(row=1, column=0, padx=5, pady=2)
        self.weight_var = tk.DoubleVar(value=75.0)
        self.weight_spin = ttk.Spinbox(
            self.person_frame, from_=40.0, to=150.0, increment=1.0,
            textvariable=self.weight_var, width=8
        )
        self.weight_spin.grid(row=1, column=1, padx=5, pady=2)
        
        # Preset
        ttk.Label(self.person_frame, text="Preset:").grid(row=2, column=0, padx=5, pady=2)
        self.preset_var = tk.StringVar(value="average_male")
        self.preset_combo = ttk.Combobox(
            self.person_frame,
            textvariable=self.preset_var,
            values=list(PERSON_PRESETS.keys()),
            state="readonly",
            width=15
        )
        self.preset_combo.grid(row=2, column=1, padx=5, pady=2)
        
        # Evolution config frame
        self.config_frame = ttk.LabelFrame(self.top_frame, text="⚙️ Evolution Config")
        
        # Population size
        ttk.Label(self.config_frame, text="Population:").grid(row=0, column=0, padx=5, pady=2)
        self.pop_var = tk.IntVar(value=30)
        self.pop_spin = ttk.Spinbox(
            self.config_frame, from_=10, to=100, increment=5,
            textvariable=self.pop_var, width=6
        )
        self.pop_spin.grid(row=0, column=1, padx=5, pady=2)
        
        # Generations
        ttk.Label(self.config_frame, text="Generations:").grid(row=1, column=0, padx=5, pady=2)
        self.gen_var = tk.IntVar(value=50)
        self.gen_spin = ttk.Spinbox(
            self.config_frame, from_=10, to=200, increment=10,
            textvariable=self.gen_var, width=6
        )
        self.gen_spin.grid(row=1, column=1, padx=5, pady=2)
        
        # Mutation rate
        ttk.Label(self.config_frame, text="Mutation:").grid(row=2, column=0, padx=5, pady=2)
        self.mutation_var = tk.DoubleVar(value=0.3)
        self.mutation_spin = ttk.Spinbox(
            self.config_frame, from_=0.1, to=0.8, increment=0.05,
            textvariable=self.mutation_var, width=6
        )
        self.mutation_spin.grid(row=2, column=1, padx=5, pady=2)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.top_frame)
        
        self.start_btn = ttk.Button(
            self.control_frame, text="▶ Start", command=self._on_start
        )
        self.stop_btn = ttk.Button(
            self.control_frame, text="⏹ Stop", command=self._on_stop, state="disabled"
        )
        self.reset_btn = ttk.Button(
            self.control_frame, text="↺ Reset", command=self._on_reset
        )
        
        # === MIDDLE: Canvas ===
        self.canvas_frame = ttk.LabelFrame(self, text="🔬 Plate Evolution")
        
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=700,
            height=400,
            bg=Colors.BG_DEEPEST,
            highlightthickness=0
        )
        
        # === BOTTOM FRAME: Status ===
        self.bottom_frame = ttk.Frame(self)
        
        # Fitness scores frame
        self.fitness_frame = ttk.LabelFrame(self.bottom_frame, text="📊 Fitness Scores")
        
        self.flatness_label = ttk.Label(self.fitness_frame, text="Flatness:")
        self.flatness_bar = ttk.Progressbar(self.fitness_frame, length=120, mode='determinate')
        self.flatness_value = ttk.Label(self.fitness_frame, text="0%")
        
        self.spine_label = ttk.Label(self.fitness_frame, text="Spine:")
        self.spine_bar = ttk.Progressbar(self.fitness_frame, length=120, mode='determinate')
        self.spine_value = ttk.Label(self.fitness_frame, text="0%")
        
        self.mass_label = ttk.Label(self.fitness_frame, text="Mass:")
        self.mass_bar = ttk.Progressbar(self.fitness_frame, length=120, mode='determinate')
        self.mass_value = ttk.Label(self.fitness_frame, text="0%")
        
        self.total_label = ttk.Label(self.fitness_frame, text="TOTAL:", font=('Helvetica', 10, 'bold'))
        self.total_value = ttk.Label(self.fitness_frame, text="0.000", font=('Helvetica', 12, 'bold'))
        
        # Progress frame
        self.progress_frame = ttk.LabelFrame(self.bottom_frame, text="📈 Progress")
        
        self.gen_label = ttk.Label(self.progress_frame, text="Generation: 0/0")
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=200, mode='determinate')
        self.best_label = ttk.Label(self.progress_frame, text="Best: -")
        self.time_label = ttk.Label(self.progress_frame, text="Time: 0.0s")
        
        # Genome info
        self.genome_label = ttk.Label(
            self.progress_frame, 
            text="Plate: -",
            font=('Helvetica', 9)
        )
    
    def _layout_widgets(self):
        """Posiziona widgets."""
        
        # Top frame
        self.top_frame.pack(fill='x', padx=10, pady=5)
        
        self.person_frame.pack(side='left', padx=5, pady=5)
        self.config_frame.pack(side='left', padx=15, pady=5)
        self.control_frame.pack(side='left', padx=15, pady=5)
        
        self.start_btn.pack(side='left', padx=3)
        self.stop_btn.pack(side='left', padx=3)
        self.reset_btn.pack(side='left', padx=3)
        
        # Canvas
        self.canvas_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bottom frame
        self.bottom_frame.pack(fill='x', padx=10, pady=5)
        
        # Fitness scores layout
        self.fitness_frame.pack(side='left', padx=5, pady=5)
        
        self.flatness_label.grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.flatness_bar.grid(row=0, column=1, padx=5, pady=2)
        self.flatness_value.grid(row=0, column=2, padx=5, pady=2)
        
        self.spine_label.grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.spine_bar.grid(row=1, column=1, padx=5, pady=2)
        self.spine_value.grid(row=1, column=2, padx=5, pady=2)
        
        self.mass_label.grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.mass_bar.grid(row=2, column=1, padx=5, pady=2)
        self.mass_value.grid(row=2, column=2, padx=5, pady=2)
        
        self.total_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.total_value.grid(row=3, column=1, columnspan=2, padx=5, pady=5)
        
        # Progress layout
        self.progress_frame.pack(side='left', padx=15, pady=5)
        
        self.gen_label.pack(anchor='w', padx=5, pady=2)
        self.progress_bar.pack(padx=5, pady=2)
        self.best_label.pack(anchor='w', padx=5, pady=2)
        self.time_label.pack(anchor='w', padx=5, pady=2)
        self.genome_label.pack(anchor='w', padx=5, pady=2)
    
    def _bind_events(self):
        """Collega eventi."""
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_selected)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_preset_selected(self, event):
        """Preset persona selezionato."""
        preset_name = self.preset_var.get()
        if preset_name in PERSON_PRESETS:
            preset = PERSON_PRESETS[preset_name]
            self.height_var.set(preset.height_m)
            self.weight_var.set(preset.weight_kg)
            self._update_person()
    
    def _update_person(self):
        """Aggiorna modello persona."""
        self._person = Person(
            height_m=self.height_var.get(),
            weight_kg=self.weight_var.get()
        )
        self._draw_initial_plate()
    
    def _on_start(self):
        """Avvia evoluzione."""
        if self._is_running:
            return
        
        self._is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        # Update person
        self._update_person()
        
        # Config
        config = EvolutionConfig(
            population_size=self.pop_var.get(),
            n_generations=self.gen_var.get(),
            mutation_rate=self.mutation_var.get(),
        )
        
        # Create optimizer
        self._optimizer = EvolutionaryOptimizer(
            person=self._person,
            config=config,
        )
        
        # Run in thread
        self._evolution_thread = threading.Thread(
            target=self._run_evolution,
            daemon=True
        )
        self._evolution_thread.start()
    
    def _on_stop(self):
        """Ferma evoluzione."""
        self._is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def _on_reset(self):
        """Reset stato."""
        self._on_stop()
        self._best_genome = None
        self._best_fitness = None
        self._update_fitness_display(None)
        self._draw_initial_plate()
        self.gen_label.config(text="Generation: 0/0")
        self.progress_bar['value'] = 0
        self.best_label.config(text="Best: -")
        self.time_label.config(text="Time: 0.0s")
        self.genome_label.config(text="Plate: -")
    
    def _on_canvas_resize(self, event):
        """Canvas ridimensionato."""
        if self._best_genome:
            self._draw_plate(self._best_genome, self._best_fitness)
        else:
            self._draw_initial_plate()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evolution Thread
    # ─────────────────────────────────────────────────────────────────────────
    
    def _run_evolution(self):
        """Esegue evoluzione in thread separato."""
        def callback(state: EvolutionState) -> bool:
            # Queue update for main thread
            self._update_queue.put(state)
            return self._is_running
        
        try:
            best = self._optimizer.run(callback=callback, verbose=False)
        except Exception as e:
            print(f"Evolution error: {e}")
        finally:
            self._is_running = False
            # Schedule UI update
            self.after(0, lambda: self.start_btn.config(state="normal"))
            self.after(0, lambda: self.stop_btn.config(state="disabled"))
    
    def _poll_updates(self):
        """Poll for updates from evolution thread."""
        try:
            while True:
                state = self._update_queue.get_nowait()
                self._handle_evolution_update(state)
        except queue.Empty:
            pass
        
        # Schedule next poll
        self.after(50, self._poll_updates)
    
    def _handle_evolution_update(self, state: EvolutionState):
        """Gestisce update dall'evoluzione."""
        self._best_genome = state.best_genome
        self._best_fitness = state.best_fitness
        
        # Update progress
        max_gen = self.gen_var.get()
        self.gen_label.config(text=f"Generation: {state.generation}/{max_gen}")
        self.progress_bar['value'] = (state.generation / max_gen) * 100
        self.best_label.config(text=f"Best: {state.best_fitness.total_fitness:.4f}")
        self.time_label.config(text=f"Time: {state.elapsed_time:.1f}s")
        
        # Update genome info
        g = state.best_genome
        self.genome_label.config(
            text=f"Plate: {g.contour_type.value} {g.length:.2f}×{g.width:.2f}m, "
                 f"{g.thickness_base*1000:.1f}mm"
        )
        
        # Update fitness display
        self._update_fitness_display(state.best_fitness)
        
        # Update visualization
        self._draw_plate(state.best_genome, state.best_fitness)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_fitness_display(self, fitness: Optional[FitnessResult]):
        """Aggiorna display fitness scores."""
        if fitness is None:
            self.flatness_bar['value'] = 0
            self.flatness_value.config(text="0%")
            self.spine_bar['value'] = 0
            self.spine_value.config(text="0%")
            self.mass_bar['value'] = 0
            self.mass_value.config(text="0%")
            self.total_value.config(text="0.000")
            return
        
        # Flatness
        flat_pct = fitness.flatness_score * 100
        self.flatness_bar['value'] = flat_pct
        self.flatness_value.config(text=f"{flat_pct:.0f}%")
        
        # Spine
        spine_pct = fitness.spine_coupling_score * 100
        self.spine_bar['value'] = spine_pct
        self.spine_value.config(text=f"{spine_pct:.0f}%")
        
        # Mass
        mass_pct = fitness.low_mass_score * 100
        self.mass_bar['value'] = mass_pct
        self.mass_value.config(text=f"{mass_pct:.0f}%")
        
        # Total
        self.total_value.config(text=f"{fitness.total_fitness:.3f}")
    
    def _draw_initial_plate(self):
        """Disegna tavola iniziale (rettangolo base)."""
        if self._person is None:
            self._update_person()
        
        initial_genome = PlateGenome(
            length=self._person.recommended_plate_length,
            width=self._person.recommended_plate_width,
            contour_type=ContourType.RECTANGLE,
        )
        self._draw_plate(initial_genome, None)
    
    def _draw_plate(self, genome: PlateGenome, fitness: Optional[FitnessResult]):
        """Disegna tavola e persona sul canvas."""
        self.canvas.delete("all")
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            return
        
        # Margini e scala
        margin = 40
        plate_length = genome.length
        plate_width = genome.width
        
        # Scala per fit nel canvas (tavola orizzontale)
        scale_x = (cw - 2 * margin) / plate_length
        scale_y = (ch - 2 * margin) / plate_width
        scale = min(scale_x, scale_y)
        
        # Centro canvas
        cx = cw / 2
        cy = ch / 2
        
        # Coordinate tavola
        plate_w = plate_length * scale
        plate_h = plate_width * scale
        x0 = cx - plate_w / 2
        y0 = cy - plate_h / 2
        x1 = cx + plate_w / 2
        y1 = cy + plate_h / 2
        
        # Disegna tavola in base al contour type
        self._draw_plate_shape(genome, x0, y0, x1, y1, scale)
        
        # Disegna silhouette persona
        self._draw_person_silhouette(x0, y0, plate_w, plate_h)
        
        # Disegna zone spina
        self._draw_spine_zones(x0, y0, plate_w, plate_h)
        
        # Disegna dimensioni
        self._draw_dimensions(genome, x0, y0, x1, y1)
        
        # Disegna frequenze modi (se disponibili)
        if fitness and fitness.frequencies:
            self._draw_mode_frequencies(fitness.frequencies, x1, y0)
    
    def _draw_plate_shape(self, genome: PlateGenome, x0, y0, x1, y1, scale):
        """Disegna forma tavola."""
        ct = genome.contour_type
        
        fill_color = Colors.GOLD
        outline_color = Colors.TEXT_PRIMARY
        
        if ct in [ContourType.RECTANGLE, ContourType.GOLDEN_RECT]:
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=fill_color, outline=outline_color, width=2
            )
        
        elif ct == ContourType.ELLIPSE:
            self.canvas.create_oval(
                x0, y0, x1, y1,
                fill=fill_color, outline=outline_color, width=2
            )
        
        elif ct == ContourType.OVOID:
            # Ovoide: ellisse con un lato più stretto
            # Approssimazione con poligono
            points = []
            n_points = 50
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            
            for i in range(n_points):
                theta = 2 * np.pi * i / n_points
                # Modulazione per forma ovoide (più stretta a sinistra)
                r_mod = 1 - 0.1 * np.cos(theta)
                x = cx + rx * np.cos(theta) * r_mod
                y = cy + ry * np.sin(theta)
                points.extend([x, y])
            
            self.canvas.create_polygon(
                points,
                fill=fill_color, outline=outline_color, width=2, smooth=True
            )
        
        else:
            # Default: rettangolo
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=fill_color, outline=outline_color, width=2
            )
        
        # Disegna cutouts se presenti
        for cutout in genome.cutouts:
            cut_cx = x0 + cutout.center_x * (x1 - x0)
            cut_cy = y0 + cutout.center_y * (y1 - y0)
            cut_rx = cutout.radius_x * (x1 - x0) / 2
            cut_ry = cutout.radius_y * (y1 - y0) / 2
            
            self.canvas.create_oval(
                cut_cx - cut_rx, cut_cy - cut_ry,
                cut_cx + cut_rx, cut_cy + cut_ry,
                fill=Colors.BG_DEEPEST, outline=outline_color, width=1
            )
    
    def _draw_person_silhouette(self, x0, y0, plate_w, plate_h):
        """Disegna silhouette persona sdraiata."""
        if self._person is None:
            return
        
        # Proporzioni corpo
        head_frac = 0.12
        torso_frac = 0.35
        legs_frac = 0.53
        
        # Colori
        body_color = Colors.TEXT_TERTIARY
        
        # Testa (cerchio a destra)
        head_x = x0 + plate_w * (1 - head_frac / 2)
        head_y = y0 + plate_h / 2
        head_r = plate_h * 0.08
        
        self.canvas.create_oval(
            head_x - head_r, head_y - head_r,
            head_x + head_r, head_y + head_r,
            outline=body_color, width=2
        )
        
        # Torso (rettangolo)
        torso_x0 = x0 + plate_w * legs_frac
        torso_x1 = x0 + plate_w * (1 - head_frac)
        torso_y0 = y0 + plate_h * 0.35
        torso_y1 = y0 + plate_h * 0.65
        
        self.canvas.create_rectangle(
            torso_x0, torso_y0, torso_x1, torso_y1,
            outline=body_color, width=2
        )
        
        # Gambe (2 rettangoli)
        leg_width = plate_h * 0.12
        leg_x0 = x0 + plate_w * 0.05
        leg_x1 = torso_x0
        
        # Gamba sinistra
        self.canvas.create_rectangle(
            leg_x0, y0 + plate_h * 0.3,
            leg_x1, y0 + plate_h * 0.3 + leg_width,
            outline=body_color, width=2
        )
        
        # Gamba destra
        self.canvas.create_rectangle(
            leg_x0, y0 + plate_h * 0.7 - leg_width,
            leg_x1, y0 + plate_h * 0.7,
            outline=body_color, width=2
        )
        
        # Label orientamento
        self.canvas.create_text(
            x0 + 20, y0 + plate_h / 2,
            text="FEET", fill=body_color, font=('Helvetica', 8), anchor='w'
        )
        self.canvas.create_text(
            x0 + plate_w - 20, y0 + plate_h / 2,
            text="HEAD", fill=body_color, font=('Helvetica', 8), anchor='e'
        )
    
    def _draw_spine_zones(self, x0, y0, plate_w, plate_h):
        """Disegna zone della spina dorsale."""
        spine_y = y0 + plate_h / 2
        
        colors = {
            'lumbar': '#FF6B6B',      # Rosso
            'thoracic': '#4ECDC4',    # Teal
            'cervical': '#95E1D3',    # Verde chiaro
        }
        
        for zone_name, (start, end) in SPINE_ZONES.items():
            zone_x0 = x0 + plate_w * start
            zone_x1 = x0 + plate_w * end
            
            # Linea zona
            self.canvas.create_line(
                zone_x0, spine_y, zone_x1, spine_y,
                fill=colors.get(zone_name, '#FFFFFF'),
                width=4
            )
            
            # Label
            self.canvas.create_text(
                (zone_x0 + zone_x1) / 2, spine_y + 15,
                text=zone_name.upper(),
                fill=colors.get(zone_name, '#FFFFFF'),
                font=('Helvetica', 7)
            )
    
    def _draw_dimensions(self, genome: PlateGenome, x0, y0, x1, y1):
        """Disegna dimensioni tavola."""
        dim_color = Colors.TEXT_TERTIARY
        
        # Lunghezza (sotto)
        self.canvas.create_line(x0, y1 + 15, x1, y1 + 15, fill=dim_color, width=1)
        self.canvas.create_line(x0, y1 + 10, x0, y1 + 20, fill=dim_color, width=1)
        self.canvas.create_line(x1, y1 + 10, x1, y1 + 20, fill=dim_color, width=1)
        self.canvas.create_text(
            (x0 + x1) / 2, y1 + 25,
            text=f"{genome.length:.2f} m",
            fill=dim_color, font=('Helvetica', 9)
        )
        
        # Larghezza (destra)
        self.canvas.create_line(x1 + 15, y0, x1 + 15, y1, fill=dim_color, width=1)
        self.canvas.create_line(x1 + 10, y0, x1 + 20, y0, fill=dim_color, width=1)
        self.canvas.create_line(x1 + 10, y1, x1 + 20, y1, fill=dim_color, width=1)
        self.canvas.create_text(
            x1 + 30, (y0 + y1) / 2,
            text=f"{genome.width:.2f} m",
            fill=dim_color, font=('Helvetica', 9), angle=90
        )
    
    def _draw_mode_frequencies(self, frequencies: list, x, y):
        """Disegna lista frequenze modi."""
        freq_color = Colors.TEXT_TERTIARY
        
        text = "Modes:\n"
        for i, f in enumerate(frequencies[:6]):
            text += f"  f{i+1}: {f:.1f} Hz\n"
        
        self.canvas.create_text(
            x + 50, y + 10,
            text=text,
            fill=freq_color, font=('Helvetica', 8),
            anchor='nw'
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test standalone
    root = tk.Tk()
    root.title("Plate Designer Test")
    root.geometry("900x600")
    
    tab = PlateDesignerTab(root)
    tab.pack(fill='both', expand=True)
    
    root.mainloop()
