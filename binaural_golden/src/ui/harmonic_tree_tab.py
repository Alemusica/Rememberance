"""
HarmonicTreeTab - Therapeutic growth mode with 3D visualization
═══════════════════════════════════════════════════════════════

Generate fundamental + harmonics visualized as a tree with therapeutic growth mode.

Based on natural phyllotaxis patterns:
- Trunk = Fundamental frequency
- Branches = Harmonics at Fibonacci ratios (2f, 3f, 5f, 8f, 13f)
- Branch thickness = Amplitude (φ⁻ⁿ decay)
- Branch rotation = Phase (cumulative golden angle: n × 137.5°)

Therapeutic Growth Mode:
- Harmonics emerge progressively over configurable duration (10s to 1hr)
- Golden-ratio timed cadences: each harmonic emerges at φ⁻ⁿ × duration
- Each harmonic fades in with golden envelope
- Phases evolve (rotate) during growth, simulating tree development
- 10fps animation (can be disabled for CPU saving)

Breathe Mode:
- Grow → Sustain → Shrink → Silence → Repeat
- Configurable cycles and sustain duration

"The universe grows in spirals - so does sound"
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from golden_studio import AudioEngine

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Golden Ratio and related constants
PHI = 1.618033988749895
PHI_CONJUGATE = 0.618033988749895  # 1/φ = φ - 1

# Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Golden Angle (degrees and radians)
GOLDEN_ANGLE_DEG = 137.5077640500378
GOLDEN_ANGLE_RAD = np.radians(GOLDEN_ANGLE_DEG)


# ═══════════════════════════════════════════════════════════════════════════════
# SOUND → LIGHT COLOR MAPPING (Synesthesia)
# ═══════════════════════════════════════════════════════════════════════════════

def sound_to_light_color(frequency: float) -> str:
    """
    Map audio frequency to visible light color using musical-visual synesthesia.
    
    Maps frequency to hue using the formula:
    hue = (octave_position * 360) where octave_position = log2(f/C0) % 1
    
    This creates a repeating color wheel where each octave cycles through
    the full spectrum: C=red, D=orange, E=yellow, F=green, G=cyan, A=blue, B=violet
    
    Returns hex color string.
    """
    if frequency <= 0:
        return '#333333'
    
    # Reference: C0 ≈ 16.35 Hz
    C0 = 16.3516
    
    # Get position within octave (0.0 to 1.0)
    octave_position = (np.log2(frequency / C0)) % 1.0
    
    # Map to hue (0-360), with red at C, cycling through spectrum
    hue = octave_position * 360
    
    # Full saturation and brightness for vivid colors
    saturation = 0.9
    value = 0.95
    
    # HSV to RGB conversion
    c = value * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = value - c
    
    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    return f'#{r:02x}{g:02x}{b:02x}'


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC TREE TAB
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicTreeTab:
    """
    Generate fundamental + harmonics visualized as a tree with therapeutic growth mode.
    """
    
    def __init__(self, parent, audio_engine: 'AudioEngine'):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        # State - basic
        self.fundamental = tk.DoubleVar(value=432.0)  # Base frequency
        self.num_harmonics = tk.IntVar(value=5)       # Number of harmonics
        self.harmonic_mode = tk.StringVar(value="fibonacci")  # fibonacci or integer
        self.amplitude_decay = tk.StringVar(value="phi")  # phi, sqrt, linear
        self.amplitude = tk.DoubleVar(value=0.7)
        self.stereo_spread = tk.DoubleVar(value=0.5)  # 0 = mono, 1 = full spread
        
        # State - growth mode
        self.growth_mode = tk.BooleanVar(value=True)  # True = progressive, False = instant
        self.growth_duration = tk.IntVar(value=60)    # Duration in seconds
        self.animation_enabled = tk.BooleanVar(value=True)  # 10fps animation toggle
        self.phase_evolution = tk.BooleanVar(value=True)  # Phases rotate during growth
        self.fixed_trunk_mode = tk.BooleanVar(value=True)  # True = fundamental fixed
        
        # Breathe mode: grow → sustain → fall back to silence
        self.breathe_mode = tk.BooleanVar(value=False)  # Cycle grow/shrink
        self.breathe_cycles = tk.IntVar(value=3)        # Number of breath cycles
        self.sustain_fraction = tk.DoubleVar(value=0.2) # Fraction of cycle to hold at peak
        
        # Growth state tracking
        self._growth_timer = None
        self._growth_start_time = None
        self._current_growth_level = [0.0] * 14  # Per-harmonic growth level (0-1)
        self._is_growing = False
        self._animation_after_id = None
        self._current_cycle = 0
        self._cycle_phase = 'grow'  # 'grow', 'sustain', 'shrink'
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI with scrollable left panel"""
        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL WITH SCROLLBAR
        # ═══════════════════════════════════════════════════════════════════
        left_container = ttk.Frame(self.frame)
        left_container.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Create canvas for scrolling
        self._scroll_canvas = tk.Canvas(left_container, highlightthickness=0, width=420)
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=self._scroll_canvas.yview)
        
        # Scrollable frame inside canvas
        left_frame = ttk.LabelFrame(self._scroll_canvas, text="🌳 Harmonic Tree Controls", padding=10)
        
        # Configure scrolling
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        self._scroll_canvas.pack(side='left', fill='both', expand=True)
        
        # Create window inside canvas for the frame
        self._canvas_window = self._scroll_canvas.create_window((0, 0), window=left_frame, anchor='nw')
        
        # Bind events for scroll region update
        def _configure_scroll_region(event):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        
        def _configure_canvas_width(event):
            self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)
        
        left_frame.bind('<Configure>', _configure_scroll_region)
        self._scroll_canvas.bind('<Configure>', _configure_canvas_width)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self._scroll_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        self._scroll_canvas.bind_all('<Button-4>', lambda e: self._scroll_canvas.yview_scroll(-1, "units"))
        self._scroll_canvas.bind_all('<Button-5>', lambda e: self._scroll_canvas.yview_scroll(1, "units"))
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTROLS CONTENT
        # ═══════════════════════════════════════════════════════════════════
        
        # Fundamental frequency
        fund_frame = ttk.LabelFrame(left_frame, text="🎵 Fundamental (Trunk)", padding=5)
        fund_frame.pack(fill='x', pady=5)
        
        ttk.Label(fund_frame, text="Frequency (Hz):").pack(side='left')
        ttk.Entry(fund_frame, textvariable=self.fundamental, width=8).pack(side='left', padx=5)
        ttk.Scale(fund_frame, from_=20, to=1000, variable=self.fundamental,
                  orient='horizontal', length=200).pack(side='left', padx=5)
        
        # Frequency presets
        preset_frame = ttk.Frame(fund_frame)
        preset_frame.pack(fill='x', pady=3)
        for name, freq in [("C₄", 261.63), ("A₄", 440), ("432Hz", 432), ("φ×100", PHI * 100)]:
            ttk.Button(preset_frame, text=name, width=6,
                      command=lambda f=freq: self.fundamental.set(f)).pack(side='left', padx=2)
        
        # Harmonics configuration
        harm_frame = ttk.LabelFrame(left_frame, text="🌿 Harmonics (Branches)", padding=5)
        harm_frame.pack(fill='x', pady=5)
        
        # Number of harmonics
        num_row = ttk.Frame(harm_frame)
        num_row.pack(fill='x', pady=3)
        ttk.Label(num_row, text="Harmonics:").pack(side='left')
        ttk.Spinbox(num_row, from_=1, to=13, textvariable=self.num_harmonics, width=5).pack(side='left', padx=5)
        
        # Harmonic mode
        mode_row = ttk.Frame(harm_frame)
        mode_row.pack(fill='x', pady=3)
        ttk.Label(mode_row, text="Ratios:").pack(side='left')
        ttk.Radiobutton(mode_row, text="Fibonacci (2,3,5,8,13...)", 
                       variable=self.harmonic_mode, value="fibonacci").pack(side='left', padx=5)
        ttk.Radiobutton(mode_row, text="Integer (2,3,4,5,6...)", 
                       variable=self.harmonic_mode, value="integer").pack(side='left', padx=5)
        
        # Amplitude decay mode
        decay_row = ttk.Frame(harm_frame)
        decay_row.pack(fill='x', pady=3)
        ttk.Label(decay_row, text="Decay:").pack(side='left')
        for name, val in [("φ⁻ⁿ", "phi"), ("1/√n", "sqrt"), ("1/n", "linear")]:
            ttk.Radiobutton(decay_row, text=name, variable=self.amplitude_decay, value=val).pack(side='left', padx=5)
        
        # Phase info
        phase_info = ttk.LabelFrame(left_frame, text="🌀 Phases (Golden Angle)", padding=5)
        phase_info.pack(fill='x', pady=5)
        ttk.Label(phase_info, text="Each harmonic rotated by 137.5° (φ angle)",
                 font=('Courier', 9)).pack()
        ttk.Label(phase_info, text="n=1: 137.5°, n=2: 275°, n=3: 412.5° (≡52.5°)...",
                 font=('Courier', 9), foreground='#888').pack()
        
        # ═══════════════════════════════════════════════════════════════════
        # THERAPEUTIC GROWTH MODE
        # ═══════════════════════════════════════════════════════════════════
        growth_frame = ttk.LabelFrame(left_frame, text="🌱 Therapeutic Growth Mode", padding=5)
        growth_frame.pack(fill='x', pady=5)
        
        # Growth mode toggle
        growth_toggle = ttk.Frame(growth_frame)
        growth_toggle.pack(fill='x', pady=3)
        ttk.Checkbutton(growth_toggle, text="Progressive Growth (harmonics emerge over time)", 
                       variable=self.growth_mode).pack(side='left')
        
        # Duration selector
        duration_row = ttk.Frame(growth_frame)
        duration_row.pack(fill='x', pady=3)
        ttk.Label(duration_row, text="Duration:").pack(side='left')
        
        # Duration presets
        duration_presets = [
            ("10s", 10), ("30s", 30), ("1m", 60), ("5m", 300),
            ("10m", 600), ("30m", 1800), ("1h", 3600)
        ]
        for name, secs in duration_presets:
            ttk.Button(duration_row, text=name, width=4,
                      command=lambda s=secs: self.growth_duration.set(s)).pack(side='left', padx=1)
        
        # Custom duration entry
        custom_row = ttk.Frame(growth_frame)
        custom_row.pack(fill='x', pady=2)
        ttk.Label(custom_row, text="Custom (sec):").pack(side='left')
        ttk.Entry(custom_row, textvariable=self.growth_duration, width=6).pack(side='left', padx=5)
        self.duration_label = ttk.Label(custom_row, text="", font=('Courier', 9))
        self.duration_label.pack(side='left', padx=5)
        self._update_duration_label()
        
        # Animation toggle
        anim_row = ttk.Frame(growth_frame)
        anim_row.pack(fill='x', pady=3)
        ttk.Checkbutton(anim_row, text="Animate tree (10 fps)", 
                       variable=self.animation_enabled).pack(side='left')
        ttk.Label(anim_row, text="(disable to save CPU)", 
                 font=('Courier', 8), foreground='#888').pack(side='left', padx=5)
        
        # Phase evolution toggle
        phase_row = ttk.Frame(growth_frame)
        phase_row.pack(fill='x', pady=3)
        ttk.Checkbutton(phase_row, text="Evolve phases during growth (rotating branches)", 
                       variable=self.phase_evolution).pack(side='left')
        
        # Fixed trunk mode toggle
        trunk_row = ttk.Frame(growth_frame)
        trunk_row.pack(fill='x', pady=3)
        ttk.Label(trunk_row, text="    Rotation mode:", font=('Courier', 9)).pack(side='left')
        ttk.Radiobutton(trunk_row, text="🌲 Fixed Trunk", variable=self.fixed_trunk_mode,
                       value=True).pack(side='left', padx=5)
        ttk.Radiobutton(trunk_row, text="🌀 Whole Tree", variable=self.fixed_trunk_mode,
                       value=False).pack(side='left', padx=5)
        ttk.Label(trunk_row, text="(fundamental phase)", font=('Courier', 8), 
                 foreground='#888').pack(side='left', padx=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # BREATHE MODE: Grow → Sustain → Shrink → Silence → Repeat
        # ═══════════════════════════════════════════════════════════════════
        breathe_frame = ttk.LabelFrame(growth_frame, text="🌬️ Breathe Mode (Grow ↔ Shrink)", padding=3)
        breathe_frame.pack(fill='x', pady=3)
        
        breathe_toggle = ttk.Frame(breathe_frame)
        breathe_toggle.pack(fill='x', pady=2)
        ttk.Checkbutton(breathe_toggle, text="Enable breathing (cycle grow → fall back)", 
                       variable=self.breathe_mode).pack(side='left')
        
        breathe_params = ttk.Frame(breathe_frame)
        breathe_params.pack(fill='x', pady=2)
        ttk.Label(breathe_params, text="Cycles:").pack(side='left')
        ttk.Spinbox(breathe_params, from_=1, to=20, textvariable=self.breathe_cycles, 
                   width=4).pack(side='left', padx=3)
        ttk.Label(breathe_params, text="Sustain:").pack(side='left', padx=(10,0))
        ttk.Scale(breathe_params, from_=0.05, to=0.5, variable=self.sustain_fraction,
                 orient='horizontal', length=80).pack(side='left', padx=3)
        ttk.Label(breathe_params, text="(hold at peak)", font=('Courier', 8), 
                 foreground='#888').pack(side='left')
        
        # Growth progress bar
        self.growth_progress = ttk.Progressbar(growth_frame, mode='determinate', length=250)
        self.growth_progress.pack(fill='x', pady=5)
        self.growth_status = ttk.Label(growth_frame, text="", font=('Courier', 9))
        self.growth_status.pack()
        
        # Master controls
        master_frame = ttk.LabelFrame(left_frame, text="🎚️ Master", padding=5)
        master_frame.pack(fill='x', pady=5)
        
        amp_row = ttk.Frame(master_frame)
        amp_row.pack(fill='x', pady=2)
        ttk.Label(amp_row, text="Amplitude:").pack(side='left')
        ttk.Scale(amp_row, from_=0, to=1, variable=self.amplitude,
                  orient='horizontal', length=150).pack(side='left', padx=5)
        
        stereo_row = ttk.Frame(master_frame)
        stereo_row.pack(fill='x', pady=2)
        ttk.Label(stereo_row, text="Stereo Spread:").pack(side='left')
        ttk.Scale(stereo_row, from_=0, to=1, variable=self.stereo_spread,
                  orient='horizontal', length=150).pack(side='left', padx=5)
        
        # Playback buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ GROW & PLAY", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Right panel - Tree Visualization
        right_frame = ttk.LabelFrame(self.frame, text="🌳 Tree Visualization", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=400, height=400, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        # Info panel
        self.info_text = tk.Text(right_frame, width=45, height=8, bg='#0a0a15',
                                fg='#00ff88', font=('Courier', 9), state='disabled')
        self.info_text.pack(fill='x')
        
        self.status_var = tk.StringVar(value="Configure harmonics and press GROW & PLAY")
        ttk.Label(right_frame, textvariable=self.status_var).pack()
        
        # Bind updates
        self.fundamental.trace_add('write', self._on_param_change)
        self.num_harmonics.trace_add('write', self._on_param_change)
        self.harmonic_mode.trace_add('write', self._on_param_change)
        self.amplitude_decay.trace_add('write', self._on_param_change)
        self.amplitude.trace_add('write', self._on_param_change)
        self.stereo_spread.trace_add('write', self._on_param_change)
        self.growth_duration.trace_add('write', lambda *a: self._update_duration_label())
        
        # Initial draw
        self._draw_tree()
    
    def _update_duration_label(self):
        """Update the duration label with human-readable format"""
        try:
            secs = self.growth_duration.get()
            if secs >= 3600:
                label = f"= {secs/3600:.1f} hours"
            elif secs >= 60:
                label = f"= {secs/60:.1f} minutes"
            else:
                label = f"= {secs} seconds"
            self.duration_label.config(text=label)
        except:
            pass
    
    def _calculate_growth_schedule(self):
        """
        Calculate when each harmonic should emerge using SPREAD golden ratio cadences.
        
        Returns list of (harmonic_index, emergence_time_fraction) tuples.
        """
        n = self.num_harmonics.get() + 1  # Include fundamental
        
        # Fundamental at t=0
        schedule = [(0, 0.0)]
        
        if n <= 1:
            return schedule
        
        # Spread harmonics across 0% to 80% of duration
        max_emergence = 0.80
        
        for i in range(1, n):
            # Linear spread with slight golden weighting
            base_fraction = i / n
            
            # Apply subtle golden modulation
            golden_weight = 1.0 - (PHI_CONJUGATE ** (i * 0.5)) * 0.3
            
            emergence = base_fraction * max_emergence * golden_weight
            schedule.append((i, emergence))
        
        return schedule
    
    def _calculate_fade_envelope(self, elapsed_fraction, emergence_fraction, harmonic_index):
        """
        Calculate golden-ratio fade envelope for a harmonic.
        
        Returns amplitude multiplier (0.0 to 1.0)
        """
        if elapsed_fraction < emergence_fraction:
            return 0.0  # Not yet emerged
        
        time_since_emergence = elapsed_fraction - emergence_fraction
        
        # Longer fade-in durations
        base_fade = 0.15
        golden_factor = 1.0 - (harmonic_index * 0.02)
        fade_in_duration = max(0.10, base_fade * golden_factor)
        
        if time_since_emergence >= fade_in_duration:
            return 1.0  # Fully grown
        
        # Smooth cosine fade-in
        progress = time_since_emergence / fade_in_duration
        return 0.5 * (1 - np.cos(np.pi * progress))
    
    def _calculate_harmonics(self, apply_growth=False, elapsed_fraction=0.0):
        """
        Calculate harmonic frequencies, amplitudes, phases, and stereo positions.
        
        Returns:
            (frequencies, amplitudes, phases, positions)
        """
        fund = self.fundamental.get()
        n = self.num_harmonics.get()
        mode = self.harmonic_mode.get()
        decay = self.amplitude_decay.get()
        spread = self.stereo_spread.get()
        
        # ═══════════════════════════════════════════════════════════════════
        # FREQUENCIES: Fibonacci or Integer ratios
        # ═══════════════════════════════════════════════════════════════════
        
        if mode == "fibonacci":
            fib_ratios = [FIBONACCI[i+2] for i in range(n)]  # Start from F(3)=2
            frequencies = [fund] + [fund * r for r in fib_ratios]
        else:
            frequencies = [fund] + [fund * (i + 2) for i in range(n)]
        
        # ═══════════════════════════════════════════════════════════════════
        # AMPLITUDES: Various decay modes
        # ═══════════════════════════════════════════════════════════════════
        
        total = len(frequencies)
        if decay == "phi":
            amplitudes = [PHI_CONJUGATE ** i for i in range(total)]
        elif decay == "sqrt":
            amplitudes = [1.0 / np.sqrt(i + 1) for i in range(total)]
        else:
            amplitudes = [1.0 / (i + 1) for i in range(total)]
        
        # ═══════════════════════════════════════════════════════════════════
        # GROWTH MODE: Progressive amplitude envelope
        # ═══════════════════════════════════════════════════════════════════
        
        if apply_growth and self.growth_mode.get():
            schedule = self._calculate_growth_schedule()
            for i in range(total):
                emergence = schedule[i][1] if i < len(schedule) else 0.95
                envelope = self._calculate_fade_envelope(elapsed_fraction, emergence, i)
                amplitudes[i] *= envelope
                self._current_growth_level[i] = envelope
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASES: Cumulative Golden Angle (phyllotaxis pattern)
        # ═══════════════════════════════════════════════════════════════════
        
        base_phases = [(i * GOLDEN_ANGLE_RAD) % (2 * np.pi) for i in range(total)]
        
        # Phase evolution during growth
        if apply_growth and self.phase_evolution.get():
            phase_offset = elapsed_fraction * 2 * np.pi
            
            if self.fixed_trunk_mode.get():
                # Fundamental stays at 0°, harmonics rotate
                phases = [0.0]
                phases += [(base_phases[i] + phase_offset * PHI_CONJUGATE ** i) % (2 * np.pi) 
                          for i in range(1, total)]
            else:
                # All phases rotate including fundamental
                phases = [(base_phases[i] + phase_offset * PHI_CONJUGATE ** i) % (2 * np.pi) 
                         for i in range(total)]
        else:
            phases = base_phases
        
        # ═══════════════════════════════════════════════════════════════════
        # STEREO: Spiral positioning based on golden angle
        # ═══════════════════════════════════════════════════════════════════
        
        positions = []
        for i in range(total):
            angle = i * GOLDEN_ANGLE_RAD
            if apply_growth and self.phase_evolution.get():
                angle += elapsed_fraction * np.pi * 0.5
            pan = np.sin(angle) * spread
            positions.append(pan)
        
        return frequencies, amplitudes, phases, positions
    
    def _draw_tree(self, elapsed_fraction=0.0):
        """
        Draw 3D isometric harmonic tree visualization.
        """
        self.canvas.delete('all')
        
        is_growing = self._is_growing and self.growth_mode.get()
        frequencies, amplitudes, phases, positions = self._calculate_harmonics(
            apply_growth=is_growing, elapsed_fraction=elapsed_fraction)
        
        # ═══════════════════════════════════════════════════════════════════
        # SWISS DESIGN: Clean dark background with subtle grid
        # ═══════════════════════════════════════════════════════════════════
        self.canvas.configure(bg='#0a0a0f')
        
        # Subtle grid
        grid_color = '#15151f'
        for x in range(0, 400, 40):
            self.canvas.create_line(x, 0, x, 400, fill=grid_color, width=1)
        for y in range(0, 400, 40):
            self.canvas.create_line(0, y, 400, y, fill=grid_color, width=1)
        
        cx, cy = 200, 320
        
        # ═══════════════════════════════════════════════════════════════════
        # 3D ISOMETRIC PROJECTION
        # ═══════════════════════════════════════════════════════════════════
        def iso_project(x3d, y3d, z3d):
            iso_angle = np.radians(30)
            x2d = cx + (x3d - z3d) * np.cos(iso_angle)
            y2d = cy - y3d - (x3d + z3d) * np.sin(iso_angle) * 0.5
            return x2d, y2d
        
        # Ground plane circles
        ground_color = '#1a1a2a'
        for r in [60, 100, 140]:
            points = []
            for angle in range(0, 360, 15):
                rad = np.radians(angle)
                x3d, z3d = r * np.cos(rad), r * np.sin(rad)
                x2d, y2d = iso_project(x3d, 0, z3d)
                points.extend([x2d, y2d])
            if len(points) >= 4:
                self.canvas.create_polygon(points, outline=ground_color, fill='', width=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # TRUNK (Fundamental)
        # ═══════════════════════════════════════════════════════════════════
        trunk_growth = self._current_growth_level[0] if is_growing else 1.0
        trunk_height = 120 * trunk_growth
        trunk_width = max(3, 12 * amplitudes[0] * trunk_growth)
        
        trunk_color = sound_to_light_color(frequencies[0])
        trunk_shadow = '#3d2817'
        
        if trunk_height > 0:
            x2d_base, y2d_base = iso_project(0, 0, 0)
            x2d_top, y2d_top = iso_project(0, trunk_height, 0)
            
            offset = trunk_width * 0.3
            self.canvas.create_line(x2d_base - offset, y2d_base,
                                   x2d_top - offset, y2d_top,
                                   fill=trunk_shadow, width=trunk_width * 0.7)
            self.canvas.create_line(x2d_base, y2d_base, x2d_top, y2d_top,
                                   fill=trunk_color, width=trunk_width,
                                   capstyle='round')
        
        # ═══════════════════════════════════════════════════════════════════
        # BRANCHES (Harmonics)
        # ═══════════════════════════════════════════════════════════════════
        branch_origin_y = trunk_height
        
        branches = []
        
        for i in range(1, len(frequencies)):
            growth = self._current_growth_level[i] if is_growing else 1.0
            
            if growth < 0.01:
                continue
            
            golden_angle = i * GOLDEN_ANGLE_DEG
            angle_rad = np.radians(golden_angle)
            
            spiral_radius = 25 * np.sqrt(i) * growth
            layer_height = branch_origin_y + (i * 15 * PHI_CONJUGATE) * growth
            
            x3d = spiral_radius * np.cos(angle_rad)
            z3d = spiral_radius * np.sin(angle_rad)
            y3d = layer_height + 20 * growth
            
            branch_length = (40 + 30 * amplitudes[i]) * growth
            branch_width = max(1, (8 - i * 0.3) * growth)
            
            color = sound_to_light_color(frequencies[i])
            
            if growth < 1.0:
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                r, g, b = int(r * growth), int(g * growth), int(b * growth)
                color = f'#{r:02x}{g:02x}{b:02x}'
            
            branches.append({
                'i': i,
                'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
                'length': branch_length,
                'width': branch_width,
                'color': color,
                'growth': growth,
                'freq': frequencies[i],
                'depth': z3d
            })
        
        # Sort by depth (back to front)
        branches.sort(key=lambda b: b['depth'])
        
        # Draw branches
        for branch in branches:
            i = branch['i']
            x3d, y3d, z3d = branch['x3d'], branch['y3d'], branch['z3d']
            
            start_x2d, start_y2d = iso_project(0, branch_origin_y + i * 5, 0)
            end_x2d, end_y2d = iso_project(x3d, y3d, z3d)
            
            self.canvas.create_line(start_x2d, start_y2d, end_x2d, end_y2d,
                                   fill=branch['color'], width=branch['width'],
                                   capstyle='round')
            
            # Node at tip
            node_size = (4 + amplitudes[0] * 6) * branch['growth']
            if node_size >= 2:
                self.canvas.create_oval(
                    end_x2d - node_size + 1, end_y2d - node_size + 1,
                    end_x2d + node_size + 1, end_y2d + node_size + 1,
                    fill='#000000', outline='')
                self.canvas.create_oval(
                    end_x2d - node_size, end_y2d - node_size,
                    end_x2d + node_size, end_y2d + node_size,
                    fill=branch['color'], outline='')
                hl_size = node_size * 0.4
                self.canvas.create_oval(
                    end_x2d - hl_size - node_size*0.3, end_y2d - hl_size - node_size*0.3,
                    end_x2d + hl_size - node_size*0.3, end_y2d + hl_size - node_size*0.3,
                    fill='#ffffff', outline='')
            
            # Labels
            if i <= 5 and branch['growth'] > 0.6:
                label_offset = 18
                label_x = end_x2d + (label_offset if end_x2d > cx else -label_offset)
                anchor = 'w' if end_x2d > cx else 'e'
                self.canvas.create_text(label_x, end_y2d,
                                       text=f"{branch['freq']:.0f}",
                                       fill=branch['color'], 
                                       font=('Helvetica Neue', 9),
                                       anchor=anchor)
        
        # ═══════════════════════════════════════════════════════════════════
        # GOLDEN SPIRAL
        # ═══════════════════════════════════════════════════════════════════
        if trunk_height > 40:
            spiral_points = []
            spiral_growth = elapsed_fraction if is_growing else 1.0
            num_points = int(40 * spiral_growth)
            for j in range(max(2, num_points)):
                angle = j * GOLDEN_ANGLE_RAD * 0.4
                r = 6 * np.sqrt(j * 0.3) * spiral_growth
                x3d = r * np.cos(angle)
                z3d = r * np.sin(angle)
                y3d = trunk_height * 0.7 + j * 0.5
                x2d, y2d = iso_project(x3d, y3d, z3d)
                spiral_points.extend([x2d, y2d])
            
            if len(spiral_points) >= 4:
                self.canvas.create_line(spiral_points, fill='#ffd700', 
                                       width=1.5, smooth=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # SWISS TYPOGRAPHY: Title and Info
        # ═══════════════════════════════════════════════════════════════════
        mode = self.harmonic_mode.get()
        
        if is_growing:
            percent = int(elapsed_fraction * 100)
            title = f"{percent}%"
            subtitle = f"growing · {mode}"
        else:
            title = "φ"
            subtitle = f"harmonic tree · {mode}"
        
        self.canvas.create_text(20, 20, text=title,
                               fill='#ffffff', font=('Helvetica Neue', 24, 'bold'),
                               anchor='nw')
        self.canvas.create_text(20, 50, text=subtitle,
                               fill='#888888', font=('Helvetica Neue', 10),
                               anchor='nw')
        
        fund_color = sound_to_light_color(frequencies[0])
        self.canvas.create_text(20, 380, text=f"ƒ₀ = {frequencies[0]:.1f} Hz",
                               fill=fund_color, font=('Helvetica Neue', 11),
                               anchor='sw')
        
        self.canvas.create_text(380, 380, 
                               text="sound→light",
                               fill='#555555', font=('Helvetica Neue', 8),
                               anchor='se')
    
    def _update_info(self, elapsed_fraction=0.0):
        """Update the info panel with current harmonics"""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        
        is_growing = self._is_growing and self.growth_mode.get()
        frequencies, amplitudes, phases, positions = self._calculate_harmonics(
            apply_growth=is_growing, elapsed_fraction=elapsed_fraction)
        
        mode = self.harmonic_mode.get()
        decay = self.amplitude_decay.get()
        
        if is_growing:
            self.info_text.insert('end', f"═══ HARMONIC TREE (Growing {int(elapsed_fraction*100)}%) ═══\n\n")
        else:
            self.info_text.insert('end', f"═══ HARMONIC TREE ({mode}/{decay}) ═══\n\n")
        
        self.info_text.insert('end', f"{'#':<3} {'Freq (Hz)':<10} {'Amp':<8} {'Phase°':<10} {'Growth':<8}\n")
        self.info_text.insert('end', "─" * 45 + "\n")
        
        for i, (f, a, p, pos) in enumerate(zip(frequencies, amplitudes, phases, positions)):
            phase_deg = np.degrees(p)
            growth = self._current_growth_level[i] if is_growing else 1.0
            growth_str = f"{growth*100:.0f}%" if is_growing else "100%"
            name = "Fund" if i == 0 else f"H{i}"
            
            if growth < 0.01:
                self.info_text.insert('end', f"{name:<3} {'(emerging...)':<38}\n")
            else:
                self.info_text.insert('end', 
                    f"{name:<3} {f:<10.1f} {a:<8.3f} {phase_deg:<10.1f} {growth_str:<8}\n")
        
        self.info_text.config(state='disabled')
    
    def _on_param_change(self, *args):
        """Handle parameter changes - update visualization and audio"""
        try:
            elapsed_fraction = 0.0
            if self._is_growing and self._growth_start_time:
                elapsed = time.time() - self._growth_start_time
                duration = self.growth_duration.get()
                elapsed_fraction = min(elapsed / duration, 1.0)
            
            self._draw_tree(elapsed_fraction)
            self._update_info(elapsed_fraction)
            
            if self.audio.is_playing():
                self._update_audio(elapsed_fraction)
        except:
            pass
    
    def _update_audio(self, elapsed_fraction=0.0):
        """Update audio parameters in real-time"""
        is_growing = self._is_growing and self.growth_mode.get()
        frequencies, amplitudes, phases, positions = self._calculate_harmonics(
            apply_growth=is_growing, elapsed_fraction=elapsed_fraction)
        self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
    
    def _play(self):
        """Start playing the harmonic tree with optional growth mode"""
        amp = self.amplitude.get()
        
        # Reset growth state
        self._current_growth_level = [0.0] * 14
        self._is_growing = self.growth_mode.get()
        self._growth_start_time = time.time()
        
        if self._is_growing:
            self._current_growth_level[0] = 1.0  # Fundamental starts immediately
            frequencies, amplitudes, phases, positions = self._calculate_harmonics(
                apply_growth=True, elapsed_fraction=0.0)
            
            self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                      master_amplitude=amp)
            
            self._growth_callback()
            
            duration = self.growth_duration.get()
            self.status_var.set(f"🌱 Growing over {duration}s - Tree emerging...")
        else:
            for i in range(14):
                self._current_growth_level[i] = 1.0
            frequencies, amplitudes, phases, positions = self._calculate_harmonics()
            
            self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                      master_amplitude=amp)
            
            fund = self.fundamental.get()
            n = self.num_harmonics.get()
            self.status_var.set(f"🔊 Playing: {fund:.1f} Hz + {n} harmonics")
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
    
    def _calculate_breathe_envelope(self, elapsed_fraction):
        """
        Calculate envelope for breathe mode (grow → sustain → shrink).
        
        Returns growth multiplier (0.0 to 1.0)
        """
        sustain = self.sustain_fraction.get()
        grow_fraction = (1.0 - sustain) / 2
        shrink_fraction = grow_fraction
        
        if elapsed_fraction < grow_fraction:
            t = elapsed_fraction / grow_fraction
            return 0.5 * (1 - np.cos(np.pi * t))
        elif elapsed_fraction < grow_fraction + sustain:
            return 1.0
        else:
            t = (elapsed_fraction - grow_fraction - sustain) / shrink_fraction
            return 0.5 * (1 + np.cos(np.pi * t))
    
    def _growth_callback(self):
        """Timer callback for growth animation - runs at 10fps"""
        if not self._is_growing or not self.audio.is_playing():
            return
        
        elapsed = time.time() - self._growth_start_time
        duration = self.growth_duration.get()
        
        if self.breathe_mode.get():
            # Breathe mode: multiple cycles of grow/shrink
            total_cycles = self.breathe_cycles.get()
            cycle_duration = duration / total_cycles
            
            current_cycle = int(elapsed / cycle_duration)
            cycle_elapsed = elapsed % cycle_duration
            cycle_fraction = cycle_elapsed / cycle_duration
            
            breathe_envelope = self._calculate_breathe_envelope(cycle_fraction)
            
            for i in range(14):
                self._current_growth_level[i] = breathe_envelope
            
            overall_fraction = elapsed / duration
            self.growth_progress['value'] = overall_fraction * 100
            
            phase = "🌱" if cycle_fraction < 0.4 else ("💚" if cycle_fraction < 0.6 else "🍂")
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            total_min, total_sec = divmod(duration, 60)
            self.growth_status.config(
                text=f"{phase} Cycle {current_cycle+1}/{total_cycles} | {elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}")
            
            self._update_audio(breathe_envelope)
            
            if self.animation_enabled.get():
                self._draw_tree(breathe_envelope)
                self._update_info(breathe_envelope)
            
            if elapsed >= duration:
                self._is_growing = False
                self.status_var.set(f"🌬️ Breathing complete - {total_cycles} cycles")
                self.growth_status.config(text="Breathing complete ✓")
                self._stop()
                return
        else:
            # Normal grow-only mode
            elapsed_fraction = min(elapsed / duration, 1.0)
            
            self.growth_progress['value'] = elapsed_fraction * 100
            
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            total_min, total_sec = divmod(duration, 60)
            self.growth_status.config(text=f"{elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}")
            
            self._update_audio(elapsed_fraction)
            
            if self.animation_enabled.get():
                self._draw_tree(elapsed_fraction)
                self._update_info(elapsed_fraction)
            
            if elapsed_fraction >= 1.0:
                self._is_growing = False
                self.status_var.set(f"🌳 Fully grown - {self.num_harmonics.get()+1} harmonics playing")
                self.growth_status.config(text="Growth complete ✓")
                for i in range(14):
                    self._current_growth_level[i] = 1.0
                self._draw_tree(1.0)
                self._update_info(1.0)
                return
        
        # Schedule next frame (100ms = 10fps)
        self._animation_after_id = self.frame.after(100, self._growth_callback)
    
    def _stop(self):
        """Stop playback"""
        self._is_growing = False
        if self._animation_after_id:
            self.frame.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        
        self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.growth_progress['value'] = 0
        self.growth_status.config(text="")
        
        # Reset growth levels for display
        for i in range(14):
            self._current_growth_level[i] = 1.0
        self._draw_tree()
        self._update_info()
        
        self.status_var.set("Configure harmonics and press GROW & PLAY")
