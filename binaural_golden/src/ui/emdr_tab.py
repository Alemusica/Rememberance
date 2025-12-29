"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  EMDR TAB - BILATERAL AUDIO STIMULATION                      ║
║                                                                              ║
║   Eye Movement Desensitization and Reprocessing (EMDR) Audio Implementation  ║
║                                                                              ║
║   Based on clinical research:                                                ║
║   - Bilateral stimulation at 0.5-2 Hz (standard EMDR protocol)               ║
║   - Audio tones alternating L/R (validated alternative to eye movements)     ║
║   - Activates parasympathetic nervous system (vagal tone)                    ║
║   - Linked to REM sleep mechanisms for memory processing                     ║
║                                                                              ║
║   Enhanced with sacred geometry:                                             ║
║   - φ (Golden Ratio) phase relationships between hemispheres                 ║
║   - Solfeggio frequencies for specific healing intentions                    ║
║   - Golden envelope curves for gentle onset/offset                           ║
║   - Fibonacci-timed progression for trauma annealing journeys               ║
║                                                                              ║
║   "Integrate hemispheres, anneal traumas, restore order"                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
import time
import numpy as np
from typing import Optional, Dict, List, Tuple

# Try importing from core module, fallback to local definitions
try:
    from core.audio_engine import AudioEngine
    from core.golden_math import PHI, PHI_CONJUGATE, golden_fade
except ImportError:
    PHI = 1.618033988749895
    PHI_CONJUGATE = 0.6180339887498949
    
    def golden_fade(t, fade_in=True):
        """Golden ratio based fade curve using φ exponent."""
        t = max(0.0, min(1.0, t))
        if fade_in:
            base = (1 - np.cos(t * np.pi)) / 2.0
            return base ** PHI_CONJUGATE
        else:
            base = (1 - np.cos((1 - t) * np.pi)) / 2.0
            return 1.0 - (base ** PHI_CONJUGATE)
    
    AudioEngine = None

# Sacred frequency constants
try:
    from golden_constants import (
        SOLFEGGIO_FREQUENCIES, GOLDEN_ANGLE_DEG, GOLDEN_ANGLE_RAD,
        generate_golden_phases, golden_ease
    )
except ImportError:
    SOLFEGGIO_FREQUENCIES = {
        174: {'name': 'UT', 'property': 'Pain reduction, safety'},
        285: {'name': 'RE', 'property': 'Tissue healing'},
        396: {'name': 'MI', 'property': 'Liberating guilt and fear'},
        417: {'name': 'FA', 'property': 'Facilitating change'},
        528: {'name': 'SOL', 'property': 'DNA repair, miracles'},
        639: {'name': 'LA', 'property': 'Relationships, connection'},
        741: {'name': 'SI', 'property': 'Expression, solutions'},
        852: {'name': 'TI', 'property': 'Spiritual awakening'},
        963: {'name': 'DO', 'property': 'Divine consciousness'},
    }
    GOLDEN_ANGLE_DEG = 137.5077640500378546
    GOLDEN_ANGLE_RAD = np.radians(GOLDEN_ANGLE_DEG)
    
    def generate_golden_phases(n):
        return np.array([(i * GOLDEN_ANGLE_RAD) % (2 * np.pi) for i in range(n)])
    
    def golden_ease(t):
        theta = t * np.pi * PHI
        return (1.0 - np.cos(theta * PHI_CONJUGATE)) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# EMDR PROTOCOL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Bilateral stimulation speeds (Hz) - research validated range
EMDR_SPEED_MIN = 0.5   # Slow, grounding
EMDR_SPEED_MAX = 2.0   # Standard EMDR
EMDR_SPEED_DEFAULT = 1.0  # 1 Hz = 1 bilateral cycle per second

# Bilateral modes
BILATERAL_MODES = {
    "standard": "Standard EMDR - Clean alternation",
    "golden_phase": "φ Phase - 137.5° hemispheric offset",
    "breathing": "Respiratory - Synced to 6 breaths/min",
    "theta_pulse": "Theta Pulse - 5Hz carrier modulated bilaterally",
}

# Sacred frequency presets for EMDR
EMDR_FREQUENCIES = {
    "trauma_release": {
        "freq": 396,  # Solfeggio MI - Liberating guilt and fear
        "name": "🌊 Trauma Release (396Hz)",
        "description": "Solfeggio MI frequency for liberating guilt and fear",
    },
    "change_facilitation": {
        "freq": 417,  # Solfeggio FA - Facilitating change  
        "name": "🔄 Change Facilitation (417Hz)",
        "description": "Solfeggio FA frequency for facilitating change",
    },
    "dna_repair": {
        "freq": 528,  # Solfeggio SOL - DNA repair
        "name": "✨ DNA Repair (528Hz)",
        "description": "Solfeggio SOL - 'Miracle tone' for DNA repair",
    },
    "connection": {
        "freq": 639,  # Solfeggio LA - Relationships
        "name": "💚 Connection (639Hz)",
        "description": "Solfeggio LA for healing relationships",
    },
    "sacred_432": {
        "freq": 432,  # Verdi tuning
        "name": "🎵 Sacred 432Hz",
        "description": "Verdi/Schumann resonance tuning",
    },
    "gamma_40": {
        "freq": 40,  # Gamma brainwave entrainment (needs carrier)
        "name": "🧠 Gamma 40Hz",
        "description": "Gamma brainwave for heightened cognition",
    },
    "alpha_10": {
        "freq": 10,  # Alpha brainwave (needs carrier)
        "name": "🧘 Alpha 10Hz",
        "description": "Alpha brainwave for relaxed awareness",
    },
    "theta_6": {
        "freq": 6,  # Theta brainwave (needs carrier)
        "name": "🌙 Theta 6Hz",
        "description": "Theta brainwave for deep meditation, REM",
    },
}

# Trauma annealing journey programs (inspired by metallurgical annealing)
ANNEALING_PROGRAMS = {
    "gentle_release": {
        "name": "🌅 Gentle Release",
        "duration_min": 5,
        "description": "Soft entry into bilateral processing",
        "phases": [
            {"name": "Ground", "duration_pct": 0.20, "freq": 396, "speed": 0.5},
            {"name": "Open", "duration_pct": 0.30, "freq": 417, "speed": 0.8},
            {"name": "Process", "duration_pct": 0.30, "freq": 528, "speed": 1.0},
            {"name": "Integrate", "duration_pct": 0.20, "freq": 639, "speed": 0.6},
        ],
    },
    "deep_processing": {
        "name": "🔥 Deep Processing",
        "duration_min": 10,
        "description": "Intensive bilateral for stubborn patterns",
        "phases": [
            {"name": "Warm-up", "duration_pct": 0.10, "freq": 432, "speed": 0.5},
            {"name": "Activate", "duration_pct": 0.15, "freq": 396, "speed": 0.8},
            {"name": "Intensify", "duration_pct": 0.25, "freq": 417, "speed": 1.2},
            {"name": "Peak", "duration_pct": 0.20, "freq": 528, "speed": 1.5},
            {"name": "Cool-down", "duration_pct": 0.15, "freq": 639, "speed": 0.8},
            {"name": "Integrate", "duration_pct": 0.15, "freq": 852, "speed": 0.5},
        ],
    },
    "hemispheric_sync": {
        "name": "🧠 Hemispheric Sync",
        "duration_min": 8,
        "description": "Focus on bilateral integration with golden phases",
        "phases": [
            {"name": "Left Activate", "duration_pct": 0.25, "freq": 432, "speed": 0.7, "phase_mode": "left_lead"},
            {"name": "Right Activate", "duration_pct": 0.25, "freq": 432, "speed": 0.7, "phase_mode": "right_lead"},
            {"name": "Merge", "duration_pct": 0.30, "freq": 528, "speed": 1.0, "phase_mode": "golden_sync"},
            {"name": "Unity", "duration_pct": 0.20, "freq": 639, "speed": 0.5, "phase_mode": "coherent"},
        ],
    },
    "golden_spiral": {
        "name": "🌀 Golden Spiral",
        "duration_min": 12,
        "description": "Fibonacci-timed phases, golden ratio transitions",
        "phases": [
            # Fibonacci-proportioned phases: 1:1:2:3:5:8 → normalized
            {"name": "Seed", "duration_pct": 0.05, "freq": 174, "speed": 0.5},
            {"name": "Sprout", "duration_pct": 0.05, "freq": 285, "speed": 0.6},
            {"name": "Grow", "duration_pct": 0.10, "freq": 396, "speed": 0.8},
            {"name": "Expand", "duration_pct": 0.15, "freq": 417, "speed": 1.0},
            {"name": "Flower", "duration_pct": 0.25, "freq": 528, "speed": 1.2},
            {"name": "Fruit", "duration_pct": 0.40, "freq": 639, "speed": 0.8},
        ],
    },
}


class EMDRTab:
    """
    EMDR Bilateral Audio Stimulation Tab.
    
    Implements clinically-validated bilateral audio stimulation with
    sacred geometry enhancements for hemispheric integration and
    trauma processing.
    """
    
    def __init__(self, parent: tk.Frame, audio_engine: 'AudioEngine'):
        """
        Initialize EMDR Tab.
        
        Args:
            parent: Parent frame (notebook)
            audio_engine: Shared audio engine instance
        """
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)  # This is the frame added to notebook
        
        # State
        self._is_playing = False
        self._is_journey_playing = False
        self._bilateral_timer = None
        self._journey_timer = None
        self._bilateral_phase = 0.0  # 0 = left, 0.5 = right, wraps at 1.0
        self._start_time = 0.0
        self._journey_start_time = 0.0
        
        # Current parameters
        self._current_freq = 432.0
        self._current_speed = 1.0
        self._current_mode = "standard"
        self._current_program = None
        self._current_phase_index = 0
        
        # Golden phase offset between hemispheres
        self._phase_offset_rad = GOLDEN_ANGLE_RAD  # 137.5° default
        
        # Build UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the EMDR tab interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.frame, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════════════
        header = ttk.Frame(main_frame)
        header.pack(fill='x', pady=(0, 10))
        
        ttk.Label(header, text="🧠 EMDR - Bilateral Audio Stimulation", 
                  font=('Helvetica', 14, 'bold')).pack(side='left')
        
        # Status indicator
        self.status_label = ttk.Label(header, text="⚪ Ready", 
                                       font=('Helvetica', 10))
        self.status_label.pack(side='right')
        
        # ══════════════════════════════════════════════════════════════════════
        # LEFT COLUMN - MANUAL CONTROLS
        # ══════════════════════════════════════════════════════════════════════
        columns = ttk.Frame(main_frame)
        columns.pack(fill='both', expand=True)
        
        left_col = ttk.LabelFrame(columns, text="Manual Bilateral", padding=10)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Frequency Selection
        freq_frame = ttk.Frame(left_col)
        freq_frame.pack(fill='x', pady=5)
        
        ttk.Label(freq_frame, text="Frequency:").pack(side='left')
        
        self.freq_var = tk.StringVar(value="sacred_432")
        freq_combo = ttk.Combobox(freq_frame, textvariable=self.freq_var, 
                                   state='readonly', width=25)
        freq_combo['values'] = [EMDR_FREQUENCIES[k]['name'] for k in EMDR_FREQUENCIES.keys()]
        freq_combo.pack(side='left', padx=5)
        freq_combo.bind('<<ComboboxSelected>>', self._on_freq_change)
        
        # Custom frequency entry
        ttk.Label(freq_frame, text="Hz:").pack(side='left', padx=(10, 0))
        self.custom_freq = tk.DoubleVar(value=432.0)
        freq_spin = ttk.Spinbox(freq_frame, from_=20, to=2000, 
                                 textvariable=self.custom_freq, width=8)
        freq_spin.pack(side='left', padx=5)
        
        # Bilateral Speed
        speed_frame = ttk.Frame(left_col)
        speed_frame.pack(fill='x', pady=5)
        
        ttk.Label(speed_frame, text="Bilateral Speed:").pack(side='left')
        
        self.speed_var = tk.DoubleVar(value=EMDR_SPEED_DEFAULT)
        speed_scale = ttk.Scale(speed_frame, from_=EMDR_SPEED_MIN, to=EMDR_SPEED_MAX,
                                 variable=self.speed_var, orient='horizontal', length=150)
        speed_scale.pack(side='left', padx=5)
        speed_scale.bind('<Motion>', self._on_speed_change)
        
        self.speed_label = ttk.Label(speed_frame, text="1.0 Hz", width=8)
        self.speed_label.pack(side='left')
        
        # Speed presets
        preset_frame = ttk.Frame(left_col)
        preset_frame.pack(fill='x', pady=5)
        
        for speed, label in [(0.5, "Slow"), (1.0, "EMDR"), (1.5, "Fast"), (2.0, "Rapid")]:
            btn = ttk.Button(preset_frame, text=label, width=6,
                            command=lambda s=speed: self._set_speed(s))
            btn.pack(side='left', padx=2)
        
        # Bilateral Mode
        mode_frame = ttk.Frame(left_col)
        mode_frame.pack(fill='x', pady=5)
        
        ttk.Label(mode_frame, text="Mode:").pack(side='left')
        
        self.mode_var = tk.StringVar(value="standard")
        for mode, desc in BILATERAL_MODES.items():
            rb = ttk.Radiobutton(mode_frame, text=mode.replace('_', ' ').title(),
                                  variable=self.mode_var, value=mode,
                                  command=self._on_mode_change)
            rb.pack(side='left', padx=5)
        
        # Phase offset (for golden_phase mode)
        phase_frame = ttk.Frame(left_col)
        phase_frame.pack(fill='x', pady=5)
        
        ttk.Label(phase_frame, text="φ Phase Offset:").pack(side='left')
        
        self.phase_offset_var = tk.DoubleVar(value=GOLDEN_ANGLE_DEG)
        phase_scale = ttk.Scale(phase_frame, from_=0, to=180,
                                 variable=self.phase_offset_var, orient='horizontal', length=120)
        phase_scale.pack(side='left', padx=5)
        
        self.phase_label = ttk.Label(phase_frame, text=f"{GOLDEN_ANGLE_DEG:.1f}°", width=10)
        self.phase_label.pack(side='left')
        
        # Golden angle preset button
        ttk.Button(phase_frame, text="φ", width=3,
                   command=lambda: self.phase_offset_var.set(GOLDEN_ANGLE_DEG)).pack(side='left', padx=2)
        
        # Amplitude
        amp_frame = ttk.Frame(left_col)
        amp_frame.pack(fill='x', pady=5)
        
        ttk.Label(amp_frame, text="Amplitude:").pack(side='left')
        
        self.amplitude_var = tk.DoubleVar(value=0.7)
        amp_scale = ttk.Scale(amp_frame, from_=0.0, to=1.0,
                               variable=self.amplitude_var, orient='horizontal', length=150)
        amp_scale.pack(side='left', padx=5)
        
        self.amp_label = ttk.Label(amp_frame, text="70%", width=6)
        self.amp_label.pack(side='left')
        
        # Play/Stop buttons
        btn_frame = ttk.Frame(left_col)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Start Bilateral", 
                                    command=self._start_bilateral)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", 
                                    command=self._stop_bilateral, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Bilateral visualization
        viz_frame = ttk.LabelFrame(left_col, text="Bilateral Visualization", padding=5)
        viz_frame.pack(fill='x', pady=10)
        
        self.viz_canvas = tk.Canvas(viz_frame, height=60, bg='black')
        self.viz_canvas.pack(fill='x')
        
        # Draw initial state
        self._draw_bilateral_viz(0.0)
        
        # ══════════════════════════════════════════════════════════════════════
        # RIGHT COLUMN - ANNEALING JOURNEYS
        # ══════════════════════════════════════════════════════════════════════
        right_col = ttk.LabelFrame(columns, text="Trauma Annealing Journeys", padding=10)
        right_col.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Journey selection
        journey_frame = ttk.Frame(right_col)
        journey_frame.pack(fill='x', pady=5)
        
        ttk.Label(journey_frame, text="Program:").pack(side='left')
        
        self.journey_var = tk.StringVar(value="gentle_release")
        journey_combo = ttk.Combobox(journey_frame, textvariable=self.journey_var,
                                      state='readonly', width=30)
        journey_combo['values'] = [ANNEALING_PROGRAMS[k]['name'] for k in ANNEALING_PROGRAMS.keys()]
        journey_combo.pack(side='left', padx=5)
        journey_combo.bind('<<ComboboxSelected>>', self._on_journey_change)
        
        # Journey description
        self.journey_desc = ttk.Label(right_col, text="", wraplength=300)
        self.journey_desc.pack(fill='x', pady=5)
        self._update_journey_description()
        
        # Duration
        dur_frame = ttk.Frame(right_col)
        dur_frame.pack(fill='x', pady=5)
        
        ttk.Label(dur_frame, text="Duration:").pack(side='left')
        
        self.duration_var = tk.DoubleVar(value=5.0)
        dur_spin = ttk.Spinbox(dur_frame, from_=1, to=30, increment=1,
                                textvariable=self.duration_var, width=6)
        dur_spin.pack(side='left', padx=5)
        ttk.Label(dur_frame, text="minutes").pack(side='left')
        
        # Journey progress
        prog_frame = ttk.LabelFrame(right_col, text="Journey Progress", padding=5)
        prog_frame.pack(fill='x', pady=10)
        
        self.journey_progress = ttk.Progressbar(prog_frame, mode='determinate', length=250)
        self.journey_progress.pack(fill='x')
        
        self.journey_phase_label = ttk.Label(prog_frame, text="Ready to begin")
        self.journey_phase_label.pack(pady=5)
        
        self.journey_time_label = ttk.Label(prog_frame, text="")
        self.journey_time_label.pack()
        
        # Journey buttons
        journey_btn_frame = ttk.Frame(right_col)
        journey_btn_frame.pack(fill='x', pady=10)
        
        self.journey_play_btn = ttk.Button(journey_btn_frame, text="🌅 Begin Journey",
                                            command=self._start_journey)
        self.journey_play_btn.pack(side='left', padx=5)
        
        self.journey_stop_btn = ttk.Button(journey_btn_frame, text="⏹ End Journey",
                                            command=self._stop_journey, state='disabled')
        self.journey_stop_btn.pack(side='left', padx=5)
        
        # ══════════════════════════════════════════════════════════════════════
        # BOTTOM - INFO PANEL
        # ══════════════════════════════════════════════════════════════════════
        info_frame = ttk.LabelFrame(main_frame, text="ℹ️ EMDR Information", padding=10)
        info_frame.pack(fill='x', pady=(10, 0))
        
        info_text = """
EMDR (Eye Movement Desensitization and Reprocessing) is a clinically validated 
psychotherapy technique. Bilateral audio stimulation (alternating L/R tones) 
is an accepted alternative to eye movements, activating the parasympathetic 
nervous system through vagal tone and mimicking REM sleep memory processing.

Sacred geometry enhancements: Golden ratio (φ) phase relationships between 
hemispheres, Solfeggio healing frequencies, and Fibonacci-timed journeys 
for deep integration.

⚠️ This is a wellness tool, not medical treatment. Consult a professional for trauma work.
"""
        ttk.Label(info_frame, text=info_text.strip(), wraplength=700,
                  justify='left').pack()
    
    # ══════════════════════════════════════════════════════════════════════════
    # UI CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _on_freq_change(self, event=None):
        """Handle frequency preset change."""
        selected = self.freq_var.get()
        for key, data in EMDR_FREQUENCIES.items():
            if data['name'] == selected:
                self.custom_freq.set(data['freq'])
                break
    
    def _on_speed_change(self, event=None):
        """Handle speed slider change."""
        speed = self.speed_var.get()
        self.speed_label.config(text=f"{speed:.1f} Hz")
        self._current_speed = speed
    
    def _on_mode_change(self):
        """Handle bilateral mode change."""
        self._current_mode = self.mode_var.get()
    
    def _on_journey_change(self, event=None):
        """Handle journey program change."""
        self._update_journey_description()
    
    def _set_speed(self, speed: float):
        """Set bilateral speed from preset."""
        self.speed_var.set(speed)
        self.speed_label.config(text=f"{speed:.1f} Hz")
        self._current_speed = speed
    
    def _update_journey_description(self):
        """Update journey description text."""
        selected = self.journey_var.get()
        for key, program in ANNEALING_PROGRAMS.items():
            if program['name'] == selected:
                self.journey_desc.config(text=program['description'])
                self.duration_var.set(program['duration_min'])
                break
    
    # ══════════════════════════════════════════════════════════════════════════
    # BILATERAL VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _draw_bilateral_viz(self, phase: float):
        """
        Draw bilateral visualization showing L/R activation.
        
        Args:
            phase: 0.0 to 1.0, where 0-0.5 is left active, 0.5-1.0 is right active
        """
        canvas = self.viz_canvas
        canvas.delete('all')
        
        w = canvas.winfo_width() if canvas.winfo_width() > 1 else 300
        h = canvas.winfo_height() if canvas.winfo_height() > 1 else 60
        
        # Background
        canvas.create_rectangle(0, 0, w, h, fill='#111111', outline='')
        
        # Calculate L/R intensities with golden fade
        if phase < 0.5:
            # Left dominant
            left_intensity = golden_fade(1.0 - phase * 2, fade_in=True)
            right_intensity = golden_fade(phase * 2, fade_in=True) * 0.2
        else:
            # Right dominant
            left_intensity = golden_fade((phase - 0.5) * 2, fade_in=True) * 0.2
            right_intensity = golden_fade(1.0 - (phase - 0.5) * 2, fade_in=True)
        
        # Left hemisphere (blue/purple)
        left_color = self._intensity_to_color(left_intensity, (100, 100, 255))
        canvas.create_oval(20, 5, 20 + h - 10, h - 5, fill=left_color, outline='#333')
        canvas.create_text(20 + (h - 10) // 2, h // 2, text="L", fill='white', font=('Helvetica', 12, 'bold'))
        
        # Right hemisphere (gold/orange)
        right_color = self._intensity_to_color(right_intensity, (255, 180, 80))
        canvas.create_oval(w - 20 - (h - 10), 5, w - 20, h - 5, fill=right_color, outline='#333')
        canvas.create_text(w - 20 - (h - 10) // 2, h // 2, text="R", fill='white', font=('Helvetica', 12, 'bold'))
        
        # Center line (corpus callosum metaphor)
        canvas.create_line(w // 2, 10, w // 2, h - 10, fill='#444', width=2)
        
        # Wave indicator
        wave_x = int(20 + (h - 10) // 2 + (w - 40 - (h - 10)) * phase)
        canvas.create_oval(wave_x - 8, h // 2 - 8, wave_x + 8, h // 2 + 8, 
                          fill='#00ff88', outline='white', width=2)
    
    def _intensity_to_color(self, intensity: float, base_rgb: tuple) -> str:
        """Convert intensity (0-1) to hex color."""
        r = int(base_rgb[0] * intensity + 30 * (1 - intensity))
        g = int(base_rgb[1] * intensity + 30 * (1 - intensity))
        b = int(base_rgb[2] * intensity + 30 * (1 - intensity))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    # ══════════════════════════════════════════════════════════════════════════
    # BILATERAL AUDIO GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _start_bilateral(self):
        """Start bilateral audio stimulation."""
        if self._is_playing:
            return
        
        self._is_playing = True
        self._bilateral_phase = 0.0
        self._start_time = time.time()
        self._current_freq = self.custom_freq.get()
        self._current_speed = self.speed_var.get()
        self._phase_offset_rad = np.radians(self.phase_offset_var.get())
        
        # Update UI
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="🟢 Playing")
        
        # Start audio - use spectral mode with 2 frequencies (L/R)
        amplitude = self.amplitude_var.get()
        
        # Initial state: left active
        self.audio.start_spectral(
            frequencies=[self._current_freq, self._current_freq],
            amplitudes=[amplitude, 0.0],  # Left active, right silent
            phases=[0.0, self._phase_offset_rad],  # Golden phase offset
            positions=[-1.0, 1.0],  # Hard L/R pan
            master_amplitude=0.9
        )
        
        # Start bilateral animation
        self._bilateral_callback()
    
    def _stop_bilateral(self):
        """Stop bilateral audio stimulation."""
        self._is_playing = False
        
        if self._bilateral_timer:
            self.frame.after_cancel(self._bilateral_timer)
            self._bilateral_timer = None
        
        self.audio.stop()
        
        # Update UI
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="⚪ Stopped")
        self._draw_bilateral_viz(0.0)
    
    def _bilateral_callback(self):
        """Animation callback for bilateral stimulation."""
        if not self._is_playing:
            return
        
        # Calculate bilateral phase (0-1 wrapping)
        elapsed = time.time() - self._start_time
        self._bilateral_phase = (elapsed * self._current_speed) % 1.0
        
        # Calculate L/R amplitudes based on mode
        mode = self.mode_var.get()
        amplitude = self.amplitude_var.get()
        
        if mode == "standard":
            # Clean alternation with golden fade
            if self._bilateral_phase < 0.5:
                t = self._bilateral_phase * 2
                left_amp = golden_fade(1.0 - t, fade_in=True) * amplitude
                right_amp = golden_fade(t, fade_in=True) * amplitude * 0.15
            else:
                t = (self._bilateral_phase - 0.5) * 2
                left_amp = golden_fade(t, fade_in=True) * amplitude * 0.15
                right_amp = golden_fade(1.0 - t, fade_in=True) * amplitude
        
        elif mode == "golden_phase":
            # Golden phase offset mode - both play with φ phase difference
            # Creates subtle interference pattern
            base_amp = amplitude * 0.8
            mod = np.sin(self._bilateral_phase * 2 * np.pi)
            left_amp = base_amp * (0.5 + 0.5 * mod)
            right_amp = base_amp * (0.5 - 0.5 * mod)
        
        elif mode == "breathing":
            # Synchronized to ~6 breaths per minute (0.1 Hz)
            # Bilateral at self._current_speed, amplitude follows breath
            breath_phase = (elapsed * 0.1) % 1.0
            breath_envelope = 0.5 + 0.5 * np.sin(breath_phase * 2 * np.pi)
            
            if self._bilateral_phase < 0.5:
                left_amp = amplitude * breath_envelope
                right_amp = amplitude * breath_envelope * 0.2
            else:
                left_amp = amplitude * breath_envelope * 0.2
                right_amp = amplitude * breath_envelope
        
        elif mode == "theta_pulse":
            # Theta (6Hz) carrier modulated by bilateral rate
            theta_phase = (elapsed * 6.0) % 1.0
            carrier = 0.5 + 0.5 * np.sin(theta_phase * 2 * np.pi)
            
            if self._bilateral_phase < 0.5:
                left_amp = amplitude * carrier
                right_amp = amplitude * carrier * 0.1
            else:
                left_amp = amplitude * carrier * 0.1
                right_amp = amplitude * carrier
        
        else:
            # Fallback to standard
            left_amp = amplitude if self._bilateral_phase < 0.5 else 0.0
            right_amp = 0.0 if self._bilateral_phase < 0.5 else amplitude
        
        # Update audio parameters
        self.audio.set_spectral_params(
            frequencies=[self._current_freq, self._current_freq],
            amplitudes=[left_amp, right_amp],
            phases=[0.0, self._phase_offset_rad],
            positions=[-1.0, 1.0]
        )
        
        # Update visualization
        self._draw_bilateral_viz(self._bilateral_phase)
        
        # Update phase label
        self.phase_label.config(text=f"{self.phase_offset_var.get():.1f}°")
        self.amp_label.config(text=f"{int(amplitude * 100)}%")
        
        # Schedule next frame (60fps for smooth visualization)
        self._bilateral_timer = self.frame.after(16, self._bilateral_callback)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ANNEALING JOURNEY METHODS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _start_journey(self):
        """Start a trauma annealing journey."""
        if self._is_journey_playing:
            return
        
        # Find selected program
        selected = self.journey_var.get()
        for key, program in ANNEALING_PROGRAMS.items():
            if program['name'] == selected:
                self._current_program = program
                break
        
        if not self._current_program:
            return
        
        self._is_journey_playing = True
        self._journey_start_time = time.time()
        self._current_phase_index = 0
        
        # Get total duration in seconds
        self._journey_duration = self.duration_var.get() * 60
        
        # Update UI
        self.journey_play_btn.config(state='disabled')
        self.journey_stop_btn.config(state='normal')
        self.play_btn.config(state='disabled')
        self.status_label.config(text="🟢 Journey Active")
        
        # Start audio
        amplitude = self.amplitude_var.get()
        initial_phase = self._current_program['phases'][0]
        
        self.audio.start_spectral(
            frequencies=[initial_phase['freq'], initial_phase['freq']],
            amplitudes=[amplitude, 0.0],
            phases=[0.0, GOLDEN_ANGLE_RAD],
            positions=[-1.0, 1.0],
            master_amplitude=0.9
        )
        
        # Start journey callback
        self._journey_callback()
    
    def _stop_journey(self):
        """Stop the annealing journey."""
        self._is_journey_playing = False
        
        if self._journey_timer:
            self.frame.after_cancel(self._journey_timer)
            self._journey_timer = None
        
        self.audio.stop()
        
        # Update UI
        self.journey_play_btn.config(state='normal')
        self.journey_stop_btn.config(state='disabled')
        self.play_btn.config(state='normal')
        self.status_label.config(text="⚪ Journey Ended")
        self.journey_progress['value'] = 0
        self.journey_phase_label.config(text="Ready to begin")
        self.journey_time_label.config(text="")
        self._draw_bilateral_viz(0.0)
    
    def _journey_callback(self):
        """Animation callback for annealing journey."""
        if not self._is_journey_playing:
            return
        
        elapsed = time.time() - self._journey_start_time
        progress = min(elapsed / self._journey_duration, 1.0)
        
        # Update progress bar
        self.journey_progress['value'] = progress * 100
        
        # Time display
        remaining = self._journey_duration - elapsed
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        self.journey_time_label.config(text=f"{mins}:{secs:02d} remaining")
        
        # Find current phase
        phases = self._current_program['phases']
        cumulative = 0.0
        current_phase = phases[-1]
        phase_progress = 1.0
        
        for i, phase in enumerate(phases):
            phase_start = cumulative
            phase_end = cumulative + phase['duration_pct']
            
            if progress < phase_end:
                current_phase = phase
                self._current_phase_index = i
                phase_progress = (progress - phase_start) / phase['duration_pct']
                break
            
            cumulative = phase_end
        
        # Update phase label with emoji
        phase_name = current_phase['name']
        self.journey_phase_label.config(
            text=f"Phase {self._current_phase_index + 1}/{len(phases)}: {phase_name}"
        )
        
        # Get phase parameters
        freq = current_phase['freq']
        speed = current_phase['speed']
        amplitude = self.amplitude_var.get()
        
        # Smooth frequency transition between phases
        if self._current_phase_index < len(phases) - 1:
            next_phase = phases[self._current_phase_index + 1]
            # Golden fade transition in last 20% of phase
            if phase_progress > 0.8:
                blend = golden_fade((phase_progress - 0.8) * 5, fade_in=True)
                freq = freq + (next_phase['freq'] - freq) * blend
                speed = speed + (next_phase['speed'] - speed) * blend
        
        # Calculate bilateral phase
        self._bilateral_phase = (elapsed * speed) % 1.0
        
        # Calculate L/R amplitudes with golden fade
        if self._bilateral_phase < 0.5:
            t = self._bilateral_phase * 2
            left_amp = golden_fade(1.0 - t, fade_in=True) * amplitude
            right_amp = golden_fade(t, fade_in=True) * amplitude * 0.15
        else:
            t = (self._bilateral_phase - 0.5) * 2
            left_amp = golden_fade(t, fade_in=True) * amplitude * 0.15
            right_amp = golden_fade(1.0 - t, fade_in=True) * amplitude
        
        # Apply phase-specific mode if present
        phase_mode = current_phase.get('phase_mode', 'standard')
        if phase_mode == 'left_lead':
            left_amp *= 1.2
        elif phase_mode == 'right_lead':
            right_amp *= 1.2
        elif phase_mode == 'coherent':
            # Both hemispheres at same amplitude
            avg = (left_amp + right_amp) / 2
            left_amp = right_amp = avg
        elif phase_mode == 'golden_sync':
            # Golden ratio amplitude relationship
            total = left_amp + right_amp
            left_amp = total * PHI_CONJUGATE
            right_amp = total * (1 - PHI_CONJUGATE)
        
        # Update audio
        self.audio.set_spectral_params(
            frequencies=[freq, freq],
            amplitudes=[left_amp, right_amp],
            phases=[0.0, GOLDEN_ANGLE_RAD],
            positions=[-1.0, 1.0]
        )
        
        # Update visualization
        self._draw_bilateral_viz(self._bilateral_phase)
        
        # Check completion
        if progress >= 1.0:
            self._is_journey_playing = False
            self.journey_phase_label.config(text="✨ Journey Complete!")
            self.journey_play_btn.config(state='normal')
            self.journey_stop_btn.config(state='disabled')
            self.play_btn.config(state='normal')
            self.status_label.config(text="⚪ Journey Complete")
            
            # Gentle fade out
            self.frame.after(2000, lambda: self.audio.stop() if not self._is_playing else None)
            return
        
        # Schedule next frame (30fps)
        self._journey_timer = self.frame.after(33, self._journey_callback)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  EMDR TAB - STANDALONE TEST                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create test window
    root = tk.Tk()
    root.title("EMDR Tab Test")
    root.geometry("800x600")
    
    # Create mock audio engine
    class MockAudioEngine:
        def start_spectral(self, *args, **kwargs):
            print(f"start_spectral: {args}")
        def set_spectral_params(self, *args, **kwargs):
            pass  # Called frequently, don't spam
        def stop(self):
            print("stop()")
    
    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)
    
    tab = EMDRTab(frame, MockAudioEngine())
    
    root.mainloop()
