"""
Vibroacoustic Tab - Sound therapy on wooden board with dual exciters.

Physical setup:
- Spruce board: 1950mm × 600mm × 10mm on springs
- Exciters: HEAD (0mm) and FEET (1950mm) on short edges
- Listener: lying centered, ears ~150mm from head edge
- Springs: 5× (4 corners + 1 center), 15-20kg each for floor decoupling

"Feel the sound travel through your body"
"""

import tkinter as tk
from tkinter import ttk
import time
import numpy as np

# Try importing from core module, fallback to local definitions
try:
    from core.audio_engine import AudioEngine
    from core.golden_math import PHI, PHI_CONJUGATE, golden_fade
except ImportError:
    # Fallback definitions
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
    
    # AudioEngine will be passed as parameter
    AudioEngine = None

# Import soundboard panning
try:
    from soundboard_panning import SoundboardConfig, SoundboardPanner, calculate_panning
    HAS_SOUNDBOARD = True
except ImportError:
    HAS_SOUNDBOARD = False
    SoundboardConfig = None
    SoundboardPanner = None
    calculate_panning = None

# Audio constants
SAMPLE_RATE = 44100


class VibroacousticTab:
    """
    Vibroacoustic soundboard panning for therapy table with 2 exciters.
    
    Physical setup:
    - Spruce board: 2000mm × 600mm × 10mm on springs
    - Exciters: HEAD (0mm) and FEET (2000mm) on short edges
    - Listener: lying centered, ears ~150mm from head edge
    - Springs: 5× (4 corners + 1 center), 15-20kg each for floor decoupling
    
    Panning model uses:
    - ITD (Interaural Time Difference): Delay based on sound velocity in spruce
    - ILD (Interaural Level Difference): Equal-power + soft distance attenuation
    
    "Feel the sound travel through your body"
    """
    
    def __init__(self, parent, audio_engine):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        # Initialize soundboard panner
        if HAS_SOUNDBOARD:
            self.config = SoundboardConfig(sample_rate=SAMPLE_RATE)
            self.panner = SoundboardPanner(self.config)
        else:
            self.config = None
            self.panner = None
        
        # State
        self.pan_position = tk.DoubleVar(value=0.0)  # -1 (head) to +1 (feet)
        self.frequency = tk.DoubleVar(value=432.0)   # Base frequency
        self.amplitude = tk.DoubleVar(value=0.7)
        self.waveform = tk.StringVar(value="sine")
        
        # Auto-sweep mode
        self.sweep_enabled = tk.BooleanVar(value=False)
        self.sweep_duration = tk.DoubleVar(value=10.0)  # seconds per sweep
        self.sweep_mode = tk.StringVar(value="sine")  # sine, linear, golden
        
        # Sweep state
        self._sweep_timer = None
        self._sweep_start_time = None
        self._is_sweeping = False
        
        # ═══════════════════════════════════════════════════════════════════
        # CHAKRA SUNRISE STATE
        # ═══════════════════════════════════════════════════════════════════
        self._is_chakra_playing = False
        self._chakra_timer = None
        self._chakra_start_time = None
        self._journey_duration = 60.0  # Default 1 minute
        
        # Three frequencies for Chakra Sunrise
        self._freq_fourth = 0.0    # Perfect 4th (base × 4/3)
        self._freq_root = 0.0      # Root/fundamental (base × 1)
        self._freq_octave = 0.0    # Octave (base × 2)
        
        # Amplitude and pan for each frequency (smoothly interpolated by AudioEngine)
        self._fourth_amp = 0.0
        self._fourth_pan = 0.0
        self._root_amp = 0.0
        self._root_pan = 0.0
        self._octave_amp = 0.0
        self._octave_pan = 0.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI"""
        if not HAS_SOUNDBOARD:
            ttk.Label(self.frame, text="⚠️ soundboard_panning.py not found").pack(pady=50)
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL - Controls
        # ═══════════════════════════════════════════════════════════════════
        left_frame = ttk.LabelFrame(self.frame, text="🪵 Vibroacoustic Controls", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Physical setup info
        info_frame = ttk.LabelFrame(left_frame, text="📐 Physical Setup", padding=5)
        info_frame.pack(fill='x', pady=5)
        
        info_text = f"""Board: {self.config.length_mm:.0f}×{self.config.width_mm:.0f}mm spruce
Velocity: {self.config.velocity_ms:.0f} m/s (along fiber)
Max delay: {self.config.max_delay_ms:.3f} ms ({self.config.max_delay_samples} samples)
Springs: 5× (4 corners + 1 center)"""
        
        ttk.Label(info_frame, text=info_text, font=('Courier', 9), 
                 foreground='#888').pack(anchor='w')
        
        # ═══════════════════════════════════════════════════════════════════
        # PAN CONTROL - Main slider (vertical for head-feet axis)
        # ═══════════════════════════════════════════════════════════════════
        pan_frame = ttk.LabelFrame(left_frame, text="🎚️ Pan Position (Head ↔ Feet)", padding=10)
        pan_frame.pack(fill='x', pady=5)
        
        # Horizontal layout with labels
        pan_row = ttk.Frame(pan_frame)
        pan_row.pack(fill='x', pady=5)
        
        ttk.Label(pan_row, text="🧠 HEAD", font=('Helvetica', 10, 'bold'),
                 foreground='#ff6b6b').pack(side='left', padx=5)
        
        self.pan_scale = ttk.Scale(pan_row, from_=-1, to=1, variable=self.pan_position,
                                   orient='horizontal', length=250,
                                   command=self._on_pan_change)
        self.pan_scale.pack(side='left', padx=10)
        
        ttk.Label(pan_row, text="🦶 FEET", font=('Helvetica', 10, 'bold'),
                 foreground='#4ecdc4').pack(side='left', padx=5)
        
        # Pan value display
        self.pan_label = ttk.Label(pan_frame, text="Pan: 0.00 (CENTER)", 
                                   font=('Courier', 11, 'bold'))
        self.pan_label.pack(pady=5)
        
        # Quick pan presets
        preset_row = ttk.Frame(pan_frame)
        preset_row.pack(fill='x', pady=3)
        
        presets = [
            ("🧠 Head", -1.0),
            ("↑ Upper", -0.5),
            ("● Center", 0.0),
            ("↓ Lower", 0.5),
            ("🦶 Feet", 1.0),
        ]
        
        for name, pan in presets:
            ttk.Button(preset_row, text=name, width=8,
                      command=lambda p=pan: self._set_pan(p)).pack(side='left', padx=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # ITD/ILD DISPLAY
        # ═══════════════════════════════════════════════════════════════════
        itd_frame = ttk.LabelFrame(left_frame, text="📊 ITD/ILD Values", padding=5)
        itd_frame.pack(fill='x', pady=5)
        
        self.itd_label = ttk.Label(itd_frame, text="", font=('Courier', 9))
        self.itd_label.pack(anchor='w')
        self._update_itd_display()
        
        # ═══════════════════════════════════════════════════════════════════
        # SWEEP MODE (Auto-panning)
        # ═══════════════════════════════════════════════════════════════════
        sweep_frame = ttk.LabelFrame(left_frame, text="🌊 Auto-Sweep (Body Wave)", padding=5)
        sweep_frame.pack(fill='x', pady=5)
        
        sweep_toggle = ttk.Frame(sweep_frame)
        sweep_toggle.pack(fill='x', pady=2)
        ttk.Checkbutton(sweep_toggle, text="Enable sweep (sound travels head↔feet)",
                       variable=self.sweep_enabled,
                       command=self._on_sweep_toggle).pack(side='left')
        
        sweep_params = ttk.Frame(sweep_frame)
        sweep_params.pack(fill='x', pady=2)
        
        ttk.Label(sweep_params, text="Duration:").pack(side='left')
        for secs in [5, 10, 30, 60]:
            ttk.Button(sweep_params, text=f"{secs}s", width=4,
                      command=lambda s=secs: self.sweep_duration.set(s)).pack(side='left', padx=2)
        
        sweep_mode_row = ttk.Frame(sweep_frame)
        sweep_mode_row.pack(fill='x', pady=2)
        ttk.Label(sweep_mode_row, text="Mode:").pack(side='left')
        for mode in ["sine", "linear", "golden"]:
            ttk.Radiobutton(sweep_mode_row, text=mode, variable=self.sweep_mode,
                           value=mode).pack(side='left', padx=5)
        
        # Sweep progress
        self.sweep_progress = ttk.Progressbar(sweep_frame, mode='determinate', length=250)
        self.sweep_progress.pack(fill='x', pady=3)
        
        # ═══════════════════════════════════════════════════════════════════
        # CHAKRA SUNRISE - 3-frequency convergence journey
        # ═══════════════════════════════════════════════════════════════════
        chakra_frame = ttk.LabelFrame(left_frame, text="🌅 Chakra Sunrise Journey", padding=5)
        chakra_frame.pack(fill='x', pady=5)
        
        # Description
        desc_text = """3 frequencies converge at solar plexus:
• Perfect 4th (×4/3) - emerges at belly
• Root (×1) - rises from feet  
• Octave (×2) - descends from head"""
        ttk.Label(chakra_frame, text=desc_text, font=('Helvetica', 9),
                 foreground='#888', justify='left').pack(anchor='w', pady=2)
        
        # Duration selector
        dur_row = ttk.Frame(chakra_frame)
        dur_row.pack(fill='x', pady=3)
        ttk.Label(dur_row, text="Duration:").pack(side='left')
        
        self.chakra_duration = tk.DoubleVar(value=60.0)
        for label, secs in [("30s", 30), ("1m", 60), ("2m", 120), ("5m", 300)]:
            ttk.Button(dur_row, text=label, width=4,
                      command=lambda s=secs: self.chakra_duration.set(s)).pack(side='left', padx=2)
        
        # Base frequency for chakra
        base_row = ttk.Frame(chakra_frame)
        base_row.pack(fill='x', pady=3)
        ttk.Label(base_row, text="Base freq:").pack(side='left')
        self.chakra_base_freq = tk.DoubleVar(value=128.0)  # C3 area
        ttk.Entry(base_row, textvariable=self.chakra_base_freq, width=6).pack(side='left', padx=3)
        ttk.Label(base_row, text="Hz").pack(side='left')
        
        # Frequency display
        self.chakra_freq_label = ttk.Label(chakra_frame, text="", font=('Courier', 9))
        self.chakra_freq_label.pack(anchor='w', pady=2)
        self._update_chakra_freq_display()
        
        # Bind base freq change
        self.chakra_base_freq.trace_add('write', lambda *args: self._update_chakra_freq_display())
        
        # Play/Stop buttons
        chakra_btn_row = ttk.Frame(chakra_frame)
        chakra_btn_row.pack(fill='x', pady=5)
        
        self.chakra_play_btn = ttk.Button(chakra_btn_row, text="🌅 Begin Journey", 
                                          command=self._start_chakra_journey)
        self.chakra_play_btn.pack(side='left', padx=5)
        
        self.chakra_stop_btn = ttk.Button(chakra_btn_row, text="⏹ Stop", 
                                          command=self._stop_chakra_journey, state='disabled')
        self.chakra_stop_btn.pack(side='left', padx=5)
        
        # Progress bar and state
        self.chakra_progress = ttk.Progressbar(chakra_frame, mode='determinate', length=250)
        self.chakra_progress.pack(fill='x', pady=3)
        
        self.chakra_state_label = ttk.Label(chakra_frame, text="Ready", 
                                            font=('Helvetica', 10), foreground='#666')
        self.chakra_state_label.pack(anchor='w')
        
        # ═══════════════════════════════════════════════════════════════════
        # FREQUENCY & AMPLITUDE
        # ═══════════════════════════════════════════════════════════════════
        freq_frame = ttk.LabelFrame(left_frame, text="🎵 Sound Parameters", padding=5)
        freq_frame.pack(fill='x', pady=5)
        
        freq_row = ttk.Frame(freq_frame)
        freq_row.pack(fill='x', pady=2)
        ttk.Label(freq_row, text="Frequency (Hz):").pack(side='left')
        ttk.Entry(freq_row, textvariable=self.frequency, width=8).pack(side='left', padx=5)
        ttk.Scale(freq_row, from_=20, to=500, variable=self.frequency,
                  orient='horizontal', length=150,
                  command=self._on_freq_change).pack(side='left')
        
        # Frequency presets (therapeutic)
        freq_presets = ttk.Frame(freq_frame)
        freq_presets.pack(fill='x', pady=2)
        for name, freq in [("40Hz γ", 40), ("100Hz", 100), ("174Hz", 174), 
                           ("432Hz", 432), ("528Hz", 528)]:
            ttk.Button(freq_presets, text=name, width=6,
                      command=lambda f=freq: self._set_frequency(f)).pack(side='left', padx=2)
        
        amp_row = ttk.Frame(freq_frame)
        amp_row.pack(fill='x', pady=2)
        ttk.Label(amp_row, text="Amplitude:").pack(side='left')
        ttk.Scale(amp_row, from_=0, to=1, variable=self.amplitude,
                  orient='horizontal', length=150,
                  command=self._on_amp_change).pack(side='left', padx=5)
        
        wave_row = ttk.Frame(freq_frame)
        wave_row.pack(fill='x', pady=2)
        ttk.Label(wave_row, text="Waveform:").pack(side='left')
        for wf in ["sine", "golden"]:
            ttk.Radiobutton(wave_row, text=wf, variable=self.waveform,
                           value=wf, command=self._on_waveform_change).pack(side='left', padx=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # PLAYBACK BUTTONS
        # ═══════════════════════════════════════════════════════════════════
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ PLAY", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # RIGHT PANEL - Visualization
        # ═══════════════════════════════════════════════════════════════════
        right_frame = ttk.LabelFrame(self.frame, text="🛏️ Soundboard Visualization", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=300, height=450, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        self.status_var = tk.StringVar(value="Configure and press PLAY")
        ttk.Label(right_frame, textvariable=self.status_var).pack()
        
        # Initial draw
        self._draw_soundboard()
    
    def _set_pan(self, pan: float):
        """Set pan position"""
        self.pan_position.set(pan)
        self._on_pan_change(None)
    
    def _set_frequency(self, freq: float):
        """Set frequency"""
        self.frequency.set(freq)
        self._on_freq_change(None)
    
    def _on_pan_change(self, event):
        """Handle pan change"""
        pan = self.pan_position.get()
        
        # Update label
        if pan < -0.8:
            pos_name = "HEAD"
        elif pan > 0.8:
            pos_name = "FEET"
        elif abs(pan) < 0.1:
            pos_name = "CENTER"
        else:
            pos_name = "↑" if pan < 0 else "↓"
        
        self.pan_label.config(text=f"Pan: {pan:+.2f} ({pos_name})")
        
        # Update ITD/ILD display
        self._update_itd_display()
        
        # Update visualization
        self._draw_soundboard()
        
        # Update panner and audio if playing
        if self.panner:
            self.panner.pan = pan
        
        if self.audio.is_playing():
            self._update_audio()
    
    def _on_freq_change(self, event):
        """Handle frequency change"""
        if self.audio.is_playing():
            self._update_audio()
    
    def _on_amp_change(self, event):
        """Handle amplitude change"""
        if self.audio.is_playing():
            self.audio.set_amplitude(self.amplitude.get())
    
    def _on_waveform_change(self):
        """Handle waveform change"""
        if self.audio.is_playing():
            self.audio.set_waveform(self.waveform.get())
    
    def _on_sweep_toggle(self):
        """Handle sweep toggle"""
        if self.sweep_enabled.get() and self.audio.is_playing():
            self._start_sweep()
        else:
            self._stop_sweep()
    
    def _update_itd_display(self):
        """Update ITD/ILD value display"""
        if not self.config:
            return
        
        pan = self.pan_position.get()
        params = calculate_panning(pan, self.config)
        
        text = f"""HEAD exciter:  Delay {params['delay_head_ms']:.3f} ms | Gain {params['gain_head_db']:+.1f} dB
FEET exciter:  Delay {params['delay_feet_ms']:.3f} ms | Gain {params['gain_feet_db']:+.1f} dB"""
        
        self.itd_label.config(text=text)
    
    def _update_audio(self):
        """Update audio with current panning parameters"""
        if not self.panner:
            return
        
        pan = self.pan_position.get()
        freq = self.frequency.get()
        amp = self.amplitude.get()
        
        params = calculate_panning(pan, self.config)
        
        # For soundboard mode, we send to HEAD and FEET exciters
        # These go to LEFT and RIGHT channels respectively
        # HEAD = LEFT channel, FEET = RIGHT channel
        
        # Create stereo output with ITD delays applied
        # Since AudioEngine doesn't support per-channel delay natively,
        # we'll use spectral mode with 2 frequencies (same freq, different timing)
        # and position them hard left/right
        
        # For now, use simple amplitude panning (ILD only)
        # Full ITD would require modifying AudioEngine's callback
        
        frequencies = [freq, freq]
        amplitudes = [params['gain_head'] * amp, params['gain_feet'] * amp]
        phases = [0.0, 0.0]  # Same phase for both (ITD would offset this)
        positions = [-1.0, 1.0]  # HEAD=left, FEET=right
        
        self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
    
    def _draw_soundboard(self):
        """Draw top-down view of soundboard with body silhouette"""
        self.canvas.delete('all')
        
        # Canvas dimensions
        cw, ch = 300, 450
        
        # Board dimensions in pixels (scaled)
        board_w = 100  # 600mm → 100px
        board_h = 350  # 2000mm → 350px
        board_x = (cw - board_w) // 2
        board_y = (ch - board_h) // 2
        
        # ═══════════════════════════════════════════════════════════════════
        # BOARD OUTLINE
        # ═══════════════════════════════════════════════════════════════════
        self.canvas.create_rectangle(board_x, board_y, 
                                     board_x + board_w, board_y + board_h,
                                     outline='#8b4513', width=3, fill='#2d1810')
        
        # Wood grain lines
        for i in range(5, board_h, 15):
            y = board_y + i
            self.canvas.create_line(board_x + 5, y, board_x + board_w - 5, y,
                                   fill='#3d2817', width=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # SPRINGS (4 corners + 1 center)
        # ═══════════════════════════════════════════════════════════════════
        spring_color = '#666'
        spring_size = 8
        spring_positions = [
            (board_x - 15, board_y + 20),           # Top-left
            (board_x + board_w + 5, board_y + 20),  # Top-right
            (board_x - 15, board_y + board_h - 20), # Bottom-left
            (board_x + board_w + 5, board_y + board_h - 20), # Bottom-right
            (board_x + board_w // 2 - spring_size, board_y + board_h // 2),  # Center
        ]
        
        for sx, sy in spring_positions:
            # Draw spring coil
            self.canvas.create_oval(sx, sy, sx + spring_size * 2, sy + spring_size * 2,
                                   outline=spring_color, width=2)
            self.canvas.create_line(sx + spring_size, sy, 
                                   sx + spring_size, sy + spring_size * 2,
                                   fill=spring_color, width=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # EXCITERS
        # ═══════════════════════════════════════════════════════════════════
        exciter_w, exciter_h = 40, 15
        
        # HEAD exciter (top)
        head_x = board_x + (board_w - exciter_w) // 2
        head_y = board_y - exciter_h + 3
        self.canvas.create_rectangle(head_x, head_y, head_x + exciter_w, head_y + exciter_h,
                                     fill='#ff6b6b', outline='#ff4444', width=2)
        self.canvas.create_text(head_x + exciter_w // 2, head_y - 10,
                               text="🧠 HEAD", fill='#ff6b6b', font=('Helvetica', 9, 'bold'))
        
        # FEET exciter (bottom)
        feet_x = board_x + (board_w - exciter_w) // 2
        feet_y = board_y + board_h - 3
        self.canvas.create_rectangle(feet_x, feet_y, feet_x + exciter_w, feet_y + exciter_h,
                                     fill='#4ecdc4', outline='#3dbdb4', width=2)
        self.canvas.create_text(feet_x + exciter_w // 2, feet_y + exciter_h + 12,
                               text="🦶 FEET", fill='#4ecdc4', font=('Helvetica', 9, 'bold'))
        
        # ═══════════════════════════════════════════════════════════════════
        # BODY SILHOUETTE
        # ═══════════════════════════════════════════════════════════════════
        body_color = '#444'
        
        # Simple body outline (head, torso, legs)
        cx = board_x + board_w // 2
        
        # Head circle
        head_radius = 20
        head_cy = board_y + 50
        self.canvas.create_oval(cx - head_radius, head_cy - head_radius,
                               cx + head_radius, head_cy + head_radius,
                               outline=body_color, width=2)
        
        # Ears (for ITD reference)
        ear_y = head_cy
        self.canvas.create_oval(cx - head_radius - 8, ear_y - 5,
                               cx - head_radius - 2, ear_y + 5,
                               fill='#555', outline=body_color)
        self.canvas.create_oval(cx + head_radius + 2, ear_y - 5,
                               cx + head_radius + 8, ear_y + 5,
                               fill='#555', outline=body_color)
        
        # Torso (rectangle)
        torso_w, torso_h = 50, 120
        torso_y = head_cy + head_radius + 10
        self.canvas.create_rectangle(cx - torso_w // 2, torso_y,
                                     cx + torso_w // 2, torso_y + torso_h,
                                     outline=body_color, width=2)
        
        # Legs
        leg_w = 18
        leg_h = 100
        leg_y = torso_y + torso_h + 5
        
        # Left leg
        self.canvas.create_rectangle(cx - torso_w // 2 + 3, leg_y,
                                     cx - torso_w // 2 + 3 + leg_w, leg_y + leg_h,
                                     outline=body_color, width=2)
        # Right leg
        self.canvas.create_rectangle(cx + torso_w // 2 - 3 - leg_w, leg_y,
                                     cx + torso_w // 2 - 3, leg_y + leg_h,
                                     outline=body_color, width=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # SOUND SOURCE POSITION (virtual)
        # ═══════════════════════════════════════════════════════════════════
        pan = self.pan_position.get()
        
        # Map pan -1..+1 to y position on board
        source_y = board_y + board_h // 2 + int(pan * (board_h // 2 - 30))
        source_x = cx
        
        # Sound waves emanating from source
        wave_color = '#ffd700'
        for r in [15, 25, 35]:
            alpha = 1.0 - r / 50
            self.canvas.create_oval(source_x - r, source_y - r // 2,
                                   source_x + r, source_y + r // 2,
                                   outline=wave_color, width=1)
        
        # Source point
        self.canvas.create_oval(source_x - 8, source_y - 8,
                               source_x + 8, source_y + 8,
                               fill='#ffd700', outline='white', width=2)
        
        # Arrow showing direction
        if abs(pan) > 0.1:
            arrow_dir = 1 if pan > 0 else -1
            arrow_len = 20
            self.canvas.create_line(source_x, source_y,
                                   source_x, source_y + arrow_dir * arrow_len,
                                   fill='#ffd700', width=3, arrow='last')
        
        # ═══════════════════════════════════════════════════════════════════
        # SIGNAL FLOW INDICATORS
        # ═══════════════════════════════════════════════════════════════════
        if self.panner:
            params = calculate_panning(pan, self.config)
            
            # Head exciter intensity
            head_intensity = int(255 * params['gain_head'])
            head_color = f'#ff{head_intensity:02x}{head_intensity:02x}'
            self.canvas.create_rectangle(head_x + 2, head_y + 2,
                                        head_x + exciter_w - 2, head_y + exciter_h - 2,
                                        fill=head_color, outline='')
            
            # Feet exciter intensity
            feet_intensity = int(255 * params['gain_feet'])
            feet_color = f'#{feet_intensity:02x}ff{feet_intensity:02x}'
            self.canvas.create_rectangle(feet_x + 2, feet_y + 2,
                                        feet_x + exciter_w - 2, feet_y + exciter_h - 2,
                                        fill=feet_color, outline='')
        
        # ═══════════════════════════════════════════════════════════════════
        # LABELS
        # ═══════════════════════════════════════════════════════════════════
        # Position label
        if pan < -0.8:
            pos_text = "HEAD"
        elif pan > 0.8:
            pos_text = "FEET"
        elif abs(pan) < 0.1:
            pos_text = "CENTER"
        else:
            pos_text = f"{abs(pan)*100:.0f}% {'↑HEAD' if pan < 0 else '↓FEET'}"
        
        self.canvas.create_text(cw // 2, 15, text=f"Source: {pos_text}",
                               fill='#ffd700', font=('Helvetica', 11, 'bold'))
        
        # Scale indicator
        self.canvas.create_text(cw // 2, ch - 10, 
                               text="2000mm (spruce board on springs)",
                               fill='#666', font=('Courier', 8))
    
    def _play(self):
        """Start playback"""
        freq = self.frequency.get()
        amp = self.amplitude.get()
        pan = self.pan_position.get()
        
        # Initialize panner
        if self.panner:
            self.panner.set_pan_immediate(pan)
        
        params = calculate_panning(pan, self.config)
        
        # Start with spectral mode (2 channels for HEAD/FEET)
        frequencies = [freq, freq]
        amplitudes = [params['gain_head'] * amp, params['gain_feet'] * amp]
        phases = [0.0, 0.0]
        positions = [-1.0, 1.0]  # HEAD=left, FEET=right
        
        self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                  master_amplitude=1.0)
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set(f"🔊 Playing {freq:.0f}Hz | Pan: {pan:+.2f}")
        
        # Start sweep if enabled
        if self.sweep_enabled.get():
            self._start_sweep()
    
    def _stop(self):
        """Stop playback"""
        self._stop_sweep()
        self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.sweep_progress['value'] = 0
        self.status_var.set("Configure and press PLAY")
    
    def _start_sweep(self):
        """Start auto-sweep"""
        self._is_sweeping = True
        self._sweep_start_time = time.time()
        self._sweep_callback()
    
    def _stop_sweep(self):
        """Stop auto-sweep"""
        self._is_sweeping = False
        if self._sweep_timer:
            self.frame.after_cancel(self._sweep_timer)
            self._sweep_timer = None
    
    def _sweep_callback(self):
        """Sweep animation callback"""
        if not self._is_sweeping or not self.audio.is_playing():
            return
        
        elapsed = time.time() - self._sweep_start_time
        duration = self.sweep_duration.get()
        
        # Calculate progress (0 to 1)
        progress = (elapsed % duration) / duration
        
        # Calculate pan based on mode
        mode = self.sweep_mode.get()
        if mode == "sine":
            # Smooth sine wave: head → center → feet → center → head
            pan = np.sin(progress * 2 * np.pi)
        elif mode == "linear":
            # Triangle wave
            if progress < 0.5:
                pan = -1.0 + 4 * progress  # -1 → 1
            else:
                pan = 3.0 - 4 * progress   # 1 → -1
        else:  # golden
            # Golden spiral easing
            t = progress * 2
            if t <= 1:
                pan = -1.0 + 2 * (0.5 * (1 - np.cos(t * np.pi * PHI_CONJUGATE)))
            else:
                t = t - 1
                pan = 1.0 - 2 * (0.5 * (1 - np.cos(t * np.pi * PHI_CONJUGATE)))
        
        # Update pan
        self.pan_position.set(pan)
        self._on_pan_change(None)
        
        # Update progress bar
        self.sweep_progress['value'] = progress * 100
        
        # Schedule next frame (30fps)
        self._sweep_timer = self.frame.after(33, self._sweep_callback)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHAKRA SUNRISE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _update_chakra_freq_display(self):
        """Update the frequency display for Chakra Sunrise"""
        try:
            base = self.chakra_base_freq.get()
            fourth = base * 4 / 3
            octave = base * 2
            self.chakra_freq_label.config(
                text=f"4th: {fourth:.1f}Hz | Root: {base:.1f}Hz | Oct: {octave:.1f}Hz"
            )
        except:
            pass
    
    def _mm_to_pan(self, position_mm: float) -> float:
        """
        Convert body position in mm to pan value (-1 to +1).
        
        Physical setup:
        - Wooden board 1950mm with longitudinal wood fibers
        - Two speaker exciters: one at HEAD (0mm), one at FEET (1950mm)
        - Sound propagates along wood fibers
        - Pan -1.0 = 100% HEAD exciter, Pan +1.0 = 100% FEET exciter
        
        Body positions on board:
        - HEAD:          0mm   → pan = -1.000
        - SOLAR_PLEXUS:  600mm → pan = -0.385
        - SACRAL:        800mm → pan = -0.179
        - FEET:          1750mm → pan = +0.795
        
        The full range ±1.0 is intentional - it matches the physical
        speaker placement and allows true localization along the body.
        """
        return (position_mm / 975.0) - 1.0
    
    def _golden_fade(self, t: float, fade_in: bool = True) -> float:
        """
        Golden ratio based fade curve using φ exponent.
        Creates a smooth S-curve that follows golden proportions:
        - Slow start (breathing in)
        - Natural acceleration through middle  
        - Slow end (settling)
        
        The φ exponent creates a curve that feels organic.
        """
        return golden_fade(t, fade_in)
    
    def _start_chakra_journey(self):
        """Start the Chakra Sunrise journey"""
        if self._is_chakra_playing:
            return
        
        # Get parameters
        base_freq = self.chakra_base_freq.get()
        duration = self.chakra_duration.get()
        
        # Calculate the three frequencies
        self._freq_fourth = base_freq * 4 / 3   # Perfect 4th
        self._freq_root = base_freq             # Root/fundamental  
        self._freq_octave = base_freq * 2       # Octave
        
        self._journey_duration = duration
        
        # Body positions (mm from head edge on 1950mm board)
        HEAD_MM = 0.0
        SOLAR_PLEXUS_MM = 600.0   # Convergence point
        FEET_MM = 1750.0
        
        # Convert to pan values
        PAN_SOLAR_PLEXUS = self._mm_to_pan(SOLAR_PLEXUS_MM)  # ≈ -0.38
        PAN_HEAD = self._mm_to_pan(HEAD_MM)                   # -1.0
        PAN_FEET = self._mm_to_pan(FEET_MM)                   # ≈ +0.79
        
        # Initialize amplitudes and positions
        self._fourth_amp = 0.0
        self._fourth_pan = PAN_SOLAR_PLEXUS  # Always at solar plexus
        
        self._root_amp = 0.0
        self._root_pan = PAN_FEET            # Starts at feet
        
        self._octave_amp = 0.0
        self._octave_pan = PAN_HEAD          # Starts at head
        
        # Update UI
        self.chakra_play_btn.config(state='disabled')
        self.chakra_stop_btn.config(state='normal')
        
        # Start audio engine
        self._is_chakra_playing = True
        self._chakra_start_time = time.time()
        
        # Start with 3 frequencies, all silent initially
        self.audio.start_spectral(
            [self._freq_fourth, self._freq_root, self._freq_octave],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [PAN_SOLAR_PLEXUS, PAN_FEET, PAN_HEAD],
            master_amplitude=0.8
        )
        
        # Start the journey callback
        self._chakra_journey_callback()
        self.chakra_state_label.config(text=f"🌅 Journey started: {duration:.0f}s")
    
    def _stop_chakra_journey(self):
        """Stop the Chakra journey"""
        self._is_chakra_playing = False
        if self._chakra_timer:
            self.frame.after_cancel(self._chakra_timer)
            self._chakra_timer = None
        
        self.audio.stop()
        self.chakra_play_btn.config(state='normal')
        self.chakra_stop_btn.config(state='disabled')
        self.chakra_progress['value'] = 0
        self.chakra_state_label.config(text="Ready")
    
    def _chakra_journey_callback(self):
        """
        Animation callback for Chakra Sunrise journey.
        
        Body positions on 1950mm board:
        - HEAD = 0mm → pan = -1.0
        - SOLAR_PLEXUS = 600mm → pan ≈ -0.38
        - SACRAL = 800mm → pan ≈ -0.18
        - FEET = 1750mm → pan ≈ +0.79
        
        Timeline based on GOLDEN RATIO (φ ≈ 1.618):
        Using φ-based divisions: each phase relates to next by φ
        
        Phase boundaries (golden spiral):
        - Phase 1: 0.000 - 0.146  (14.6%)  4th fades in at SOLAR PLEXUS
        - Phase 2: 0.146 - 0.382  (23.6%)  Root fades in at FEET, rises
        - Phase 3: 0.382 - 0.618  (23.6%)  Root continues to SOLAR PLEXUS  
        - Phase 4: 0.618 - 0.854  (23.6%)  Octave fades in at HEAD, descends
        - Phase 5: 0.854 - 1.000  (14.6%)  CONVERGENCE at SOLAR PLEXUS
        
        The 14.6% and 23.6% come from φ: 1/φ³ ≈ 0.236, 1/φ⁴ ≈ 0.146
        """
        if not self._is_chakra_playing:
            return
        
        elapsed = time.time() - self._chakra_start_time
        duration = self._journey_duration
        progress = min(elapsed / duration, 1.0)
        
        # Update progress bar
        self.chakra_progress['value'] = progress * 100
        
        # Time display
        remaining = duration - elapsed
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        time_str = f"{mins}:{secs:02d}"
        
        # Body positions
        HEAD_MM = 0.0
        SOLAR_PLEXUS_MM = 600.0
        SACRAL_MM = 800.0
        FEET_MM = 1750.0
        
        PAN_HEAD = self._mm_to_pan(HEAD_MM)
        PAN_SOLAR_PLEXUS = self._mm_to_pan(SOLAR_PLEXUS_MM)
        PAN_SACRAL = self._mm_to_pan(SACRAL_MM)
        PAN_FEET = self._mm_to_pan(FEET_MM)
        
        # Golden phase boundaries (symmetric golden proportions: 1:φ:φ:φ:1)
        # Total = 2 + 3φ ≈ 6.854, each unit = 1/6.854
        # This creates: short intro, three golden middle phases, short outro
        GOLDEN_UNIT = 1.0 / (2.0 + 3.0 * PHI)  # ≈ 0.146
        GOLDEN_PHI_UNIT = PHI * GOLDEN_UNIT     # ≈ 0.236
        
        P1_END = GOLDEN_UNIT                                    # ≈ 0.146 (14.6%)
        P2_END = GOLDEN_UNIT + GOLDEN_PHI_UNIT                  # ≈ 0.382 (38.2%)
        P3_END = GOLDEN_UNIT + 2 * GOLDEN_PHI_UNIT              # ≈ 0.618 (61.8%)
        P4_END = GOLDEN_UNIT + 3 * GOLDEN_PHI_UNIT              # ≈ 0.854 (85.4%)
        # P5: 0.854 - 1.0 (14.6% - convergence)
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: 0 - 14.6% - Perfect 4th FADES IN at SOLAR PLEXUS
        # ═══════════════════════════════════════════════════════════════════
        if progress < P1_END:
            phase_progress = progress / P1_END
            
            self._fourth_amp = self._golden_fade(phase_progress, fade_in=True)
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = 0.0
            self._root_pan = PAN_FEET
            self._octave_amp = 0.0
            self._octave_pan = PAN_HEAD
            
            self.chakra_state_label.config(
                text=f"⚡ Phase 1: 4th emerging at solar plexus {self._fourth_amp*100:.0f}% ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: 14.6% - 38.2% - Root FADES IN at FEET, rises toward SACRAL
        # ═══════════════════════════════════════════════════════════════════
        elif progress < P2_END:
            phase_progress = (progress - P1_END) / (P2_END - P1_END)
            
            self._fourth_amp = 1.0
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = self._golden_fade(phase_progress, fade_in=True)
            pan_progress = self._golden_fade(phase_progress, fade_in=True)
            self._root_pan = PAN_FEET + (PAN_SACRAL - PAN_FEET) * pan_progress
            
            self._octave_amp = 0.0
            self._octave_pan = PAN_HEAD
            
            self.chakra_state_label.config(
                text=f"🦶 Phase 2: Root rising from feet {self._root_amp*100:.0f}% ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: 38.2% - 61.8% - Root continues from SACRAL to SOLAR PLEXUS
        # ═══════════════════════════════════════════════════════════════════
        elif progress < P3_END:
            phase_progress = (progress - P2_END) / (P3_END - P2_END)
            
            self._fourth_amp = 1.0
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = 1.0
            pan_progress = self._golden_fade(phase_progress, fade_in=True)
            self._root_pan = PAN_SACRAL + (PAN_SOLAR_PLEXUS - PAN_SACRAL) * pan_progress
            
            self._octave_amp = 0.0
            self._octave_pan = PAN_HEAD
            
            self.chakra_state_label.config(
                text=f"🌊 Phase 3: Root through sacral ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: 61.8% - 85.4% - Octave FADES IN at HEAD, descends to SOLAR PLEXUS
        # ═══════════════════════════════════════════════════════════════════
        elif progress < P4_END:
            phase_progress = (progress - P3_END) / (P4_END - P3_END)
            
            self._fourth_amp = 1.0
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = 1.0
            self._root_pan = PAN_SOLAR_PLEXUS
            
            self._octave_amp = self._golden_fade(phase_progress, fade_in=True)
            pan_progress = self._golden_fade(phase_progress, fade_in=True)
            self._octave_pan = PAN_HEAD + (PAN_SOLAR_PLEXUS - PAN_HEAD) * pan_progress
            
            self.chakra_state_label.config(
                text=f"🧠 Phase 4: Octave descending {self._octave_amp*100:.0f}% ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: 85.4% - 100% - CONVERGENCE at SOLAR PLEXUS
        # Final golden segment: sustained unity
        # ═══════════════════════════════════════════════════════════════════
        else:
            self._fourth_amp = 1.0
            self._root_amp = 1.0
            self._octave_amp = 1.0
            
            self._fourth_pan = PAN_SOLAR_PLEXUS
            self._root_pan = PAN_SOLAR_PLEXUS
            self._octave_pan = PAN_SOLAR_PLEXUS
            
            self.chakra_state_label.config(
                text=f"✨ Phase 5: CONVERGENCE at φ ({time_str})")
        
        # Update audio with current state
        self._update_chakra_audio()
        
        # Check completion
        if progress >= 1.0:
            self._is_chakra_playing = False
            self.chakra_state_label.config(text="🌅 Journey complete! All frequencies united")
            self.chakra_play_btn.config(state='normal')
            self.chakra_stop_btn.config(state='disabled')
            return
        
        # Schedule next frame (30ms for smooth golden transitions)
        self._chakra_timer = self.frame.after(30, self._chakra_journey_callback)
    
    def _update_chakra_audio(self):
        """Update audio parameters for Chakra journey"""
        frequencies = [self._freq_fourth, self._freq_root, self._freq_octave]
        amplitudes = [self._fourth_amp, self._root_amp, self._octave_amp]
        phases = [0.0, 0.0, 0.0]
        positions = [self._fourth_pan, self._root_pan, self._octave_pan]
        
        self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
