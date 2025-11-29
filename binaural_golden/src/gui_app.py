"""
Divine Golden Ratio Binaural Generator - GUI Application
=========================================================

Complete graphical interface with:
- Real-time golden spiral visualization
- Interactive parameter controls
- Live waveform display
- Annealing progress animation
- Audio generation and playback
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import struct
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

# ═══════════════════════════════════════════════════════════════════════════════
# DIVINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2                    # Golden Ratio φ = 1.618033988749895
PHI_CONJUGATE = PHI - 1                        # φ conjugate = 0.618033988749895
PHI_SQUARED = PHI * PHI                        # φ² = 2.618033988749895
SACRED_432 = 432.0                             # Universal harmony frequency
SACRED_528 = 528.0                             # DNA repair frequency

# Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Color palette (golden/cosmic theme)
COLORS = {
    'bg_dark': '#0a0a14',
    'bg_medium': '#141428',
    'bg_light': '#1e1e3c',
    'gold_primary': '#D4AF37',
    'gold_secondary': '#FFD700',
    'gold_dim': '#8B7355',
    'purple': '#4B0082',
    'cyan': '#00CED1',
    'white': '#FFFFFF',
    'text': '#E8E8E8',
    'accent': '#FF6B6B',
}

# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

def golden_spiral_interpolation(t: float) -> float:
    """Divine golden spiral easing function (NOT linear!)"""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    
    theta = t * np.pi * PHI
    golden_ease = (1 - np.cos(theta * PHI_CONJUGATE)) / 2
    
    x = (t - 0.5) * 4
    golden_sigmoid = 1 / (1 + np.exp(-x * PHI))
    
    result = golden_ease * PHI_CONJUGATE + golden_sigmoid * (1 - PHI_CONJUGATE)
    return np.clip(result, 0.0, 1.0)


def golden_spiral_points(num_points: int = 300, turns: float = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate golden spiral coordinates"""
    theta = np.linspace(0, turns * 2 * np.pi, num_points)
    b = np.log(PHI) / (np.pi / 2)
    r = np.exp(b * theta)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    max_r = np.max(np.sqrt(x**2 + y**2))
    return x / max_r, y / max_r


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenBinauralEngine:
    """High-precision binaural beat generation engine"""
    
    def __init__(self, sample_rate: int = 96000):
        self.sample_rate = sample_rate
        self.dtype = np.float64
        
    def generate_binaural_segment(
        self,
        base_freq: float,
        beat_freq: float,
        duration: float,
        amplitude: float,
        phase: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single binaural beat segment"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        
        left = amplitude * np.sin(2 * np.pi * base_freq * t + phase)
        right = amplitude * np.sin(
            2 * np.pi * (base_freq + beat_freq) * t + 
            phase + np.pi * PHI_CONJUGATE
        )
        
        return left, right
    
    def apply_golden_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Apply golden ratio envelope"""
        num_samples = len(audio)
        envelope = np.ones(num_samples, dtype=self.dtype)
        
        attack = int(num_samples * PHI_CONJUGATE * PHI_CONJUGATE * 0.1)
        decay = int(num_samples * PHI_CONJUGATE * 0.1)
        
        for i in range(min(attack, num_samples)):
            envelope[i] = golden_spiral_interpolation(i / attack)
        
        for i in range(min(decay, num_samples)):
            envelope[-(i+1)] = golden_spiral_interpolation(i / decay)
        
        return audio * envelope
    
    def generate_golden_transition(
        self,
        base_freq: float,
        beat1: float, beat2: float,
        amp1: float, amp2: float,
        phase1: float, phase2: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate golden spiral transition between states"""
        num_samples = int(duration * self.sample_rate)
        left = np.zeros(num_samples, dtype=self.dtype)
        right = np.zeros(num_samples, dtype=self.dtype)
        
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        
        for i in range(num_samples):
            factor = golden_spiral_interpolation(i / num_samples)
            
            beat = beat1 + (beat2 - beat1) * factor
            amp = amp1 + (amp2 - amp1) * factor
            phase = phase1 + (phase2 - phase1) * factor
            
            left[i] = amp * np.sin(2 * np.pi * base_freq * t[i] + phase)
            right[i] = amp * np.sin(
                2 * np.pi * (base_freq + beat) * t[i] + 
                phase + np.pi * PHI_CONJUGATE
            )
        
        return left, right
    
    def generate_annealing_sequence(
        self,
        num_stages: int,
        base_frequency: float,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete annealing sequence to silence"""
        left_channel = []
        right_channel = []
        
        for stage in range(num_stages):
            if progress_callback:
                progress_callback(stage, num_stages, f"Stage {stage+1}/{num_stages}")
            
            # Golden parameters
            beat_freq = base_frequency / (PHI ** (stage + 3))
            fib_idx = min(stage + 5, len(FIBONACCI) - 1)
            duration = FIBONACCI[fib_idx] * PHI_CONJUGATE
            transition_time = duration * PHI_CONJUGATE * PHI_CONJUGATE
            
            # Annealing progress
            annealing = stage / max(1, num_stages - 1)
            phase = np.pi * golden_spiral_interpolation(annealing)
            amplitude = (1 - golden_spiral_interpolation(annealing) * 0.95)
            amplitude *= PHI_CONJUGATE ** (stage * 0.5)
            
            # Generate segment
            left, right = self.generate_binaural_segment(
                base_frequency, beat_freq, duration, amplitude, phase
            )
            left = self.apply_golden_envelope(left)
            right = self.apply_golden_envelope(right)
            
            left_channel.append(left)
            right_channel.append(right)
            
            # Transition
            if stage < num_stages - 1:
                next_beat = base_frequency / (PHI ** (stage + 4))
                next_annealing = (stage + 1) / max(1, num_stages - 1)
                next_phase = np.pi * golden_spiral_interpolation(next_annealing)
                next_amp = (1 - golden_spiral_interpolation(next_annealing) * 0.95)
                next_amp *= PHI_CONJUGATE ** ((stage + 1) * 0.5)
                
                trans_l, trans_r = self.generate_golden_transition(
                    base_frequency,
                    beat_freq, next_beat,
                    amplitude, next_amp,
                    phase, next_phase,
                    transition_time
                )
                left_channel.append(trans_l)
                right_channel.append(trans_r)
        
        # Final silence approach
        if progress_callback:
            progress_callback(num_stages, num_stages, "Phase cancellation...")
        
        silence_duration = FIBONACCI[7] * PHI_CONJUGATE
        num_samples = int(silence_duration * self.sample_rate)
        final_left = np.zeros(num_samples, dtype=self.dtype)
        final_right = np.zeros(num_samples, dtype=self.dtype)
        
        t = np.linspace(0, silence_duration, num_samples, dtype=self.dtype)
        base_amp = PHI_CONJUGATE ** 4
        
        for i in range(num_samples):
            progress = golden_spiral_interpolation(i / num_samples)
            phase_diff = np.pi * progress
            amp = base_amp * (1 - progress)
            
            final_left[i] = amp * np.sin(2 * np.pi * base_frequency * t[i])
            final_right[i] = amp * np.sin(2 * np.pi * base_frequency * t[i] + phase_diff)
        
        left_channel.append(final_left)
        right_channel.append(final_right)
        
        return np.concatenate(left_channel), np.concatenate(right_channel)


def save_wav(filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int = 96000):
    """Save stereo audio as WAV file"""
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left = left / max_val * 0.95
        right = right / max_val * 0.95
    
    left_int = (left * 32767).astype(np.int16)
    right_int = (right * 32767).astype(np.int16)
    
    stereo = np.empty((len(left) + len(right),), dtype=np.int16)
    stereo[0::2] = left_int
    stereo[1::2] = right_int
    
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(stereo) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))
        f.write(struct.pack('<H', 4))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(stereo) * 2))
        f.write(stereo.tobytes())


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenSpiralCanvas(tk.Canvas):
    """Canvas displaying golden spiral and wave visualization"""
    
    def __init__(self, parent, width=500, height=500, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Pre-compute spiral
        self.spiral_x, self.spiral_y = golden_spiral_points(400)
        
        # Animation state
        self.phase = 0.0
        self.amplitude = 1.0
        self.beat_freq = 10.0
        self.is_animating = False
        
        self.draw_base()
    
    def draw_base(self):
        """Draw base elements"""
        # Background gradient effect (circles)
        for i in range(5, 0, -1):
            radius = min(self.width, self.height) * 0.45 * (i / 5)
            alpha = int(20 * (6 - i))
            color = f'#{alpha:02x}{alpha:02x}{int(alpha*1.5):02x}'
            self.create_oval(
                self.center_x - radius, self.center_y - radius,
                self.center_x + radius, self.center_y + radius,
                outline=color, width=1, tags="background"
            )
        
        self.draw_golden_spiral()
    
    def draw_golden_spiral(self):
        """Draw the golden spiral"""
        self.delete("spiral")
        scale = min(self.width, self.height) * 0.4
        
        for i in range(len(self.spiral_x) - 1):
            x1 = self.center_x + self.spiral_x[i] * scale
            y1 = self.center_y + self.spiral_y[i] * scale
            x2 = self.center_x + self.spiral_x[i+1] * scale
            y2 = self.center_y + self.spiral_y[i+1] * scale
            
            # Golden gradient
            progress = i / len(self.spiral_x)
            r = int(212 * (0.3 + 0.7 * progress))
            g = int(175 * (0.3 + 0.7 * progress))
            b = int(55 + 100 * (1 - progress))
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.create_line(x1, y1, x2, y2, fill=color, width=2, tags="spiral")
    
    def update_waves(self, phase: float, amplitude: float, beat_freq: float):
        """Update wave visualization"""
        self.phase = phase
        self.amplitude = amplitude
        self.beat_freq = beat_freq
        
        self.delete("waves")
        self.delete("indicator")
        
        scale = min(self.width, self.height) * 0.35 * amplitude
        
        # Left wave (gold)
        points_l = []
        for i in range(120):
            x = self.center_x + (i - 60) * 3
            y = self.center_y + np.sin(i * 0.15 + phase) * scale * 0.6
            points_l.extend([x, y])
        
        if len(points_l) >= 4:
            self.create_line(points_l, fill=COLORS['gold_primary'], 
                           width=3, smooth=True, tags="waves")
        
        # Right wave (cyan) - offset by golden phase
        points_r = []
        for i in range(120):
            x = self.center_x + (i - 60) * 3
            y = self.center_y + np.sin(i * 0.15 + phase + np.pi * PHI_CONJUGATE) * scale * 0.6
            points_r.extend([x, y])
        
        if len(points_r) >= 4:
            self.create_line(points_r, fill=COLORS['cyan'], 
                           width=3, smooth=True, tags="waves")
        
        # Phase indicator (rotating dot on spiral)
        indicator_idx = int((phase / (2 * np.pi)) * len(self.spiral_x)) % len(self.spiral_x)
        ix = self.center_x + self.spiral_x[indicator_idx] * min(self.width, self.height) * 0.4
        iy = self.center_y + self.spiral_y[indicator_idx] * min(self.width, self.height) * 0.4
        
        self.create_oval(
            ix - 8, iy - 8, ix + 8, iy + 8,
            fill=COLORS['gold_secondary'], outline=COLORS['white'], width=2,
            tags="indicator"
        )
        
        # Center phi symbol
        self.delete("phi")
        self.create_text(
            self.center_x, self.center_y,
            text="φ", font=("Helvetica", 24, "bold"),
            fill=COLORS['gold_primary'], tags="phi"
        )


class WaveformDisplay(tk.Canvas):
    """Real-time waveform display"""
    
    def __init__(self, parent, width=600, height=150, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_medium'], highlightthickness=1,
                        highlightbackground=COLORS['gold_dim'], **kwargs)
        
        self.width = width
        self.height = height
        self.center_y = height // 2
        
        # Draw center line
        self.create_line(0, self.center_y, width, self.center_y,
                        fill=COLORS['gold_dim'], dash=(4, 4))
        
        # Labels
        self.create_text(10, 10, text="L", fill=COLORS['gold_primary'],
                        font=("Helvetica", 10, "bold"), anchor="nw")
        self.create_text(10, height - 10, text="R", fill=COLORS['cyan'],
                        font=("Helvetica", 10, "bold"), anchor="sw")
    
    def update_waveform(self, left: np.ndarray, right: np.ndarray):
        """Update waveform display with audio data"""
        self.delete("waveform")
        
        # Downsample for display
        samples_to_show = min(len(left), self.width)
        step = max(1, len(left) // samples_to_show)
        
        left_display = left[::step][:self.width]
        right_display = right[::step][:self.width]
        
        scale = self.height * 0.4
        
        # Left channel (top half)
        points_l = []
        for i, sample in enumerate(left_display):
            x = i
            y = self.center_y - sample * scale * 0.8
            points_l.extend([x, y])
        
        if len(points_l) >= 4:
            self.create_line(points_l, fill=COLORS['gold_primary'],
                           width=1, tags="waveform")
        
        # Right channel (bottom half)
        points_r = []
        for i, sample in enumerate(right_display):
            x = i
            y = self.center_y + sample * scale * 0.8
            points_r.extend([x, y])
        
        if len(points_r) >= 4:
            self.create_line(points_r, fill=COLORS['cyan'],
                           width=1, tags="waveform")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenBinauralApp:
    """Main GUI Application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("✦ Divine Golden Ratio Binaural Generator ✦")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.geometry("1100x850")
        self.root.resizable(True, True)
        
        # Engine
        self.engine = GoldenBinauralEngine(sample_rate=96000)
        
        # State
        self.is_generating = False
        self.is_animating = False
        self.animation_stage = 0
        self.generated_audio = None
        
        # Variables
        self.base_freq = tk.DoubleVar(value=432.0)
        self.beat_freq = tk.DoubleVar(value=10.0)
        self.num_stages = tk.IntVar(value=8)
        self.amplitude = tk.DoubleVar(value=0.8)
        self.phase = tk.DoubleVar(value=0.0)
        
        self.setup_ui()
        self.start_animation()
    
    def setup_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ─── HEADER ───
        self.create_header(main_frame)
        
        # ─── CONTENT AREA ───
        content_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content_frame.pack(fill="both", expand=True, pady=10)
        
        # Left panel (visualization)
        left_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'])
        left_panel.pack(side="left", fill="both", expand=True)
        
        self.spiral_canvas = GoldenSpiralCanvas(left_panel, width=500, height=500)
        self.spiral_canvas.pack(pady=10)
        
        # Waveform display
        self.waveform = WaveformDisplay(left_panel, width=500, height=120)
        self.waveform.pack(pady=10)
        
        # Right panel (controls)
        right_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'])
        right_panel.pack(side="right", fill="y", padx=(20, 0))
        
        self.create_controls(right_panel)
        
        # ─── FOOTER ───
        self.create_footer(main_frame)
    
    def create_header(self, parent):
        """Create header section"""
        header = tk.Frame(parent, bg=COLORS['bg_dark'])
        header.pack(fill="x", pady=(0, 10))
        
        # Title
        title = tk.Label(
            header,
            text="✦ DIVINE GOLDEN RATIO BINAURAL ✦",
            font=("Helvetica", 22, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_dark']
        )
        title.pack()
        
        # Subtitle with phi
        subtitle = tk.Label(
            header,
            text=f"φ = {PHI:.15f}  •  Phase Cancellation Annealing",
            font=("Helvetica", 11),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_dark']
        )
        subtitle.pack()
        
        # Separator
        sep = tk.Frame(header, bg=COLORS['gold_dim'], height=2)
        sep.pack(fill="x", pady=10)
    
    def create_controls(self, parent):
        """Create control panel"""
        # Parameters section
        params_frame = tk.LabelFrame(
            parent,
            text=" ✦ Sacred Parameters ",
            font=("Helvetica", 12, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=15, pady=15
        )
        params_frame.pack(fill="x", pady=5)
        
        # Base frequency
        self.create_slider(
            params_frame, "Base Frequency (Hz)", self.base_freq,
            100, 1000, "Sacred carrier frequency"
        )
        
        # Beat frequency
        self.create_slider(
            params_frame, "Beat Frequency (Hz)", self.beat_freq,
            0.5, 40, "Binaural beat rate"
        )
        
        # Number of stages
        self.create_slider(
            params_frame, "Annealing Stages", self.num_stages,
            3, 13, "Fibonacci recommended: 5, 8, 13"
        )
        
        # Amplitude
        self.create_slider(
            params_frame, "Amplitude", self.amplitude,
            0.1, 1.0, "Output level"
        )
        
        # Presets section
        presets_frame = tk.LabelFrame(
            parent,
            text=" ✦ Divine Presets ",
            font=("Helvetica", 12, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=15, pady=15
        )
        presets_frame.pack(fill="x", pady=10)
        
        presets = [
            ("432 Hz Harmony", 432.0, 7.83, 8),
            ("528 Hz Healing", 528.0, 8.0, 8),
            ("Theta Dreams", 432.0, 5.5, 5),
            ("Deep Delta", 432.0, 2.5, 13),
            ("Alpha Relax", 432.0, 10.0, 8),
        ]
        
        for i, (name, base, beat, stages) in enumerate(presets):
            btn = tk.Button(
                presets_frame,
                text=name,
                font=("Helvetica", 9),
                fg=COLORS['bg_dark'],
                bg=COLORS['gold_secondary'],
                activebackground=COLORS['gold_primary'],
                relief="flat",
                padx=10, pady=5,
                command=lambda b=base, bt=beat, s=stages: self.apply_preset(b, bt, s)
            )
            btn.pack(fill="x", pady=2)
        
        # Generate section
        gen_frame = tk.LabelFrame(
            parent,
            text=" ✦ Generation ",
            font=("Helvetica", 12, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=15, pady=15
        )
        gen_frame.pack(fill="x", pady=10)
        
        self.generate_btn = tk.Button(
            gen_frame,
            text="✦ GENERATE SEQUENCE ✦",
            font=("Helvetica", 14, "bold"),
            fg=COLORS['bg_dark'],
            bg=COLORS['gold_primary'],
            activebackground=COLORS['gold_secondary'],
            relief="flat",
            padx=20, pady=15,
            command=self.generate_sequence
        )
        self.generate_btn.pack(fill="x", pady=5)
        
        self.save_btn = tk.Button(
            gen_frame,
            text="💾 Save WAV File",
            font=("Helvetica", 11),
            fg=COLORS['bg_dark'],
            bg=COLORS['gold_dim'],
            activebackground=COLORS['gold_secondary'],
            relief="flat",
            padx=15, pady=10,
            command=self.save_audio,
            state="disabled"
        )
        self.save_btn.pack(fill="x", pady=5)
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = tk.Label(
            gen_frame,
            textvariable=self.progress_var,
            font=("Helvetica", 10),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_medium']
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(
            gen_frame,
            length=250,
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
        
        # Info display
        info_frame = tk.LabelFrame(
            parent,
            text=" ✦ Golden Mathematics ",
            font=("Helvetica", 12, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=15, pady=15
        )
        info_frame.pack(fill="x", pady=10)
        
        info_text = f"""φ² = φ + 1 = {PHI_SQUARED:.10f}
1/φ = φ - 1 = {PHI_CONJUGATE:.10f}

Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21...
Ratio → φ as n → ∞

Golden Angle: {360/PHI_SQUARED:.2f}°"""
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=("Courier", 9),
            fg=COLORS['text'],
            bg=COLORS['bg_medium'],
            justify="left"
        )
        info_label.pack()
    
    def create_slider(self, parent, label: str, variable, min_val, max_val, tooltip: str):
        """Create a labeled slider"""
        frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        frame.pack(fill="x", pady=8)
        
        # Label
        lbl = tk.Label(
            frame,
            text=label,
            font=("Helvetica", 10),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_medium'],
            anchor="w"
        )
        lbl.pack(fill="x")
        
        # Slider frame
        slider_frame = tk.Frame(frame, bg=COLORS['bg_medium'])
        slider_frame.pack(fill="x")
        
        # Slider
        resolution = 0.1 if max_val <= 50 else 1
        slider = tk.Scale(
            slider_frame,
            from_=min_val,
            to=max_val,
            orient="horizontal",
            variable=variable,
            resolution=resolution,
            length=200,
            font=("Helvetica", 9),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            troughcolor=COLORS['purple'],
            activebackground=COLORS['gold_secondary'],
            highlightthickness=0,
            command=lambda v: self.on_parameter_change()
        )
        slider.pack(side="left", fill="x", expand=True)
        
        # Value display
        value_lbl = tk.Label(
            slider_frame,
            textvariable=variable,
            font=("Helvetica", 10, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            width=8
        )
        value_lbl.pack(side="right")
        
        # Tooltip
        tip = tk.Label(
            frame,
            text=tooltip,
            font=("Helvetica", 8),
            fg=COLORS['gold_dim'],
            bg=COLORS['bg_medium'],
            anchor="w"
        )
        tip.pack(fill="x")
    
    def create_footer(self, parent):
        """Create footer section"""
        footer = tk.Frame(parent, bg=COLORS['bg_dark'])
        footer.pack(fill="x", side="bottom", pady=(10, 0))
        
        # Separator
        sep = tk.Frame(footer, bg=COLORS['gold_dim'], height=2)
        sep.pack(fill="x", pady=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="✦ Ready • Use stereo headphones for binaural effect ✦")
        status = tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_dark']
        )
        status.pack()
    
    def apply_preset(self, base: float, beat: float, stages: int):
        """Apply a preset configuration"""
        self.base_freq.set(base)
        self.beat_freq.set(beat)
        self.num_stages.set(stages)
        self.amplitude.set(PHI_CONJUGATE)
        self.status_var.set(f"✦ Preset applied: {base}Hz • {beat}Hz beat • {stages} stages ✦")
        self.on_parameter_change()
    
    def on_parameter_change(self):
        """Called when parameters change"""
        self.spiral_canvas.update_waves(
            self.phase.get(),
            self.amplitude.get(),
            self.beat_freq.get()
        )
    
    def start_animation(self):
        """Start continuous animation"""
        self.is_animating = True
        self.animate()
    
    def animate(self):
        """Animation loop"""
        if not self.is_animating:
            return
        
        # Update phase
        current_phase = self.phase.get()
        new_phase = current_phase + 0.05 * PHI_CONJUGATE
        if new_phase > 2 * np.pi:
            new_phase -= 2 * np.pi
        self.phase.set(new_phase)
        
        # Update visualization
        self.spiral_canvas.update_waves(
            new_phase,
            self.amplitude.get(),
            self.beat_freq.get()
        )
        
        # Continue
        self.root.after(30, self.animate)
    
    def generate_sequence(self):
        """Generate the binaural sequence"""
        if self.is_generating:
            return
        
        self.is_generating = True
        self.generate_btn.config(state="disabled", text="Generating...")
        self.save_btn.config(state="disabled")
        self.progress_bar['value'] = 0
        
        # Run in thread
        thread = threading.Thread(target=self._generate_thread)
        thread.start()
    
    def _generate_thread(self):
        """Generation thread"""
        try:
            def progress_callback(stage, total, message):
                progress = (stage / total) * 100
                self.root.after(0, lambda: self.update_progress(progress, message))
            
            left, right = self.engine.generate_annealing_sequence(
                num_stages=self.num_stages.get(),
                base_frequency=self.base_freq.get(),
                progress_callback=progress_callback
            )
            
            self.generated_audio = (left, right)
            
            # Update waveform display
            display_len = min(len(left), 50000)
            self.root.after(0, lambda: self.waveform.update_waveform(
                left[:display_len], right[:display_len]
            ))
            
            duration = len(left) / self.engine.sample_rate
            self.root.after(0, lambda: self.generation_complete(duration))
            
        except Exception as e:
            self.root.after(0, lambda: self.generation_error(str(e)))
    
    def update_progress(self, progress: float, message: str):
        """Update progress display"""
        self.progress_bar['value'] = progress
        self.progress_var.set(message)
    
    def generation_complete(self, duration: float):
        """Called when generation completes"""
        self.is_generating = False
        self.generate_btn.config(state="normal", text="✦ GENERATE SEQUENCE ✦")
        self.save_btn.config(state="normal")
        self.progress_bar['value'] = 100
        self.progress_var.set(f"Complete! Duration: {duration:.1f}s ({duration/60:.1f} min)")
        self.status_var.set("✦ Generation complete! Click Save to export WAV file ✦")
    
    def generation_error(self, error: str):
        """Called on generation error"""
        self.is_generating = False
        self.generate_btn.config(state="normal", text="✦ GENERATE SEQUENCE ✦")
        self.progress_var.set(f"Error: {error}")
        self.status_var.set("✦ Error during generation ✦")
        messagebox.showerror("Generation Error", error)
    
    def save_audio(self):
        """Save generated audio to file"""
        if self.generated_audio is None:
            messagebox.showwarning("No Audio", "Please generate a sequence first.")
            return
        
        # File dialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=f"golden_binaural_{int(self.base_freq.get())}Hz.wav"
        )
        
        if filename:
            try:
                left, right = self.generated_audio
                save_wav(filename, left, right, self.engine.sample_rate)
                self.status_var.set(f"✦ Saved: {os.path.basename(filename)} ✦")
                messagebox.showinfo("Success", f"Audio saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Launch the application"""
    print("\n" + "★" * 60)
    print("  DIVINE GOLDEN RATIO BINAURAL GENERATOR")
    print("  GUI Application")
    print("★" * 60 + "\n")
    
    app = GoldenBinauralApp()
    app.run()


if __name__ == "__main__":
    main()
