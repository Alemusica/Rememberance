"""
Divine Golden Ratio Binaural Generator - GUI Application v2
============================================================

Complete graphical interface with:
- REAL-TIME AUDIO PLAYBACK
- Golden ratio proportioned UI elements
- Live golden spiral visualization
- Interactive parameter controls
- Annealing progress animation
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import struct
import os
import wave
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import subprocess
import platform

# ═══════════════════════════════════════════════════════════════════════════════
# DIVINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2                    # Golden Ratio φ = 1.618033988749895
PHI_CONJUGATE = PHI - 1                        # φ conjugate = 0.618033988749895
PHI_SQUARED = PHI * PHI                        # φ² = 2.618033988749895
PHI_CUBED = PHI * PHI * PHI                    # φ³
SACRED_432 = 432.0                             # Universal harmony frequency

# Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RATIO UI DIMENSIONS
# All dimensions follow φ relationships
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenDimensions:
    """UI dimensions based on golden ratio"""
    
    # Base unit (Fibonacci number)
    UNIT = 8
    
    # Window dimensions (φ ratio)
    WINDOW_HEIGHT = 850
    WINDOW_WIDTH = int(WINDOW_HEIGHT * PHI)  # ≈ 1375
    
    # Main panels (φ split)
    LEFT_PANEL_WIDTH = int(WINDOW_WIDTH * PHI_CONJUGATE)   # ≈ 850 (larger)
    RIGHT_PANEL_WIDTH = int(WINDOW_WIDTH / PHI_SQUARED)    # ≈ 525 (smaller)
    
    # Canvas (φ ratio)
    CANVAS_SIZE = int(LEFT_PANEL_WIDTH * PHI_CONJUGATE)    # ≈ 525
    WAVEFORM_HEIGHT = int(CANVAS_SIZE / PHI_SQUARED)       # ≈ 200
    
    # Controls
    SLIDER_WIDTH = int(RIGHT_PANEL_WIDTH * PHI_CONJUGATE)  # ≈ 325
    BUTTON_HEIGHT = int(UNIT * FIBONACCI[6])               # ≈ 104
    BUTTON_PADDING = int(UNIT * FIBONACCI[4])              # ≈ 40
    
    # Spacing (Fibonacci based)
    PAD_XS = FIBONACCI[3]    # 3
    PAD_SM = FIBONACCI[4]    # 5
    PAD_MD = FIBONACCI[5]    # 8
    PAD_LG = FIBONACCI[6]    # 13
    PAD_XL = FIBONACCI[7]    # 21
    PAD_XXL = FIBONACCI[8]   # 34
    
    # Font sizes (Fibonacci)
    FONT_XS = FIBONACCI[5]       # 8
    FONT_SM = FIBONACCI[6]       # 13
    FONT_MD = FIBONACCI[7]       # 21
    FONT_LG = FIBONACCI[8]       # 34
    FONT_XL = FIBONACCI[9]       # 55


# Color palette (golden/cosmic theme)
COLORS = {
    'bg_dark': '#0a0a14',
    'bg_medium': '#141428',
    'bg_light': '#1e1e3c',
    'gold_primary': '#D4AF37',
    'gold_secondary': '#FFD700',
    'gold_dim': '#8B7355',
    'purple': '#4B0082',
    'purple_light': '#6B238E',
    'cyan': '#00CED1',
    'cyan_dim': '#008B8B',
    'white': '#FFFFFF',
    'text': '#E8E8E8',
    'text_dim': '#888888',
    'accent': '#FF6B6B',
    'success': '#32CD32',
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
# AUDIO ENGINE WITH PLAYBACK
# ═══════════════════════════════════════════════════════════════════════════════

class AudioPlayer:
    """Cross-platform audio player"""
    
    def __init__(self):
        self.is_playing = False
        self.current_process = None
        self.temp_file = None
        
    def play(self, left: np.ndarray, right: np.ndarray, sample_rate: int = 44100):
        """Play audio using system player"""
        self.stop()
        
        # Create temp WAV file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self._save_wav(self.temp_file.name, left, right, sample_rate)
        
        # Play using system command
        system = platform.system()
        
        try:
            if system == 'Darwin':  # macOS
                self.current_process = subprocess.Popen(
                    ['afplay', self.temp_file.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif system == 'Linux':
                # Try different players
                for player in ['aplay', 'paplay', 'play']:
                    try:
                        self.current_process = subprocess.Popen(
                            [player, self.temp_file.name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        break
                    except FileNotFoundError:
                        continue
            elif system == 'Windows':
                # Use PowerShell for Windows
                self.current_process = subprocess.Popen(
                    ['powershell', '-c', f'(New-Object Media.SoundPlayer "{self.temp_file.name}").PlaySync()'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            self.is_playing = True
        except Exception as e:
            print(f"Audio playback error: {e}")
            self.is_playing = False
    
    def stop(self):
        """Stop playback"""
        if self.current_process:
            self.current_process.terminate()
            self.current_process = None
        
        if self.temp_file and os.path.exists(self.temp_file.name):
            try:
                os.unlink(self.temp_file.name)
            except:
                pass
        
        self.is_playing = False
    
    def _save_wav(self, filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int):
        """Save stereo audio as WAV"""
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)
        left = left / max_val * 0.9
        right = right / max_val * 0.9
        
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


class GoldenBinauralEngine:
    """High-precision binaural beat generation engine"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.dtype = np.float64
        self.audio_player = AudioPlayer()
        
    def generate_preview(
        self,
        base_freq: float,
        beat_freq: float,
        duration: float = 5.0,
        amplitude: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a short preview for real-time listening"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        
        # Apply golden envelope
        envelope = np.ones(num_samples, dtype=self.dtype)
        fade_samples = int(num_samples * 0.1)
        
        for i in range(fade_samples):
            envelope[i] = golden_spiral_interpolation(i / fade_samples)
            envelope[-(i+1)] = golden_spiral_interpolation(i / fade_samples)
        
        left = amplitude * envelope * np.sin(2 * np.pi * base_freq * t)
        right = amplitude * envelope * np.sin(
            2 * np.pi * (base_freq + beat_freq) * t + np.pi * PHI_CONJUGATE
        )
        
        return left, right
    
    def play_preview(self, base_freq: float, beat_freq: float, duration: float = 5.0, amplitude: float = 0.8):
        """Generate and play a preview"""
        left, right = self.generate_preview(base_freq, beat_freq, duration, amplitude)
        self.audio_player.play(left, right, self.sample_rate)
    
    def stop_playback(self):
        """Stop current playback"""
        self.audio_player.stop()
    
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
            envelope[i] = golden_spiral_interpolation(i / max(1, attack))
        
        for i in range(min(decay, num_samples)):
            envelope[-(i+1)] = golden_spiral_interpolation(i / max(1, decay))
        
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
            factor = golden_spiral_interpolation(i / max(1, num_samples))
            
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
            progress = golden_spiral_interpolation(i / max(1, num_samples))
            phase_diff = np.pi * progress
            amp = base_amp * (1 - progress)
            
            final_left[i] = amp * np.sin(2 * np.pi * base_frequency * t[i])
            final_right[i] = amp * np.sin(2 * np.pi * base_frequency * t[i] + phase_diff)
        
        left_channel.append(final_left)
        right_channel.append(final_right)
        
        return np.concatenate(left_channel), np.concatenate(right_channel)


def save_wav(filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int = 44100):
    """Save stereo audio as WAV file"""
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)
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
    
    def __init__(self, parent, size, **kwargs):
        super().__init__(parent, width=size, height=size, 
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        
        self.size = size
        self.center_x = size // 2
        self.center_y = size // 2
        
        # Pre-compute spiral
        self.spiral_x, self.spiral_y = golden_spiral_points(400)
        
        # Animation state
        self.phase = 0.0
        self.amplitude = 1.0
        self.beat_freq = 10.0
        self.is_playing = False
        
        self.draw_base()
    
    def draw_base(self):
        """Draw base elements"""
        # Background circles (golden proportions)
        for i in range(5, 0, -1):
            radius = self.size * 0.45 * (PHI_CONJUGATE ** (5-i))
            alpha = int(15 + 10 * i)
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
        scale = self.size * 0.4
        
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
    
    def update_waves(self, phase: float, amplitude: float, beat_freq: float, is_playing: bool = False):
        """Update wave visualization"""
        self.phase = phase
        self.amplitude = amplitude
        self.beat_freq = beat_freq
        self.is_playing = is_playing
        
        self.delete("waves")
        self.delete("indicator")
        self.delete("playing")
        
        scale = self.size * 0.35 * amplitude
        
        # Wave thickness based on playing state
        wave_width = 4 if is_playing else 2
        
        # Left wave (gold)
        points_l = []
        wave_len = 120
        for i in range(wave_len):
            x = self.center_x + (i - wave_len//2) * (self.size / 200)
            y = self.center_y + np.sin(i * 0.15 + phase) * scale * 0.6
            points_l.extend([x, y])
        
        if len(points_l) >= 4:
            self.create_line(points_l, fill=COLORS['gold_primary'], 
                           width=wave_width, smooth=True, tags="waves")
        
        # Right wave (cyan) - offset by golden phase
        points_r = []
        for i in range(wave_len):
            x = self.center_x + (i - wave_len//2) * (self.size / 200)
            y = self.center_y + np.sin(i * 0.15 + phase + np.pi * PHI_CONJUGATE) * scale * 0.6
            points_r.extend([x, y])
        
        if len(points_r) >= 4:
            self.create_line(points_r, fill=COLORS['cyan'], 
                           width=wave_width, smooth=True, tags="waves")
        
        # Phase indicator
        indicator_idx = int((phase / (2 * np.pi)) * len(self.spiral_x)) % len(self.spiral_x)
        ix = self.center_x + self.spiral_x[indicator_idx] * self.size * 0.4
        iy = self.center_y + self.spiral_y[indicator_idx] * self.size * 0.4
        
        indicator_size = 10 if is_playing else 6
        glow_color = COLORS['success'] if is_playing else COLORS['gold_secondary']
        
        # Glow effect when playing
        if is_playing:
            self.create_oval(
                ix - indicator_size*2, iy - indicator_size*2, 
                ix + indicator_size*2, iy + indicator_size*2,
                fill='', outline=glow_color, width=2,
                tags="indicator"
            )
        
        self.create_oval(
            ix - indicator_size, iy - indicator_size, 
            ix + indicator_size, iy + indicator_size,
            fill=glow_color, outline=COLORS['white'], width=2,
            tags="indicator"
        )
        
        # Center phi symbol
        self.delete("phi")
        phi_color = COLORS['success'] if is_playing else COLORS['gold_primary']
        self.create_text(
            self.center_x, self.center_y,
            text="φ", font=("Helvetica", int(self.size/20), "bold"),
            fill=phi_color, tags="phi"
        )
        
        # Playing indicator text
        if is_playing:
            self.create_text(
                self.center_x, self.size - 20,
                text="♪ PLAYING ♪", font=("Helvetica", 12, "bold"),
                fill=COLORS['success'], tags="playing"
            )


class WaveformDisplay(tk.Canvas):
    """Real-time waveform display with golden proportions"""
    
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_medium'], highlightthickness=2,
                        highlightbackground=COLORS['gold_dim'], **kwargs)
        
        self.width = width
        self.height = height
        self.center_y = height // 2
        
        self._draw_grid()
    
    def _draw_grid(self):
        """Draw background grid"""
        # Center line
        self.create_line(0, self.center_y, self.width, self.center_y,
                        fill=COLORS['gold_dim'], dash=(4, 4), tags="grid")
        
        # Golden division lines
        y1 = int(self.center_y - self.height * PHI_CONJUGATE / 2)
        y2 = int(self.center_y + self.height * PHI_CONJUGATE / 2)
        self.create_line(0, y1, self.width, y1, fill=COLORS['purple'], 
                        dash=(2, 4), tags="grid")
        self.create_line(0, y2, self.width, y2, fill=COLORS['purple'], 
                        dash=(2, 4), tags="grid")
        
        # Labels
        self.create_text(10, 15, text="L", fill=COLORS['gold_primary'],
                        font=("Helvetica", 10, "bold"), anchor="nw")
        self.create_text(10, self.height - 15, text="R", fill=COLORS['cyan'],
                        font=("Helvetica", 10, "bold"), anchor="sw")
    
    def update_waveform(self, left: np.ndarray, right: np.ndarray):
        """Update waveform display"""
        self.delete("waveform")
        
        samples_to_show = min(len(left), self.width)
        step = max(1, len(left) // samples_to_show)
        
        left_display = left[::step][:self.width]
        right_display = right[::step][:self.width]
        
        scale = self.height * 0.35
        
        # Left channel
        points_l = []
        for i, sample in enumerate(left_display):
            x = i * (self.width / len(left_display))
            y = self.center_y - sample * scale
            points_l.extend([x, y])
        
        if len(points_l) >= 4:
            self.create_line(points_l, fill=COLORS['gold_primary'],
                           width=1, tags="waveform")
        
        # Right channel
        points_r = []
        for i, sample in enumerate(right_display):
            x = i * (self.width / len(right_display))
            y = self.center_y + sample * scale
            points_r.extend([x, y])
        
        if len(points_r) >= 4:
            self.create_line(points_r, fill=COLORS['cyan'],
                           width=1, tags="waveform")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WITH GOLDEN PROPORTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenBinauralApp:
    """Main GUI Application with Golden Ratio Proportions"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("✦ Divine Golden Ratio Binaural Generator ✦")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Golden ratio window dimensions
        dim = GoldenDimensions
        self.root.geometry(f"{dim.WINDOW_WIDTH}x{dim.WINDOW_HEIGHT}")
        self.root.minsize(800, 600)
        
        # Engine
        self.engine = GoldenBinauralEngine(sample_rate=44100)
        
        # State
        self.is_generating = False
        self.is_playing = False
        self.animation_phase = 0
        self.generated_audio = None
        
        # Variables
        self.base_freq = tk.DoubleVar(value=432.0)
        self.beat_freq = tk.DoubleVar(value=10.0)
        self.num_stages = tk.IntVar(value=8)
        self.amplitude = tk.DoubleVar(value=PHI_CONJUGATE)  # Golden default
        self.preview_duration = tk.DoubleVar(value=8.0)  # Fibonacci
        
        self.setup_ui()
        self.start_animation()
    
    def setup_ui(self):
        """Create the user interface with golden proportions"""
        dim = GoldenDimensions
        
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill="both", expand=True, padx=dim.PAD_XL, pady=dim.PAD_LG)
        
        # ─── HEADER ───
        self.create_header(main_frame)
        
        # ─── CONTENT AREA (Golden split) ───
        content_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content_frame.pack(fill="both", expand=True, pady=dim.PAD_MD)
        
        # Left panel (φ proportion - larger)
        left_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'])
        left_panel.pack(side="left", fill="both", expand=True)
        
        # Spiral canvas (golden square)
        canvas_size = min(dim.CANVAS_SIZE, 550)
        self.spiral_canvas = GoldenSpiralCanvas(left_panel, size=canvas_size)
        self.spiral_canvas.pack(pady=dim.PAD_MD)
        
        # Waveform display (golden rectangle)
        waveform_width = canvas_size
        waveform_height = int(canvas_size / PHI)
        self.waveform = WaveformDisplay(left_panel, width=waveform_width, height=waveform_height)
        self.waveform.pack(pady=dim.PAD_MD)
        
        # Right panel (1/φ proportion - smaller)
        right_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'], width=int(dim.RIGHT_PANEL_WIDTH * 0.8))
        right_panel.pack(side="right", fill="y", padx=(dim.PAD_XL, 0))
        right_panel.pack_propagate(False)
        
        self.create_controls(right_panel)
        
        # ─── FOOTER ───
        self.create_footer(main_frame)
    
    def create_header(self, parent):
        """Create header section with golden typography"""
        dim = GoldenDimensions
        
        header = tk.Frame(parent, bg=COLORS['bg_dark'])
        header.pack(fill="x", pady=(0, dim.PAD_MD))
        
        # Title (Fibonacci font size)
        title = tk.Label(
            header,
            text="✦ DIVINE GOLDEN RATIO BINAURAL ✦",
            font=("Helvetica", dim.FONT_MD, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_dark']
        )
        title.pack()
        
        # Subtitle
        subtitle = tk.Label(
            header,
            text=f"φ = {PHI:.15f}  •  Phase Cancellation Annealing to Silence",
            font=("Helvetica", dim.FONT_SM),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_dark']
        )
        subtitle.pack()
        
        # Golden separator
        sep = tk.Frame(header, bg=COLORS['gold_dim'], height=2)
        sep.pack(fill="x", pady=dim.PAD_MD)
    
    def create_controls(self, parent):
        """Create control panel with golden proportions"""
        dim = GoldenDimensions
        
        # ═══ PLAYBACK SECTION ═══
        play_frame = tk.LabelFrame(
            parent,
            text=" ♪ Live Playback ",
            font=("Helvetica", dim.FONT_SM, "bold"),
            fg=COLORS['success'],
            bg=COLORS['bg_medium'],
            padx=dim.PAD_LG, pady=dim.PAD_LG
        )
        play_frame.pack(fill="x", pady=dim.PAD_SM)
        
        # Play/Stop button (prominent)
        self.play_btn = tk.Button(
            play_frame,
            text="▶ PLAY PREVIEW",
            font=("Helvetica", dim.FONT_SM, "bold"),
            fg=COLORS['bg_dark'],
            bg=COLORS['success'],
            activebackground=COLORS['cyan'],
            relief="flat",
            padx=dim.PAD_LG, pady=dim.PAD_MD,
            command=self.toggle_playback
        )
        self.play_btn.pack(fill="x", pady=dim.PAD_SM)
        
        # Preview duration slider
        self.create_slider(
            play_frame, "Preview Duration (s)", self.preview_duration,
            3, 21, "Fibonacci: 3, 5, 8, 13, 21"  # Fibonacci values
        )
        
        # ═══ PARAMETERS SECTION ═══
        params_frame = tk.LabelFrame(
            parent,
            text=" ✦ Sacred Parameters ",
            font=("Helvetica", dim.FONT_SM, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=dim.PAD_LG, pady=dim.PAD_LG
        )
        params_frame.pack(fill="x", pady=dim.PAD_SM)
        
        # Base frequency
        self.create_slider(
            params_frame, "Base Frequency (Hz)", self.base_freq,
            100, 963, "Sacred: 432, 528, 639, 741, 852"
        )
        
        # Beat frequency
        self.create_slider(
            params_frame, "Beat Frequency (Hz)", self.beat_freq,
            0.5, 40, "Delta<4 Theta<8 Alpha<13 Beta<30"
        )
        
        # Amplitude
        self.create_slider(
            params_frame, "Amplitude", self.amplitude,
            0.1, 1.0, f"Golden default: {PHI_CONJUGATE:.3f}"
        )
        
        # Stages
        self.create_slider(
            params_frame, "Annealing Stages", self.num_stages,
            3, 13, "Fibonacci: 3, 5, 8, 13"
        )
        
        # ═══ PRESETS SECTION ═══
        presets_frame = tk.LabelFrame(
            parent,
            text=" ✦ Divine Presets ",
            font=("Helvetica", dim.FONT_SM, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=dim.PAD_LG, pady=dim.PAD_MD
        )
        presets_frame.pack(fill="x", pady=dim.PAD_SM)
        
        presets = [
            ("🌟 432 Hz Harmony", 432.0, 7.83),
            ("💚 528 Hz Healing", 528.0, 8.0),
            ("🌙 Theta Dreams", 432.0, 5.5),
            ("🌊 Deep Delta", 432.0, 2.5),
            ("☀️ Alpha Relax", 432.0, 10.0),
        ]
        
        for name, base, beat in presets:
            btn = tk.Button(
                presets_frame,
                text=name,
                font=("Helvetica", dim.FONT_XS),
                fg=COLORS['bg_dark'],
                bg=COLORS['gold_secondary'],
                activebackground=COLORS['gold_primary'],
                relief="flat",
                padx=dim.PAD_SM, pady=dim.PAD_XS,
                command=lambda b=base, bt=beat: self.apply_preset(b, bt)
            )
            btn.pack(fill="x", pady=2)
        
        # ═══ GENERATE SECTION ═══
        gen_frame = tk.LabelFrame(
            parent,
            text=" ✦ Full Sequence ",
            font=("Helvetica", dim.FONT_SM, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            padx=dim.PAD_LG, pady=dim.PAD_LG
        )
        gen_frame.pack(fill="x", pady=dim.PAD_SM)
        
        self.generate_btn = tk.Button(
            gen_frame,
            text="✦ GENERATE FULL SEQUENCE ✦",
            font=("Helvetica", dim.FONT_SM, "bold"),
            fg=COLORS['bg_dark'],
            bg=COLORS['gold_primary'],
            activebackground=COLORS['gold_secondary'],
            relief="flat",
            padx=dim.PAD_LG, pady=dim.PAD_MD,
            command=self.generate_sequence
        )
        self.generate_btn.pack(fill="x", pady=dim.PAD_XS)
        
        # Buttons row
        btn_row = tk.Frame(gen_frame, bg=COLORS['bg_medium'])
        btn_row.pack(fill="x", pady=dim.PAD_XS)
        
        self.save_btn = tk.Button(
            btn_row,
            text="💾 Save",
            font=("Helvetica", dim.FONT_XS),
            fg=COLORS['bg_dark'],
            bg=COLORS['gold_dim'],
            relief="flat",
            padx=dim.PAD_MD, pady=dim.PAD_SM,
            command=self.save_audio,
            state="disabled"
        )
        self.save_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        
        self.play_full_btn = tk.Button(
            btn_row,
            text="▶ Play Full",
            font=("Helvetica", dim.FONT_XS),
            fg=COLORS['bg_dark'],
            bg=COLORS['cyan_dim'],
            relief="flat",
            padx=dim.PAD_MD, pady=dim.PAD_SM,
            command=self.play_full_sequence,
            state="disabled"
        )
        self.play_full_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = tk.Label(
            gen_frame,
            textvariable=self.progress_var,
            font=("Helvetica", dim.FONT_XS),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_medium']
        )
        self.progress_label.pack(pady=dim.PAD_XS)
        
        self.progress_bar = ttk.Progressbar(gen_frame, length=200, mode='determinate')
        self.progress_bar.pack(pady=dim.PAD_XS, fill="x")
    
    def create_slider(self, parent, label: str, variable, min_val, max_val, tooltip: str):
        """Create a labeled slider with golden proportions"""
        dim = GoldenDimensions
        
        frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        frame.pack(fill="x", pady=dim.PAD_SM)
        
        # Label row
        label_row = tk.Frame(frame, bg=COLORS['bg_medium'])
        label_row.pack(fill="x")
        
        lbl = tk.Label(
            label_row,
            text=label,
            font=("Helvetica", dim.FONT_XS),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_medium'],
            anchor="w"
        )
        lbl.pack(side="left")
        
        value_lbl = tk.Label(
            label_row,
            textvariable=variable,
            font=("Helvetica", dim.FONT_XS, "bold"),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            width=8
        )
        value_lbl.pack(side="right")
        
        # Slider
        resolution = 0.1 if max_val <= 50 else 1
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient="horizontal",
            variable=variable,
            resolution=resolution,
            font=("Helvetica", 8),
            fg=COLORS['gold_primary'],
            bg=COLORS['bg_medium'],
            troughcolor=COLORS['purple'],
            activebackground=COLORS['gold_secondary'],
            highlightthickness=0,
            showvalue=False,
            command=lambda v: self.on_parameter_change()
        )
        slider.pack(fill="x")
        
        # Tooltip
        tip = tk.Label(
            frame,
            text=tooltip,
            font=("Helvetica", 7),
            fg=COLORS['text_dim'],
            bg=COLORS['bg_medium'],
            anchor="w"
        )
        tip.pack(fill="x")
    
    def create_footer(self, parent):
        """Create footer section"""
        dim = GoldenDimensions
        
        footer = tk.Frame(parent, bg=COLORS['bg_dark'])
        footer.pack(fill="x", side="bottom", pady=(dim.PAD_MD, 0))
        
        # Separator
        sep = tk.Frame(footer, bg=COLORS['gold_dim'], height=1)
        sep.pack(fill="x", pady=(0, dim.PAD_SM))
        
        # Status
        self.status_var = tk.StringVar(value="✦ Ready • Use stereo headphones for binaural effect ✦")
        status = tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Helvetica", dim.FONT_XS),
            fg=COLORS['gold_secondary'],
            bg=COLORS['bg_dark']
        )
        status.pack()
        
        # Golden math info
        info = tk.Label(
            footer,
            text=f"φ² = φ+1 = {PHI_SQUARED:.6f}  •  1/φ = φ-1 = {PHI_CONJUGATE:.6f}  •  Golden Angle = {360/PHI_SQUARED:.2f}°",
            font=("Helvetica", 8),
            fg=COLORS['text_dim'],
            bg=COLORS['bg_dark']
        )
        info.pack()
    
    def apply_preset(self, base: float, beat: float):
        """Apply a preset configuration"""
        self.base_freq.set(base)
        self.beat_freq.set(beat)
        self.amplitude.set(PHI_CONJUGATE)
        self.status_var.set(f"✦ Preset: {base}Hz base • {beat}Hz beat ✦")
        self.on_parameter_change()
    
    def on_parameter_change(self):
        """Called when parameters change"""
        pass  # Animation handles updates
    
    def toggle_playback(self):
        """Toggle audio preview playback"""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start audio preview"""
        self.is_playing = True
        self.play_btn.config(text="⏹ STOP", bg=COLORS['accent'])
        self.status_var.set("♪ Playing binaural preview... Use headphones! ♪")
        
        # Play in thread
        thread = threading.Thread(target=self._playback_thread)
        thread.daemon = True
        thread.start()
    
    def _playback_thread(self):
        """Playback thread"""
        try:
            self.engine.play_preview(
                self.base_freq.get(),
                self.beat_freq.get(),
                self.preview_duration.get(),
                self.amplitude.get()
            )
            
            # Wait for playback to finish
            import time
            time.sleep(self.preview_duration.get() + 0.5)
            
        finally:
            self.root.after(0, self.stop_playback)
    
    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        self.engine.stop_playback()
        self.play_btn.config(text="▶ PLAY PREVIEW", bg=COLORS['success'])
        self.status_var.set("✦ Ready ✦")
    
    def start_animation(self):
        """Start continuous animation"""
        self.animate()
    
    def animate(self):
        """Animation loop"""
        # Update phase
        self.animation_phase += 0.05 * PHI_CONJUGATE
        if self.animation_phase > 2 * np.pi:
            self.animation_phase -= 2 * np.pi
        
        # Update visualization
        self.spiral_canvas.update_waves(
            self.animation_phase,
            self.amplitude.get(),
            self.beat_freq.get(),
            self.is_playing
        )
        
        # Continue (Fibonacci interval)
        self.root.after(34, self.animate)  # ~29fps (Fibonacci)
    
    def generate_sequence(self):
        """Generate the full binaural sequence"""
        if self.is_generating:
            return
        
        self.stop_playback()
        self.is_generating = True
        self.generate_btn.config(state="disabled", text="Generating...")
        self.save_btn.config(state="disabled")
        self.play_full_btn.config(state="disabled")
        self.progress_bar['value'] = 0
        
        thread = threading.Thread(target=self._generate_thread)
        thread.start()
    
    def _generate_thread(self):
        """Generation thread"""
        try:
            def progress_callback(stage, total, message):
                progress = (stage / max(1, total)) * 100
                self.root.after(0, lambda: self.update_progress(progress, message))
            
            left, right = self.engine.generate_annealing_sequence(
                num_stages=self.num_stages.get(),
                base_frequency=self.base_freq.get(),
                progress_callback=progress_callback
            )
            
            self.generated_audio = (left, right)
            
            # Update waveform
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
        self.generate_btn.config(state="normal", text="✦ GENERATE FULL SEQUENCE ✦")
        self.save_btn.config(state="normal")
        self.play_full_btn.config(state="normal")
        self.progress_bar['value'] = 100
        self.progress_var.set(f"✓ Complete! {duration:.1f}s ({duration/60:.1f} min)")
        self.status_var.set("✦ Sequence ready! Save or play the full audio ✦")
    
    def generation_error(self, error: str):
        """Called on generation error"""
        self.is_generating = False
        self.generate_btn.config(state="normal", text="✦ GENERATE FULL SEQUENCE ✦")
        self.progress_var.set(f"Error: {error}")
        messagebox.showerror("Error", error)
    
    def play_full_sequence(self):
        """Play the generated full sequence"""
        if self.generated_audio is None:
            return
        
        self.status_var.set("♪ Playing full sequence... ♪")
        left, right = self.generated_audio
        self.engine.audio_player.play(left, right, self.engine.sample_rate)
    
    def save_audio(self):
        """Save generated audio"""
        if self.generated_audio is None:
            messagebox.showwarning("No Audio", "Generate a sequence first.")
            return
        
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
                messagebox.showinfo("Success", f"Saved:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        """Clean up on close"""
        self.engine.stop_playback()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "★" * 55)
    print("  DIVINE GOLDEN RATIO BINAURAL GENERATOR v2")
    print("  With Real-Time Audio Playback")
    print("★" * 55 + "\n")
    
    app = GoldenBinauralApp()
    app.run()


if __name__ == "__main__":
    main()
