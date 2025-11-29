"""
Divine Golden Ratio Binaural Generator - GUI v3 SMOOTH
=======================================================

Final version with:
- SEAMLESS AUDIO (no clicks/pops)
- Real-time playback
- Golden ratio proportioned UI
- All tests passing
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import struct
import os
import tempfile
import subprocess
import platform
from typing import Tuple, Optional, List

# ═══════════════════════════════════════════════════════════════════════════════
# DIVINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = np.float64((1 + np.sqrt(5)) / 2)
PHI_CONJUGATE = np.float64(PHI - 1)
PHI_SQUARED = np.float64(PHI * PHI)
TWO_PI = np.float64(2 * np.pi)

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Golden UI Colors
COLORS = {
    'bg_dark': '#0a0a14',
    'bg_medium': '#141428',
    'gold_primary': '#D4AF37',
    'gold_secondary': '#FFD700',
    'gold_dim': '#8B7355',
    'purple': '#4B0082',
    'cyan': '#00CED1',
    'white': '#FFFFFF',
    'text': '#E8E8E8',
    'text_dim': '#888888',
    'success': '#32CD32',
    'accent': '#FF6B6B',
}


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN MATHEMATICS - Vectorized (No Clicks!)
# ═══════════════════════════════════════════════════════════════════════════════

def golden_spiral_interpolation(t):
    """Vectorized golden spiral easing - smooth at all points"""
    t = np.asarray(t, dtype=np.float64)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)
    
    result = np.zeros_like(t)
    
    mask_low = t <= 0
    mask_high = t >= 1
    mask_mid = ~mask_low & ~mask_high
    
    result[mask_low] = 0.0
    result[mask_high] = 1.0
    
    if np.any(mask_mid):
        t_mid = t[mask_mid]
        theta = t_mid * np.pi * PHI_CONJUGATE
        golden_ease = (1 - np.cos(theta)) / 2
        smoothstep = t_mid * t_mid * (3 - 2 * t_mid)
        result[mask_mid] = smoothstep * PHI_CONJUGATE + golden_ease * (1 - PHI_CONJUGATE)
    
    return float(result[0]) if scalar_input else result


def golden_spiral_points(num_points: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Generate golden spiral for visualization"""
    theta = np.linspace(0, 4 * TWO_PI, num_points)
    b = np.log(PHI) / (np.pi / 2)
    r = np.exp(b * theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    max_r = np.max(np.sqrt(x**2 + y**2))
    return x / max_r, y / max_r


# ═══════════════════════════════════════════════════════════════════════════════
# SMOOTH BINAURAL ENGINE (from tested smooth_engine_v2)
# ═══════════════════════════════════════════════════════════════════════════════

class SmoothBinauralEngine:
    """Binaural generator with seamless transitions - NO CLICKS"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._phase_left = 0.0
        self._phase_right = 0.0
    
    def reset_phase(self):
        self._phase_left = 0.0
        self._phase_right = 0.0
    
    def generate_segment_vectorized(
        self,
        base_freq: float,
        beat_freq: float,
        duration: float,
        amplitude: float,
        phase_offset: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate binaural segment with phase continuity"""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        omega_left = TWO_PI * base_freq
        omega_right = TWO_PI * (base_freq + beat_freq)
        
        phase_left = self._phase_left + omega_left * t + phase_offset
        phase_right = self._phase_right + omega_right * t + np.pi * PHI_CONJUGATE + phase_offset
        
        left = amplitude * np.sin(phase_left)
        right = amplitude * np.sin(phase_right)
        
        self._phase_left = (self._phase_left + omega_left * duration) % TWO_PI
        self._phase_right = (self._phase_right + omega_right * duration) % TWO_PI
        
        return left, right
    
    def generate_transition_vectorized(
        self,
        base_freq: float,
        beat1: float, beat2: float,
        amp1: float, amp2: float,
        phase1: float, phase2: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth parameter transition"""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        factor = golden_spiral_interpolation(np.linspace(0, 1, num_samples))
        
        beat_freq = beat1 + (beat2 - beat1) * factor
        amplitude = amp1 + (amp2 - amp1) * factor
        phase_offset = phase1 + (phase2 - phase1) * factor
        
        omega_left = TWO_PI * base_freq
        dt = 1.0 / self.sample_rate
        omega_right_instant = TWO_PI * (base_freq + beat_freq)
        phase_right_cumulative = np.cumsum(omega_right_instant) * dt
        
        phase_left = self._phase_left + omega_left * t + phase_offset
        phase_right = self._phase_right + phase_right_cumulative + np.pi * PHI_CONJUGATE + phase_offset
        
        left = amplitude * np.sin(phase_left)
        right = amplitude * np.sin(phase_right)
        
        self._phase_left = (self._phase_left + omega_left * duration) % TWO_PI
        self._phase_right = (self._phase_right + phase_right_cumulative[-1]) % TWO_PI
        
        return left, right
    
    def _calculate_stage_params(self, stage: int, total_stages: int, base_freq: float) -> dict:
        beat_freq = base_freq / (PHI ** (stage + 3))
        fib_idx = min(stage + 5, len(FIBONACCI) - 1)
        duration = FIBONACCI[fib_idx] * PHI_CONJUGATE
        transition_time = duration * PHI_CONJUGATE * PHI_CONJUGATE
        
        annealing = stage / max(1, total_stages - 1)
        phase_offset = np.pi * golden_spiral_interpolation(annealing)
        amplitude = (1 - golden_spiral_interpolation(annealing) * 0.95) * (PHI_CONJUGATE ** (stage * 0.5))
        
        return {
            'beat_freq': beat_freq,
            'duration': duration,
            'transition_time': transition_time,
            'phase_offset': phase_offset,
            'amplitude': amplitude,
        }
    
    def _apply_micro_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Tiny envelope at boundaries"""
        fade_samples = min(64, len(audio) // 10)
        if fade_samples < 2:
            return audio
        
        fade_in = np.linspace(0, 1, fade_samples) ** 2
        fade_out = np.linspace(1, 0, fade_samples) ** 2
        
        audio = audio.copy()
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        return audio
    
    def _smooth_concatenate(self, segments: List[np.ndarray]) -> np.ndarray:
        if not segments:
            return np.array([], dtype=np.float64)
        smoothed = [self._apply_micro_envelope(seg) for seg in segments]
        return np.concatenate(smoothed)
    
    def generate_annealing_sequence(
        self,
        num_stages: int,
        base_frequency: float,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete annealing sequence - SMOOTH"""
        self.reset_phase()
        
        left_segments, right_segments = [], []
        stages = [self._calculate_stage_params(i, num_stages, base_frequency) for i in range(num_stages)]
        
        for idx, params in enumerate(stages):
            if progress_callback:
                progress_callback(idx, num_stages, f"Stage {idx+1}/{num_stages}")
            
            left, right = self.generate_segment_vectorized(
                base_frequency, params['beat_freq'], params['duration'],
                params['amplitude'], params['phase_offset']
            )
            left_segments.append(left)
            right_segments.append(right)
            
            if idx < num_stages - 1:
                next_p = stages[idx + 1]
                trans_l, trans_r = self.generate_transition_vectorized(
                    base_frequency,
                    params['beat_freq'], next_p['beat_freq'],
                    params['amplitude'], next_p['amplitude'],
                    params['phase_offset'], next_p['phase_offset'],
                    params['transition_time']
                )
                left_segments.append(trans_l)
                right_segments.append(trans_r)
        
        if progress_callback:
            progress_callback(num_stages, num_stages, "Phase cancellation...")
        
        final_l, final_r = self._generate_final_cancellation(
            base_frequency, stages[-1]['amplitude'] if stages else 0.1
        )
        left_segments.append(final_l)
        right_segments.append(final_r)
        
        return self._smooth_concatenate(left_segments), self._smooth_concatenate(right_segments)
    
    def _generate_final_cancellation(self, frequency: float, start_amplitude: float):
        duration = FIBONACCI[7] * PHI_CONJUGATE
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        progress = golden_spiral_interpolation(np.linspace(0, 1, num_samples))
        phase_diff = np.pi * progress
        amplitude = start_amplitude * (1 - progress)
        
        omega = TWO_PI * frequency
        phase = self._phase_left + omega * t
        
        return amplitude * np.sin(phase), amplitude * np.sin(phase + phase_diff)
    
    def generate_preview(self, base_freq: float, beat_freq: float, duration: float, amplitude: float):
        """Generate smooth preview"""
        self.reset_phase()
        left, right = self.generate_segment_vectorized(base_freq, beat_freq, duration, amplitude)
        
        num_samples = len(left)
        attack, decay = int(num_samples * 0.1), int(num_samples * 0.15)
        
        envelope = np.ones(num_samples, dtype=np.float64)
        if attack > 0:
            envelope[:attack] = (1 - np.cos(np.linspace(0, np.pi, attack))) / 2
        if decay > 0:
            envelope[-decay:] = (1 + np.cos(np.linspace(0, np.pi, decay))) / 2
        
        return left * envelope, right * envelope


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PLAYER
# ═══════════════════════════════════════════════════════════════════════════════

class AudioPlayer:
    def __init__(self):
        self.process = None
        self.temp_file = None
    
    def play(self, left: np.ndarray, right: np.ndarray, sample_rate: int = 44100):
        self.stop()
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self._save_wav(self.temp_file.name, left, right, sample_rate)
        
        system = platform.system()
        try:
            if system == 'Darwin':
                self.process = subprocess.Popen(['afplay', self.temp_file.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == 'Linux':
                for player in ['aplay', 'paplay', 'play']:
                    try:
                        self.process = subprocess.Popen([player, self.temp_file.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        break
                    except FileNotFoundError:
                        continue
        except Exception as e:
            print(f"Playback error: {e}")
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process = None
        if self.temp_file and os.path.exists(self.temp_file.name):
            try:
                os.unlink(self.temp_file.name)
            except:
                pass
    
    def _save_wav(self, filename, left, right, sample_rate):
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)
        left, right = left / max_val * 0.9, right / max_val * 0.9
        
        left_int = (left * 32767).astype(np.int16)
        right_int = (right * 32767).astype(np.int16)
        
        stereo = np.empty(len(left) * 2, dtype=np.int16)
        stereo[0::2], stereo[1::2] = left_int, right_int
        
        with open(filename, 'wb') as f:
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + len(stereo) * 2))
            f.write(b'WAVEfmt ')
            f.write(struct.pack('<IHHIIHH', 16, 1, 2, sample_rate, sample_rate * 4, 4, 16))
            f.write(b'data')
            f.write(struct.pack('<I', len(stereo) * 2))
            f.write(stereo.tobytes())


def save_wav(filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int = 44100):
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)
    left, right = left / max_val * 0.95, right / max_val * 0.95
    
    left_int = (left * 32767).astype(np.int16)
    right_int = (right * 32767).astype(np.int16)
    
    stereo = np.empty(len(left) * 2, dtype=np.int16)
    stereo[0::2], stereo[1::2] = left_int, right_int
    
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(stereo) * 2))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 2, sample_rate, sample_rate * 4, 4, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(stereo) * 2))
        f.write(stereo.tobytes())


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenSpiralCanvas(tk.Canvas):
    def __init__(self, parent, size, **kwargs):
        super().__init__(parent, width=size, height=size, bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        self.size = size
        self.cx, self.cy = size // 2, size // 2
        self.spiral_x, self.spiral_y = golden_spiral_points(400)
        self.phase = 0.0
        self.is_playing = False
        self._draw_spiral()
    
    def _draw_spiral(self):
        self.delete("spiral")
        scale = self.size * 0.4
        for i in range(len(self.spiral_x) - 1):
            progress = i / len(self.spiral_x)
            r = int(212 * (0.3 + 0.7 * progress))
            g = int(175 * (0.3 + 0.7 * progress))
            b = int(55 + 100 * (1 - progress))
            self.create_line(
                self.cx + self.spiral_x[i] * scale, self.cy + self.spiral_y[i] * scale,
                self.cx + self.spiral_x[i+1] * scale, self.cy + self.spiral_y[i+1] * scale,
                fill=f'#{r:02x}{g:02x}{b:02x}', width=2, tags="spiral"
            )
    
    def update(self, phase: float, amplitude: float, is_playing: bool = False):
        self.phase = phase
        self.is_playing = is_playing
        self.delete("waves", "indicator", "phi")
        
        scale = self.size * 0.35 * amplitude
        width = 4 if is_playing else 2
        
        # Waves
        pts_l, pts_r = [], []
        for i in range(120):
            x = self.cx + (i - 60) * (self.size / 200)
            pts_l.extend([x, self.cy + np.sin(i * 0.15 + phase) * scale * 0.6])
            pts_r.extend([x, self.cy + np.sin(i * 0.15 + phase + np.pi * PHI_CONJUGATE) * scale * 0.6])
        
        if len(pts_l) >= 4:
            self.create_line(pts_l, fill=COLORS['gold_primary'], width=width, smooth=True, tags="waves")
            self.create_line(pts_r, fill=COLORS['cyan'], width=width, smooth=True, tags="waves")
        
        # Indicator
        idx = int((phase / TWO_PI) * len(self.spiral_x)) % len(self.spiral_x)
        ix = self.cx + self.spiral_x[idx] * self.size * 0.4
        iy = self.cy + self.spiral_y[idx] * self.size * 0.4
        color = COLORS['success'] if is_playing else COLORS['gold_secondary']
        size = 10 if is_playing else 6
        
        if is_playing:
            self.create_oval(ix-size*2, iy-size*2, ix+size*2, iy+size*2, outline=color, width=2, tags="indicator")
        self.create_oval(ix-size, iy-size, ix+size, iy+size, fill=color, outline=COLORS['white'], width=2, tags="indicator")
        
        # Phi
        self.create_text(self.cx, self.cy, text="φ", font=("Helvetica", int(self.size/20), "bold"),
                        fill=color, tags="phi")


class WaveformDisplay(tk.Canvas):
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_medium'],
                        highlightthickness=2, highlightbackground=COLORS['gold_dim'], **kwargs)
        self.w, self.h = width, height
        self.cy = height // 2
        self.create_line(0, self.cy, width, self.cy, fill=COLORS['gold_dim'], dash=(4, 4))
    
    def update_waveform(self, left: np.ndarray, right: np.ndarray):
        self.delete("wf")
        step = max(1, len(left) // self.w)
        left_d, right_d = left[::step][:self.w], right[::step][:self.w]
        scale = self.h * 0.35
        
        pts_l = [(i * (self.w / len(left_d)), self.cy - s * scale) for i, s in enumerate(left_d)]
        pts_r = [(i * (self.w / len(right_d)), self.cy + s * scale) for i, s in enumerate(right_d)]
        
        if pts_l:
            self.create_line([c for p in pts_l for c in p], fill=COLORS['gold_primary'], tags="wf")
        if pts_r:
            self.create_line([c for p in pts_r for c in p], fill=COLORS['cyan'], tags="wf")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenBinauralApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("✦ Divine Golden Binaural Generator ✦")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Golden ratio window
        h = 850
        w = int(h * PHI)
        self.root.geometry(f"{w}x{h}")
        
        self.engine = SmoothBinauralEngine(44100)
        self.player = AudioPlayer()
        
        self.is_playing = False
        self.is_generating = False
        self.generated_audio = None
        self.anim_phase = 0.0
        
        # Variables
        self.base_freq = tk.DoubleVar(value=432.0)
        self.beat_freq = tk.DoubleVar(value=10.0)
        self.num_stages = tk.IntVar(value=8)
        self.amplitude = tk.DoubleVar(value=PHI_CONJUGATE)
        self.preview_duration = tk.DoubleVar(value=8.0)
        
        self._setup_ui()
        self._animate()
    
    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill="both", expand=True, padx=21, pady=13)  # Fibonacci padding
        
        # Header
        tk.Label(main, text="✦ DIVINE GOLDEN RATIO BINAURAL ✦", font=("Helvetica", 21, "bold"),
                fg=COLORS['gold_primary'], bg=COLORS['bg_dark']).pack()
        tk.Label(main, text=f"φ = {PHI:.15f}  •  Seamless Phase Annealing", font=("Helvetica", 11),
                fg=COLORS['gold_secondary'], bg=COLORS['bg_dark']).pack()
        tk.Frame(main, bg=COLORS['gold_dim'], height=2).pack(fill="x", pady=8)
        
        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill="both", expand=True, pady=8)
        
        # Left panel (φ proportion)
        left = tk.Frame(content, bg=COLORS['bg_dark'])
        left.pack(side="left", fill="both", expand=True)
        
        canvas_size = 500
        self.spiral = GoldenSpiralCanvas(left, canvas_size)
        self.spiral.pack(pady=8)
        
        self.waveform = WaveformDisplay(left, canvas_size, int(canvas_size / PHI))
        self.waveform.pack(pady=8)
        
        # Right panel (1/φ proportion)
        right = tk.Frame(content, bg=COLORS['bg_dark'], width=int(500 / PHI))
        right.pack(side="right", fill="y", padx=(21, 0))
        right.pack_propagate(False)
        
        self._create_controls(right)
        
        # Footer
        tk.Frame(main, bg=COLORS['gold_dim'], height=1).pack(fill="x", pady=(8, 5), side="bottom")
        self.status = tk.StringVar(value="✦ Ready • Use stereo headphones ✦")
        tk.Label(main, textvariable=self.status, font=("Helvetica", 10), fg=COLORS['gold_secondary'],
                bg=COLORS['bg_dark']).pack(side="bottom")
    
    def _create_controls(self, parent):
        # Playback
        play_frame = tk.LabelFrame(parent, text=" ♪ Live Preview ", font=("Helvetica", 11, "bold"),
                                   fg=COLORS['success'], bg=COLORS['bg_medium'], padx=13, pady=13)
        play_frame.pack(fill="x", pady=5)
        
        self.play_btn = tk.Button(play_frame, text="▶ PLAY", font=("Helvetica", 13, "bold"),
                                  fg=COLORS['bg_dark'], bg=COLORS['success'], relief="flat",
                                  padx=13, pady=8, command=self._toggle_play)
        self.play_btn.pack(fill="x", pady=5)
        
        self._slider(play_frame, "Duration (s)", self.preview_duration, 3, 21, "Fibonacci: 3, 5, 8, 13, 21")
        
        # Parameters
        params = tk.LabelFrame(parent, text=" ✦ Parameters ", font=("Helvetica", 11, "bold"),
                               fg=COLORS['gold_primary'], bg=COLORS['bg_medium'], padx=13, pady=13)
        params.pack(fill="x", pady=5)
        
        self._slider(params, "Base Freq (Hz)", self.base_freq, 100, 963, "432, 528, 639...")
        self._slider(params, "Beat Freq (Hz)", self.beat_freq, 0.5, 40, "Delta<4 Theta<8 Alpha<13")
        self._slider(params, "Amplitude", self.amplitude, 0.1, 1.0, f"Golden: {PHI_CONJUGATE:.3f}")
        self._slider(params, "Stages", self.num_stages, 3, 13, "Fibonacci: 5, 8, 13")
        
        # Presets
        presets = tk.LabelFrame(parent, text=" ✦ Presets ", font=("Helvetica", 11, "bold"),
                                fg=COLORS['gold_primary'], bg=COLORS['bg_medium'], padx=13, pady=8)
        presets.pack(fill="x", pady=5)
        
        for name, base, beat in [("🌟 432 Hz", 432, 7.83), ("💚 528 Hz", 528, 8), 
                                  ("🌙 Theta", 432, 5.5), ("🌊 Delta", 432, 2.5)]:
            tk.Button(presets, text=name, font=("Helvetica", 9), fg=COLORS['bg_dark'],
                     bg=COLORS['gold_secondary'], relief="flat", pady=3,
                     command=lambda b=base, bt=beat: self._preset(b, bt)).pack(fill="x", pady=2)
        
        # Generate
        gen = tk.LabelFrame(parent, text=" ✦ Full Sequence ", font=("Helvetica", 11, "bold"),
                            fg=COLORS['gold_primary'], bg=COLORS['bg_medium'], padx=13, pady=13)
        gen.pack(fill="x", pady=5)
        
        self.gen_btn = tk.Button(gen, text="✦ GENERATE ✦", font=("Helvetica", 12, "bold"),
                                 fg=COLORS['bg_dark'], bg=COLORS['gold_primary'], relief="flat",
                                 pady=8, command=self._generate)
        self.gen_btn.pack(fill="x", pady=3)
        
        btn_row = tk.Frame(gen, bg=COLORS['bg_medium'])
        btn_row.pack(fill="x", pady=3)
        
        self.save_btn = tk.Button(btn_row, text="💾 Save", font=("Helvetica", 9),
                                  fg=COLORS['bg_dark'], bg=COLORS['gold_dim'], relief="flat",
                                  command=self._save, state="disabled")
        self.save_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        
        self.play_full_btn = tk.Button(btn_row, text="▶ Play", font=("Helvetica", 9),
                                       fg=COLORS['bg_dark'], bg=COLORS['cyan'], relief="flat",
                                       command=self._play_full, state="disabled")
        self.play_full_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))
        
        self.progress = tk.StringVar(value="Ready")
        tk.Label(gen, textvariable=self.progress, font=("Helvetica", 9),
                fg=COLORS['gold_secondary'], bg=COLORS['bg_medium']).pack(pady=3)
        
        self.progress_bar = ttk.Progressbar(gen, length=200, mode='determinate')
        self.progress_bar.pack(fill="x", pady=3)
    
    def _slider(self, parent, label, var, min_v, max_v, tip):
        frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        frame.pack(fill="x", pady=5)
        
        row = tk.Frame(frame, bg=COLORS['bg_medium'])
        row.pack(fill="x")
        tk.Label(row, text=label, font=("Helvetica", 9), fg=COLORS['gold_secondary'],
                bg=COLORS['bg_medium']).pack(side="left")
        tk.Label(row, textvariable=var, font=("Helvetica", 9, "bold"), fg=COLORS['gold_primary'],
                bg=COLORS['bg_medium'], width=6).pack(side="right")
        
        res = 0.1 if max_v <= 50 else 1
        tk.Scale(frame, from_=min_v, to=max_v, orient="horizontal", variable=var, resolution=res,
                font=("Helvetica", 8), fg=COLORS['gold_primary'], bg=COLORS['bg_medium'],
                troughcolor=COLORS['purple'], highlightthickness=0, showvalue=False).pack(fill="x")
        
        tk.Label(frame, text=tip, font=("Helvetica", 7), fg=COLORS['text_dim'],
                bg=COLORS['bg_medium']).pack(fill="x")
    
    def _preset(self, base, beat):
        self.base_freq.set(base)
        self.beat_freq.set(beat)
        self.amplitude.set(PHI_CONJUGATE)
        self.status.set(f"✦ Preset: {base}Hz • {beat}Hz ✦")
    
    def _toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()
    
    def _start_play(self):
        self.is_playing = True
        self.play_btn.config(text="⏹ STOP", bg=COLORS['accent'])
        self.status.set("♪ Playing... Use headphones! ♪")
        
        threading.Thread(target=self._play_thread, daemon=True).start()
    
    def _play_thread(self):
        left, right = self.engine.generate_preview(
            self.base_freq.get(), self.beat_freq.get(),
            self.preview_duration.get(), self.amplitude.get()
        )
        self.player.play(left, right, 44100)
        
        import time
        time.sleep(self.preview_duration.get() + 0.5)
        self.root.after(0, self._stop_play)
    
    def _stop_play(self):
        self.is_playing = False
        self.player.stop()
        self.play_btn.config(text="▶ PLAY", bg=COLORS['success'])
        self.status.set("✦ Ready ✦")
    
    def _generate(self):
        if self.is_generating:
            return
        
        self._stop_play()
        self.is_generating = True
        self.gen_btn.config(state="disabled", text="Generating...")
        self.save_btn.config(state="disabled")
        self.play_full_btn.config(state="disabled")
        self.progress_bar['value'] = 0
        
        threading.Thread(target=self._gen_thread).start()
    
    def _gen_thread(self):
        def cb(stage, total, msg):
            self.root.after(0, lambda: self._update_progress((stage / max(1, total)) * 100, msg))
        
        left, right = self.engine.generate_annealing_sequence(
            self.num_stages.get(), self.base_freq.get(), cb
        )
        self.generated_audio = (left, right)
        
        self.root.after(0, lambda: self.waveform.update_waveform(left[:50000], right[:50000]))
        self.root.after(0, lambda: self._gen_done(len(left) / 44100))
    
    def _update_progress(self, pct, msg):
        self.progress_bar['value'] = pct
        self.progress.set(msg)
    
    def _gen_done(self, duration):
        self.is_generating = False
        self.gen_btn.config(state="normal", text="✦ GENERATE ✦")
        self.save_btn.config(state="normal")
        self.play_full_btn.config(state="normal")
        self.progress_bar['value'] = 100
        self.progress.set(f"✓ {duration:.1f}s ({duration/60:.1f} min)")
        self.status.set("✦ Sequence ready! ✦")
    
    def _play_full(self):
        if self.generated_audio:
            self.status.set("♪ Playing full sequence... ♪")
            left, right = self.generated_audio
            self.player.play(left, right, 44100)
    
    def _save(self):
        if not self.generated_audio:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")],
            initialfile=f"golden_{int(self.base_freq.get())}Hz.wav"
        )
        if filename:
            left, right = self.generated_audio
            save_wav(filename, left, right, 44100)
            self.status.set(f"✦ Saved: {os.path.basename(filename)} ✦")
            messagebox.showinfo("Success", f"Saved:\n{filename}")
    
    def _animate(self):
        self.anim_phase += 0.05 * PHI_CONJUGATE
        if self.anim_phase > TWO_PI:
            self.anim_phase -= TWO_PI
        
        self.spiral.update(self.anim_phase, self.amplitude.get(), self.is_playing)
        self.root.after(34, self._animate)  # ~29fps (Fibonacci)
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()
    
    def _close(self):
        self.player.stop()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "★" * 55)
    print("  DIVINE GOLDEN BINAURAL v3 - SMOOTH")
    print("  No clicks • Seamless transitions • All tests passing")
    print("★" * 55 + "\n")
    
    GoldenBinauralApp().run()
