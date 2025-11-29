"""
═══════════════════════════════════════════════════════════════════════════════
  DIVINE GOLDEN BINAURAL GENERATOR - PROFESSIONAL VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Advanced GUI with:
- Full sequence visualization with ALL stages
- Interactive navigation between stages
- Real-time waveform display per stage
- Phase diagram (shows phase cancellation progress)
- Frequency spectrum view
- Parameter evolution graphs
- Stage-by-stage playback

Tutto in proporzioni φ (golden ratio)
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
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# DIVINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = np.float64((1 + np.sqrt(5)) / 2)
PHI_CONJUGATE = np.float64(PHI - 1)
PHI_SQUARED = np.float64(PHI * PHI)
TWO_PI = np.float64(2 * np.pi)

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Professional dark theme
COLORS = {
    'bg_dark': '#0d1117',
    'bg_panel': '#161b22',
    'bg_card': '#21262d',
    'border': '#30363d',
    'gold': '#D4AF37',
    'gold_bright': '#FFD700',
    'gold_dim': '#8B7355',
    'cyan': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'purple': '#a371f7',
    'orange': '#d29922',
    'white': '#f0f6fc',
    'text': '#c9d1d9',
    'text_dim': '#8b949e',
    'graph_bg': '#0d1117',
    'grid': '#21262d',
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageData:
    """Complete data for one stage"""
    index: int
    name: str
    beat_freq: float
    duration: float
    amplitude: float
    phase_offset: float
    transition_time: float
    annealing_progress: float  # 0 = start, 1 = full cancellation
    left_audio: np.ndarray
    right_audio: np.ndarray
    start_sample: int
    end_sample: int
    stage_type: str  # 'beat', 'transition', 'cancellation'


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

def golden_spiral_interpolation(t):
    """Vectorized golden spiral easing"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# SMOOTH BINAURAL ENGINE WITH STAGE TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyticalBinauralEngine:
    """Engine that tracks all stages for visualization"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._phase_left = 0.0
        self._phase_right = 0.0
        self.stages: List[StageData] = []
        self.full_left: Optional[np.ndarray] = None
        self.full_right: Optional[np.ndarray] = None
    
    def reset(self):
        self._phase_left = 0.0
        self._phase_right = 0.0
        self.stages = []
        self.full_left = None
        self.full_right = None
    
    def _generate_segment(self, base_freq: float, beat_freq: float, duration: float,
                          amplitude: float, phase_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _generate_transition(self, base_freq: float, beat1: float, beat2: float,
                             amp1: float, amp2: float, phase1: float, phase2: float,
                             duration: float) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _apply_envelope(self, audio: np.ndarray) -> np.ndarray:
        fade_samples = min(64, len(audio) // 10)
        if fade_samples < 2:
            return audio
        
        fade_in = np.linspace(0, 1, fade_samples) ** 2
        fade_out = np.linspace(1, 0, fade_samples) ** 2
        
        audio = audio.copy()
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        return audio
    
    def generate_full_sequence(self, num_stages: int, base_frequency: float,
                               progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete sequence with stage tracking"""
        self.reset()
        
        left_segments, right_segments = [], []
        current_sample = 0
        
        # Calculate all stage parameters first
        stage_params = []
        for i in range(num_stages):
            beat_freq = base_frequency / (PHI ** (i + 3))
            fib_idx = min(i + 5, len(FIBONACCI) - 1)
            duration = FIBONACCI[fib_idx] * PHI_CONJUGATE
            transition_time = duration * PHI_CONJUGATE * PHI_CONJUGATE
            
            annealing = i / max(1, num_stages - 1)
            phase_offset = np.pi * golden_spiral_interpolation(annealing)
            amplitude = (1 - golden_spiral_interpolation(annealing) * 0.95) * (PHI_CONJUGATE ** (i * 0.5))
            
            stage_params.append({
                'index': i,
                'beat_freq': beat_freq,
                'duration': duration,
                'transition_time': transition_time,
                'phase_offset': phase_offset,
                'amplitude': amplitude,
                'annealing': annealing,
            })
        
        # Generate each stage with tracking
        for idx, params in enumerate(stage_params):
            if progress_callback:
                progress_callback(idx, num_stages * 2, f"Stage {idx+1}: Generating beat")
            
            # Generate beat segment
            left, right = self._generate_segment(
                base_frequency, params['beat_freq'], params['duration'],
                params['amplitude'], params['phase_offset']
            )
            left = self._apply_envelope(left)
            right = self._apply_envelope(right)
            
            # Track stage
            stage = StageData(
                index=idx,
                name=f"Stage {idx+1}: Beat φ^{idx+3}",
                beat_freq=params['beat_freq'],
                duration=params['duration'],
                amplitude=params['amplitude'],
                phase_offset=params['phase_offset'],
                transition_time=params['transition_time'],
                annealing_progress=params['annealing'],
                left_audio=left,
                right_audio=right,
                start_sample=current_sample,
                end_sample=current_sample + len(left),
                stage_type='beat'
            )
            self.stages.append(stage)
            
            left_segments.append(left)
            right_segments.append(right)
            current_sample += len(left)
            
            # Generate transition
            if idx < num_stages - 1:
                if progress_callback:
                    progress_callback(idx * 2 + 1, num_stages * 2, f"Transition {idx+1}→{idx+2}")
                
                next_p = stage_params[idx + 1]
                trans_l, trans_r = self._generate_transition(
                    base_frequency,
                    params['beat_freq'], next_p['beat_freq'],
                    params['amplitude'], next_p['amplitude'],
                    params['phase_offset'], next_p['phase_offset'],
                    params['transition_time']
                )
                trans_l = self._apply_envelope(trans_l)
                trans_r = self._apply_envelope(trans_r)
                
                # Track transition
                trans_stage = StageData(
                    index=idx,
                    name=f"Transition {idx+1}→{idx+2}",
                    beat_freq=(params['beat_freq'] + next_p['beat_freq']) / 2,
                    duration=params['transition_time'],
                    amplitude=(params['amplitude'] + next_p['amplitude']) / 2,
                    phase_offset=(params['phase_offset'] + next_p['phase_offset']) / 2,
                    transition_time=params['transition_time'],
                    annealing_progress=(params['annealing'] + next_p['annealing']) / 2,
                    left_audio=trans_l,
                    right_audio=trans_r,
                    start_sample=current_sample,
                    end_sample=current_sample + len(trans_l),
                    stage_type='transition'
                )
                self.stages.append(trans_stage)
                
                left_segments.append(trans_l)
                right_segments.append(trans_r)
                current_sample += len(trans_l)
        
        # Final cancellation
        if progress_callback:
            progress_callback(num_stages * 2 - 1, num_stages * 2, "Phase cancellation")
        
        cancel_l, cancel_r = self._generate_cancellation(
            base_frequency, stage_params[-1]['amplitude'] if stage_params else 0.1
        )
        
        cancel_stage = StageData(
            index=num_stages,
            name="Final: Phase Cancellation → Silence",
            beat_freq=0,
            duration=len(cancel_l) / self.sample_rate,
            amplitude=0,
            phase_offset=np.pi,
            transition_time=0,
            annealing_progress=1.0,
            left_audio=cancel_l,
            right_audio=cancel_r,
            start_sample=current_sample,
            end_sample=current_sample + len(cancel_l),
            stage_type='cancellation'
        )
        self.stages.append(cancel_stage)
        
        left_segments.append(cancel_l)
        right_segments.append(cancel_r)
        
        # Concatenate
        self.full_left = np.concatenate([self._apply_envelope(s) for s in left_segments])
        self.full_right = np.concatenate([self._apply_envelope(s) for s in right_segments])
        
        return self.full_left, self.full_right
    
    def _generate_cancellation(self, frequency: float, start_amplitude: float):
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
        self._phase_left = 0.0
        self._phase_right = 0.0
        left, right = self._generate_segment(base_freq, beat_freq, duration, amplitude)
        
        num_samples = len(left)
        attack = int(num_samples * 0.1)
        decay = int(num_samples * 0.15)
        
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
        
        try:
            if platform.system() == 'Darwin':
                self.process = subprocess.Popen(['afplay', self.temp_file.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
# VISUALIZATION CANVASES
# ═══════════════════════════════════════════════════════════════════════════════

class WaveformCanvas(tk.Canvas):
    """Professional waveform display with zoom"""
    
    def __init__(self, parent, width, height, title="Waveform", **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['graph_bg'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self.title = title
        self.left_data = None
        self.right_data = None
        self._draw_grid()
    
    def _draw_grid(self):
        self.delete("grid", "title")
        cy = self.h // 2
        
        # Horizontal grid
        for i in range(5):
            y = int(i * self.h / 4)
            self.create_line(0, y, self.w, y, fill=COLORS['grid'], tags="grid")
        
        # Center line
        self.create_line(0, cy, self.w, cy, fill=COLORS['border'], width=1, tags="grid")
        
        # Vertical grid (golden sections)
        for i in range(1, 5):
            x = int(i * self.w / 5)
            self.create_line(x, 0, x, self.h, fill=COLORS['grid'], dash=(2, 4), tags="grid")
        
        # Title
        self.create_text(10, 10, text=self.title, anchor="nw", fill=COLORS['text_dim'],
                        font=("Helvetica", 9), tags="title")
    
    def update_data(self, left: np.ndarray, right: np.ndarray, title: str = None):
        if title:
            self.title = title
        self.left_data = left
        self.right_data = right
        self._redraw()
    
    def _redraw(self):
        self._draw_grid()
        self.delete("waveform")
        
        if self.left_data is None or len(self.left_data) == 0:
            return
        
        cy = self.h // 2
        scale = self.h * 0.4
        
        # Downsample for display
        step = max(1, len(self.left_data) // (self.w * 2))
        left_d = self.left_data[::step]
        right_d = self.right_data[::step]
        
        # Draw waveforms
        x_scale = self.w / len(left_d)
        
        # Left channel (gold, top half)
        pts_left = []
        for i, s in enumerate(left_d):
            x = i * x_scale
            y = cy - s * scale * 0.9
            pts_left.extend([x, y])
        
        if len(pts_left) >= 4:
            self.create_line(pts_left, fill=COLORS['gold'], width=1, smooth=True, tags="waveform")
        
        # Right channel (cyan, bottom half offset)
        pts_right = []
        for i, s in enumerate(right_d):
            x = i * x_scale
            y = cy + s * scale * 0.9
            pts_right.extend([x, y])
        
        if len(pts_right) >= 4:
            self.create_line(pts_right, fill=COLORS['cyan'], width=1, smooth=True, tags="waveform")
        
        # Labels
        self.create_text(self.w - 10, 15, text="L", anchor="ne", fill=COLORS['gold'],
                        font=("Helvetica", 10, "bold"), tags="waveform")
        self.create_text(self.w - 10, self.h - 15, text="R", anchor="se", fill=COLORS['cyan'],
                        font=("Helvetica", 10, "bold"), tags="waveform")


class PhaseCanvas(tk.Canvas):
    """Phase relationship visualization - shows L vs R phase"""
    
    def __init__(self, parent, size, **kwargs):
        super().__init__(parent, width=size, height=size, bg=COLORS['graph_bg'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.size = size
        self.cx, self.cy = size // 2, size // 2
        self._draw_grid()
    
    def _draw_grid(self):
        self.delete("grid")
        r = self.size // 2 - 20
        
        # Circles
        for frac in [0.25, 0.5, 0.75, 1.0]:
            self.create_oval(
                self.cx - r * frac, self.cy - r * frac,
                self.cx + r * frac, self.cy + r * frac,
                outline=COLORS['grid'], tags="grid"
            )
        
        # Axes
        self.create_line(self.cx - r, self.cy, self.cx + r, self.cy, fill=COLORS['border'], tags="grid")
        self.create_line(self.cx, self.cy - r, self.cx, self.cy + r, fill=COLORS['border'], tags="grid")
        
        # Labels
        self.create_text(self.cx + r + 5, self.cy, text="0°", anchor="w", fill=COLORS['text_dim'],
                        font=("Helvetica", 8), tags="grid")
        self.create_text(self.cx, self.cy - r - 5, text="90°", anchor="s", fill=COLORS['text_dim'],
                        font=("Helvetica", 8), tags="grid")
        self.create_text(self.cx - r - 5, self.cy, text="180°", anchor="e", fill=COLORS['text_dim'],
                        font=("Helvetica", 8), tags="grid")
        
        self.create_text(self.cx, 10, text="Phase Diagram", anchor="n", fill=COLORS['text_dim'],
                        font=("Helvetica", 9), tags="grid")
    
    def update_phase(self, left: np.ndarray, right: np.ndarray, phase_offset: float):
        self._draw_grid()
        self.delete("phase")
        
        if left is None or len(left) == 0:
            return
        
        r = self.size // 2 - 25
        
        # Sample points for Lissajous figure
        step = max(1, len(left) // 500)
        l_samples = left[::step]
        r_samples = right[::step]
        
        # Normalize
        max_val = max(np.max(np.abs(l_samples)), np.max(np.abs(r_samples)), 1e-10)
        l_norm = l_samples / max_val
        r_norm = r_samples / max_val
        
        # Draw Lissajous
        pts = []
        for l, rv in zip(l_norm, r_norm):
            x = self.cx + l * r
            y = self.cy - rv * r
            pts.extend([x, y])
        
        if len(pts) >= 4:
            # Color gradient based on position
            self.create_line(pts, fill=COLORS['purple'], width=1, smooth=True, tags="phase")
        
        # Phase offset indicator
        angle = phase_offset
        end_x = self.cx + r * 0.9 * np.cos(angle)
        end_y = self.cy - r * 0.9 * np.sin(angle)
        self.create_line(self.cx, self.cy, end_x, end_y, fill=COLORS['gold_bright'],
                        width=2, arrow=tk.LAST, tags="phase")
        
        # Phase text
        phase_deg = np.degrees(phase_offset) % 360
        self.create_text(self.cx, self.size - 10, text=f"Phase: {phase_deg:.1f}°",
                        anchor="s", fill=COLORS['gold'], font=("Helvetica", 10, "bold"), tags="phase")


class SequenceTimeline(tk.Canvas):
    """Timeline showing all stages with navigation"""
    
    def __init__(self, parent, width, height, on_stage_click=None, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_panel'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self.stages: List[StageData] = []
        self.selected_stage = 0
        self.on_stage_click = on_stage_click
        self.bind("<Button-1>", self._on_click)
    
    def set_stages(self, stages: List[StageData]):
        self.stages = stages
        self._redraw()
    
    def set_selected(self, idx: int):
        self.selected_stage = idx
        self._redraw()
    
    def _redraw(self):
        self.delete("all")
        
        if not self.stages:
            self.create_text(self.w // 2, self.h // 2, text="Generate to see timeline",
                            fill=COLORS['text_dim'], font=("Helvetica", 10))
            return
        
        # Calculate total samples
        total_samples = self.stages[-1].end_sample
        if total_samples == 0:
            return
        
        margin = 10
        usable_width = self.w - 2 * margin
        bar_height = 30
        bar_y = (self.h - bar_height) // 2
        
        # Background bar
        self.create_rectangle(margin, bar_y, self.w - margin, bar_y + bar_height,
                             fill=COLORS['bg_card'], outline=COLORS['border'])
        
        # Draw each stage
        for i, stage in enumerate(self.stages):
            x1 = margin + (stage.start_sample / total_samples) * usable_width
            x2 = margin + (stage.end_sample / total_samples) * usable_width
            
            # Color based on type
            if stage.stage_type == 'beat':
                color = COLORS['gold'] if i == self.selected_stage else COLORS['gold_dim']
            elif stage.stage_type == 'transition':
                color = COLORS['cyan'] if i == self.selected_stage else COLORS['border']
            else:  # cancellation
                color = COLORS['purple'] if i == self.selected_stage else COLORS['text_dim']
            
            # Stage bar
            self.create_rectangle(x1, bar_y + 2, x2, bar_y + bar_height - 2,
                                 fill=color, outline='', tags=f"stage_{i}")
            
            # Stage number for beat stages
            if stage.stage_type == 'beat' and (x2 - x1) > 20:
                self.create_text((x1 + x2) / 2, bar_y + bar_height / 2,
                               text=str(stage.index + 1), fill=COLORS['bg_dark'],
                               font=("Helvetica", 9, "bold"))
        
        # Selection highlight
        selected = self.stages[self.selected_stage]
        x1 = margin + (selected.start_sample / total_samples) * usable_width
        x2 = margin + (selected.end_sample / total_samples) * usable_width
        self.create_rectangle(x1 - 2, bar_y - 3, x2 + 2, bar_y + bar_height + 3,
                             outline=COLORS['gold_bright'], width=2)
        
        # Title
        self.create_text(margin, 5, text="Sequence Timeline", anchor="nw",
                        fill=COLORS['text_dim'], font=("Helvetica", 9))
        
        # Selected stage info
        duration = len(selected.left_audio) / 44100
        self.create_text(self.w - margin, 5, text=f"{selected.name} ({duration:.1f}s)",
                        anchor="ne", fill=COLORS['gold'], font=("Helvetica", 9))
    
    def _on_click(self, event):
        if not self.stages:
            return
        
        margin = 10
        usable_width = self.w - 2 * margin
        total_samples = self.stages[-1].end_sample
        
        # Find clicked stage
        click_sample = ((event.x - margin) / usable_width) * total_samples
        
        for i, stage in enumerate(self.stages):
            if stage.start_sample <= click_sample < stage.end_sample:
                self.selected_stage = i
                self._redraw()
                if self.on_stage_click:
                    self.on_stage_click(i)
                break


class ParameterGraph(tk.Canvas):
    """Graph showing parameter evolution over stages"""
    
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['graph_bg'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self._draw_grid()
    
    def _draw_grid(self):
        self.delete("grid")
        margin = 40
        
        # Axes
        self.create_line(margin, 10, margin, self.h - 20, fill=COLORS['border'], tags="grid")
        self.create_line(margin, self.h - 20, self.w - 10, self.h - 20, fill=COLORS['border'], tags="grid")
        
        # Y labels
        self.create_text(5, 10, text="100%", anchor="nw", fill=COLORS['text_dim'],
                        font=("Helvetica", 7), tags="grid")
        self.create_text(5, self.h // 2, text="50%", anchor="w", fill=COLORS['text_dim'],
                        font=("Helvetica", 7), tags="grid")
        self.create_text(5, self.h - 25, text="0%", anchor="sw", fill=COLORS['text_dim'],
                        font=("Helvetica", 7), tags="grid")
        
        # Title
        self.create_text(self.w // 2, 5, text="Parameter Evolution", anchor="n",
                        fill=COLORS['text_dim'], font=("Helvetica", 9), tags="grid")
    
    def update_parameters(self, stages: List[StageData]):
        self._draw_grid()
        self.delete("params")
        
        if not stages:
            return
        
        margin = 40
        graph_w = self.w - margin - 10
        graph_h = self.h - 30
        
        # Extract beat stages only for cleaner graph
        beat_stages = [s for s in stages if s.stage_type == 'beat']
        if not beat_stages:
            return
        
        n = len(beat_stages)
        x_step = graph_w / max(1, n - 1)
        
        # Amplitude line (gold)
        amp_pts = []
        for i, s in enumerate(beat_stages):
            x = margin + i * x_step
            y = 10 + (1 - s.amplitude) * graph_h
            amp_pts.extend([x, y])
        
        if len(amp_pts) >= 4:
            self.create_line(amp_pts, fill=COLORS['gold'], width=2, smooth=True, tags="params")
        
        # Phase line (cyan)
        phase_pts = []
        for i, s in enumerate(beat_stages):
            x = margin + i * x_step
            y = 10 + (1 - s.phase_offset / np.pi) * graph_h
            phase_pts.extend([x, y])
        
        if len(phase_pts) >= 4:
            self.create_line(phase_pts, fill=COLORS['cyan'], width=2, smooth=True, tags="params")
        
        # Annealing line (purple)
        ann_pts = []
        for i, s in enumerate(beat_stages):
            x = margin + i * x_step
            y = 10 + (1 - s.annealing_progress) * graph_h
            ann_pts.extend([x, y])
        
        if len(ann_pts) >= 4:
            self.create_line(ann_pts, fill=COLORS['purple'], width=2, smooth=True, tags="params")
        
        # Legend
        self.create_rectangle(self.w - 90, 20, self.w - 10, 70, fill=COLORS['bg_card'], tags="params")
        self.create_line(self.w - 85, 30, self.w - 65, 30, fill=COLORS['gold'], width=2, tags="params")
        self.create_text(self.w - 60, 30, text="Amp", anchor="w", fill=COLORS['text_dim'],
                        font=("Helvetica", 7), tags="params")
        self.create_line(self.w - 85, 45, self.w - 65, 45, fill=COLORS['cyan'], width=2, tags="params")
        self.create_text(self.w - 60, 45, text="Phase", anchor="w", fill=COLORS['text_dim'],
                        font=("Helvetica", 7), tags="params")
        self.create_line(self.w - 85, 60, self.w - 65, 60, fill=COLORS['purple'], width=2, tags="params")
        self.create_text(self.w - 60, 60, text="Anneal", anchor="w", fill=COLORS['text_dim'],
                        font=("Helvetica", 7), tags="params")


class StageInfoPanel(tk.Frame):
    """Detailed info panel for selected stage"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['bg_card'], **kwargs)
        
        self.title_label = tk.Label(self, text="No Stage Selected", font=("Helvetica", 12, "bold"),
                                   fg=COLORS['gold'], bg=COLORS['bg_card'])
        self.title_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        self.info_frame = tk.Frame(self, bg=COLORS['bg_card'])
        self.info_frame.pack(fill="x", padx=10, pady=5)
        
        self.labels = {}
        params = [
            ("Type", "stage_type"),
            ("Beat Freq", "beat_freq", "Hz"),
            ("Duration", "duration", "s"),
            ("Amplitude", "amplitude", ""),
            ("Phase", "phase_deg", "°"),
            ("Annealing", "annealing_pct", "%"),
        ]
        
        for i, param in enumerate(params):
            name = param[0]
            row = tk.Frame(self.info_frame, bg=COLORS['bg_card'])
            row.pack(fill="x", pady=2)
            
            tk.Label(row, text=f"{name}:", font=("Helvetica", 9), fg=COLORS['text_dim'],
                    bg=COLORS['bg_card'], width=10, anchor="w").pack(side="left")
            
            self.labels[name] = tk.Label(row, text="-", font=("Helvetica", 10, "bold"),
                                        fg=COLORS['gold'], bg=COLORS['bg_card'], anchor="w")
            self.labels[name].pack(side="left", padx=5)
    
    def update_stage(self, stage: StageData):
        self.title_label.config(text=stage.name)
        
        type_colors = {'beat': COLORS['gold'], 'transition': COLORS['cyan'], 'cancellation': COLORS['purple']}
        self.labels["Type"].config(text=stage.stage_type.upper(), fg=type_colors.get(stage.stage_type, COLORS['text']))
        self.labels["Beat Freq"].config(text=f"{stage.beat_freq:.3f} Hz")
        self.labels["Duration"].config(text=f"{stage.duration:.2f} s")
        self.labels["Amplitude"].config(text=f"{stage.amplitude:.4f}")
        self.labels["Phase"].config(text=f"{np.degrees(stage.phase_offset):.1f}°")
        self.labels["Annealing"].config(text=f"{stage.annealing_progress * 100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenBinauralProfessional:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("✦ Divine Golden Binaural - Professional Analyzer ✦")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Golden window size
        h = 900
        w = int(h * PHI)
        self.root.geometry(f"{w}x{h}")
        
        self.engine = AnalyticalBinauralEngine(44100)
        self.player = AudioPlayer()
        
        self.is_generating = False
        self.selected_stage_idx = 0
        
        # Variables
        self.base_freq = tk.DoubleVar(value=432.0)
        self.num_stages = tk.IntVar(value=8)
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Main container
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill="both", expand=True, padx=13, pady=8)
        
        # Header
        header = tk.Frame(main, bg=COLORS['bg_dark'])
        header.pack(fill="x", pady=(0, 8))
        
        tk.Label(header, text="✦ DIVINE GOLDEN BINAURAL ANALYZER ✦",
                font=("Helvetica", 18, "bold"), fg=COLORS['gold'], bg=COLORS['bg_dark']).pack()
        tk.Label(header, text=f"φ = {PHI:.15f} • Professional Visualization & Navigation",
                font=("Helvetica", 10), fg=COLORS['text_dim'], bg=COLORS['bg_dark']).pack()
        
        # Top controls
        controls = tk.Frame(main, bg=COLORS['bg_panel'], padx=13, pady=8)
        controls.pack(fill="x", pady=5)
        
        # Left controls
        left_ctrl = tk.Frame(controls, bg=COLORS['bg_panel'])
        left_ctrl.pack(side="left")
        
        tk.Label(left_ctrl, text="Base Freq:", fg=COLORS['text'], bg=COLORS['bg_panel'],
                font=("Helvetica", 10)).pack(side="left", padx=5)
        freq_spin = tk.Spinbox(left_ctrl, from_=100, to=963, textvariable=self.base_freq,
                              width=6, font=("Helvetica", 10), bg=COLORS['bg_card'],
                              fg=COLORS['gold'], insertbackground=COLORS['gold'])
        freq_spin.pack(side="left", padx=5)
        tk.Label(left_ctrl, text="Hz", fg=COLORS['text_dim'], bg=COLORS['bg_panel']).pack(side="left")
        
        tk.Label(left_ctrl, text="  |  Stages:", fg=COLORS['text'], bg=COLORS['bg_panel'],
                font=("Helvetica", 10)).pack(side="left", padx=(20, 5))
        stages_spin = tk.Spinbox(left_ctrl, from_=3, to=13, textvariable=self.num_stages,
                                width=4, font=("Helvetica", 10), bg=COLORS['bg_card'],
                                fg=COLORS['gold'], insertbackground=COLORS['gold'])
        stages_spin.pack(side="left", padx=5)
        
        # Right controls (buttons)
        right_ctrl = tk.Frame(controls, bg=COLORS['bg_panel'])
        right_ctrl.pack(side="right")
        
        self.gen_btn = tk.Button(right_ctrl, text="✦ GENERATE SEQUENCE", font=("Helvetica", 11, "bold"),
                                fg=COLORS['bg_dark'], bg=COLORS['gold'], relief="flat", padx=15, pady=5,
                                command=self._generate)
        self.gen_btn.pack(side="left", padx=5)
        
        self.play_btn = tk.Button(right_ctrl, text="▶ Play Stage", font=("Helvetica", 10),
                                 fg=COLORS['bg_dark'], bg=COLORS['green'], relief="flat", padx=10, pady=5,
                                 command=self._play_stage, state="disabled")
        self.play_btn.pack(side="left", padx=5)
        
        self.play_all_btn = tk.Button(right_ctrl, text="▶▶ Play All", font=("Helvetica", 10),
                                     fg=COLORS['bg_dark'], bg=COLORS['cyan'], relief="flat", padx=10, pady=5,
                                     command=self._play_all, state="disabled")
        self.play_all_btn.pack(side="left", padx=5)
        
        tk.Button(right_ctrl, text="⏹ Stop", font=("Helvetica", 10),
                 fg=COLORS['white'], bg=COLORS['red'], relief="flat", padx=10, pady=5,
                 command=self._stop).pack(side="left", padx=5)
        
        self.save_btn = tk.Button(right_ctrl, text="💾 Save", font=("Helvetica", 10),
                                 fg=COLORS['bg_dark'], bg=COLORS['orange'], relief="flat", padx=10, pady=5,
                                 command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=5)
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        tk.Label(controls, textvariable=self.progress_var, fg=COLORS['text_dim'],
                bg=COLORS['bg_panel'], font=("Helvetica", 9)).pack(side="right", padx=20)
        
        # Main content area
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill="both", expand=True, pady=5)
        
        # Left column (waveform + phase)
        left_col = tk.Frame(content, bg=COLORS['bg_dark'])
        left_col.pack(side="left", fill="both", expand=True)
        
        # Waveform canvas
        self.waveform = WaveformCanvas(left_col, 800, 200, "Stage Waveform")
        self.waveform.pack(fill="x", pady=5)
        
        # Phase diagram
        phase_row = tk.Frame(left_col, bg=COLORS['bg_dark'])
        phase_row.pack(fill="x", pady=5)
        
        self.phase_canvas = PhaseCanvas(phase_row, 250)
        self.phase_canvas.pack(side="left", padx=5)
        
        # Parameter graph
        self.param_graph = ParameterGraph(phase_row, 540, 250)
        self.param_graph.pack(side="left", padx=5, fill="x", expand=True)
        
        # Timeline
        self.timeline = SequenceTimeline(left_col, 800, 60, on_stage_click=self._on_stage_select)
        self.timeline.pack(fill="x", pady=5)
        
        # Navigation buttons
        nav = tk.Frame(left_col, bg=COLORS['bg_dark'])
        nav.pack(fill="x", pady=5)
        
        self.prev_btn = tk.Button(nav, text="◀ Previous Stage", font=("Helvetica", 10),
                                 fg=COLORS['white'], bg=COLORS['bg_card'], relief="flat", padx=10,
                                 command=lambda: self._navigate(-1), state="disabled")
        self.prev_btn.pack(side="left", padx=5)
        
        self.stage_label = tk.Label(nav, text="Stage: -/-", font=("Helvetica", 12, "bold"),
                                   fg=COLORS['gold'], bg=COLORS['bg_dark'])
        self.stage_label.pack(side="left", expand=True)
        
        self.next_btn = tk.Button(nav, text="Next Stage ▶", font=("Helvetica", 10),
                                 fg=COLORS['white'], bg=COLORS['bg_card'], relief="flat", padx=10,
                                 command=lambda: self._navigate(1), state="disabled")
        self.next_btn.pack(side="right", padx=5)
        
        # Right column (info panel)
        right_col = tk.Frame(content, bg=COLORS['bg_dark'], width=280)
        right_col.pack(side="right", fill="y", padx=(10, 0))
        right_col.pack_propagate(False)
        
        # Stage info
        tk.Label(right_col, text="Selected Stage", font=("Helvetica", 11, "bold"),
                fg=COLORS['text'], bg=COLORS['bg_dark']).pack(pady=(0, 5))
        
        self.info_panel = StageInfoPanel(right_col)
        self.info_panel.pack(fill="x", pady=5)
        
        # Presets
        presets = tk.LabelFrame(right_col, text=" Quick Presets ", font=("Helvetica", 10),
                               fg=COLORS['gold'], bg=COLORS['bg_card'], padx=10, pady=5)
        presets.pack(fill="x", pady=10)
        
        for name, freq, stages in [("🌟 432Hz/8", 432, 8), ("💚 528Hz/5", 528, 5),
                                    ("🌙 Schumann", 432, 13), ("⚡ Quick 3", 432, 3)]:
            tk.Button(presets, text=name, font=("Helvetica", 9), fg=COLORS['bg_dark'],
                     bg=COLORS['gold_dim'], relief="flat", pady=2,
                     command=lambda f=freq, s=stages: self._preset(f, s)).pack(fill="x", pady=2)
        
        # Golden info
        info = tk.LabelFrame(right_col, text=" Golden Constants ", font=("Helvetica", 10),
                            fg=COLORS['gold'], bg=COLORS['bg_card'], padx=10, pady=5)
        info.pack(fill="x", pady=10)
        
        constants = [
            ("φ", f"{PHI:.10f}"),
            ("1/φ", f"{PHI_CONJUGATE:.10f}"),
            ("φ²", f"{PHI_SQUARED:.10f}"),
            ("φ² = φ+1", f"{PHI + 1:.10f}"),
        ]
        
        for name, val in constants:
            row = tk.Frame(info, bg=COLORS['bg_card'])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{name}:", fg=COLORS['text_dim'], bg=COLORS['bg_card'],
                    font=("Helvetica", 9), width=6, anchor="w").pack(side="left")
            tk.Label(row, text=val, fg=COLORS['gold'], bg=COLORS['bg_card'],
                    font=("Helvetica", 9, "bold")).pack(side="left")
        
        # Status bar
        status = tk.Frame(main, bg=COLORS['bg_panel'], height=25)
        status.pack(fill="x", side="bottom", pady=(5, 0))
        
        self.status_var = tk.StringVar(value="✦ Ready • Generate a sequence to analyze ✦")
        tk.Label(status, textvariable=self.status_var, fg=COLORS['gold'], bg=COLORS['bg_panel'],
                font=("Helvetica", 9)).pack(side="left", padx=10)
        
        self.total_duration = tk.StringVar(value="")
        tk.Label(status, textvariable=self.total_duration, fg=COLORS['text_dim'], bg=COLORS['bg_panel'],
                font=("Helvetica", 9)).pack(side="right", padx=10)
    
    def _preset(self, freq, stages):
        self.base_freq.set(freq)
        self.num_stages.set(stages)
    
    def _generate(self):
        if self.is_generating:
            return
        
        self.is_generating = True
        self.gen_btn.config(state="disabled", text="Generating...")
        self.play_btn.config(state="disabled")
        self.play_all_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")
        
        threading.Thread(target=self._gen_thread, daemon=True).start()
    
    def _gen_thread(self):
        def progress_cb(curr, total, msg):
            self.root.after(0, lambda: self.progress_var.set(msg))
        
        self.engine.generate_full_sequence(
            self.num_stages.get(), self.base_freq.get(), progress_cb
        )
        
        self.root.after(0, self._on_generate_complete)
    
    def _on_generate_complete(self):
        self.is_generating = False
        self.gen_btn.config(state="normal", text="✦ GENERATE SEQUENCE")
        self.play_btn.config(state="normal")
        self.play_all_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.prev_btn.config(state="normal")
        self.next_btn.config(state="normal")
        
        # Update visualizations
        self.timeline.set_stages(self.engine.stages)
        self.param_graph.update_parameters(self.engine.stages)
        
        # Select first stage
        self.selected_stage_idx = 0
        self._update_stage_view()
        
        # Update status
        total_samples = len(self.engine.full_left)
        duration = total_samples / 44100
        self.total_duration.set(f"Total: {duration:.1f}s ({duration/60:.1f} min) • {len(self.engine.stages)} segments")
        self.status_var.set(f"✦ Sequence ready with {len(self.engine.stages)} stages ✦")
        self.progress_var.set("Complete!")
    
    def _on_stage_select(self, idx: int):
        self.selected_stage_idx = idx
        self._update_stage_view()
    
    def _navigate(self, direction: int):
        if not self.engine.stages:
            return
        
        new_idx = self.selected_stage_idx + direction
        if 0 <= new_idx < len(self.engine.stages):
            self.selected_stage_idx = new_idx
            self.timeline.set_selected(new_idx)
            self._update_stage_view()
    
    def _update_stage_view(self):
        if not self.engine.stages:
            return
        
        stage = self.engine.stages[self.selected_stage_idx]
        
        # Update waveform
        self.waveform.update_data(stage.left_audio, stage.right_audio, stage.name)
        
        # Update phase
        self.phase_canvas.update_phase(stage.left_audio, stage.right_audio, stage.phase_offset)
        
        # Update info panel
        self.info_panel.update_stage(stage)
        
        # Update stage label
        self.stage_label.config(text=f"Stage: {self.selected_stage_idx + 1}/{len(self.engine.stages)}")
        
        # Update timeline
        self.timeline.set_selected(self.selected_stage_idx)
    
    def _play_stage(self):
        if not self.engine.stages:
            return
        
        stage = self.engine.stages[self.selected_stage_idx]
        self.status_var.set(f"♪ Playing: {stage.name}")
        self.player.play(stage.left_audio, stage.right_audio, 44100)
    
    def _play_all(self):
        if self.engine.full_left is None:
            return
        
        self.status_var.set("♪ Playing full sequence...")
        self.player.play(self.engine.full_left, self.engine.full_right, 44100)
    
    def _stop(self):
        self.player.stop()
        self.status_var.set("✦ Stopped ✦")
    
    def _save(self):
        if self.engine.full_left is None:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")],
            initialfile=f"golden_{int(self.base_freq.get())}Hz_{self.num_stages.get()}stages.wav"
        )
        
        if filename:
            save_wav(filename, self.engine.full_left, self.engine.full_right, 44100)
            self.status_var.set(f"✦ Saved: {os.path.basename(filename)} ✦")
            messagebox.showinfo("Success", f"Saved:\n{filename}\n\nDuration: {len(self.engine.full_left)/44100:.1f}s")
    
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
    print("\n" + "═" * 60)
    print("  DIVINE GOLDEN BINAURAL - PROFESSIONAL ANALYZER")
    print("  Full visualization • Stage navigation • Phase analysis")
    print("═" * 60 + "\n")
    
    GoldenBinauralProfessional().run()
