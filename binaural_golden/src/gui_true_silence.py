"""
═══════════════════════════════════════════════════════════════════════════════
  DIVINE GOLDEN BINAURAL - TRUE SILENCE ANNEALING
═══════════════════════════════════════════════════════════════════════════════

CORRETTO: Porta VERAMENTE verso il silenzio tramite:

1. FASE L-R → 180° (π) = cancellazione perfetta
2. AMPIEZZA → 0 
3. BEAT FREQUENCY → 0 (convergenza)
4. Mix L+R → silenzio quando fase = 180°

La cancellazione di fase funziona così:
- L = sin(ωt)
- R = sin(ωt + π) = -sin(ωt)
- L + R = 0 (silenzio perfetto quando fase = 180°)

Con angoli sacri 137.5°, α⁻¹, DNA, etc.
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
from typing import Tuple, List, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = np.float64((1 + np.sqrt(5)) / 2)
PHI_CONJUGATE = np.float64(PHI - 1)
PHI_SQUARED = np.float64(PHI * PHI)
TWO_PI = np.float64(2 * np.pi)

# SACRED ANGLES
GOLDEN_ANGLE_DEG = 360.0 / (PHI ** 2)              # 137.5077°
GOLDEN_ANGLE_RAD = np.radians(GOLDEN_ANGLE_DEG)
ALPHA_INVERSE = 137.035999084                       # Fine structure constant
DNA_TWIST_MAJOR = 34.3
DNA_TWIST_MINOR = 36.0
PENTAGON_INTERIOR = 108.0
PENTAGRAM_POINT = 36.0
GIZA_ANGLE = 51.827

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

COLORS = {
    'bg_dark': '#0d1117', 'bg_panel': '#161b22', 'bg_card': '#21262d',
    'border': '#30363d', 'gold': '#D4AF37', 'gold_bright': '#FFD700',
    'gold_dim': '#8B7355', 'cyan': '#58a6ff', 'green': '#3fb950',
    'red': '#f85149', 'purple': '#a371f7', 'orange': '#d29922',
    'white': '#f0f6fc', 'text': '#c9d1d9', 'text_dim': '#8b949e',
}


def golden_spiral_interpolation(t):
    """Non-linear golden easing [0,1] → [0,1]"""
    t = np.asarray(t, dtype=np.float64)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    
    result = np.zeros_like(t)
    result[t <= 0] = 0.0
    result[t >= 1] = 1.0
    
    mask = (t > 0) & (t < 1)
    if np.any(mask):
        tm = t[mask]
        theta = tm * np.pi * PHI_CONJUGATE
        golden_ease = (1 - np.cos(theta)) / 2
        smoothstep = tm * tm * (3 - 2 * tm)
        result[mask] = smoothstep * PHI_CONJUGATE + golden_ease * (1 - PHI_CONJUGATE)
    
    return float(result[0]) if scalar else result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE SEQUENCES - ALL MUST END AT π (180°) FOR CANCELLATION!
# ═══════════════════════════════════════════════════════════════════════════════

def get_annealing_phase_sequence(num_stages: int, mode: str = 'golden_to_pi') -> List[dict]:
    """
    Generate phase sequence that ALWAYS ends at π (180°) for cancellation.
    Returns dict with phase_offset and phase_diff_lr (the L-R difference).
    
    KEY INSIGHT: For cancellation, we need L and R to be 180° out of phase.
    So phase_diff_lr must approach π.
    """
    stages = []
    
    if mode == 'golden_to_pi':
        # Start at golden angle, smoothly approach 180° for cancellation
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            annealed = golden_spiral_interpolation(progress)
            
            # Phase difference L-R: starts at golden angle, ends at π
            phase_diff = GOLDEN_ANGLE_RAD * (1 - annealed) + np.pi * annealed
            
            # Base phase offset (just for variety, less important)
            base_phase = (i * GOLDEN_ANGLE_RAD) % TWO_PI
            
            stages.append({
                'phase_diff_lr': phase_diff,  # THIS is what creates cancellation
                'base_phase': base_phase,
                'progress': progress,
            })
    
    elif mode == 'fine_structure_to_pi':
        # Start at α⁻¹ (137.036°), approach 180°
        alpha_rad = np.radians(ALPHA_INVERSE)
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            annealed = golden_spiral_interpolation(progress)
            
            phase_diff = alpha_rad * (1 - annealed) + np.pi * annealed
            base_phase = (i * alpha_rad) % TWO_PI
            
            stages.append({
                'phase_diff_lr': phase_diff,
                'base_phase': base_phase,
                'progress': progress,
            })
    
    elif mode == 'dna_to_pi':
        # DNA twist angles accumulating toward 180°
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            annealed = golden_spiral_interpolation(progress)
            
            # Accumulate DNA angles
            dna_accum = sum([np.radians(DNA_TWIST_MAJOR if j % 2 == 0 else DNA_TWIST_MINOR) 
                           for j in range(i + 1)]) % TWO_PI
            
            # Blend toward π
            phase_diff = dna_accum * (1 - annealed) + np.pi * annealed
            
            stages.append({
                'phase_diff_lr': phase_diff,
                'base_phase': dna_accum,
                'progress': progress,
            })
    
    elif mode == 'pentagon_to_pi':
        # Pentagon vertices (72° steps) toward 180°
        pentagon_rad = np.radians(72.0)
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            annealed = golden_spiral_interpolation(progress)
            
            pent_phase = ((i % 5) * pentagon_rad) % TWO_PI
            phase_diff = pent_phase * (1 - annealed) + np.pi * annealed
            
            stages.append({
                'phase_diff_lr': phase_diff,
                'base_phase': pent_phase,
                'progress': progress,
            })
    
    elif mode == 'sacred_blend':
        # Blend of multiple sacred angles
        sacred = [GOLDEN_ANGLE_RAD, np.radians(ALPHA_INVERSE), 
                  np.radians(DNA_TWIST_MINOR * 5), np.radians(PENTAGON_INTERIOR)]
        
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            annealed = golden_spiral_interpolation(progress)
            
            # Weighted average of sacred angles
            w = np.array([PHI_CONJUGATE, 1-PHI_CONJUGATE, PHI_CONJUGATE**2, PHI_CONJUGATE**3])
            w = w / np.sum(w)
            sacred_mix = np.sum([a * ww for a, ww in zip(sacred, w)])
            
            base = (sacred_mix * (i + 1)) % TWO_PI
            phase_diff = base * (1 - annealed) + np.pi * annealed
            
            stages.append({
                'phase_diff_lr': phase_diff,
                'base_phase': base,
                'progress': progress,
            })
    
    else:  # direct_to_pi
        # Simple direct approach to π
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            phase_diff = np.pi * golden_spiral_interpolation(progress)
            
            stages.append({
                'phase_diff_lr': phase_diff,
                'base_phase': 0,
                'progress': progress,
            })
    
    return stages


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageData:
    index: int
    name: str
    beat_freq: float
    duration: float
    amplitude: float
    phase_diff_lr: float      # L-R phase difference (KEY for cancellation!)
    phase_diff_degrees: float
    annealing: float          # 0=start, 1=silence
    left_audio: np.ndarray
    right_audio: np.ndarray
    mixed_audio: np.ndarray   # L+R to show cancellation
    start_sample: int
    end_sample: int
    stage_type: str
    rms_mono: float           # RMS of mixed signal (should → 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TRUE ANNEALING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TrueAnnealingEngine:
    """
    Engine that ACTUALLY produces silence through phase cancellation.
    
    The key is: when L and R are 180° out of phase, L + R = 0
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._phase = 0.0
        self.stages: List[StageData] = []
        self.full_left: Optional[np.ndarray] = None
        self.full_right: Optional[np.ndarray] = None
        self.full_mixed: Optional[np.ndarray] = None
    
    def reset(self):
        self._phase = 0.0
        self.stages = []
        self.full_left = None
        self.full_right = None
        self.full_mixed = None
    
    def _generate_segment(self, base_freq: float, beat_freq: float, duration: float,
                          amplitude: float, phase_diff_lr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate binaural with specific L-R phase difference.
        
        L = amp * sin(ω_L * t + base_phase)
        R = amp * sin(ω_R * t + base_phase + phase_diff_lr)
        
        When phase_diff_lr = π, and beat_freq ≈ 0:
        L + R ≈ amp*sin(θ) + amp*sin(θ + π) = amp*sin(θ) - amp*sin(θ) = 0
        """
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        omega_left = TWO_PI * base_freq
        omega_right = TWO_PI * (base_freq + beat_freq)
        
        # Continuous phase
        phase_left = self._phase + omega_left * t
        phase_right = self._phase + omega_right * t + phase_diff_lr
        
        left = amplitude * np.sin(phase_left)
        right = amplitude * np.sin(phase_right)
        
        # Mixed signal shows cancellation
        mixed = (left + right) / 2
        
        # Update phase accumulator
        self._phase = (self._phase + omega_left * duration) % TWO_PI
        
        return left, right, mixed
    
    def _generate_transition(self, base_freq: float, 
                             beat1: float, beat2: float,
                             amp1: float, amp2: float,
                             diff1: float, diff2: float,
                             duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Smooth transition with phase continuity"""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        interp = golden_spiral_interpolation(np.linspace(0, 1, num_samples))
        
        beat_freq = beat1 + (beat2 - beat1) * interp
        amplitude = amp1 + (amp2 - amp1) * interp
        phase_diff = diff1 + (diff2 - diff1) * interp
        
        omega_left = TWO_PI * base_freq
        dt = 1.0 / self.sample_rate
        omega_right = TWO_PI * (base_freq + beat_freq)
        
        phase_left = self._phase + omega_left * t
        phase_right = self._phase + np.cumsum(omega_right) * dt + phase_diff
        
        left = amplitude * np.sin(phase_left)
        right = amplitude * np.sin(phase_right)
        mixed = (left + right) / 2
        
        self._phase = (self._phase + omega_left * duration) % TWO_PI
        
        return left, right, mixed
    
    def _apply_envelope(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        fade = min(64, n // 10)
        if fade < 2:
            return audio
        
        audio = audio.copy()
        audio[:fade] *= np.linspace(0, 1, fade) ** 2
        audio[-fade:] *= np.linspace(1, 0, fade) ** 2
        return audio
    
    def generate_sequence(self, num_stages: int, base_frequency: float,
                          phase_mode: str = 'golden_to_pi',
                          progress_callback=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete annealing sequence to TRUE SILENCE"""
        self.reset()
        
        left_segs, right_segs, mixed_segs = [], [], []
        current_sample = 0
        
        # Get phase sequence
        phase_seq = get_annealing_phase_sequence(num_stages, phase_mode)
        
        # Calculate stage parameters with TRUE annealing
        params = []
        for i in range(num_stages):
            ps = phase_seq[i]
            progress = ps['progress']
            annealed = golden_spiral_interpolation(progress)
            
            # Beat frequency → 0 (convergence)
            base_beat = base_frequency / (PHI ** 3)  # Start beat
            beat_freq = base_beat * (1 - annealed ** 0.5)  # → 0
            
            # Amplitude → 0 
            # Use stronger decay: cubic annealing
            amplitude = (1 - annealed) ** 1.5  # Goes to 0!
            
            # Duration from Fibonacci
            fib_idx = min(i + 5, len(FIBONACCI) - 1)
            duration = FIBONACCI[fib_idx] * PHI_CONJUGATE
            transition_time = duration * PHI_CONJUGATE ** 2
            
            params.append({
                'index': i,
                'beat_freq': beat_freq,
                'duration': duration,
                'amplitude': amplitude,
                'phase_diff_lr': ps['phase_diff_lr'],
                'transition_time': transition_time,
                'annealing': annealed,
            })
        
        # Generate stages
        for idx, p in enumerate(params):
            if progress_callback:
                progress_callback(idx, num_stages * 2, 
                    f"Stage {idx+1}: {np.degrees(p['phase_diff_lr']):.1f}° | Amp: {p['amplitude']:.3f}")
            
            left, right, mixed = self._generate_segment(
                base_frequency, p['beat_freq'], p['duration'],
                p['amplitude'], p['phase_diff_lr']
            )
            
            left = self._apply_envelope(left)
            right = self._apply_envelope(right)
            mixed = self._apply_envelope(mixed)
            
            # Calculate RMS of mixed (should decrease toward 0)
            rms = np.sqrt(np.mean(mixed ** 2))
            
            stage = StageData(
                index=idx,
                name=f"Stage {idx+1}",
                beat_freq=p['beat_freq'],
                duration=p['duration'],
                amplitude=p['amplitude'],
                phase_diff_lr=p['phase_diff_lr'],
                phase_diff_degrees=np.degrees(p['phase_diff_lr']),
                annealing=p['annealing'],
                left_audio=left,
                right_audio=right,
                mixed_audio=mixed,
                start_sample=current_sample,
                end_sample=current_sample + len(left),
                stage_type='beat',
                rms_mono=rms
            )
            self.stages.append(stage)
            
            left_segs.append(left)
            right_segs.append(right)
            mixed_segs.append(mixed)
            current_sample += len(left)
            
            # Transition
            if idx < num_stages - 1:
                if progress_callback:
                    progress_callback(idx * 2 + 1, num_stages * 2, f"Transition {idx+1}→{idx+2}")
                
                np_ = params[idx + 1]
                tl, tr, tm = self._generate_transition(
                    base_frequency,
                    p['beat_freq'], np_['beat_freq'],
                    p['amplitude'], np_['amplitude'],
                    p['phase_diff_lr'], np_['phase_diff_lr'],
                    p['transition_time']
                )
                
                tl = self._apply_envelope(tl)
                tr = self._apply_envelope(tr)
                tm = self._apply_envelope(tm)
                
                trans_rms = np.sqrt(np.mean(tm ** 2))
                
                trans = StageData(
                    index=idx,
                    name=f"Trans {idx+1}→{idx+2}",
                    beat_freq=(p['beat_freq'] + np_['beat_freq']) / 2,
                    duration=p['transition_time'],
                    amplitude=(p['amplitude'] + np_['amplitude']) / 2,
                    phase_diff_lr=(p['phase_diff_lr'] + np_['phase_diff_lr']) / 2,
                    phase_diff_degrees=np.degrees((p['phase_diff_lr'] + np_['phase_diff_lr']) / 2),
                    annealing=(p['annealing'] + np_['annealing']) / 2,
                    left_audio=tl,
                    right_audio=tr,
                    mixed_audio=tm,
                    start_sample=current_sample,
                    end_sample=current_sample + len(tl),
                    stage_type='transition',
                    rms_mono=trans_rms
                )
                self.stages.append(trans)
                
                left_segs.append(tl)
                right_segs.append(tr)
                mixed_segs.append(tm)
                current_sample += len(tl)
        
        # Final silence stage - perfect cancellation
        if progress_callback:
            progress_callback(num_stages * 2 - 1, num_stages * 2, "Final: Perfect Silence (π)")
        
        final_l, final_r, final_m = self._generate_final_silence(base_frequency)
        
        final_rms = np.sqrt(np.mean(final_m ** 2))
        
        final = StageData(
            index=num_stages,
            name="SILENCE (180° = π)",
            beat_freq=0,
            duration=len(final_l) / self.sample_rate,
            amplitude=0,
            phase_diff_lr=np.pi,
            phase_diff_degrees=180.0,
            annealing=1.0,
            left_audio=final_l,
            right_audio=final_r,
            mixed_audio=final_m,
            start_sample=current_sample,
            end_sample=current_sample + len(final_l),
            stage_type='silence',
            rms_mono=final_rms
        )
        self.stages.append(final)
        
        left_segs.append(final_l)
        right_segs.append(final_r)
        mixed_segs.append(final_m)
        
        self.full_left = np.concatenate(left_segs)
        self.full_right = np.concatenate(right_segs)
        self.full_mixed = np.concatenate(mixed_segs)
        
        return self.full_left, self.full_right, self.full_mixed
    
    def _generate_final_silence(self, frequency: float):
        """Final stage: amplitude → 0, phase → π, result → silence"""
        duration = FIBONACCI[8] * PHI_CONJUGATE
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        progress = golden_spiral_interpolation(np.linspace(0, 1, num_samples))
        
        # Phase difference approaches π perfectly
        phase_diff = np.pi * (0.9 + 0.1 * progress)  # 0.9π → π
        
        # Amplitude decays to 0
        start_amp = 0.1  # Already low from previous stage
        amplitude = start_amp * (1 - progress) ** 2  # Quadratic decay to 0
        
        omega = TWO_PI * frequency
        phase = self._phase + omega * t
        
        left = amplitude * np.sin(phase)
        right = amplitude * np.sin(phase + phase_diff)
        mixed = (left + right) / 2  # This should be very close to 0
        
        return left, right, mixed


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
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
    left_int, right_int = (left * 32767).astype(np.int16), (right * 32767).astype(np.int16)
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

class PhaseCanvas(tk.Canvas):
    """Shows L-R phase difference approaching 180°"""
    def __init__(self, parent, size, **kwargs):
        super().__init__(parent, width=size, height=size, bg=COLORS['bg_dark'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.size = size
        self.cx, self.cy = size // 2, size // 2
        self.r = size // 2 - 30
        self._draw_base()
    
    def _draw_base(self):
        self.delete("base")
        
        # Circle
        self.create_oval(self.cx - self.r, self.cy - self.r,
                        self.cx + self.r, self.cy + self.r,
                        outline=COLORS['border'], width=2, tags="base")
        
        # 180° marker (target!)
        x180 = self.cx - self.r
        self.create_line(self.cx, self.cy, x180, self.cy, fill=COLORS['red'], width=3, tags="base")
        self.create_text(x180 - 10, self.cy, text="180°", fill=COLORS['red'], 
                        font=("Helvetica", 10, "bold"), anchor="e", tags="base")
        self.create_text(x180 - 10, self.cy + 15, text="SILENCE", fill=COLORS['red'],
                        font=("Helvetica", 8), anchor="e", tags="base")
        
        # Other markers
        for deg, label in [(0, "0°"), (90, "90°"), (137.5, "φ"), (270, "270°")]:
            rad = np.radians(deg)
            x = self.cx + self.r * 1.1 * np.cos(-rad + np.pi/2)
            y = self.cy - self.r * 1.1 * np.sin(-rad + np.pi/2)
            self.create_text(x, y, text=label, fill=COLORS['text_dim'], font=("Helvetica", 8), tags="base")
    
    def update(self, phase_diff: float, history: List[float] = None):
        self._draw_base()
        self.delete("phase")
        
        # History trail
        if history:
            for i, ph in enumerate(history):
                alpha = 0.2 + 0.8 * (i / len(history))
                rad = -ph + np.pi / 2
                x = self.cx + self.r * 0.7 * np.cos(rad)
                y = self.cy - self.r * 0.7 * np.sin(rad)
                s = 2 + 4 * (i / len(history))
                self.create_oval(x-s, y-s, x+s, y+s, fill=COLORS['gold_dim'], outline='', tags="phase")
        
        # Current phase vector
        rad = -phase_diff + np.pi / 2
        x = self.cx + self.r * 0.9 * np.cos(rad)
        y = self.cy - self.r * 0.9 * np.sin(rad)
        
        self.create_line(self.cx, self.cy, x, y, fill=COLORS['gold_bright'], width=3, arrow=tk.LAST, tags="phase")
        self.create_oval(x-8, y-8, x+8, y+8, fill=COLORS['gold_bright'], outline=COLORS['white'], width=2, tags="phase")
        
        # Degree text
        deg = np.degrees(phase_diff)
        color = COLORS['green'] if deg > 170 else COLORS['gold']
        self.create_text(self.cx, self.size - 12, text=f"L-R Phase: {deg:.1f}°",
                        fill=color, font=("Helvetica", 11, "bold"), tags="phase")


class AnnealingGraph(tk.Canvas):
    """Shows amplitude, phase, RMS all trending to silence"""
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_dark'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self._draw_grid()
    
    def _draw_grid(self):
        self.delete("grid")
        m = 50  # margin
        
        # Axes
        self.create_line(m, 20, m, self.h - 20, fill=COLORS['border'], tags="grid")
        self.create_line(m, self.h - 20, self.w - 10, self.h - 20, fill=COLORS['border'], tags="grid")
        
        # Y labels
        self.create_text(m - 5, 25, text="100%", anchor="e", fill=COLORS['text_dim'], font=("Helvetica", 7), tags="grid")
        self.create_text(m - 5, self.h // 2, text="50%", anchor="e", fill=COLORS['text_dim'], font=("Helvetica", 7), tags="grid")
        self.create_text(m - 5, self.h - 25, text="0%", anchor="e", fill=COLORS['text_dim'], font=("Helvetica", 7), tags="grid")
        
        # 180° line
        y180 = 25 + (self.h - 45) * (1 - 180/360)
        self.create_line(m, y180, self.w - 10, y180, fill=COLORS['red'], dash=(4, 4), tags="grid")
        self.create_text(self.w - 10, y180 - 8, text="180° (silence)", anchor="e", 
                        fill=COLORS['red'], font=("Helvetica", 7), tags="grid")
        
        self.create_text(self.w // 2, 8, text="Annealing Progress → Silence", anchor="n",
                        fill=COLORS['text'], font=("Helvetica", 9), tags="grid")
    
    def update(self, stages: List[StageData]):
        self._draw_grid()
        self.delete("data")
        
        if not stages:
            return
        
        beat_stages = [s for s in stages if s.stage_type == 'beat']
        if not beat_stages:
            return
        
        n = len(beat_stages)
        m = 50
        gw = self.w - m - 10
        gh = self.h - 45
        step = gw / max(1, n - 1)
        
        # Amplitude line (gold) - should → 0
        amp_pts = []
        for i, s in enumerate(beat_stages):
            x = m + i * step
            y = 25 + gh * (1 - s.amplitude)
            amp_pts.extend([x, y])
        
        if len(amp_pts) >= 4:
            self.create_line(amp_pts, fill=COLORS['gold'], width=2, smooth=True, tags="data")
        
        # Phase difference line (cyan) - should → 180°
        phase_pts = []
        for i, s in enumerate(beat_stages):
            x = m + i * step
            y = 25 + gh * (1 - s.phase_diff_degrees / 360)
            phase_pts.extend([x, y])
        
        if len(phase_pts) >= 4:
            self.create_line(phase_pts, fill=COLORS['cyan'], width=2, smooth=True, tags="data")
        
        # RMS line (purple) - should → 0
        rms_pts = []
        max_rms = max(s.rms_mono for s in beat_stages) or 1
        for i, s in enumerate(beat_stages):
            x = m + i * step
            y = 25 + gh * (1 - s.rms_mono / max_rms)
            rms_pts.extend([x, y])
        
        if len(rms_pts) >= 4:
            self.create_line(rms_pts, fill=COLORS['purple'], width=2, smooth=True, tags="data")
        
        # Legend
        self.create_rectangle(self.w - 100, 25, self.w - 10, 75, fill=COLORS['bg_card'], tags="data")
        self.create_line(self.w - 95, 35, self.w - 75, 35, fill=COLORS['gold'], width=2, tags="data")
        self.create_text(self.w - 70, 35, text="Amp", anchor="w", fill=COLORS['text_dim'], font=("Helvetica", 7), tags="data")
        self.create_line(self.w - 95, 50, self.w - 75, 50, fill=COLORS['cyan'], width=2, tags="data")
        self.create_text(self.w - 70, 50, text="Phase", anchor="w", fill=COLORS['text_dim'], font=("Helvetica", 7), tags="data")
        self.create_line(self.w - 95, 65, self.w - 75, 65, fill=COLORS['purple'], width=2, tags="data")
        self.create_text(self.w - 70, 65, text="RMS→0", anchor="w", fill=COLORS['text_dim'], font=("Helvetica", 7), tags="data")


class WaveformCanvas(tk.Canvas):
    """Shows L, R, and MIXED (L+R) - mixed should → flat line (silence)"""
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_dark'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
    
    def update(self, left: np.ndarray, right: np.ndarray, mixed: np.ndarray, title: str = ""):
        self.delete("all")
        
        if left is None or len(left) == 0:
            return
        
        # Title
        self.create_text(10, 10, text=title, anchor="nw", fill=COLORS['text'], font=("Helvetica", 9))
        
        cy = self.h // 2
        scale = self.h * 0.35
        
        step = max(1, len(left) // (self.w * 2))
        ld, rd, md = left[::step], right[::step], mixed[::step]
        xs = self.w / len(ld)
        
        # Draw L (gold, upper)
        pts_l = []
        for i, v in enumerate(ld):
            pts_l.extend([i * xs, cy - v * scale])
        if len(pts_l) >= 4:
            self.create_line(pts_l, fill=COLORS['gold'], width=1, tags="wf")
        
        # Draw R (cyan, lower)
        pts_r = []
        for i, v in enumerate(rd):
            pts_r.extend([i * xs, cy + v * scale * 0.5])
        if len(pts_r) >= 4:
            self.create_line(pts_r, fill=COLORS['cyan'], width=1, tags="wf")
        
        # Draw MIXED (purple, should be flat when cancelled!)
        pts_m = []
        for i, v in enumerate(md):
            pts_m.extend([i * xs, cy + v * scale * 1.5])
        if len(pts_m) >= 4:
            self.create_line(pts_m, fill=COLORS['purple'], width=2, tags="wf")
        
        # Labels
        self.create_text(self.w - 5, cy - scale * 0.5, text="L", anchor="e", fill=COLORS['gold'], font=("Helvetica", 8, "bold"))
        self.create_text(self.w - 5, cy + scale * 0.25, text="R", anchor="e", fill=COLORS['cyan'], font=("Helvetica", 8, "bold"))
        self.create_text(self.w - 5, cy + scale * 0.75, text="L+R", anchor="e", fill=COLORS['purple'], font=("Helvetica", 8, "bold"))
        
        # RMS of mixed
        rms = np.sqrt(np.mean(mixed ** 2))
        color = COLORS['green'] if rms < 0.05 else COLORS['orange'] if rms < 0.2 else COLORS['text']
        self.create_text(self.w // 2, self.h - 10, text=f"Mixed RMS: {rms:.4f} {'(SILENCE!)' if rms < 0.01 else ''}",
                        fill=color, font=("Helvetica", 9, "bold"))


class Timeline(tk.Canvas):
    def __init__(self, parent, width, height, on_click=None, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_panel'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self.stages = []
        self.selected = 0
        self.on_click = on_click
        self.bind("<Button-1>", self._click)
    
    def set_stages(self, stages):
        self.stages = stages
        self._draw()
    
    def set_selected(self, idx):
        self.selected = idx
        self._draw()
    
    def _draw(self):
        self.delete("all")
        if not self.stages:
            return
        
        total = self.stages[-1].end_sample
        m, bh = 10, 30
        by = (self.h - bh) // 2
        
        for i, s in enumerate(self.stages):
            x1 = m + (s.start_sample / total) * (self.w - 2*m)
            x2 = m + (s.end_sample / total) * (self.w - 2*m)
            
            # Color by annealing level
            ann = s.annealing
            if s.stage_type == 'silence':
                color = COLORS['green']
            elif ann > 0.8:
                color = COLORS['purple'] if i == self.selected else '#553366'
            elif ann > 0.5:
                color = COLORS['cyan'] if i == self.selected else '#335566'
            else:
                color = COLORS['gold'] if i == self.selected else COLORS['gold_dim']
            
            self.create_rectangle(x1, by, x2, by + bh, fill=color, outline='')
        
        # Selection
        s = self.stages[self.selected]
        x1 = m + (s.start_sample / total) * (self.w - 2*m)
        x2 = m + (s.end_sample / total) * (self.w - 2*m)
        self.create_rectangle(x1-2, by-3, x2+2, by+bh+3, outline=COLORS['white'], width=2)
        
        # Info
        self.create_text(self.w // 2, 8, text=f"{s.name} | Phase: {s.phase_diff_degrees:.1f}° | Amp: {s.amplitude:.3f} | RMS: {s.rms_mono:.4f}",
                        fill=COLORS['gold'], font=("Helvetica", 9))
    
    def _click(self, e):
        if not self.stages:
            return
        m = 10
        total = self.stages[-1].end_sample
        pos = ((e.x - m) / (self.w - 2*m)) * total
        
        for i, s in enumerate(self.stages):
            if s.start_sample <= pos < s.end_sample:
                self.selected = i
                self._draw()
                if self.on_click:
                    self.on_click(i)
                break


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TrueAnnealingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("✦ TRUE SILENCE ANNEALING - Phase Cancellation ✦")
        self.root.configure(bg=COLORS['bg_dark'])
        
        h = 900
        w = int(h * PHI)
        self.root.geometry(f"{w}x{h}")
        
        self.engine = TrueAnnealingEngine(44100)
        self.player = AudioPlayer()
        self.is_generating = False
        self.selected = 0
        
        self.base_freq = tk.DoubleVar(value=432.0)
        self.num_stages = tk.IntVar(value=8)
        self.phase_mode = tk.StringVar(value='golden_to_pi')
        
        self._setup()
    
    def _setup(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill="both", expand=True, padx=13, pady=8)
        
        # Header
        tk.Label(main, text="✦ TRUE SILENCE ANNEALING ✦", font=("Helvetica", 18, "bold"),
                fg=COLORS['gold'], bg=COLORS['bg_dark']).pack()
        tk.Label(main, text="Phase L-R → 180° = Perfect Cancellation | Amplitude → 0 | Beat → 0 | L+R → SILENCE",
                font=("Helvetica", 9), fg=COLORS['cyan'], bg=COLORS['bg_dark']).pack()
        
        # Controls
        ctrl = tk.Frame(main, bg=COLORS['bg_panel'], padx=10, pady=8)
        ctrl.pack(fill="x", pady=5)
        
        left_c = tk.Frame(ctrl, bg=COLORS['bg_panel'])
        left_c.pack(side="left")
        
        tk.Label(left_c, text="Base:", fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side="left")
        tk.Spinbox(left_c, from_=100, to=963, textvariable=self.base_freq, width=5,
                  bg=COLORS['bg_card'], fg=COLORS['gold']).pack(side="left", padx=3)
        
        tk.Label(left_c, text="Stages:", fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side="left", padx=(10, 0))
        tk.Spinbox(left_c, from_=3, to=13, textvariable=self.num_stages, width=3,
                  bg=COLORS['bg_card'], fg=COLORS['gold']).pack(side="left", padx=3)
        
        tk.Label(left_c, text="Mode:", fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side="left", padx=(10, 0))
        ttk.Combobox(left_c, textvariable=self.phase_mode, width=16, state="readonly",
                    values=['golden_to_pi', 'fine_structure_to_pi', 'dna_to_pi', 
                           'pentagon_to_pi', 'sacred_blend', 'direct_to_pi']).pack(side="left", padx=3)
        
        right_c = tk.Frame(ctrl, bg=COLORS['bg_panel'])
        right_c.pack(side="right")
        
        self.gen_btn = tk.Button(right_c, text="✦ GENERATE → SILENCE", font=("Helvetica", 10, "bold"),
                                fg=COLORS['bg_dark'], bg=COLORS['gold'], command=self._generate)
        self.gen_btn.pack(side="left", padx=3)
        
        self.play_btn = tk.Button(right_c, text="▶", fg=COLORS['bg_dark'], bg=COLORS['green'],
                                 command=self._play_stage, state="disabled")
        self.play_btn.pack(side="left", padx=3)
        
        self.play_all = tk.Button(right_c, text="▶▶", fg=COLORS['bg_dark'], bg=COLORS['cyan'],
                                 command=self._play_all, state="disabled")
        self.play_all.pack(side="left", padx=3)
        
        tk.Button(right_c, text="⏹", fg=COLORS['white'], bg=COLORS['red'],
                 command=self._stop).pack(side="left", padx=3)
        
        self.save_btn = tk.Button(right_c, text="💾", fg=COLORS['bg_dark'], bg=COLORS['orange'],
                                 command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=3)
        
        self.prog = tk.StringVar(value="Ready")
        tk.Label(ctrl, textvariable=self.prog, fg=COLORS['text_dim'], bg=COLORS['bg_panel']).pack(side="right", padx=10)
        
        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill="both", expand=True, pady=5)
        
        left = tk.Frame(content, bg=COLORS['bg_dark'])
        left.pack(side="left", fill="both", expand=True)
        
        # Visualizations row
        viz = tk.Frame(left, bg=COLORS['bg_dark'])
        viz.pack(fill="x", pady=5)
        
        self.phase_canvas = PhaseCanvas(viz, 220)
        self.phase_canvas.pack(side="left", padx=5)
        
        self.annealing_graph = AnnealingGraph(viz, 550, 220)
        self.annealing_graph.pack(side="left", padx=5, fill="x", expand=True)
        
        # Waveform
        self.waveform = WaveformCanvas(left, 780, 180)
        self.waveform.pack(fill="x", pady=5)
        
        # Timeline
        self.timeline = Timeline(left, 780, 60, on_click=self._select)
        self.timeline.pack(fill="x", pady=5)
        
        # Nav
        nav = tk.Frame(left, bg=COLORS['bg_dark'])
        nav.pack(fill="x", pady=5)
        
        self.prev_btn = tk.Button(nav, text="◀", fg=COLORS['white'], bg=COLORS['bg_card'],
                                 command=lambda: self._nav(-1), state="disabled")
        self.prev_btn.pack(side="left", padx=5)
        
        self.stage_lbl = tk.Label(nav, text="-", fg=COLORS['gold'], bg=COLORS['bg_dark'],
                                 font=("Helvetica", 11, "bold"))
        self.stage_lbl.pack(side="left", expand=True)
        
        self.next_btn = tk.Button(nav, text="▶", fg=COLORS['white'], bg=COLORS['bg_card'],
                                 command=lambda: self._nav(1), state="disabled")
        self.next_btn.pack(side="right", padx=5)
        
        # Right panel - info
        right = tk.Frame(content, bg=COLORS['bg_dark'], width=300)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        
        info = tk.LabelFrame(right, text=" Stage Details ", fg=COLORS['gold'], bg=COLORS['bg_card'], padx=10, pady=10)
        info.pack(fill="x", pady=5)
        
        self.info = {}
        for p in ["Type", "Phase L-R", "Amplitude", "Beat Freq", "Duration", "RMS Mono", "Annealing"]:
            r = tk.Frame(info, bg=COLORS['bg_card'])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=f"{p}:", fg=COLORS['text_dim'], bg=COLORS['bg_card'], width=10, anchor="w").pack(side="left")
            self.info[p] = tk.Label(r, text="-", fg=COLORS['gold'], bg=COLORS['bg_card'], font=("Helvetica", 10, "bold"))
            self.info[p].pack(side="left")
        
        # Explanation
        exp = tk.LabelFrame(right, text=" How Cancellation Works ", fg=COLORS['cyan'], bg=COLORS['bg_card'], padx=10, pady=5)
        exp.pack(fill="x", pady=10)
        
        explanation = """When L and R are 180° apart:
L = sin(θ)
R = sin(θ + π) = -sin(θ)

L + R = sin(θ) - sin(θ) = 0

→ PERFECT SILENCE

The purple "L+R" line in the
waveform should become FLAT
as we approach 180°."""
        
        tk.Label(exp, text=explanation, fg=COLORS['text_dim'], bg=COLORS['bg_card'],
                font=("Helvetica", 8), justify="left").pack()
        
        # Status
        self.status = tk.StringVar(value="✦ Generate to see true phase cancellation → silence ✦")
        tk.Label(main, textvariable=self.status, fg=COLORS['gold'], bg=COLORS['bg_dark'],
                font=("Helvetica", 9)).pack(side="bottom", pady=5)
    
    def _generate(self):
        if self.is_generating:
            return
        
        self.is_generating = True
        self.gen_btn.config(state="disabled")
        
        threading.Thread(target=self._gen_thread, daemon=True).start()
    
    def _gen_thread(self):
        def cb(c, t, m):
            self.root.after(0, lambda: self.prog.set(m))
        
        self.engine.generate_sequence(self.num_stages.get(), self.base_freq.get(), self.phase_mode.get(), cb)
        self.root.after(0, self._done)
    
    def _done(self):
        self.is_generating = False
        self.gen_btn.config(state="normal")
        self.play_btn.config(state="normal")
        self.play_all.config(state="normal")
        self.save_btn.config(state="normal")
        self.prev_btn.config(state="normal")
        self.next_btn.config(state="normal")
        
        self.timeline.set_stages(self.engine.stages)
        self.annealing_graph.update(self.engine.stages)
        
        self.selected = 0
        self._update()
        
        dur = len(self.engine.full_left) / 44100
        final_rms = self.engine.stages[-1].rms_mono if self.engine.stages else 0
        self.status.set(f"✦ {len(self.engine.stages)} stages | {dur:.1f}s | Final RMS: {final_rms:.6f} {'← SILENCE!' if final_rms < 0.01 else ''} ✦")
    
    def _select(self, idx):
        self.selected = idx
        self._update()
    
    def _nav(self, d):
        if not self.engine.stages:
            return
        new = self.selected + d
        if 0 <= new < len(self.engine.stages):
            self.selected = new
            self.timeline.set_selected(new)
            self._update()
    
    def _update(self):
        if not self.engine.stages:
            return
        
        s = self.engine.stages[self.selected]
        
        self.waveform.update(s.left_audio, s.right_audio, s.mixed_audio, s.name)
        
        phases = [st.phase_diff_lr for st in self.engine.stages[:self.selected+1] if st.stage_type == 'beat']
        self.phase_canvas.update(s.phase_diff_lr, phases)
        
        self.timeline.set_selected(self.selected)
        self.stage_lbl.config(text=f"Stage {self.selected + 1}/{len(self.engine.stages)} | {s.name}")
        
        self.info["Type"].config(text=s.stage_type.upper())
        self.info["Phase L-R"].config(text=f"{s.phase_diff_degrees:.2f}°")
        self.info["Amplitude"].config(text=f"{s.amplitude:.5f}")
        self.info["Beat Freq"].config(text=f"{s.beat_freq:.4f} Hz")
        self.info["Duration"].config(text=f"{s.duration:.2f} s")
        self.info["RMS Mono"].config(text=f"{s.rms_mono:.6f}")
        self.info["Annealing"].config(text=f"{s.annealing * 100:.1f}%")
    
    def _play_stage(self):
        if self.engine.stages:
            s = self.engine.stages[self.selected]
            self.player.play(s.left_audio, s.right_audio)
    
    def _play_all(self):
        if self.engine.full_left is not None:
            self.player.play(self.engine.full_left, self.engine.full_right)
    
    def _stop(self):
        self.player.stop()
    
    def _save(self):
        if self.engine.full_left is None:
            return
        
        fn = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")],
            initialfile=f"true_silence_{self.phase_mode.get()}_{int(self.base_freq.get())}Hz.wav")
        if fn:
            save_wav(fn, self.engine.full_left, self.engine.full_right)
            messagebox.showinfo("Saved", f"Saved:\n{fn}\n\nFinal RMS: {self.engine.stages[-1].rms_mono:.6f}")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", lambda: (self.player.stop(), self.root.destroy()))
        self.root.mainloop()


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  TRUE SILENCE ANNEALING")
    print("  Phase L-R → 180° = Cancellation")
    print("  L + R → 0 = SILENCE")
    print("═" * 60 + "\n")
    
    TrueAnnealingApp().run()
