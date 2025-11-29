"""
═══════════════════════════════════════════════════════════════════════════════
  DIVINE GOLDEN BINAURAL GENERATOR - SACRED GEOMETRY EDITION
═══════════════════════════════════════════════════════════════════════════════

Con TUTTI gli angoli aurei sacri:
- 137.5° - Angolo aureo (fillotassi, girasoli, galassie spirali)
- 137.035999... - Costante struttura fine α⁻¹ (costante fondamentale universo)
- 222.5° - Complemento aureo
- 34°/36° - DNA helix twist angles  
- 72°, 108° - Pentagono/pentagramma sacro
- 144° - Doppio angolo aureo
- 51.83° - Angolo piramide di Giza

Fasi che seguono questi angoli divini per massima coerenza cosmica
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
import math

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS - THE DIVINE MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

# Golden Ratio
PHI = np.float64((1 + np.sqrt(5)) / 2)                    # 1.618033988749895
PHI_CONJUGATE = np.float64(PHI - 1)                        # 0.618033988749895 = 1/φ
PHI_SQUARED = np.float64(PHI * PHI)                        # 2.618033988749895 = φ + 1
PHI_CUBED = np.float64(PHI ** 3)                           # 4.236067977499790
SQRT_5 = np.float64(np.sqrt(5))                            # 2.236067977499790

# === SACRED ANGLES (in degrees) ===

# THE GOLDEN ANGLE - Most important!
GOLDEN_ANGLE_DEG = 360.0 / (PHI ** 2)                      # 137.5077640500378°
GOLDEN_ANGLE_RAD = np.radians(GOLDEN_ANGLE_DEG)            # 2.39996322972865 rad

# FINE STRUCTURE CONSTANT - α⁻¹ ≈ 137.035999...
# This is THE fundamental constant of the universe (electromagnetic coupling)
ALPHA_INVERSE = 137.035999084                               # CODATA 2018 value
FINE_STRUCTURE_ANGLE_RAD = np.radians(ALPHA_INVERSE)       # As angle

# COMPLEMENTARY GOLDEN ANGLE
GOLDEN_COMPLEMENT_DEG = 360.0 - GOLDEN_ANGLE_DEG           # 222.4922359499622°
GOLDEN_COMPLEMENT_RAD = np.radians(GOLDEN_COMPLEMENT_DEG)

# DNA HELIX ANGLES
DNA_TWIST_MAJOR = 34.3                                      # Major groove angle
DNA_TWIST_MINOR = 36.0                                      # Minor groove / bases per turn
DNA_TWIST_RAD_MAJOR = np.radians(DNA_TWIST_MAJOR)
DNA_TWIST_RAD_MINOR = np.radians(DNA_TWIST_MINOR)
DNA_PITCH_ANGLE = 28.0                                      # Helix pitch angle
DNA_PITCH_RAD = np.radians(DNA_PITCH_ANGLE)

# PENTAGON/PENTAGRAM SACRED GEOMETRY
PENTAGON_INTERIOR = 108.0                                   # Interior angle
PENTAGON_EXTERIOR = 72.0                                    # Exterior angle (360/5)
PENTAGRAM_POINT = 36.0                                      # Point angle
PENTAGON_RAD = np.radians(PENTAGON_INTERIOR)
PENTAGRAM_RAD = np.radians(PENTAGRAM_POINT)

# DOUBLE GOLDEN
DOUBLE_GOLDEN_DEG = 2 * GOLDEN_ANGLE_DEG                   # ~275°
GOLDEN_QUARTER = GOLDEN_ANGLE_DEG / 4                      # ~34.38° (close to DNA!)

# PYRAMID OF GIZA
GIZA_ANGLE = 51.827                                         # Face angle
GIZA_RAD = np.radians(GIZA_ANGLE)

# PLANCK ANGLE (Planck length relationship)
PLANCK_ANGLE_DEG = 360.0 * PHI_CONJUGATE * PHI_CONJUGATE * PHI_CONJUGATE  # ~85.4°

# PI relationships
PI_GOLDEN = np.pi * PHI_CONJUGATE                          # ~1.9416...
TAU_GOLDEN = 2 * np.pi * PHI_CONJUGATE                     # ~3.8832...

TWO_PI = np.float64(2 * np.pi)

# Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]

# Lucas numbers (related to golden ratio)
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843]

# Sacred frequencies
SACRED_FREQUENCIES = {
    'schumann': 7.83,           # Earth resonance
    'sacred_om': 136.1,         # Om frequency
    'universal': 432.0,         # Cosmic tuning
    'love': 528.0,              # Solfeggio - DNA repair
    'liberation': 639.0,        # Solfeggio
    'dna': 528.0,               # DNA resonance
    'golden_freq': 432 * PHI_CONJUGATE,  # ~267 Hz
}

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED ANGLE SEQUENCES FOR PHASE PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def get_sacred_phase_sequence(num_stages: int, mode: str = 'golden_spiral') -> List[float]:
    """
    Generate phase sequence using sacred angles
    
    Modes:
    - 'golden_spiral': Each phase += golden angle (137.5°)
    - 'fine_structure': Based on α⁻¹ = 137.036°
    - 'dna_helix': DNA twist angles
    - 'pentagon': Pentagonal progression
    - 'fibonacci_angular': Fibonacci-based angles
    - 'cosmic': Combines multiple sacred ratios
    """
    phases = []
    
    if mode == 'golden_spiral':
        # Classic golden angle progression - like sunflower seeds
        for i in range(num_stages):
            phase = (i * GOLDEN_ANGLE_RAD) % TWO_PI
            phases.append(phase)
    
    elif mode == 'fine_structure':
        # Using the fine structure constant - fundamental physics
        for i in range(num_stages):
            # Phase based on α⁻¹, approaching π for cancellation
            progress = i / max(1, num_stages - 1)
            # Start at α⁻¹ degrees, end at 180°
            angle_deg = ALPHA_INVERSE + (180 - ALPHA_INVERSE) * golden_spiral_interpolation(progress)
            phases.append(np.radians(angle_deg))
    
    elif mode == 'dna_helix':
        # DNA-inspired phase progression
        for i in range(num_stages):
            # Alternating between major and minor groove angles
            if i % 2 == 0:
                base_angle = DNA_TWIST_MAJOR
            else:
                base_angle = DNA_TWIST_MINOR
            
            # Accumulate with golden scaling
            cumulative = sum([DNA_TWIST_RAD_MAJOR if j % 2 == 0 else DNA_TWIST_RAD_MINOR 
                            for j in range(i + 1)])
            phase = cumulative % TWO_PI
            
            # Scale toward π for annealing
            progress = i / max(1, num_stages - 1)
            phase = phase * (1 - progress) + np.pi * progress
            phases.append(phase)
    
    elif mode == 'pentagon':
        # Pentagonal sacred geometry
        pentagon_angles = [0, 72, 144, 216, 288]  # Pentagon vertices
        for i in range(num_stages):
            # Cycle through pentagon, scaling toward 180°
            base = pentagon_angles[i % 5]
            progress = i / max(1, num_stages - 1)
            angle = base + (180 - base) * golden_spiral_interpolation(progress)
            phases.append(np.radians(angle))
    
    elif mode == 'fibonacci_angular':
        # Fibonacci numbers as angles
        for i in range(num_stages):
            fib_idx = min(i + 3, len(FIBONACCI) - 1)
            # Fibonacci number mod 360, scaled by golden ratio
            angle = (FIBONACCI[fib_idx] * PHI_CONJUGATE) % 360
            progress = i / max(1, num_stages - 1)
            angle = angle + (180 - angle) * golden_spiral_interpolation(progress)
            phases.append(np.radians(angle))
    
    elif mode == 'cosmic':
        # Combines multiple sacred constants
        sacred_angles = [
            GOLDEN_ANGLE_DEG,      # 137.5°
            ALPHA_INVERSE,          # 137.036°
            DNA_TWIST_MINOR * 4,    # 144° (Fibonacci!)
            PENTAGON_INTERIOR,      # 108°
            GIZA_ANGLE * 2,         # 103.65°
            PENTAGRAM_POINT * 3,    # 108°
        ]
        
        for i in range(num_stages):
            # Weighted combination of sacred angles
            weights = golden_spiral_interpolation(np.linspace(0, 1, len(sacred_angles)))
            weights = weights / np.sum(weights)
            
            base_angle = np.sum([a * w for a, w in zip(sacred_angles, weights)])
            base_angle = (base_angle * (i + 1) / PHI) % 360
            
            progress = i / max(1, num_stages - 1)
            angle = base_angle + (180 - base_angle) * golden_spiral_interpolation(progress)
            phases.append(np.radians(angle))
    
    else:
        # Default: simple golden progression to π
        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)
            phases.append(np.pi * golden_spiral_interpolation(progress))
    
    return phases


# Colors
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
}


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
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SacredStageData:
    """Complete data for one stage with sacred geometry info"""
    index: int
    name: str
    beat_freq: float
    duration: float
    amplitude: float
    phase_offset: float          # in radians
    phase_degrees: float         # human readable
    sacred_angle_name: str       # which sacred angle this relates to
    transition_time: float
    annealing_progress: float
    left_audio: np.ndarray
    right_audio: np.ndarray
    start_sample: int
    end_sample: int
    stage_type: str


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED BINAURAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SacredBinauralEngine:
    """Engine with sacred geometry phase progressions"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._phase_left = 0.0
        self._phase_right = 0.0
        self.stages: List[SacredStageData] = []
        self.full_left: Optional[np.ndarray] = None
        self.full_right: Optional[np.ndarray] = None
        self.phase_mode = 'golden_spiral'
    
    def reset(self):
        self._phase_left = 0.0
        self._phase_right = 0.0
        self.stages = []
        self.full_left = None
        self.full_right = None
    
    def _get_sacred_angle_name(self, angle_rad: float) -> str:
        """Identify which sacred angle this phase is closest to"""
        angle_deg = np.degrees(angle_rad) % 360
        
        sacred_refs = [
            (GOLDEN_ANGLE_DEG, "Golden Angle (φ)"),
            (ALPHA_INVERSE, "Fine Structure (α⁻¹)"),
            (DNA_TWIST_MAJOR, "DNA Major"),
            (DNA_TWIST_MINOR, "DNA Minor"),
            (PENTAGON_INTERIOR, "Pentagon 108°"),
            (PENTAGON_EXTERIOR, "Pentagon 72°"),
            (PENTAGRAM_POINT, "Pentagram 36°"),
            (144.0, "Fibonacci 144°"),
            (GIZA_ANGLE, "Giza Pyramid"),
            (180.0, "Cancellation π"),
            (90.0, "Quadrature"),
            (60.0, "Hexagon"),
            (45.0, "Octagon"),
        ]
        
        closest = min(sacred_refs, key=lambda x: min(abs(angle_deg - x[0]), abs(angle_deg - x[0] + 360), abs(angle_deg - x[0] - 360)))
        return closest[1]
    
    def _generate_segment(self, base_freq: float, beat_freq: float, duration: float,
                          amplitude: float, phase_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        omega_left = TWO_PI * base_freq
        omega_right = TWO_PI * (base_freq + beat_freq)
        
        phase_left = self._phase_left + omega_left * t + phase_offset
        phase_right = self._phase_right + omega_right * t + phase_offset + GOLDEN_ANGLE_RAD
        
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
        phase_right = self._phase_right + phase_right_cumulative + GOLDEN_ANGLE_RAD + phase_offset
        
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
                               phase_mode: str = 'golden_spiral',
                               progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete sequence with sacred phase progression"""
        self.reset()
        self.phase_mode = phase_mode
        
        left_segments, right_segments = [], []
        current_sample = 0
        
        # Get sacred phase sequence
        sacred_phases = get_sacred_phase_sequence(num_stages, phase_mode)
        
        # Calculate all stage parameters
        stage_params = []
        for i in range(num_stages):
            beat_freq = base_frequency / (PHI ** (i + 3))
            fib_idx = min(i + 5, len(FIBONACCI) - 1)
            duration = FIBONACCI[fib_idx] * PHI_CONJUGATE
            transition_time = duration * PHI_CONJUGATE * PHI_CONJUGATE
            
            annealing = i / max(1, num_stages - 1)
            phase_offset = sacred_phases[i]
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
        
        # Generate each stage
        for idx, params in enumerate(stage_params):
            if progress_callback:
                progress_callback(idx, num_stages * 2, f"Stage {idx+1}: {self._get_sacred_angle_name(params['phase_offset'])}")
            
            left, right = self._generate_segment(
                base_frequency, params['beat_freq'], params['duration'],
                params['amplitude'], params['phase_offset']
            )
            left = self._apply_envelope(left)
            right = self._apply_envelope(right)
            
            stage = SacredStageData(
                index=idx,
                name=f"Stage {idx+1}: {phase_mode.replace('_', ' ').title()}",
                beat_freq=params['beat_freq'],
                duration=params['duration'],
                amplitude=params['amplitude'],
                phase_offset=params['phase_offset'],
                phase_degrees=np.degrees(params['phase_offset']),
                sacred_angle_name=self._get_sacred_angle_name(params['phase_offset']),
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
            
            # Transition
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
                
                trans_stage = SacredStageData(
                    index=idx,
                    name=f"Transition {idx+1}→{idx+2}",
                    beat_freq=(params['beat_freq'] + next_p['beat_freq']) / 2,
                    duration=params['transition_time'],
                    amplitude=(params['amplitude'] + next_p['amplitude']) / 2,
                    phase_offset=(params['phase_offset'] + next_p['phase_offset']) / 2,
                    phase_degrees=np.degrees((params['phase_offset'] + next_p['phase_offset']) / 2),
                    sacred_angle_name="Transition",
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
            progress_callback(num_stages * 2 - 1, num_stages * 2, "Phase Cancellation → Silence")
        
        cancel_l, cancel_r = self._generate_cancellation(
            base_frequency, stage_params[-1]['amplitude'] if stage_params else 0.1
        )
        
        cancel_stage = SacredStageData(
            index=num_stages,
            name="Final: Cancellation (180° = π)",
            beat_freq=0,
            duration=len(cancel_l) / self.sample_rate,
            amplitude=0,
            phase_offset=np.pi,
            phase_degrees=180.0,
            sacred_angle_name="Perfect Cancellation π",
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
        
        self.full_left = np.concatenate([self._apply_envelope(s) for s in left_segments])
        self.full_right = np.concatenate([self._apply_envelope(s) for s in right_segments])
        
        return self.full_left, self.full_right
    
    def _generate_cancellation(self, frequency: float, start_amplitude: float):
        duration = FIBONACCI[7] * PHI_CONJUGATE
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        progress = golden_spiral_interpolation(np.linspace(0, 1, num_samples))
        phase_diff = np.pi * progress  # Approach π (180°) for perfect cancellation
        amplitude = start_amplitude * (1 - progress)
        
        omega = TWO_PI * frequency
        phase = self._phase_left + omega * t
        
        return amplitude * np.sin(phase), amplitude * np.sin(phase + phase_diff)


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

class SacredGeometryCanvas(tk.Canvas):
    """Visualizes sacred angles on a circle"""
    
    def __init__(self, parent, size, **kwargs):
        super().__init__(parent, width=size, height=size, bg=COLORS['bg_dark'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.size = size
        self.cx, self.cy = size // 2, size // 2
        self.radius = size // 2 - 30
        self._draw_base()
    
    def _draw_base(self):
        self.delete("base")
        
        # Main circle
        self.create_oval(self.cx - self.radius, self.cy - self.radius,
                        self.cx + self.radius, self.cy + self.radius,
                        outline=COLORS['border'], width=2, tags="base")
        
        # Cardinal directions
        for angle, label in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
            rad = np.radians(angle)
            x = self.cx + (self.radius + 15) * np.cos(-rad + np.pi/2)
            y = self.cy - (self.radius + 15) * np.sin(-rad + np.pi/2)
            self.create_text(x, y, text=label, fill=COLORS['text_dim'], font=("Helvetica", 8), tags="base")
        
        # Draw sacred angle markers
        sacred_angles = [
            (GOLDEN_ANGLE_DEG, COLORS['gold'], "φ"),
            (ALPHA_INVERSE, COLORS['cyan'], "α⁻¹"),
            (144, COLORS['purple'], "Fib"),
            (108, COLORS['green'], "⬠"),
            (DNA_TWIST_MINOR, COLORS['orange'], "DNA"),
        ]
        
        for angle, color, label in sacred_angles:
            rad = np.radians(angle)
            x1 = self.cx + (self.radius - 10) * np.cos(-rad + np.pi/2)
            y1 = self.cy - (self.radius - 10) * np.sin(-rad + np.pi/2)
            x2 = self.cx + (self.radius + 5) * np.cos(-rad + np.pi/2)
            y2 = self.cy - (self.radius + 5) * np.sin(-rad + np.pi/2)
            self.create_line(x1, y1, x2, y2, fill=color, width=2, tags="base")
    
    def update_phase(self, phase_rad: float, phases_history: List[float] = None):
        self._draw_base()
        self.delete("phase")
        
        # Draw phase history if available
        if phases_history:
            for i, ph in enumerate(phases_history):
                alpha = 0.3 + 0.7 * (i / len(phases_history))
                x = self.cx + self.radius * 0.6 * np.cos(-ph + np.pi/2)
                y = self.cy - self.radius * 0.6 * np.sin(-ph + np.pi/2)
                size = 3 + 3 * (i / len(phases_history))
                self.create_oval(x-size, y-size, x+size, y+size, fill=COLORS['gold_dim'], outline='', tags="phase")
        
        # Current phase vector
        x = self.cx + self.radius * 0.9 * np.cos(-phase_rad + np.pi/2)
        y = self.cy - self.radius * 0.9 * np.sin(-phase_rad + np.pi/2)
        self.create_line(self.cx, self.cy, x, y, fill=COLORS['gold_bright'], width=3, arrow=tk.LAST, tags="phase")
        
        # Phase point
        self.create_oval(x-6, y-6, x+6, y+6, fill=COLORS['gold_bright'], outline=COLORS['white'], width=2, tags="phase")
        
        # Angle text
        deg = np.degrees(phase_rad) % 360
        self.create_text(self.cx, self.size - 15, text=f"{deg:.2f}°", fill=COLORS['gold'],
                        font=("Helvetica", 12, "bold"), tags="phase")


class WaveformCanvas(tk.Canvas):
    def __init__(self, parent, width, height, title="Waveform", **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_dark'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self.title = title
        self._draw_grid()
    
    def _draw_grid(self):
        self.delete("grid")
        cy = self.h // 2
        self.create_line(0, cy, self.w, cy, fill=COLORS['border'], tags="grid")
        self.create_text(10, 10, text=self.title, anchor="nw", fill=COLORS['text_dim'], font=("Helvetica", 9), tags="grid")
    
    def update_data(self, left: np.ndarray, right: np.ndarray, title: str = None):
        if title:
            self.title = title
        self._draw_grid()
        self.delete("wf")
        
        if left is None or len(left) == 0:
            return
        
        cy = self.h // 2
        scale = self.h * 0.4
        step = max(1, len(left) // (self.w * 2))
        left_d, right_d = left[::step], right[::step]
        x_scale = self.w / len(left_d)
        
        pts_l = []
        pts_r = []
        for i, (l, r) in enumerate(zip(left_d, right_d)):
            x = i * x_scale
            pts_l.extend([x, cy - l * scale])
            pts_r.extend([x, cy + r * scale])
        
        if len(pts_l) >= 4:
            self.create_line(pts_l, fill=COLORS['gold'], width=1, smooth=True, tags="wf")
            self.create_line(pts_r, fill=COLORS['cyan'], width=1, smooth=True, tags="wf")


class SequenceTimeline(tk.Canvas):
    def __init__(self, parent, width, height, on_click=None, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_panel'],
                        highlightthickness=1, highlightbackground=COLORS['border'], **kwargs)
        self.w, self.h = width, height
        self.stages = []
        self.selected = 0
        self.on_click = on_click
        self.bind("<Button-1>", self._handle_click)
    
    def set_stages(self, stages):
        self.stages = stages
        self._redraw()
    
    def set_selected(self, idx):
        self.selected = idx
        self._redraw()
    
    def _redraw(self):
        self.delete("all")
        if not self.stages:
            return
        
        total = self.stages[-1].end_sample
        margin = 10
        bar_h = 30
        bar_y = (self.h - bar_h) // 2
        
        for i, s in enumerate(self.stages):
            x1 = margin + (s.start_sample / total) * (self.w - 2*margin)
            x2 = margin + (s.end_sample / total) * (self.w - 2*margin)
            
            if s.stage_type == 'beat':
                color = COLORS['gold'] if i == self.selected else COLORS['gold_dim']
            elif s.stage_type == 'transition':
                color = COLORS['cyan'] if i == self.selected else COLORS['border']
            else:
                color = COLORS['purple'] if i == self.selected else COLORS['text_dim']
            
            self.create_rectangle(x1, bar_y, x2, bar_y + bar_h, fill=color, outline='')
        
        # Selection
        s = self.stages[self.selected]
        x1 = margin + (s.start_sample / total) * (self.w - 2*margin)
        x2 = margin + (s.end_sample / total) * (self.w - 2*margin)
        self.create_rectangle(x1-2, bar_y-3, x2+2, bar_y+bar_h+3, outline=COLORS['gold_bright'], width=2)
        
        self.create_text(self.w//2, 8, text=f"{s.name} • {s.phase_degrees:.1f}° ({s.sacred_angle_name})",
                        fill=COLORS['gold'], font=("Helvetica", 9))
    
    def _handle_click(self, event):
        if not self.stages:
            return
        margin = 10
        total = self.stages[-1].end_sample
        click_pos = ((event.x - margin) / (self.w - 2*margin)) * total
        
        for i, s in enumerate(self.stages):
            if s.start_sample <= click_pos < s.end_sample:
                self.selected = i
                self._redraw()
                if self.on_click:
                    self.on_click(i)
                break


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class SacredBinauralApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("✦ Sacred Geometry Binaural Generator ✦")
        self.root.configure(bg=COLORS['bg_dark'])
        
        h = 920
        w = int(h * PHI)
        self.root.geometry(f"{w}x{h}")
        
        self.engine = SacredBinauralEngine(44100)
        self.player = AudioPlayer()
        self.is_generating = False
        self.selected_idx = 0
        
        self.base_freq = tk.DoubleVar(value=432.0)
        self.num_stages = tk.IntVar(value=8)
        self.phase_mode = tk.StringVar(value='golden_spiral')
        
        self._setup_ui()
    
    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill="both", expand=True, padx=13, pady=8)
        
        # Header
        tk.Label(main, text="✦ SACRED GEOMETRY BINAURAL GENERATOR ✦",
                font=("Helvetica", 18, "bold"), fg=COLORS['gold'], bg=COLORS['bg_dark']).pack()
        
        constants_text = f"φ={PHI:.10f} • α⁻¹={ALPHA_INVERSE} • Golden∠={GOLDEN_ANGLE_DEG:.4f}°"
        tk.Label(main, text=constants_text, font=("Helvetica", 9), fg=COLORS['text_dim'], bg=COLORS['bg_dark']).pack()
        
        # Controls
        ctrl = tk.Frame(main, bg=COLORS['bg_panel'], padx=10, pady=8)
        ctrl.pack(fill="x", pady=5)
        
        left_ctrl = tk.Frame(ctrl, bg=COLORS['bg_panel'])
        left_ctrl.pack(side="left")
        
        tk.Label(left_ctrl, text="Base Freq:", fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side="left")
        tk.Spinbox(left_ctrl, from_=100, to=963, textvariable=self.base_freq, width=6,
                  bg=COLORS['bg_card'], fg=COLORS['gold']).pack(side="left", padx=5)
        
        tk.Label(left_ctrl, text="Stages:", fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side="left", padx=(10, 0))
        tk.Spinbox(left_ctrl, from_=3, to=13, textvariable=self.num_stages, width=4,
                  bg=COLORS['bg_card'], fg=COLORS['gold']).pack(side="left", padx=5)
        
        tk.Label(left_ctrl, text="Phase Mode:", fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side="left", padx=(10, 0))
        modes = ttk.Combobox(left_ctrl, textvariable=self.phase_mode, width=15, state="readonly",
                            values=['golden_spiral', 'fine_structure', 'dna_helix', 'pentagon', 'fibonacci_angular', 'cosmic'])
        modes.pack(side="left", padx=5)
        
        right_ctrl = tk.Frame(ctrl, bg=COLORS['bg_panel'])
        right_ctrl.pack(side="right")
        
        self.gen_btn = tk.Button(right_ctrl, text="✦ GENERATE", font=("Helvetica", 11, "bold"),
                                fg=COLORS['bg_dark'], bg=COLORS['gold'], command=self._generate)
        self.gen_btn.pack(side="left", padx=5)
        
        self.play_btn = tk.Button(right_ctrl, text="▶ Play", fg=COLORS['bg_dark'], bg=COLORS['green'],
                                 command=self._play_stage, state="disabled")
        self.play_btn.pack(side="left", padx=5)
        
        self.play_all_btn = tk.Button(right_ctrl, text="▶▶ All", fg=COLORS['bg_dark'], bg=COLORS['cyan'],
                                     command=self._play_all, state="disabled")
        self.play_all_btn.pack(side="left", padx=5)
        
        tk.Button(right_ctrl, text="⏹", fg=COLORS['white'], bg=COLORS['red'], command=self._stop).pack(side="left", padx=5)
        
        self.save_btn = tk.Button(right_ctrl, text="💾", fg=COLORS['bg_dark'], bg=COLORS['orange'],
                                 command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        tk.Label(ctrl, textvariable=self.progress_var, fg=COLORS['text_dim'], bg=COLORS['bg_panel']).pack(side="right", padx=10)
        
        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill="both", expand=True, pady=5)
        
        left = tk.Frame(content, bg=COLORS['bg_dark'])
        left.pack(side="left", fill="both", expand=True)
        
        # Sacred geometry display
        geo_row = tk.Frame(left, bg=COLORS['bg_dark'])
        geo_row.pack(fill="x", pady=5)
        
        self.sacred_canvas = SacredGeometryCanvas(geo_row, 250)
        self.sacred_canvas.pack(side="left", padx=5)
        
        # Sacred constants panel
        const_frame = tk.LabelFrame(geo_row, text=" Sacred Constants ", fg=COLORS['gold'],
                                   bg=COLORS['bg_card'], padx=10, pady=5)
        const_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        constants = [
            ("Golden Angle", f"{GOLDEN_ANGLE_DEG:.6f}°", "φ⁻² × 360°"),
            ("Fine Structure α⁻¹", f"{ALPHA_INVERSE}", "Universe constant"),
            ("DNA Major", f"{DNA_TWIST_MAJOR}°", "Helix groove"),
            ("DNA Minor", f"{DNA_TWIST_MINOR}°", "Bases/turn"),
            ("Pentagon", f"{PENTAGON_INTERIOR}°", "Sacred geometry"),
            ("Pentagram", f"{PENTAGRAM_POINT}°", "Star point"),
            ("Fibonacci", "144°", "F(12)"),
            ("Giza", f"{GIZA_ANGLE}°", "Pyramid face"),
        ]
        
        for i, (name, val, desc) in enumerate(constants):
            row = tk.Frame(const_frame, bg=COLORS['bg_card'])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, fg=COLORS['text_dim'], bg=COLORS['bg_card'], width=15, anchor="w",
                    font=("Helvetica", 8)).pack(side="left")
            tk.Label(row, text=val, fg=COLORS['gold'], bg=COLORS['bg_card'], width=12,
                    font=("Helvetica", 9, "bold")).pack(side="left")
            tk.Label(row, text=desc, fg=COLORS['text_dim'], bg=COLORS['bg_card'],
                    font=("Helvetica", 7)).pack(side="left")
        
        # Waveform
        self.waveform = WaveformCanvas(left, 750, 180)
        self.waveform.pack(fill="x", pady=5)
        
        # Timeline
        self.timeline = SequenceTimeline(left, 750, 60, on_click=self._on_stage_select)
        self.timeline.pack(fill="x", pady=5)
        
        # Navigation
        nav = tk.Frame(left, bg=COLORS['bg_dark'])
        nav.pack(fill="x", pady=5)
        
        self.prev_btn = tk.Button(nav, text="◀ Prev", fg=COLORS['white'], bg=COLORS['bg_card'],
                                 command=lambda: self._navigate(-1), state="disabled")
        self.prev_btn.pack(side="left", padx=5)
        
        self.stage_label = tk.Label(nav, text="Stage: -/-", fg=COLORS['gold'], bg=COLORS['bg_dark'],
                                   font=("Helvetica", 12, "bold"))
        self.stage_label.pack(side="left", expand=True)
        
        self.next_btn = tk.Button(nav, text="Next ▶", fg=COLORS['white'], bg=COLORS['bg_card'],
                                 command=lambda: self._navigate(1), state="disabled")
        self.next_btn.pack(side="right", padx=5)
        
        # Right panel - stage info
        right = tk.Frame(content, bg=COLORS['bg_dark'], width=350)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        
        info = tk.LabelFrame(right, text=" Current Stage ", fg=COLORS['gold'], bg=COLORS['bg_card'], padx=10, pady=10)
        info.pack(fill="x", pady=5)
        
        self.info_labels = {}
        params = ["Type", "Beat Freq", "Duration", "Amplitude", "Phase", "Sacred Angle", "Annealing"]
        
        for p in params:
            row = tk.Frame(info, bg=COLORS['bg_card'])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{p}:", fg=COLORS['text_dim'], bg=COLORS['bg_card'], width=12, anchor="w").pack(side="left")
            self.info_labels[p] = tk.Label(row, text="-", fg=COLORS['gold'], bg=COLORS['bg_card'],
                                          font=("Helvetica", 10, "bold"))
            self.info_labels[p].pack(side="left")
        
        # Phase modes explanation
        modes_info = tk.LabelFrame(right, text=" Phase Modes ", fg=COLORS['gold'], bg=COLORS['bg_card'], padx=10, pady=5)
        modes_info.pack(fill="x", pady=10)
        
        mode_descs = [
            ("golden_spiral", "Sunflower seed pattern"),
            ("fine_structure", "α⁻¹ physics constant"),
            ("dna_helix", "DNA twist angles"),
            ("pentagon", "Sacred geometry 72°/108°"),
            ("fibonacci_angular", "Fibonacci as degrees"),
            ("cosmic", "Combined sacred ratios"),
        ]
        
        for mode, desc in mode_descs:
            row = tk.Frame(modes_info, bg=COLORS['bg_card'])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=mode.replace('_', ' ').title(), fg=COLORS['cyan'], bg=COLORS['bg_card'],
                    width=14, anchor="w", font=("Helvetica", 8)).pack(side="left")
            tk.Label(row, text=desc, fg=COLORS['text_dim'], bg=COLORS['bg_card'],
                    font=("Helvetica", 7)).pack(side="left")
        
        # Status
        self.status_var = tk.StringVar(value="✦ Ready • Select a phase mode and generate ✦")
        tk.Label(main, textvariable=self.status_var, fg=COLORS['gold'], bg=COLORS['bg_dark'],
                font=("Helvetica", 9)).pack(side="bottom", pady=5)
    
    def _generate(self):
        if self.is_generating:
            return
        
        self.is_generating = True
        self.gen_btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.play_all_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        
        threading.Thread(target=self._gen_thread, daemon=True).start()
    
    def _gen_thread(self):
        def cb(curr, total, msg):
            self.root.after(0, lambda: self.progress_var.set(msg))
        
        self.engine.generate_full_sequence(
            self.num_stages.get(), self.base_freq.get(), self.phase_mode.get(), cb
        )
        self.root.after(0, self._on_complete)
    
    def _on_complete(self):
        self.is_generating = False
        self.gen_btn.config(state="normal")
        self.play_btn.config(state="normal")
        self.play_all_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.prev_btn.config(state="normal")
        self.next_btn.config(state="normal")
        
        self.timeline.set_stages(self.engine.stages)
        self.selected_idx = 0
        self._update_view()
        
        duration = len(self.engine.full_left) / 44100
        self.status_var.set(f"✦ {len(self.engine.stages)} stages • {duration:.1f}s • Mode: {self.phase_mode.get()} ✦")
    
    def _on_stage_select(self, idx):
        self.selected_idx = idx
        self._update_view()
    
    def _navigate(self, d):
        if not self.engine.stages:
            return
        new_idx = self.selected_idx + d
        if 0 <= new_idx < len(self.engine.stages):
            self.selected_idx = new_idx
            self.timeline.set_selected(new_idx)
            self._update_view()
    
    def _update_view(self):
        if not self.engine.stages:
            return
        
        s = self.engine.stages[self.selected_idx]
        
        self.waveform.update_data(s.left_audio, s.right_audio, s.name)
        
        # Sacred geometry with phase history
        phases_so_far = [st.phase_offset for st in self.engine.stages[:self.selected_idx+1] if st.stage_type == 'beat']
        self.sacred_canvas.update_phase(s.phase_offset, phases_so_far)
        
        self.stage_label.config(text=f"Stage: {self.selected_idx + 1}/{len(self.engine.stages)}")
        
        self.info_labels["Type"].config(text=s.stage_type.upper())
        self.info_labels["Beat Freq"].config(text=f"{s.beat_freq:.4f} Hz")
        self.info_labels["Duration"].config(text=f"{s.duration:.2f} s")
        self.info_labels["Amplitude"].config(text=f"{s.amplitude:.5f}")
        self.info_labels["Phase"].config(text=f"{s.phase_degrees:.2f}°")
        self.info_labels["Sacred Angle"].config(text=s.sacred_angle_name)
        self.info_labels["Annealing"].config(text=f"{s.annealing_progress * 100:.1f}%")
    
    def _play_stage(self):
        if self.engine.stages:
            s = self.engine.stages[self.selected_idx]
            self.player.play(s.left_audio, s.right_audio)
    
    def _play_all(self):
        if self.engine.full_left is not None:
            self.player.play(self.engine.full_left, self.engine.full_right)
    
    def _stop(self):
        self.player.stop()
    
    def _save(self):
        if self.engine.full_left is None:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")],
            initialfile=f"sacred_{self.phase_mode.get()}_{int(self.base_freq.get())}Hz.wav"
        )
        if filename:
            save_wav(filename, self.engine.full_left, self.engine.full_right)
            messagebox.showinfo("Saved", f"Saved:\n{filename}")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", lambda: (self.player.stop(), self.root.destroy()))
        self.root.mainloop()


if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  SACRED GEOMETRY BINAURAL GENERATOR")
    print(f"  Golden Angle: {GOLDEN_ANGLE_DEG:.6f}° • α⁻¹: {ALPHA_INVERSE}")
    print(f"  DNA: {DNA_TWIST_MAJOR}°/{DNA_TWIST_MINOR}° • Pentagon: {PENTAGON_INTERIOR}°")
    print("═" * 65 + "\n")
    
    SacredBinauralApp().run()
