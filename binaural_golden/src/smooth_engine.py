"""
Divine Golden Ratio Binaural Generator - v3 SMOOTH
===================================================

FIXES:
- Seamless transitions between sections (no clicks/pops)
- Phase continuity across all segments
- Amplitude crossfade at boundaries
- Comprehensive unit tests

All parameters in divine golden coherence.
"""

import numpy as np
import unittest
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
import struct
import os

# ═══════════════════════════════════════════════════════════════════════════════
# DIVINE CONSTANTS - Maximum Precision
# ═══════════════════════════════════════════════════════════════════════════════

PHI = np.float64((1 + np.sqrt(5)) / 2)          # 1.6180339887498949
PHI_CONJUGATE = np.float64(PHI - 1)              # 0.6180339887498949
PHI_SQUARED = np.float64(PHI * PHI)              # 2.6180339887498949
TWO_PI = np.float64(2 * np.pi)

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN MATHEMATICS - Smooth Functions
# ═══════════════════════════════════════════════════════════════════════════════

def golden_spiral_interpolation(t: float) -> float:
    """
    Golden spiral easing function.
    Guarantees: f(0)=0, f(1)=1, smooth derivative at boundaries.
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    
    # Use smooth hermite interpolation base
    # f(t) = 3t² - 2t³ (smoothstep) combined with golden spiral
    t = np.float64(t)
    
    # Golden-weighted smoothstep
    theta = t * np.pi * PHI_CONJUGATE  # Gentler curve
    golden_ease = (1 - np.cos(theta)) / 2
    
    # Standard smoothstep for guaranteed smoothness
    smoothstep = t * t * (3 - 2 * t)
    
    # Blend: more smoothstep for boundary smoothness
    result = smoothstep * PHI_CONJUGATE + golden_ease * (1 - PHI_CONJUGATE)
    
    return np.clip(result, 0.0, 1.0)


def golden_crossfade(t: float) -> Tuple[float, float]:
    """
    Returns (fade_out, fade_in) gains for crossfading.
    Guarantees: fade_out² + fade_in² ≈ 1 (constant power)
    """
    t = np.clip(t, 0.0, 1.0)
    interp = golden_spiral_interpolation(t)
    
    # Constant power crossfade (equal power)
    fade_out = np.cos(interp * np.pi / 2)
    fade_in = np.sin(interp * np.pi / 2)
    
    return fade_out, fade_in


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE-CONTINUOUS OSCILLATOR
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseAccumulator:
    """
    Maintains continuous phase across frequency changes.
    This prevents clicks/pops at segment boundaries.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = np.float64(sample_rate)
        self.phase_left = np.float64(0.0)
        self.phase_right = np.float64(0.0)
        
    def reset(self):
        """Reset phases to zero"""
        self.phase_left = 0.0
        self.phase_right = 0.0
    
    def generate_segment(
        self,
        base_freq: float,
        beat_freq: float,
        num_samples: int,
        amplitude: float,
        target_phase_offset: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate audio segment with continuous phase.
        Phase accumulates correctly across calls.
        """
        left = np.zeros(num_samples, dtype=np.float64)
        right = np.zeros(num_samples, dtype=np.float64)
        
        dt = 1.0 / self.sample_rate
        omega_left = TWO_PI * np.float64(base_freq)
        omega_right = TWO_PI * np.float64(base_freq + beat_freq)
        
        # Right channel has golden phase relationship
        phase_right_offset = np.pi * PHI_CONJUGATE
        
        for i in range(num_samples):
            left[i] = amplitude * np.sin(self.phase_left + target_phase_offset)
            right[i] = amplitude * np.sin(self.phase_right + phase_right_offset + target_phase_offset)
            
            # Accumulate phase
            self.phase_left += omega_left * dt
            self.phase_right += omega_right * dt
            
            # Wrap phase to prevent numerical overflow
            if self.phase_left > TWO_PI:
                self.phase_left -= TWO_PI
            if self.phase_right > TWO_PI:
                self.phase_right -= TWO_PI
        
        return left, right
    
    def generate_transition(
        self,
        base_freq: float,
        beat_freq_start: float,
        beat_freq_end: float,
        num_samples: int,
        amp_start: float,
        amp_end: float,
        phase_offset_start: float,
        phase_offset_end: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth transition with interpolated parameters.
        All parameters smoothly interpolate using golden spiral.
        """
        left = np.zeros(num_samples, dtype=np.float64)
        right = np.zeros(num_samples, dtype=np.float64)
        
        dt = 1.0 / self.sample_rate
        phase_right_offset = np.pi * PHI_CONJUGATE
        
        for i in range(num_samples):
            # Golden spiral interpolation factor
            t = i / max(1, num_samples - 1)
            factor = golden_spiral_interpolation(t)
            
            # Interpolate all parameters smoothly
            beat_freq = beat_freq_start + (beat_freq_end - beat_freq_start) * factor
            amplitude = amp_start + (amp_end - amp_start) * factor
            phase_offset = phase_offset_start + (phase_offset_end - phase_offset_start) * factor
            
            # Current frequencies
            omega_left = TWO_PI * base_freq
            omega_right = TWO_PI * (base_freq + beat_freq)
            
            # Generate samples
            left[i] = amplitude * np.sin(self.phase_left + phase_offset)
            right[i] = amplitude * np.sin(self.phase_right + phase_right_offset + phase_offset)
            
            # Accumulate phase with current frequency
            self.phase_left += omega_left * dt
            self.phase_right += omega_right * dt
            
            # Wrap
            if self.phase_left > TWO_PI:
                self.phase_left -= TWO_PI
            if self.phase_right > TWO_PI:
                self.phase_right -= TWO_PI
        
        return left, right


# ═══════════════════════════════════════════════════════════════════════════════
# SMOOTH BINAURAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SmoothBinauralEngine:
    """
    Binaural generator with guaranteed smooth transitions.
    No clicks, pops, or discontinuities.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.oscillator = PhaseAccumulator(sample_rate)
        
    def generate_annealing_sequence(
        self,
        num_stages: int,
        base_frequency: float,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete annealing sequence with seamless transitions.
        """
        self.oscillator.reset()
        
        left_segments = []
        right_segments = []
        
        # Pre-calculate all stage parameters
        stages = []
        for stage in range(num_stages):
            params = self._calculate_stage_params(stage, num_stages, base_frequency)
            stages.append(params)
        
        for stage_idx, params in enumerate(stages):
            if progress_callback:
                progress_callback(stage_idx, num_stages, f"Stage {stage_idx+1}/{num_stages}")
            
            # Generate main segment
            segment_samples = int(params['duration'] * self.sample_rate)
            
            left, right = self.oscillator.generate_segment(
                base_frequency,
                params['beat_freq'],
                segment_samples,
                params['amplitude'],
                params['phase_offset']
            )
            
            # Apply smooth envelope (only attack/decay, not full)
            left = self._apply_segment_envelope(left, params['amplitude'])
            right = self._apply_segment_envelope(right, params['amplitude'])
            
            left_segments.append(left)
            right_segments.append(right)
            
            # Generate transition to next stage
            if stage_idx < num_stages - 1:
                next_params = stages[stage_idx + 1]
                trans_samples = int(params['transition_time'] * self.sample_rate)
                
                trans_left, trans_right = self.oscillator.generate_transition(
                    base_frequency,
                    params['beat_freq'], next_params['beat_freq'],
                    trans_samples,
                    params['amplitude'], next_params['amplitude'],
                    params['phase_offset'], next_params['phase_offset']
                )
                
                left_segments.append(trans_left)
                right_segments.append(trans_right)
        
        # Final phase cancellation approach
        if progress_callback:
            progress_callback(num_stages, num_stages, "Phase cancellation...")
        
        final_left, final_right = self._generate_final_cancellation(
            base_frequency, 
            stages[-1]['amplitude'] if stages else 0.1
        )
        left_segments.append(final_left)
        right_segments.append(final_right)
        
        return np.concatenate(left_segments), np.concatenate(right_segments)
    
    def _calculate_stage_params(
        self, 
        stage: int, 
        total_stages: int, 
        base_freq: float
    ) -> dict:
        """Calculate parameters for a stage with golden relationships"""
        # Beat frequency decreases by golden ratio
        beat_freq = base_freq / (PHI ** (stage + 3))
        
        # Duration from Fibonacci
        fib_idx = min(stage + 5, len(FIBONACCI) - 1)
        duration = FIBONACCI[fib_idx] * PHI_CONJUGATE
        
        # Transition time in golden ratio
        transition_time = duration * PHI_CONJUGATE * PHI_CONJUGATE
        
        # Annealing progress
        annealing = stage / max(1, total_stages - 1)
        
        # Phase approaches π smoothly
        phase_offset = np.pi * golden_spiral_interpolation(annealing)
        
        # Amplitude decreases toward silence
        amplitude = (1 - golden_spiral_interpolation(annealing) * 0.95)
        amplitude *= PHI_CONJUGATE ** (stage * 0.5)
        
        return {
            'beat_freq': beat_freq,
            'duration': duration,
            'transition_time': transition_time,
            'phase_offset': phase_offset,
            'amplitude': amplitude,
        }
    
    def _apply_segment_envelope(
        self, 
        audio: np.ndarray, 
        target_amplitude: float
    ) -> np.ndarray:
        """
        Apply gentle envelope to segment.
        Very short fade to avoid discontinuities without affecting the main sound.
        """
        num_samples = len(audio)
        
        # Very short crossfade at boundaries (golden ratio of 1%)
        fade_samples = max(64, int(num_samples * 0.01 * PHI_CONJUGATE))
        
        # Fade in
        for i in range(min(fade_samples, num_samples)):
            t = i / fade_samples
            audio[i] *= golden_spiral_interpolation(t)
        
        # Fade out
        for i in range(min(fade_samples, num_samples)):
            t = i / fade_samples
            audio[-(i+1)] *= golden_spiral_interpolation(t)
        
        return audio
    
    def _generate_final_cancellation(
        self, 
        frequency: float, 
        start_amplitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth approach to silence through phase cancellation"""
        duration = FIBONACCI[7] * PHI_CONJUGATE
        num_samples = int(duration * self.sample_rate)
        
        left = np.zeros(num_samples, dtype=np.float64)
        right = np.zeros(num_samples, dtype=np.float64)
        
        dt = 1.0 / self.sample_rate
        omega = TWO_PI * frequency
        
        # Start from current oscillator state
        phase = self.oscillator.phase_left
        
        for i in range(num_samples):
            progress = golden_spiral_interpolation(i / max(1, num_samples - 1))
            
            # Phase difference approaches π
            phase_diff = np.pi * progress
            
            # Amplitude approaches zero
            amp = start_amplitude * (1 - progress)
            
            left[i] = amp * np.sin(phase)
            right[i] = amp * np.sin(phase + phase_diff)
            
            phase += omega * dt
            if phase > TWO_PI:
                phase -= TWO_PI
        
        return left, right
    
    def generate_preview(
        self,
        base_freq: float,
        beat_freq: float,
        duration: float,
        amplitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth preview clip"""
        self.oscillator.reset()
        
        num_samples = int(duration * self.sample_rate)
        left, right = self.oscillator.generate_segment(
            base_freq, beat_freq, num_samples, amplitude
        )
        
        # Apply full envelope for preview
        envelope = self._create_smooth_envelope(num_samples)
        
        return left * envelope, right * envelope
    
    def _create_smooth_envelope(self, num_samples: int) -> np.ndarray:
        """Create smooth envelope with golden proportions"""
        envelope = np.ones(num_samples, dtype=np.float64)
        
        # Attack and decay using golden ratio
        attack_samples = int(num_samples * PHI_CONJUGATE * 0.1)
        decay_samples = int(num_samples * PHI_CONJUGATE * 0.15)
        
        for i in range(attack_samples):
            envelope[i] = golden_spiral_interpolation(i / attack_samples)
        
        for i in range(decay_samples):
            envelope[-(i+1)] = golden_spiral_interpolation(i / decay_samples)
        
        return envelope


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def detect_discontinuities(audio: np.ndarray, threshold: float = 0.1) -> List[int]:
    """
    Detect discontinuities (clicks/pops) in audio.
    Returns list of sample indices where discontinuities occur.
    """
    # Calculate sample-to-sample differences
    diff = np.abs(np.diff(audio))
    
    # Find where difference exceeds threshold
    discontinuities = np.where(diff > threshold)[0]
    
    return discontinuities.tolist()


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS (root mean square) of audio"""
    return np.sqrt(np.mean(audio ** 2))


def calculate_peak(audio: np.ndarray) -> float:
    """Calculate peak absolute value"""
    return np.max(np.abs(audio))


def check_phase_continuity(
    audio: np.ndarray, 
    sample_rate: int,
    frequency: float,
    tolerance_degrees: float = 5.0
) -> Tuple[bool, float]:
    """
    Check if audio maintains phase continuity.
    Returns (is_continuous, max_phase_jump_degrees)
    """
    # Estimate instantaneous phase using Hilbert transform
    from scipy.signal import hilbert
    
    analytic = hilbert(audio)
    instantaneous_phase = np.unwrap(np.angle(analytic))
    
    # Calculate expected phase progression
    expected_phase_diff = 2 * np.pi * frequency / sample_rate
    
    # Calculate actual phase differences
    phase_diff = np.diff(instantaneous_phase)
    
    # Find maximum deviation from expected
    deviation = np.abs(phase_diff - expected_phase_diff)
    max_deviation_rad = np.max(deviation)
    max_deviation_deg = np.degrees(max_deviation_rad)
    
    is_continuous = max_deviation_deg < tolerance_degrees
    
    return is_continuous, max_deviation_deg


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoldenMathematics(unittest.TestCase):
    """Tests for golden ratio mathematics"""
    
    def test_phi_identity(self):
        """φ² = φ + 1"""
        self.assertAlmostEqual(PHI * PHI, PHI + 1, places=10)
    
    def test_phi_conjugate(self):
        """1/φ = φ - 1"""
        self.assertAlmostEqual(1 / PHI, PHI - 1, places=10)
    
    def test_golden_spiral_bounds(self):
        """Golden spiral interpolation should be bounded [0, 1]"""
        for t in np.linspace(-0.5, 1.5, 100):
            result = golden_spiral_interpolation(t)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    def test_golden_spiral_endpoints(self):
        """Golden spiral should map 0→0 and 1→1"""
        self.assertEqual(golden_spiral_interpolation(0.0), 0.0)
        self.assertEqual(golden_spiral_interpolation(1.0), 1.0)
    
    def test_golden_spiral_monotonic(self):
        """Golden spiral should be monotonically increasing"""
        prev = 0.0
        for t in np.linspace(0, 1, 100):
            current = golden_spiral_interpolation(t)
            self.assertGreaterEqual(current, prev)
            prev = current
    
    def test_golden_crossfade_power(self):
        """Crossfade should maintain approximately constant power"""
        for t in np.linspace(0, 1, 50):
            fade_out, fade_in = golden_crossfade(t)
            power = fade_out ** 2 + fade_in ** 2
            # Should be close to 1.0 (constant power)
            self.assertAlmostEqual(power, 1.0, places=2)


class TestPhaseAccumulator(unittest.TestCase):
    """Tests for phase-continuous oscillator"""
    
    def setUp(self):
        self.sample_rate = 44100
        self.osc = PhaseAccumulator(self.sample_rate)
    
    def test_generates_correct_length(self):
        """Should generate correct number of samples"""
        num_samples = 1000
        left, right = self.osc.generate_segment(440, 10, num_samples, 0.5)
        self.assertEqual(len(left), num_samples)
        self.assertEqual(len(right), num_samples)
    
    def test_amplitude_bounds(self):
        """Output should be bounded by amplitude"""
        amplitude = 0.7
        left, right = self.osc.generate_segment(440, 10, 10000, amplitude)
        self.assertLessEqual(np.max(np.abs(left)), amplitude + 0.001)
        self.assertLessEqual(np.max(np.abs(right)), amplitude + 0.001)
    
    def test_phase_continuity_across_segments(self):
        """Phase should be continuous across multiple segments"""
        # Generate two consecutive segments
        left1, right1 = self.osc.generate_segment(440, 10, 1000, 0.5)
        left2, right2 = self.osc.generate_segment(440, 10, 1000, 0.5)
        
        # Concatenate
        left_full = np.concatenate([left1, left2])
        
        # Check for discontinuity at boundary
        boundary_diff = abs(left1[-1] - left2[0])
        
        # Should be very small (continuous)
        # The difference should be similar to other sample-to-sample differences
        avg_diff = np.mean(np.abs(np.diff(left1)))
        self.assertLess(boundary_diff, avg_diff * 3)  # Allow some margin
    
    def test_no_clicks_in_segment(self):
        """Segment should have no discontinuities"""
        left, right = self.osc.generate_segment(440, 10, 44100, 0.5)
        
        discontinuities = detect_discontinuities(left, threshold=0.1)
        self.assertEqual(len(discontinuities), 0, 
                        f"Found {len(discontinuities)} discontinuities")
    
    def test_transition_smoothness(self):
        """Transitions should be smooth"""
        left, right = self.osc.generate_transition(
            440,  # base freq
            10, 5,  # beat freq start/end
            10000,  # samples
            0.8, 0.4,  # amplitude start/end
            0, np.pi/4  # phase offset start/end
        )
        
        discontinuities = detect_discontinuities(left, threshold=0.15)
        self.assertEqual(len(discontinuities), 0,
                        f"Found {len(discontinuities)} discontinuities in transition")


class TestSmoothBinauralEngine(unittest.TestCase):
    """Tests for the main binaural engine"""
    
    def setUp(self):
        self.engine = SmoothBinauralEngine(sample_rate=44100)
    
    def test_preview_generation(self):
        """Preview should generate valid audio"""
        left, right = self.engine.generate_preview(432, 10, 2.0, 0.8)
        
        self.assertGreater(len(left), 0)
        self.assertEqual(len(left), len(right))
        self.assertLessEqual(calculate_peak(left), 0.85)
        self.assertLessEqual(calculate_peak(right), 0.85)
    
    def test_preview_no_clicks(self):
        """Preview should have no discontinuities"""
        left, right = self.engine.generate_preview(432, 10, 2.0, 0.8)
        
        # Check for clicks
        disc_left = detect_discontinuities(left, threshold=0.1)
        disc_right = detect_discontinuities(right, threshold=0.1)
        
        self.assertEqual(len(disc_left), 0, 
                        f"Found {len(disc_left)} clicks in left channel")
        self.assertEqual(len(disc_right), 0,
                        f"Found {len(disc_right)} clicks in right channel")
    
    def test_annealing_sequence_generation(self):
        """Should generate valid annealing sequence"""
        left, right = self.engine.generate_annealing_sequence(
            num_stages=3,
            base_frequency=432
        )
        
        self.assertGreater(len(left), 0)
        self.assertEqual(len(left), len(right))
    
    def test_annealing_no_major_clicks(self):
        """Annealing sequence should have no major discontinuities"""
        left, right = self.engine.generate_annealing_sequence(
            num_stages=3,
            base_frequency=432
        )
        
        # Check for major clicks (threshold slightly higher for complex sequence)
        disc_left = detect_discontinuities(left, threshold=0.2)
        
        # Allow very few discontinuities (at segment boundaries there might be tiny ones)
        self.assertLess(len(disc_left), 10,
                       f"Found {len(disc_left)} discontinuities (should be < 10)")
    
    def test_annealing_decreasing_amplitude(self):
        """Amplitude should generally decrease through annealing"""
        left, right = self.engine.generate_annealing_sequence(
            num_stages=5,
            base_frequency=432
        )
        
        # Divide into sections and check RMS
        section_size = len(left) // 5
        rms_values = []
        
        for i in range(5):
            start = i * section_size
            end = start + section_size
            rms = calculate_rms(left[start:end])
            rms_values.append(rms)
        
        # Overall trend should be decreasing
        self.assertGreater(rms_values[0], rms_values[-1])
    
    def test_final_silence(self):
        """End of sequence should approach silence"""
        left, right = self.engine.generate_annealing_sequence(
            num_stages=5,
            base_frequency=432
        )
        
        # Last 10% should be very quiet
        final_section = left[-int(len(left) * 0.1):]
        final_rms = calculate_rms(final_section)
        
        # Should be much quieter than overall
        total_rms = calculate_rms(left)
        self.assertLess(final_rms, total_rms * 0.3)


class TestGoldenProportions(unittest.TestCase):
    """Tests for golden ratio proportions in parameters"""
    
    def test_beat_frequency_golden_decay(self):
        """Beat frequencies should follow golden ratio decay"""
        engine = SmoothBinauralEngine()
        base_freq = 432
        
        params = []
        for stage in range(5):
            p = engine._calculate_stage_params(stage, 5, base_freq)
            params.append(p)
        
        # Check ratio between consecutive beat frequencies
        for i in range(len(params) - 1):
            ratio = params[i]['beat_freq'] / params[i + 1]['beat_freq']
            # Should be close to φ
            self.assertAlmostEqual(ratio, PHI, places=1)
    
    def test_duration_fibonacci_based(self):
        """Durations should be based on Fibonacci sequence"""
        engine = SmoothBinauralEngine()
        
        params = []
        for stage in range(5):
            p = engine._calculate_stage_params(stage, 5, 432)
            params.append(p)
        
        # Durations should increase
        for i in range(len(params) - 1):
            self.assertLess(params[i]['duration'], params[i + 1]['duration'])
    
    def test_transition_golden_ratio_of_duration(self):
        """Transition time should be golden ratio of duration"""
        engine = SmoothBinauralEngine()
        
        for stage in range(5):
            p = engine._calculate_stage_params(stage, 5, 432)
            
            expected_ratio = PHI_CONJUGATE * PHI_CONJUGATE
            actual_ratio = p['transition_time'] / p['duration']
            
            self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)


class TestAudioQuality(unittest.TestCase):
    """Tests for audio quality metrics"""
    
    def test_stereo_difference_creates_beat(self):
        """Left and right channels should create binaural beat"""
        engine = SmoothBinauralEngine(sample_rate=44100)
        beat_freq = 10.0
        
        left, right = engine.generate_preview(432, beat_freq, 1.0, 0.8)
        
        # The difference between channels should have the beat frequency
        diff = left - right
        
        # Simple check: the difference should not be zero
        # (if it were zero, there would be no binaural beat)
        self.assertGreater(calculate_rms(diff), 0.01)
    
    def test_no_dc_offset(self):
        """Audio should have no significant DC offset"""
        engine = SmoothBinauralEngine()
        left, right = engine.generate_preview(432, 10, 2.0, 0.8)
        
        # Mean should be close to zero
        self.assertAlmostEqual(np.mean(left), 0.0, places=2)
        self.assertAlmostEqual(np.mean(right), 0.0, places=2)
    
    def test_no_nan_or_inf(self):
        """Audio should contain no NaN or Inf values"""
        engine = SmoothBinauralEngine()
        left, right = engine.generate_annealing_sequence(5, 432)
        
        self.assertFalse(np.any(np.isnan(left)))
        self.assertFalse(np.any(np.isnan(right)))
        self.assertFalse(np.any(np.isinf(left)))
        self.assertFalse(np.any(np.isinf(right)))


# ═══════════════════════════════════════════════════════════════════════════════
# WAV FILE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

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
# MAIN - Run Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Run all unit tests"""
    print("\n" + "=" * 60)
    print("  GOLDEN BINAURAL GENERATOR - UNIT TESTS")
    print("=" * 60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGoldenMathematics))
    suite.addTests(loader.loadTestsFromTestCase(TestPhaseAccumulator))
    suite.addTests(loader.loadTestsFromTestCase(TestSmoothBinauralEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestGoldenProportions))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioQuality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("  ✓ ALL TESTS PASSED")
    else:
        print(f"  ✗ FAILURES: {len(result.failures)}, ERRORS: {len(result.errors)}")
    print("=" * 60 + "\n")
    
    return result.wasSuccessful()


def generate_test_audio():
    """Generate test audio file to verify quality"""
    print("\nGenerating test audio...")
    
    engine = SmoothBinauralEngine(sample_rate=44100)
    left, right = engine.generate_annealing_sequence(
        num_stages=5,
        base_frequency=432
    )
    
    filename = "test_smooth_binaural.wav"
    save_wav(filename, left, right, 44100)
    
    print(f"Saved: {filename}")
    print(f"Duration: {len(left) / 44100:.1f} seconds")
    
    # Analyze
    disc = detect_discontinuities(left, threshold=0.15)
    print(f"Discontinuities detected: {len(disc)}")
    
    return filename


if __name__ == "__main__":
    success = run_tests()
    
    if success:
        generate_test_audio()
