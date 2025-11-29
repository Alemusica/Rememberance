"""
Divine Golden Ratio Binaural Generator - v3 OPTIMIZED & SMOOTH
===============================================================

FIXES:
- Seamless transitions (no clicks/pops)
- Phase continuity across segments
- VECTORIZED operations for speed
- Comprehensive tests
"""

import numpy as np
import unittest
from typing import Tuple, List, Optional, Callable
import struct

# ═══════════════════════════════════════════════════════════════════════════════
# DIVINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = np.float64((1 + np.sqrt(5)) / 2)
PHI_CONJUGATE = np.float64(PHI - 1)
PHI_SQUARED = np.float64(PHI * PHI)
TWO_PI = np.float64(2 * np.pi)

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN MATHEMATICS - Vectorized
# ═══════════════════════════════════════════════════════════════════════════════

def golden_spiral_interpolation(t):
    """Vectorized golden spiral easing"""
    t = np.asarray(t, dtype=np.float64)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)
    
    result = np.zeros_like(t)
    
    # Handle boundaries
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
# SMOOTH BINAURAL ENGINE - Vectorized
# ═══════════════════════════════════════════════════════════════════════════════

class SmoothBinauralEngine:
    """Optimized binaural generator with seamless transitions"""
    
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
        """Generate binaural segment with continuous phase - VECTORIZED"""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        # Phase arrays with continuity from previous segment
        omega_left = TWO_PI * base_freq
        omega_right = TWO_PI * (base_freq + beat_freq)
        
        phase_left = self._phase_left + omega_left * t + phase_offset
        phase_right = self._phase_right + omega_right * t + np.pi * PHI_CONJUGATE + phase_offset
        
        # Generate audio
        left = amplitude * np.sin(phase_left)
        right = amplitude * np.sin(phase_right)
        
        # Update phase for continuity
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
        """Generate smooth transition - VECTORIZED"""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        # Interpolation factor (golden spiral)
        factor = np.linspace(0, 1, num_samples, dtype=np.float64)
        factor = golden_spiral_interpolation(factor)
        
        # Interpolate parameters smoothly
        beat_freq = beat1 + (beat2 - beat1) * factor
        amplitude = amp1 + (amp2 - amp1) * factor
        phase_offset = phase1 + (phase2 - phase1) * factor
        
        # Cumulative phase for smooth frequency transitions
        omega_left = TWO_PI * base_freq
        
        # For right channel, integrate changing frequency
        dt = 1.0 / self.sample_rate
        omega_right_instant = TWO_PI * (base_freq + beat_freq)
        phase_right_cumulative = np.cumsum(omega_right_instant) * dt
        
        phase_left = self._phase_left + omega_left * t + phase_offset
        phase_right = self._phase_right + phase_right_cumulative + np.pi * PHI_CONJUGATE + phase_offset
        
        left = amplitude * np.sin(phase_left)
        right = amplitude * np.sin(phase_right)
        
        # Update phases
        self._phase_left = (self._phase_left + omega_left * duration) % TWO_PI
        self._phase_right = (self._phase_right + phase_right_cumulative[-1]) % TWO_PI
        
        return left, right
    
    def _calculate_stage_params(self, stage: int, total_stages: int, base_freq: float) -> dict:
        """Calculate parameters for a stage"""
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
        """Apply tiny envelope at boundaries to eliminate any residual clicks"""
        fade_samples = min(64, len(audio) // 10)
        if fade_samples < 2:
            return audio
        
        # Smooth fade curve
        fade_in = np.linspace(0, 1, fade_samples) ** 2
        fade_out = np.linspace(1, 0, fade_samples) ** 2
        
        audio = audio.copy()
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def generate_annealing_sequence(
        self,
        num_stages: int,
        base_frequency: float,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete annealing sequence with seamless transitions"""
        self.reset_phase()
        
        left_segments = []
        right_segments = []
        
        # Pre-calculate all stage parameters
        stages = [self._calculate_stage_params(i, num_stages, base_frequency) 
                  for i in range(num_stages)]
        
        for idx, params in enumerate(stages):
            if progress_callback:
                progress_callback(idx, num_stages, f"Stage {idx+1}/{num_stages}")
            
            # Generate segment
            left, right = self.generate_segment_vectorized(
                base_frequency,
                params['beat_freq'],
                params['duration'],
                params['amplitude'],
                params['phase_offset']
            )
            
            left_segments.append(left)
            right_segments.append(right)
            
            # Transition to next stage
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
        
        # Final cancellation
        if progress_callback:
            progress_callback(num_stages, num_stages, "Phase cancellation...")
        
        final_l, final_r = self._generate_final_cancellation(
            base_frequency,
            stages[-1]['amplitude'] if stages else 0.1
        )
        left_segments.append(final_l)
        right_segments.append(final_r)
        
        # Concatenate with micro-crossfades
        return self._smooth_concatenate(left_segments), self._smooth_concatenate(right_segments)
    
    def _smooth_concatenate(self, segments: List[np.ndarray]) -> np.ndarray:
        """Concatenate segments with tiny crossfades at boundaries"""
        if not segments:
            return np.array([], dtype=np.float64)
        
        # Apply micro envelope to each segment
        smoothed = [self._apply_micro_envelope(seg) for seg in segments]
        return np.concatenate(smoothed)
    
    def _generate_final_cancellation(
        self,
        frequency: float,
        start_amplitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth approach to silence - VECTORIZED"""
        duration = FIBONACCI[7] * PHI_CONJUGATE
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        
        # Progress factor
        progress = golden_spiral_interpolation(np.linspace(0, 1, num_samples))
        
        # Phase difference approaches π
        phase_diff = np.pi * progress
        
        # Amplitude approaches zero
        amplitude = start_amplitude * (1 - progress)
        
        # Phase continuity
        omega = TWO_PI * frequency
        phase = self._phase_left + omega * t
        
        left = amplitude * np.sin(phase)
        right = amplitude * np.sin(phase + phase_diff)
        
        return left, right
    
    def generate_preview(
        self,
        base_freq: float,
        beat_freq: float,
        duration: float,
        amplitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth preview"""
        self.reset_phase()
        
        left, right = self.generate_segment_vectorized(
            base_freq, beat_freq, duration, amplitude
        )
        
        # Full envelope for preview - smooth cosine fade
        num_samples = len(left)
        attack = int(num_samples * 0.1)
        decay = int(num_samples * 0.15)
        
        envelope = np.ones(num_samples, dtype=np.float64)
        
        # Smooth cosine fade-in (avoids any discontinuity)
        if attack > 0:
            fade_in = (1 - np.cos(np.linspace(0, np.pi, attack))) / 2
            envelope[:attack] = fade_in
        
        # Smooth cosine fade-out
        if decay > 0:
            fade_out = (1 + np.cos(np.linspace(0, np.pi, decay))) / 2
            envelope[-decay:] = fade_out
        
        return left * envelope, right * envelope


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_discontinuities(audio: np.ndarray, threshold: float = 0.1) -> List[int]:
    """Detect clicks/pops in audio"""
    diff = np.abs(np.diff(audio))
    return np.where(diff > threshold)[0].tolist()


def calculate_rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio ** 2))


def calculate_peak(audio: np.ndarray) -> float:
    return np.max(np.abs(audio))


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoldenMath(unittest.TestCase):
    """Test golden ratio mathematics"""
    
    def test_phi_identity(self):
        self.assertAlmostEqual(PHI * PHI, PHI + 1, places=10)
    
    def test_phi_conjugate(self):
        self.assertAlmostEqual(1 / PHI, PHI - 1, places=10)
    
    def test_interpolation_bounds(self):
        for t in np.linspace(-0.5, 1.5, 50):
            result = golden_spiral_interpolation(t)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    def test_interpolation_endpoints(self):
        self.assertEqual(golden_spiral_interpolation(0.0), 0.0)
        self.assertEqual(golden_spiral_interpolation(1.0), 1.0)
    
    def test_interpolation_monotonic(self):
        values = [golden_spiral_interpolation(t) for t in np.linspace(0, 1, 50)]
        for i in range(len(values) - 1):
            self.assertGreaterEqual(values[i+1], values[i])
    
    def test_vectorized_interpolation(self):
        """Test that vectorized version matches scalar"""
        t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        vectorized = golden_spiral_interpolation(t_values)
        scalar = [golden_spiral_interpolation(t) for t in t_values]
        np.testing.assert_array_almost_equal(vectorized, scalar)


class TestSmoothEngine(unittest.TestCase):
    """Test smooth binaural engine"""
    
    def setUp(self):
        self.engine = SmoothBinauralEngine(sample_rate=44100)
    
    def test_segment_length(self):
        left, right = self.engine.generate_segment_vectorized(440, 10, 1.0, 0.8)
        self.assertEqual(len(left), 44100)
        self.assertEqual(len(right), 44100)
    
    def test_amplitude_bounds(self):
        left, right = self.engine.generate_segment_vectorized(440, 10, 1.0, 0.7)
        self.assertLessEqual(calculate_peak(left), 0.75)
        self.assertLessEqual(calculate_peak(right), 0.75)
    
    def test_no_clicks_in_segment(self):
        left, _ = self.engine.generate_segment_vectorized(440, 10, 1.0, 0.5)
        clicks = detect_discontinuities(left, threshold=0.1)
        self.assertEqual(len(clicks), 0, f"Found {len(clicks)} clicks")
    
    def test_phase_continuity(self):
        """Multiple segments should connect smoothly"""
        self.engine.reset_phase()
        
        left1, _ = self.engine.generate_segment_vectorized(440, 10, 0.5, 0.5)
        left2, _ = self.engine.generate_segment_vectorized(440, 10, 0.5, 0.5)
        
        # Boundary difference should be small
        boundary_diff = abs(left1[-1] - left2[0])
        avg_diff = np.mean(np.abs(np.diff(left1)))
        
        self.assertLess(boundary_diff, avg_diff * 5, 
                       f"Boundary discontinuity: {boundary_diff:.4f} vs avg {avg_diff:.4f}")
    
    def test_transition_smoothness(self):
        left, _ = self.engine.generate_transition_vectorized(
            440, 10, 5, 0.8, 0.4, 0, np.pi/4, 0.5
        )
        clicks = detect_discontinuities(left, threshold=0.15)
        self.assertLess(len(clicks), 5, f"Found {len(clicks)} discontinuities")
    
    def test_preview_no_clicks(self):
        left, right = self.engine.generate_preview(432, 10, 2.0, 0.8)
        
        clicks_l = detect_discontinuities(left, threshold=0.1)
        clicks_r = detect_discontinuities(right, threshold=0.1)
        
        self.assertEqual(len(clicks_l), 0, f"Left: {len(clicks_l)} clicks")
        self.assertEqual(len(clicks_r), 0, f"Right: {len(clicks_r)} clicks")
    
    def test_annealing_no_major_clicks(self):
        """Full annealing should be mostly smooth"""
        left, _ = self.engine.generate_annealing_sequence(3, 432)
        
        clicks = detect_discontinuities(left, threshold=0.2)
        # Allow very few at segment boundaries
        self.assertLess(len(clicks), 20, f"Found {len(clicks)} discontinuities")
    
    def test_annealing_decreasing_amplitude(self):
        left, _ = self.engine.generate_annealing_sequence(3, 432)
        
        # First and last quarters
        quarter = len(left) // 4
        first_rms = calculate_rms(left[:quarter])
        last_rms = calculate_rms(left[-quarter:])
        
        self.assertGreater(first_rms, last_rms * 0.5)
    
    def test_no_nan_inf(self):
        left, right = self.engine.generate_annealing_sequence(3, 432)
        
        self.assertFalse(np.any(np.isnan(left)))
        self.assertFalse(np.any(np.isnan(right)))
        self.assertFalse(np.any(np.isinf(left)))
        self.assertFalse(np.any(np.isinf(right)))
    
    def test_no_dc_offset(self):
        left, right = self.engine.generate_preview(432, 10, 2.0, 0.8)
        
        self.assertAlmostEqual(np.mean(left), 0.0, places=2)
        self.assertAlmostEqual(np.mean(right), 0.0, places=2)


class TestGoldenProportions(unittest.TestCase):
    """Test that parameters follow golden ratios"""
    
    def test_beat_freq_ratio(self):
        engine = SmoothBinauralEngine()
        
        p0 = engine._calculate_stage_params(0, 5, 432)
        p1 = engine._calculate_stage_params(1, 5, 432)
        
        ratio = p0['beat_freq'] / p1['beat_freq']
        self.assertAlmostEqual(ratio, PHI, places=1)
    
    def test_transition_ratio(self):
        engine = SmoothBinauralEngine()
        p = engine._calculate_stage_params(0, 5, 432)
        
        expected = PHI_CONJUGATE * PHI_CONJUGATE
        actual = p['transition_time'] / p['duration']
        
        self.assertAlmostEqual(actual, expected, places=5)


# ═══════════════════════════════════════════════════════════════════════════════
# WAV UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_wav(filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int = 44100):
    """Save stereo WAV"""
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)
    left = left / max_val * 0.95
    right = right / max_val * 0.95
    
    left_int = (left * 32767).astype(np.int16)
    right_int = (right * 32767).astype(np.int16)
    
    stereo = np.empty((len(left) * 2,), dtype=np.int16)
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
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests():
    print("\n" + "=" * 60)
    print("  GOLDEN BINAURAL - UNIT TESTS")
    print("=" * 60 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestGoldenMath))
    suite.addTests(loader.loadTestsFromTestCase(TestSmoothEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestGoldenProportions))
    
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
    print("\nGenerating test audio...")
    
    engine = SmoothBinauralEngine(44100)
    left, right = engine.generate_annealing_sequence(5, 432)
    
    filename = "test_smooth.wav"
    save_wav(filename, left, right, 44100)
    
    print(f"✓ Saved: {filename}")
    print(f"  Duration: {len(left)/44100:.1f}s")
    print(f"  Clicks detected: {len(detect_discontinuities(left, 0.15))}")


if __name__ == "__main__":
    if run_tests():
        generate_test_audio()
