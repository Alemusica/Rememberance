"""
Harmonic Tree Module
====================

Generate fundamental + harmonics with Fibonacci ratios and golden angle phases.
Therapeutic growth mode with progressive harmonic emergence.

Features:
- Fibonacci harmonic series (f, 2f, 3f, 5f, 8f, 13f...)
- Golden angle phase relationships (137.5°)
- φ-based amplitude decay
- Progressive growth with golden-timed emergence
- Stereo panning for spatial depth
"""

import numpy as np
import time
from typing import List, Optional
from enum import Enum


PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498949
GOLDEN_ANGLE_RAD = np.radians(137.5077640500378546)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


class HarmonicMode(Enum):
    """Harmonic series type"""
    FIBONACCI = "fibonacci"  # 1f, 2f, 3f, 5f, 8f, 13f...
    INTEGER = "integer"      # 1f, 2f, 3f, 4f, 5f, 6f...
    PHI = "phi"             # 1f, φf, φ²f, φ³f...


class AmplitudeDecay(Enum):
    """Amplitude decay curve"""
    PHI = "phi"          # a_n = φ^(-n)
    SQRT = "sqrt"        # a_n = 1/√n
    LINEAR = "linear"    # a_n = 1/n


class HarmonicTreeGenerator:
    """
    Generate harmonic tree with progressive growth.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Parameters
        self.fundamental = 432.0
        self.num_harmonics = 5
        self.harmonic_mode = HarmonicMode.FIBONACCI
        self.amplitude_decay = AmplitudeDecay.PHI
        self.master_amplitude = 0.7
        self.stereo_spread = 0.5
        
        # Growth mode
        self.growth_enabled = True
        self.growth_duration = 60.0  # seconds
        self.phase_evolution = True
        
        # State
        self.phases = np.zeros(13)  # Phase accumulators
        self.growth_levels = np.zeros(13)  # 0-1 for each harmonic
        self.time_elapsed = 0.0
        self.started = False
    
    def set_parameters(self, 
                      fundamental: Optional[float] = None,
                      num_harmonics: Optional[int] = None,
                      harmonic_mode: Optional[HarmonicMode] = None,
                      amplitude_decay: Optional[AmplitudeDecay] = None,
                      amplitude: Optional[float] = None,
                      stereo_spread: Optional[float] = None):
        """Update parameters"""
        if fundamental is not None:
            self.fundamental = fundamental
        if num_harmonics is not None:
            self.num_harmonics = min(num_harmonics, 12)
        if harmonic_mode is not None:
            self.harmonic_mode = harmonic_mode
        if amplitude_decay is not None:
            self.amplitude_decay = amplitude_decay
        if amplitude is not None:
            self.master_amplitude = np.clip(amplitude, 0.0, 1.0)
        if stereo_spread is not None:
            self.stereo_spread = np.clip(stereo_spread, 0.0, 1.0)
    
    def set_growth(self, enabled: bool, duration: float = 60.0, phase_evolution: bool = True):
        """Configure growth mode"""
        self.growth_enabled = enabled
        self.growth_duration = duration
        self.phase_evolution = phase_evolution
    
    def start(self):
        """Start generation"""
        self.started = True
        self.time_elapsed = 0.0
        if not self.growth_enabled:
            # Instant mode - all harmonics at full level
            self.growth_levels[:] = 1.0
    
    def stop(self):
        """Stop generation"""
        self.started = False
    
    def _get_harmonic_frequencies(self) -> List[float]:
        """Get harmonic frequencies based on mode"""
        freqs = []
        
        if self.harmonic_mode == HarmonicMode.FIBONACCI:
            for i in range(self.num_harmonics):
                ratio = FIBONACCI[i]
                freqs.append(self.fundamental * ratio)
        
        elif self.harmonic_mode == HarmonicMode.INTEGER:
            for i in range(self.num_harmonics):
                freqs.append(self.fundamental * (i + 1))
        
        elif self.harmonic_mode == HarmonicMode.PHI:
            for i in range(self.num_harmonics):
                ratio = PHI ** i
                freqs.append(self.fundamental * ratio)
        
        return freqs
    
    def _get_harmonic_amplitudes(self) -> np.ndarray:
        """Get harmonic amplitudes with decay"""
        amps = np.zeros(self.num_harmonics)
        
        for i in range(self.num_harmonics):
            if self.amplitude_decay == AmplitudeDecay.PHI:
                amps[i] = PHI ** (-i)
            elif self.amplitude_decay == AmplitudeDecay.SQRT:
                amps[i] = 1.0 / np.sqrt(i + 1)
            elif self.amplitude_decay == AmplitudeDecay.LINEAR:
                amps[i] = 1.0 / (i + 1)
        
        # Normalize
        amps /= np.max(amps)
        return amps
    
    def _update_growth(self, dt: float):
        """Update growth levels"""
        if not self.growth_enabled or self.time_elapsed >= self.growth_duration:
            self.growth_levels[:self.num_harmonics] = 1.0
            return
        
        # Each harmonic emerges at φ^(-n) × duration
        for i in range(self.num_harmonics):
            emergence_time = (PHI ** (-i)) * self.growth_duration
            
            if self.time_elapsed >= emergence_time:
                # Fade in over 2 seconds
                fade_duration = 2.0
                fade_elapsed = self.time_elapsed - emergence_time
                self.growth_levels[i] = min(1.0, fade_elapsed / fade_duration)
    
    def generate_frame(self, num_frames: int) -> np.ndarray:
        """
        Generate audio frame.
        
        Returns:
            np.ndarray of shape (num_frames, 2) - stereo
        """
        if not self.started:
            return np.zeros((num_frames, 2), dtype=np.float32)
        
        dt = num_frames / self.sample_rate
        t = np.arange(num_frames) / self.sample_rate
        
        # Update growth
        self._update_growth(dt)
        
        # Get frequencies and amplitudes
        freqs = self._get_harmonic_frequencies()
        base_amps = self._get_harmonic_amplitudes()
        
        # Generate harmonics
        output = np.zeros((num_frames, 2))
        
        for i in range(self.num_harmonics):
            freq = freqs[i]
            amp = base_amps[i] * self.growth_levels[i] * self.master_amplitude
            
            # Phase evolution during growth
            if self.phase_evolution and self.growth_enabled:
                phase_offset = (i * GOLDEN_ANGLE_RAD) * (self.time_elapsed / self.growth_duration)
            else:
                phase_offset = i * GOLDEN_ANGLE_RAD
            
            # Generate harmonic
            phase_array = self.phases[i] + 2 * np.pi * freq * t + phase_offset
            harmonic = np.sin(phase_array) * amp
            
            # Update phase accumulator
            self.phases[i] = (self.phases[i] + 2 * np.pi * freq * dt) % (2 * np.pi)
            
            # Stereo panning using golden angle
            pan_angle = (i * GOLDEN_ANGLE_RAD) * self.stereo_spread
            left_gain = np.cos(pan_angle)
            right_gain = np.sin(pan_angle)
            
            output[:, 0] += harmonic * left_gain
            output[:, 1] += harmonic * right_gain
        
        # Update time
        self.time_elapsed += dt
        
        return output.astype(np.float32)


# Example usage
if __name__ == "__main__":
    print("🌳 Harmonic Tree Generator Test\n")
    
    gen = HarmonicTreeGenerator(sample_rate=48000)
    gen.set_parameters(
        fundamental=432,
        num_harmonics=5,
        harmonic_mode=HarmonicMode.FIBONACCI,
        amplitude=0.5
    )
    gen.set_growth(enabled=True, duration=10.0)
    
    print(f"✓ Fundamental: {gen.fundamental} Hz")
    print(f"✓ Harmonics: {gen.num_harmonics}")
    print(f"✓ Growth duration: {gen.growth_duration}s")
    
    gen.start()
    
    # Generate 1 second
    frames = []
    for i in range(int(48000 / 2048)):
        frame = gen.generate_frame(2048)
        frames.append(frame)
    
    audio = np.vstack(frames)
    print(f"\n✓ Generated {audio.shape} audio")
