"""
Spectral Sound Module
======================

Transform atomic spectral lines into audio.
Maps optical frequencies to audio range while preserving intensity ratios.

Features:
- Real spectral data (Hydrogen, Helium, Oxygen, etc.)
- Linear frequency scaling: optical → audio
- Relative intensities preserved
- Phase modes: incoherent, coherent, golden, fibonacci
- Element library with emission lines
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498949
C = 299792458  # Speed of light m/s


class PhaseMode(Enum):
    """Phase relationship modes"""
    INCOHERENT = "incoherent"  # Random phases (realistic)
    COHERENT = "coherent"      # All phases = 0
    GOLDEN = "golden"          # Golden angle spacing
    FIBONACCI = "fibonacci"    # Fibonacci-based phases


@dataclass
class SpectralLine:
    """Atomic spectral line"""
    name: str
    wavelength_nm: float
    intensity_relative: float
    frequency_optical: float = 0.0
    
    def __post_init__(self):
        # Calculate optical frequency from wavelength
        self.frequency_optical = C / (self.wavelength_nm * 1e-9)


# Spectral databases
HYDROGEN_BALMER = [
    SpectralLine("H-α", 656.281, 1.0),
    SpectralLine("H-β", 486.135, 0.44),
    SpectralLine("H-γ", 434.047, 0.17),
    SpectralLine("H-δ", 410.174, 0.08),
]

OXYGEN_ATMOSPHERIC = [
    SpectralLine("O-760", 760.0, 1.0),
    SpectralLine("O-630", 630.0, 0.8),
    SpectralLine("O-557", 557.7, 0.6),
]

HELIUM_VISIBLE = [
    SpectralLine("He-D3", 587.562, 1.0),
    SpectralLine("He-447", 447.1, 0.7),
    SpectralLine("He-501", 501.6, 0.6),
]

ELEMENTS = {
    "hydrogen": HYDROGEN_BALMER,
    "oxygen": OXYGEN_ATMOSPHERIC,
    "helium": HELIUM_VISIBLE,
}


class SpectralSoundGenerator:
    """
    Generate audio from atomic spectral lines.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Parameters
        self.element = "hydrogen"
        self.phase_mode = PhaseMode.GOLDEN
        self.amplitude = 0.5
        self.frequency_multiplier = 1.0  # Scale all frequencies
        
        # State
        self.phases = {}
        self.started = False
    
    def set_parameters(self,
                      element: Optional[str] = None,
                      phase_mode: Optional[PhaseMode] = None,
                      amplitude: Optional[float] = None,
                      freq_mult: Optional[float] = None):
        """Update parameters"""
        if element is not None and element in ELEMENTS:
            self.element = element
        if phase_mode is not None:
            self.phase_mode = phase_mode
        if amplitude is not None:
            self.amplitude = np.clip(amplitude, 0.0, 1.0)
        if freq_mult is not None:
            self.frequency_multiplier = freq_mult
    
    def start(self):
        """Start generation"""
        self.started = True
        self.phases = {}
    
    def stop(self):
        """Stop generation"""
        self.started = False
    
    def _optical_to_audio(self, freq_optical: float) -> float:
        """
        Scale optical frequency to audio range.
        
        Optical: ~400-700 THz
        Audio: 20-20000 Hz
        
        Linear scaling: f_audio = (f_optical - min) / (max - min) * (20000 - 20) + 20
        """
        # Visible spectrum range
        min_optical = C / (700e-9)  # Red ~430 THz
        max_optical = C / (400e-9)  # Violet ~750 THz
        
        # Linear map to audio range
        normalized = (freq_optical - min_optical) / (max_optical - min_optical)
        audio_freq = normalized * (20000 - 20) + 20
        
        return np.clip(audio_freq, 20, 20000) * self.frequency_multiplier
    
    def _get_phases(self, lines: List[SpectralLine]) -> np.ndarray:
        """Get phases for spectral lines based on mode"""
        n = len(lines)
        
        if self.phase_mode == PhaseMode.INCOHERENT:
            return np.random.uniform(0, 2 * np.pi, n)
        
        elif self.phase_mode == PhaseMode.COHERENT:
            return np.zeros(n)
        
        elif self.phase_mode == PhaseMode.GOLDEN:
            golden_angle = np.radians(137.5077640500378546)
            return np.array([i * golden_angle for i in range(n)])
        
        elif self.phase_mode == PhaseMode.FIBONACCI:
            fib = [1, 1, 2, 3, 5, 8, 13, 21, 34]
            return np.array([fib[i % len(fib)] * np.pi / 8 for i in range(n)])
    
    def generate_frame(self, num_frames: int) -> np.ndarray:
        """
        Generate audio frame.
        
        Returns:
            np.ndarray of shape (num_frames, 2) - stereo
        """
        if not self.started:
            return np.zeros((num_frames, 2), dtype=np.float32)
        
        t = np.arange(num_frames) / self.sample_rate
        dt = num_frames / self.sample_rate
        
        # Get spectral lines for current element
        lines = ELEMENTS[self.element]
        
        # Initialize phases if needed
        if not self.phases:
            phase_offsets = self._get_phases(lines)
            for i, line in enumerate(lines):
                self.phases[line.name] = phase_offsets[i]
        
        # Generate audio
        output = np.zeros((num_frames, 2))
        
        for i, line in enumerate(lines):
            # Convert to audio frequency
            audio_freq = self._optical_to_audio(line.frequency_optical)
            
            # Amplitude from relative intensity
            amp = line.intensity_relative * self.amplitude / len(lines)
            
            # Generate sine wave
            phase_array = self.phases[line.name] + 2 * np.pi * audio_freq * t
            wave = np.sin(phase_array) * amp
            
            # Update phase accumulator
            self.phases[line.name] = (self.phases[line.name] + 
                                     2 * np.pi * audio_freq * dt) % (2 * np.pi)
            
            # Stereo panning based on wavelength (red = left, blue = right)
            # Shorter wavelength = more right
            pan = (line.wavelength_nm - 400) / (700 - 400)  # 0 to 1
            left_gain = np.cos(pan * np.pi / 2)
            right_gain = np.sin(pan * np.pi / 2)
            
            output[:, 0] += wave * left_gain
            output[:, 1] += wave * right_gain
        
        return output.astype(np.float32)


# Example usage
if __name__ == "__main__":
    print("⚛️ Spectral Sound Generator Test\n")
    
    gen = SpectralSoundGenerator(sample_rate=48000)
    gen.set_parameters(
        element="hydrogen",
        phase_mode=PhaseMode.GOLDEN,
        amplitude=0.5
    )
    
    print(f"✓ Element: {gen.element}")
    print(f"✓ Lines: {len(ELEMENTS[gen.element])}")
    print(f"✓ Phase mode: {gen.phase_mode.value}")
    
    gen.start()
    
    # Generate 1 second
    frames = []
    for i in range(int(48000 / 2048)):
        frame = gen.generate_frame(2048)
        frames.append(frame)
    
    audio = np.vstack(frames)
    print(f"\n✓ Generated {audio.shape} audio")
