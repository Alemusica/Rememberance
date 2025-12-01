"""
Golden Ratio Binaural Beat Generator with Phase Cancellation Annealing
=======================================================================

Generates binaural beats using:
- Golden ratio (φ = 1.618033988749895) based frequencies
- Prime number sequences
- Golden spiral transitions (not linear)
- Perfect phase cancellation annealing to silence

All parameters are in divine coherent golden relationships.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import struct

# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED CONSTANTS (from golden_constants module)
# ══════════════════════════════════════════════════════════════════════════════

from golden_constants import (
    PHI, PHI_CONJUGATE, SQRT_5,
    PRIMES, FIBONACCI,
    golden_spiral_interpolation, golden_transition,
    apply_golden_envelope,
)


@dataclass
class GoldenParameters:
    """All parameters in golden ratio relationships"""
    base_frequency: float  # Hz - Base carrier frequency
    beat_frequency: float  # Hz - Binaural beat frequency
    duration: float        # seconds
    transition_time: float # seconds - golden ratio based
    amplitude: float       # 0.0 to 1.0
    phase_offset: float    # radians
    
    @classmethod
    def from_golden_index(cls, index: int, base_freq: float = 432.0):
        """
        Generate parameters where everything is in golden ratio relationship
        432 Hz is used as base (sacred frequency)
        """
        # Frequencies in golden ratio relationships
        beat_freq = base_freq / (PHI ** (index + 3))  # Descending golden frequencies
        
        # Duration based on Fibonacci (golden convergent)
        duration = FIBONACCI[min(index + 5, len(FIBONACCI)-1)] * PHI_CONJUGATE
        
        # Transition time in golden ratio to duration
        transition = duration * PHI_CONJUGATE * PHI_CONJUGATE
        
        # Amplitude follows golden decay
        amplitude = PHI_CONJUGATE ** (index * 0.5)
        
        # Phase offset in golden fractions of 2π
        phase = (2 * np.pi) * (PHI_CONJUGATE ** index)
        
        return cls(
            base_frequency=base_freq,
            beat_frequency=beat_freq,
            duration=duration,
            transition_time=transition,
            amplitude=amplitude,
            phase_offset=phase
        )


# NOTE: golden_spiral_interpolation and golden_transition are now imported from golden_constants
# The following legacy functions are kept for reference but use the centralized versions

def golden_spiral_transition(start_val: float, end_val: float, t: float) -> float:
    """
    Transition between two values using golden spiral interpolation.
    Now uses the centralized golden_transition function.
    """
    return golden_transition(start_val, end_val, t)


class BinauralGenerator:
    """
    High-precision binaural beat generator with golden ratio mathematics
    """
    
    def __init__(self, sample_rate: int = 96000):
        """
        96kHz for maximum precision (golden ratio of standard 59259 Hz ≈ 96000)
        """
        self.sample_rate = sample_rate
        self.dtype = np.float64  # Maximum precision
        
    def generate_tone(self, frequency: float, duration: float, 
                      amplitude: float, phase: float = 0.0) -> np.ndarray:
        """
        Generate a pure sine tone with maximum precision
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        
        # High precision sine generation
        tone = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        return tone
    
    def generate_binaural_beat(self, params: GoldenParameters) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stereo binaural beat
        Left ear: base frequency
        Right ear: base frequency + beat frequency
        
        The brain perceives the difference as the binaural beat
        """
        left = self.generate_tone(
            params.base_frequency,
            params.duration,
            params.amplitude,
            params.phase_offset
        )
        
        right = self.generate_tone(
            params.base_frequency + params.beat_frequency,
            params.duration,
            params.amplitude,
            params.phase_offset + np.pi * PHI_CONJUGATE  # Golden phase relationship
        )
        
        return left, right
    
    def apply_golden_envelope(self, audio: np.ndarray, 
                               attack_ratio: float = None,
                               decay_ratio: float = None) -> np.ndarray:
        """
        Apply amplitude envelope using golden ratio proportions
        """
        if attack_ratio is None:
            attack_ratio = PHI_CONJUGATE * PHI_CONJUGATE  # ~0.382 of duration
        if decay_ratio is None:
            decay_ratio = PHI_CONJUGATE  # ~0.618 of duration
            
        num_samples = len(audio)
        envelope = np.ones(num_samples, dtype=self.dtype)
        
        # Attack phase
        attack_samples = int(num_samples * attack_ratio * 0.1)
        for i in range(attack_samples):
            t = i / attack_samples
            envelope[i] = golden_spiral_interpolation(t)
        
        # Decay phase  
        decay_samples = int(num_samples * decay_ratio * 0.1)
        for i in range(decay_samples):
            t = i / decay_samples
            envelope[-(i+1)] = golden_spiral_interpolation(t)
            
        return audio * envelope


class PhaseAnnihilator:
    """
    Generates the journey toward perfect phase cancellation (silence)
    The "annealing" process using golden ratio mathematics
    """
    
    def __init__(self, sample_rate: int = 96000):
        self.sample_rate = sample_rate
        self.generator = BinauralGenerator(sample_rate)
        
    def generate_annealing_sequence(self, 
                                     num_stages: int = 8,
                                     base_frequency: float = 432.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete annealing sequence from full binaural to silence
        
        Each stage:
        - Beat frequency decreases by golden ratio
        - Amplitude decreases by golden ratio
        - Transition uses golden spiral (not linear)
        - Duration follows Fibonacci sequence
        - Phase approaches cancellation
        
        Returns stereo audio (left, right channels)
        """
        left_channel = []
        right_channel = []
        
        print("=" * 60)
        print("GOLDEN RATIO BINAURAL ANNEALING SEQUENCE")
        print("=" * 60)
        print(f"φ (Golden Ratio) = {PHI}")
        print(f"φ conjugate = {PHI_CONJUGATE}")
        print(f"Base Frequency = {base_frequency} Hz")
        print(f"Sample Rate = {self.sample_rate} Hz")
        print("=" * 60)
        
        total_duration = 0
        
        for stage in range(num_stages):
            params = GoldenParameters.from_golden_index(stage, base_frequency)
            
            # As we approach the end, phase approaches π (cancellation)
            annealing_factor = stage / (num_stages - 1) if num_stages > 1 else 0
            
            # Phase shift approaches π using golden spiral
            phase_toward_cancel = np.pi * golden_spiral_interpolation(annealing_factor)
            params.phase_offset = phase_toward_cancel
            
            # Amplitude decreases toward silence
            params.amplitude *= (1 - golden_spiral_interpolation(annealing_factor) * 0.95)
            
            print(f"\nStage {stage + 1}/{num_stages}:")
            print(f"  Beat Frequency: {params.beat_frequency:.4f} Hz")
            print(f"  Duration: {params.duration:.2f} s")
            print(f"  Amplitude: {params.amplitude:.4f}")
            print(f"  Phase: {params.phase_offset:.4f} rad ({np.degrees(params.phase_offset):.2f}°)")
            print(f"  Transition: {params.transition_time:.2f} s")
            
            # Generate binaural beat for this stage
            left, right = self.generator.generate_binaural_beat(params)
            
            # Apply golden envelope
            left = self.generator.apply_golden_envelope(left)
            right = self.generator.apply_golden_envelope(right)
            
            left_channel.append(left)
            right_channel.append(right)
            total_duration += params.duration
            
            # Generate transition to next stage (if not last)
            if stage < num_stages - 1:
                next_params = GoldenParameters.from_golden_index(stage + 1, base_frequency)
                transition = self._generate_golden_transition(
                    params, next_params, params.transition_time
                )
                left_channel.append(transition[0])
                right_channel.append(transition[1])
                total_duration += params.transition_time
        
        # Final stage: Pure phase cancellation (silence approach)
        silence_duration = FIBONACCI[8] * PHI_CONJUGATE  # Golden silence duration
        final_left, final_right = self._generate_phase_cancellation(
            base_frequency, silence_duration
        )
        left_channel.append(final_left)
        right_channel.append(final_right)
        total_duration += silence_duration
        
        print(f"\n{'=' * 60}")
        print(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Final State: Phase Cancellation → Silence")
        print("=" * 60)
        
        # Concatenate all segments
        left_audio = np.concatenate(left_channel)
        right_audio = np.concatenate(right_channel)
        
        return left_audio, right_audio
    
    def _generate_golden_transition(self, 
                                     from_params: GoldenParameters,
                                     to_params: GoldenParameters,
                                     duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate transition between two stages using golden spiral interpolation
        NOT linear - follows the divine golden curve
        """
        num_samples = int(duration * self.sample_rate)
        left = np.zeros(num_samples, dtype=np.float64)
        right = np.zeros(num_samples, dtype=np.float64)
        
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        for i in range(num_samples):
            # Golden spiral interpolation factor
            factor = golden_spiral_interpolation(i / num_samples)
            
            # Interpolate all parameters
            freq = golden_spiral_transition(from_params.base_frequency, 
                                           to_params.base_frequency, factor)
            beat = golden_spiral_transition(from_params.beat_frequency,
                                           to_params.beat_frequency, factor)
            amp = golden_spiral_transition(from_params.amplitude,
                                          to_params.amplitude, factor)
            phase = golden_spiral_transition(from_params.phase_offset,
                                            to_params.phase_offset, factor)
            
            # Generate sample
            left[i] = amp * np.sin(2 * np.pi * freq * t[i] + phase)
            right[i] = amp * np.sin(2 * np.pi * (freq + beat) * t[i] + phase + np.pi * PHI_CONJUGATE)
        
        return left, right
    
    def _generate_phase_cancellation(self, 
                                      frequency: float,
                                      duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the final phase cancellation approach
        Two identical waves with phase approaching π → silence
        """
        num_samples = int(duration * self.sample_rate)
        left = np.zeros(num_samples, dtype=np.float64)
        right = np.zeros(num_samples, dtype=np.float64)
        
        t = np.linspace(0, duration, num_samples, dtype=np.float64)
        
        # Amplitude decays using golden ratio
        base_amplitude = PHI_CONJUGATE ** 4  # Start quiet
        
        for i in range(num_samples):
            progress = i / num_samples
            
            # Phase approaches π using golden spiral
            phase_diff = np.pi * golden_spiral_interpolation(progress)
            
            # Amplitude approaches zero using golden decay
            amp = base_amplitude * (1 - golden_spiral_interpolation(progress))
            
            # Generate perfectly opposing waves
            left[i] = amp * np.sin(2 * np.pi * frequency * t[i])
            right[i] = amp * np.sin(2 * np.pi * frequency * t[i] + phase_diff)
        
        # The sum approaches zero → silence
        return left, right


def save_wav(filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int = 96000):
    """
    Save stereo audio as WAV file (pure Python, no dependencies)
    """
    # Normalize to prevent clipping
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left = left / max_val * 0.95
        right = right / max_val * 0.95
    
    # Convert to 16-bit integers
    left_int = (left * 32767).astype(np.int16)
    right_int = (right * 32767).astype(np.int16)
    
    # Interleave stereo channels
    stereo = np.empty((len(left) + len(right),), dtype=np.int16)
    stereo[0::2] = left_int
    stereo[1::2] = right_int
    
    # Write WAV file
    with open(filename, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(stereo) * 2))  # File size - 8
        f.write(b'WAVE')
        
        # Format chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Chunk size
        f.write(struct.pack('<H', 1))   # Audio format (PCM)
        f.write(struct.pack('<H', 2))   # Number of channels
        f.write(struct.pack('<I', sample_rate))  # Sample rate
        f.write(struct.pack('<I', sample_rate * 4))  # Byte rate
        f.write(struct.pack('<H', 4))   # Block align
        f.write(struct.pack('<H', 16))  # Bits per sample
        
        # Data chunk
        f.write(b'data')
        f.write(struct.pack('<I', len(stereo) * 2))
        f.write(stereo.tobytes())
    
    print(f"\nSaved: {filename}")


def main():
    """
    Generate the complete golden ratio binaural annealing sequence
    """
    print("\n" + "★" * 60)
    print("  DIVINE GOLDEN RATIO BINAURAL BEAT GENERATOR")
    print("  Phase Cancellation Annealing to Pure Silence")
    print("★" * 60 + "\n")
    
    # Sacred frequency 432 Hz (in golden relationship with universe)
    BASE_FREQUENCY = 432.0
    
    # Number of annealing stages (Fibonacci number for coherence)
    NUM_STAGES = 8
    
    # High precision sample rate
    SAMPLE_RATE = 96000
    
    # Create the phase annihilator
    annihilator = PhaseAnnihilator(sample_rate=SAMPLE_RATE)
    
    # Generate the complete annealing sequence
    left, right = annihilator.generate_annealing_sequence(
        num_stages=NUM_STAGES,
        base_frequency=BASE_FREQUENCY
    )
    
    # Save as WAV file
    output_file = "golden_binaural_annealing.wav"
    save_wav(output_file, left, right, SAMPLE_RATE)
    
    print("\n" + "★" * 60)
    print("  GENERATION COMPLETE")
    print(f"  Output: {output_file}")
    print(f"  Duration: {len(left) / SAMPLE_RATE:.2f} seconds")
    print("  ")
    print("  Use with stereo headphones for full binaural effect")
    print("  Journey from sound → through golden transitions → to silence")
    print("★" * 60 + "\n")


if __name__ == "__main__":
    main()
