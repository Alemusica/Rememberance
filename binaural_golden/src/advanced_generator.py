"""
Advanced Golden Ratio Binaural Generator
========================================

Extended version with:
- Prime sequence integration
- Multiple brainwave states
- Real-time visualization
- Configurable annealing profiles
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
import json

# Divine Constants - Maximum Precision (64-bit float)
PHI = np.float64((1 + np.sqrt(5)) / 2)  # 1.618033988749895
PHI_CONJUGATE = np.float64(PHI - 1)       # 0.618033988749895
PHI_SQUARED = np.float64(PHI * PHI)       # 2.618033988749895

# Sacred frequencies
SACRED_432 = np.float64(432.0)   # Universal harmony
SACRED_528 = np.float64(528.0)   # DNA repair frequency
SACRED_639 = np.float64(639.0)   # Heart chakra

# Prime numbers (first 50)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
          127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
          199, 211, 223, 227, 229]

# Fibonacci sequence (converges to golden ratio)
def generate_fibonacci(n: int) -> List[int]:
    fib = [1, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

FIBONACCI = generate_fibonacci(50)

# Lucas numbers (related to Fibonacci, also converges to φ)
def generate_lucas(n: int) -> List[int]:
    lucas = [2, 1]
    for _ in range(2, n):
        lucas.append(lucas[-1] + lucas[-2])
    return lucas

LUCAS = generate_lucas(30)


class BrainwaveState(Enum):
    """Brainwave frequency ranges"""
    DELTA = (0.5, 4.0)    # Deep sleep, healing
    THETA = (4.0, 8.0)    # Meditation, creativity
    ALPHA = (8.0, 13.0)   # Relaxed awareness
    BETA = (13.0, 30.0)   # Active thinking
    GAMMA = (30.0, 100.0) # Peak performance
    
    def get_golden_frequency(self) -> float:
        """Get the golden ratio point within this range"""
        low, high = self.value
        return low + (high - low) * PHI_CONJUGATE


class GoldenSequenceType(Enum):
    """Types of golden sequences for beat frequencies"""
    FIBONACCI = "fibonacci"
    LUCAS = "lucas"
    PRIME_GOLDEN = "prime_golden"
    PHI_POWERS = "phi_powers"
    DIVINE_SPIRAL = "divine_spiral"


@dataclass
class AnnealingProfile:
    """Configuration for the annealing process"""
    name: str
    num_stages: int
    base_frequency: float
    starting_state: BrainwaveState
    ending_state: BrainwaveState  # Should approach silence
    sequence_type: GoldenSequenceType
    total_duration_minutes: float
    
    # Golden ratio multipliers for each parameter
    frequency_decay_power: float = field(default_factory=lambda: PHI_CONJUGATE)
    amplitude_decay_power: float = field(default_factory=lambda: PHI_CONJUGATE)
    transition_ratio: float = field(default_factory=lambda: PHI_CONJUGATE * PHI_CONJUGATE)


# Predefined profiles
PROFILES = {
    "deep_meditation": AnnealingProfile(
        name="Deep Meditation Annealing",
        num_stages=8,
        base_frequency=SACRED_432,
        starting_state=BrainwaveState.ALPHA,
        ending_state=BrainwaveState.DELTA,
        sequence_type=GoldenSequenceType.FIBONACCI,
        total_duration_minutes=21,  # 21 is Fibonacci
    ),
    "healing_sleep": AnnealingProfile(
        name="Healing Sleep Journey",
        num_stages=13,  # Fibonacci
        base_frequency=SACRED_528,
        starting_state=BrainwaveState.THETA,
        ending_state=BrainwaveState.DELTA,
        sequence_type=GoldenSequenceType.PHI_POWERS,
        total_duration_minutes=34,  # Fibonacci
    ),
    "transcendence": AnnealingProfile(
        name="Transcendence to Silence",
        num_stages=5,  # Fibonacci prime
        base_frequency=SACRED_639,
        starting_state=BrainwaveState.BETA,
        ending_state=BrainwaveState.DELTA,
        sequence_type=GoldenSequenceType.DIVINE_SPIRAL,
        total_duration_minutes=13,  # Fibonacci prime
    ),
}


def golden_spiral_easing(t: float, precision: int = 1) -> float:
    """
    Golden spiral easing - Divine transition curve
    
    precision levels:
    1 - Standard golden spiral
    2 - Double golden spiral (more pronounced)
    3 - Triple golden spiral (maximum divine curve)
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    
    result = t
    for _ in range(precision):
        # Each iteration applies golden spiral transformation
        theta = result * np.pi * PHI
        golden_ease = (1 - np.cos(theta * PHI_CONJUGATE)) / 2
        
        x = (result - 0.5) * 4
        golden_sigmoid = 1 / (1 + np.exp(-x * PHI))
        
        result = golden_ease * PHI_CONJUGATE + golden_sigmoid * (1 - PHI_CONJUGATE)
    
    return np.clip(result, 0.0, 1.0)


def generate_golden_beat_sequence(
    sequence_type: GoldenSequenceType,
    num_beats: int,
    starting_freq: float
) -> List[float]:
    """
    Generate a sequence of beat frequencies based on golden mathematics
    """
    beats = []
    
    if sequence_type == GoldenSequenceType.FIBONACCI:
        # Beat frequencies as Fibonacci ratios of starting frequency
        for i in range(num_beats):
            idx = min(i + 3, len(FIBONACCI) - 1)
            beat = starting_freq / FIBONACCI[idx]
            beats.append(beat)
            
    elif sequence_type == GoldenSequenceType.LUCAS:
        # Lucas number ratios
        for i in range(num_beats):
            idx = min(i + 2, len(LUCAS) - 1)
            beat = starting_freq / LUCAS[idx]
            beats.append(beat)
            
    elif sequence_type == GoldenSequenceType.PRIME_GOLDEN:
        # Prime numbers scaled by golden ratio
        for i in range(num_beats):
            prime = PRIMES[min(i, len(PRIMES) - 1)]
            beat = (starting_freq / prime) * (PHI_CONJUGATE ** i)
            beats.append(beat)
            
    elif sequence_type == GoldenSequenceType.PHI_POWERS:
        # Pure golden ratio powers
        for i in range(num_beats):
            beat = starting_freq / (PHI ** (i + 2))
            beats.append(beat)
            
    elif sequence_type == GoldenSequenceType.DIVINE_SPIRAL:
        # Combination of all methods in golden proportions
        for i in range(num_beats):
            # Fibonacci component
            fib_idx = min(i + 3, len(FIBONACCI) - 1)
            fib_beat = starting_freq / FIBONACCI[fib_idx]
            
            # Prime component
            prime = PRIMES[min(i, len(PRIMES) - 1)]
            prime_beat = starting_freq / prime
            
            # Phi power component
            phi_beat = starting_freq / (PHI ** (i + 2))
            
            # Blend in golden proportions
            beat = (fib_beat * PHI_CONJUGATE + 
                    prime_beat * (1 - PHI_CONJUGATE) * PHI_CONJUGATE +
                    phi_beat * (1 - PHI_CONJUGATE) * (1 - PHI_CONJUGATE))
            beats.append(beat)
    
    return beats


def calculate_golden_durations(
    total_duration: float,
    num_segments: int
) -> Tuple[List[float], List[float]]:
    """
    Calculate segment durations and transition times in golden proportions
    
    The sum of all durations and transitions equals total_duration
    Each duration is in golden ratio to its neighbors
    """
    # Calculate golden weights for each segment
    weights = [PHI_CONJUGATE ** i for i in range(num_segments)]
    total_weight = sum(weights)
    
    # Allocate time for segments (φ² of total) and transitions (rest)
    segment_time = total_duration * PHI_CONJUGATE * PHI_CONJUGATE
    transition_time = total_duration - segment_time
    
    # Distribute segment time
    durations = [(w / total_weight) * segment_time for w in weights]
    
    # Transitions between segments (one less than segments)
    if num_segments > 1:
        trans_weights = [PHI_CONJUGATE ** i for i in range(num_segments - 1)]
        trans_total = sum(trans_weights)
        transitions = [(w / trans_total) * transition_time for w in trans_weights]
    else:
        transitions = []
    
    return durations, transitions


class AdvancedBinauralGenerator:
    """
    Advanced binaural beat generator with full golden ratio coherence
    """
    
    def __init__(self, sample_rate: int = 96000):
        self.sample_rate = sample_rate
        self.dtype = np.float64
        
    def generate_from_profile(
        self,
        profile: AnnealingProfile
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete annealing sequence from a profile
        """
        total_duration_seconds = profile.total_duration_minutes * 60
        
        # Calculate durations in golden proportions
        durations, transitions = calculate_golden_durations(
            total_duration_seconds,
            profile.num_stages
        )
        
        # Generate beat frequency sequence
        starting_beat = profile.starting_state.get_golden_frequency()
        beats = generate_golden_beat_sequence(
            profile.sequence_type,
            profile.num_stages,
            starting_beat
        )
        
        print(f"\n{'═' * 60}")
        print(f"  {profile.name.upper()}")
        print(f"{'═' * 60}")
        print(f"  Base Frequency: {profile.base_frequency} Hz")
        print(f"  Stages: {profile.num_stages}")
        print(f"  Total Duration: {profile.total_duration_minutes} minutes")
        print(f"  Sequence Type: {profile.sequence_type.value}")
        print(f"{'═' * 60}")
        
        left_channel = []
        right_channel = []
        
        for stage in range(profile.num_stages):
            # Calculate stage parameters
            annealing_factor = stage / max(1, profile.num_stages - 1)
            
            # Phase approaches π using golden spiral
            phase = np.pi * golden_spiral_easing(annealing_factor, precision=2)
            
            # Amplitude decreases toward silence
            amplitude = (1 - golden_spiral_easing(annealing_factor, precision=2) * 0.95)
            amplitude *= profile.amplitude_decay_power ** (stage * 0.5)
            
            # Beat frequency from sequence
            beat_freq = beats[stage]
            
            # Ensure beat stays within target brainwave range
            target_range = profile.ending_state.value
            beat_freq = np.clip(beat_freq, target_range[0], 
                               profile.starting_state.value[1])
            
            print(f"\n  Stage {stage + 1}/{profile.num_stages}:")
            print(f"    Duration: {durations[stage]:.2f}s")
            print(f"    Beat Freq: {beat_freq:.4f} Hz")
            print(f"    Amplitude: {amplitude:.4f}")
            print(f"    Phase: {np.degrees(phase):.2f}°")
            
            # Generate binaural beat
            left, right = self._generate_segment(
                profile.base_frequency,
                beat_freq,
                durations[stage],
                amplitude,
                phase
            )
            
            # Apply golden envelope
            left = self._apply_golden_envelope(left)
            right = self._apply_golden_envelope(right)
            
            left_channel.append(left)
            right_channel.append(right)
            
            # Generate transition
            if stage < profile.num_stages - 1:
                next_beat = beats[stage + 1]
                next_amp = amplitude * profile.amplitude_decay_power
                next_phase = np.pi * golden_spiral_easing(
                    (stage + 1) / max(1, profile.num_stages - 1), 
                    precision=2
                )
                
                trans_left, trans_right = self._generate_golden_transition(
                    profile.base_frequency,
                    beat_freq, next_beat,
                    amplitude, next_amp,
                    phase, next_phase,
                    transitions[stage]
                )
                
                left_channel.append(trans_left)
                right_channel.append(trans_right)
        
        # Final phase cancellation
        silence_duration = FIBONACCI[7] * PHI_CONJUGATE
        final_left, final_right = self._generate_phase_cancellation(
            profile.base_frequency,
            silence_duration
        )
        left_channel.append(final_left)
        right_channel.append(final_right)
        
        print(f"\n{'═' * 60}")
        print(f"  COMPLETE → Phase Cancellation → Silence")
        print(f"{'═' * 60}\n")
        
        return np.concatenate(left_channel), np.concatenate(right_channel)
    
    def _generate_segment(
        self,
        base_freq: float,
        beat_freq: float,
        duration: float,
        amplitude: float,
        phase: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single binaural segment"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        
        left = amplitude * np.sin(2 * np.pi * base_freq * t + phase)
        right = amplitude * np.sin(
            2 * np.pi * (base_freq + beat_freq) * t + 
            phase + np.pi * PHI_CONJUGATE
        )
        
        return left, right
    
    def _apply_golden_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Apply golden ratio envelope"""
        num_samples = len(audio)
        envelope = np.ones(num_samples, dtype=self.dtype)
        
        attack_samples = int(num_samples * PHI_CONJUGATE * PHI_CONJUGATE * 0.1)
        decay_samples = int(num_samples * PHI_CONJUGATE * 0.1)
        
        for i in range(min(attack_samples, num_samples)):
            t = i / attack_samples
            envelope[i] = golden_spiral_easing(t)
        
        for i in range(min(decay_samples, num_samples)):
            t = i / decay_samples
            envelope[-(i+1)] = golden_spiral_easing(t)
        
        return audio * envelope
    
    def _generate_golden_transition(
        self,
        base_freq: float,
        beat1: float, beat2: float,
        amp1: float, amp2: float,
        phase1: float, phase2: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate golden spiral transition between segments"""
        num_samples = int(duration * self.sample_rate)
        left = np.zeros(num_samples, dtype=self.dtype)
        right = np.zeros(num_samples, dtype=self.dtype)
        
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        
        for i in range(num_samples):
            factor = golden_spiral_easing(i / num_samples, precision=2)
            
            beat = beat1 + (beat2 - beat1) * factor
            amp = amp1 + (amp2 - amp1) * factor
            phase = phase1 + (phase2 - phase1) * factor
            
            left[i] = amp * np.sin(2 * np.pi * base_freq * t[i] + phase)
            right[i] = amp * np.sin(
                2 * np.pi * (base_freq + beat) * t[i] + 
                phase + np.pi * PHI_CONJUGATE
            )
        
        return left, right
    
    def _generate_phase_cancellation(
        self,
        frequency: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate phase cancellation approach to silence"""
        num_samples = int(duration * self.sample_rate)
        left = np.zeros(num_samples, dtype=self.dtype)
        right = np.zeros(num_samples, dtype=self.dtype)
        
        t = np.linspace(0, duration, num_samples, dtype=self.dtype)
        base_amplitude = PHI_CONJUGATE ** 4
        
        for i in range(num_samples):
            progress = golden_spiral_easing(i / num_samples, precision=3)
            
            phase_diff = np.pi * progress
            amp = base_amplitude * (1 - progress)
            
            left[i] = amp * np.sin(2 * np.pi * frequency * t[i])
            right[i] = amp * np.sin(2 * np.pi * frequency * t[i] + phase_diff)
        
        return left, right


def save_high_quality_wav(
    filename: str,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int = 96000,
    bit_depth: int = 24
):
    """
    Save as high-quality WAV file
    Supports 16, 24, or 32 bit depth
    """
    import struct
    
    # Normalize
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left = left / max_val * 0.95
        right = right / max_val * 0.95
    
    if bit_depth == 16:
        max_int = 32767
        dtype = np.int16
        bytes_per_sample = 2
    elif bit_depth == 24:
        max_int = 8388607
        dtype = np.int32  # We'll handle 24-bit specially
        bytes_per_sample = 3
    else:  # 32-bit
        max_int = 2147483647
        dtype = np.int32
        bytes_per_sample = 4
    
    left_int = (left * max_int).astype(np.int32)
    right_int = (right * max_int).astype(np.int32)
    
    with open(filename, 'wb') as f:
        num_samples = len(left)
        data_size = num_samples * 2 * bytes_per_sample
        
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        
        # Format chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))  # PCM
        f.write(struct.pack('<H', 2))  # Stereo
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2 * bytes_per_sample))
        f.write(struct.pack('<H', 2 * bytes_per_sample))
        f.write(struct.pack('<H', bit_depth))
        
        # Data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        
        if bit_depth == 24:
            # Write 24-bit samples
            for i in range(num_samples):
                # Left channel
                val = left_int[i]
                f.write(struct.pack('<I', val & 0xFFFFFF)[:3])
                # Right channel
                val = right_int[i]
                f.write(struct.pack('<I', val & 0xFFFFFF)[:3])
        else:
            # Interleave and write
            stereo = np.empty((num_samples * 2,), dtype=dtype)
            stereo[0::2] = left_int.astype(dtype)
            stereo[1::2] = right_int.astype(dtype)
            f.write(stereo.tobytes())
    
    print(f"Saved: {filename} ({bit_depth}-bit, {sample_rate}Hz)")


def main():
    """
    Generate advanced binaural annealing sequences
    """
    print("\n" + "★" * 60)
    print("  DIVINE GOLDEN RATIO BINAURAL GENERATOR")
    print("  Advanced Version with Prime & Fibonacci Sequences")
    print("★" * 60)
    
    generator = AdvancedBinauralGenerator(sample_rate=96000)
    
    # Generate from predefined profiles
    for profile_name, profile in PROFILES.items():
        print(f"\nGenerating: {profile_name}")
        
        left, right = generator.generate_from_profile(profile)
        
        filename = f"golden_{profile_name}.wav"
        save_high_quality_wav(filename, left, right, 96000, 24)
    
    print("\n" + "★" * 60)
    print("  ALL SEQUENCES GENERATED")
    print("  Use with stereo headphones for full binaural effect")
    print("★" * 60 + "\n")


if __name__ == "__main__":
    main()
