"""
Vibroacoustic Panning Module
=============================

Physical model for vibroacoustic soundboard with multiple exciters.
Implements ITD/ILD panning with propagation delays and golden ratio positioning.

Hardware Setup:
- Spruce board: 1950mm × 600mm × 10mm
- Exciters: Head (0mm) and Feet (1950mm) on short edges
- Sound velocity in spruce: 5500 m/s
- Max delay: ~0.35ms ≈ 17 samples @48kHz

Features:
- Real-time ITD/ILD panning
- Golden ratio frequency distribution
- Propagation delay modeling
- Frequency-dependent attenuation
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


# Physical constants
SPRUCE_VELOCITY_MS = 5500.0  # m/s along grain
BOARD_LENGTH_MM = 1950.0     # mm between exciters
PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498949


class PanAxis(Enum):
    """Panning axis"""
    HEAD_FEET = "head_feet"      # Longitudinal (vibroacoustic default)
    LEFT_RIGHT = "left_right"    # Lateral (traditional stereo)


class VibroacousticPanner:
    """
    Physical vibroacoustic panning engine.
    
    Supports:
    - 2-channel (head/feet or L/R)
    - 4-channel (quad: HL, HR, FL, FR)
    - ITD (Inter-Transducer Delay) from propagation
    - ILD (Inter-Transducer Level Difference) from attenuation
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 num_channels: int = 2,
                 board_length_mm: float = BOARD_LENGTH_MM,
                 velocity_ms: float = SPRUCE_VELOCITY_MS,
                 attenuation_alpha: float = 0.4):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            num_channels: 2 or 4 channels
            board_length_mm: Distance between exciters (mm)
            velocity_ms: Sound velocity in medium (m/s)
            attenuation_alpha: Attenuation factor (0=none, 1=inverse distance)
        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.board_length_mm = board_length_mm
        self.velocity_ms = velocity_ms
        self.attenuation_alpha = attenuation_alpha
        
        # Calculate max delay
        self.max_delay_ms = (board_length_mm / 1000.0) / velocity_ms * 1000
        self.max_delay_samples = int(self.max_delay_ms * sample_rate / 1000.0)
        
        # Delay buffers for each channel
        self.delay_buffers = [
            np.zeros(self.max_delay_samples + 1) for _ in range(num_channels)
        ]
        self.buffer_pos = 0
        
        print(f"✓ Vibroacoustic panner initialized:")
        print(f"  Channels: {num_channels}")
        print(f"  Board length: {board_length_mm}mm")
        print(f"  Max delay: {self.max_delay_ms:.2f}ms ({self.max_delay_samples} samples)")
    
    def pan_2ch(self, audio_mono: np.ndarray, position: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pan mono audio to 2 channels with ITD/ILD.
        
        Args:
            audio_mono: Mono audio samples
            position: Pan position -1 (head/left) to +1 (feet/right)
        
        Returns:
            (channel_1, channel_2) audio arrays
        """
        position = np.clip(position, -1.0, 1.0)
        
        # Calculate distances from virtual source to each exciter
        # Position -1: at head (0mm), +1: at feet (board_length_mm)
        source_pos_mm = (position + 1) / 2 * self.board_length_mm
        
        dist_head_mm = abs(source_pos_mm - 0)
        dist_feet_mm = abs(source_pos_mm - self.board_length_mm)
        
        # ITD: Propagation delay
        delay_head_ms = (dist_head_mm / 1000.0) / self.velocity_ms * 1000
        delay_feet_ms = (dist_feet_mm / 1000.0) / self.velocity_ms * 1000
        
        delay_head_samples = int(delay_head_ms * self.sample_rate / 1000.0)
        delay_feet_samples = int(delay_feet_ms * self.sample_rate / 1000.0)
        
        # ILD: Amplitude attenuation (soft for resonant board)
        if self.attenuation_alpha > 0:
            # Normalize distances to 0-1 range
            dist_max = self.board_length_mm
            gain_head = 1.0 - (dist_head_mm / dist_max) ** self.attenuation_alpha
            gain_feet = 1.0 - (dist_feet_mm / dist_max) ** self.attenuation_alpha
        else:
            gain_head = 1.0
            gain_feet = 1.0
        
        # Normalize gains to preserve energy
        total_gain = gain_head + gain_feet
        if total_gain > 0:
            gain_head /= total_gain
            gain_feet /= total_gain
        
        # Apply delays and gains
        head_channel = self._apply_delay(audio_mono, delay_head_samples) * gain_head
        feet_channel = self._apply_delay(audio_mono, delay_feet_samples) * gain_feet
        
        return head_channel, feet_channel
    
    def pan_4ch(self, audio_mono: np.ndarray, 
                position_hf: float, position_lr: float) -> Tuple[np.ndarray, ...]:
        """
        Pan mono audio to 4 channels (quad).
        
        Args:
            audio_mono: Mono audio samples
            position_hf: Head/Feet position -1 to +1
            position_lr: Left/Right position -1 to +1
        
        Returns:
            (HL, HR, FL, FR) audio arrays
        """
        # First pan H/F axis
        head_audio, feet_audio = self.pan_2ch(audio_mono, position_hf)
        
        # Then pan L/R axis for each
        # For L/R, we use simpler constant-power panning (no physical delay needed)
        position_lr = np.clip(position_lr, -1.0, 1.0)
        
        # Constant power pan law
        pan_angle = (position_lr + 1) / 2 * np.pi / 2  # 0 to π/2
        gain_left = np.cos(pan_angle)
        gain_right = np.sin(pan_angle)
        
        # Apply L/R panning to head and feet separately
        HL = head_audio * gain_left
        HR = head_audio * gain_right
        FL = feet_audio * gain_left
        FR = feet_audio * gain_right
        
        return HL, HR, FL, FR
    
    def _apply_delay(self, audio: np.ndarray, delay_samples: int) -> np.ndarray:
        """
        Apply delay to audio using delay buffer.
        
        Args:
            audio: Input audio samples
            delay_samples: Delay in samples
        
        Returns:
            Delayed audio
        """
        if delay_samples <= 0:
            return audio
        
        delay_samples = min(delay_samples, self.max_delay_samples)
        
        # Simple delay line (circular buffer could be added for efficiency)
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        return delayed


class GoldenFrequencyPanner:
    """
    Golden ratio frequency distribution for vibroacoustic therapy.
    
    Distributes harmonics using φ-based spacing and pans them
    across the soundboard using golden angle positions.
    """
    
    def __init__(self, panner: VibroacousticPanner, base_freq: float = 432.0):
        """
        Args:
            panner: VibroacousticPanner instance
            base_freq: Base frequency (Hz)
        """
        self.panner = panner
        self.base_freq = base_freq
        
        # Golden angle for position distribution
        self.golden_angle_deg = 137.5077640500378546
        self.golden_angle_rad = np.radians(self.golden_angle_deg)
    
    def generate_harmonic_field(self, 
                                num_harmonics: int,
                                duration_sec: float,
                                amplitude: float = 0.3) -> np.ndarray:
        """
        Generate harmonic field with golden ratio distribution.
        
        Args:
            num_harmonics: Number of harmonics to generate
            duration_sec: Duration in seconds
            amplitude: Overall amplitude
        
        Returns:
            np.ndarray of shape (num_samples, num_channels)
        """
        num_samples = int(duration_sec * self.panner.sample_rate)
        t = np.arange(num_samples) / self.panner.sample_rate
        
        # Initialize output
        if self.panner.num_channels == 2:
            output = np.zeros((num_samples, 2))
        else:
            output = np.zeros((num_samples, 4))
        
        # Generate each harmonic
        for i in range(num_harmonics):
            # Frequency: Use golden ratio spacing
            freq = self.base_freq * (PHI ** (i * PHI_CONJUGATE))
            
            # Position: Use golden angle
            position = (i * self.golden_angle_rad) % (2 * np.pi)
            # Map to -1 to +1
            position = np.cos(position)
            
            # Generate harmonic
            harmonic = np.sin(2 * np.pi * freq * t) * amplitude / num_harmonics
            
            # Pan to position
            if self.panner.num_channels == 2:
                ch1, ch2 = self.panner.pan_2ch(harmonic, position)
                output[:, 0] += ch1
                output[:, 1] += ch2
            else:
                # For 4ch, add some L/R variation using phase
                position_lr = np.sin(position * PHI)
                HL, HR, FL, FR = self.panner.pan_4ch(harmonic, position, position_lr)
                output[:, 0] += HL
                output[:, 1] += HR
                output[:, 2] += FL
                output[:, 3] += FR
        
        return output


class VibroacousticProgram:
    """
    Vibroacoustic therapy program with dynamic panning.
    
    Moves sound sources along the body using golden ratio timing
    and frequency modulation.
    """
    
    def __init__(self, 
                 panner: VibroacousticPanner,
                 frequency: float = 432.0,
                 modulation_freq: float = 0.1):
        """
        Args:
            panner: VibroacousticPanner instance
            frequency: Carrier frequency (Hz)
            modulation_freq: Panning modulation frequency (Hz)
        """
        self.panner = panner
        self.frequency = frequency
        self.modulation_freq = modulation_freq
        self.time = 0.0
    
    def generate_frame(self, num_frames: int) -> np.ndarray:
        """
        Generate audio frame with dynamic panning.
        
        Args:
            num_frames: Number of frames to generate
        
        Returns:
            np.ndarray of shape (num_frames, num_channels)
        """
        t = np.arange(num_frames) / self.panner.sample_rate + self.time
        
        # Generate carrier
        carrier = np.sin(2 * np.pi * self.frequency * t)
        
        # Dynamic panning position (moves along H/F axis)
        # Use golden ratio phase for smooth motion
        position = np.sin(2 * np.pi * self.modulation_freq * t + PHI)
        
        # Pan each sample
        if self.panner.num_channels == 2:
            output = np.zeros((num_frames, 2))
            for i in range(num_frames):
                ch1, ch2 = self.panner.pan_2ch(carrier[i:i+1], position[i])
                output[i, 0] = ch1[0]
                output[i, 1] = ch2[0]
        else:
            output = np.zeros((num_frames, 4))
            position_lr = np.sin(2 * np.pi * self.modulation_freq * PHI * t)
            for i in range(num_frames):
                HL, HR, FL, FR = self.panner.pan_4ch(
                    carrier[i:i+1], position[i], position_lr[i]
                )
                output[i, 0] = HL[0]
                output[i, 1] = HR[0]
                output[i, 2] = FL[0]
                output[i, 3] = FR[0]
        
        # Update time
        self.time += num_frames / self.panner.sample_rate
        
        return output.astype(np.float32)


# Example usage
if __name__ == "__main__":
    print("🎵 Vibroacoustic Panning Module Test\n")
    
    # Create panner
    panner = VibroacousticPanner(
        sample_rate=48000,
        num_channels=2,
        board_length_mm=1950.0,
        velocity_ms=5500.0,
        attenuation_alpha=0.4
    )
    
    # Test static panning
    print("\n🧪 Testing static pan at center...")
    test_audio = np.sin(2 * np.pi * 432 * np.arange(48000) / 48000)
    head, feet = panner.pan_2ch(test_audio, 0.0)  # Center position
    print(f"✓ Head RMS: {np.sqrt(np.mean(head**2)):.3f}")
    print(f"✓ Feet RMS: {np.sqrt(np.mean(feet**2)):.3f}")
    
    # Test golden frequency distribution
    print("\n🌟 Testing golden harmonic field...")
    golden_panner = GoldenFrequencyPanner(panner, base_freq=432.0)
    harmonic_field = golden_panner.generate_harmonic_field(
        num_harmonics=8,
        duration_sec=1.0,
        amplitude=0.3
    )
    print(f"✓ Generated field shape: {harmonic_field.shape}")
    
    # Test dynamic program
    print("\n🌊 Testing dynamic vibroacoustic program...")
    program = VibroacousticProgram(
        panner=panner,
        frequency=432.0,
        modulation_freq=0.1  # 10 second cycle
    )
    frame = program.generate_frame(2048)
    print(f"✓ Generated frame shape: {frame.shape}")
    
    print("\n✓ All tests complete")
