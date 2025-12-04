"""
Soundboard Physical Panning Module
==================================

Physical model for vibroacoustic soundboard with two exciters (HEAD/FEET)
mounted on spruce board with spring isolation.

Setup:
    - Spruce board: 2000mm × 600mm × 10mm
    - Exciters: Head (0mm) and Feet (2000mm) on short edges
    - Listener: Lying centered, ears ~150mm from head edge
    - Springs: 5× (4 corners + 1 center), 15-20kg each for floor decoupling

Physics:
    - Sound velocity in spruce (longitudinal): 5500 m/s
    - Max propagation delay (2m): 0.36ms ≈ 17 samples @48kHz
    - Resonant board on springs = soft attenuation (α ≈ 0.4)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS - SPRUCE SOUNDBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# Sound velocity in spruce (Abete)
SPRUCE_VELOCITY_LONGITUDINAL = 5500.0  # m/s along grain (fibra lungo la tavola)
SPRUCE_VELOCITY_TRANSVERSE = 1800.0    # m/s across grain
SOUND_VELOCITY_AIR = 343.0             # m/s at 20°C (for reference)

# Board dimensions (Brico standard spruce)
BOARD_LENGTH_MM = 1950.0   # mm (actual distance between exciter centers)
BOARD_WIDTH_MM = 600.0     # mm
BOARD_THICKNESS_MM = 10.0  # mm

# Exciter: Visaton EX 60S (body-shaker transducer)
# Specifications (from official Visaton datasheet):
#   - Power: 25W RMS
#   - Impedance: 8 ohm nominal (~8-25Ω curve)
#   - Resonance Fs: ~65 Hz (impedance peak from graph)
#   - Frequency range: 50 Hz - 20 kHz (-3dB)
#   - SPL peak: ~80 dB @ 300 Hz
#   - Magnet: Neodymium
#   - Note: dip ~500-600 Hz, rise at HF 10-20 kHz
EXCITER_MODEL = "Visaton EX 60S"
EXCITER_POWER_W = 25.0          # Watts RMS
EXCITER_IMPEDANCE_OHM = 8.0     # Ohms nominal
EXCITER_RESONANCE_HZ = 65.0     # Hz (Fs from impedance curve)
EXCITER_FREQ_MIN_HZ = 50.0      # Hz (-3dB point)
EXCITER_FREQ_MAX_HZ = 20000.0   # Hz
EXCITER_SPL_PEAK_HZ = 300.0     # Hz (maximum efficiency)

# Amp: Behringer EPQ304 in stereo mode
# Specifications:
#   - 4 channels, Class D
#   - 40W RMS @ 8 ohm per channel
#   - 65W RMS @ 4 ohm per channel
#   - 130W bridged @ 8 ohm
AMP_MODEL = "Behringer EPQ304 (stereo)"
AMP_POWER_8OHM_W = 40.0         # Watts RMS per channel @ 8 ohm
AMP_POWER_4OHM_W = 65.0         # Watts RMS per channel @ 4 ohm
AMP_CHANNELS = 4

# Exciter positions (head-feet axis)
EXCITER_HEAD_POS_MM = 0.0      # Head exciter at top edge
EXCITER_FEET_POS_MM = 1950.0   # Feet exciter at bottom edge (center-to-center)

# Listener position
LISTENER_CENTER_MM = 1000.0    # Center of board
EARS_FROM_HEAD_EDGE_MM = 150.0 # Ears position (~15cm from head edge)

# Spring isolation
NUM_SPRINGS = 5                # 4 corners + 1 center
SPRING_LOAD_KG = 17.5          # Average 15-20 kg per spring

# Attenuation factor (low for resonant board on springs)
ATTENUATION_ALPHA = 0.4        # Soft decay (0.0 = no decay, 1.0 = inverse distance)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class PanAxis(Enum):
    """Panning axis orientation."""
    HEAD_FEET = "head_feet"      # Longitudinal (default for vibroacoustic)
    LEFT_RIGHT = "left_right"    # Lateral (traditional stereo)


@dataclass
class SoundboardConfig:
    """Configuration for soundboard physical panning."""
    
    # Board dimensions
    length_mm: float = BOARD_LENGTH_MM
    width_mm: float = BOARD_WIDTH_MM
    thickness_mm: float = BOARD_THICKNESS_MM
    
    # Sound velocity (use longitudinal for grain along length)
    velocity_ms: float = SPRUCE_VELOCITY_LONGITUDINAL
    
    # Exciter positions (center-to-center: 1950mm)
    exciter_head_mm: float = EXCITER_HEAD_POS_MM
    exciter_feet_mm: float = EXCITER_FEET_POS_MM
    
    # Hardware info
    exciter_model: str = EXCITER_MODEL
    exciter_power_w: float = EXCITER_POWER_W
    exciter_impedance_ohm: float = EXCITER_IMPEDANCE_OHM
    exciter_freq_min_hz: float = EXCITER_FREQ_MIN_HZ
    exciter_freq_max_hz: float = EXCITER_FREQ_MAX_HZ
    amp_model: str = AMP_MODEL
    amp_power_w: float = AMP_POWER_8OHM_W
    
    # Listener
    ears_pos_mm: float = EARS_FROM_HEAD_EDGE_MM
    
    # Attenuation
    attenuation_alpha: float = ATTENUATION_ALPHA
    
    # Sample rate
    sample_rate: int = 48000
    
    # Derived values (computed)
    _max_delay_ms: float = field(init=False, repr=False)
    _max_delay_samples: int = field(init=False, repr=False)
    
    def __post_init__(self):
        """Compute derived values."""
        distance_m = self.length_mm / 1000.0
        self._max_delay_ms = (distance_m / self.velocity_ms) * 1000.0
        self._max_delay_samples = int(np.ceil(self._max_delay_ms / 1000.0 * self.sample_rate))
    
    @property
    def max_delay_ms(self) -> float:
        """Maximum delay in milliseconds."""
        return self._max_delay_ms
    
    @property
    def max_delay_samples(self) -> int:
        """Maximum delay in samples."""
        return self._max_delay_samples
    
    def get_info(self) -> str:
        """Return configuration info string."""
        return (
            f"Soundboard Config:\n"
            f"  Board: {self.length_mm}×{self.width_mm}×{self.thickness_mm} mm\n"
            f"  Velocity: {self.velocity_ms} m/s (spruce longitudinal)\n"
            f"  Max delay: {self.max_delay_ms:.3f} ms = {self.max_delay_samples} samples @{self.sample_rate}Hz\n"
            f"  Attenuation α: {self.attenuation_alpha}\n"
            f"  Exciter: {self.exciter_model} ({self.exciter_power_w}W @ {self.exciter_impedance_ohm}Ω)\n"
            f"  Amp: {self.amp_model} ({self.amp_power_w}W/ch @ 8Ω)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DELAY LINE WITH FRACTIONAL INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════════

class FractionalDelayLine:
    """
    Circular buffer delay line with linear interpolation for fractional delays.
    
    Supports sub-sample precision for accurate ITD modeling.
    """
    
    def __init__(self, max_delay_samples: int, num_channels: int = 1):
        """
        Initialize delay line.
        
        Args:
            max_delay_samples: Maximum delay in samples
            num_channels: Number of audio channels
        """
        # Add extra samples for interpolation safety
        self.buffer_size = max_delay_samples + 4
        self.num_channels = num_channels
        self.buffer = np.zeros((num_channels, self.buffer_size), dtype=np.float64)
        self.write_pos = 0
        self.max_delay = max_delay_samples
    
    def write(self, samples: np.ndarray):
        """
        Write samples to the delay line.
        
        Args:
            samples: Array of shape (num_channels,) or (num_channels, num_samples)
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        num_samples = samples.shape[1]
        
        for i in range(num_samples):
            self.buffer[:, self.write_pos] = samples[:, i]
            self.write_pos = (self.write_pos + 1) % self.buffer_size
    
    def read(self, delay_samples: float, channel: int = 0) -> float:
        """
        Read from delay line with fractional interpolation.
        
        Args:
            delay_samples: Delay in samples (can be fractional)
            channel: Channel index
            
        Returns:
            Interpolated sample value
        """
        # Clamp delay
        delay_samples = np.clip(delay_samples, 0, self.max_delay)
        
        # Calculate read position
        read_pos = self.write_pos - 1 - delay_samples
        
        # Integer and fractional parts
        read_pos_int = int(np.floor(read_pos))
        frac = read_pos - read_pos_int
        
        # Wrap positions
        pos0 = read_pos_int % self.buffer_size
        pos1 = (read_pos_int + 1) % self.buffer_size
        
        # Linear interpolation
        sample0 = self.buffer[channel, pos0]
        sample1 = self.buffer[channel, pos1]
        
        return sample0 + frac * (sample1 - sample0)
    
    def read_block(self, delay_samples: float, num_samples: int, channel: int = 0) -> np.ndarray:
        """
        Read a block of samples with constant delay.
        
        Args:
            delay_samples: Delay in samples (can be fractional)
            num_samples: Number of samples to read
            channel: Channel index
            
        Returns:
            Array of interpolated samples
        """
        output = np.zeros(num_samples, dtype=np.float64)
        
        delay_samples = np.clip(delay_samples, 0, self.max_delay)
        read_pos_int = int(np.floor(self.write_pos - 1 - delay_samples))
        frac = (self.write_pos - 1 - delay_samples) - read_pos_int
        
        for i in range(num_samples):
            pos0 = (read_pos_int - i) % self.buffer_size
            pos1 = (read_pos_int - i + 1) % self.buffer_size
            sample0 = self.buffer[channel, pos0]
            sample1 = self.buffer[channel, pos1]
            output[i] = sample0 + frac * (sample1 - sample0)
        
        return output[::-1]  # Reverse to get correct time order
    
    def clear(self):
        """Clear the delay buffer."""
        self.buffer.fill(0)
        self.write_pos = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ITD / ILD CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_itd(
    pan: float,
    config: SoundboardConfig
) -> Tuple[float, float]:
    """
    Calculate Interaural Time Difference (ITD) for head-feet panning.
    
    Args:
        pan: Pan position from -1.0 (head) to +1.0 (feet), 0.0 = center
        config: Soundboard configuration
        
    Returns:
        (delay_head_ms, delay_feet_ms) - Delays for each exciter in milliseconds
    """
    # Normalize pan to 0-1 range
    # pan = -1 → all head (delay_head=0, delay_feet=max)
    # pan = +1 → all feet (delay_head=max, delay_feet=0)
    # pan = 0  → center (both at half delay)
    
    pan_normalized = (pan + 1.0) / 2.0  # 0 to 1
    
    delay_head_ms = pan_normalized * config.max_delay_ms
    delay_feet_ms = (1.0 - pan_normalized) * config.max_delay_ms
    
    return delay_head_ms, delay_feet_ms


def calculate_itd_samples(
    pan: float,
    config: SoundboardConfig
) -> Tuple[float, float]:
    """
    Calculate ITD in samples (fractional for sub-sample precision).
    
    Args:
        pan: Pan position from -1.0 (head) to +1.0 (feet)
        config: Soundboard configuration
        
    Returns:
        (delay_head_samples, delay_feet_samples)
    """
    delay_head_ms, delay_feet_ms = calculate_itd(pan, config)
    
    delay_head_samples = delay_head_ms / 1000.0 * config.sample_rate
    delay_feet_samples = delay_feet_ms / 1000.0 * config.sample_rate
    
    return delay_head_samples, delay_feet_samples


def calculate_ild(
    pan: float,
    config: SoundboardConfig,
    use_equal_power: bool = True
) -> Tuple[float, float]:
    """
    Calculate Interaural Level Difference (ILD) for head-feet panning.
    
    Combines equal-power panning with soft distance attenuation for
    resonant soundboard on springs.
    
    Args:
        pan: Pan position from -1.0 (head) to +1.0 (feet)
        config: Soundboard configuration
        use_equal_power: Use equal power pan law (recommended)
        
    Returns:
        (gain_head, gain_feet) - Amplitude gains for each exciter (0.0 to 1.0)
    """
    if use_equal_power:
        # Equal power panning (constant loudness)
        # pan = -1 → head=1, feet=0
        # pan = 0  → head=0.707, feet=0.707
        # pan = +1 → head=0, feet=1
        pan_normalized = (pan + 1.0) / 2.0  # 0 to 1
        
        gain_head = np.cos(pan_normalized * np.pi / 2)
        gain_feet = np.sin(pan_normalized * np.pi / 2)
    else:
        # Linear panning (simpler but -6dB at center)
        pan_normalized = (pan + 1.0) / 2.0
        gain_head = 1.0 - pan_normalized
        gain_feet = pan_normalized
    
    # Apply soft distance attenuation (for resonant board, keep it subtle)
    if config.attenuation_alpha > 0:
        # Virtual source position (0 to length_mm based on pan)
        source_pos = (pan + 1.0) / 2.0 * config.length_mm
        
        # Distance from source to each exciter
        dist_to_head = abs(source_pos - config.exciter_head_mm) + 1  # +1 to avoid div/0
        dist_to_feet = abs(source_pos - config.exciter_feet_mm) + 1
        
        # Soft attenuation: (ref_distance / distance)^alpha
        ref_dist = config.length_mm / 2  # Reference at center
        
        atten_head = (ref_dist / dist_to_head) ** config.attenuation_alpha
        atten_feet = (ref_dist / dist_to_feet) ** config.attenuation_alpha
        
        # Normalize to maintain overall level
        max_atten = max(atten_head, atten_feet)
        atten_head /= max_atten
        atten_feet /= max_atten
        
        # Combine with equal power
        gain_head *= atten_head
        gain_feet *= atten_feet
    
    return gain_head, gain_feet


def calculate_panning(
    pan: float,
    config: SoundboardConfig
) -> dict:
    """
    Calculate complete panning parameters (ITD + ILD).
    
    Args:
        pan: Pan position from -1.0 (head) to +1.0 (feet)
        config: Soundboard configuration
        
    Returns:
        Dictionary with all panning parameters
    """
    delay_head_ms, delay_feet_ms = calculate_itd(pan, config)
    delay_head_samples, delay_feet_samples = calculate_itd_samples(pan, config)
    gain_head, gain_feet = calculate_ild(pan, config)
    
    return {
        'pan': pan,
        'delay_head_ms': delay_head_ms,
        'delay_feet_ms': delay_feet_ms,
        'delay_head_samples': delay_head_samples,
        'delay_feet_samples': delay_feet_samples,
        'gain_head': gain_head,
        'gain_feet': gain_feet,
        'gain_head_db': 20 * np.log10(gain_head + 1e-10),
        'gain_feet_db': 20 * np.log10(gain_feet + 1e-10),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SOUNDBOARD PANNER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SoundboardPanner:
    """
    Real-time soundboard panner with ITD and ILD.
    
    Converts mono input to stereo output for HEAD and FEET exciters
    with physical delay and amplitude modeling.
    """
    
    def __init__(self, config: Optional[SoundboardConfig] = None):
        """
        Initialize the soundboard panner.
        
        Args:
            config: Soundboard configuration (uses defaults if None)
        """
        self.config = config or SoundboardConfig()
        
        # Delay lines for each output channel
        self.delay_head = FractionalDelayLine(self.config.max_delay_samples + 10)
        self.delay_feet = FractionalDelayLine(self.config.max_delay_samples + 10)
        
        # Current pan position
        self._pan = 0.0
        self._target_pan = 0.0
        
        # Smoothing for pan changes (avoid clicks)
        self._pan_smooth_samples = int(0.010 * self.config.sample_rate)  # 10ms
        
        # Current parameters
        self._update_params()
    
    @property
    def pan(self) -> float:
        """Current pan position."""
        return self._pan
    
    @pan.setter
    def pan(self, value: float):
        """Set pan position with smoothing."""
        self._target_pan = np.clip(value, -1.0, 1.0)
    
    def set_pan_immediate(self, value: float):
        """Set pan position immediately (no smoothing)."""
        self._pan = np.clip(value, -1.0, 1.0)
        self._target_pan = self._pan
        self._update_params()
    
    def _update_params(self):
        """Update internal parameters from current pan."""
        self._params = calculate_panning(self._pan, self.config)
    
    def process_sample(self, mono_sample: float) -> Tuple[float, float]:
        """
        Process a single sample.
        
        Args:
            mono_sample: Input mono sample
            
        Returns:
            (head_sample, feet_sample) for the two exciters
        """
        # Smooth pan changes
        if self._pan != self._target_pan:
            diff = self._target_pan - self._pan
            step = diff / self._pan_smooth_samples
            self._pan += step
            if abs(self._pan - self._target_pan) < 0.001:
                self._pan = self._target_pan
            self._update_params()
        
        # Write to delay lines
        self.delay_head.write(np.array([mono_sample]))
        self.delay_feet.write(np.array([mono_sample]))
        
        # Read with ITD delays
        head_sample = self.delay_head.read(self._params['delay_head_samples'])
        feet_sample = self.delay_feet.read(self._params['delay_feet_samples'])
        
        # Apply ILD gains
        head_sample *= self._params['gain_head']
        feet_sample *= self._params['gain_feet']
        
        return head_sample, feet_sample
    
    def process_block(self, mono_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a block of samples.
        
        Args:
            mono_block: Input mono samples (1D array)
            
        Returns:
            (head_block, feet_block) arrays for the two exciters
        """
        num_samples = len(mono_block)
        head_out = np.zeros(num_samples, dtype=np.float64)
        feet_out = np.zeros(num_samples, dtype=np.float64)
        
        for i in range(num_samples):
            head_out[i], feet_out[i] = self.process_sample(mono_block[i])
        
        return head_out, feet_out
    
    def get_params(self) -> dict:
        """Get current panning parameters."""
        return self._params.copy()
    
    def reset(self):
        """Reset delay lines and state."""
        self.delay_head.clear()
        self.delay_feet.clear()
        self._pan = 0.0
        self._target_pan = 0.0
        self._update_params()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_panning_table(config: Optional[SoundboardConfig] = None):
    """Print a table of panning values for reference."""
    config = config or SoundboardConfig()
    
    print("═" * 70)
    print("  SOUNDBOARD PANNING TABLE - Head/Feet Axis")
    print("═" * 70)
    print(f"  {config.get_info()}")
    print("═" * 70)
    print()
    print(f"{'Pan':>6} │ {'Position':^10} │ {'Delay H':>8} │ {'Delay F':>8} │ {'Gain H':>7} │ {'Gain F':>7}")
    print(f"{'':>6} │ {'':^10} │ {'(ms)':>8} │ {'(ms)':>8} │ {'(dB)':>7} │ {'(dB)':>7}")
    print("─" * 70)
    
    for pan in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
        params = calculate_panning(pan, config)
        
        if pan == -1:
            pos = "HEAD"
        elif pan == 1:
            pos = "FEET"
        elif pan == 0:
            pos = "CENTER"
        else:
            pos = f"{'↑' if pan < 0 else '↓'} {abs(pan):.2f}"
        
        print(f"{pan:+6.2f} │ {pos:^10} │ {params['delay_head_ms']:8.3f} │ "
              f"{params['delay_feet_ms']:8.3f} │ {params['gain_head_db']:7.2f} │ "
              f"{params['gain_feet_db']:7.2f}")
    
    print("═" * 70)


def demo_sweep(duration_sec: float = 5.0, sample_rate: int = 48000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a demo sweep from head to feet.
    
    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        (head_signal, feet_signal) arrays
    """
    config = SoundboardConfig(sample_rate=sample_rate)
    panner = SoundboardPanner(config)
    
    num_samples = int(duration_sec * sample_rate)
    
    # Generate test tone (440 Hz)
    t = np.linspace(0, duration_sec, num_samples)
    mono_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Sweep pan from head to feet
    pan_sweep = np.linspace(-1, 1, num_samples)
    
    head_out = np.zeros(num_samples)
    feet_out = np.zeros(num_samples)
    
    for i in range(num_samples):
        panner.pan = pan_sweep[i]
        head_out[i], feet_out[i] = panner.process_sample(mono_signal[i])
    
    return head_out, feet_out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🪵 Soundboard Physical Panning Module\n")
    
    # Show configuration
    config = SoundboardConfig()
    print(config.get_info())
    print()
    
    # Print panning table
    print_panning_table(config)
