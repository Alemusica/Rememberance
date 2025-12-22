"""
EMDR Bilateral Audio Module
============================

Backend-agnostic EMDR bilateral stimulation engine.
Extracted from UI layer for headless operation on Raspberry Pi.

Clinical Protocol:
- 0.5-2 Hz bilateral stimulation (validated by research)
- Alternating L/R audio tones
- Multiple modes: standard, golden_phase, breathing, theta_pulse

Sacred Geometry Enhancement:
- φ phase relationships (137.5°)
- Solfeggio frequency integration
- Golden ratio envelope curves
- Fibonacci-timed trauma annealing journeys
"""

import numpy as np
import time
from typing import Optional, Dict, List, Callable
from enum import Enum


# Golden ratio constants
PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498949
GOLDEN_ANGLE_RAD = np.radians(137.5077640500378546)

# EMDR protocol constants
EMDR_SPEED_MIN = 0.5   # Hz - Slow, grounding
EMDR_SPEED_MAX = 2.0   # Hz - Standard EMDR maximum
EMDR_SPEED_DEFAULT = 1.0  # Hz


class BilateralMode(Enum):
    """Bilateral stimulation modes"""
    STANDARD = "standard"           # Clean L/R alternation
    GOLDEN_PHASE = "golden_phase"   # 137.5° phase offset
    BREATHING = "breathing"         # 6 breaths/min sync
    THETA_PULSE = "theta_pulse"     # 5Hz carrier with bilateral modulation


# Sacred frequency presets (Solfeggio scale + others)
EMDR_FREQUENCIES = {
    "trauma_release": 396,       # MI - Liberating guilt/fear
    "change_facilitation": 417,  # FA - Facilitating change
    "dna_repair": 528,           # SOL - DNA repair, miracles
    "connection": 639,           # LA - Relationships
    "sacred_432": 432,           # Verdi tuning
    "gamma_40": 40,              # Gamma brainwave
    "alpha_10": 10,              # Alpha brainwave
    "theta_6": 6,                # Theta brainwave
}


def golden_fade(t: float, fade_in: bool = True) -> float:
    """
    Golden ratio based fade curve using φ exponent.
    
    Args:
        t: Time 0-1
        fade_in: True for fade in, False for fade out
    
    Returns:
        Fade value 0-1
    """
    t = max(0.0, min(1.0, t))
    if fade_in:
        base = (1 - np.cos(t * np.pi)) / 2.0
        return base ** PHI_CONJUGATE
    else:
        base = (1 - np.cos((1 - t) * np.pi)) / 2.0
        return 1.0 - (base ** PHI_CONJUGATE)


class EMDRGenerator:
    """
    EMDR bilateral audio generator.
    
    Generates stereo audio with L/R alternation at specified speed.
    Can be driven by any audio backend (PyAudio, ALSA, etc.)
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # EMDR parameters
        self.bilateral_speed = EMDR_SPEED_DEFAULT  # Hz
        self.carrier_freq = 432.0  # Base frequency
        self.mode = BilateralMode.STANDARD
        self.amplitude = 0.5
        
        # Phase tracking
        self.bilateral_phase = 0.0  # Bilateral alternation phase
        self.carrier_phase = 0.0    # Carrier tone phase
        
        # Fade state
        self.fade_in_duration = 1.0   # seconds
        self.fade_out_duration = 1.0  # seconds
        self.time_elapsed = 0.0
        self.fade_state = "idle"  # "idle", "fading_in", "playing", "fading_out"
        
    def set_parameters(self, speed: Optional[float] = None, 
                      freq: Optional[float] = None,
                      mode: Optional[BilateralMode] = None,
                      amplitude: Optional[float] = None):
        """Update EMDR parameters in real-time"""
        if speed is not None:
            self.bilateral_speed = np.clip(speed, EMDR_SPEED_MIN, EMDR_SPEED_MAX)
        if freq is not None:
            self.carrier_freq = freq
        if mode is not None:
            self.mode = mode
        if amplitude is not None:
            self.amplitude = np.clip(amplitude, 0.0, 1.0)
    
    def start(self):
        """Start playback with fade-in"""
        self.fade_state = "fading_in"
        self.time_elapsed = 0.0
    
    def stop(self):
        """Begin fade-out"""
        if self.fade_state != "idle":
            self.fade_state = "fading_out"
            self.time_elapsed = 0.0
    
    def generate_frame(self, num_frames: int) -> np.ndarray:
        """
        Generate EMDR audio frame.
        
        Args:
            num_frames: Number of frames to generate
        
        Returns:
            np.ndarray of shape (num_frames, 2) - stereo audio
        """
        
        # Time array for this frame
        t = np.arange(num_frames) / self.sample_rate
        dt = num_frames / self.sample_rate
        
        # Generate carrier tone
        carrier_phase_array = self.carrier_phase + 2 * np.pi * self.carrier_freq * t
        carrier = np.sin(carrier_phase_array)
        
        # Update carrier phase
        self.carrier_phase = (self.carrier_phase + 
                             2 * np.pi * self.carrier_freq * dt) % (2 * np.pi)
        
        # Generate bilateral modulation based on mode
        bilateral_phase_array = self.bilateral_phase + 2 * np.pi * self.bilateral_speed * t
        
        if self.mode == BilateralMode.STANDARD:
            # Simple L/R alternation: left when sin > 0, right when sin < 0
            bilateral_mod = np.sin(bilateral_phase_array)
            left_amp = np.maximum(bilateral_mod, 0)  # Positive half
            right_amp = np.maximum(-bilateral_mod, 0)  # Negative half
            
        elif self.mode == BilateralMode.GOLDEN_PHASE:
            # Golden angle (137.5°) phase offset between hemispheres
            left_mod = np.sin(bilateral_phase_array)
            right_mod = np.sin(bilateral_phase_array + GOLDEN_ANGLE_RAD)
            left_amp = (left_mod + 1) / 2  # 0 to 1
            right_amp = (right_mod + 1) / 2
            
        elif self.mode == BilateralMode.BREATHING:
            # Sync to breathing: 6 breaths/min = 0.1 Hz
            # Use sine wave to modulate amplitude smoothly
            breathing_freq = 0.1
            breathing_phase = 2 * np.pi * breathing_freq * t
            breath_cycle = (np.sin(breathing_phase) + 1) / 2  # 0 to 1
            
            # Bilateral alternates with breath
            bilateral_mod = np.sin(bilateral_phase_array)
            left_amp = breath_cycle * np.maximum(bilateral_mod, 0)
            right_amp = breath_cycle * np.maximum(-bilateral_mod, 0)
            
        elif self.mode == BilateralMode.THETA_PULSE:
            # 5Hz theta carrier with bilateral modulation
            theta_carrier = np.sin(2 * np.pi * 5 * t)
            bilateral_mod = np.sin(bilateral_phase_array)
            left_amp = (theta_carrier + 1) / 2 * np.maximum(bilateral_mod, 0)
            right_amp = (theta_carrier + 1) / 2 * np.maximum(-bilateral_mod, 0)
        
        else:
            # Fallback to standard
            bilateral_mod = np.sin(bilateral_phase_array)
            left_amp = np.maximum(bilateral_mod, 0)
            right_amp = np.maximum(-bilateral_mod, 0)
        
        # Update bilateral phase
        self.bilateral_phase = (self.bilateral_phase + 
                               2 * np.pi * self.bilateral_speed * dt) % (2 * np.pi)
        
        # Apply amplitude envelopes
        left_channel = carrier * left_amp
        right_channel = carrier * right_amp
        
        # Apply fade envelope
        if self.fade_state == "fading_in":
            for i in range(num_frames):
                frame_time = self.time_elapsed + i / self.sample_rate
                if frame_time < self.fade_in_duration:
                    fade = golden_fade(frame_time / self.fade_in_duration, fade_in=True)
                    left_channel[i] *= fade
                    right_channel[i] *= fade
                else:
                    self.fade_state = "playing"
                    break
        
        elif self.fade_state == "fading_out":
            for i in range(num_frames):
                frame_time = self.time_elapsed + i / self.sample_rate
                if frame_time < self.fade_out_duration:
                    fade = golden_fade(frame_time / self.fade_out_duration, fade_in=False)
                    left_channel[i] *= fade
                    right_channel[i] *= fade
                else:
                    self.fade_state = "idle"
                    # Zero out remaining frames
                    left_channel[i:] = 0
                    right_channel[i:] = 0
                    break
        
        elif self.fade_state == "idle":
            # Silent
            left_channel[:] = 0
            right_channel[:] = 0
        
        # Apply master amplitude
        left_channel *= self.amplitude
        right_channel *= self.amplitude
        
        # Update time elapsed
        self.time_elapsed += dt
        
        # Stack to stereo
        stereo = np.column_stack([left_channel, right_channel])
        
        return stereo.astype(np.float32)


class EMDRJourney:
    """
    EMDR trauma annealing journey with multiple phases.
    
    Automatically transitions through phases with different frequencies,
    speeds, and modes. Uses Fibonacci-timed progressions.
    """
    
    def __init__(self, generator: EMDRGenerator, program: Dict):
        """
        Args:
            generator: EMDRGenerator instance
            program: Journey program dict with phases
        """
        self.generator = generator
        self.program = program
        self.phases = program["phases"]
        self.total_duration = program["duration_min"] * 60  # seconds
        
        # State tracking
        self.current_phase_index = 0
        self.phase_start_time = 0.0
        self.journey_start_time = 0.0
        self.running = False
        
        # Calculate absolute phase durations
        self.phase_durations = []
        for phase in self.phases:
            duration = phase["duration_pct"] * self.total_duration
            self.phase_durations.append(duration)
    
    def start(self):
        """Start journey"""
        self.current_phase_index = 0
        self.journey_start_time = time.time()
        self.phase_start_time = self.journey_start_time
        self.running = True
        self.generator.start()
        
        # Apply first phase parameters
        self._apply_phase(0)
    
    def stop(self):
        """Stop journey"""
        self.running = False
        self.generator.stop()
    
    def _apply_phase(self, phase_index: int):
        """Apply phase parameters to generator"""
        if phase_index >= len(self.phases):
            return
        
        phase = self.phases[phase_index]
        self.generator.set_parameters(
            speed=phase.get("speed", 1.0),
            freq=phase.get("freq", 432),
            mode=BilateralMode.GOLDEN_PHASE if "phase_mode" in phase else BilateralMode.STANDARD
        )
    
    def update(self):
        """Update journey state (call periodically)"""
        if not self.running:
            return
        
        current_time = time.time()
        elapsed_in_phase = current_time - self.phase_start_time
        
        # Check if phase completed
        if elapsed_in_phase >= self.phase_durations[self.current_phase_index]:
            # Move to next phase
            self.current_phase_index += 1
            
            if self.current_phase_index >= len(self.phases):
                # Journey complete
                self.stop()
                return
            
            # Start next phase
            self.phase_start_time = current_time
            self._apply_phase(self.current_phase_index)
    
    def get_progress(self) -> Dict:
        """Get journey progress info"""
        if not self.running:
            return {"running": False, "progress": 0.0, "phase": None}
        
        current_time = time.time()
        total_elapsed = current_time - self.journey_start_time
        progress = total_elapsed / self.total_duration
        
        phase = self.phases[self.current_phase_index]
        phase_elapsed = current_time - self.phase_start_time
        phase_duration = self.phase_durations[self.current_phase_index]
        phase_progress = phase_elapsed / phase_duration
        
        return {
            "running": True,
            "progress": progress,
            "phase_name": phase["name"],
            "phase_index": self.current_phase_index,
            "phase_progress": phase_progress,
        }


# Preset journey programs
ANNEALING_PROGRAMS = {
    "gentle_release": {
        "name": "🌅 Gentle Release",
        "duration_min": 5,
        "description": "Soft entry into bilateral processing",
        "phases": [
            {"name": "Ground", "duration_pct": 0.20, "freq": 396, "speed": 0.5},
            {"name": "Open", "duration_pct": 0.30, "freq": 417, "speed": 0.8},
            {"name": "Process", "duration_pct": 0.30, "freq": 528, "speed": 1.0},
            {"name": "Integrate", "duration_pct": 0.20, "freq": 639, "speed": 0.6},
        ],
    },
    "deep_processing": {
        "name": "🔥 Deep Processing",
        "duration_min": 10,
        "description": "Intensive bilateral for stubborn patterns",
        "phases": [
            {"name": "Warm-up", "duration_pct": 0.10, "freq": 432, "speed": 0.5},
            {"name": "Activate", "duration_pct": 0.15, "freq": 396, "speed": 0.8},
            {"name": "Intensify", "duration_pct": 0.25, "freq": 417, "speed": 1.2},
            {"name": "Peak", "duration_pct": 0.20, "freq": 528, "speed": 1.5},
            {"name": "Cool-down", "duration_pct": 0.15, "freq": 639, "speed": 0.8},
            {"name": "Integrate", "duration_pct": 0.15, "freq": 852, "speed": 0.5},
        ],
    },
    "hemispheric_sync": {
        "name": "🧠 Hemispheric Sync",
        "duration_min": 8,
        "description": "Bilateral integration with golden phases",
        "phases": [
            {"name": "Left Activate", "duration_pct": 0.25, "freq": 432, "speed": 0.7},
            {"name": "Right Activate", "duration_pct": 0.25, "freq": 432, "speed": 0.7},
            {"name": "Merge", "duration_pct": 0.30, "freq": 528, "speed": 1.0},
            {"name": "Unity", "duration_pct": 0.20, "freq": 639, "speed": 0.5},
        ],
    },
    "golden_spiral": {
        "name": "🌀 Golden Spiral",
        "duration_min": 12,
        "description": "Fibonacci-timed phases, golden transitions",
        "phases": [
            {"name": "Seed", "duration_pct": 0.05, "freq": 174, "speed": 0.5},
            {"name": "Sprout", "duration_pct": 0.05, "freq": 285, "speed": 0.6},
            {"name": "Grow", "duration_pct": 0.10, "freq": 396, "speed": 0.8},
            {"name": "Expand", "duration_pct": 0.15, "freq": 417, "speed": 1.0},
            {"name": "Flower", "duration_pct": 0.25, "freq": 528, "speed": 1.2},
            {"name": "Fruit", "duration_pct": 0.40, "freq": 639, "speed": 0.8},
        ],
    },
}


# Example usage
if __name__ == "__main__":
    print("🧠 EMDR Bilateral Audio Generator Test\n")
    
    # Create generator
    generator = EMDRGenerator(sample_rate=44100)
    generator.set_parameters(
        speed=1.0,  # 1 Hz bilateral
        freq=432,   # Sacred 432Hz
        mode=BilateralMode.GOLDEN_PHASE,
        amplitude=0.5
    )
    
    # Generate 2 seconds of audio
    print("Generating 2 seconds of bilateral audio...")
    frames = []
    total_frames = 44100 * 2  # 2 seconds
    buffer_size = 2048
    
    generator.start()
    while len(frames) * buffer_size < total_frames:
        frame = generator.generate_frame(buffer_size)
        frames.append(frame)
    
    audio = np.vstack(frames)
    print(f"✓ Generated {audio.shape} audio (L/R stereo)")
    
    # Test journey
    print("\n🌅 Testing Gentle Release journey...")
    journey = EMDRJourney(generator, ANNEALING_PROGRAMS["gentle_release"])
    journey.start()
    
    # Simulate updates
    for i in range(10):
        journey.update()
        progress = journey.get_progress()
        if progress["running"]:
            print(f"  Phase: {progress['phase_name']} ({progress['phase_progress']*100:.1f}%)")
        time.sleep(0.5)
    
    journey.stop()
    print("✓ Journey test complete")
