#!/usr/bin/env python3
"""
Test Chakra Sunrise Journey - Debug audio glitches
Simulates GUI behavior with 108Hz base, 2 minute duration
"""

import numpy as np
import time
import threading
import sys

# Constants
SAMPLE_RATE = 44100
PHI = 1.618033988749895
PHI_CONJUGATE = 0.618033988749895

print("=" * 70)
print("CHAKRA SUNRISE DEBUG TEST")
print("Base: 108 Hz | Duration: 2 minutes")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE (simplified from golden_studio.py)
# ══════════════════════════════════════════════════════════════════════════════

class DebugAudioEngine:
    """Audio engine with detailed logging for debugging clicks"""
    
    def __init__(self):
        self.playing = False
        self.mode = "spectral"
        self.amplitude = 0.8
        
        # Spectral mode
        self.spectral_frequencies = []
        self.spectral_phases = []
        self.stereo_positions = []
        
        # Smooth interpolation
        self._target_amplitudes = []
        self._current_amplitudes = []
        self._target_positions = []
        self._current_positions = []
        
        self.lock = threading.Lock()
        
        # Debug tracking
        self.frame_count = 0
        self.last_output_left = 0.0
        self.last_output_right = 0.0
        self.max_discontinuity_left = 0.0
        self.max_discontinuity_right = 0.0
        self.discontinuity_frames = []
        
    def set_spectral_params(self, frequencies, amplitudes, phases=None, positions=None):
        """Set target parameters (will be smoothly interpolated)"""
        with self.lock:
            self.spectral_frequencies = list(zip(
                frequencies,
                amplitudes,
                phases or [0.0] * len(frequencies)
            ))
            self.stereo_positions = positions or [0.0] * len(frequencies)
            
            self._target_amplitudes = list(amplitudes)
            self._target_positions = list(positions) if positions else [0.0] * len(frequencies)
            
            if len(self._current_amplitudes) != len(frequencies):
                self._current_amplitudes = list(amplitudes)
                self._current_positions = list(self._target_positions)
            
            if len(self.spectral_phases) != len(frequencies):
                self.spectral_phases = [0.0] * len(frequencies)
    
    def generate_chunk(self, frame_count=1024):
        """Generate audio chunk with debugging"""
        with self.lock:
            freq_data = list(self.spectral_frequencies)
            target_amps = list(self._target_amplitudes) if self._target_amplitudes else []
            target_pans = list(self._target_positions) if self._target_positions else []
            current_amps = list(self._current_amplitudes) if self._current_amplitudes else []
            current_pans = list(self._current_positions) if self._current_positions else []
            master_amp = self.amplitude
        
        if not freq_data:
            return np.zeros(frame_count * 2, dtype=np.float32)
        
        output_left = np.zeros(frame_count, dtype=np.float32)
        output_right = np.zeros(frame_count, dtype=np.float32)
        
        # Ensure arrays are right size
        while len(self.spectral_phases) < len(freq_data):
            self.spectral_phases.append(0.0)
        while len(current_amps) < len(freq_data):
            current_amps.append(0.0)
        while len(current_pans) < len(freq_data):
            current_pans.append(0.0)
        while len(target_amps) < len(freq_data):
            target_amps.append(0.0)
        while len(target_pans) < len(freq_data):
            target_pans.append(0.0)
        
        # Smoothing coefficients - ULTRA SLOW for click-free audio
        # At 44100 Hz, we need ~150-250ms transitions
        amp_smooth = 0.00012   # ~190ms to reach 63%
        pan_smooth = 0.00008   # ~280ms to reach 63%
        
        for idx, (freq, _, phase_off) in enumerate(freq_data):
            phase_inc = 2 * np.pi * freq / SAMPLE_RATE
            
            for i in range(frame_count):
                # Smooth interpolation
                amp_diff = target_amps[idx] - current_amps[idx]
                current_amps[idx] += amp_diff * amp_smooth
                
                pan_diff = target_pans[idx] - current_pans[idx]
                current_pans[idx] += pan_diff * pan_smooth
                
                freq_amp = current_amps[idx]
                pan = current_pans[idx]
                
                # Pan law with minimum
                pan_angle = (pan + 1) * np.pi / 4
                left_gain = max(0.02, np.cos(pan_angle))
                right_gain = max(0.02, np.sin(pan_angle))
                
                sample = freq_amp * np.sin(self.spectral_phases[idx] + phase_off)
                output_left[i] += sample * left_gain
                output_right[i] += sample * right_gain
                
                self.spectral_phases[idx] += phase_inc
                if self.spectral_phases[idx] > 2 * np.pi:
                    self.spectral_phases[idx] -= 2 * np.pi
        
        # Store smoothed values
        with self.lock:
            self._current_amplitudes = current_amps
            self._current_positions = current_pans
        
        # Apply master amplitude - NO dynamic limiting (causes clicks)
        output_left *= master_amp
        output_right *= master_amp
        
        # Simple hard clip
        np.clip(output_left, -1.0, 1.0, out=output_left)
        np.clip(output_right, -1.0, 1.0, out=output_right)
        
        # ═══════════════════════════════════════════════════════════════════
        # DEBUG: Check for discontinuities
        # ═══════════════════════════════════════════════════════════════════
        
        # Check first sample vs last sample of previous chunk
        if self.frame_count > 0:
            disc_left = abs(output_left[0] - self.last_output_left)
            disc_right = abs(output_right[0] - self.last_output_right)
            
            # Threshold for "click" - a sudden jump
            CLICK_THRESHOLD = 0.05
            
            if disc_left > CLICK_THRESHOLD or disc_right > CLICK_THRESHOLD:
                self.discontinuity_frames.append({
                    'frame': self.frame_count,
                    'time': self.frame_count / SAMPLE_RATE,
                    'disc_left': disc_left,
                    'disc_right': disc_right,
                    'prev_left': self.last_output_left,
                    'prev_right': self.last_output_right,
                    'curr_left': output_left[0],
                    'curr_right': output_right[0],
                    'current_amps': list(current_amps),
                    'current_pans': list(current_pans),
                    'target_amps': list(target_amps),
                    'target_pans': list(target_pans),
                })
            
            self.max_discontinuity_left = max(self.max_discontinuity_left, disc_left)
            self.max_discontinuity_right = max(self.max_discontinuity_right, disc_right)
        
        # Store last samples
        self.last_output_left = output_left[-1]
        self.last_output_right = output_right[-1]
        self.frame_count += frame_count
        
        # Also check for internal discontinuities within the chunk
        for i in range(1, frame_count):
            disc_l = abs(output_left[i] - output_left[i-1])
            disc_r = abs(output_right[i] - output_right[i-1])
            
            # Very high threshold for within-chunk (only catches extreme jumps)
            if disc_l > 0.1 or disc_r > 0.1:
                self.discontinuity_frames.append({
                    'frame': self.frame_count - frame_count + i,
                    'time': (self.frame_count - frame_count + i) / SAMPLE_RATE,
                    'disc_left': disc_l,
                    'disc_right': disc_r,
                    'type': 'internal',
                    'sample_index': i,
                })
        
        return output_left, output_right


# ══════════════════════════════════════════════════════════════════════════════
# CHAKRA SUNRISE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def mm_to_pan(position_mm):
    """Convert body position in mm to pan value (-1 to +1)"""
    return (position_mm / 975.0) - 1.0

def golden_fade(t, fade_in=True):
    """
    Golden ratio based fade curve using φ exponent.
    Creates a smooth S-curve that follows golden proportions.
    """
    t = max(0.0, min(1.0, t))  # Clamp to 0-1
    
    if fade_in:
        # Raised cosine with golden exponent
        base = (1 - np.cos(t * np.pi)) / 2.0
        return base ** PHI_CONJUGATE  # ≈ 0.618 exponent
    else:
        base = (1 - np.cos((1 - t) * np.pi)) / 2.0
        return 1.0 - (base ** PHI_CONJUGATE)


def simulate_chakra_journey(base_freq=108.0, duration=120.0, log_interval=5.0):
    """
    Simulate the Chakra Sunrise journey exactly as the GUI does
    """
    print(f"\n{'='*70}")
    print(f"SIMULATING CHAKRA SUNRISE")
    print(f"Base frequency: {base_freq} Hz")
    print(f"Duration: {duration} seconds")
    print(f"{'='*70}\n")
    
    # Calculate frequencies
    freq_fourth = base_freq * 4 / 3   # Perfect 4th = 144 Hz
    freq_root = base_freq             # Root = 108 Hz
    freq_octave = base_freq * 2       # Octave = 216 Hz
    
    print(f"Frequencies:")
    print(f"  Perfect 4th: {freq_fourth:.1f} Hz")
    print(f"  Root:        {freq_root:.1f} Hz")
    print(f"  Octave:      {freq_octave:.1f} Hz")
    
    # Body positions
    HEAD_MM = 0.0
    SOLAR_PLEXUS_MM = 600.0
    SACRAL_MM = 800.0
    FEET_MM = 1750.0
    
    PAN_HEAD = mm_to_pan(HEAD_MM)
    PAN_SOLAR_PLEXUS = mm_to_pan(SOLAR_PLEXUS_MM)
    PAN_SACRAL = mm_to_pan(SACRAL_MM)
    PAN_FEET = mm_to_pan(FEET_MM)
    
    print(f"\nPan positions:")
    print(f"  HEAD:          {PAN_HEAD:.3f}")
    print(f"  SOLAR_PLEXUS:  {PAN_SOLAR_PLEXUS:.3f}")
    print(f"  SACRAL:        {PAN_SACRAL:.3f}")
    print(f"  FEET:          {PAN_FEET:.3f}")
    
    # Create audio engine
    engine = DebugAudioEngine()
    
    # Initial state
    fourth_amp = 0.0
    fourth_pan = PAN_SOLAR_PLEXUS
    root_amp = 0.0
    root_pan = PAN_FEET
    octave_amp = 0.0
    octave_pan = PAN_HEAD
    
    # Set initial params
    engine.set_spectral_params(
        [freq_fourth, freq_root, freq_octave],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [PAN_SOLAR_PLEXUS, PAN_FEET, PAN_HEAD]
    )
    
    # Simulation parameters
    # GUI updates every 30ms, audio generates 1024 samples at a time
    gui_update_interval = 0.030  # 30ms
    audio_chunk_size = 1024
    audio_chunk_duration = audio_chunk_size / SAMPLE_RATE
    
    # We'll simulate time steps
    time_step = 0.001  # 1ms steps for smooth simulation
    
    start_time = time.time()
    sim_time = 0.0
    last_gui_update = 0.0
    last_audio_gen = 0.0
    last_log = 0.0
    
    print(f"\n{'='*70}")
    print("STARTING SIMULATION...")
    print(f"{'='*70}\n")
    
    phase_names = ["Phase 1: 4th emerging", "Phase 2: Root from feet", 
                   "Phase 3: Root through sacral", "Phase 4: Octave descending",
                   "Phase 5: CONVERGENCE"]
    current_phase = -1
    
    # Golden phase boundaries (symmetric: 1:φ:φ:φ:1)
    GOLDEN_UNIT = 1.0 / (2.0 + 3.0 * PHI)  # ≈ 0.146
    GOLDEN_PHI_UNIT = PHI * GOLDEN_UNIT     # ≈ 0.236
    
    P1_END = GOLDEN_UNIT                                    # ≈ 0.146
    P2_END = GOLDEN_UNIT + GOLDEN_PHI_UNIT                  # ≈ 0.382
    P3_END = GOLDEN_UNIT + 2 * GOLDEN_PHI_UNIT              # ≈ 0.618
    P4_END = GOLDEN_UNIT + 3 * GOLDEN_PHI_UNIT              # ≈ 0.854
    
    while sim_time < duration:
        progress = sim_time / duration
        
        # Determine current phase (golden boundaries)
        if progress < P1_END:
            new_phase = 0
        elif progress < P2_END:
            new_phase = 1
        elif progress < P3_END:
            new_phase = 2
        elif progress < P4_END:
            new_phase = 3
        else:
            new_phase = 4
        
        if new_phase != current_phase:
            current_phase = new_phase
            print(f"[{sim_time:6.1f}s] === {phase_names[current_phase]} ===")
        
        # ═══════════════════════════════════════════════════════════════════
        # GUI UPDATE (every 30ms) - updates target amplitudes/positions
        # ═══════════════════════════════════════════════════════════════════
        if sim_time - last_gui_update >= gui_update_interval:
            last_gui_update = sim_time
            
            # Calculate new targets based on phase (golden boundaries)
            if progress < P1_END:
                phase_progress = progress / P1_END
                fourth_amp = golden_fade(phase_progress, fade_in=True)
                fourth_pan = PAN_SOLAR_PLEXUS
                root_amp = 0.0
                root_pan = PAN_FEET
                octave_amp = 0.0
                octave_pan = PAN_HEAD
                
            elif progress < P2_END:
                phase_progress = (progress - P1_END) / (P2_END - P1_END)
                fourth_amp = 1.0
                fourth_pan = PAN_SOLAR_PLEXUS
                root_amp = golden_fade(phase_progress, fade_in=True)
                pan_progress = golden_fade(phase_progress, fade_in=True)
                root_pan = PAN_FEET + (PAN_SACRAL - PAN_FEET) * pan_progress
                octave_amp = 0.0
                octave_pan = PAN_HEAD
                
            elif progress < P3_END:
                phase_progress = (progress - P2_END) / (P3_END - P2_END)
                fourth_amp = 1.0
                fourth_pan = PAN_SOLAR_PLEXUS
                root_amp = 1.0
                pan_progress = golden_fade(phase_progress, fade_in=True)
                root_pan = PAN_SACRAL + (PAN_SOLAR_PLEXUS - PAN_SACRAL) * pan_progress
                octave_amp = 0.0
                octave_pan = PAN_HEAD
                
            elif progress < P4_END:
                phase_progress = (progress - P3_END) / (P4_END - P3_END)
                fourth_amp = 1.0
                fourth_pan = PAN_SOLAR_PLEXUS
                root_amp = 1.0
                root_pan = PAN_SOLAR_PLEXUS
                octave_amp = golden_fade(phase_progress, fade_in=True)
                pan_progress = golden_fade(phase_progress, fade_in=True)
                octave_pan = PAN_HEAD + (PAN_SOLAR_PLEXUS - PAN_HEAD) * pan_progress
                
            else:
                fourth_amp = 1.0
                root_amp = 1.0
                octave_amp = 1.0
                fourth_pan = PAN_SOLAR_PLEXUS
                root_pan = PAN_SOLAR_PLEXUS
                octave_pan = PAN_SOLAR_PLEXUS
            
            # Update audio engine (this is what GUI does)
            engine.set_spectral_params(
                [freq_fourth, freq_root, freq_octave],
                [fourth_amp, root_amp, octave_amp],
                [0.0, 0.0, 0.0],
                [fourth_pan, root_pan, octave_pan]
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # AUDIO GENERATION (simulates audio callback)
        # ═══════════════════════════════════════════════════════════════════
        if sim_time - last_audio_gen >= audio_chunk_duration:
            last_audio_gen = sim_time
            left, right = engine.generate_chunk(audio_chunk_size)
        
        # ═══════════════════════════════════════════════════════════════════
        # LOGGING
        # ═══════════════════════════════════════════════════════════════════
        if sim_time - last_log >= log_interval:
            last_log = sim_time
            print(f"[{sim_time:6.1f}s] Progress: {progress*100:5.1f}% | "
                  f"Amps: 4th={fourth_amp:.2f} root={root_amp:.2f} oct={octave_amp:.2f} | "
                  f"Pans: 4th={fourth_pan:+.2f} root={root_pan:+.2f} oct={octave_pan:+.2f}")
            
            if engine.discontinuity_frames:
                print(f"         ⚠️  {len(engine.discontinuity_frames)} discontinuities detected!")
        
        sim_time += time_step
    
    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE - ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total frames generated: {engine.frame_count}")
    print(f"Max discontinuity LEFT:  {engine.max_discontinuity_left:.6f}")
    print(f"Max discontinuity RIGHT: {engine.max_discontinuity_right:.6f}")
    print(f"Total discontinuities detected: {len(engine.discontinuity_frames)}")
    
    if engine.discontinuity_frames:
        print(f"\n{'='*70}")
        print("DISCONTINUITY DETAILS (potential clicks)")
        print(f"{'='*70}\n")
        
        for i, disc in enumerate(engine.discontinuity_frames[:20]):  # First 20
            print(f"Discontinuity #{i+1} at {disc['time']:.3f}s (frame {disc['frame']}):")
            print(f"  Left:  prev={disc.get('prev_left', 0):.4f} → curr={disc.get('curr_left', 0):.4f} (jump={disc['disc_left']:.4f})")
            print(f"  Right: prev={disc.get('prev_right', 0):.4f} → curr={disc.get('curr_right', 0):.4f} (jump={disc['disc_right']:.4f})")
            if 'current_amps' in disc:
                print(f"  Current amps: {disc['current_amps']}")
                print(f"  Target amps:  {disc['target_amps']}")
                print(f"  Current pans: {disc['current_pans']}")
                print(f"  Target pans:  {disc['target_pans']}")
            print()
        
        if len(engine.discontinuity_frames) > 20:
            print(f"... and {len(engine.discontinuity_frames) - 20} more")
        
        # Analyze when clicks happen
        print(f"\n{'='*70}")
        print("TIMING ANALYSIS")
        print(f"{'='*70}\n")
        
        # Group by phase
        phase_boundaries = [0.20 * duration, 0.40 * duration, 0.60 * duration, 0.80 * duration]
        phase_counts = [0, 0, 0, 0, 0]
        
        for disc in engine.discontinuity_frames:
            t = disc['time']
            if t < phase_boundaries[0]:
                phase_counts[0] += 1
            elif t < phase_boundaries[1]:
                phase_counts[1] += 1
            elif t < phase_boundaries[2]:
                phase_counts[2] += 1
            elif t < phase_boundaries[3]:
                phase_counts[3] += 1
            else:
                phase_counts[4] += 1
        
        for i, (name, count) in enumerate(zip(phase_names, phase_counts)):
            print(f"  {name}: {count} discontinuities")
    
    else:
        print("\n✅ NO DISCONTINUITIES DETECTED - Audio should be click-free!")
    
    return engine


if __name__ == "__main__":
    engine = simulate_chakra_journey(base_freq=108.0, duration=120.0, log_interval=10.0)
