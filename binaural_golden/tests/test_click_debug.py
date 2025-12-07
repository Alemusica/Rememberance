#!/usr/bin/env python3
"""
Debug script to find the source of clicking in chakra journey.
Tests audio generation at the exact moment when octave fades in at HEAD position.
"""

import numpy as np
import pyaudio
import time
import threading

SAMPLE_RATE = 44100

class ClickDebugger:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.playing = False
        self.lock = threading.Lock()
        
        # Three frequencies like chakra journey
        self.freq_fourth = 174.0    # Perfect 4th
        self.freq_root = 130.5      # Root  
        self.freq_octave = 261.0    # Octave
        
        # Amplitude and pan targets (what the journey sets)
        self._target_amps = [1.0, 1.0, 0.0]  # 4th full, root full, octave starting
        self._target_pans = [-0.38, -0.38, -1.0]  # solar plexus, solar plexus, HEAD
        
        # Current smoothed values
        self._current_amps = [1.0, 1.0, 0.0]
        self._current_pans = [-0.38, -0.38, -1.0]
        
        # Phase accumulators
        self.phases = [0.0, 0.0, 0.0]
        
        # Debug: track discontinuities
        self.last_left = 0.0
        self.last_right = 0.0
        self.click_count = 0
        self.max_jump = 0.0
        
    def _generate_chunk(self, frame_count):
        """Generate audio chunk with click detection"""
        output_left = np.zeros(frame_count, dtype=np.float32)
        output_right = np.zeros(frame_count, dtype=np.float32)
        
        frequencies = [self.freq_fourth, self.freq_root, self.freq_octave]
        
        # Smoothing coefficients
        amp_smooth = 0.0008
        pan_smooth = 0.0003
        
        for idx, freq in enumerate(frequencies):
            phase_inc = 2 * np.pi * freq / SAMPLE_RATE
            
            for i in range(frame_count):
                # Smooth interpolation
                amp_diff = self._target_amps[idx] - self._current_amps[idx]
                self._current_amps[idx] += amp_diff * amp_smooth
                
                pan_diff = self._target_pans[idx] - self._current_pans[idx]
                self._current_pans[idx] += pan_diff * pan_smooth
                
                freq_amp = self._current_amps[idx]
                pan = self._current_pans[idx]
                
                # Pan law
                pan_angle = (pan + 1) * np.pi / 4
                left_gain = np.cos(pan_angle)
                right_gain = np.sin(pan_angle)
                
                sample = freq_amp * np.sin(self.phases[idx])
                output_left[i] += sample * left_gain
                output_right[i] += sample * right_gain
                
                self.phases[idx] += phase_inc
                if self.phases[idx] > 2 * np.pi:
                    self.phases[idx] -= 2 * np.pi
        
        # Check for clicks (large sample-to-sample jumps)
        if frame_count > 0:
            # Check first sample against last from previous buffer
            jump_left = abs(output_left[0] - self.last_left)
            jump_right = abs(output_right[0] - self.last_right)
            max_jump = max(jump_left, jump_right)
            
            if max_jump > 0.1:  # Threshold for audible click
                self.click_count += 1
                if max_jump > self.max_jump:
                    self.max_jump = max_jump
                    print(f"⚠️ CLICK DETECTED! Jump: {max_jump:.4f} (L: {self.last_left:.4f}→{output_left[0]:.4f})")
            
            self.last_left = output_left[-1]
            self.last_right = output_right[-1]
        
        # Soft limiting
        peak = max(np.max(np.abs(output_left)), np.max(np.abs(output_right)), 0.001)
        if peak > 0.95:
            ratio = 0.95 + (peak - 0.95) * 0.3
            output_left *= (0.95 / peak) * (ratio / 0.95)
            output_right *= (0.95 / peak) * (ratio / 0.95)
        
        output_left *= 0.5  # Master volume
        output_right *= 0.5
        
        # Interleave
        output = np.empty(frame_count * 2, dtype=np.float32)
        output[0::2] = output_left
        output[1::2] = output_right
        
        return output.tobytes()
    
    def _callback(self, in_data, frame_count, time_info, status):
        if not self.playing:
            return (None, pyaudio.paComplete)
        data = self._generate_chunk(frame_count)
        return (data, pyaudio.paContinue)
    
    def test_octave_fade_in(self):
        """Simulate phase 4: octave fading in at HEAD"""
        print("\n" + "="*60)
        print("TEST: Simulating octave fade-in at HEAD position")
        print("This is where the click occurs in the chakra journey")
        print("="*60 + "\n")
        
        # Start with phase 3 end state (4th + root at solar plexus, octave silent at head)
        self._current_amps = [1.0, 1.0, 0.0]
        self._current_pans = [-0.38, -0.38, -1.0]
        self._target_amps = [1.0, 1.0, 0.0]
        self._target_pans = [-0.38, -0.38, -1.0]
        
        # Reset click detection
        self.click_count = 0
        self.max_jump = 0.0
        self.last_left = 0.0
        self.last_right = 0.0
        
        # Open stream
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=1024,
            stream_callback=self._callback
        )
        
        self.playing = True
        stream.start_stream()
        
        print("Phase 3 state: 4th + root at solar plexus, octave silent")
        time.sleep(1)
        
        # Now simulate phase 4: fade in octave at HEAD
        print("\n>>> Starting octave fade-in at HEAD (-1.0 pan)...")
        
        # Gradual fade in over 5 seconds (like the actual journey)
        for i in range(50):
            progress = i / 50.0
            # Golden fade approximation
            golden_progress = progress * progress * (3 - 2 * progress)  # smoothstep
            
            self._target_amps[2] = golden_progress  # Octave amplitude
            # Pan stays at HEAD initially, then moves
            pan_progress = golden_progress
            self._target_pans[2] = -1.0 + (-0.38 - (-1.0)) * pan_progress
            
            time.sleep(0.1)
            print(f"  Octave amp: {self._current_amps[2]:.3f} → {self._target_amps[2]:.3f}, pan: {self._current_pans[2]:.3f}")
        
        print("\nOctave fully faded in, holding...")
        time.sleep(2)
        
        self.playing = False
        stream.stop_stream()
        stream.close()
        
        print(f"\n{'='*60}")
        print(f"TEST COMPLETE")
        print(f"Clicks detected: {self.click_count}")
        print(f"Max jump: {self.max_jump:.4f}")
        print(f"{'='*60}\n")
        
        return self.click_count
    
    def test_buffer_boundary(self):
        """Test specifically for buffer boundary discontinuities"""
        print("\n" + "="*60)
        print("TEST: Buffer boundary discontinuity check")
        print("="*60 + "\n")
        
        self._current_amps = [1.0, 1.0, 1.0]
        self._current_pans = [-0.38, -0.38, -1.0]
        self._target_amps = [1.0, 1.0, 1.0]
        self._target_pans = [-0.38, -0.38, -1.0]
        
        # Generate multiple consecutive buffers and check continuity
        buffer_sizes = [256, 512, 1024, 2048]
        
        for buf_size in buffer_sizes:
            self.last_left = 0.0
            self.last_right = 0.0
            self.click_count = 0
            self.max_jump = 0.0
            self.phases = [0.0, 0.0, 0.0]
            
            # Generate 100 consecutive buffers
            for _ in range(100):
                self._generate_chunk(buf_size)
            
            print(f"Buffer size {buf_size}: {self.click_count} clicks, max jump: {self.max_jump:.6f}")
        
        return self.click_count
    
    def test_pan_at_extremes(self):
        """Test panning at extreme positions (-1.0 HEAD)"""
        print("\n" + "="*60)
        print("TEST: Pan at extreme positions (HEAD = -1.0)")
        print("="*60 + "\n")
        
        # Test what happens when pan is exactly -1.0
        pan = -1.0
        pan_angle = (pan + 1) * np.pi / 4  # = 0
        left_gain = np.cos(pan_angle)   # = 1.0
        right_gain = np.sin(pan_angle)  # = 0.0
        
        print(f"Pan = -1.0 (HEAD):")
        print(f"  pan_angle = {pan_angle:.4f} rad ({np.degrees(pan_angle):.1f}°)")
        print(f"  left_gain = {left_gain:.4f}")
        print(f"  right_gain = {right_gain:.4f}")
        
        # When right_gain is 0 and we're adding samples, any discontinuity
        # in the left channel would be very audible
        
        # Test a small pan change from -1.0
        pan2 = -0.99
        pan_angle2 = (pan2 + 1) * np.pi / 4
        left_gain2 = np.cos(pan_angle2)
        right_gain2 = np.sin(pan_angle2)
        
        print(f"\nPan = -0.99:")
        print(f"  left_gain = {left_gain2:.4f} (change: {left_gain2 - left_gain:.6f})")
        print(f"  right_gain = {right_gain2:.4f} (change: {right_gain2 - right_gain:.6f})")
        
        # The issue: when octave fades in at pan=-1.0, right channel is 0
        # but as soon as pan changes even slightly, right channel suddenly has signal
        print(f"\n⚠️ POTENTIAL ISSUE: Right channel jumps from 0 to {right_gain2:.4f}")
        print("   If octave amp is 1.0, this creates a sudden +{:.4f} jump!".format(right_gain2))
        
        return right_gain2
    
    def cleanup(self):
        self.p.terminate()


if __name__ == "__main__":
    debugger = ClickDebugger()
    
    try:
        # Test 1: Pan at extremes (mathematical check)
        debugger.test_pan_at_extremes()
        
        # Test 2: Buffer boundary check
        debugger.test_buffer_boundary()
        
        # Test 3: Actual octave fade-in simulation
        print("\nPress Enter to start audio test (octave fade-in)...")
        input()
        debugger.test_octave_fade_in()
        
    finally:
        debugger.cleanup()
