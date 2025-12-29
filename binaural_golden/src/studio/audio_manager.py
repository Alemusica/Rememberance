"""
Audio Manager - Real-time audio engine for Golden Sound Studio.

Extracted from golden_studio.py for Pi5 deployment optimization.
Provides callback-based audio with real-time parameter updates.
"""

import numpy as np
import threading
from typing import Optional

# PyAudio
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Import constants
from golden_constants import (
    PHI, PHI_CONJUGATE, SAMPLE_RATE
)


class AudioEngine:
    """
    Callback-based audio engine with REAL-TIME parameter updates.
    
    Changes to parameters (frequency, phase, amplitude, etc.) are applied
    IMMEDIATELY without stopping playback - no glitches, smooth transitions.
    """
    
    def __init__(self):
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.playing = False
        
        # === BINAURAL MODE parameters ===
        self.mode = "binaural"  # "binaural", "spectral", "molecular"
        
        # Binaural parameters (real-time updateable)
        self.freq_left = 432.0
        self.freq_right = 440.0
        self.phase_offset = np.radians(137.5)  # Phase angle in radians
        self.amplitude = 0.7
        self.waveform_mode = "sine"
        self.mono_mix = False  # If True, outputs (L+R)/2 to both channels (TEST mode)
        
        # Phase accumulators (continuous across callbacks)
        self.phase_left = 0.0
        self.phase_right = 0.0
        
        # === SPECTRAL/MOLECULAR MODE parameters ===
        self.spectral_frequencies = []  # List of (freq, amplitude, phase)
        self.spectral_phases = []  # Phase accumulators for each frequency
        self.stereo_positions = []  # -1 to 1 for panning
        
        # === SMOOTH PARAMETER INTERPOLATION ===
        # These track current values that smoothly approach targets
        self._target_amplitudes = []   # Target amplitude for each frequency
        self._current_amplitudes = []  # Current (smoothed) amplitude
        self._target_positions = []    # Target pan position for each frequency
        self._current_positions = []   # Current (smoothed) pan position
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Devices
        self.devices = []
        self.selected_device = None
        self._scan_devices()
    
    def _scan_devices(self):
        """Scan for audio output devices"""
        if not HAS_PYAUDIO:
            print("⚠️ PyAudio not available")
            return
        try:
            pa = pyaudio.PyAudio()
            self.devices = []
            count = pa.get_device_count()
            print(f"📡 Scanning {count} audio devices...")
            for i in range(count):
                try:
                    info = pa.get_device_info_by_index(i)
                    if info['maxOutputChannels'] > 0:
                        self.devices.append({
                            'index': i,
                            'name': info['name'],
                            'channels': info['maxOutputChannels'],
                        })
                        print(f"  ✓ {i}: {info['name']} ({info['maxOutputChannels']} ch)")
                except Exception as e:
                    print(f"  ✗ Device {i}: {e}")
            pa.terminate()
            print(f"📡 Found {len(self.devices)} output devices")
        except Exception as e:
            print(f"❌ Error scanning devices: {e}")
            import traceback
            traceback.print_exc()
    
    def get_device_names(self):
        return [d['name'] for d in self.devices]
    
    def set_device(self, idx: int):
        if 0 <= idx < len(self.devices):
            self.selected_device = self.devices[idx]['index']
    
    # ══════════════════════════════════════════════════════════════════════════
    # REAL-TIME PARAMETER SETTERS (call these while audio is playing!)
    # ══════════════════════════════════════════════════════════════════════════
    
    def set_binaural_params(self, freq_left: float, freq_right: float, 
                            phase_angle_deg: float, amplitude: float, 
                            waveform: str = "sine"):
        """Update binaural parameters in real-time (no glitches)"""
        with self.lock:
            self.freq_left = freq_left
            self.freq_right = freq_right
            self.phase_offset = np.radians(phase_angle_deg)
            self.amplitude = amplitude
            self.waveform_mode = waveform
    
    def set_frequencies(self, freq_left: float, freq_right: float):
        """Update frequencies only"""
        with self.lock:
            self.freq_left = freq_left
            self.freq_right = freq_right
    
    def set_phase_angle(self, angle_deg: float):
        """Update phase angle only"""
        with self.lock:
            self.phase_offset = np.radians(angle_deg)
    
    def set_amplitude(self, amp: float):
        """Update amplitude only"""
        with self.lock:
            self.amplitude = max(0.0, min(1.0, amp))
    
    def set_waveform(self, waveform: str):
        """Update waveform mode"""
        with self.lock:
            self.waveform_mode = waveform
    
    def set_mono_mix(self, enabled: bool):
        """Enable/disable mono mix mode (for phase cancellation testing)"""
        with self.lock:
            self.mono_mix = enabled
    
    def set_spectral_params(self, frequencies: list, amplitudes: list, 
                            phases: list = None, positions: list = None):
        """
        Update spectral/molecular parameters in real-time.
        
        Uses smooth interpolation to prevent clicks - targets are set here,
        actual values smoothly approach targets in the audio callback.
        
        frequencies: list of Hz values
        amplitudes: list of amplitude values [0,1]
        phases: optional list of phase offsets (radians)
        positions: optional list of stereo positions [-1, 1]
        """
        with self.lock:
            self.spectral_frequencies = list(zip(
                frequencies, 
                amplitudes,
                phases or [0.0] * len(frequencies)
            ))
            self.stereo_positions = positions or [0.0] * len(frequencies)
            
            # Set TARGET values for smooth interpolation
            self._target_amplitudes = list(amplitudes)
            self._target_positions = list(positions) if positions else [0.0] * len(frequencies)
            
            # Initialize current values if not set (first call)
            if len(self._current_amplitudes) != len(frequencies):
                self._current_amplitudes = list(amplitudes)
                self._current_positions = list(self._target_positions)
            
            # Initialize phase accumulators if needed
            if len(self.spectral_phases) != len(frequencies):
                self.spectral_phases = [0.0] * len(frequencies)
    
    # ══════════════════════════════════════════════════════════════════════════
    # GOLDEN WAVE GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _golden_wave_vectorized(self, phases: np.ndarray, reversed: bool = True) -> np.ndarray:
        """Generate golden wave samples (vectorized)"""
        theta = phases % (2 * np.pi)
        if reversed:
            theta = 2 * np.pi - theta
        
        t = theta / (2 * np.pi)
        rise = PHI_CONJUGATE
        
        result = np.zeros_like(t)
        mask_rise = t < rise
        result[mask_rise] = np.sin(np.pi * t[mask_rise] / rise / 2)
        result[~mask_rise] = np.cos(np.pi * (t[~mask_rise] - rise) / (1 - rise) / 2)
        return result
    
    def _golden_wave_sample(self, phase: float, reversed: bool = True) -> float:
        """Generate single golden wave sample (for backwards compatibility)"""
        theta = phase % (2 * np.pi)
        if reversed:
            theta = 2 * np.pi - theta
        
        t = theta / (2 * np.pi)
        rise = PHI_CONJUGATE
        
        if t < rise:
            return np.sin(np.pi * t / rise / 2)
        else:
            return np.cos(np.pi * (t - rise) / (1 - rise) / 2)
    
    # ══════════════════════════════════════════════════════════════════════════
    # AUDIO GENERATION (callback) - VECTORIZED
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_binaural_chunk(self, frame_count: int) -> bytes:
        """Generate binaural stereo chunk - VECTORIZED VERSION"""
        with self.lock:
            freq_l = self.freq_left
            freq_r = self.freq_right
            phase_off = self.phase_offset
            amp = self.amplitude
            waveform = self.waveform_mode
            mono = self.mono_mix
        
        # Phase increments
        phase_inc_left = 2 * np.pi * freq_l / SAMPLE_RATE
        phase_inc_right = 2 * np.pi * freq_r / SAMPLE_RATE
        
        # Generate phase arrays for the entire buffer (vectorized)
        t = np.arange(frame_count, dtype=np.float32)
        phases_left = self.phase_left + phase_inc_left * t
        phases_right = self.phase_right + phase_off + phase_inc_right * t
        
        # Generate waveforms (vectorized)
        if waveform == "sine":
            left_samples = amp * np.sin(phases_left)
            right_samples = amp * np.sin(phases_right)
        elif waveform == "golden":
            left_samples = amp * self._golden_wave_vectorized(phases_left, reversed=False)
            right_samples = amp * self._golden_wave_vectorized(phases_right, reversed=False)
        else:  # golden_reversed
            left_samples = amp * self._golden_wave_vectorized(phases_left, reversed=True)
            right_samples = amp * self._golden_wave_vectorized(phases_right, reversed=True)
        
        # Update phase accumulators for next buffer
        self.phase_left = (self.phase_left + phase_inc_left * frame_count) % (2 * np.pi)
        self.phase_right = (self.phase_right + phase_inc_right * frame_count) % (2 * np.pi)
        
        # Interleave stereo output
        output = np.empty(frame_count * 2, dtype=np.float32)
        
        if mono:
            mono_samples = (left_samples + right_samples) / 2.0
            output[0::2] = mono_samples
            output[1::2] = mono_samples
        else:
            output[0::2] = left_samples
            output[1::2] = right_samples
        
        return output.tobytes()
    
    def _generate_spectral_chunk(self, frame_count: int) -> bytes:
        """
        Generate spectral/molecular stereo chunk with multiple frequencies.
        
        VECTORIZED VERSION - Uses NumPy for high performance.
        Smooth interpolation happens per-buffer (not per-sample) for efficiency.
        """
        with self.lock:
            freq_data = list(self.spectral_frequencies)
            target_amps = np.array(self._target_amplitudes) if self._target_amplitudes else np.array([])
            target_pans = np.array(self._target_positions) if self._target_positions else np.array([])
            current_amps = np.array(self._current_amplitudes) if self._current_amplitudes else np.array([])
            current_pans = np.array(self._current_positions) if self._current_positions else np.array([])
            master_amp = self.amplitude
        
        if not freq_data:
            return np.zeros(frame_count * 2, dtype=np.float32).tobytes()
        
        n_freqs = len(freq_data)
        
        # Ensure arrays are correct size
        if len(self.spectral_phases) != n_freqs:
            self.spectral_phases = [0.0] * n_freqs
        if len(current_amps) != n_freqs:
            current_amps = np.zeros(n_freqs)
        if len(current_pans) != n_freqs:
            current_pans = np.zeros(n_freqs)
        if len(target_amps) != n_freqs:
            target_amps = np.zeros(n_freqs)
        if len(target_pans) != n_freqs:
            target_pans = np.zeros(n_freqs)
        
        # Per-buffer smoothing (fast exponential approach)
        # At 44100 Hz with 1024 buffer, we get ~43 buffers/sec
        # smooth_factor = 0.15 means ~63% in ~7 buffers (~160ms)
        amp_smooth = 0.12
        pan_smooth = 0.08
        
        # Smooth current values toward targets (per-buffer, not per-sample)
        current_amps = current_amps + (target_amps - current_amps) * amp_smooth
        current_pans = current_pans + (target_pans - current_pans) * pan_smooth
        
        # Store smoothed values back
        with self.lock:
            self._current_amplitudes = current_amps.tolist()
            self._current_positions = current_pans.tolist()
        
        # VECTORIZED GENERATION
        output_left = np.zeros(frame_count, dtype=np.float32)
        output_right = np.zeros(frame_count, dtype=np.float32)
        
        # Time array for this buffer
        t = np.arange(frame_count, dtype=np.float32)
        
        for idx, (freq, _, phase_off) in enumerate(freq_data):
            # Phase increment per sample
            phase_inc = 2 * np.pi * freq / SAMPLE_RATE
            
            # Generate full buffer of phases
            phases = self.spectral_phases[idx] + phase_off + phase_inc * t
            
            # Generate waveform (vectorized)
            samples = current_amps[idx] * np.sin(phases)
            
            # Pan law (equal power)
            pan = current_pans[idx]
            pan_angle = (pan + 1) * np.pi / 4
            left_gain = max(0.02, np.cos(pan_angle))
            right_gain = max(0.02, np.sin(pan_angle))
            
            # Accumulate to output
            output_left += samples * left_gain
            output_right += samples * right_gain
            
            # Update phase accumulator for next buffer
            self.spectral_phases[idx] = (self.spectral_phases[idx] + phase_inc * frame_count) % (2 * np.pi)
        
        # Apply master amplitude
        output_left *= master_amp
        output_right *= master_amp
        
        # Hard clip
        np.clip(output_left, -1.0, 1.0, out=output_left)
        np.clip(output_right, -1.0, 1.0, out=output_right)
        
        # Interleave
        output = np.empty(frame_count * 2, dtype=np.float32)
        output[0::2] = output_left
        output[1::2] = output_right
        
        return output.tobytes()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - generates audio in real-time"""
        if not self.playing:
            return (None, pyaudio.paComplete)
        
        if self.mode == "binaural":
            data = self._generate_binaural_chunk(frame_count)
        else:  # spectral or molecular
            data = self._generate_spectral_chunk(frame_count)
        
        return (data, pyaudio.paContinue)
    
    # ══════════════════════════════════════════════════════════════════════════
    # START/STOP
    # ══════════════════════════════════════════════════════════════════════════
    
    def start_binaural(self, freq_left: float, freq_right: float,
                       phase_angle_deg: float, amplitude: float,
                       waveform: str = "sine"):
        """Start continuous binaural playback"""
        self.stop()
        
        self.mode = "binaural"
        self.set_binaural_params(freq_left, freq_right, phase_angle_deg, amplitude, waveform)
        self.phase_left = 0.0
        self.phase_right = 0.0
        
        self._start_stream()
    
    def start_spectral(self, frequencies: list, amplitudes: list,
                       phases: list = None, positions: list = None,
                       master_amplitude: float = 0.7):
        """Start continuous spectral/molecular playback"""
        self.stop()
        
        self.mode = "spectral"
        self.amplitude = master_amplitude
        self.set_spectral_params(frequencies, amplitudes, phases, positions)
        self.spectral_phases = [0.0] * len(frequencies)
        
        self._start_stream()
    
    def _start_stream(self):
        """Start the audio stream"""
        if not HAS_PYAUDIO:
            print("PyAudio not available")
            return
        
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            stream_params = {
                'format': pyaudio.paFloat32,
                'channels': 2,
                'rate': SAMPLE_RATE,
                'output': True,
                'frames_per_buffer': 1024,
                'stream_callback': self._audio_callback
            }
            
            if self.selected_device is not None:
                stream_params['output_device_index'] = self.selected_device
            
            self.stream = self.pyaudio_instance.open(**stream_params)
            self.playing = True
            
            print(f"🔊 Audio started ({self.mode} mode)")
            
        except Exception as e:
            print(f"Audio error: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop playback"""
        self.playing = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
    
    def is_playing(self):
        return self.playing
