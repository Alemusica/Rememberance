"""
Audio Engine - Real-time audio generation
==========================================

Callback-based audio engine with REAL-TIME parameter updates.
Changes to parameters (frequency, phase, amplitude, etc.) are applied
IMMEDIATELY without stopping playback - no glitches, smooth transitions.
"""

import numpy as np
import threading
from typing import Optional, List

# PyAudio import
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("⚠️ PyAudio not available")

from .golden_math import PHI, PHI_CONJUGATE

# Audio constants
SAMPLE_RATE: int = 44100


class AudioEngine:
    """
    Callback-based audio engine with REAL-TIME parameter updates.
    
    Supports two modes:
    - binaural: Two frequencies with phase offset (for binaural beats)
    - spectral: Multiple frequencies with individual positions (for vibroacoustic)
    """
    
    def __init__(self):
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.playing = False
        
        # === BINAURAL MODE parameters ===
        self.mode = "binaural"  # "binaural", "spectral"
        
        # Binaural parameters (real-time updateable)
        self.freq_left = 432.0
        self.freq_right = 440.0
        self.phase_offset = np.radians(137.5)  # Phase angle in radians
        self.amplitude = 0.7
        self.waveform_mode = "sine"
        self.mono_mix = False  # If True, outputs (L+R)/2 to both channels
        
        # Phase accumulators (continuous across callbacks)
        self.phase_left = 0.0
        self.phase_right = 0.0
        
        # === SPECTRAL MODE parameters ===
        self.spectral_frequencies = []  # List of (freq, amplitude, phase)
        self.spectral_phases = []  # Phase accumulators for each frequency
        self.stereo_positions = []  # -1 to 1 for panning
        
        # === SMOOTH PARAMETER INTERPOLATION ===
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
    
    # ══════════════════════════════════════════════════════════════════════════
    # DEVICE MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    
    def _scan_devices(self):
        """Scan for audio output devices"""
        if not HAS_PYAUDIO:
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
    
    def get_device_names(self) -> List[str]:
        return [d['name'] for d in self.devices]
    
    def set_device(self, idx: int):
        if 0 <= idx < len(self.devices):
            self.selected_device = self.devices[idx]['index']
    
    # ══════════════════════════════════════════════════════════════════════════
    # REAL-TIME PARAMETER SETTERS
    # ══════════════════════════════════════════════════════════════════════════
    
    def set_binaural_params(self, freq_left: float, freq_right: float, 
                            phase_angle_deg: float, amplitude: float, 
                            waveform: str = "sine"):
        """Update binaural parameters in real-time"""
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
        """Enable/disable mono mix mode"""
        with self.lock:
            self.mono_mix = enabled
    
    def set_spectral_params(self, frequencies: list, amplitudes: list, 
                            phases: list = None, positions: list = None):
        """
        Update spectral parameters in real-time.
        
        Uses smooth interpolation to prevent clicks.
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
            
            # Initialize current values if not set
            if len(self._current_amplitudes) != len(frequencies):
                self._current_amplitudes = list(amplitudes)
                self._current_positions = list(self._target_positions)
            
            # Initialize phase accumulators if needed
            if len(self.spectral_phases) != len(frequencies):
                self.spectral_phases = [0.0] * len(frequencies)
    
    # ══════════════════════════════════════════════════════════════════════════
    # GOLDEN WAVE GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _golden_wave_sample(self, phase: float, reversed: bool = True) -> float:
        """Generate single golden wave sample"""
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
    # AUDIO GENERATION (callback)
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_binaural_chunk(self, frame_count: int) -> bytes:
        """Generate binaural stereo chunk"""
        with self.lock:
            freq_l = self.freq_left
            freq_r = self.freq_right
            phase_off = self.phase_offset
            amp = self.amplitude
            waveform = self.waveform_mode
            mono = self.mono_mix
        
        phase_inc_left = 2 * np.pi * freq_l / SAMPLE_RATE
        phase_inc_right = 2 * np.pi * freq_r / SAMPLE_RATE
        
        output = np.empty(frame_count * 2, dtype=np.float32)
        
        for i in range(frame_count):
            if waveform == "sine":
                left_sample = amp * np.sin(self.phase_left)
                right_sample = amp * np.sin(self.phase_right + phase_off)
            elif waveform == "golden":
                left_sample = amp * self._golden_wave_sample(self.phase_left, reversed=False)
                right_sample = amp * self._golden_wave_sample(self.phase_right + phase_off, reversed=False)
            else:  # golden_reversed
                left_sample = amp * self._golden_wave_sample(self.phase_left, reversed=True)
                right_sample = amp * self._golden_wave_sample(self.phase_right + phase_off, reversed=True)
            
            if mono:
                mono_sample = (left_sample + right_sample) / 2.0
                output[i * 2] = mono_sample
                output[i * 2 + 1] = mono_sample
            else:
                output[i * 2] = left_sample
                output[i * 2 + 1] = right_sample
            
            self.phase_left += phase_inc_left
            self.phase_right += phase_inc_right
            
            if self.phase_left > 2 * np.pi:
                self.phase_left -= 2 * np.pi
            if self.phase_right > 2 * np.pi:
                self.phase_right -= 2 * np.pi
        
        return output.tobytes()
    
    def _generate_spectral_chunk(self, frame_count: int) -> bytes:
        """
        Generate spectral stereo chunk with multiple frequencies.
        Uses PER-SAMPLE smooth interpolation to prevent clicks.
        """
        with self.lock:
            freq_data = list(self.spectral_frequencies)
            target_amps = list(self._target_amplitudes) if self._target_amplitudes else []
            target_pans = list(self._target_positions) if self._target_positions else []
            current_amps = list(self._current_amplitudes) if self._current_amplitudes else []
            current_pans = list(self._current_positions) if self._current_positions else []
            master_amp = self.amplitude
        
        if not freq_data:
            return np.zeros(frame_count * 2, dtype=np.float32).tobytes()
        
        output_left = np.zeros(frame_count, dtype=np.float32)
        output_right = np.zeros(frame_count, dtype=np.float32)
        
        # Ensure we have enough accumulators
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
        
        # Smoothing coefficients (per sample)
        amp_smooth = 0.00012   # ~190ms to reach 63%
        pan_smooth = 0.00008   # ~280ms to reach 63%
        
        for idx, (freq, _, phase_off) in enumerate(freq_data):
            phase_inc = 2 * np.pi * freq / SAMPLE_RATE
            
            for i in range(frame_count):
                # Smooth amplitude toward target
                amp_diff = target_amps[idx] - current_amps[idx]
                current_amps[idx] += amp_diff * amp_smooth
                
                # Smooth pan toward target
                pan_diff = target_pans[idx] - current_pans[idx]
                current_pans[idx] += pan_diff * pan_smooth
                
                freq_amp = current_amps[idx]
                pan = current_pans[idx]
                
                # Pan law: equal power
                pan_angle = (pan + 1) * np.pi / 4
                left_gain = max(0.02, np.cos(pan_angle))
                right_gain = max(0.02, np.sin(pan_angle))
                
                sample = freq_amp * np.sin(self.spectral_phases[idx] + phase_off)
                output_left[i] += sample * left_gain
                output_right[i] += sample * right_gain
                
                self.spectral_phases[idx] += phase_inc
                if self.spectral_phases[idx] > 2 * np.pi:
                    self.spectral_phases[idx] -= 2 * np.pi
        
        # Store updated smooth values back
        with self.lock:
            self._current_amplitudes = current_amps
            self._current_positions = current_pans
        
        # Apply master amplitude and hard clip
        output_left *= master_amp
        output_right *= master_amp
        np.clip(output_left, -1.0, 1.0, out=output_left)
        np.clip(output_right, -1.0, 1.0, out=output_right)
        
        # Interleave
        output = np.empty(frame_count * 2, dtype=np.float32)
        output[0::2] = output_left
        output[1::2] = output_right
        
        return output.tobytes()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback"""
        if status:
            # Status flags: paInputUnderflow, paInputOverflow, paOutputUnderflow, paOutputOverflow
            print(f"⚠️ Audio callback status: {status} (frame_count={frame_count})")
        
        if not self.playing:
            return (None, pyaudio.paComplete)
        
        if self.mode == "binaural":
            data = self._generate_binaural_chunk(frame_count)
        else:
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
        """Start continuous spectral playback"""
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
    
    def is_playing(self) -> bool:
        return self.playing
