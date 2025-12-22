#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GOLDEN SOUND STUDIO - UNIFIED INTERFACE                   ║
║                                                                              ║
║   Three apps in one:                                                         ║
║   🎵 TAB 1: Binaural Beats - Phase angle control, sacred geometry           ║
║   ⚛️ TAB 2: Spectral Sound - Play atomic elements                           ║
║   🧪 TAB 3: Molecular Sound - Play molecules with bond angles as phases     ║
║                                                                              ║
║   "The universe is made of vibrations - let's hear them"                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import subprocess
import sys
import os
import time
from typing import Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED CONSTANTS (from golden_constants module)
# ══════════════════════════════════════════════════════════════════════════════

from golden_constants import (
    PHI, PHI_CONJUGATE, SAMPLE_RATE, SACRED_ANGLES,
    GOLDEN_ANGLE_DEG, GOLDEN_ANGLE_RAD,
    golden_wave_sample, apply_golden_envelope, golden_ease,
    generate_golden_phases, fibonacci_harmonics, phi_amplitude_decay,
    FIBONACCI,
    sound_to_light_color, harmonic_color_palette, SOLFEGGIO_FREQUENCIES,
    SOUNDBOARD_CONFIG,
)

# PyAudio
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Import our modules
try:
    from spectral_sound import SpectralSounder, PhaseMode
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False
    print("⚠️ spectral_sound.py not found")

try:
    from molecular_sound import MolecularSounder, MOLECULES_DB
    HAS_MOLECULAR = True
except ImportError:
    HAS_MOLECULAR = False
    print("⚠️ molecular_sound.py not found")

# Import soundboard panning module
try:
    from soundboard_panning import (
        SoundboardConfig, SoundboardPanner, 
        calculate_panning, print_panning_table,
        SPRUCE_VELOCITY_LONGITUDINAL, BOARD_LENGTH_MM
    )
    HAS_SOUNDBOARD = True
except ImportError:
    HAS_SOUNDBOARD = False
    print("⚠️ soundboard_panning.py not found")

# Import EMDR bilateral stimulation module
try:
    from ui.emdr_tab import EMDRTab
    HAS_EMDR = True
except ImportError:
    HAS_EMDR = False
    print("⚠️ ui/emdr_tab.py not found")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE - REAL-TIME PARAMETER UPDATES, NO GLITCHES
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: BINAURAL BEATS
# ══════════════════════════════════════════════════════════════════════════════

class BinauralTab:
    """Binaural beats with phase angle control - CONTINUOUS PLAYBACK"""
    
    def __init__(self, parent, audio_engine: AudioEngine):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        # State
        self.base_freq = tk.DoubleVar(value=432.0)  # Base frequency
        self.beat_freq = tk.DoubleVar(value=8.0)    # BEAT FREQUENCY (battimenti)
        self.freq_left = tk.DoubleVar(value=432.0)
        self.freq_right = tk.DoubleVar(value=440.0)
        self.phase_angle = tk.DoubleVar(value=137.5)
        self.amplitude = tk.DoubleVar(value=0.7)
        self.waveform = tk.StringVar(value="golden_reversed")
        self.link_mode = tk.StringVar(value="beat")  # "beat" = use battimenti, "manual" = manual L/R
        
        self._setup_ui()
        self._update_frequencies()  # Initial sync
    
    def _setup_ui(self):
        """Build the UI with scrollable controls panel"""
        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL WITH SCROLLBAR
        # ═══════════════════════════════════════════════════════════════════
        left_container = ttk.Frame(self.frame)
        left_container.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Create canvas for scrolling
        self._scroll_canvas = tk.Canvas(left_container, highlightthickness=0, width=420)
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=self._scroll_canvas.yview)
        
        # Scrollable frame inside canvas
        left_frame = ttk.LabelFrame(self._scroll_canvas, text="🎛️ Controls", padding=10)
        
        # Configure scrolling
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        self._scroll_canvas.pack(side='left', fill='both', expand=True)
        
        # Create window inside canvas for the frame
        self._canvas_window = self._scroll_canvas.create_window((0, 0), window=left_frame, anchor='nw')
        
        # Bind events for scroll region update
        def _configure_scroll_region(event):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        
        def _configure_canvas_width(event):
            self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)
        
        left_frame.bind('<Configure>', _configure_scroll_region)
        self._scroll_canvas.bind('<Configure>', _configure_canvas_width)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self._scroll_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTROLS CONTENT
        # ═══════════════════════════════════════════════════════════════════
        
        # === BATTIMENTI (Beat Frequency) - MAIN CONTROL ===
        beat_frame = ttk.LabelFrame(left_frame, text="🌀 BATTIMENTI (Beat Frequency)", padding=10)
        beat_frame.pack(fill='x', pady=5)
        
        # Mode selector
        mode_frame = ttk.Frame(beat_frame)
        mode_frame.pack(fill='x', pady=5)
        ttk.Radiobutton(mode_frame, text="🔗 Battimenti Mode", variable=self.link_mode,
                       value="beat", command=self._update_frequencies).pack(side='left', padx=10)
        ttk.Radiobutton(mode_frame, text="✋ Manual L/R", variable=self.link_mode,
                       value="manual").pack(side='left', padx=10)
        
        # Base frequency (carrier)
        base_row = ttk.Frame(beat_frame)
        base_row.pack(fill='x', pady=3)
        ttk.Label(base_row, text="Base Freq (Hz):", width=15).pack(side='left')
        ttk.Entry(base_row, textvariable=self.base_freq, width=8).pack(side='left')
        base_scale = ttk.Scale(base_row, from_=20, to=800, variable=self.base_freq,
                               orient='horizontal', length=180, command=lambda e: self._update_frequencies())
        base_scale.pack(side='left', padx=5)
        
        # Beat frequency (battimenti) - THE KEY SLIDER
        beat_row = ttk.Frame(beat_frame)
        beat_row.pack(fill='x', pady=3)
        ttk.Label(beat_row, text="Battimenti (Hz):", width=15).pack(side='left')
        self.beat_entry = ttk.Entry(beat_row, textvariable=self.beat_freq, width=8)
        self.beat_entry.pack(side='left')
        beat_scale = ttk.Scale(beat_row, from_=0.5, to=50, variable=self.beat_freq,
                               orient='horizontal', length=180, command=lambda e: self._update_frequencies())
        beat_scale.pack(side='left', padx=5)
        
        # Beat presets
        preset_row = ttk.Frame(beat_frame)
        preset_row.pack(fill='x', pady=5)
        ttk.Label(preset_row, text="Presets:").pack(side='left')
        
        beat_presets = [
            ("δ 2Hz", 2), ("θ 6Hz", 6), ("α 10Hz", 10), 
            ("β 20Hz", 20), ("γ 40Hz", 40), ("φ", PHI)
        ]
        
        for name, beat in beat_presets:
            btn = ttk.Button(preset_row, text=name, width=6,
                           command=lambda b=beat: self._set_beat_preset(b))
            btn.pack(side='left', padx=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # MUSICAL INTERVALS - Root + Interval shortcuts
        # ═══════════════════════════════════════════════════════════════════
        interval_frame = ttk.LabelFrame(left_frame, text="🎼 Musical Intervals (L=Root, R=Interval)", padding=5)
        interval_frame.pack(fill='x', pady=5)
        
        # Musical interval ratios (just intonation)
        # These create consonant relationships between L and R
        intervals_row1 = ttk.Frame(interval_frame)
        intervals_row1.pack(fill='x', pady=2)
        intervals_row2 = ttk.Frame(interval_frame)
        intervals_row2.pack(fill='x', pady=2)
        
        musical_intervals = [
            # Row 1: Basic intervals
            ("Unison", 1, 1),        # 1:1
            ("m2", 16, 15),          # Minor 2nd
            ("M2", 9, 8),            # Major 2nd  
            ("m3", 6, 5),            # Minor 3rd
            ("M3", 5, 4),            # Major 3rd (just)
            ("P4", 4, 3),            # Perfect 4th
            # Row 2: Upper intervals
            ("Tritone", 45, 32),     # Tritone
            ("P5", 3, 2),            # Perfect 5th
            ("m6", 8, 5),            # Minor 6th
            ("M6", 5, 3),            # Major 6th
            ("m7", 9, 5),            # Minor 7th
            ("Octave", 2, 1),        # Octave
        ]
        
        for i, (name, num, den) in enumerate(musical_intervals[:6]):
            btn = ttk.Button(intervals_row1, text=name, width=7,
                           command=lambda n=num, d=den: self._set_interval(n, d))
            btn.pack(side='left', padx=1)
        
        for i, (name, num, den) in enumerate(musical_intervals[6:]):
            btn = ttk.Button(intervals_row2, text=name, width=7,
                           command=lambda n=num, d=den: self._set_interval(n, d))
            btn.pack(side='left', padx=1)
        
        # Preset save/load section
        preset_save_frame = ttk.LabelFrame(left_frame, text="💾 Presets", padding=5)
        preset_save_frame.pack(fill='x', pady=5)
        
        preset_btn_row = ttk.Frame(preset_save_frame)
        preset_btn_row.pack(fill='x', pady=2)
        
        self.preset_name_var = tk.StringVar(value="my_preset")
        ttk.Entry(preset_btn_row, textvariable=self.preset_name_var, width=15).pack(side='left', padx=2)
        ttk.Button(preset_btn_row, text="Save", command=self._save_preset).pack(side='left', padx=2)
        ttk.Button(preset_btn_row, text="Load", command=self._load_preset).pack(side='left', padx=2)
        
        # Quick preset buttons
        quick_preset_row = ttk.Frame(preset_save_frame)
        quick_preset_row.pack(fill='x', pady=2)
        
        quick_presets = [
            ("Meditation", 432, 7.83, 137.5),   # Schumann + Golden angle
            ("Focus", 440, 14, 108),             # Beta + Pentagon angle
            ("Sleep", 396, 3, 51.84),            # Deep delta + Pyramid
            ("Healing", 528, 8, 137.5),          # Solfeggio + Golden
        ]
        
        for name, base, beat, phase in quick_presets:
            btn = ttk.Button(quick_preset_row, text=name, width=10,
                           command=lambda b=base, bt=beat, p=phase: self._apply_quick_preset(b, bt, p))
            btn.pack(side='left', padx=1)
        
        # Calculated frequencies display (read-only when in beat mode)
        calc_frame = ttk.LabelFrame(left_frame, text="📊 Resulting Frequencies", padding=5)
        calc_frame.pack(fill='x', pady=5)
        
        self.calc_label = ttk.Label(calc_frame, text="L: 432.0 Hz | R: 440.0 Hz | Beat: 8.0 Hz",
                                    font=('Courier', 10, 'bold'))
        self.calc_label.pack(pady=5)
        
        # Manual frequency controls (shown but linked in beat mode)
        manual_frame = ttk.LabelFrame(left_frame, text="Manual Frequencies (active in Manual mode)", padding=5)
        manual_frame.pack(fill='x', pady=5)
        
        ttk.Label(manual_frame, text="Left (Hz):").grid(row=0, column=0, sticky='w')
        ttk.Entry(manual_frame, textvariable=self.freq_left, width=10).grid(row=0, column=1)
        ttk.Scale(manual_frame, from_=20, to=1000, variable=self.freq_left, 
                  orient='horizontal', length=150).grid(row=0, column=2)
        
        ttk.Label(manual_frame, text="Right (Hz):").grid(row=1, column=0, sticky='w')
        ttk.Entry(manual_frame, textvariable=self.freq_right, width=10).grid(row=1, column=1)
        ttk.Scale(manual_frame, from_=20, to=1000, variable=self.freq_right,
                  orient='horizontal', length=150).grid(row=1, column=2)
        
        # Phase angle
        phase_frame = ttk.LabelFrame(left_frame, text="Phase Angle", padding=5)
        phase_frame.pack(fill='x', pady=5)
        
        ttk.Label(phase_frame, text="Phase (°):").pack(side='left')
        ttk.Entry(phase_frame, textvariable=self.phase_angle, width=10).pack(side='left')
        ttk.Scale(phase_frame, from_=0, to=360, variable=self.phase_angle,
                  orient='horizontal', length=200).pack(side='left', padx=5)
        
        # Sacred angle presets
        sacred_frame = ttk.LabelFrame(left_frame, text="Sacred Angles", padding=5)
        sacred_frame.pack(fill='x', pady=5)
        
        for i, (name, angle) in enumerate(SACRED_ANGLES.items()):
            short_name = name.split('(')[0].strip()[:15]
            btn = ttk.Button(sacred_frame, text=f"{short_name} ({angle:.1f}°)", width=20,
                           command=lambda a=angle: self.phase_angle.set(a))
            btn.grid(row=i//2, column=i%2, padx=2, pady=1)
        
        # Waveform
        wave_frame = ttk.LabelFrame(left_frame, text="Waveform", padding=5)
        wave_frame.pack(fill='x', pady=5)
        
        for wf in ["sine", "golden", "golden_reversed"]:
            ttk.Radiobutton(wave_frame, text=wf, variable=self.waveform, 
                          value=wf).pack(side='left', padx=10)
        
        # Amplitude only (no duration - continuous!)
        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill='x', pady=5)
        
        ttk.Label(param_frame, text="Amplitude:").pack(side='left', padx=(20, 0))
        ttk.Scale(param_frame, from_=0, to=1, variable=self.amplitude,
                  orient='horizontal', length=150).pack(side='left')
        
        # MONO MIX - Phase Cancellation Test Mode
        mono_frame = ttk.LabelFrame(left_frame, text="🧪 Phase Cancellation Test", padding=5)
        mono_frame.pack(fill='x', pady=5)
        
        self.mono_mix_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mono_frame, text="MONO MIX (L+R)/2", variable=self.mono_mix_var,
                        command=self._on_mono_change).pack(side='left')
        ttk.Label(mono_frame, text="← At 180° with SAME freq = SILENCE!", 
                  foreground='red').pack(side='left', padx=10)
        
        # Quick test buttons
        test_frame = ttk.Frame(mono_frame)
        test_frame.pack(fill='x', pady=3)
        ttk.Button(test_frame, text="Test 180° Cancel", 
                   command=self._test_180_cancel).pack(side='left', padx=2)
        ttk.Button(test_frame, text="Test 0° Sum", 
                   command=self._test_0_sum).pack(side='left', padx=2)
        
        # Playback buttons - BIG
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ PLAY (continuous)", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text="🌀 3D Scope", command=self._launch_3d).pack(side='left', padx=5)
        
        # Right panel - Visualization
        right_frame = ttk.LabelFrame(self.frame, text="📊 Visualization", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=300, height=300, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        # Info label
        self.info_var = tk.StringVar(value="Ready - Audio will play continuously until STOP")
        ttk.Label(right_frame, textvariable=self.info_var, wraplength=280).pack()
        
        # Initial draw
        self._draw_phase_circle()
        
        # Bind updates for visualization AND real-time audio
        self.phase_angle.trace_add('write', self._on_phase_change)
        self.beat_freq.trace_add('write', lambda *args: self._update_frequencies())
        self.base_freq.trace_add('write', lambda *args: self._update_frequencies())
        self.amplitude.trace_add('write', self._on_amplitude_change)
        self.waveform.trace_add('write', self._on_waveform_change)
        self.freq_left.trace_add('write', self._on_freq_change)
        self.freq_right.trace_add('write', self._on_freq_change)
    
    def _set_beat_preset(self, beat: float):
        """Set beat frequency preset"""
        self.beat_freq.set(beat)
        self._update_frequencies()
    
    def _set_interval(self, numerator: int, denominator: int):
        """Set musical interval: Right = Left × (num/den)"""
        self.link_mode.set("manual")
        base = self.base_freq.get()
        self.freq_left.set(base)
        self.freq_right.set(base * numerator / denominator)
        self._update_frequencies()
        
        # Calculate interval name for display
        ratio = numerator / denominator
        cents = 1200 * np.log2(ratio)
        self.info_var.set(f"Interval: {numerator}:{denominator} = {ratio:.4f} ({cents:.1f} cents)")
    
    def _apply_quick_preset(self, base: float, beat: float, phase: float):
        """Apply a quick preset with base, beat, and phase"""
        self.link_mode.set("beat")
        self.base_freq.set(base)
        self.beat_freq.set(beat)
        self.phase_angle.set(phase)
        self._update_frequencies()
    
    def _save_preset(self):
        """Save current settings to a preset file"""
        import json
        name = self.preset_name_var.get().strip()
        if not name:
            return
        
        preset = {
            'base_freq': self.base_freq.get(),
            'beat_freq': self.beat_freq.get(),
            'freq_left': self.freq_left.get(),
            'freq_right': self.freq_right.get(),
            'phase_angle': self.phase_angle.get(),
            'amplitude': self.amplitude.get(),
            'waveform': self.waveform.get(),
            'link_mode': self.link_mode.get(),
        }
        
        preset_dir = os.path.join(os.path.dirname(__file__), 'presets')
        os.makedirs(preset_dir, exist_ok=True)
        
        filepath = os.path.join(preset_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(preset, f, indent=2)
        
        self.info_var.set(f"💾 Saved: {name}.json")
    
    def _load_preset(self):
        """Load settings from a preset file"""
        import json
        name = self.preset_name_var.get().strip()
        if not name:
            return
        
        preset_dir = os.path.join(os.path.dirname(__file__), 'presets')
        filepath = os.path.join(preset_dir, f"{name}.json")
        
        if not os.path.exists(filepath):
            self.info_var.set(f"⚠️ Preset not found: {name}")
            return
        
        with open(filepath, 'r') as f:
            preset = json.load(f)
        
        self.base_freq.set(preset.get('base_freq', 432))
        self.beat_freq.set(preset.get('beat_freq', 8))
        self.freq_left.set(preset.get('freq_left', 432))
        self.freq_right.set(preset.get('freq_right', 440))
        self.phase_angle.set(preset.get('phase_angle', 137.5))
        self.amplitude.set(preset.get('amplitude', 0.7))
        self.waveform.set(preset.get('waveform', 'golden_reversed'))
        self.link_mode.set(preset.get('link_mode', 'beat'))
        
        self._update_frequencies()
        self.info_var.set(f"📂 Loaded: {name}.json")

    def _update_frequencies(self):
        """Update L/R frequencies based on beat mode - AND UPDATE AUDIO IN REAL-TIME"""
        try:
            if self.link_mode.get() == "beat":
                base = self.base_freq.get()
                beat = self.beat_freq.get()
                self.freq_left.set(base)
                self.freq_right.set(base + beat)
            
            # Update display
            l = self.freq_left.get()
            r = self.freq_right.get()
            b = abs(r - l)
            self.calc_label.config(text=f"L: {l:.1f} Hz | R: {r:.1f} Hz | Beat: {b:.2f} Hz")
            
            # === REAL-TIME UPDATE: if playing, update audio engine immediately ===
            if self.audio.is_playing():
                self.audio.set_frequencies(l, r)
                
        except:
            pass
    
    def _on_phase_change(self, *args):
        """Called when phase angle changes - update audio in real-time"""
        try:
            phase = self.phase_angle.get()
            self._draw_phase_circle()
            
            if self.audio.is_playing():
                self.audio.set_phase_angle(phase)
        except:
            pass
    
    def _on_amplitude_change(self, *args):
        """Called when amplitude changes - update audio in real-time"""
        try:
            amp = self.amplitude.get()
            if self.audio.is_playing():
                self.audio.set_amplitude(amp)
        except:
            pass
    
    def _on_waveform_change(self, *args):
        """Called when waveform changes - update audio in real-time"""
        try:
            wf = self.waveform.get()
            if self.audio.is_playing():
                self.audio.set_waveform(wf)
        except:
            pass
    
    def _on_freq_change(self, *args):
        """Called when manual frequencies change - update audio in real-time"""
        try:
            if self.link_mode.get() == "manual":
                l = self.freq_left.get()
                r = self.freq_right.get()
                b = abs(r - l)
                self.calc_label.config(text=f"L: {l:.1f} Hz | R: {r:.1f} Hz | Beat: {b:.2f} Hz")
                
                if self.audio.is_playing():
                    self.audio.set_frequencies(l, r)
        except:
            pass
    
    def _on_mono_change(self):
        """Called when mono mix checkbox changes"""
        mono = self.mono_mix_var.get()
        self.audio.set_mono_mix(mono)
        if mono:
            self.info_var.set("⚠️ MONO MIX ON: L+R summed. 180° with same freq = SILENCE!")
        else:
            self.info_var.set("STEREO: L and R go to separate ears (binaural effect)")
    
    def _test_180_cancel(self):
        """Quick test: Set 180° phase, same freq, mono - should be SILENCE"""
        self.link_mode.set("manual")
        self.freq_left.set(440.0)
        self.freq_right.set(440.0)  # SAME frequency!
        self.phase_angle.set(180.0)  # 180° = opposite phase
        self.mono_mix_var.set(True)
        self._on_mono_change()
        self._update_frequencies()
        if not self.audio.is_playing():
            self._play()
        self.info_var.set("🔇 TEST: 440Hz + 440Hz @ 180° MONO → SILENCE!")
    
    def _test_0_sum(self):
        """Quick test: Set 0° phase, same freq, mono - should be LOUD"""
        self.link_mode.set("manual")
        self.freq_left.set(440.0)
        self.freq_right.set(440.0)
        self.phase_angle.set(0.0)  # 0° = same phase
        self.mono_mix_var.set(True)
        self._on_mono_change()
        self._update_frequencies()
        if not self.audio.is_playing():
            self._play()
        self.info_var.set("🔊 TEST: 440Hz + 440Hz @ 0° MONO → CONSTRUCTIVE!")
    
    def _draw_phase_circle(self):
        """Draw phase visualization"""
        self.canvas.delete('all')
        
        cx, cy, r = 150, 150, 120
        phase = self.phase_angle.get()
        
        # Circle
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline='#333', width=2)
        
        # Angle markers
        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle - 90)
            x, y = cx + r * np.cos(rad), cy + r * np.sin(rad)
            self.canvas.create_text(x, y, text=f"{angle}°", fill='#666', font=('Courier', 8))
        
        # Left vector (reference)
        self.canvas.create_line(cx, cy, cx, cy - r * 0.8, fill='#ff6b6b', width=3, arrow='last')
        self.canvas.create_text(cx, cy - r - 10, text="L", fill='#ff6b6b', font=('Helvetica', 10, 'bold'))
        
        # Right vector (phase shifted)
        rad = np.radians(phase - 90)
        rx, ry = cx + r * 0.8 * np.cos(rad), cy + r * 0.8 * np.sin(rad)
        self.canvas.create_line(cx, cy, rx, ry, fill='#4ecdc4', width=3, arrow='last')
        self.canvas.create_text(rx + 15 * np.cos(rad), ry + 15 * np.sin(rad), 
                               text="R", fill='#4ecdc4', font=('Helvetica', 10, 'bold'))
        
        # Phase arc
        self.canvas.create_arc(cx-40, cy-40, cx+40, cy+40, 
                              start=90, extent=-phase, outline='#ffd700', width=2, style='arc')
        
        # Phase value
        self.canvas.create_text(cx, cy + r + 20, text=f"Phase: {phase:.2f}°", 
                               fill='#ffd700', font=('Courier', 10, 'bold'))
        
        # Show beat frequency
        beat = abs(self.freq_right.get() - self.freq_left.get())
        self.canvas.create_text(cx, cy + r + 40, text=f"Beat: {beat:.2f} Hz",
                               fill='#00ff88', font=('Courier', 10, 'bold'))
    
    def _generate_binaural(self, duration: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate binaural beat signal - short buffer for looping"""
        freq_l = self.freq_left.get()
        freq_r = self.freq_right.get()
        phase = np.radians(self.phase_angle.get())
        amp = self.amplitude.get()
        wf = self.waveform.get()
        
        # Generate a buffer (will be looped)
        num_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate waveforms
        if wf == "sine":
            left = amp * np.sin(2 * np.pi * freq_l * t)
            right = amp * np.sin(2 * np.pi * freq_r * t + phase)
        else:
            # Golden wave
            left = amp * self._golden_wave(2 * np.pi * freq_l * t, reversed=(wf == "golden_reversed"))
            right = amp * self._golden_wave(2 * np.pi * freq_r * t + phase, reversed=(wf == "golden_reversed"))
        
        # Smooth crossfade for seamless loop
        crossfade_samples = int(SAMPLE_RATE * 0.02)  # 20ms crossfade
        fade_in = np.linspace(0, 1, crossfade_samples)
        fade_out = np.linspace(1, 0, crossfade_samples)
        
        left[:crossfade_samples] *= fade_in
        left[-crossfade_samples:] *= fade_out
        right[:crossfade_samples] *= fade_in
        right[-crossfade_samples:] *= fade_out
        
        return left, right
    
    def _golden_wave(self, phase: np.ndarray, reversed: bool = True) -> np.ndarray:
        """Generate PHI-based waveform"""
        theta = phase % (2 * np.pi)
        if reversed:
            theta = 2 * np.pi - theta
        
        t = theta / (2 * np.pi)
        rise = PHI_CONJUGATE
        
        wave = np.zeros_like(t)
        rising = t < rise
        wave = np.where(rising, np.sin(np.pi * t / rise / 2), 0)
        wave = np.where(~rising, np.cos(np.pi * (t - rise) / (1 - rise) / 2), wave)
        
        return wave
    
    def _golden_envelope(self, length: int) -> np.ndarray:
        """Golden ratio envelope"""
        attack = int(length * PHI_CONJUGATE * PHI_CONJUGATE * 0.1)
        release = int(length * PHI_CONJUGATE * 0.2)
        
        env = np.ones(length)
        
        for i in range(attack):
            env[i] = (1 - np.cos(np.pi * i / attack)) / 2
        
        for i in range(release):
            env[length - 1 - i] = (1 - np.cos(np.pi * i / release)) / 2
        
        return env
    
    def _play(self):
        """Start CONTINUOUS callback-based playback"""
        freq_l = self.freq_left.get()
        freq_r = self.freq_right.get()
        phase = self.phase_angle.get()
        amp = self.amplitude.get()
        wf = self.waveform.get()
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        beat = abs(freq_r - freq_l)
        self.info_var.set(f"🔊 Playing... L:{freq_l:.0f}Hz R:{freq_r:.0f}Hz Beat:{beat:.2f}Hz")
        
        # Use callback-based continuous playback
        self.audio.start_binaural(freq_l, freq_r, phase, amp, wf)
    
    def _stop(self):
        """Stop playback"""
        self.audio.stop()
        self._on_playback_done()
    
    def _on_playback_done(self):
        """Callback when playback ends"""
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.info_var.set("Ready - Press PLAY for continuous audio")
    
    def _launch_3d(self):
        """Launch 3D oscilloscope"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scope_script = os.path.join(script_dir, 'oscilloscope_3d.py')
        
        cmd = [
            sys.executable, scope_script,
            '--freq-left', str(self.freq_left.get()),
            '--freq-right', str(self.freq_right.get()),
            '--phase', str(self.phase_angle.get()),
            '--amplitude', str(self.amplitude.get()),
            '--waveform', self.waveform.get()
        ]
        
        try:
            subprocess.Popen(cmd)
            self.info_var.set("🌀 3D Scope launched")
        except Exception as e:
            self.info_var.set(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SPECTRAL SOUND (Elements)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralTab:
    """Play atomic elements as sound"""
    
    def __init__(self, parent, audio_engine: AudioEngine):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        if HAS_SPECTRAL:
            self.sounder = SpectralSounder()
        else:
            self.sounder = None
        
        # State
        self.element = tk.StringVar()
        self.duration = tk.DoubleVar(value=3.0)
        self.phase_mode = tk.StringVar(value="GOLDEN")
        self.output_mode = tk.StringVar(value="stereo")
        self.beat_freq = tk.DoubleVar(value=7.83)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI"""
        if not HAS_SPECTRAL:
            ttk.Label(self.frame, text="⚠️ spectral_sound.py not found").pack(pady=50)
            return
        
        # Left panel
        left_frame = ttk.LabelFrame(self.frame, text="⚛️ Element Selection", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Element dropdown
        elements = self.sounder.get_element_names()
        ttk.Label(left_frame, text="Select Element:").pack(anchor='w')
        combo = ttk.Combobox(left_frame, textvariable=self.element, values=elements, 
                            state='readonly', width=30)
        combo.pack(fill='x', pady=5)
        combo.bind('<<ComboboxSelected>>', self._on_element_select)
        
        # Quick presets
        preset_frame = ttk.LabelFrame(left_frame, text="Quick Presets", padding=5)
        preset_frame.pack(fill='x', pady=5)
        
        presets = [
            ("🔴 Hydrogen", "Hydrogen-Balmer"),
            ("🟡 Helium", "Helium"),
            ("🟠 Sodium", "Sodium"),
            ("🔵 Neon", "Neon"),
            ("⚪ Mercury", "Mercury"),
            ("🟢 Oxygen", "Oxygen"),
        ]
        
        for i, (label, elem) in enumerate(presets):
            btn = ttk.Button(preset_frame, text=label, width=12,
                           command=lambda e=elem: self._select_element(e))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
        
        # Parameters
        param_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=5)
        param_frame.pack(fill='x', pady=5)
        
        ttk.Label(param_frame, text="Duration (s):").grid(row=0, column=0, sticky='w')
        ttk.Scale(param_frame, from_=1, to=10, variable=self.duration,
                  orient='horizontal', length=150).grid(row=0, column=1)
        
        ttk.Label(param_frame, text="Phase Mode:").grid(row=1, column=0, sticky='w')
        for i, mode in enumerate(["INCOHERENT", "COHERENT", "GOLDEN", "FIBONACCI"]):
            ttk.Radiobutton(param_frame, text=mode, variable=self.phase_mode,
                          value=mode).grid(row=2+i//2, column=i%2, sticky='w')
        
        ttk.Label(param_frame, text="Output:").grid(row=4, column=0, sticky='w')
        for i, mode in enumerate(["mono", "stereo", "binaural"]):
            ttk.Radiobutton(param_frame, text=mode, variable=self.output_mode,
                          value=mode).grid(row=4, column=i+1, sticky='w')
        
        ttk.Label(param_frame, text="Beat Freq (Hz):").grid(row=5, column=0, sticky='w')
        ttk.Scale(param_frame, from_=1, to=40, variable=self.beat_freq,
                  orient='horizontal', length=150).grid(row=5, column=1, columnspan=2)
        
        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ PLAY", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text="💾 SAVE", command=self._save).pack(side='left', padx=5)
        
        # Right panel - Spectrum
        right_frame = ttk.LabelFrame(self.frame, text="📊 Spectrum", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=350, height=250, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        # Info text
        self.info_text = tk.Text(right_frame, width=40, height=10, bg='#0a0a15', 
                                 fg='#00ff88', font=('Courier', 9), state='disabled')
        self.info_text.pack(fill='both', expand=True)
        
        self.status_var = tk.StringVar(value="Select an element")
        ttk.Label(right_frame, textvariable=self.status_var).pack()
        
        # Bind parameter changes for real-time updates
        self.phase_mode.trace_add('write', self._on_param_change)
        self.output_mode.trace_add('write', self._on_param_change)
        self.beat_freq.trace_add('write', self._on_param_change)
    
    def _select_element(self, element: str):
        """Select an element"""
        self.element.set(element)
        self._on_element_select(None)
    
    def _on_element_select(self, event):
        """Handle element selection"""
        element = self.element.get()
        if element and self.sounder:
            self._draw_spectrum(element)
            self._update_info(element)
            self.status_var.set(f"✅ {element}")
    
    def _draw_spectrum(self, element: str):
        """Draw element spectrum"""
        self.canvas.delete('all')
        
        lines = self.sounder.get_spectral_lines(element)
        if not lines:
            return
        
        scaled = self.sounder.scale_to_audio(lines)
        
        # Axes
        self.canvas.create_line(30, 220, 330, 220, fill='#333', width=2)
        self.canvas.create_line(30, 220, 30, 20, fill='#333', width=2)
        
        # Bars
        bar_width = 280 / max(len(lines), 1)
        colors = ['#ff6b6b', '#ffd700', '#00ff88', '#4ecdc4', '#ff00ff', '#00bfff', '#ff8c00']
        
        for i, ((freq, amp), line) in enumerate(zip(scaled, lines)):
            x = 40 + i * bar_width + bar_width/2
            height = amp * 180
            color = colors[i % len(colors)]
            
            self.canvas.create_rectangle(x - bar_width/3, 220 - height,
                                        x + bar_width/3, 220, fill=color)
            self.canvas.create_text(x, 230, text=f"{int(freq)}", fill='#666', font=('Courier', 7))
        
        self.canvas.create_text(180, 10, text=f"{element}", fill='#ffd700', 
                               font=('Helvetica', 11, 'bold'))
    
    def _update_info(self, element: str):
        """Update info panel"""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        
        lines = self.sounder.get_spectral_lines(element)
        if not lines:
            self.info_text.insert('end', "No data")
            self.info_text.config(state='disabled')
            return
        
        scaled = self.sounder.scale_to_audio(lines)
        
        self.info_text.insert('end', f"═══ {element} ═══\n\n")
        self.info_text.insert('end', f"{'Line':<8} {'λ(nm)':<8} {'f(Hz)':<8} {'Amp':<6}\n")
        self.info_text.insert('end', "─" * 32 + "\n")
        
        for line, (f_audio, amp) in zip(lines, scaled):
            self.info_text.insert('end', 
                f"{line.name[:7]:<8} {line.wavelength_nm:<8.1f} {f_audio:<8.0f} {amp:<6.2f}\n")
        
        self.info_text.config(state='disabled')
    
    def _play(self):
        """Play element sound - CONTINUOUS until STOP"""
        element = self.element.get()
        if not element:
            messagebox.showwarning("Warning", "Select an element first!")
            return
        
        try:
            # Get spectral lines
            lines = self.sounder.get_spectral_lines(element)
            if not lines:
                messagebox.showerror("Error", f"No spectral lines for {element}")
                return
            
            # Scale to audio frequencies
            scaled = self.sounder.scale_to_audio(lines)
            frequencies = [f for f, a in scaled]
            amplitudes = [a for f, a in scaled]
            
            # Generate phases based on mode
            phase_mode = PhaseMode[self.phase_mode.get()]
            phases = list(self.sounder.generate_phases(len(lines), phase_mode))
            
            # Generate stereo positions based on output mode
            output = self.output_mode.get()
            if output == "mono":
                positions = [0.0] * len(frequencies)  # center
            elif output == "stereo":
                # Spread across stereo field based on frequency
                positions = list(np.linspace(-0.8, 0.8, len(frequencies)))
            else:  # binaural - create binaural effect
                # Alternate left/right with beat frequency offset
                positions = [(-1.0 if i % 2 == 0 else 1.0) for i in range(len(frequencies))]
                # Add slight frequency shift for binaural beat on odd frequencies
                beat = self.beat_freq.get()
                frequencies = [f + (beat if i % 2 == 1 else 0) for i, f in enumerate(frequencies)]
            
            # Start continuous streaming
            self.audio.start_spectral(frequencies, amplitudes, phases, positions, 
                                      master_amplitude=0.7)
            
            self.play_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_var.set("🔊 Playing continuously...")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def _stop(self):
        self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set(f"✅ {self.element.get()}")
    
    def _on_done(self):
        """Legacy callback - not used with continuous playback"""
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set(f"✅ {self.element.get()}")
    
    def _save(self):
        """Save to WAV"""
        element = self.element.get()
        if not element:
            return
        
        filename = f"{element.lower().replace('-', '_')}.wav"
        try:
            left, right = self.sounder.generate_element_stereo(
                element, self.duration.get(), PhaseMode[self.phase_mode.get()])
            self.sounder.save_wav(left, filename, stereo=True, right_channel=right)
            self.status_var.set(f"💾 Saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _on_param_change(self, *args):
        """Update audio parameters in real-time when sliders change"""
        if not self.audio.is_playing():
            return
        
        element = self.element.get()
        if not element:
            return
        
        try:
            # Regenerate parameters for streaming audio
            lines = self.sounder.get_spectral_lines(element)
            if not lines:
                return
            
            scaled = self.sounder.scale_to_audio(lines)
            frequencies = [f for f, a in scaled]
            amplitudes = [a for f, a in scaled]
            
            phase_mode = PhaseMode[self.phase_mode.get()]
            phases = list(self.sounder.generate_phases(len(lines), phase_mode))
            
            output = self.output_mode.get()
            if output == "mono":
                positions = [0.0] * len(frequencies)
            elif output == "stereo":
                positions = list(np.linspace(-0.8, 0.8, len(frequencies)))
            else:  # binaural
                positions = [(-1.0 if i % 2 == 0 else 1.0) for i in range(len(frequencies))]
                beat = self.beat_freq.get()
                frequencies = [f + (beat if i % 2 == 1 else 0) for i, f in enumerate(frequencies)]
            
            self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
        except:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MOLECULAR SOUND
# ══════════════════════════════════════════════════════════════════════════════

class MolecularTab:
    """Play molecules with bond angles as phases"""
    
    def __init__(self, parent, audio_engine: AudioEngine):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        if HAS_MOLECULAR:
            self.sounder = MolecularSounder()
        else:
            self.sounder = None
        
        # State
        self.molecule = tk.StringVar()
        self.duration = tk.DoubleVar(value=4.0)
        self.output_mode = tk.StringVar(value="molecular")
        self.beat_freq = tk.DoubleVar(value=7.83)
        self.use_spectral = tk.BooleanVar(value=True)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI"""
        if not HAS_MOLECULAR:
            ttk.Label(self.frame, text="⚠️ molecular_sound.py not found").pack(pady=50)
            return
        
        # Left panel
        left_frame = ttk.LabelFrame(self.frame, text="🧪 Molecule Selection", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Molecule dropdown
        molecules = self.sounder.get_available_molecules()
        ttk.Label(left_frame, text="Select Molecule:").pack(anchor='w')
        combo = ttk.Combobox(left_frame, textvariable=self.molecule, values=molecules,
                            state='readonly', width=20)
        combo.pack(fill='x', pady=5)
        combo.bind('<<ComboboxSelected>>', self._on_molecule_select)
        
        # Quick presets with descriptions
        preset_frame = ttk.LabelFrame(left_frame, text="Molecules", padding=5)
        preset_frame.pack(fill='x', pady=5)
        
        presets = [
            ("💧 H₂O", "H2O", "Water (104.5°)"),
            ("☁️ CO₂", "CO2", "CO₂ (180°)"),
            ("🔥 CH₄", "CH4", "Methane (109.5°)"),
            ("💨 NH₃", "NH3", "Ammonia (107.3°)"),
            ("🌀 O₃", "O3", "Ozone (116.8°)"),
            ("💀 H₂S", "H2S", "H₂S (92.1°)"),
            ("🏭 SO₂", "SO2", "SO₂ (119°)"),
            ("🏙️ NO₂", "NO2", "NO₂ (134°)"),
        ]
        
        for i, (icon, formula, desc) in enumerate(presets):
            btn = ttk.Button(preset_frame, text=f"{icon} {formula}", width=12,
                           command=lambda f=formula: self._select_molecule(f))
            btn.grid(row=i//2, column=i%2, padx=2, pady=2)
        
        # Parameters
        param_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=5)
        param_frame.pack(fill='x', pady=5)
        
        ttk.Label(param_frame, text="Duration (s):").grid(row=0, column=0, sticky='w')
        ttk.Scale(param_frame, from_=1, to=10, variable=self.duration,
                  orient='horizontal', length=150).grid(row=0, column=1)
        
        ttk.Label(param_frame, text="Output Mode:").grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(param_frame, text="Molecular", variable=self.output_mode,
                       value="molecular").grid(row=2, column=0, sticky='w')
        ttk.Radiobutton(param_frame, text="Binaural", variable=self.output_mode,
                       value="binaural").grid(row=2, column=1, sticky='w')
        
        ttk.Label(param_frame, text="Beat Freq (Hz):").grid(row=3, column=0, sticky='w')
        ttk.Scale(param_frame, from_=1, to=40, variable=self.beat_freq,
                  orient='horizontal', length=150).grid(row=3, column=1)
        
        ttk.Checkbutton(param_frame, text="Use real spectral lines", 
                       variable=self.use_spectral).grid(row=4, column=0, columnspan=2, sticky='w')
        
        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ PLAY", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text="💾 SAVE", command=self._save).pack(side='left', padx=5)
        
        # Right panel - Visualization
        right_frame = ttk.LabelFrame(self.frame, text="🔬 Molecular Structure", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=350, height=250, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        # Info text
        self.info_text = tk.Text(right_frame, width=40, height=12, bg='#0a0a15',
                                fg='#00ff88', font=('Courier', 9), state='disabled')
        self.info_text.pack(fill='both', expand=True)
        
        self.status_var = tk.StringVar(value="Select a molecule")
        ttk.Label(right_frame, textvariable=self.status_var).pack()
        
        # Bind parameter changes for real-time updates
        self.output_mode.trace_add('write', self._on_param_change)
        self.beat_freq.trace_add('write', self._on_param_change)
        self.use_spectral.trace_add('write', self._on_param_change)
    
    def _select_molecule(self, formula: str):
        """Select a molecule"""
        self.molecule.set(formula)
        self._on_molecule_select(None)
    
    def _on_molecule_select(self, event):
        """Handle molecule selection"""
        formula = self.molecule.get()
        if formula and self.sounder:
            mol = self.sounder.get_molecule(formula)
            if mol:
                self._draw_molecule(mol)
                self._update_info(mol)
                self.status_var.set(f"✅ {mol.name}")
    
    def _draw_molecule(self, mol):
        """Draw molecule structure"""
        self.canvas.delete('all')
        
        cx, cy = 175, 125
        scale = 80
        
        # Draw bonds
        for bond in mol.bonds:
            a1 = mol.atoms[bond.atom1_idx]
            a2 = mol.atoms[bond.atom2_idx]
            
            x1 = cx + a1.position[0] * scale
            y1 = cy - a1.position[1] * scale
            x2 = cx + a2.position[0] * scale
            y2 = cy - a2.position[1] * scale
            
            # Multiple lines for double/triple bonds
            for offset in range(bond.order):
                dy = (offset - (bond.order-1)/2) * 3
                self.canvas.create_line(x1, y1+dy, x2, y2+dy, fill='#666', width=2)
        
        # Draw atoms
        colors = {
            'H': '#ffffff', 'O': '#ff4444', 'C': '#333333', 'N': '#4444ff',
            'S': '#ffff00', 'Cl': '#44ff44', 'F': '#88ff88'
        }
        
        for atom in mol.atoms:
            x = cx + atom.position[0] * scale
            y = cy - atom.position[1] * scale
            r = 15 if atom.symbol != 'H' else 10
            
            color = colors.get(atom.symbol, '#888888')
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='white')
            self.canvas.create_text(x, y, text=atom.symbol, fill='white' if atom.symbol not in ['H', 'S'] else 'black',
                                   font=('Helvetica', 10, 'bold'))
        
        # Draw angle arc if available
        if mol.bond_angles:
            angle = mol.bond_angles[0]
            self.canvas.create_text(cx, 230, text=f"Bond Angle: {angle}° → Phase: {np.radians(angle):.3f} rad",
                                   fill='#ffd700', font=('Courier', 10))
        
        self.canvas.create_text(175, 15, text=f"{mol.name} ({mol.formula})",
                               fill='#ffd700', font=('Helvetica', 12, 'bold'))
    
    def _update_info(self, mol):
        """Update info panel"""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        
        self.info_text.insert('end', f"═══ {mol.name} ({mol.formula}) ═══\n\n")
        self.info_text.insert('end', f"Symmetry: {mol.symmetry}\n")
        self.info_text.insert('end', f"Dipole: {mol.dipole_moment:.2f} D\n\n")
        
        self.info_text.insert('end', "ATOMS:\n")
        for atom in mol.atoms:
            self.info_text.insert('end', 
                f"  {atom.symbol}: mass={atom.mass:.3f}, χ={atom.electronegativity:.2f}\n")
        
        self.info_text.insert('end', "\nBOND ANGLES → PHASES:\n")
        for i, angle in enumerate(mol.bond_angles):
            phase = np.radians(angle)
            self.info_text.insert('end', f"  Angle {i+1}: {angle:>7.2f}° → {phase:.4f} rad\n")
        
        self.info_text.config(state='disabled')
    
    def _play(self):
        """Play molecule sound - CONTINUOUS with REAL SPECTRAL LINES until STOP"""
        formula = self.molecule.get()
        if not formula:
            messagebox.showwarning("Warning", "Select a molecule first!")
            return
        
        mol = self.sounder.get_molecule(formula)
        if not mol:
            return
        
        mode = self.output_mode.get()
        use_spectral = self.use_spectral.get()
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # EXTRACT REAL SPECTRAL FREQUENCIES FROM EACH ATOM IN THE MOLECULE
            # This is the PRECISE molecular sound based on:
            # 1. Real spectral lines (Balmer for H, real lines for O, etc.)
            # 2. Phase from bond angles (104.5° for H₂O, 180° for CO₂, etc.)
            # 3. Stereo position from atomic mass
            # ═══════════════════════════════════════════════════════════════════
            
            all_frequencies = []
            all_amplitudes = []
            all_phases = []
            all_positions = []
            
            # Calculate base phases from bond angles
            base_phases = []
            if mol.bond_angles:
                for angle in mol.bond_angles:
                    base_phases.append(self.sounder.angle_to_phase(angle))
            else:
                base_phases = [0.0]
            
            total_mass = mol.total_mass
            
            # Element name mapping for spectral data
            element_names = {
                'H': 'Hydrogen-Balmer',
                'O': 'Oxygen',
                'C': 'Carbon',
                'N': 'Nitrogen',
                'S': 'Sulfur',
                'He': 'Helium',
                'Na': 'Sodium',
                'Ne': 'Neon',
                'Hg': 'Mercury',
                'Fe': 'Iron',
                'Ca': 'Calcium',
            }
            
            # For each atom, get its REAL spectral lines
            for i, atom in enumerate(mol.atoms):
                # Phase from bond angle ONLY for this atom
                # First atom: phase = 0 (reference)
                # Other atoms: phase = bond angle in radians
                if i == 0:
                    atom_base_phase = 0.0  # Reference atom, no phase shift
                else:
                    phase_idx = (i - 1) % len(base_phases)
                    atom_base_phase = base_phases[phase_idx]  # Bond angle phase ONLY
                
                # Stereo position from mass
                pan = self.sounder.mass_to_pan(atom.mass, total_mass)
                position = (pan - 0.5) * 2  # Convert [0,1] to [-1, 1]
                
                if use_spectral:
                    # Try to get REAL spectral lines for this element
                    element_name = element_names.get(atom.symbol)
                    if element_name and self.sounder.spectral_sounder:
                        try:
                            lines = self.sounder.spectral_sounder.get_spectral_lines(element_name)
                            if lines:
                                # Scale to audio frequencies
                                scaled = self.sounder.spectral_sounder.scale_to_audio(lines)
                                
                                # Add each spectral line
                                # Use SAME phases as SpectralTab for consistency!
                                for j, (freq, amp) in enumerate(scaled):
                                    # Phase: bond angle phase for this atom
                                    # (NO extra golden offset - must match single element!)
                                    line_phase = atom_base_phase
                                    
                                    all_frequencies.append(freq)
                                    all_amplitudes.append(amp * 0.8)  # Scale down a bit
                                    all_phases.append(line_phase)
                                    all_positions.append(position)
                                continue
                        except:
                            pass
                
                # Fallback: use bond length frequency if no spectral data
                if mol.bonds and i < len(mol.bonds):
                    bond = mol.bonds[min(i, len(mol.bonds)-1)]
                    freq = self.sounder.bond_length_to_frequency(bond.length)
                else:
                    freq = 200.0 + i * 100.0
                
                amp = self.sounder.electronegativity_to_amplitude(atom.electronegativity)
                
                all_frequencies.append(freq)
                all_amplitudes.append(amp)
                all_phases.append(atom_base_phase)
                all_positions.append(position)
            
            # If binaural mode, create binaural effect
            if mode == "binaural":
                beat = self.beat_freq.get()
                # Add slightly shifted frequencies to alternating channels
                new_freqs = []
                new_amps = []
                new_phases = []
                new_positions = []
                
                for i, (f, a, p, pos) in enumerate(zip(all_frequencies, all_amplitudes, all_phases, all_positions)):
                    # Original on left
                    new_freqs.append(f)
                    new_amps.append(a)
                    new_phases.append(p)
                    new_positions.append(-0.9)
                    
                    # Shifted on right (binaural beat)
                    new_freqs.append(f + beat)
                    new_amps.append(a)
                    new_phases.append(p)
                    new_positions.append(0.9)
                
                all_frequencies = new_freqs
                all_amplitudes = new_amps
                all_phases = new_phases
                all_positions = new_positions
            
            # Start continuous streaming with REAL spectral data
            self.audio.start_spectral(all_frequencies, all_amplitudes, all_phases, all_positions,
                                      master_amplitude=0.7)
            
            self.play_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_var.set(f"🔊 {mol.name}: {len(all_frequencies)} spectral lines")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def _stop(self):
        self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        mol = self.sounder.get_molecule(self.molecule.get())
        if mol:
            self.status_var.set(f"✅ {mol.name}")
    
    def _on_done(self):
        """Legacy callback - not used with continuous playback"""
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        mol = self.sounder.get_molecule(self.molecule.get())
        if mol:
            self.status_var.set(f"✅ {mol.name}")
    
    def _save(self):
        """Save to WAV"""
        formula = self.molecule.get()
        if not formula:
            return
        
        mol = self.sounder.get_molecule(formula)
        if not mol:
            return
        
        filename = f"{formula.lower()}_molecular.wav"
        try:
            left, right = self.sounder.generate_molecule_sound(
                mol, self.duration.get(), use_spectral=self.use_spectral.get())
            self.sounder.save_wav(left, right, filename)
            self.status_var.set(f"💾 Saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _on_param_change(self, *args):
        """Update audio parameters in real-time when settings change - WITH REAL SPECTRAL LINES"""
        if not self.audio.is_playing():
            return
        
        formula = self.molecule.get()
        if not formula:
            return
        
        mol = self.sounder.get_molecule(formula)
        if not mol:
            return
        
        mode = self.output_mode.get()
        use_spectral = self.use_spectral.get()
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # REAL-TIME UPDATE WITH REAL SPECTRAL LINES
            # Same precise logic as _play()
            # ═══════════════════════════════════════════════════════════════════
            
            all_frequencies = []
            all_amplitudes = []
            all_phases = []
            all_positions = []
            
            base_phases = []
            if mol.bond_angles:
                for angle in mol.bond_angles:
                    base_phases.append(self.sounder.angle_to_phase(angle))
            else:
                base_phases = [0.0]
            
            total_mass = mol.total_mass
            
            element_names = {
                'H': 'Hydrogen-Balmer',
                'O': 'Oxygen',
                'C': 'Carbon',
                'N': 'Nitrogen',
                'S': 'Sulfur',
                'He': 'Helium',
                'Na': 'Sodium',
                'Ne': 'Neon',
                'Hg': 'Mercury',
                'Fe': 'Iron',
                'Ca': 'Calcium',
            }
            
            for i, atom in enumerate(mol.atoms):
                phase_idx = i % len(base_phases)
                atom_base_phase = base_phases[phase_idx] + (2 * np.pi * PHI_CONJUGATE * i)
                
                pan = self.sounder.mass_to_pan(atom.mass, total_mass)
                position = (pan - 0.5) * 2
                
                if use_spectral:
                    element_name = element_names.get(atom.symbol)
                    if element_name and self.sounder.spectral_sounder:
                        try:
                            lines = self.sounder.spectral_sounder.get_spectral_lines(element_name)
                            if lines:
                                scaled = self.sounder.spectral_sounder.scale_to_audio(lines)
                                for j, (freq, amp) in enumerate(scaled):
                                    line_phase = atom_base_phase + (2 * np.pi * PHI_CONJUGATE * j)
                                    all_frequencies.append(freq)
                                    all_amplitudes.append(amp * 0.8)
                                    all_phases.append(line_phase)
                                    all_positions.append(position)
                                continue
                        except:
                            pass
                
                # Fallback
                if mol.bonds and i < len(mol.bonds):
                    bond = mol.bonds[min(i, len(mol.bonds)-1)]
                    freq = self.sounder.bond_length_to_frequency(bond.length)
                else:
                    freq = 200.0 + i * 100.0
                
                amp = self.sounder.electronegativity_to_amplitude(atom.electronegativity)
                
                all_frequencies.append(freq)
                all_amplitudes.append(amp)
                all_phases.append(atom_base_phase)
                all_positions.append(position)
            
            if mode == "binaural":
                beat = self.beat_freq.get()
                new_freqs = []
                new_amps = []
                new_phases = []
                new_positions = []
                
                for i, (f, a, p, pos) in enumerate(zip(all_frequencies, all_amplitudes, all_phases, all_positions)):
                    new_freqs.append(f)
                    new_amps.append(a)
                    new_phases.append(p)
                    new_positions.append(-0.9)
                    
                    new_freqs.append(f + beat)
                    new_amps.append(a)
                    new_phases.append(p)
                    new_positions.append(0.9)
                
                all_frequencies = new_freqs
                all_amplitudes = new_amps
                all_phases = new_phases
                all_positions = new_positions
            
            self.audio.set_spectral_params(all_frequencies, all_amplitudes, all_phases, all_positions)
        except:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: HARMONIC TREE - THERAPEUTIC GROWTH MODE
# ══════════════════════════════════════════════════════════════════════════════

class HarmonicTreeTab:
    """
    Generate fundamental + harmonics visualized as a tree with therapeutic growth mode.
    
    Based on natural phyllotaxis patterns:
    - Trunk = Fundamental frequency
    - Branches = Harmonics at Fibonacci ratios (2f, 3f, 5f, 8f, 13f)
    - Branch thickness = Amplitude (φ⁻ⁿ decay)
    - Branch rotation = Phase (cumulative golden angle: n × 137.5°)
    
    Therapeutic Growth Mode:
    - Harmonics emerge progressively over configurable duration (10s to 1hr)
    - Golden-ratio timed cadences: each harmonic emerges at φ⁻ⁿ × duration
    - Each harmonic fades in with golden envelope
    - Phases evolve (rotate) during growth, simulating tree development
    - 10fps animation (can be disabled for CPU saving)
    
    "The universe grows in spirals - so does sound"
    """
    
    def __init__(self, parent, audio_engine: AudioEngine):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        # State - basic
        self.fundamental = tk.DoubleVar(value=432.0)  # Base frequency
        self.num_harmonics = tk.IntVar(value=5)       # Number of harmonics
        self.harmonic_mode = tk.StringVar(value="fibonacci")  # fibonacci or integer
        self.amplitude_decay = tk.StringVar(value="phi")  # phi, sqrt, linear
        self.amplitude = tk.DoubleVar(value=0.7)
        self.stereo_spread = tk.DoubleVar(value=0.5)  # 0 = mono, 1 = full spread
        
        # State - growth mode
        self.growth_mode = tk.BooleanVar(value=True)  # True = progressive, False = instant
        self.growth_duration = tk.IntVar(value=60)    # Duration in seconds
        self.animation_enabled = tk.BooleanVar(value=True)  # 10fps animation toggle
        self.phase_evolution = tk.BooleanVar(value=True)  # Phases rotate during growth
        self.fixed_trunk_mode = tk.BooleanVar(value=True)  # True = fundamental fixed, False = whole tree rotates
        
        # Breathe mode: grow → sustain → fall back to silence
        self.breathe_mode = tk.BooleanVar(value=False)  # Cycle grow/shrink
        self.breathe_cycles = tk.IntVar(value=3)        # Number of breath cycles
        self.sustain_fraction = tk.DoubleVar(value=0.2) # Fraction of cycle to hold at peak
        
        # Growth state tracking
        self._growth_timer = None
        self._growth_start_time = None
        self._current_growth_level = [0.0] * 14  # Per-harmonic growth level (0-1)
        self._is_growing = False
        self._animation_after_id = None
        self._current_cycle = 0
        self._cycle_phase = 'grow'  # 'grow', 'sustain', 'shrink'
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI with scrollable left panel"""
        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL WITH SCROLLBAR
        # ═══════════════════════════════════════════════════════════════════
        left_container = ttk.Frame(self.frame)
        left_container.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Create canvas for scrolling
        self._scroll_canvas = tk.Canvas(left_container, highlightthickness=0, width=420)
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=self._scroll_canvas.yview)
        
        # Scrollable frame inside canvas
        left_frame = ttk.LabelFrame(self._scroll_canvas, text="🌳 Harmonic Tree Controls", padding=10)
        
        # Configure scrolling
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        self._scroll_canvas.pack(side='left', fill='both', expand=True)
        
        # Create window inside canvas for the frame
        self._canvas_window = self._scroll_canvas.create_window((0, 0), window=left_frame, anchor='nw')
        
        # Bind events for scroll region update
        def _configure_scroll_region(event):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        
        def _configure_canvas_width(event):
            self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)
        
        left_frame.bind('<Configure>', _configure_scroll_region)
        self._scroll_canvas.bind('<Configure>', _configure_canvas_width)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self._scroll_canvas.bind_all('<MouseWheel>', _on_mousewheel)  # Windows/Linux
        self._scroll_canvas.bind_all('<Button-4>', lambda e: self._scroll_canvas.yview_scroll(-1, "units"))  # Linux scroll up
        self._scroll_canvas.bind_all('<Button-5>', lambda e: self._scroll_canvas.yview_scroll(1, "units"))   # Linux scroll down
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTROLS CONTENT
        # ═══════════════════════════════════════════════════════════════════
        
        # Fundamental frequency
        fund_frame = ttk.LabelFrame(left_frame, text="🎵 Fundamental (Trunk)", padding=5)
        fund_frame.pack(fill='x', pady=5)
        
        ttk.Label(fund_frame, text="Frequency (Hz):").pack(side='left')
        ttk.Entry(fund_frame, textvariable=self.fundamental, width=8).pack(side='left', padx=5)
        ttk.Scale(fund_frame, from_=20, to=1000, variable=self.fundamental,
                  orient='horizontal', length=200).pack(side='left', padx=5)
        
        # Frequency presets
        preset_frame = ttk.Frame(fund_frame)
        preset_frame.pack(fill='x', pady=3)
        for name, freq in [("C₄", 261.63), ("A₄", 440), ("432Hz", 432), ("φ×100", PHI * 100)]:
            ttk.Button(preset_frame, text=name, width=6,
                      command=lambda f=freq: self.fundamental.set(f)).pack(side='left', padx=2)
        
        # Harmonics configuration
        harm_frame = ttk.LabelFrame(left_frame, text="🌿 Harmonics (Branches)", padding=5)
        harm_frame.pack(fill='x', pady=5)
        
        # Number of harmonics
        num_row = ttk.Frame(harm_frame)
        num_row.pack(fill='x', pady=3)
        ttk.Label(num_row, text="Harmonics:").pack(side='left')
        ttk.Spinbox(num_row, from_=1, to=13, textvariable=self.num_harmonics, width=5).pack(side='left', padx=5)
        
        # Harmonic mode
        mode_row = ttk.Frame(harm_frame)
        mode_row.pack(fill='x', pady=3)
        ttk.Label(mode_row, text="Ratios:").pack(side='left')
        ttk.Radiobutton(mode_row, text="Fibonacci (2,3,5,8,13...)", 
                       variable=self.harmonic_mode, value="fibonacci").pack(side='left', padx=5)
        ttk.Radiobutton(mode_row, text="Integer (2,3,4,5,6...)", 
                       variable=self.harmonic_mode, value="integer").pack(side='left', padx=5)
        
        # Amplitude decay mode
        decay_row = ttk.Frame(harm_frame)
        decay_row.pack(fill='x', pady=3)
        ttk.Label(decay_row, text="Decay:").pack(side='left')
        for name, val in [("φ⁻ⁿ", "phi"), ("1/√n", "sqrt"), ("1/n", "linear")]:
            ttk.Radiobutton(decay_row, text=name, variable=self.amplitude_decay, value=val).pack(side='left', padx=5)
        
        # Phase info
        phase_info = ttk.LabelFrame(left_frame, text="🌀 Phases (Golden Angle)", padding=5)
        phase_info.pack(fill='x', pady=5)
        ttk.Label(phase_info, text="Each harmonic rotated by 137.5° (φ angle)",
                 font=('Courier', 9)).pack()
        ttk.Label(phase_info, text="n=1: 137.5°, n=2: 275°, n=3: 412.5° (≡52.5°)...",
                 font=('Courier', 9), foreground='#888').pack()
        
        # ═══════════════════════════════════════════════════════════════════
        # THERAPEUTIC GROWTH MODE
        # ═══════════════════════════════════════════════════════════════════
        growth_frame = ttk.LabelFrame(left_frame, text="🌱 Therapeutic Growth Mode", padding=5)
        growth_frame.pack(fill='x', pady=5)
        
        # Growth mode toggle
        growth_toggle = ttk.Frame(growth_frame)
        growth_toggle.pack(fill='x', pady=3)
        ttk.Checkbutton(growth_toggle, text="Progressive Growth (harmonics emerge over time)", 
                       variable=self.growth_mode).pack(side='left')
        
        # Duration selector
        duration_row = ttk.Frame(growth_frame)
        duration_row.pack(fill='x', pady=3)
        ttk.Label(duration_row, text="Duration:").pack(side='left')
        
        # Duration presets
        duration_presets = [
            ("10s", 10), ("30s", 30), ("1m", 60), ("5m", 300),
            ("10m", 600), ("30m", 1800), ("1h", 3600)
        ]
        for name, secs in duration_presets:
            ttk.Button(duration_row, text=name, width=4,
                      command=lambda s=secs: self.growth_duration.set(s)).pack(side='left', padx=1)
        
        # Custom duration entry
        custom_row = ttk.Frame(growth_frame)
        custom_row.pack(fill='x', pady=2)
        ttk.Label(custom_row, text="Custom (sec):").pack(side='left')
        ttk.Entry(custom_row, textvariable=self.growth_duration, width=6).pack(side='left', padx=5)
        self.duration_label = ttk.Label(custom_row, text="", font=('Courier', 9))
        self.duration_label.pack(side='left', padx=5)
        self._update_duration_label()
        
        # Animation toggle
        anim_row = ttk.Frame(growth_frame)
        anim_row.pack(fill='x', pady=3)
        ttk.Checkbutton(anim_row, text="Animate tree (10 fps)", 
                       variable=self.animation_enabled).pack(side='left')
        ttk.Label(anim_row, text="(disable to save CPU)", 
                 font=('Courier', 8), foreground='#888').pack(side='left', padx=5)
        
        # Phase evolution toggle
        phase_row = ttk.Frame(growth_frame)
        phase_row.pack(fill='x', pady=3)
        ttk.Checkbutton(phase_row, text="Evolve phases during growth (rotating branches)", 
                       variable=self.phase_evolution).pack(side='left')
        
        # Fixed trunk mode toggle (only visible when phase evolution is enabled)
        trunk_row = ttk.Frame(growth_frame)
        trunk_row.pack(fill='x', pady=3)
        ttk.Label(trunk_row, text="    Rotation mode:", font=('Courier', 9)).pack(side='left')
        ttk.Radiobutton(trunk_row, text="🌲 Fixed Trunk", variable=self.fixed_trunk_mode,
                       value=True).pack(side='left', padx=5)
        ttk.Radiobutton(trunk_row, text="🌀 Whole Tree", variable=self.fixed_trunk_mode,
                       value=False).pack(side='left', padx=5)
        ttk.Label(trunk_row, text="(fundamental phase)", font=('Courier', 8), 
                 foreground='#888').pack(side='left', padx=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # BREATHE MODE: Grow → Sustain → Shrink → Silence → Repeat
        # ═══════════════════════════════════════════════════════════════════
        breathe_frame = ttk.LabelFrame(growth_frame, text="🌬️ Breathe Mode (Grow ↔ Shrink)", padding=3)
        breathe_frame.pack(fill='x', pady=3)
        
        breathe_toggle = ttk.Frame(breathe_frame)
        breathe_toggle.pack(fill='x', pady=2)
        ttk.Checkbutton(breathe_toggle, text="Enable breathing (cycle grow → fall back)", 
                       variable=self.breathe_mode).pack(side='left')
        
        breathe_params = ttk.Frame(breathe_frame)
        breathe_params.pack(fill='x', pady=2)
        ttk.Label(breathe_params, text="Cycles:").pack(side='left')
        ttk.Spinbox(breathe_params, from_=1, to=20, textvariable=self.breathe_cycles, 
                   width=4).pack(side='left', padx=3)
        ttk.Label(breathe_params, text="Sustain:").pack(side='left', padx=(10,0))
        ttk.Scale(breathe_params, from_=0.05, to=0.5, variable=self.sustain_fraction,
                 orient='horizontal', length=80).pack(side='left', padx=3)
        ttk.Label(breathe_params, text="(hold at peak)", font=('Courier', 8), 
                 foreground='#888').pack(side='left')
        
        # Growth progress bar
        self.growth_progress = ttk.Progressbar(growth_frame, mode='determinate', length=250)
        self.growth_progress.pack(fill='x', pady=5)
        self.growth_status = ttk.Label(growth_frame, text="", font=('Courier', 9))
        self.growth_status.pack()
        
        # Master controls
        master_frame = ttk.LabelFrame(left_frame, text="🎚️ Master", padding=5)
        master_frame.pack(fill='x', pady=5)
        
        amp_row = ttk.Frame(master_frame)
        amp_row.pack(fill='x', pady=2)
        ttk.Label(amp_row, text="Amplitude:").pack(side='left')
        ttk.Scale(amp_row, from_=0, to=1, variable=self.amplitude,
                  orient='horizontal', length=150).pack(side='left', padx=5)
        
        stereo_row = ttk.Frame(master_frame)
        stereo_row.pack(fill='x', pady=2)
        ttk.Label(stereo_row, text="Stereo Spread:").pack(side='left')
        ttk.Scale(stereo_row, from_=0, to=1, variable=self.stereo_spread,
                  orient='horizontal', length=150).pack(side='left', padx=5)
        
        # Playback buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ GROW & PLAY", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Right panel - Tree Visualization
        right_frame = ttk.LabelFrame(self.frame, text="🌳 Tree Visualization", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=400, height=400, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        # Info panel
        self.info_text = tk.Text(right_frame, width=45, height=8, bg='#0a0a15',
                                fg='#00ff88', font=('Courier', 9), state='disabled')
        self.info_text.pack(fill='x')
        
        self.status_var = tk.StringVar(value="Configure harmonics and press GROW & PLAY")
        ttk.Label(right_frame, textvariable=self.status_var).pack()
        
        # Bind updates
        self.fundamental.trace_add('write', self._on_param_change)
        self.num_harmonics.trace_add('write', self._on_param_change)
        self.harmonic_mode.trace_add('write', self._on_param_change)
        self.amplitude_decay.trace_add('write', self._on_param_change)
        self.amplitude.trace_add('write', self._on_param_change)
        self.stereo_spread.trace_add('write', self._on_param_change)
        self.growth_duration.trace_add('write', lambda *a: self._update_duration_label())
        
        # Initial draw
        self._draw_tree()
    
    def _update_duration_label(self):
        """Update the duration label with human-readable format"""
        try:
            secs = self.growth_duration.get()
            if secs >= 3600:
                label = f"= {secs/3600:.1f} hours"
            elif secs >= 60:
                label = f"= {secs/60:.1f} minutes"
            else:
                label = f"= {secs} seconds"
            self.duration_label.config(text=label)
        except:
            pass
    
    def _calculate_growth_schedule(self):
        """
        Calculate when each harmonic should emerge using SPREAD golden ratio cadences.
        
        Returns list of (harmonic_index, emergence_time_fraction) tuples.
        
        FIXED TIMING: Harmonics now spread evenly across the duration with golden weighting.
        Each harmonic gets a fair portion of time to emerge and grow.
        """
        n = self.num_harmonics.get() + 1  # Include fundamental
        
        # More spread-out timing: divide duration into n segments
        # but weight by golden ratio for natural feel
        schedule = [(0, 0.0)]  # Fundamental at t=0
        
        if n <= 1:
            return schedule
        
        # Spread harmonics across 0% to 80% of duration
        # This gives each harmonic time to fully grow
        max_emergence = 0.80  # Last harmonic starts at 80%, has 20% to grow
        
        for i in range(1, n):
            # Linear spread with slight golden weighting
            # Each harmonic gets roughly equal time
            base_fraction = i / n  # Linear: 0.2, 0.4, 0.6, 0.8 for 5 harmonics
            
            # Apply subtle golden modulation (not too aggressive)
            golden_weight = 1.0 - (PHI_CONJUGATE ** (i * 0.5)) * 0.3
            
            emergence = base_fraction * max_emergence * golden_weight
            schedule.append((i, emergence))
        
        return schedule
    
    def _calculate_fade_envelope(self, elapsed_fraction, emergence_fraction, harmonic_index):
        """
        Calculate golden-ratio fade envelope for a harmonic.
        
        FIXED: Longer fade-in durations so harmonics grow more gracefully.
        All harmonics get minimum 10% of total duration to fade in.
        
        Returns amplitude multiplier (0.0 to 1.0)
        """
        if elapsed_fraction < emergence_fraction:
            return 0.0  # Not yet emerged
        
        time_since_emergence = elapsed_fraction - emergence_fraction
        
        # FIXED: Much longer fade-in durations
        # Base duration of 15% of total, with slight golden reduction for later harmonics
        # Minimum 10% fade-in for all harmonics
        base_fade = 0.15
        golden_factor = 1.0 - (harmonic_index * 0.02)  # Only 2% reduction per harmonic
        fade_in_duration = max(0.10, base_fade * golden_factor)
        
        if time_since_emergence >= fade_in_duration:
            return 1.0  # Fully grown
        
        # Smooth cosine fade-in (more natural than linear)
        progress = time_since_emergence / fade_in_duration
        # Cosine easing: slow start, fast middle, slow end
        return 0.5 * (1 - np.cos(np.pi * progress))
    
    def _calculate_harmonics(self, apply_growth=False, elapsed_fraction=0.0):
        """
        Calculate harmonic frequencies, amplitudes, phases, and stereo positions.
        
        Args:
            apply_growth: If True, modulate amplitudes based on growth state
            elapsed_fraction: Current progress through growth cycle (0.0 to 1.0)
        
        Returns:
            (frequencies, amplitudes, phases, positions)
        """
        fund = self.fundamental.get()
        n = self.num_harmonics.get()
        mode = self.harmonic_mode.get()
        decay = self.amplitude_decay.get()
        spread = self.stereo_spread.get()
        
        # ═══════════════════════════════════════════════════════════════════
        # FREQUENCIES: Fibonacci or Integer ratios
        # ═══════════════════════════════════════════════════════════════════
        
        if mode == "fibonacci":
            # Use Fibonacci ratios: 2, 3, 5, 8, 13, 21...
            fib_ratios = [FIBONACCI[i+2] for i in range(n)]  # Start from F(3)=2
            frequencies = [fund] + [fund * r for r in fib_ratios]
        else:
            # Integer harmonics: 2, 3, 4, 5, 6...
            frequencies = [fund] + [fund * (i + 2) for i in range(n)]
        
        # ═══════════════════════════════════════════════════════════════════
        # AMPLITUDES: Various decay modes
        # ═══════════════════════════════════════════════════════════════════
        
        total = len(frequencies)
        if decay == "phi":
            # Golden ratio decay: φ⁻ⁿ
            amplitudes = [PHI_CONJUGATE ** i for i in range(total)]
        elif decay == "sqrt":
            # 1/√n decay (gentler)
            amplitudes = [1.0 / np.sqrt(i + 1) for i in range(total)]
        else:
            # 1/n decay (classic)
            amplitudes = [1.0 / (i + 1) for i in range(total)]
        
        # ═══════════════════════════════════════════════════════════════════
        # GROWTH MODE: Progressive amplitude envelope
        # ═══════════════════════════════════════════════════════════════════
        
        if apply_growth and self.growth_mode.get():
            schedule = self._calculate_growth_schedule()
            for i in range(total):
                emergence = schedule[i][1] if i < len(schedule) else 0.95
                envelope = self._calculate_fade_envelope(elapsed_fraction, emergence, i)
                amplitudes[i] *= envelope
                self._current_growth_level[i] = envelope
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASES: Cumulative Golden Angle (phyllotaxis pattern)
        # ═══════════════════════════════════════════════════════════════════
        
        # Each harmonic is rotated by n × 137.5°
        # This is the pattern found in sunflower seeds!
        base_phases = [(i * GOLDEN_ANGLE_RAD) % (2 * np.pi) for i in range(total)]
        
        # Phase evolution during growth (rotating branches)
        if apply_growth and self.phase_evolution.get():
            # Rotate phases over time - slower for later harmonics
            phase_offset = elapsed_fraction * 2 * np.pi
            
            if self.fixed_trunk_mode.get():
                # Mode: Fixed Trunk - Fundamental stays at 0°, harmonics rotate
                phases = [0.0]  # Fundamental fixed at phase 0
                phases += [(base_phases[i] + phase_offset * PHI_CONJUGATE ** i) % (2 * np.pi) 
                          for i in range(1, total)]
            else:
                # Mode: Whole Tree - All phases rotate including fundamental
                phases = [(base_phases[i] + phase_offset * PHI_CONJUGATE ** i) % (2 * np.pi) 
                         for i in range(total)]
        else:
            phases = base_phases
        
        # ═══════════════════════════════════════════════════════════════════
        # STEREO: Spiral positioning based on golden angle
        # ═══════════════════════════════════════════════════════════════════
        
        # Fundamental at center, harmonics spiral outward
        positions = []
        for i in range(total):
            # Use golden angle for stereo position
            angle = i * GOLDEN_ANGLE_RAD
            if apply_growth and self.phase_evolution.get():
                angle += elapsed_fraction * np.pi * 0.5  # Subtle stereo movement
            pan = np.sin(angle) * spread  # -spread to +spread
            positions.append(pan)
        
        return frequencies, amplitudes, phases, positions
    
    def _draw_tree(self, elapsed_fraction=0.0):
        """
        Draw 3D isometric harmonic tree visualization.
        
        Features:
        - Isometric 3D projection for depth
        - Sound → Light spectrum colors (synesthesia mapping)
        - Swiss typography (Helvetica, clean grid)
        - Golden angle phyllotaxis pattern
        """
        self.canvas.delete('all')
        
        is_growing = self._is_growing and self.growth_mode.get()
        frequencies, amplitudes, phases, positions = self._calculate_harmonics(
            apply_growth=is_growing, elapsed_fraction=elapsed_fraction)
        
        # ═══════════════════════════════════════════════════════════════════
        # SWISS DESIGN: Clean dark background with subtle grid
        # ═══════════════════════════════════════════════════════════════════
        self.canvas.configure(bg='#0a0a0f')  # Deep dark blue-black
        
        # Subtle grid for Swiss design aesthetic
        grid_color = '#15151f'
        for x in range(0, 400, 40):
            self.canvas.create_line(x, 0, x, 400, fill=grid_color, width=1)
        for y in range(0, 400, 40):
            self.canvas.create_line(0, y, 400, y, fill=grid_color, width=1)
        
        # Canvas center for 3D projection
        cx, cy = 200, 320
        
        # ═══════════════════════════════════════════════════════════════════
        # 3D ISOMETRIC PROJECTION HELPERS
        # ═══════════════════════════════════════════════════════════════════
        def iso_project(x3d, y3d, z3d):
            """Convert 3D coords to 2D isometric projection"""
            # Isometric angles: 30° for x-axis, -30° for z-axis
            iso_angle = np.radians(30)
            x2d = cx + (x3d - z3d) * np.cos(iso_angle)
            y2d = cy - y3d - (x3d + z3d) * np.sin(iso_angle) * 0.5
            return x2d, y2d
        
        # ═══════════════════════════════════════════════════════════════════
        # 3D GROUND PLANE (subtle reference)
        # ═══════════════════════════════════════════════════════════════════
        ground_color = '#1a1a2a'
        for r in [60, 100, 140]:
            points = []
            for angle in range(0, 360, 15):
                rad = np.radians(angle)
                x3d, z3d = r * np.cos(rad), r * np.sin(rad)
                x2d, y2d = iso_project(x3d, 0, z3d)
                points.extend([x2d, y2d])
            if len(points) >= 4:
                self.canvas.create_polygon(points, outline=ground_color, fill='', width=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # TRUNK (Fundamental frequency) - 3D cylinder effect
        # ═══════════════════════════════════════════════════════════════════
        trunk_growth = self._current_growth_level[0] if is_growing else 1.0
        trunk_height = 120 * trunk_growth
        trunk_width = max(3, 12 * amplitudes[0] * trunk_growth)
        
        # Trunk color based on fundamental frequency
        trunk_color = sound_to_light_color(frequencies[0])
        trunk_shadow = '#3d2817'
        
        if trunk_height > 0:
            # Draw trunk with 3D shading (left side darker)
            x2d_base, y2d_base = iso_project(0, 0, 0)
            x2d_top, y2d_top = iso_project(0, trunk_height, 0)
            
            # Shadow side
            offset = trunk_width * 0.3
            self.canvas.create_line(x2d_base - offset, y2d_base,
                                   x2d_top - offset, y2d_top,
                                   fill=trunk_shadow, width=trunk_width * 0.7)
            # Main trunk
            self.canvas.create_line(x2d_base, y2d_base, x2d_top, y2d_top,
                                   fill=trunk_color, width=trunk_width,
                                   capstyle='round')
        
        # ═══════════════════════════════════════════════════════════════════
        # BRANCHES (Harmonics) - 3D Golden Angle Phyllotaxis
        # ═══════════════════════════════════════════════════════════════════
        branch_origin_y = trunk_height
        
        # Collect branches for depth sorting (draw back-to-front)
        branches = []
        
        for i in range(1, len(frequencies)):
            growth = self._current_growth_level[i] if is_growing else 1.0
            
            if growth < 0.01:
                continue
            
            # Golden angle rotation (137.5° per branch)
            golden_angle = i * GOLDEN_ANGLE_DEG
            angle_rad = np.radians(golden_angle)
            
            # 3D spiral: branches grow outward and upward
            # Radius increases with sqrt(i) for Fermat spiral
            spiral_radius = 25 * np.sqrt(i) * growth
            
            # Height increases with golden ratio for each layer
            layer_height = branch_origin_y + (i * 15 * PHI_CONJUGATE) * growth
            
            # 3D position of branch tip
            x3d = spiral_radius * np.cos(angle_rad)
            z3d = spiral_radius * np.sin(angle_rad)
            y3d = layer_height + 20 * growth  # Slight upward tilt
            
            # Branch length and width
            branch_length = (40 + 30 * amplitudes[i]) * growth
            branch_width = max(1, (8 - i * 0.3) * growth)
            
            # SOUND → LIGHT COLOR MAPPING
            color = sound_to_light_color(frequencies[i])
            
            # Fade color during emergence
            if growth < 1.0:
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                r, g, b = int(r * growth), int(g * growth), int(b * growth)
                color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Calculate depth for sorting (z-order)
            depth = z3d
            
            branches.append({
                'i': i,
                'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
                'length': branch_length,
                'width': branch_width,
                'color': color,
                'growth': growth,
                'freq': frequencies[i],
                'depth': depth
            })
        
        # Sort by depth (back to front)
        branches.sort(key=lambda b: b['depth'])
        
        # Draw branches
        for branch in branches:
            i = branch['i']
            x3d, y3d, z3d = branch['x3d'], branch['y3d'], branch['z3d']
            
            # Branch start (at trunk)
            start_x2d, start_y2d = iso_project(0, branch_origin_y + i * 5, 0)
            
            # Branch end (tip)
            end_x2d, end_y2d = iso_project(x3d, y3d, z3d)
            
            # Draw branch line
            self.canvas.create_line(start_x2d, start_y2d, end_x2d, end_y2d,
                                   fill=branch['color'], width=branch['width'],
                                   capstyle='round')
            
            # Draw node/sphere at tip (3D effect with highlight)
            node_size = (4 + amplitudes[0] * 6) * branch['growth']
            if node_size >= 2:
                # Shadow
                self.canvas.create_oval(
                    end_x2d - node_size + 1, end_y2d - node_size + 1,
                    end_x2d + node_size + 1, end_y2d + node_size + 1,
                    fill='#000000', outline='')
                # Main node
                self.canvas.create_oval(
                    end_x2d - node_size, end_y2d - node_size,
                    end_x2d + node_size, end_y2d + node_size,
                    fill=branch['color'], outline='')
                # Highlight
                hl_size = node_size * 0.4
                self.canvas.create_oval(
                    end_x2d - hl_size - node_size*0.3, end_y2d - hl_size - node_size*0.3,
                    end_x2d + hl_size - node_size*0.3, end_y2d + hl_size - node_size*0.3,
                    fill='#ffffff', outline='')
            
            # Swiss typography labels (Helvetica, clean positioning)
            if i <= 5 and branch['growth'] > 0.6:
                label_offset = 18
                label_x = end_x2d + (label_offset if end_x2d > cx else -label_offset)
                anchor = 'w' if end_x2d > cx else 'e'
                self.canvas.create_text(label_x, end_y2d,
                                       text=f"{branch['freq']:.0f}",
                                       fill=branch['color'], 
                                       font=('Helvetica Neue', 9),
                                       anchor=anchor)
        
        # ═══════════════════════════════════════════════════════════════════
        # GOLDEN SPIRAL (3D decorative element)
        # ═══════════════════════════════════════════════════════════════════
        if trunk_height > 40:
            spiral_points = []
            spiral_growth = elapsed_fraction if is_growing else 1.0
            num_points = int(40 * spiral_growth)
            for j in range(max(2, num_points)):
                angle = j * GOLDEN_ANGLE_RAD * 0.4
                r = 6 * np.sqrt(j * 0.3) * spiral_growth
                x3d = r * np.cos(angle)
                z3d = r * np.sin(angle)
                y3d = trunk_height * 0.7 + j * 0.5
                x2d, y2d = iso_project(x3d, y3d, z3d)
                spiral_points.extend([x2d, y2d])
            
            if len(spiral_points) >= 4:
                self.canvas.create_line(spiral_points, fill='#ffd700', 
                                       width=1.5, smooth=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # SWISS TYPOGRAPHY: Title and Info
        # ═══════════════════════════════════════════════════════════════════
        mode = self.harmonic_mode.get()
        
        # Minimal Swiss-style title
        if is_growing:
            percent = int(elapsed_fraction * 100)
            title = f"{percent}%"
            subtitle = f"growing · {mode}"
        else:
            title = "φ"
            subtitle = f"harmonic tree · {mode}"
        
        # Title - large, bold
        self.canvas.create_text(20, 20, text=title,
                               fill='#ffffff', font=('Helvetica Neue', 24, 'bold'),
                               anchor='nw')
        # Subtitle - small, light
        self.canvas.create_text(20, 50, text=subtitle,
                               fill='#888888', font=('Helvetica Neue', 10),
                               anchor='nw')
        
        # Fundamental frequency display
        fund_color = sound_to_light_color(frequencies[0])
        self.canvas.create_text(20, 380, text=f"ƒ₀ = {frequencies[0]:.1f} Hz",
                               fill=fund_color, font=('Helvetica Neue', 11),
                               anchor='sw')
        
        # Color legend (small, bottom right)
        legend_x = 380
        legend_y = 380
        self.canvas.create_text(legend_x, legend_y, 
                               text="sound→light",
                               fill='#555555', font=('Helvetica Neue', 8),
                               anchor='se')
    
    def _draw_golden_spiral(self, cx, cy, growth=1.0):
        """Draw a small golden spiral at the center"""
        points = []
        scale = 8 * growth
        num_points = int(50 * growth)
        for i in range(max(2, num_points)):
            angle = i * GOLDEN_ANGLE_RAD * 0.3
            r = scale * np.sqrt(i * 0.5)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.extend([x, y])
        
        if len(points) >= 4:
            self.canvas.create_line(points, fill='#ffd700', width=1, smooth=True)
    
    def _update_info(self, elapsed_fraction=0.0):
        """Update the info panel with current harmonics"""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        
        is_growing = self._is_growing and self.growth_mode.get()
        frequencies, amplitudes, phases, positions = self._calculate_harmonics(
            apply_growth=is_growing, elapsed_fraction=elapsed_fraction)
        
        mode = self.harmonic_mode.get()
        decay = self.amplitude_decay.get()
        
        if is_growing:
            self.info_text.insert('end', f"═══ HARMONIC TREE (Growing {int(elapsed_fraction*100)}%) ═══\n\n")
        else:
            self.info_text.insert('end', f"═══ HARMONIC TREE ({mode}/{decay}) ═══\n\n")
        
        self.info_text.insert('end', f"{'#':<3} {'Freq (Hz)':<10} {'Amp':<8} {'Phase°':<10} {'Growth':<8}\n")
        self.info_text.insert('end', "─" * 45 + "\n")
        
        for i, (f, a, p, pos) in enumerate(zip(frequencies, amplitudes, phases, positions)):
            phase_deg = np.degrees(p)
            growth = self._current_growth_level[i] if is_growing else 1.0
            growth_str = f"{growth*100:.0f}%" if is_growing else "100%"
            name = "Fund" if i == 0 else f"H{i}"
            
            # Show emerged vs not-emerged
            if growth < 0.01:
                self.info_text.insert('end', f"{name:<3} {'(emerging...)':<38}\n")
            else:
                self.info_text.insert('end', 
                    f"{name:<3} {f:<10.1f} {a:<8.3f} {phase_deg:<10.1f} {growth_str:<8}\n")
        
        self.info_text.config(state='disabled')
    
    def _on_param_change(self, *args):
        """Handle parameter changes - update visualization and audio"""
        try:
            elapsed_fraction = 0.0
            if self._is_growing and self._growth_start_time:
                elapsed = time.time() - self._growth_start_time
                duration = self.growth_duration.get()
                elapsed_fraction = min(elapsed / duration, 1.0)
            
            self._draw_tree(elapsed_fraction)
            self._update_info(elapsed_fraction)
            
            # If playing, update audio in real-time
            if self.audio.is_playing():
                self._update_audio(elapsed_fraction)
        except:
            pass
    
    def _update_audio(self, elapsed_fraction=0.0):
        """Update audio parameters in real-time"""
        is_growing = self._is_growing and self.growth_mode.get()
        frequencies, amplitudes, phases, positions = self._calculate_harmonics(
            apply_growth=is_growing, elapsed_fraction=elapsed_fraction)
        self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
    
    def _play(self):
        """Start playing the harmonic tree with optional growth mode"""
        amp = self.amplitude.get()
        
        # Reset growth state
        self._current_growth_level = [0.0] * 14
        self._is_growing = self.growth_mode.get()
        self._growth_start_time = time.time()
        
        if self._is_growing:
            # Start with only fundamental, harmonics will emerge
            self._current_growth_level[0] = 1.0  # Fundamental starts immediately
            frequencies, amplitudes, phases, positions = self._calculate_harmonics(
                apply_growth=True, elapsed_fraction=0.0)
            
            # Start audio with initial state
            self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                      master_amplitude=amp)
            
            # Start growth timer
            self._growth_callback()
            
            duration = self.growth_duration.get()
            self.status_var.set(f"🌱 Growing over {duration}s - Tree emerging...")
        else:
            # Instant mode - all harmonics at once
            for i in range(14):
                self._current_growth_level[i] = 1.0
            frequencies, amplitudes, phases, positions = self._calculate_harmonics()
            
            self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                      master_amplitude=amp)
            
            fund = self.fundamental.get()
            n = self.num_harmonics.get()
            self.status_var.set(f"🔊 Playing: {fund:.1f} Hz + {n} harmonics")
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
    
    def _calculate_breathe_envelope(self, elapsed_fraction):
        """
        Calculate envelope for breathe mode (grow → sustain → shrink).
        
        One cycle:
        - 40% grow (0.0 → 1.0)
        - 20% sustain (hold at 1.0)  
        - 40% shrink (1.0 → 0.0)
        
        Returns growth multiplier (0.0 to 1.0)
        """
        sustain = self.sustain_fraction.get()
        grow_fraction = (1.0 - sustain) / 2
        shrink_fraction = grow_fraction
        
        if elapsed_fraction < grow_fraction:
            # Growing phase
            t = elapsed_fraction / grow_fraction
            return 0.5 * (1 - np.cos(np.pi * t))  # Smooth ease
        elif elapsed_fraction < grow_fraction + sustain:
            # Sustain phase (hold at peak)
            return 1.0
        else:
            # Shrinking phase
            t = (elapsed_fraction - grow_fraction - sustain) / shrink_fraction
            return 0.5 * (1 + np.cos(np.pi * t))  # Smooth ease down
    
    def _growth_callback(self):
        """Timer callback for growth animation - runs at 10fps. Supports breathe mode."""
        if not self._is_growing or not self.audio.is_playing():
            return
        
        elapsed = time.time() - self._growth_start_time
        duration = self.growth_duration.get()
        
        if self.breathe_mode.get():
            # Breathe mode: multiple cycles of grow/shrink
            total_cycles = self.breathe_cycles.get()
            cycle_duration = duration / total_cycles
            
            # Which cycle are we in?
            current_cycle = int(elapsed / cycle_duration)
            cycle_elapsed = elapsed % cycle_duration
            cycle_fraction = cycle_elapsed / cycle_duration
            
            # Calculate envelope within this cycle
            breathe_envelope = self._calculate_breathe_envelope(cycle_fraction)
            
            # Apply to all growth levels uniformly (simpler for breathe)
            for i in range(14):
                self._current_growth_level[i] = breathe_envelope
            
            # Overall progress
            overall_fraction = elapsed / duration
            self.growth_progress['value'] = overall_fraction * 100
            
            # Status
            phase = "🌱" if cycle_fraction < 0.4 else ("💚" if cycle_fraction < 0.6 else "🍂")
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            total_min, total_sec = divmod(duration, 60)
            self.growth_status.config(
                text=f"{phase} Cycle {current_cycle+1}/{total_cycles} | {elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}")
            
            # Update audio and visualization
            self._update_audio(breathe_envelope)  # Use envelope directly
            
            if self.animation_enabled.get():
                self._draw_tree(breathe_envelope)
                self._update_info(breathe_envelope)
            
            # Check if all cycles complete
            if elapsed >= duration:
                self._is_growing = False
                self.status_var.set(f"🌬️ Breathing complete - {total_cycles} cycles")
                self.growth_status.config(text="Breathing complete ✓")
                self._stop()
                return
        else:
            # Normal grow-only mode
            elapsed_fraction = min(elapsed / duration, 1.0)
            
            # Update progress bar
            self.growth_progress['value'] = elapsed_fraction * 100
            
            # Format elapsed time
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            total_min, total_sec = divmod(duration, 60)
            self.growth_status.config(text=f"{elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}")
            
            # Update audio with new growth state
            self._update_audio(elapsed_fraction)
            
            # Update visualization (if animation enabled)
            if self.animation_enabled.get():
                self._draw_tree(elapsed_fraction)
                self._update_info(elapsed_fraction)
            
            if elapsed_fraction >= 1.0:
                # Growth complete
                self._is_growing = False
                self.status_var.set(f"🌳 Fully grown - {self.num_harmonics.get()+1} harmonics playing")
                self.growth_status.config(text="Growth complete ✓")
                # Set all growth levels to 1.0
                for i in range(14):
                    self._current_growth_level[i] = 1.0
                # Final update
                self._draw_tree(1.0)
                self._update_info(1.0)
                return
        
        # Schedule next frame (100ms = 10fps)
        self._animation_after_id = self.frame.after(100, self._growth_callback)
    
    def _stop(self):
        """Stop playback"""
        self._is_growing = False
        if self._animation_after_id:
            self.frame.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        
        self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.growth_progress['value'] = 0
        self.growth_status.config(text="")
        
        # Reset growth levels for display
        for i in range(14):
            self._current_growth_level[i] = 1.0
        self._draw_tree()
        self._update_info()
        
        self.status_var.set("Configure harmonics and press GROW & PLAY")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: VIBROACOUSTIC SOUNDBOARD - Physical Panning for Head-Feet Axis
# ══════════════════════════════════════════════════════════════════════════════

class VibroacousticTab:
    """
    Vibroacoustic soundboard panning for therapy table with 2 exciters.
    
    Physical setup:
    - Spruce board: 2000mm × 600mm × 10mm on springs
    - Exciters: HEAD (0mm) and FEET (2000mm) on short edges
    - Listener: lying centered, ears ~150mm from head edge
    - Springs: 5× (4 corners + 1 center), 15-20kg each for floor decoupling
    
    Panning model uses:
    - ITD (Interaural Time Difference): Delay based on sound velocity in spruce
    - ILD (Interaural Level Difference): Equal-power + soft distance attenuation
    
    "Feel the sound travel through your body"
    """
    
    def __init__(self, parent, audio_engine: AudioEngine):
        self.parent = parent
        self.audio = audio_engine
        self.frame = ttk.Frame(parent)
        
        # Initialize soundboard panner
        if HAS_SOUNDBOARD:
            self.config = SoundboardConfig(sample_rate=SAMPLE_RATE)
            self.panner = SoundboardPanner(self.config)
        else:
            self.config = None
            self.panner = None
        
        # State
        self.pan_position = tk.DoubleVar(value=0.0)  # -1 (head) to +1 (feet)
        self.frequency = tk.DoubleVar(value=432.0)   # Base frequency
        self.amplitude = tk.DoubleVar(value=0.7)
        self.waveform = tk.StringVar(value="sine")
        
        # Auto-sweep mode
        self.sweep_enabled = tk.BooleanVar(value=False)
        self.sweep_duration = tk.DoubleVar(value=10.0)  # seconds per sweep
        self.sweep_mode = tk.StringVar(value="sine")  # sine, linear, golden
        
        # Sweep state
        self._sweep_timer = None
        self._sweep_start_time = None
        self._is_sweeping = False
        
        # ═══════════════════════════════════════════════════════════════════
        # CHAKRA SUNRISE STATE
        # ═══════════════════════════════════════════════════════════════════
        self._is_chakra_playing = False
        self._chakra_timer = None
        self._chakra_start_time = None
        self._journey_duration = 60.0  # Default 1 minute
        
        # Three frequencies for Chakra Sunrise
        self._freq_fourth = 0.0    # Perfect 4th (base × 4/3)
        self._freq_root = 0.0      # Root/fundamental (base × 1)
        self._freq_octave = 0.0    # Octave (base × 2)
        
        # Amplitude and pan for each frequency (smoothly interpolated by AudioEngine)
        self._fourth_amp = 0.0
        self._fourth_pan = 0.0
        self._root_amp = 0.0
        self._root_pan = 0.0
        self._octave_amp = 0.0
        self._octave_pan = 0.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI"""
        if not HAS_SOUNDBOARD:
            ttk.Label(self.frame, text="⚠️ soundboard_panning.py not found").pack(pady=50)
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL - Controls
        # ═══════════════════════════════════════════════════════════════════
        left_frame = ttk.LabelFrame(self.frame, text="🪵 Vibroacoustic Controls", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Physical setup info
        info_frame = ttk.LabelFrame(left_frame, text="📐 Physical Setup", padding=5)
        info_frame.pack(fill='x', pady=5)
        
        info_text = f"""Board: {self.config.length_mm:.0f}×{self.config.width_mm:.0f}mm spruce
Velocity: {self.config.velocity_ms:.0f} m/s (along fiber)
Max delay: {self.config.max_delay_ms:.3f} ms ({self.config.max_delay_samples} samples)
Springs: 5× (4 corners + 1 center)"""
        
        ttk.Label(info_frame, text=info_text, font=('Courier', 9), 
                 foreground='#888').pack(anchor='w')
        
        # ═══════════════════════════════════════════════════════════════════
        # PAN CONTROL - Main slider (vertical for head-feet axis)
        # ═══════════════════════════════════════════════════════════════════
        pan_frame = ttk.LabelFrame(left_frame, text="🎚️ Pan Position (Head ↔ Feet)", padding=10)
        pan_frame.pack(fill='x', pady=5)
        
        # Horizontal layout with labels
        pan_row = ttk.Frame(pan_frame)
        pan_row.pack(fill='x', pady=5)
        
        ttk.Label(pan_row, text="🧠 HEAD", font=('Helvetica', 10, 'bold'),
                 foreground='#ff6b6b').pack(side='left', padx=5)
        
        self.pan_scale = ttk.Scale(pan_row, from_=-1, to=1, variable=self.pan_position,
                                   orient='horizontal', length=250,
                                   command=self._on_pan_change)
        self.pan_scale.pack(side='left', padx=10)
        
        ttk.Label(pan_row, text="🦶 FEET", font=('Helvetica', 10, 'bold'),
                 foreground='#4ecdc4').pack(side='left', padx=5)
        
        # Pan value display
        self.pan_label = ttk.Label(pan_frame, text="Pan: 0.00 (CENTER)", 
                                   font=('Courier', 11, 'bold'))
        self.pan_label.pack(pady=5)
        
        # Quick pan presets
        preset_row = ttk.Frame(pan_frame)
        preset_row.pack(fill='x', pady=3)
        
        presets = [
            ("🧠 Head", -1.0),
            ("↑ Upper", -0.5),
            ("● Center", 0.0),
            ("↓ Lower", 0.5),
            ("🦶 Feet", 1.0),
        ]
        
        for name, pan in presets:
            ttk.Button(preset_row, text=name, width=8,
                      command=lambda p=pan: self._set_pan(p)).pack(side='left', padx=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # ITD/ILD DISPLAY
        # ═══════════════════════════════════════════════════════════════════
        itd_frame = ttk.LabelFrame(left_frame, text="📊 ITD/ILD Values", padding=5)
        itd_frame.pack(fill='x', pady=5)
        
        self.itd_label = ttk.Label(itd_frame, text="", font=('Courier', 9))
        self.itd_label.pack(anchor='w')
        self._update_itd_display()
        
        # ═══════════════════════════════════════════════════════════════════
        # SWEEP MODE (Auto-panning)
        # ═══════════════════════════════════════════════════════════════════
        sweep_frame = ttk.LabelFrame(left_frame, text="🌊 Auto-Sweep (Body Wave)", padding=5)
        sweep_frame.pack(fill='x', pady=5)
        
        sweep_toggle = ttk.Frame(sweep_frame)
        sweep_toggle.pack(fill='x', pady=2)
        ttk.Checkbutton(sweep_toggle, text="Enable sweep (sound travels head↔feet)",
                       variable=self.sweep_enabled,
                       command=self._on_sweep_toggle).pack(side='left')
        
        sweep_params = ttk.Frame(sweep_frame)
        sweep_params.pack(fill='x', pady=2)
        
        ttk.Label(sweep_params, text="Duration:").pack(side='left')
        for secs in [5, 10, 30, 60]:
            ttk.Button(sweep_params, text=f"{secs}s", width=4,
                      command=lambda s=secs: self.sweep_duration.set(s)).pack(side='left', padx=2)
        
        sweep_mode_row = ttk.Frame(sweep_frame)
        sweep_mode_row.pack(fill='x', pady=2)
        ttk.Label(sweep_mode_row, text="Mode:").pack(side='left')
        for mode in ["sine", "linear", "golden"]:
            ttk.Radiobutton(sweep_mode_row, text=mode, variable=self.sweep_mode,
                           value=mode).pack(side='left', padx=5)
        
        # Sweep progress
        self.sweep_progress = ttk.Progressbar(sweep_frame, mode='determinate', length=250)
        self.sweep_progress.pack(fill='x', pady=3)
        
        # ═══════════════════════════════════════════════════════════════════
        # CHAKRA SUNRISE - 3-frequency convergence journey
        # ═══════════════════════════════════════════════════════════════════
        chakra_frame = ttk.LabelFrame(left_frame, text="🌅 Chakra Sunrise Journey", padding=5)
        chakra_frame.pack(fill='x', pady=5)
        
        # Description
        desc_text = """3 frequencies converge at solar plexus:
• Perfect 4th (×4/3) - emerges at belly
• Root (×1) - rises from feet  
• Octave (×2) - descends from head"""
        ttk.Label(chakra_frame, text=desc_text, font=('Helvetica', 9),
                 foreground='#888', justify='left').pack(anchor='w', pady=2)
        
        # Duration selector
        dur_row = ttk.Frame(chakra_frame)
        dur_row.pack(fill='x', pady=3)
        ttk.Label(dur_row, text="Duration:").pack(side='left')
        
        self.chakra_duration = tk.DoubleVar(value=60.0)
        for label, secs in [("30s", 30), ("1m", 60), ("2m", 120), ("5m", 300)]:
            ttk.Button(dur_row, text=label, width=4,
                      command=lambda s=secs: self.chakra_duration.set(s)).pack(side='left', padx=2)
        
        # Base frequency for chakra
        base_row = ttk.Frame(chakra_frame)
        base_row.pack(fill='x', pady=3)
        ttk.Label(base_row, text="Base freq:").pack(side='left')
        self.chakra_base_freq = tk.DoubleVar(value=128.0)  # C3 area
        ttk.Entry(base_row, textvariable=self.chakra_base_freq, width=6).pack(side='left', padx=3)
        ttk.Label(base_row, text="Hz").pack(side='left')
        
        # Frequency display
        self.chakra_freq_label = ttk.Label(chakra_frame, text="", font=('Courier', 9))
        self.chakra_freq_label.pack(anchor='w', pady=2)
        self._update_chakra_freq_display()
        
        # Bind base freq change
        self.chakra_base_freq.trace_add('write', lambda *args: self._update_chakra_freq_display())
        
        # Play/Stop buttons
        chakra_btn_row = ttk.Frame(chakra_frame)
        chakra_btn_row.pack(fill='x', pady=5)
        
        self.chakra_play_btn = ttk.Button(chakra_btn_row, text="🌅 Begin Journey", 
                                          command=self._start_chakra_journey)
        self.chakra_play_btn.pack(side='left', padx=5)
        
        self.chakra_stop_btn = ttk.Button(chakra_btn_row, text="⏹ Stop", 
                                          command=self._stop_chakra_journey, state='disabled')
        self.chakra_stop_btn.pack(side='left', padx=5)
        
        # Progress bar and state
        self.chakra_progress = ttk.Progressbar(chakra_frame, mode='determinate', length=250)
        self.chakra_progress.pack(fill='x', pady=3)
        
        self.chakra_state_label = ttk.Label(chakra_frame, text="Ready", 
                                            font=('Helvetica', 10), foreground='#666')
        self.chakra_state_label.pack(anchor='w')
        
        # ═══════════════════════════════════════════════════════════════════
        # FREQUENCY & AMPLITUDE
        # ═══════════════════════════════════════════════════════════════════
        freq_frame = ttk.LabelFrame(left_frame, text="🎵 Sound Parameters", padding=5)
        freq_frame.pack(fill='x', pady=5)
        
        freq_row = ttk.Frame(freq_frame)
        freq_row.pack(fill='x', pady=2)
        ttk.Label(freq_row, text="Frequency (Hz):").pack(side='left')
        ttk.Entry(freq_row, textvariable=self.frequency, width=8).pack(side='left', padx=5)
        ttk.Scale(freq_row, from_=20, to=500, variable=self.frequency,
                  orient='horizontal', length=150,
                  command=self._on_freq_change).pack(side='left')
        
        # Frequency presets (therapeutic)
        freq_presets = ttk.Frame(freq_frame)
        freq_presets.pack(fill='x', pady=2)
        for name, freq in [("40Hz γ", 40), ("100Hz", 100), ("174Hz", 174), 
                           ("432Hz", 432), ("528Hz", 528)]:
            ttk.Button(freq_presets, text=name, width=6,
                      command=lambda f=freq: self._set_frequency(f)).pack(side='left', padx=2)
        
        amp_row = ttk.Frame(freq_frame)
        amp_row.pack(fill='x', pady=2)
        ttk.Label(amp_row, text="Amplitude:").pack(side='left')
        ttk.Scale(amp_row, from_=0, to=1, variable=self.amplitude,
                  orient='horizontal', length=150,
                  command=self._on_amp_change).pack(side='left', padx=5)
        
        wave_row = ttk.Frame(freq_frame)
        wave_row.pack(fill='x', pady=2)
        ttk.Label(wave_row, text="Waveform:").pack(side='left')
        for wf in ["sine", "golden"]:
            ttk.Radiobutton(wave_row, text=wf, variable=self.waveform,
                           value=wf, command=self._on_waveform_change).pack(side='left', padx=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # PLAYBACK BUTTONS
        # ═══════════════════════════════════════════════════════════════════
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ PLAY", command=self._play)
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # RIGHT PANEL - Visualization
        # ═══════════════════════════════════════════════════════════════════
        right_frame = ttk.LabelFrame(self.frame, text="🛏️ Soundboard Visualization", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=300, height=450, bg='#0a0a15')
        self.canvas.pack(pady=10)
        
        self.status_var = tk.StringVar(value="Configure and press PLAY")
        ttk.Label(right_frame, textvariable=self.status_var).pack()
        
        # Initial draw
        self._draw_soundboard()
    
    def _set_pan(self, pan: float):
        """Set pan position"""
        self.pan_position.set(pan)
        self._on_pan_change(None)
    
    def _set_frequency(self, freq: float):
        """Set frequency"""
        self.frequency.set(freq)
        self._on_freq_change(None)
    
    def _on_pan_change(self, event):
        """Handle pan change"""
        pan = self.pan_position.get()
        
        # Update label
        if pan < -0.8:
            pos_name = "HEAD"
        elif pan > 0.8:
            pos_name = "FEET"
        elif abs(pan) < 0.1:
            pos_name = "CENTER"
        else:
            pos_name = "↑" if pan < 0 else "↓"
        
        self.pan_label.config(text=f"Pan: {pan:+.2f} ({pos_name})")
        
        # Update ITD/ILD display
        self._update_itd_display()
        
        # Update visualization
        self._draw_soundboard()
        
        # Update panner and audio if playing
        if self.panner:
            self.panner.pan = pan
        
        if self.audio.is_playing():
            self._update_audio()
    
    def _on_freq_change(self, event):
        """Handle frequency change"""
        if self.audio.is_playing():
            self._update_audio()
    
    def _on_amp_change(self, event):
        """Handle amplitude change"""
        if self.audio.is_playing():
            self.audio.set_amplitude(self.amplitude.get())
    
    def _on_waveform_change(self):
        """Handle waveform change"""
        if self.audio.is_playing():
            self.audio.set_waveform(self.waveform.get())
    
    def _on_sweep_toggle(self):
        """Handle sweep toggle"""
        if self.sweep_enabled.get() and self.audio.is_playing():
            self._start_sweep()
        else:
            self._stop_sweep()
    
    def _update_itd_display(self):
        """Update ITD/ILD value display"""
        if not self.config:
            return
        
        pan = self.pan_position.get()
        params = calculate_panning(pan, self.config)
        
        text = f"""HEAD exciter:  Delay {params['delay_head_ms']:.3f} ms | Gain {params['gain_head_db']:+.1f} dB
FEET exciter:  Delay {params['delay_feet_ms']:.3f} ms | Gain {params['gain_feet_db']:+.1f} dB"""
        
        self.itd_label.config(text=text)
    
    def _update_audio(self):
        """Update audio with current panning parameters"""
        if not self.panner:
            return
        
        pan = self.pan_position.get()
        freq = self.frequency.get()
        amp = self.amplitude.get()
        
        params = calculate_panning(pan, self.config)
        
        # For soundboard mode, we send to HEAD and FEET exciters
        # These go to LEFT and RIGHT channels respectively
        # HEAD = LEFT channel, FEET = RIGHT channel
        
        # Create stereo output with ITD delays applied
        # Since AudioEngine doesn't support per-channel delay natively,
        # we'll use spectral mode with 2 frequencies (same freq, different timing)
        # and position them hard left/right
        
        # For now, use simple amplitude panning (ILD only)
        # Full ITD would require modifying AudioEngine's callback
        
        frequencies = [freq, freq]
        amplitudes = [params['gain_head'] * amp, params['gain_feet'] * amp]
        phases = [0.0, 0.0]  # Same phase for both (ITD would offset this)
        positions = [-1.0, 1.0]  # HEAD=left, FEET=right
        
        self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
    
    def _draw_soundboard(self):
        """Draw top-down view of soundboard with body silhouette"""
        self.canvas.delete('all')
        
        # Canvas dimensions
        cw, ch = 300, 450
        
        # Board dimensions in pixels (scaled)
        board_w = 100  # 600mm → 100px
        board_h = 350  # 2000mm → 350px
        board_x = (cw - board_w) // 2
        board_y = (ch - board_h) // 2
        
        # ═══════════════════════════════════════════════════════════════════
        # BOARD OUTLINE
        # ═══════════════════════════════════════════════════════════════════
        self.canvas.create_rectangle(board_x, board_y, 
                                     board_x + board_w, board_y + board_h,
                                     outline='#8b4513', width=3, fill='#2d1810')
        
        # Wood grain lines
        for i in range(5, board_h, 15):
            y = board_y + i
            self.canvas.create_line(board_x + 5, y, board_x + board_w - 5, y,
                                   fill='#3d2817', width=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # SPRINGS (4 corners + 1 center)
        # ═══════════════════════════════════════════════════════════════════
        spring_color = '#666'
        spring_size = 8
        spring_positions = [
            (board_x - 15, board_y + 20),           # Top-left
            (board_x + board_w + 5, board_y + 20),  # Top-right
            (board_x - 15, board_y + board_h - 20), # Bottom-left
            (board_x + board_w + 5, board_y + board_h - 20), # Bottom-right
            (board_x + board_w // 2 - spring_size, board_y + board_h // 2),  # Center
        ]
        
        for sx, sy in spring_positions:
            # Draw spring coil
            self.canvas.create_oval(sx, sy, sx + spring_size * 2, sy + spring_size * 2,
                                   outline=spring_color, width=2)
            self.canvas.create_line(sx + spring_size, sy, 
                                   sx + spring_size, sy + spring_size * 2,
                                   fill=spring_color, width=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # EXCITERS
        # ═══════════════════════════════════════════════════════════════════
        exciter_w, exciter_h = 40, 15
        
        # HEAD exciter (top)
        head_x = board_x + (board_w - exciter_w) // 2
        head_y = board_y - exciter_h + 3
        self.canvas.create_rectangle(head_x, head_y, head_x + exciter_w, head_y + exciter_h,
                                     fill='#ff6b6b', outline='#ff4444', width=2)
        self.canvas.create_text(head_x + exciter_w // 2, head_y - 10,
                               text="🧠 HEAD", fill='#ff6b6b', font=('Helvetica', 9, 'bold'))
        
        # FEET exciter (bottom)
        feet_x = board_x + (board_w - exciter_w) // 2
        feet_y = board_y + board_h - 3
        self.canvas.create_rectangle(feet_x, feet_y, feet_x + exciter_w, feet_y + exciter_h,
                                     fill='#4ecdc4', outline='#3dbdb4', width=2)
        self.canvas.create_text(feet_x + exciter_w // 2, feet_y + exciter_h + 12,
                               text="🦶 FEET", fill='#4ecdc4', font=('Helvetica', 9, 'bold'))
        
        # ═══════════════════════════════════════════════════════════════════
        # BODY SILHOUETTE
        # ═══════════════════════════════════════════════════════════════════
        body_color = '#444'
        
        # Simple body outline (head, torso, legs)
        cx = board_x + board_w // 2
        
        # Head circle
        head_radius = 20
        head_cy = board_y + 50
        self.canvas.create_oval(cx - head_radius, head_cy - head_radius,
                               cx + head_radius, head_cy + head_radius,
                               outline=body_color, width=2)
        
        # Ears (for ITD reference)
        ear_y = head_cy
        self.canvas.create_oval(cx - head_radius - 8, ear_y - 5,
                               cx - head_radius - 2, ear_y + 5,
                               fill='#555', outline=body_color)
        self.canvas.create_oval(cx + head_radius + 2, ear_y - 5,
                               cx + head_radius + 8, ear_y + 5,
                               fill='#555', outline=body_color)
        
        # Torso (rectangle)
        torso_w, torso_h = 50, 120
        torso_y = head_cy + head_radius + 10
        self.canvas.create_rectangle(cx - torso_w // 2, torso_y,
                                     cx + torso_w // 2, torso_y + torso_h,
                                     outline=body_color, width=2)
        
        # Legs
        leg_w = 18
        leg_h = 100
        leg_y = torso_y + torso_h + 5
        
        # Left leg
        self.canvas.create_rectangle(cx - torso_w // 2 + 3, leg_y,
                                     cx - torso_w // 2 + 3 + leg_w, leg_y + leg_h,
                                     outline=body_color, width=2)
        # Right leg
        self.canvas.create_rectangle(cx + torso_w // 2 - 3 - leg_w, leg_y,
                                     cx + torso_w // 2 - 3, leg_y + leg_h,
                                     outline=body_color, width=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # SOUND SOURCE POSITION (virtual)
        # ═══════════════════════════════════════════════════════════════════
        pan = self.pan_position.get()
        
        # Map pan -1..+1 to y position on board
        source_y = board_y + board_h // 2 + int(pan * (board_h // 2 - 30))
        source_x = cx
        
        # Sound waves emanating from source
        wave_color = '#ffd700'
        for r in [15, 25, 35]:
            alpha = 1.0 - r / 50
            self.canvas.create_oval(source_x - r, source_y - r // 2,
                                   source_x + r, source_y + r // 2,
                                   outline=wave_color, width=1)
        
        # Source point
        self.canvas.create_oval(source_x - 8, source_y - 8,
                               source_x + 8, source_y + 8,
                               fill='#ffd700', outline='white', width=2)
        
        # Arrow showing direction
        if abs(pan) > 0.1:
            arrow_dir = 1 if pan > 0 else -1
            arrow_len = 20
            self.canvas.create_line(source_x, source_y,
                                   source_x, source_y + arrow_dir * arrow_len,
                                   fill='#ffd700', width=3, arrow='last')
        
        # ═══════════════════════════════════════════════════════════════════
        # SIGNAL FLOW INDICATORS
        # ═══════════════════════════════════════════════════════════════════
        if self.panner:
            params = calculate_panning(pan, self.config)
            
            # Head exciter intensity
            head_intensity = int(255 * params['gain_head'])
            head_color = f'#ff{head_intensity:02x}{head_intensity:02x}'
            self.canvas.create_rectangle(head_x + 2, head_y + 2,
                                        head_x + exciter_w - 2, head_y + exciter_h - 2,
                                        fill=head_color, outline='')
            
            # Feet exciter intensity
            feet_intensity = int(255 * params['gain_feet'])
            feet_color = f'#{feet_intensity:02x}ff{feet_intensity:02x}'
            self.canvas.create_rectangle(feet_x + 2, feet_y + 2,
                                        feet_x + exciter_w - 2, feet_y + exciter_h - 2,
                                        fill=feet_color, outline='')
        
        # ═══════════════════════════════════════════════════════════════════
        # LABELS
        # ═══════════════════════════════════════════════════════════════════
        # Position label
        if pan < -0.8:
            pos_text = "HEAD"
        elif pan > 0.8:
            pos_text = "FEET"
        elif abs(pan) < 0.1:
            pos_text = "CENTER"
        else:
            pos_text = f"{abs(pan)*100:.0f}% {'↑HEAD' if pan < 0 else '↓FEET'}"
        
        self.canvas.create_text(cw // 2, 15, text=f"Source: {pos_text}",
                               fill='#ffd700', font=('Helvetica', 11, 'bold'))
        
        # Scale indicator
        self.canvas.create_text(cw // 2, ch - 10, 
                               text="2000mm (spruce board on springs)",
                               fill='#666', font=('Courier', 8))
    
    def _play(self):
        """Start playback"""
        freq = self.frequency.get()
        amp = self.amplitude.get()
        pan = self.pan_position.get()
        
        # Initialize panner
        if self.panner:
            self.panner.set_pan_immediate(pan)
        
        params = calculate_panning(pan, self.config)
        
        # Start with spectral mode (2 channels for HEAD/FEET)
        frequencies = [freq, freq]
        amplitudes = [params['gain_head'] * amp, params['gain_feet'] * amp]
        phases = [0.0, 0.0]
        positions = [-1.0, 1.0]  # HEAD=left, FEET=right
        
        self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                  master_amplitude=1.0)
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set(f"🔊 Playing {freq:.0f}Hz | Pan: {pan:+.2f}")
        
        # Start sweep if enabled
        if self.sweep_enabled.get():
            self._start_sweep()
    
    def _stop(self):
        """Stop playback"""
        self._stop_sweep()
        self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.sweep_progress['value'] = 0
        self.status_var.set("Configure and press PLAY")
    
    def _start_sweep(self):
        """Start auto-sweep"""
        self._is_sweeping = True
        self._sweep_start_time = time.time()
        self._sweep_callback()
    
    def _stop_sweep(self):
        """Stop auto-sweep"""
        self._is_sweeping = False
        if self._sweep_timer:
            self.frame.after_cancel(self._sweep_timer)
            self._sweep_timer = None
    
    def _sweep_callback(self):
        """Sweep animation callback"""
        if not self._is_sweeping or not self.audio.is_playing():
            return
        
        elapsed = time.time() - self._sweep_start_time
        duration = self.sweep_duration.get()
        
        # Calculate progress (0 to 1)
        progress = (elapsed % duration) / duration
        
        # Calculate pan based on mode
        mode = self.sweep_mode.get()
        if mode == "sine":
            # Smooth sine wave: head → center → feet → center → head
            pan = np.sin(progress * 2 * np.pi)
        elif mode == "linear":
            # Triangle wave
            if progress < 0.5:
                pan = -1.0 + 4 * progress  # -1 → 1
            else:
                pan = 3.0 - 4 * progress   # 1 → -1
        else:  # golden
            # Golden spiral easing
            t = progress * 2
            if t <= 1:
                pan = -1.0 + 2 * (0.5 * (1 - np.cos(t * np.pi * PHI_CONJUGATE)))
            else:
                t = t - 1
                pan = 1.0 - 2 * (0.5 * (1 - np.cos(t * np.pi * PHI_CONJUGATE)))
        
        # Update pan
        self.pan_position.set(pan)
        self._on_pan_change(None)
        
        # Update progress bar
        self.sweep_progress['value'] = progress * 100
        
        # Schedule next frame (30fps)
        self._sweep_timer = self.frame.after(33, self._sweep_callback)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHAKRA SUNRISE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _update_chakra_freq_display(self):
        """Update the frequency display for Chakra Sunrise"""
        try:
            base = self.chakra_base_freq.get()
            fourth = base * 4 / 3
            octave = base * 2
            self.chakra_freq_label.config(
                text=f"4th: {fourth:.1f}Hz | Root: {base:.1f}Hz | Oct: {octave:.1f}Hz"
            )
        except:
            pass
    
    def _mm_to_pan(self, position_mm):
        """
        Convert body position in mm to pan value (-1 to +1).
        
        Physical setup:
        - Wooden board 1950mm with longitudinal wood fibers
        - Two speaker exciters: one at HEAD (0mm), one at FEET (1950mm)
        - Sound propagates along wood fibers
        - Pan -1.0 = 100% HEAD exciter, Pan +1.0 = 100% FEET exciter
        
        Body positions on board:
        - HEAD:          0mm   → pan = -1.000
        - SOLAR_PLEXUS:  600mm → pan = -0.385
        - SACRAL:        800mm → pan = -0.179
        - FEET:          1750mm → pan = +0.795
        
        The full range ±1.0 is intentional - it matches the physical
        speaker placement and allows true localization along the body.
        """
        return (position_mm / 975.0) - 1.0
    
    def _golden_fade(self, t, fade_in=True):
        """
        Golden ratio based fade curve using φ exponent.
        Creates a smooth S-curve that follows golden proportions:
        - Slow start (breathing in)
        - Natural acceleration through middle  
        - Slow end (settling)
        
        The φ exponent creates a curve that feels organic.
        """
        t = max(0.0, min(1.0, t))  # Clamp to 0-1
        
        if fade_in:
            # Raised cosine with golden exponent for smooth start/end
            # cos goes 1→-1, so (1-cos)/2 goes 0→1
            base = (1 - np.cos(t * np.pi)) / 2.0
            # Apply golden exponent for organic curve
            return base ** PHI_CONJUGATE  # ≈ 0.618 exponent = faster rise
        else:
            # Fade out: mirror of fade in
            base = (1 - np.cos((1 - t) * np.pi)) / 2.0
            return 1.0 - (base ** PHI_CONJUGATE)
    
    def _start_chakra_journey(self):
        """Start the Chakra Sunrise journey"""
        if self._is_chakra_playing:
            return
        
        # Get parameters
        base_freq = self.chakra_base_freq.get()
        duration = self.chakra_duration.get()
        
        # Calculate the three frequencies
        self._freq_fourth = base_freq * 4 / 3   # Perfect 4th
        self._freq_root = base_freq             # Root/fundamental  
        self._freq_octave = base_freq * 2       # Octave
        
        self._journey_duration = duration
        
        # Body positions (mm from head edge on 1950mm board)
        HEAD_MM = 0.0
        SOLAR_PLEXUS_MM = 600.0   # Convergence point
        FEET_MM = 1750.0
        
        # Convert to pan values
        PAN_SOLAR_PLEXUS = self._mm_to_pan(SOLAR_PLEXUS_MM)  # ≈ -0.38
        PAN_HEAD = self._mm_to_pan(HEAD_MM)                   # -1.0
        PAN_FEET = self._mm_to_pan(FEET_MM)                   # ≈ +0.79
        
        # Initialize amplitudes and positions
        self._fourth_amp = 0.0
        self._fourth_pan = PAN_SOLAR_PLEXUS  # Always at solar plexus
        
        self._root_amp = 0.0
        self._root_pan = PAN_FEET            # Starts at feet
        
        self._octave_amp = 0.0
        self._octave_pan = PAN_HEAD          # Starts at head
        
        # Update UI
        self.chakra_play_btn.config(state='disabled')
        self.chakra_stop_btn.config(state='normal')
        
        # Start audio engine
        self._is_chakra_playing = True
        self._chakra_start_time = time.time()
        
        # Start with 3 frequencies, all silent initially
        self.audio.start_spectral(
            [self._freq_fourth, self._freq_root, self._freq_octave],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [PAN_SOLAR_PLEXUS, PAN_FEET, PAN_HEAD],
            master_amplitude=0.8
        )
        
        # Start the journey callback
        self._chakra_journey_callback()
        self.chakra_state_label.config(text=f"🌅 Journey started: {duration:.0f}s")
    
    def _stop_chakra_journey(self):
        """Stop the Chakra journey"""
        self._is_chakra_playing = False
        if self._chakra_timer:
            self.frame.after_cancel(self._chakra_timer)
            self._chakra_timer = None
        
        self.audio.stop()
        self.chakra_play_btn.config(state='normal')
        self.chakra_stop_btn.config(state='disabled')
        self.chakra_progress['value'] = 0
        self.chakra_state_label.config(text="Ready")
    
    def _chakra_journey_callback(self):
        """
        Animation callback for Chakra Sunrise journey.
        
        Body positions on 1950mm board:
        - HEAD = 0mm → pan = -1.0
        - SOLAR_PLEXUS = 600mm → pan ≈ -0.38
        - SACRAL = 800mm → pan ≈ -0.18
        - FEET = 1750mm → pan ≈ +0.79
        
        Timeline based on GOLDEN RATIO (φ ≈ 1.618):
        Using φ-based divisions: each phase relates to next by φ
        
        Phase boundaries (golden spiral):
        - Phase 1: 0.000 - 0.146  (14.6%)  4th fades in at SOLAR PLEXUS
        - Phase 2: 0.146 - 0.382  (23.6%)  Root fades in at FEET, rises
        - Phase 3: 0.382 - 0.618  (23.6%)  Root continues to SOLAR PLEXUS  
        - Phase 4: 0.618 - 0.854  (23.6%)  Octave fades in at HEAD, descends
        - Phase 5: 0.854 - 1.000  (14.6%)  CONVERGENCE at SOLAR PLEXUS
        
        The 14.6% and 23.6% come from φ: 1/φ³ ≈ 0.236, 1/φ⁴ ≈ 0.146
        """
        if not self._is_chakra_playing:
            return
        
        elapsed = time.time() - self._chakra_start_time
        duration = self._journey_duration
        progress = min(elapsed / duration, 1.0)
        
        # Update progress bar
        self.chakra_progress['value'] = progress * 100
        
        # Time display
        remaining = duration - elapsed
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        time_str = f"{mins}:{secs:02d}"
        
        # Body positions
        HEAD_MM = 0.0
        SOLAR_PLEXUS_MM = 600.0
        SACRAL_MM = 800.0
        FEET_MM = 1750.0
        
        PAN_HEAD = self._mm_to_pan(HEAD_MM)
        PAN_SOLAR_PLEXUS = self._mm_to_pan(SOLAR_PLEXUS_MM)
        PAN_SACRAL = self._mm_to_pan(SACRAL_MM)
        PAN_FEET = self._mm_to_pan(FEET_MM)
        
        # Golden phase boundaries (symmetric golden proportions: 1:φ:φ:φ:1)
        # Total = 2 + 3φ ≈ 6.854, each unit = 1/6.854
        # This creates: short intro, three golden middle phases, short outro
        GOLDEN_UNIT = 1.0 / (2.0 + 3.0 * PHI)  # ≈ 0.146
        GOLDEN_PHI_UNIT = PHI * GOLDEN_UNIT     # ≈ 0.236
        
        P1_END = GOLDEN_UNIT                                    # ≈ 0.146 (14.6%)
        P2_END = GOLDEN_UNIT + GOLDEN_PHI_UNIT                  # ≈ 0.382 (38.2%)
        P3_END = GOLDEN_UNIT + 2 * GOLDEN_PHI_UNIT              # ≈ 0.618 (61.8%)
        P4_END = GOLDEN_UNIT + 3 * GOLDEN_PHI_UNIT              # ≈ 0.854 (85.4%)
        # P5: 0.854 - 1.0 (14.6% - convergence)
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: 0 - 23.6% - Perfect 4th FADES IN at SOLAR PLEXUS
        # ═══════════════════════════════════════════════════════════════════
        if progress < P1_END:
            phase_progress = progress / P1_END
            
            self._fourth_amp = self._golden_fade(phase_progress, fade_in=True)
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = 0.0
            self._root_pan = PAN_FEET
            self._octave_amp = 0.0
            self._octave_pan = PAN_HEAD
            
            self.chakra_state_label.config(
                text=f"⚡ Phase 1: 4th emerging at solar plexus {self._fourth_amp*100:.0f}% ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: 23.6% - 38.2% - Root FADES IN at FEET, rises toward SACRAL
        # ═══════════════════════════════════════════════════════════════════
        elif progress < P2_END:
            phase_progress = (progress - P1_END) / (P2_END - P1_END)
            
            self._fourth_amp = 1.0
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = self._golden_fade(phase_progress, fade_in=True)
            pan_progress = self._golden_fade(phase_progress, fade_in=True)
            self._root_pan = PAN_FEET + (PAN_SACRAL - PAN_FEET) * pan_progress
            
            self._octave_amp = 0.0
            self._octave_pan = PAN_HEAD
            
            self.chakra_state_label.config(
                text=f"🦶 Phase 2: Root rising from feet {self._root_amp*100:.0f}% ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: 38.2% - 61.8% - Root continues from SACRAL to SOLAR PLEXUS
        # ═══════════════════════════════════════════════════════════════════
        elif progress < P3_END:
            phase_progress = (progress - P2_END) / (P3_END - P2_END)
            
            self._fourth_amp = 1.0
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = 1.0
            pan_progress = self._golden_fade(phase_progress, fade_in=True)
            self._root_pan = PAN_SACRAL + (PAN_SOLAR_PLEXUS - PAN_SACRAL) * pan_progress
            
            self._octave_amp = 0.0
            self._octave_pan = PAN_HEAD
            
            self.chakra_state_label.config(
                text=f"🌊 Phase 3: Root through sacral ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: 61.8% - 85.4% - Octave FADES IN at HEAD, descends to SOLAR PLEXUS
        # ═══════════════════════════════════════════════════════════════════
        elif progress < P4_END:
            phase_progress = (progress - P3_END) / (P4_END - P3_END)
            
            self._fourth_amp = 1.0
            self._fourth_pan = PAN_SOLAR_PLEXUS
            
            self._root_amp = 1.0
            self._root_pan = PAN_SOLAR_PLEXUS
            
            self._octave_amp = self._golden_fade(phase_progress, fade_in=True)
            pan_progress = self._golden_fade(phase_progress, fade_in=True)
            self._octave_pan = PAN_HEAD + (PAN_SOLAR_PLEXUS - PAN_HEAD) * pan_progress
            
            self.chakra_state_label.config(
                text=f"🧠 Phase 4: Octave descending {self._octave_amp*100:.0f}% ({time_str})")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: 85.4% - 100% - CONVERGENCE at SOLAR PLEXUS
        # Final golden segment: sustained unity
        # ═══════════════════════════════════════════════════════════════════
        else:
            self._fourth_amp = 1.0
            self._root_amp = 1.0
            self._octave_amp = 1.0
            
            self._fourth_pan = PAN_SOLAR_PLEXUS
            self._root_pan = PAN_SOLAR_PLEXUS
            self._octave_pan = PAN_SOLAR_PLEXUS
            
            self.chakra_state_label.config(
                text=f"✨ Phase 5: CONVERGENCE at φ ({time_str})")
        
        # Update audio with current state
        self._update_chakra_audio()
        
        # Check completion
        if progress >= 1.0:
            self._is_chakra_playing = False
            self.chakra_state_label.config(text="🌅 Journey complete! All frequencies united")
            self.chakra_play_btn.config(state='normal')
            self.chakra_stop_btn.config(state='disabled')
            return
        
        # Schedule next frame (30ms for smooth golden transitions)
        self._chakra_timer = self.frame.after(30, self._chakra_journey_callback)
    
    def _update_chakra_audio(self):
        """Update audio parameters for Chakra journey"""
        frequencies = [self._freq_fourth, self._freq_root, self._freq_octave]
        amplitudes = [self._fourth_amp, self._root_amp, self._octave_amp]
        phases = [0.0, 0.0, 0.0]
        positions = [self._fourth_pan, self._root_pan, self._octave_pan]
        
        self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class GoldenSoundStudio:
    """Main application with tabbed interface"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌀 GOLDEN SOUND STUDIO")
        self.root.geometry("950x700")
        
        # Shared audio engine
        self.audio = AudioEngine()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the main UI"""
        # Header
        header = tk.Frame(self.root, bg='#1a1a2e')
        header.pack(fill='x')
        
        title = tk.Label(header, text="🌀 GOLDEN SOUND STUDIO 🌀",
                        font=('Helvetica', 20, 'bold'), fg='#ffd700', bg='#1a1a2e')
        title.pack(pady=10)
        
        subtitle = tk.Label(header, 
                           text="Binaural Beats • Atomic Spectra • Molecular Geometry",
                           font=('Helvetica', 11), fg='#888', bg='#1a1a2e')
        subtitle.pack()
        
        # Audio device selector
        device_frame = tk.Frame(header, bg='#1a1a2e')
        device_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(device_frame, text="Audio Device:", fg='#888', bg='#1a1a2e',
                font=('Courier', 9)).pack(side='left')
        
        # Re-scan devices to ensure fresh list
        self.audio._scan_devices()
        devices = self.audio.get_device_names()
        print(f"🔊 Found {len(devices)} audio devices: {devices}")
        
        self.device_var = tk.StringVar(value=devices[0] if devices else "Default")
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var,
                                   values=devices, state='readonly', width=50)
        self.device_combo.pack(side='left', padx=10)
        self.device_combo.bind('<<ComboboxSelected>>', self._on_device_change)
        
        # Refresh button
        ttk.Button(device_frame, text="🔄", width=3, 
                  command=self._refresh_devices).pack(side='left', padx=5)
        
        # Notebook (tabs)
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Helvetica', 11, 'bold'), padding=[20, 10])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.binaural_tab = BinauralTab(self.notebook, self.audio)
        self.spectral_tab = SpectralTab(self.notebook, self.audio)
        self.molecular_tab = MolecularTab(self.notebook, self.audio)
        self.harmonic_tree_tab = HarmonicTreeTab(self.notebook, self.audio)
        self.vibroacoustic_tab = VibroacousticTab(self.notebook, self.audio)
        
        # EMDR Tab (bilateral audio stimulation for hemispheric integration)
        if HAS_EMDR:
            self.emdr_tab = EMDRTab(self.notebook, self.audio)
        
        self.notebook.add(self.binaural_tab.frame, text="🎵 Binaural Beats")
        self.notebook.add(self.spectral_tab.frame, text="⚛️ Spectral Sound")
        self.notebook.add(self.molecular_tab.frame, text="🧪 Molecular Sound")
        self.notebook.add(self.harmonic_tree_tab.frame, text="🌳 Harmonic Tree")
        self.notebook.add(self.vibroacoustic_tab.frame, text="🪵 Vibroacoustic")
        
        # Add EMDR tab if available
        if HAS_EMDR:
            self.notebook.add(self.emdr_tab.frame, text="🧠 EMDR")
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#1a1a2e')
        status_frame.pack(fill='x')
        
        self.status = tk.Label(status_frame, text="Ready", font=('Courier', 9),
                              fg='#00ff88', bg='#1a1a2e')
        self.status.pack(pady=5)
        
        # Bindings
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_device_change(self, event):
        """Handle device selection change"""
        idx = self.audio.get_device_names().index(self.device_var.get())
        self.audio.set_device(idx)
        print(f"🔊 Selected device: {self.device_var.get()} (index {idx})")
    
    def _refresh_devices(self):
        """Refresh the device list"""
        self.audio._scan_devices()
        devices = self.audio.get_device_names()
        self.device_combo['values'] = devices
        if devices:
            self.device_var.set(devices[0])
        print(f"🔄 Refreshed: {len(devices)} devices found")
    
    def _on_close(self):
        """Handle window close"""
        self.audio.stop()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GOLDEN SOUND STUDIO                                       ║
║                                                                              ║
║   🎵 Tab 1: Binaural Beats - Phase angle control, sacred geometry           ║
║   ⚛️ Tab 2: Spectral Sound - Play atomic elements (H, He, O, Na...)         ║
║   🧪 Tab 3: Molecular Sound - Play molecules (H₂O, CO₂, CH₄...)             ║
║   🌳 Tab 4: Harmonic Tree - Fundamental + Fibonacci harmonics               ║
║   🪵 Tab 5: Vibroacoustic - Soundboard panning (HEAD↔FEET)                  ║
║   🧠 Tab 6: EMDR - Bilateral audio, hemispheric integration, annealing      ║
║                                                                              ║
║   Based on natural phyllotaxis patterns:                                     ║
║   • Harmonics at Fibonacci ratios (2f, 3f, 5f, 8f, 13f)                     ║
║   • Phases rotate by Golden Angle (137.5°) like sunflower seeds             ║
║   • Amplitudes decay by φ⁻ⁿ (natural growth pattern)                        ║
║   • EMDR bilateral stimulation for trauma processing                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        self.root.mainloop()


if __name__ == "__main__":
    app = GoldenSoundStudio()
    app.run()
