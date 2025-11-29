#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DUAL FREQUENCY PHASE ANGLE CONTROLLER                     ║
║                                                                              ║
║   Set LEFT and RIGHT frequencies independently                               ║
║   Control PHASE DIFFERENCE as ANGLE (degrees)                                ║
║   Visualize sacred angles: Golden 137.5°, Fine Structure 137.036°, etc.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import struct
import math

# ══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895  # Golden Ratio
PHI_CONJUGATE = 0.618033988749895  # 1/φ = φ-1

# Sacred Angles in DEGREES
SACRED_ANGLES = {
    "Golden Angle (360°/φ²)": 360.0 / (PHI * PHI),  # ≈ 137.5077°
    "Fine Structure (α⁻¹)": 137.035999084,
    "DNA Helix (per base)": 34.3,
    "DNA Turn (360°/10.5)": 360.0 / 10.5,  # ≈ 34.29°
    "Pentagon Internal": 108.0,
    "Pentagon External": 72.0,
    "Pyramid Giza Slope": 51.8392,
    "Cancellation (180°)": 180.0,
    "Quadrature (90°)": 90.0,
    "Unity (0°)": 0.0,
    "Golden × φ": (360.0 / (PHI * PHI)) * PHI,  # ≈ 222.5°
    "Fine Struct / φ": 137.035999084 / PHI,  # ≈ 84.7°
}

SAMPLE_RATE = 44100

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AudioDeviceManager:
    """Manage audio devices and get available outputs"""
    
    def __init__(self):
        self.pyaudio = None
        self.devices = []
        self._scan_devices()
    
    def _scan_devices(self):
        """Scan for available audio output devices"""
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            self.devices = []
            
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                # Only output devices (maxOutputChannels > 0)
                if info['maxOutputChannels'] > 0:
                    self.devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxOutputChannels'],
                        'sample_rate': int(info['defaultSampleRate']),
                        'host_api': self.pyaudio.get_host_api_info_by_index(info['hostApi'])['name']
                    })
            
            self.pyaudio.terminate()
            self.pyaudio = None
            
        except Exception as e:
            print(f"Error scanning devices: {e}")
            self.devices = []
    
    def get_device_list(self) -> list:
        """Get list of device names for UI"""
        return [f"{d['name']} ({d['host_api']})" for d in self.devices]
    
    def get_device_index(self, selection_idx: int) -> int:
        """Get PyAudio device index from selection index"""
        if 0 <= selection_idx < len(self.devices):
            return self.devices[selection_idx]['index']
        return None
    
    def get_device_info(self, selection_idx: int) -> dict:
        """Get device info from selection index"""
        if 0 <= selection_idx < len(self.devices):
            return self.devices[selection_idx]
        return None
    
    def refresh(self):
        """Rescan devices"""
        self._scan_devices()


class DualFrequencyEngine:
    """Generate stereo audio with independent L/R frequencies and phase angle control
    
    Uses PHI-based waveform (Golden Wave) instead of pure sine, running REVERSED
    """
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.freq_left = 432.0
        self.freq_right = 440.0
        self.phase_angle_deg = 137.5  # Golden angle default
        self.amplitude = 0.7
        
        # Phase accumulators for continuous playback
        self.phase_left = 0.0
        self.phase_right = 0.0
        
        # Waveform mode
        self.waveform_mode = "golden_reversed"  # "sine", "golden", "golden_reversed"
        
        # Audio device
        self.device_index = None  # None = default device
        
        self.playing = False
        self.lock = threading.Lock()
    
    def set_waveform_mode(self, mode: str):
        """Set waveform: 'sine', 'golden', 'golden_reversed'"""
        with self.lock:
            self.waveform_mode = mode
    
    def set_device(self, device_index: int):
        """Set output device index (None for default)"""
        with self.lock:
            self.device_index = device_index
    
    def set_frequencies(self, freq_left: float, freq_right: float):
        with self.lock:
            self.freq_left = freq_left
            self.freq_right = freq_right
    
    def set_phase_angle(self, angle_deg: float):
        """Set phase difference in DEGREES"""
        with self.lock:
            self.phase_angle_deg = angle_deg
    
    def set_amplitude(self, amp: float):
        with self.lock:
            self.amplitude = max(0.0, min(1.0, amp))
    
    def deg_to_rad(self, deg: float) -> float:
        return deg * np.pi / 180.0
    
    def golden_wave(self, phase: np.ndarray, reversed: bool = True) -> np.ndarray:
        """
        Generate PHI-based waveform (Golden Wave)
        
        Instead of sin(θ), uses a waveform shaped by golden ratio:
        - Base: golden spiral interpolation
        - Asymmetric: rise/fall in φ ratio
        - Harmonic: includes φ-weighted harmonics
        
        If reversed=True, the wave runs backwards (time-reversed)
        """
        # Normalize phase to [0, 2π]
        theta = phase % (2 * np.pi)
        
        if reversed:
            # Reverse the phase progression
            theta = 2 * np.pi - theta
        
        # Normalized position in cycle [0, 1]
        t = theta / (2 * np.pi)
        
        # Golden wave components:
        
        # 1. Asymmetric rise/fall using golden ratio
        # Rise takes φ⁻¹ of the cycle, fall takes (1-φ⁻¹)
        rise_portion = PHI_CONJUGATE  # ≈ 0.618
        
        wave = np.zeros_like(t)
        
        # Rising phase (0 to rise_portion)
        rising = t < rise_portion
        t_rise = t[rising] / rise_portion  # Normalize to [0,1]
        # Golden spiral easing for rise
        wave[rising] = self._golden_ease(t_rise)
        
        # Falling phase (rise_portion to 1)
        falling = ~rising
        t_fall = (t[falling] - rise_portion) / (1 - rise_portion)  # Normalize to [0,1]
        # Golden spiral easing for fall (inverted)
        wave[falling] = 1 - self._golden_ease(t_fall)
        
        # 2. Scale to [-1, 1]
        wave = wave * 2 - 1
        
        # 3. Add golden harmonics
        # φ-weighted second harmonic
        harmonic2 = np.sin(2 * theta) * (PHI_CONJUGATE ** 2)
        # φ-weighted third harmonic (reversed relationship)
        harmonic3 = np.sin(3 * theta) * (PHI_CONJUGATE ** 3)
        
        # Blend: main wave + harmonics in golden ratio
        golden_blend = (
            wave * PHI_CONJUGATE + 
            harmonic2 * (1 - PHI_CONJUGATE) * PHI_CONJUGATE +
            harmonic3 * (1 - PHI_CONJUGATE) * (1 - PHI_CONJUGATE)
        )
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(golden_blend))
        if max_val > 0:
            golden_blend = golden_blend / max_val
        
        return golden_blend
    
    def _golden_ease(self, t: np.ndarray) -> np.ndarray:
        """Golden spiral interpolation for smooth easing"""
        # Clamp
        t = np.clip(t, 0, 1)
        
        # Golden spiral easing
        theta = t * np.pi * PHI
        golden_ease = (1.0 - np.cos(theta * PHI_CONJUGATE)) / 2.0
        
        # Golden sigmoid blend
        x = (t - 0.5) * 4.0
        golden_sigmoid = 1.0 / (1.0 + np.exp(-x * PHI))
        
        # Blend with golden weights
        result = golden_ease * PHI_CONJUGATE + golden_sigmoid * (1 - PHI_CONJUGATE)
        
        return np.clip(result, 0, 1)
    
    def deg_to_rad(self, deg: float) -> float:
        return deg * np.pi / 180.0
    
    def generate_chunk(self, num_samples: int) -> bytes:
        """Generate stereo audio chunk with current parameters using Golden Wave"""
        with self.lock:
            freq_l = self.freq_left
            freq_r = self.freq_right
            phase_offset = self.deg_to_rad(self.phase_angle_deg)
            amp = self.amplitude
            mode = self.waveform_mode
        
        # Phase increments (NEGATIVE for reversed direction)
        if "reversed" in mode:
            delta_phase_l = -2 * np.pi * freq_l / self.sample_rate
            delta_phase_r = -2 * np.pi * freq_r / self.sample_rate
        else:
            delta_phase_l = 2 * np.pi * freq_l / self.sample_rate
            delta_phase_r = 2 * np.pi * freq_r / self.sample_rate
        
        # Generate with accumulated phase for continuity
        phases_l = self.phase_left + np.cumsum(np.full(num_samples, delta_phase_l))
        phases_r = self.phase_right + phase_offset + np.cumsum(np.full(num_samples, delta_phase_r))
        
        # Update accumulated phases (keep in valid range)
        self.phase_left = phases_l[-1] % (2 * np.pi)
        self.phase_right = (phases_r[-1] - phase_offset) % (2 * np.pi)
        
        # Generate signals based on waveform mode
        if mode == "sine":
            left = amp * np.sin(phases_l)
            right = amp * np.sin(phases_r)
        elif mode == "golden":
            left = amp * self.golden_wave(phases_l, reversed=False)
            right = amp * self.golden_wave(phases_r, reversed=False)
        else:  # golden_reversed (default)
            left = amp * self.golden_wave(phases_l, reversed=True)
            right = amp * self.golden_wave(phases_r, reversed=True)
        
        # Interleave stereo
        stereo = np.empty(num_samples * 2, dtype=np.float32)
        stereo[0::2] = left.astype(np.float32)
        stereo[1::2] = right.astype(np.float32)
        
        return stereo.tobytes()
    
    def get_instantaneous_values(self) -> dict:
        """Get current L, R, and sum values for visualization"""
        with self.lock:
            phase_offset = self.deg_to_rad(self.phase_angle_deg)
            mode = self.waveform_mode
        
        phase_l = np.array([self.phase_left])
        phase_r = np.array([self.phase_right + phase_offset])
        
        if mode == "sine":
            l = np.sin(self.phase_left)
            r = np.sin(self.phase_right + phase_offset)
        else:
            reversed_mode = "reversed" in mode
            l = self.golden_wave(phase_l, reversed=reversed_mode)[0]
            r = self.golden_wave(phase_r, reversed=reversed_mode)[0]
        
        return {
            'left': l,
            'right': r,
            'sum': l + r,
            'phase_deg': self.phase_angle_deg,
            'freq_left': self.freq_left,
            'freq_right': self.freq_right,
            'waveform': mode,
        }

# ══════════════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class PhaseAngleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Dual Frequency Phase Angle Controller")
        
        # Golden ratio dimensions
        width = 1100
        height = int(width / PHI) + 80  # Extra space for audio settings
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='#1a1a2e')
        
        # Audio device manager
        self.device_manager = AudioDeviceManager()
        self.selected_device_idx = 0  # Default device
        
        self.engine = DualFrequencyEngine(SAMPLE_RATE)
        self.pyaudio = None
        self.stream = None
        
        self._setup_ui()
        self._start_visualization()
    
    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), 
                       background='#1a1a2e', foreground='#ffd700')
        style.configure('Info.TLabel', font=('Courier', 11), 
                       background='#1a1a2e', foreground='#00ff88')
        style.configure('Value.TLabel', font=('Courier', 14, 'bold'), 
                       background='#1a1a2e', foreground='#00ffff')
        
        # Main container
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title = tk.Label(main, text="✦ DUAL FREQUENCY PHASE ANGLE CONTROLLER ✦",
                        font=('Helvetica', 18, 'bold'), fg='#ffd700', bg='#1a1a2e')
        title.pack(pady=(0, 20))
        
        # Content frame (two columns)
        content = tk.Frame(main, bg='#1a1a2e')
        content.pack(fill='both', expand=True)
        
        # Left column: Controls
        left_col = tk.Frame(content, bg='#1a1a2e')
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Right column: Visualization
        right_col = tk.Frame(content, bg='#1a1a2e')
        
        # Audio device section at top of left column
        self._create_audio_device_controls(left_col)
        right_col.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        self._create_frequency_controls(left_col)
        self._create_phase_controls(left_col)
        self._create_sacred_presets(left_col)
        self._create_playback_controls(left_col)
        self._create_visualization(right_col)
    
    def _create_audio_device_controls(self, parent):
        """Create audio device selection controls"""
        frame = tk.LabelFrame(parent, text="🔊 AUDIO OUTPUT DEVICE", 
                             font=('Helvetica', 12, 'bold'),
                             fg='#00ff88', bg='#1a1a2e', bd=2)
        frame.pack(fill='x', pady=10)
        
        # Device selection
        dev_frame = tk.Frame(frame, bg='#1a1a2e')
        dev_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(dev_frame, text="Device:", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        # Dropdown for devices
        self.device_var = tk.StringVar()
        device_list = self.device_manager.get_device_list()
        if device_list:
            self.device_var.set(device_list[0])
        else:
            device_list = ["No audio devices found"]
            self.device_var.set(device_list[0])
        
        self.device_combo = ttk.Combobox(dev_frame, textvariable=self.device_var,
                                         values=device_list, state='readonly',
                                         width=50, font=('Courier', 9))
        self.device_combo.pack(side='left', padx=10, fill='x', expand=True)
        self.device_combo.bind('<<ComboboxSelected>>', self._on_device_change)
        
        # Refresh button
        refresh_btn = tk.Button(dev_frame, text="↻", command=self._refresh_devices,
                               bg='#3d3d5c', fg='#00ff88', font=('Helvetica', 12),
                               width=3)
        refresh_btn.pack(side='right', padx=5)
        
        # Device info
        info_frame = tk.Frame(frame, bg='#1a1a2e')
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.device_info_label = tk.Label(info_frame, text="", 
                                          font=('Courier', 9), fg='#666', bg='#1a1a2e')
        self.device_info_label.pack(side='left')
        
        # Update device info
        self._update_device_info()
    
    def _on_device_change(self, event=None):
        """Handle device selection change"""
        device_list = self.device_manager.get_device_list()
        selected = self.device_var.get()
        
        try:
            idx = device_list.index(selected)
            self.selected_device_idx = idx
            device_info = self.device_manager.get_device_info(idx)
            if device_info:
                self.engine.set_device(device_info['index'])
            self._update_device_info()
            
            # If playing, restart with new device
            if self.engine.playing:
                self._stop_audio()
                self._start_audio()
                
        except ValueError:
            pass
    
    def _refresh_devices(self):
        """Refresh audio device list"""
        self.device_manager.refresh()
        device_list = self.device_manager.get_device_list()
        
        if device_list:
            self.device_combo['values'] = device_list
            self.device_var.set(device_list[0])
            self.selected_device_idx = 0
        else:
            self.device_combo['values'] = ["No audio devices found"]
            self.device_var.set("No audio devices found")
        
        self._update_device_info()
    
    def _update_device_info(self):
        """Update device info display"""
        device_info = self.device_manager.get_device_info(self.selected_device_idx)
        if device_info:
            info_text = f"Channels: {device_info['channels']} | " \
                       f"Sample Rate: {device_info['sample_rate']} Hz | " \
                       f"API: {device_info['host_api']}"
            self.device_info_label.config(text=info_text)
        else:
            self.device_info_label.config(text="No device selected")
    
    def _create_frequency_controls(self, parent):
        frame = tk.LabelFrame(parent, text="⚡ FREQUENCIES (Hz)", 
                             font=('Helvetica', 12, 'bold'),
                             fg='#ffd700', bg='#1a1a2e', bd=2)
        frame.pack(fill='x', pady=10)
        
        # Left frequency
        lf = tk.Frame(frame, bg='#1a1a2e')
        lf.pack(fill='x', padx=10, pady=5)
        
        tk.Label(lf, text="LEFT:", font=('Courier', 11), 
                fg='#ff6b6b', bg='#1a1a2e').pack(side='left')
        
        self.freq_left_var = tk.DoubleVar(value=432.0)
        self.freq_left_scale = tk.Scale(lf, from_=20, to=1000, resolution=0.1,
                                        orient='horizontal', variable=self.freq_left_var,
                                        command=self._on_freq_change,
                                        bg='#2d2d44', fg='#ff6b6b', 
                                        troughcolor='#1a1a2e', highlightthickness=0,
                                        length=300)
        self.freq_left_scale.pack(side='left', fill='x', expand=True, padx=10)
        
        self.freq_left_label = tk.Label(lf, text="432.0 Hz", font=('Courier', 11, 'bold'),
                                        fg='#ff6b6b', bg='#1a1a2e', width=10)
        self.freq_left_label.pack(side='right')
        
        # Right frequency
        rf = tk.Frame(frame, bg='#1a1a2e')
        rf.pack(fill='x', padx=10, pady=5)
        
        tk.Label(rf, text="RIGHT:", font=('Courier', 11), 
                fg='#4ecdc4', bg='#1a1a2e').pack(side='left')
        
        self.freq_right_var = tk.DoubleVar(value=440.0)
        self.freq_right_scale = tk.Scale(rf, from_=20, to=1000, resolution=0.1,
                                         orient='horizontal', variable=self.freq_right_var,
                                         command=self._on_freq_change,
                                         bg='#2d2d44', fg='#4ecdc4',
                                         troughcolor='#1a1a2e', highlightthickness=0,
                                         length=300)
        self.freq_right_scale.pack(side='left', fill='x', expand=True, padx=10)
        
        self.freq_right_label = tk.Label(rf, text="440.0 Hz", font=('Courier', 11, 'bold'),
                                         fg='#4ecdc4', bg='#1a1a2e', width=10)
        self.freq_right_label.pack(side='right')
        
        # Beat frequency display
        bf = tk.Frame(frame, bg='#1a1a2e')
        bf.pack(fill='x', padx=10, pady=5)
        
        tk.Label(bf, text="BEAT FREQ:", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        self.beat_label = tk.Label(bf, text="8.0 Hz", font=('Courier', 11, 'bold'),
                                   fg='#ffd700', bg='#1a1a2e')
        self.beat_label.pack(side='left', padx=10)
        
        # === DIRECT BEAT FREQUENCY CONTROL ===
        beat_ctrl_frame = tk.LabelFrame(parent, text="🎯 DIRECT BEAT FREQUENCY CONTROL", 
                                        font=('Helvetica', 12, 'bold'),
                                        fg='#00ffff', bg='#1a1a2e', bd=2)
        beat_ctrl_frame.pack(fill='x', pady=10)
        
        # Base frequency (carrier)
        base_f = tk.Frame(beat_ctrl_frame, bg='#1a1a2e')
        base_f.pack(fill='x', padx=10, pady=5)
        
        tk.Label(base_f, text="BASE (carrier):", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        self.base_freq_var = tk.DoubleVar(value=432.0)
        self.base_freq_entry = tk.Entry(base_f, textvariable=self.base_freq_var,
                                        font=('Courier', 12), width=10,
                                        bg='#2d2d44', fg='#ffd700', insertbackground='#ffd700')
        self.base_freq_entry.pack(side='left', padx=10)
        
        tk.Label(base_f, text="Hz", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        # Beat frequency entry
        beat_f = tk.Frame(beat_ctrl_frame, bg='#1a1a2e')
        beat_f.pack(fill='x', padx=10, pady=5)
        
        tk.Label(beat_f, text="BEAT FREQ:", font=('Courier', 10), 
                fg='#00ffff', bg='#1a1a2e').pack(side='left')
        
        self.direct_beat_var = tk.DoubleVar(value=8.0)
        self.direct_beat_entry = tk.Entry(beat_f, textvariable=self.direct_beat_var,
                                          font=('Courier', 14, 'bold'), width=10,
                                          bg='#2d2d44', fg='#00ffff', insertbackground='#00ffff')
        self.direct_beat_entry.pack(side='left', padx=10)
        
        tk.Label(beat_f, text="Hz", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        # Quick beat presets
        preset_f = tk.Frame(beat_ctrl_frame, bg='#1a1a2e')
        preset_f.pack(fill='x', padx=10, pady=5)
        
        beat_presets = [
            ("δ Delta\n0.5-4Hz", 2.0),
            ("θ Theta\n4-8Hz", 6.0),
            ("α Alpha\n8-13Hz", 10.0),
            ("β Beta\n13-30Hz", 20.0),
            ("γ Gamma\n30-100Hz", 40.0),
            ("φ Golden\n≈7.83Hz", 432.0 / (PHI ** 4)),  # Schumann × golden
        ]
        
        for name, freq in beat_presets:
            btn = tk.Button(preset_f, text=name, 
                           command=lambda f=freq: self._set_direct_beat(f),
                           bg='#2d2d44', fg='#00ffff', font=('Courier', 8),
                           width=8, height=2)
            btn.pack(side='left', padx=2)
        
        # Apply button
        apply_f = tk.Frame(beat_ctrl_frame, bg='#1a1a2e')
        apply_f.pack(fill='x', padx=10, pady=5)
        
        self.apply_beat_btn = tk.Button(apply_f, text="⚡ APPLY BEAT FREQUENCY", 
                                        command=self._apply_direct_beat,
                                        bg='#006464', fg='#00ffff',
                                        font=('Helvetica', 11, 'bold'),
                                        width=25)
        self.apply_beat_btn.pack(pady=5)
        
        # Result display
        self.beat_result_label = tk.Label(apply_f, 
                                          text="L: 432.0 Hz | R: 440.0 Hz | Beat: 8.0 Hz",
                                          font=('Courier', 10), fg='#00ff88', bg='#1a1a2e')
        self.beat_result_label.pack()
        
        # Bind Enter key
        self.direct_beat_entry.bind('<Return>', lambda e: self._apply_direct_beat())
        self.base_freq_entry.bind('<Return>', lambda e: self._apply_direct_beat())
        
        # === WAVEFORM SELECTION ===
        wave_frame = tk.LabelFrame(parent, text="〰️ WAVEFORM TYPE (PHI-based)", 
                                   font=('Helvetica', 12, 'bold'),
                                   fg='#ff00ff', bg='#1a1a2e', bd=2)
        wave_frame.pack(fill='x', pady=10)
        
        wave_inner = tk.Frame(wave_frame, bg='#1a1a2e')
        wave_inner.pack(fill='x', padx=10, pady=5)
        
        self.waveform_var = tk.StringVar(value="golden_reversed")
        
        waveforms = [
            ("🔄 Golden Reversed\n(PHI wave, backward)", "golden_reversed", "#ff00ff"),
            ("✨ Golden Forward\n(PHI wave, forward)", "golden", "#ffd700"),
            ("〰️ Pure Sine\n(Standard)", "sine", "#00ff88"),
        ]
        
        for text, mode, color in waveforms:
            btn = tk.Radiobutton(wave_inner, text=text, variable=self.waveform_var,
                                value=mode, command=self._on_waveform_change,
                                bg='#2d2d44', fg=color, selectcolor='#1a1a2e',
                                activebackground='#3d3d5c', activeforeground=color,
                                font=('Courier', 9), indicatoron=0, width=18, height=2,
                                borderwidth=2, relief='raised')
            btn.pack(side='left', padx=5, pady=5)
        
        # Waveform info
        self.wave_info_label = tk.Label(wave_frame, 
            text="φ Golden Wave: Rise=φ⁻¹, Fall=1-φ⁻¹ | Harmonics: 2nd×φ⁻², 3rd×φ⁻³ | Direction: REVERSED ←",
            font=('Courier', 8), fg='#888', bg='#1a1a2e')
        self.wave_info_label.pack(pady=5)
    
    def _set_direct_beat(self, beat_freq: float):
        """Set beat frequency from preset"""
        self.direct_beat_var.set(round(beat_freq, 4))
        self._apply_direct_beat()
    
    def _on_waveform_change(self):
        """Handle waveform type change"""
        mode = self.waveform_var.get()
        self.engine.set_waveform_mode(mode)
        
        # Update info label
        if mode == "golden_reversed":
            info = "φ Golden Wave: Rise=φ⁻¹, Fall=1-φ⁻¹ | Harmonics: 2nd×φ⁻², 3rd×φ⁻³ | Direction: REVERSED ←"
        elif mode == "golden":
            info = "φ Golden Wave: Rise=φ⁻¹, Fall=1-φ⁻¹ | Harmonics: 2nd×φ⁻², 3rd×φ⁻³ | Direction: FORWARD →"
        else:
            info = "Standard sine wave: sin(2πft) | No harmonics | Direction: FORWARD →"
        
        self.wave_info_label.config(text=info)
    
    def _apply_direct_beat(self):
        """Apply direct beat frequency - calculates L and R automatically"""
        try:
            base = self.base_freq_var.get()
            beat = self.direct_beat_var.get()
            
            # Calculate L and R frequencies
            # L = base, R = base + beat (or base - beat/2, base + beat/2 for centered)
            freq_left = base
            freq_right = base + beat
            
            # Update the sliders
            self.freq_left_var.set(freq_left)
            self.freq_right_var.set(freq_right)
            
            # Update engine
            self.engine.set_frequencies(freq_left, freq_right)
            
            # Update labels
            self.freq_left_label.config(text=f"{freq_left:.1f} Hz")
            self.freq_right_label.config(text=f"{freq_right:.1f} Hz")
            self.beat_label.config(text=f"{beat:.2f} Hz")
            
            # Update result display
            self.beat_result_label.config(
                text=f"L: {freq_left:.1f} Hz | R: {freq_right:.1f} Hz | Beat: {beat:.2f} Hz"
            )
            
        except Exception as e:
            print(f"Error applying beat: {e}")
    
    def _create_phase_controls(self, parent):
        frame = tk.LabelFrame(parent, text="◐ PHASE ANGLE (degrees)", 
                             font=('Helvetica', 12, 'bold'),
                             fg='#ffd700', bg='#1a1a2e', bd=2)
        frame.pack(fill='x', pady=10)
        
        # Main phase slider
        pf = tk.Frame(frame, bg='#1a1a2e')
        pf.pack(fill='x', padx=10, pady=5)
        
        self.phase_var = tk.DoubleVar(value=137.5)
        self.phase_scale = tk.Scale(pf, from_=0, to=360, resolution=0.001,
                                    orient='horizontal', variable=self.phase_var,
                                    command=self._on_phase_change,
                                    bg='#2d2d44', fg='#ffd700',
                                    troughcolor='#1a1a2e', highlightthickness=0,
                                    length=380)
        self.phase_scale.pack(fill='x')
        
        # Phase value display
        vf = tk.Frame(frame, bg='#1a1a2e')
        vf.pack(fill='x', padx=10, pady=5)
        
        self.phase_deg_label = tk.Label(vf, text="137.5000°", 
                                        font=('Courier', 20, 'bold'),
                                        fg='#ffd700', bg='#1a1a2e')
        self.phase_deg_label.pack(side='left')
        
        self.phase_rad_label = tk.Label(vf, text="= 2.3999 rad", 
                                        font=('Courier', 12),
                                        fg='#888', bg='#1a1a2e')
        self.phase_rad_label.pack(side='left', padx=20)
        
        # Fine tune entry
        ef = tk.Frame(frame, bg='#1a1a2e')
        ef.pack(fill='x', padx=10, pady=5)
        
        tk.Label(ef, text="Exact angle:", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        self.phase_entry = tk.Entry(ef, font=('Courier', 12), width=15,
                                    bg='#2d2d44', fg='#ffd700', insertbackground='#ffd700')
        self.phase_entry.pack(side='left', padx=10)
        self.phase_entry.insert(0, "137.5")
        self.phase_entry.bind('<Return>', self._on_phase_entry)
        
        tk.Button(ef, text="Set", command=self._on_phase_entry,
                 bg='#3d3d5c', fg='#ffd700', font=('Courier', 10)).pack(side='left')
    
    def _create_sacred_presets(self, parent):
        frame = tk.LabelFrame(parent, text="✦ SACRED ANGLE PRESETS", 
                             font=('Helvetica', 12, 'bold'),
                             fg='#ffd700', bg='#1a1a2e', bd=2)
        frame.pack(fill='x', pady=10)
        
        # Create buttons for each sacred angle
        row = 0
        col = 0
        max_cols = 3
        
        btn_frame = tk.Frame(frame, bg='#1a1a2e')
        btn_frame.pack(fill='x', padx=5, pady=5)
        
        for name, angle in SACRED_ANGLES.items():
            # Truncate name if too long
            short_name = name[:18] + ".." if len(name) > 20 else name
            
            btn = tk.Button(btn_frame, text=f"{short_name}\n{angle:.4f}°",
                           command=lambda a=angle: self._set_sacred_angle(a),
                           bg='#2d2d44', fg='#00ff88', font=('Courier', 8),
                           width=14, height=2)
            btn.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(max_cols):
            btn_frame.columnconfigure(i, weight=1)
    
    def _create_playback_controls(self, parent):
        frame = tk.Frame(parent, bg='#1a1a2e')
        frame.pack(fill='x', pady=10)
        
        # Amplitude
        af = tk.Frame(frame, bg='#1a1a2e')
        af.pack(fill='x', pady=5)
        
        tk.Label(af, text="AMPLITUDE:", font=('Courier', 10), 
                fg='#888', bg='#1a1a2e').pack(side='left')
        
        self.amp_var = tk.DoubleVar(value=0.7)
        self.amp_scale = tk.Scale(af, from_=0, to=1, resolution=0.01,
                                  orient='horizontal', variable=self.amp_var,
                                  command=self._on_amp_change,
                                  bg='#2d2d44', fg='#00ff88',
                                  troughcolor='#1a1a2e', highlightthickness=0,
                                  length=200)
        self.amp_scale.pack(side='left', padx=10)
        
        # Buttons frame
        btn_frame = tk.Frame(frame, bg='#1a1a2e')
        btn_frame.pack(fill='x', pady=5)
        
        # Play button
        self.play_btn = tk.Button(btn_frame, text="▶ PLAY", 
                                  command=self._toggle_play,
                                  bg='#006400', fg='white',
                                  font=('Helvetica', 14, 'bold'),
                                  width=12, height=2)
        self.play_btn.pack(side='left', padx=10, pady=10)
        
        # 3D Oscilloscope button
        self.scope_btn = tk.Button(btn_frame, text="🌀 3D SCOPE", 
                                   command=self._launch_3d_oscilloscope,
                                   bg='#8B008B', fg='white',
                                   font=('Helvetica', 14, 'bold'),
                                   width=12, height=2)
        self.scope_btn.pack(side='left', padx=10, pady=10)
    
    def _create_visualization(self, parent):
        frame = tk.LabelFrame(parent, text="📊 PHASE VISUALIZATION", 
                             font=('Helvetica', 12, 'bold'),
                             fg='#ffd700', bg='#1a1a2e', bd=2)
        frame.pack(fill='both', expand=True)
        
        # Canvas for phase circle
        self.viz_canvas = tk.Canvas(frame, width=350, height=350,
                                    bg='#0d0d1a', highlightthickness=0)
        self.viz_canvas.pack(pady=10)
        
        # Info labels
        info_frame = tk.Frame(frame, bg='#1a1a2e')
        info_frame.pack(fill='x', padx=10)
        
        self.left_val_label = tk.Label(info_frame, text="L: 0.000", 
                                       font=('Courier', 12), fg='#ff6b6b', bg='#1a1a2e')
        self.left_val_label.pack(side='left', expand=True)
        
        self.right_val_label = tk.Label(info_frame, text="R: 0.000", 
                                        font=('Courier', 12), fg='#4ecdc4', bg='#1a1a2e')
        self.right_val_label.pack(side='left', expand=True)
        
        self.sum_val_label = tk.Label(info_frame, text="L+R: 0.000", 
                                      font=('Courier', 12, 'bold'), fg='#ffd700', bg='#1a1a2e')
        self.sum_val_label.pack(side='left', expand=True)
    
    def _draw_phase_circle(self, phase_deg):
        """Draw the phase visualization circle"""
        self.viz_canvas.delete('all')
        
        cx, cy = 175, 175  # Center
        r = 140  # Radius
        
        # Draw main circle
        self.viz_canvas.create_oval(cx-r, cy-r, cx+r, cy+r, 
                                    outline='#333', width=2)
        
        # Draw angle markers
        for angle in [0, 90, 180, 270]:
            rad = math.radians(angle - 90)  # -90 to start from top
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            self.viz_canvas.create_text(x, y, text=f"{angle}°", 
                                        fill='#666', font=('Courier', 10))
        
        # Draw sacred angle markers
        sacred_display = [
            (137.5, '#ffd700', 'φ'),  # Golden
            (137.036, '#ff6b6b', 'α'),  # Fine structure
            (180, '#00ff88', '⊕'),  # Cancellation
        ]
        
        for angle, color, symbol in sacred_display:
            rad = math.radians(angle - 90)
            x = cx + (r + 20) * math.cos(rad)
            y = cy + (r + 20) * math.sin(rad)
            self.viz_canvas.create_text(x, y, text=symbol, 
                                        fill=color, font=('Helvetica', 12, 'bold'))
        
        # Draw LEFT vector (reference at 0°)
        left_rad = math.radians(-90)  # Top
        left_x = cx + r * 0.9 * math.cos(left_rad)
        left_y = cy + r * 0.9 * math.sin(left_rad)
        self.viz_canvas.create_line(cx, cy, left_x, left_y, 
                                    fill='#ff6b6b', width=3, arrow='last')
        self.viz_canvas.create_text(left_x, left_y - 15, text="L", 
                                    fill='#ff6b6b', font=('Helvetica', 12, 'bold'))
        
        # Draw RIGHT vector (at phase_deg offset)
        right_rad = math.radians(phase_deg - 90)
        right_x = cx + r * 0.9 * math.cos(right_rad)
        right_y = cy + r * 0.9 * math.sin(right_rad)
        self.viz_canvas.create_line(cx, cy, right_x, right_y, 
                                    fill='#4ecdc4', width=3, arrow='last')
        self.viz_canvas.create_text(right_x + 15 * math.cos(right_rad), 
                                    right_y + 15 * math.sin(right_rad), 
                                    text="R", fill='#4ecdc4', font=('Helvetica', 12, 'bold'))
        
        # Draw arc showing phase difference
        if phase_deg > 0:
            arc_r = r * 0.5
            self.viz_canvas.create_arc(cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r,
                                       start=90, extent=-phase_deg,
                                       outline='#ffd700', width=2, style='arc')
        
        # Draw sum vector
        # Sum of unit vectors at 0° and phase_deg°
        sum_x = math.cos(math.radians(-90)) + math.cos(math.radians(phase_deg - 90))
        sum_y = math.sin(math.radians(-90)) + math.sin(math.radians(phase_deg - 90))
        sum_mag = math.sqrt(sum_x**2 + sum_y**2) / 2  # Normalize
        
        if sum_mag > 0.01:
            sum_angle = math.atan2(sum_y, sum_x)
            vec_x = cx + r * 0.6 * sum_mag * math.cos(sum_angle)
            vec_y = cy + r * 0.6 * sum_mag * math.sin(sum_angle)
            self.viz_canvas.create_line(cx, cy, vec_x, vec_y, 
                                        fill='#ffd700', width=4, arrow='last',
                                        dash=(5, 3))
        
        # Phase angle text in center
        self.viz_canvas.create_text(cx, cy + 50, text=f"{phase_deg:.4f}°",
                                    fill='#ffd700', font=('Courier', 16, 'bold'))
        
        # Sum magnitude indicator
        sum_text = f"|L+R| = {sum_mag*2:.3f}"
        color = '#00ff88' if sum_mag < 0.1 else '#ffd700' if sum_mag < 0.5 else '#ff6b6b'
        self.viz_canvas.create_text(cx, cy + 75, text=sum_text,
                                    fill=color, font=('Courier', 12))
    
    def _on_freq_change(self, _=None):
        fl = self.freq_left_var.get()
        fr = self.freq_right_var.get()
        
        self.freq_left_label.config(text=f"{fl:.1f} Hz")
        self.freq_right_label.config(text=f"{fr:.1f} Hz")
        self.beat_label.config(text=f"{abs(fr - fl):.2f} Hz")
        
        self.engine.set_frequencies(fl, fr)
    
    def _on_phase_change(self, _=None):
        phase = self.phase_var.get()
        rad = phase * math.pi / 180
        
        self.phase_deg_label.config(text=f"{phase:.4f}°")
        self.phase_rad_label.config(text=f"= {rad:.4f} rad")
        self.phase_entry.delete(0, tk.END)
        self.phase_entry.insert(0, f"{phase:.6f}")
        
        self.engine.set_phase_angle(phase)
        self._draw_phase_circle(phase)
    
    def _on_phase_entry(self, _=None):
        try:
            val = float(self.phase_entry.get())
            val = val % 360  # Wrap to 0-360
            self.phase_var.set(val)
            self._on_phase_change()
        except ValueError:
            pass
    
    def _on_amp_change(self, _=None):
        self.engine.set_amplitude(self.amp_var.get())
    
    def _set_sacred_angle(self, angle: float):
        self.phase_var.set(angle)
        self._on_phase_change()
    
    def _toggle_play(self):
        if self.engine.playing:
            self._stop_audio()
        else:
            self._start_audio()
    
    def _launch_3d_oscilloscope(self):
        """Launch the 3D oscilloscope in a separate process with current parameters"""
        import subprocess
        import sys
        import os
        
        # Get current parameters
        freq_left = self.freq_left_var.get()
        freq_right = self.freq_right_var.get()
        phase_angle = self.phase_var.get()
        amplitude = self.amp_var.get()
        waveform = self.engine.waveform_mode
        
        # Get device index
        device_info = self.device_manager.get_device_info(self.selected_device_idx)
        device_index = device_info['index'] if device_info else None
        
        # Path to oscilloscope script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scope_script = os.path.join(script_dir, 'oscilloscope_3d.py')
        
        # Build command
        cmd = [
            sys.executable, scope_script,
            '--freq-left', str(freq_left),
            '--freq-right', str(freq_right),
            '--phase', str(phase_angle),
            '--amplitude', str(amplitude),
            '--waveform', waveform
        ]
        
        if device_index is not None:
            cmd.extend(['--device', str(device_index)])
        
        # Launch in separate process
        try:
            subprocess.Popen(cmd)
            print(f"🌀 3D Oscilloscope launched with:")
            print(f"   L: {freq_left:.2f} Hz, R: {freq_right:.2f} Hz")
            print(f"   Phase: {phase_angle:.3f}°")
            print(f"   Waveform: {waveform}")
        except Exception as e:
            print(f"Error launching oscilloscope: {e}")
    
    def _start_audio(self):
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            
            def callback(in_data, frame_count, time_info, status):
                data = self.engine.generate_chunk(frame_count)
                return (data, pyaudio.paContinue)
            
            # Get selected device index
            device_info = self.device_manager.get_device_info(self.selected_device_idx)
            device_index = device_info['index'] if device_info else None
            
            # Build stream parameters
            stream_params = {
                'format': pyaudio.paFloat32,
                'channels': 2,
                'rate': SAMPLE_RATE,
                'output': True,
                'frames_per_buffer': 1024,
                'stream_callback': callback
            }
            
            # Add device index if specified
            if device_index is not None:
                stream_params['output_device_index'] = device_index
            
            self.stream = self.pyaudio.open(**stream_params)
            
            self.engine.playing = True
            self.play_btn.config(text="⏹ STOP", bg='#8b0000')
            
            # Update status
            if device_info:
                print(f"Audio started on: {device_info['name']}")
            
        except Exception as e:
            print(f"Audio error: {e}")
            import traceback
            traceback.print_exc()
    
    def _stop_audio(self):
        self.engine.playing = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
        
        self.play_btn.config(text="▶ PLAY", bg='#006400')
    
    def _start_visualization(self):
        """Start visualization update loop"""
        self._draw_phase_circle(self.phase_var.get())
        self._update_values()
    
    def _update_values(self):
        """Update value displays"""
        if self.engine.playing:
            vals = self.engine.get_instantaneous_values()
            self.left_val_label.config(text=f"L: {vals['left']:+.3f}")
            self.right_val_label.config(text=f"R: {vals['right']:+.3f}")
            self.sum_val_label.config(text=f"L+R: {vals['sum']:+.3f}")
        
        self.root.after(50, self._update_values)
    
    def cleanup(self):
        self._stop_audio()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DUAL FREQUENCY PHASE ANGLE CONTROLLER                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Control TWO frequencies independently (Left & Right channels)               ║
║  Set PHASE DIFFERENCE as ANGLE in degrees                                    ║
║                                                                              ║
║  SACRED ANGLES:                                                              ║
║    • Golden Angle:     137.5077° = 360°/φ²                                  ║
║    • Fine Structure:   137.0360° = α⁻¹                                      ║
║    • DNA Helix:         34.3°    = rotation per base pair                   ║
║    • Pentagon:         108.0°    = internal angle                           ║
║    • Pyramid Giza:      51.84°   = slope angle                              ║
║    • Cancellation:     180.0°    = phase cancellation → SILENCE             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    root = tk.Tk()
    app = PhaseAngleGUI(root)
    
    def on_close():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
