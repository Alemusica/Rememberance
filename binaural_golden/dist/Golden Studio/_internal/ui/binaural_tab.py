"""
Binaural Tab - Binaural beats with phase angle control.

Features:
- Continuous real-time playback
- Beat frequency (battimenti) control
- Musical intervals (just intonation)
- Sacred angle presets
- Phase cancellation testing
- 3D oscilloscope launch
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import sys
import subprocess
import json
from typing import Tuple

# Try importing from core module
try:
    from core.audio_engine import AudioEngine
    from core.golden_math import PHI, PHI_CONJUGATE
except ImportError:
    PHI = 1.618033988749895
    PHI_CONJUGATE = 0.6180339887498949
    AudioEngine = None

# Audio constants
SAMPLE_RATE = 44100

# Sacred angles for phase
SACRED_ANGLES = {
    "Golden Angle (φ)": 137.5077640500378,
    "Pentagon (72°)": 72.0,
    "Pentagram (36°)": 36.0,
    "Hexagon (60°)": 60.0,
    "Pyramid (51.84°)": 51.84,
    "DNA Twist (34.38°)": 34.377468,
    "Quarter (90°)": 90.0,
    "Opposite (180°)": 180.0,
}


class BinauralTab:
    """Binaural beats with phase angle control - CONTINUOUS PLAYBACK"""
    
    def __init__(self, parent, audio_engine):
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
        
        intervals_row1 = ttk.Frame(interval_frame)
        intervals_row1.pack(fill='x', pady=2)
        intervals_row2 = ttk.Frame(interval_frame)
        intervals_row2.pack(fill='x', pady=2)
        
        musical_intervals = [
            ("Unison", 1, 1), ("m2", 16, 15), ("M2", 9, 8),
            ("m3", 6, 5), ("M3", 5, 4), ("P4", 4, 3),
            ("Tritone", 45, 32), ("P5", 3, 2), ("m6", 8, 5),
            ("M6", 5, 3), ("m7", 9, 5), ("Octave", 2, 1),
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
            ("Meditation", 432, 7.83, 137.5),
            ("Focus", 440, 14, 108),
            ("Sleep", 396, 3, 51.84),
            ("Healing", 528, 8, 137.5),
        ]
        
        for name, base, beat, phase in quick_presets:
            btn = ttk.Button(quick_preset_row, text=name, width=10,
                           command=lambda b=base, bt=beat, p=phase: self._apply_quick_preset(b, bt, p))
            btn.pack(side='left', padx=1)
        
        # Calculated frequencies display
        calc_frame = ttk.LabelFrame(left_frame, text="📊 Resulting Frequencies", padding=5)
        calc_frame.pack(fill='x', pady=5)
        
        self.calc_label = ttk.Label(calc_frame, text="L: 432.0 Hz | R: 440.0 Hz | Beat: 8.0 Hz",
                                    font=('Courier', 10, 'bold'))
        self.calc_label.pack(pady=5)
        
        # Manual frequency controls
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
        
        # Amplitude
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
        
        test_frame = ttk.Frame(mono_frame)
        test_frame.pack(fill='x', pady=3)
        ttk.Button(test_frame, text="Test 180° Cancel", 
                   command=self._test_180_cancel).pack(side='left', padx=2)
        ttk.Button(test_frame, text="Test 0° Sum", 
                   command=self._test_0_sum).pack(side='left', padx=2)
        
        # Playback buttons
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
        
        self.info_var = tk.StringVar(value="Ready - Audio will play continuously until STOP")
        ttk.Label(right_frame, textvariable=self.info_var, wraplength=280).pack()
        
        self._draw_phase_circle()
        
        # Bind updates
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
        
        ratio = numerator / denominator
        cents = 1200 * np.log2(ratio)
        self.info_var.set(f"Interval: {numerator}:{denominator} = {ratio:.4f} ({cents:.1f} cents)")
    
    def _apply_quick_preset(self, base: float, beat: float, phase: float):
        """Apply a quick preset"""
        self.link_mode.set("beat")
        self.base_freq.set(base)
        self.beat_freq.set(beat)
        self.phase_angle.set(phase)
        self._update_frequencies()
    
    def _save_preset(self):
        """Save current settings to preset file"""
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
        
        preset_dir = os.path.join(os.path.dirname(__file__), '..', 'presets')
        os.makedirs(preset_dir, exist_ok=True)
        
        filepath = os.path.join(preset_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(preset, f, indent=2)
        
        self.info_var.set(f"💾 Saved: {name}.json")
    
    def _load_preset(self):
        """Load settings from preset file"""
        name = self.preset_name_var.get().strip()
        if not name:
            return
        
        preset_dir = os.path.join(os.path.dirname(__file__), '..', 'presets')
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
        """Update L/R frequencies - AND UPDATE AUDIO IN REAL-TIME"""
        try:
            if self.link_mode.get() == "beat":
                base = self.base_freq.get()
                beat = self.beat_freq.get()
                self.freq_left.set(base)
                self.freq_right.set(base + beat)
            
            l = self.freq_left.get()
            r = self.freq_right.get()
            b = abs(r - l)
            self.calc_label.config(text=f"L: {l:.1f} Hz | R: {r:.1f} Hz | Beat: {b:.2f} Hz")
            
            if self.audio and self.audio.is_playing():
                self.audio.set_frequencies(l, r)
        except:
            pass
    
    def _on_phase_change(self, *args):
        """Update audio in real-time when phase changes"""
        try:
            phase = self.phase_angle.get()
            self._draw_phase_circle()
            
            if self.audio and self.audio.is_playing():
                self.audio.set_phase_angle(phase)
        except:
            pass
    
    def _on_amplitude_change(self, *args):
        """Update audio when amplitude changes"""
        try:
            amp = self.amplitude.get()
            if self.audio and self.audio.is_playing():
                self.audio.set_amplitude(amp)
        except:
            pass
    
    def _on_waveform_change(self, *args):
        """Update audio when waveform changes"""
        try:
            wf = self.waveform.get()
            if self.audio and self.audio.is_playing():
                self.audio.set_waveform(wf)
        except:
            pass
    
    def _on_freq_change(self, *args):
        """Update audio when manual frequencies change"""
        try:
            if self.link_mode.get() == "manual":
                l = self.freq_left.get()
                r = self.freq_right.get()
                b = abs(r - l)
                self.calc_label.config(text=f"L: {l:.1f} Hz | R: {r:.1f} Hz | Beat: {b:.2f} Hz")
                
                if self.audio and self.audio.is_playing():
                    self.audio.set_frequencies(l, r)
        except:
            pass
    
    def _on_mono_change(self):
        """Handle mono mix checkbox change"""
        mono = self.mono_mix_var.get()
        if self.audio:
            self.audio.set_mono_mix(mono)
        if mono:
            self.info_var.set("⚠️ MONO MIX ON: L+R summed. 180° with same freq = SILENCE!")
        else:
            self.info_var.set("STEREO: L and R go to separate ears (binaural effect)")
    
    def _test_180_cancel(self):
        """Test: 180° phase, same freq, mono = SILENCE"""
        self.link_mode.set("manual")
        self.freq_left.set(440.0)
        self.freq_right.set(440.0)
        self.phase_angle.set(180.0)
        self.mono_mix_var.set(True)
        self._on_mono_change()
        self._update_frequencies()
        if self.audio and not self.audio.is_playing():
            self._play()
        self.info_var.set("🔇 TEST: 440Hz + 440Hz @ 180° MONO → SILENCE!")
    
    def _test_0_sum(self):
        """Test: 0° phase, same freq, mono = LOUD"""
        self.link_mode.set("manual")
        self.freq_left.set(440.0)
        self.freq_right.set(440.0)
        self.phase_angle.set(0.0)
        self.mono_mix_var.set(True)
        self._on_mono_change()
        self._update_frequencies()
        if self.audio and not self.audio.is_playing():
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
        
        # Beat frequency
        beat = abs(self.freq_right.get() - self.freq_left.get())
        self.canvas.create_text(cx, cy + r + 40, text=f"Beat: {beat:.2f} Hz",
                               fill='#00ff88', font=('Courier', 10, 'bold'))
    
    def _play(self):
        """Start CONTINUOUS callback-based playback"""
        if not self.audio:
            self.info_var.set("Audio engine not available")
            return
        
        freq_l = self.freq_left.get()
        freq_r = self.freq_right.get()
        phase = self.phase_angle.get()
        amp = self.amplitude.get()
        wf = self.waveform.get()
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        beat = abs(freq_r - freq_l)
        self.info_var.set(f"🔊 Playing... L:{freq_l:.0f}Hz R:{freq_r:.0f}Hz Beat:{beat:.2f}Hz")
        
        self.audio.start_binaural(freq_l, freq_r, phase, amp, wf)
    
    def _stop(self):
        """Stop playback"""
        if self.audio:
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
        scope_script = os.path.join(script_dir, '..', 'oscilloscope_3d.py')
        
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
