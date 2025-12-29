"""
SpectralTab - Play atomic elements as sound
═══════════════════════════════════════════

Each chemical element has a unique emission spectrum.
This tab converts those spectral lines to audible frequencies.

Features:
- Element selection with presets
- Phase modes: INCOHERENT, COHERENT, GOLDEN, FIBONACCI
- Output modes: mono, stereo, binaural
- Real-time spectrum visualization
- WAV export

Based on NIST atomic spectra database.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from golden_studio import AudioEngine

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from spectral_sound import SpectralSounder, PhaseMode
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False
    SpectralSounder = None
    PhaseMode = None


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL TAB
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralTab:
    """Play atomic elements as sound"""
    
    def __init__(self, parent, audio_engine: 'AudioEngine'):
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
        """Stop playback"""
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
