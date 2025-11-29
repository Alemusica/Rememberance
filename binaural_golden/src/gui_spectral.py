#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GUI SPECTRAL SOUND - SUONA GLI ELEMENTI                   ║
║                                                                              ║
║   Interfaccia per generare suoni dalle linee spettrali atomiche             ║
║   Tavola periodica sonora!                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
from typing import Optional

# Import del modulo spectral_sound
from spectral_sound import (
    SpectralSounder, PhaseMode, SpectralLine,
    PHI, PHI_CONJUGATE, SAMPLE_RATE
)

# PyAudio per riproduzione
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("⚠️ PyAudio non disponibile - solo generazione file")


class SpectralSoundGUI:
    """GUI per suonare gli elementi"""
    
    def __init__(self):
        self.sounder = SpectralSounder()
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.playing = False
        self.current_signal: Optional[np.ndarray] = None
        self.current_signal_stereo: Optional[tuple] = None
        self.playback_position = 0
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Configura l'interfaccia grafica"""
        self.root = tk.Tk()
        self.root.title("🌈 SPECTRAL SOUND - Suona gli Elementi")
        self.root.geometry("900x750")
        self.root.configure(bg='#0d0d1a')
        
        # Header
        header = tk.Label(
            self.root,
            text="🌈 SPECTRAL SOUND 🎵",
            font=('Helvetica', 24, 'bold'),
            fg='#ffd700',
            bg='#0d0d1a'
        )
        header.pack(pady=10)
        
        subtitle = tk.Label(
            self.root,
            text="Trasforma le linee spettrali atomiche in suono",
            font=('Helvetica', 12),
            fg='#888',
            bg='#0d0d1a'
        )
        subtitle.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#0d0d1a')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel: Element selection
        left_panel = tk.Frame(main_frame, bg='#1a1a2e', bd=2, relief='groove')
        left_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        self._create_element_selector(left_panel)
        self._create_parameters_panel(left_panel)
        self._create_playback_controls(left_panel)
        
        # Right panel: Visualization
        right_panel = tk.Frame(main_frame, bg='#1a1a2e', bd=2, relief='groove')
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        self._create_spectrum_display(right_panel)
        self._create_info_panel(right_panel)
        
        # Status bar
        self.status_var = tk.StringVar(value="Seleziona un elemento per iniziare")
        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Courier', 10),
            fg='#00ff88',
            bg='#0d0d1a'
        )
        status.pack(pady=5)
        
        # Bindings
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_element_selector(self, parent):
        """Crea il selettore degli elementi"""
        frame = tk.LabelFrame(
            parent,
            text="⚛️ SELEZIONA ELEMENTO",
            font=('Helvetica', 12, 'bold'),
            fg='#ffd700',
            bg='#1a1a2e',
            bd=2
        )
        frame.pack(fill='x', padx=10, pady=10)
        
        # Dropdown elementi
        self.element_var = tk.StringVar()
        elements = self.sounder.get_element_names()
        
        self.element_combo = ttk.Combobox(
            frame,
            textvariable=self.element_var,
            values=elements,
            state='readonly',
            width=30,
            font=('Courier', 11)
        )
        self.element_combo.pack(pady=10)
        self.element_combo.bind('<<ComboboxSelected>>', self._on_element_select)
        
        # Preset buttons
        presets_frame = tk.Frame(frame, bg='#1a1a2e')
        presets_frame.pack(fill='x', pady=5)
        
        presets = [
            ("🔴 H", "Hydrogen-Balmer"),
            ("🟡 He", "Helium"),
            ("🟠 Na", "Sodium"),
            ("🔵 Ne", "Neon"),
            ("⚪ Hg", "Mercury"),
            ("🟢 O", "Oxygen"),
        ]
        
        for i, (label, element) in enumerate(presets):
            btn = tk.Button(
                presets_frame,
                text=label,
                command=lambda e=element: self._select_element(e),
                bg='#2d2d44',
                fg='white',
                font=('Helvetica', 10),
                width=6
            )
            btn.grid(row=i//3, column=i%3, padx=3, pady=3)
    
    def _create_parameters_panel(self, parent):
        """Crea il pannello dei parametri"""
        frame = tk.LabelFrame(
            parent,
            text="🎛️ PARAMETRI",
            font=('Helvetica', 12, 'bold'),
            fg='#ffd700',
            bg='#1a1a2e',
            bd=2
        )
        frame.pack(fill='x', padx=10, pady=10)
        
        # Duration
        dur_frame = tk.Frame(frame, bg='#1a1a2e')
        dur_frame.pack(fill='x', pady=5)
        
        tk.Label(dur_frame, text="Durata (s):", fg='#888', bg='#1a1a2e',
                font=('Courier', 10)).pack(side='left', padx=5)
        
        self.duration_var = tk.DoubleVar(value=3.0)
        duration_scale = tk.Scale(
            dur_frame,
            from_=0.5, to=10.0,
            resolution=0.5,
            orient='horizontal',
            variable=self.duration_var,
            bg='#2d2d44',
            fg='#00ff88',
            troughcolor='#1a1a2e',
            highlightthickness=0,
            length=200
        )
        duration_scale.pack(side='left', padx=10)
        
        # Phase mode
        phase_frame = tk.Frame(frame, bg='#1a1a2e')
        phase_frame.pack(fill='x', pady=5)
        
        tk.Label(phase_frame, text="Modalità Fase:", fg='#888', bg='#1a1a2e',
                font=('Courier', 10)).pack(side='left', padx=5)
        
        self.phase_var = tk.StringVar(value="GOLDEN")
        phase_modes = [
            ("Incoerente (Quantistico)", "INCOHERENT"),
            ("Coerente (Armonica)", "COHERENT"),
            ("Golden Ratio", "GOLDEN"),
            ("Fibonacci", "FIBONACCI")
        ]
        
        for text, mode in phase_modes:
            rb = tk.Radiobutton(
                phase_frame,
                text=text,
                variable=self.phase_var,
                value=mode,
                bg='#1a1a2e',
                fg='#4ecdc4',
                selectcolor='#2d2d44',
                font=('Courier', 9)
            )
            rb.pack(anchor='w', padx=20)
        
        # Audio range
        range_frame = tk.Frame(frame, bg='#1a1a2e')
        range_frame.pack(fill='x', pady=5)
        
        tk.Label(range_frame, text="Range Audio:", fg='#888', bg='#1a1a2e',
                font=('Courier', 10)).pack(side='left', padx=5)
        
        tk.Label(range_frame, text="Min:", fg='#666', bg='#1a1a2e',
                font=('Courier', 9)).pack(side='left')
        self.freq_min_var = tk.IntVar(value=50)
        tk.Entry(range_frame, textvariable=self.freq_min_var, width=6,
                bg='#2d2d44', fg='white').pack(side='left', padx=2)
        
        tk.Label(range_frame, text="Max:", fg='#666', bg='#1a1a2e',
                font=('Courier', 9)).pack(side='left', padx=5)
        self.freq_max_var = tk.IntVar(value=4000)
        tk.Entry(range_frame, textvariable=self.freq_max_var, width=6,
                bg='#2d2d44', fg='white').pack(side='left', padx=2)
        
        # Beat frequency (for binaural)
        beat_frame = tk.Frame(frame, bg='#1a1a2e')
        beat_frame.pack(fill='x', pady=5)
        
        tk.Label(beat_frame, text="Beat Freq (Hz):", fg='#888', bg='#1a1a2e',
                font=('Courier', 10)).pack(side='left', padx=5)
        
        self.beat_var = tk.DoubleVar(value=7.83)  # Schumann
        beat_scale = tk.Scale(
            beat_frame,
            from_=0.5, to=40.0,
            resolution=0.5,
            orient='horizontal',
            variable=self.beat_var,
            bg='#2d2d44',
            fg='#ff6b6b',
            troughcolor='#1a1a2e',
            highlightthickness=0,
            length=150
        )
        beat_scale.pack(side='left', padx=10)
        
        # Beat presets
        beat_presets = tk.Frame(frame, bg='#1a1a2e')
        beat_presets.pack(fill='x', pady=2)
        
        for name, freq in [("Schumann", 7.83), ("Theta", 6.0), ("Alpha", 10.0), ("Beta", 20.0)]:
            btn = tk.Button(
                beat_presets,
                text=name,
                command=lambda f=freq: self.beat_var.set(f),
                bg='#3d3d5c',
                fg='white',
                font=('Helvetica', 8),
                width=8
            )
            btn.pack(side='left', padx=2)
    
    def _create_playback_controls(self, parent):
        """Crea i controlli di riproduzione"""
        frame = tk.LabelFrame(
            parent,
            text="▶️ RIPRODUZIONE",
            font=('Helvetica', 12, 'bold'),
            fg='#ffd700',
            bg='#1a1a2e',
            bd=2
        )
        frame.pack(fill='x', padx=10, pady=10)
        
        # Mode selection
        mode_frame = tk.Frame(frame, bg='#1a1a2e')
        mode_frame.pack(fill='x', pady=5)
        
        self.output_mode_var = tk.StringVar(value="stereo")
        modes = [("Mono", "mono"), ("Stereo", "stereo"), ("Binaural", "binaural")]
        
        for text, mode in modes:
            rb = tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.output_mode_var,
                value=mode,
                bg='#1a1a2e',
                fg='#ff6b6b',
                selectcolor='#2d2d44',
                font=('Courier', 10)
            )
            rb.pack(side='left', padx=10)
        
        # Buttons
        btn_frame = tk.Frame(frame, bg='#1a1a2e')
        btn_frame.pack(fill='x', pady=10)
        
        self.play_btn = tk.Button(
            btn_frame,
            text="▶ SUONA",
            command=self._on_play,
            bg='#006400',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            width=10,
            height=2
        )
        self.play_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⏹ STOP",
            command=self._on_stop,
            bg='#8b0000',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            width=10,
            height=2,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=10)
        
        self.save_btn = tk.Button(
            btn_frame,
            text="💾 SALVA",
            command=self._on_save,
            bg='#4a4a8a',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            width=10
        )
        self.save_btn.pack(side='left', padx=10)
    
    def _create_spectrum_display(self, parent):
        """Crea il display dello spettro"""
        frame = tk.LabelFrame(
            parent,
            text="📊 SPETTRO",
            font=('Helvetica', 12, 'bold'),
            fg='#ffd700',
            bg='#1a1a2e',
            bd=2
        )
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas per visualizzazione
        self.spectrum_canvas = tk.Canvas(
            frame,
            width=350,
            height=250,
            bg='#0a0a15',
            highlightthickness=0
        )
        self.spectrum_canvas.pack(pady=10)
        
        # Draw initial empty spectrum
        self._draw_empty_spectrum()
    
    def _create_info_panel(self, parent):
        """Crea il pannello informazioni"""
        frame = tk.LabelFrame(
            parent,
            text="ℹ️ INFORMAZIONI LINEE",
            font=('Helvetica', 12, 'bold'),
            fg='#ffd700',
            bg='#1a1a2e',
            bd=2
        )
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Text widget per info
        self.info_text = tk.Text(
            frame,
            width=40,
            height=12,
            bg='#0a0a15',
            fg='#00ff88',
            font=('Courier', 9),
            state='disabled'
        )
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _draw_empty_spectrum(self):
        """Disegna spettro vuoto"""
        self.spectrum_canvas.delete('all')
        
        # Axes
        self.spectrum_canvas.create_line(30, 220, 330, 220, fill='#333', width=2)  # X
        self.spectrum_canvas.create_line(30, 220, 30, 20, fill='#333', width=2)    # Y
        
        # Labels
        self.spectrum_canvas.create_text(180, 240, text="Frequenza Audio (Hz)",
                                         fill='#666', font=('Courier', 8))
        self.spectrum_canvas.create_text(15, 120, text="Amp", angle=90,
                                         fill='#666', font=('Courier', 8))
        
        # Message
        self.spectrum_canvas.create_text(180, 120, 
                                         text="Seleziona un elemento",
                                         fill='#444', font=('Helvetica', 12))
    
    def _draw_spectrum(self, element: str):
        """Disegna lo spettro dell'elemento"""
        self.spectrum_canvas.delete('all')
        
        lines = self.sounder.get_spectral_lines(element)
        if not lines:
            self._draw_empty_spectrum()
            return
        
        # Update sounder range
        self.sounder.audio_min = self.freq_min_var.get()
        self.sounder.audio_max = self.freq_max_var.get()
        
        scaled = self.sounder.scale_to_audio(lines)
        
        # Axes
        self.spectrum_canvas.create_line(30, 220, 330, 220, fill='#333', width=2)
        self.spectrum_canvas.create_line(30, 220, 30, 20, fill='#333', width=2)
        
        # Draw bars
        bar_width = 250 / len(lines)
        colors = ['#ff6b6b', '#ffd700', '#00ff88', '#4ecdc4', '#ff00ff', '#00bfff', '#ff8c00']
        
        for i, ((freq, amp), line) in enumerate(zip(scaled, lines)):
            # Map frequency to X position
            x = 40 + i * bar_width + bar_width/2
            
            # Height from amplitude
            height = amp * 180
            
            color = colors[i % len(colors)]
            
            # Bar
            self.spectrum_canvas.create_rectangle(
                x - bar_width/3, 220 - height,
                x + bar_width/3, 220,
                fill=color, outline=''
            )
            
            # Label
            self.spectrum_canvas.create_text(
                x, 230, text=f"{int(freq)}", fill='#666', font=('Courier', 7)
            )
            self.spectrum_canvas.create_text(
                x, 220 - height - 10, text=line.name.split('-')[-1] if '-' in line.name else line.name[:3],
                fill=color, font=('Courier', 8, 'bold')
            )
        
        # Title
        self.spectrum_canvas.create_text(
            180, 10, text=f"Spettro Audio: {element}",
            fill='#ffd700', font=('Helvetica', 10, 'bold')
        )
    
    def _update_info(self, element: str):
        """Aggiorna le informazioni sulle linee"""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        
        lines = self.sounder.get_spectral_lines(element)
        if not lines:
            self.info_text.insert('end', "Nessuna informazione disponibile")
            self.info_text.config(state='disabled')
            return
        
        # Update range
        self.sounder.audio_min = self.freq_min_var.get()
        self.sounder.audio_max = self.freq_max_var.get()
        scaled = self.sounder.scale_to_audio(lines)
        
        self.info_text.insert('end', f"╔═══ {element} ═══╗\n\n")
        self.info_text.insert('end', f"{'Linea':<8} {'λ(nm)':<8} {'f(Hz)':<8} {'Amp':<6}\n")
        self.info_text.insert('end', "─" * 35 + "\n")
        
        for line, (f_audio, amp) in zip(lines, scaled):
            short_name = line.name[:7]
            self.info_text.insert('end', 
                f"{short_name:<8} {line.wavelength_nm:<8.1f} {f_audio:<8.0f} {amp:<6.2f}\n")
        
        self.info_text.insert('end', "\n" + "─" * 35 + "\n")
        self.info_text.insert('end', f"Totale linee: {len(lines)}\n")
        
        self.info_text.config(state='disabled')
    
    def _select_element(self, element: str):
        """Seleziona un elemento"""
        self.element_var.set(element)
        self._on_element_select(None)
    
    def _on_element_select(self, event):
        """Gestisce la selezione di un elemento"""
        element = self.element_var.get()
        if element:
            self._draw_spectrum(element)
            self._update_info(element)
            self.status_var.set(f"✅ Selezionato: {element}")
    
    def _get_phase_mode(self) -> PhaseMode:
        """Ottiene la modalità fase selezionata"""
        mode_str = self.phase_var.get()
        return PhaseMode[mode_str]
    
    def _on_play(self):
        """Avvia la riproduzione"""
        element = self.element_var.get()
        if not element:
            messagebox.showwarning("Attenzione", "Seleziona un elemento!")
            return
        
        if not HAS_PYAUDIO:
            messagebox.showerror("Errore", "PyAudio non disponibile")
            return
        
        # Update sounder
        self.sounder.audio_min = self.freq_min_var.get()
        self.sounder.audio_max = self.freq_max_var.get()
        
        # Generate sound
        duration = self.duration_var.get()
        phase_mode = self._get_phase_mode()
        output_mode = self.output_mode_var.get()
        
        try:
            if output_mode == "mono":
                self.current_signal = self.sounder.generate_element_sound(
                    element, duration, phase_mode
                )
                self.current_signal_stereo = None
            elif output_mode == "stereo":
                left, right = self.sounder.generate_element_stereo(
                    element, duration, phase_mode
                )
                self.current_signal_stereo = (left, right)
                self.current_signal = None
            else:  # binaural
                left, right = self.sounder.generate_binaural_element(
                    element, self.beat_var.get(), duration, phase_mode
                )
                self.current_signal_stereo = (left, right)
                self.current_signal = None
            
            # Start playback
            self._start_playback()
            
        except Exception as e:
            messagebox.showerror("Errore", str(e))
    
    def _start_playback(self):
        """Avvia la riproduzione audio"""
        self.playing = True
        self.playback_position = 0
        
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("🔊 Riproduzione in corso...")
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self._audio_playback)
        self.audio_thread.start()
    
    def _audio_playback(self):
        """Thread di riproduzione audio"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            if self.current_signal_stereo:
                left, right = self.current_signal_stereo
                # Interleave
                stereo = np.empty(len(left) * 2, dtype=np.float32)
                stereo[0::2] = left.astype(np.float32)
                stereo[1::2] = right.astype(np.float32)
                channels = 2
                data = stereo.tobytes()
            else:
                channels = 1
                data = self.current_signal.astype(np.float32).tobytes()
            
            self.stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=SAMPLE_RATE,
                output=True
            )
            
            # Play in chunks
            chunk_size = 1024 * channels * 4  # bytes
            position = 0
            
            while position < len(data) and self.playing:
                chunk = data[position:position + chunk_size]
                if chunk:
                    self.stream.write(chunk)
                position += chunk_size
            
            # Cleanup
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio.terminate()
            
        except Exception as e:
            print(f"Audio error: {e}")
        
        finally:
            self.playing = False
            self.root.after(0, self._playback_finished)
    
    def _playback_finished(self):
        """Chiamata quando la riproduzione termina"""
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("✅ Riproduzione completata")
    
    def _on_stop(self):
        """Ferma la riproduzione"""
        self.playing = False
        self.status_var.set("⏹ Fermato")
    
    def _on_save(self):
        """Salva il suono corrente"""
        element = self.element_var.get()
        if not element:
            messagebox.showwarning("Attenzione", "Seleziona un elemento!")
            return
        
        # Update sounder
        self.sounder.audio_min = self.freq_min_var.get()
        self.sounder.audio_max = self.freq_max_var.get()
        
        duration = self.duration_var.get()
        phase_mode = self._get_phase_mode()
        output_mode = self.output_mode_var.get()
        
        filename = f"{element.lower().replace('-', '_')}_{output_mode}.wav"
        
        try:
            if output_mode == "mono":
                signal = self.sounder.generate_element_sound(element, duration, phase_mode)
                self.sounder.save_wav(signal, filename)
            else:
                if output_mode == "stereo":
                    left, right = self.sounder.generate_element_stereo(element, duration, phase_mode)
                else:
                    left, right = self.sounder.generate_binaural_element(
                        element, self.beat_var.get(), duration, phase_mode
                    )
                self.sounder.save_wav(left, filename, stereo=True, right_channel=right)
            
            self.status_var.set(f"💾 Salvato: {filename}")
            messagebox.showinfo("Salvato", f"File salvato: {filename}")
            
        except Exception as e:
            messagebox.showerror("Errore", str(e))
    
    def _on_close(self):
        """Gestisce la chiusura"""
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        self.root.destroy()
    
    def run(self):
        """Avvia la GUI"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPECTRAL SOUND GUI                                        ║
║                    Suonare gli elementi della tavola periodica               ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        self.root.mainloop()


if __name__ == "__main__":
    app = SpectralSoundGUI()
    app.run()
