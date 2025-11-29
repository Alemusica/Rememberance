"""
Real-time Golden Binaural Visualization & Audio
================================================

Interactive interface for:
- Real-time audio generation
- Golden spiral visualization
- Parameter control
- Annealing progress display
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import queue
import struct
import wave
from dataclasses import dataclass
from typing import Optional, Callable

# Divine Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_CONJUGATE = PHI - 1

# Colors in golden ratio RGB values
GOLD_PRIMARY = "#D4AF37"
GOLD_SECONDARY = "#FFD700"
DIVINE_PURPLE = "#4B0082"
COSMIC_BLUE = "#191970"


def golden_spiral_points(num_points: int = 200, turns: float = 5) -> tuple:
    """Generate points on a golden spiral for visualization"""
    theta = np.linspace(0, turns * 2 * np.pi, num_points)
    # Golden spiral: r = a * e^(b*θ) where b = ln(φ)/(π/2)
    b = np.log(PHI) / (np.pi / 2)
    r = np.exp(b * theta)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Normalize to [-1, 1]
    max_r = np.max(np.sqrt(x**2 + y**2))
    return x / max_r, y / max_r


class GoldenSpiralCanvas:
    """Canvas for drawing golden spiral visualizations"""
    
    def __init__(self, parent, width=400, height=400):
        self.canvas = tk.Canvas(
            parent, 
            width=width, 
            height=height, 
            bg=COSMIC_BLUE,
            highlightthickness=0
        )
        self.canvas.pack(pady=10)
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Pre-compute golden spiral
        self.spiral_x, self.spiral_y = golden_spiral_points(300)
        
        # Draw initial spiral
        self.draw_golden_spiral()
        
        # Animation state
        self.phase = 0
        self.amplitude = 1.0
        
    def draw_golden_spiral(self):
        """Draw the base golden spiral"""
        self.canvas.delete("spiral")
        
        scale = min(self.width, self.height) * 0.4
        
        # Draw spiral segments with golden color gradient
        for i in range(len(self.spiral_x) - 1):
            x1 = self.center_x + self.spiral_x[i] * scale
            y1 = self.center_y + self.spiral_y[i] * scale
            x2 = self.center_x + self.spiral_x[i+1] * scale
            y2 = self.center_y + self.spiral_y[i+1] * scale
            
            # Color intensity based on position
            intensity = int(155 + 100 * (i / len(self.spiral_x)))
            color = f"#{intensity:02x}{int(intensity*0.7):02x}37"
            
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=2,
                tags="spiral"
            )
    
    def update_visualization(self, phase: float, amplitude: float, beat_freq: float):
        """Update visualization based on current audio state"""
        self.phase = phase
        self.amplitude = amplitude
        
        self.canvas.delete("waves")
        
        # Draw two interfering waves
        scale = min(self.width, self.height) * 0.35 * amplitude
        
        # Left wave (base frequency representation)
        points1 = []
        for i in range(100):
            x = self.center_x + (i - 50) * 3
            y = self.center_y + np.sin(i * 0.2 + self.phase) * scale * 0.5
            points1.extend([x, y])
        
        if len(points1) >= 4:
            self.canvas.create_line(
                points1,
                fill=GOLD_PRIMARY,
                width=2,
                smooth=True,
                tags="waves"
            )
        
        # Right wave (beat frequency offset)
        points2 = []
        for i in range(100):
            x = self.center_x + (i - 50) * 3
            y = self.center_y + np.sin(i * 0.2 + self.phase + np.pi * PHI_CONJUGATE) * scale * 0.5
            points2.extend([x, y])
        
        if len(points2) >= 4:
            self.canvas.create_line(
                points2,
                fill=GOLD_SECONDARY,
                width=2,
                smooth=True,
                tags="waves"
            )
        
        # Draw phase indicator
        indicator_angle = self.phase
        indicator_radius = 30
        ix = self.center_x + np.cos(indicator_angle) * indicator_radius
        iy = self.center_y - np.sin(indicator_angle) * indicator_radius
        
        self.canvas.delete("indicator")
        self.canvas.create_oval(
            ix - 5, iy - 5, ix + 5, iy + 5,
            fill=GOLD_PRIMARY,
            outline=GOLD_SECONDARY,
            width=2,
            tags="indicator"
        )


class GoldenBinauralApp:
    """Main application for golden binaural generation"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Divine Golden Binaural Generator")
        self.root.configure(bg=COSMIC_BLUE)
        self.root.geometry("800x700")
        
        # State
        self.is_playing = False
        self.current_stage = 0
        self.total_stages = 8
        
        # Parameters (with golden ratio defaults)
        self.base_freq = tk.DoubleVar(value=432.0)
        self.beat_freq = tk.DoubleVar(value=10.0)
        self.amplitude = tk.DoubleVar(value=0.8)
        self.phase = tk.DoubleVar(value=0.0)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg=COSMIC_BLUE)
        title_frame.pack(pady=10)
        
        title = tk.Label(
            title_frame,
            text="✦ DIVINE GOLDEN RATIO BINAURAL ✦",
            font=("Helvetica", 18, "bold"),
            fg=GOLD_PRIMARY,
            bg=COSMIC_BLUE
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text=f"φ = {PHI:.10f}",
            font=("Helvetica", 10),
            fg=GOLD_SECONDARY,
            bg=COSMIC_BLUE
        )
        subtitle.pack()
        
        # Visualization canvas
        self.spiral_canvas = GoldenSpiralCanvas(self.root)
        
        # Parameters frame
        params_frame = tk.LabelFrame(
            self.root,
            text="Sacred Parameters",
            font=("Helvetica", 12, "bold"),
            fg=GOLD_PRIMARY,
            bg=COSMIC_BLUE
        )
        params_frame.pack(padx=20, pady=10, fill="x")
        
        # Base frequency
        self.create_parameter_slider(
            params_frame,
            "Base Frequency (Hz)",
            self.base_freq,
            100, 1000,
            0
        )
        
        # Beat frequency
        self.create_parameter_slider(
            params_frame,
            "Beat Frequency (Hz)",
            self.beat_freq,
            0.5, 40,
            1
        )
        
        # Amplitude
        self.create_parameter_slider(
            params_frame,
            "Amplitude",
            self.amplitude,
            0, 1,
            2
        )
        
        # Phase
        self.create_parameter_slider(
            params_frame,
            "Phase (radians)",
            self.phase,
            0, np.pi,
            3
        )
        
        # Progress frame
        progress_frame = tk.Frame(self.root, bg=COSMIC_BLUE)
        progress_frame.pack(pady=10, fill="x", padx=20)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Stage: 0 / 8 | Annealing Progress: 0%",
            font=("Helvetica", 11),
            fg=GOLD_SECONDARY,
            bg=COSMIC_BLUE
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
        
        # Control buttons
        controls_frame = tk.Frame(self.root, bg=COSMIC_BLUE)
        controls_frame.pack(pady=15)
        
        style = ttk.Style()
        style.configure(
            "Golden.TButton",
            font=("Helvetica", 11, "bold")
        )
        
        self.generate_btn = tk.Button(
            controls_frame,
            text="✦ Generate Sequence ✦",
            font=("Helvetica", 12, "bold"),
            fg=COSMIC_BLUE,
            bg=GOLD_PRIMARY,
            padx=20,
            pady=10,
            command=self.generate_sequence
        )
        self.generate_btn.pack(side="left", padx=10)
        
        self.anneal_btn = tk.Button(
            controls_frame,
            text="⟳ Start Annealing",
            font=("Helvetica", 12, "bold"),
            fg=COSMIC_BLUE,
            bg=GOLD_SECONDARY,
            padx=20,
            pady=10,
            command=self.start_annealing
        )
        self.anneal_btn.pack(side="left", padx=10)
        
        # Preset buttons
        presets_frame = tk.LabelFrame(
            self.root,
            text="Divine Presets",
            font=("Helvetica", 12, "bold"),
            fg=GOLD_PRIMARY,
            bg=COSMIC_BLUE
        )
        presets_frame.pack(padx=20, pady=10, fill="x")
        
        presets = [
            ("432 Hz Meditation", 432.0, 7.83),  # Schumann resonance
            ("528 Hz Healing", 528.0, 8.0),
            ("639 Hz Heart", 639.0, 10.5),
            ("Theta Dreams", 432.0, 5.5),
            ("Deep Delta", 432.0, 2.5),
        ]
        
        for name, base, beat in presets:
            btn = tk.Button(
                presets_frame,
                text=name,
                font=("Helvetica", 9),
                fg=COSMIC_BLUE,
                bg=GOLD_SECONDARY,
                command=lambda b=base, bt=beat: self.apply_preset(b, bt)
            )
            btn.pack(side="left", padx=5, pady=5)
        
        # Status bar
        self.status = tk.Label(
            self.root,
            text="Ready • All parameters in golden ratio coherence",
            font=("Helvetica", 9),
            fg=GOLD_SECONDARY,
            bg=COSMIC_BLUE
        )
        self.status.pack(side="bottom", pady=5)
    
    def create_parameter_slider(self, parent, label, variable, min_val, max_val, row):
        """Create a labeled slider for a parameter"""
        frame = tk.Frame(parent, bg=COSMIC_BLUE)
        frame.pack(fill="x", padx=10, pady=5)
        
        lbl = tk.Label(
            frame,
            text=label,
            font=("Helvetica", 10),
            fg=GOLD_SECONDARY,
            bg=COSMIC_BLUE,
            width=20,
            anchor="w"
        )
        lbl.pack(side="left")
        
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient="horizontal",
            variable=variable,
            resolution=0.01 if max_val <= 10 else 1,
            length=300,
            fg=GOLD_PRIMARY,
            bg=COSMIC_BLUE,
            troughcolor=DIVINE_PURPLE,
            highlightthickness=0
        )
        slider.pack(side="left", padx=10)
        
        value_lbl = tk.Label(
            frame,
            textvariable=variable,
            font=("Helvetica", 10),
            fg=GOLD_PRIMARY,
            bg=COSMIC_BLUE,
            width=10
        )
        value_lbl.pack(side="left")
    
    def apply_preset(self, base: float, beat: float):
        """Apply a preset configuration"""
        self.base_freq.set(base)
        self.beat_freq.set(beat)
        # Set amplitude and phase in golden ratio
        self.amplitude.set(PHI_CONJUGATE)
        self.phase.set(np.pi * PHI_CONJUGATE)
        self.status.config(text=f"Applied preset: {base}Hz base, {beat}Hz beat")
        self.update_visualization()
    
    def update_visualization(self):
        """Update the spiral visualization"""
        self.spiral_canvas.update_visualization(
            self.phase.get(),
            self.amplitude.get(),
            self.beat_freq.get()
        )
    
    def generate_sequence(self):
        """Generate and save the binaural sequence"""
        self.status.config(text="Generating golden binaural sequence...")
        self.root.update()
        
        # Import generator
        from golden_core import PhaseAnnihilator, save_wav
        
        # Generate
        annihilator = PhaseAnnihilator(sample_rate=96000)
        left, right = annihilator.generate_annealing_sequence(
            num_stages=self.total_stages,
            base_frequency=self.base_freq.get()
        )
        
        # Save
        filename = f"golden_binaural_{int(self.base_freq.get())}Hz.wav"
        save_wav(filename, left, right, 96000)
        
        self.status.config(text=f"Saved: {filename}")
    
    def start_annealing(self):
        """Start the visual annealing demonstration"""
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.anneal_btn.config(text="⏹ Stop")
            self.animate_annealing()
        else:
            self.anneal_btn.config(text="⟳ Start Annealing")
    
    def animate_annealing(self):
        """Animate the annealing process"""
        if not self.is_playing:
            return
        
        # Update phase using golden spiral
        current_phase = self.phase.get()
        new_phase = current_phase + 0.1 * PHI_CONJUGATE
        if new_phase > np.pi:
            new_phase = 0
            self.current_stage += 1
            if self.current_stage > self.total_stages:
                self.current_stage = 0
                self.is_playing = False
                self.anneal_btn.config(text="⟳ Start Annealing")
                self.status.config(text="Annealing complete → Silence achieved")
                return
        
        self.phase.set(new_phase)
        
        # Update amplitude (decrease toward silence)
        progress = self.current_stage / self.total_stages
        new_amp = (1 - progress * 0.9) * PHI_CONJUGATE
        self.amplitude.set(new_amp)
        
        # Update beat frequency
        new_beat = self.beat_freq.get() * (PHI_CONJUGATE ** (progress * 0.5))
        if new_beat < 0.5:
            new_beat = 0.5
        self.beat_freq.set(new_beat)
        
        # Update progress display
        self.progress_label.config(
            text=f"Stage: {self.current_stage} / {self.total_stages} | "
                 f"Annealing Progress: {int(progress * 100)}%"
        )
        self.progress_bar['value'] = progress * 100
        
        # Update visualization
        self.update_visualization()
        
        # Continue animation
        self.root.after(50, self.animate_annealing)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Launch the application"""
    print("\n" + "★" * 50)
    print("  DIVINE GOLDEN BINAURAL INTERFACE")
    print("★" * 50 + "\n")
    
    app = GoldenBinauralApp()
    app.run()


if __name__ == "__main__":
    main()
