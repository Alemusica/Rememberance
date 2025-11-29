#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    3D GOLDEN WAVE OSCILLOSCOPE                               ║
║                                                                              ║
║   Real-time 3D visualization of PHI-based binaural waves                     ║
║   Inspired by Lyapunov exponents and strange attractors                      ║
║                                                                              ║
║   X = Left channel amplitude                                                 ║
║   Y = Right channel amplitude                                                ║
║   Z = Phase difference (or time)                                             ║
║                                                                              ║
║   Creates Lissajous-like 3D figures that reveal the golden structure         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import threading
import time
from collections import deque

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Installing matplotlib...")

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# ══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
PHI_CONJUGATE = 0.618033988749895
GOLDEN_ANGLE_RAD = 2 * np.pi / (PHI * PHI)  # ≈ 2.399 rad ≈ 137.5°

# Colors based on golden ratio
GOLDEN_COLORS = [
    '#FFD700',  # Gold
    '#FF6B6B',  # Coral (Left)
    '#4ECDC4',  # Cyan (Right)
    '#FF00FF',  # Magenta (Phase)
    '#00FF88',  # Green (Sum)
]

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN WAVE GENERATOR (same as GUI)
# ══════════════════════════════════════════════════════════════════════════════

class GoldenWaveGenerator:
    """Generate PHI-based waveforms for visualization"""
    
    def __init__(self):
        self.freq_left = 432.0
        self.freq_right = 440.0
        self.phase_angle_deg = 137.5
        self.amplitude = 0.8
        self.waveform_mode = "golden_reversed"
        
        # Phase accumulators
        self.phase_left = 0.0
        self.phase_right = 0.0
        
        self.lock = threading.Lock()
    
    def set_params(self, freq_l, freq_r, phase_deg, amp, mode):
        with self.lock:
            self.freq_left = freq_l
            self.freq_right = freq_r
            self.phase_angle_deg = phase_deg
            self.amplitude = amp
            self.waveform_mode = mode
    
    def golden_wave(self, phase, reversed=True):
        """Generate PHI-based waveform"""
        theta = phase % (2 * np.pi)
        if reversed:
            theta = 2 * np.pi - theta
        
        t = theta / (2 * np.pi)
        rise_portion = PHI_CONJUGATE
        
        wave = np.zeros_like(t)
        rising = t < rise_portion
        t_rise = np.where(rising, t / rise_portion, 0)
        wave = np.where(rising, self._golden_ease(t_rise), 0)
        
        falling = ~rising
        t_fall = np.where(falling, (t - rise_portion) / (1 - rise_portion), 0)
        wave = np.where(falling, 1 - self._golden_ease(t_fall), wave)
        
        wave = wave * 2 - 1
        
        # Add golden harmonics
        harmonic2 = np.sin(2 * theta) * (PHI_CONJUGATE ** 2)
        harmonic3 = np.sin(3 * theta) * (PHI_CONJUGATE ** 3)
        
        golden_blend = (
            wave * PHI_CONJUGATE + 
            harmonic2 * (1 - PHI_CONJUGATE) * PHI_CONJUGATE +
            harmonic3 * (1 - PHI_CONJUGATE) * (1 - PHI_CONJUGATE)
        )
        
        max_val = np.max(np.abs(golden_blend))
        if max_val > 0:
            golden_blend = golden_blend / max_val
        
        return golden_blend
    
    def _golden_ease(self, t):
        t = np.clip(t, 0, 1)
        theta = t * np.pi * PHI
        golden_ease = (1.0 - np.cos(theta * PHI_CONJUGATE)) / 2.0
        x = (t - 0.5) * 4.0
        golden_sigmoid = 1.0 / (1.0 + np.exp(-x * PHI))
        return np.clip(golden_ease * PHI_CONJUGATE + golden_sigmoid * (1 - PHI_CONJUGATE), 0, 1)
    
    def generate_samples(self, num_samples):
        """Generate L, R, and phase data"""
        with self.lock:
            freq_l = self.freq_left
            freq_r = self.freq_right
            phase_offset = self.phase_angle_deg * np.pi / 180
            amp = self.amplitude
            mode = self.waveform_mode
        
        # Phase increments
        reversed_mode = "reversed" in mode
        sign = -1 if reversed_mode else 1
        
        delta_l = sign * 2 * np.pi * freq_l / SAMPLE_RATE
        delta_r = sign * 2 * np.pi * freq_r / SAMPLE_RATE
        
        phases_l = self.phase_left + np.cumsum(np.full(num_samples, delta_l))
        phases_r = self.phase_right + phase_offset + np.cumsum(np.full(num_samples, delta_r))
        
        self.phase_left = phases_l[-1] % (2 * np.pi)
        self.phase_right = (phases_r[-1] - phase_offset) % (2 * np.pi)
        
        if mode == "sine":
            left = amp * np.sin(phases_l)
            right = amp * np.sin(phases_r)
        else:
            left = amp * self.golden_wave(phases_l, reversed=reversed_mode)
            right = amp * self.golden_wave(phases_r, reversed=reversed_mode)
        
        # Phase difference over time (for Z axis)
        phase_diff = (phases_r - phases_l) % (2 * np.pi)
        
        return left, right, phase_diff, phases_l


# ══════════════════════════════════════════════════════════════════════════════
# 3D OSCILLOSCOPE WITH LYAPUNOV-STYLE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class Oscilloscope3D:
    """Real-time 3D oscilloscope with golden ratio aesthetics"""
    
    def __init__(self):
        self.generator = GoldenWaveGenerator()
        
        # Data buffers - longer trail for 3D effect
        self.trail_length = 4000  # Points in the 3D trail
        self.x_data = deque(maxlen=self.trail_length)
        self.y_data = deque(maxlen=self.trail_length)
        self.z_data = deque(maxlen=self.trail_length)
        self.colors = deque(maxlen=self.trail_length)
        
        # Time accumulator for Z-axis
        self.time_acc = 0.0
        
        # Visualization mode
        self.viz_mode = "lissajous_3d"  # "lissajous_3d", "attractor", "helix", "torus"
        
        # Audio device index (None = default)
        self.device_index = None
        
        # Audio
        self.pyaudio = None
        self.stream = None
        self.playing = False
        
        # Animation
        self.fig = None
        self.ax = None
        self.scatter = None
        self.line = None
        
        # Rotation
        self.auto_rotate = True
        self.rotation_angle = 0
        
        self.running = True
    
    def start_audio(self):
        """Start audio playback"""
        if not HAS_PYAUDIO:
            print("PyAudio not available")
            return
        
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            def callback(in_data, frame_count, time_info, status):
                left, right, _, _ = self.generator.generate_samples(frame_count)
                stereo = np.empty(frame_count * 2, dtype=np.float32)
                stereo[0::2] = left.astype(np.float32)
                stereo[1::2] = right.astype(np.float32)
                return (stereo.tobytes(), pyaudio.paContinue)
            
            # Open stream with optional device selection
            stream_kwargs = {
                'format': pyaudio.paFloat32,
                'channels': 2,
                'rate': SAMPLE_RATE,
                'output': True,
                'frames_per_buffer': BUFFER_SIZE,
                'stream_callback': callback
            }
            
            # Add device index if specified
            if self.device_index is not None:
                stream_kwargs['output_device_index'] = self.device_index
            
            self.stream = self.pyaudio.open(**stream_kwargs)
            self.playing = True
            print("Audio started")
        except Exception as e:
            print(f"Audio error: {e}")
    
    def stop_audio(self):
        """Stop audio"""
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
    
    def update_data(self):
        """Generate new data points for visualization"""
        num_samples = 100  # Samples per update
        
        left, right, phase_diff, phases = self.generator.generate_samples(num_samples)
        
        # Different 3D mappings based on mode
        if self.viz_mode == "lissajous_3d":
            # Classic Lissajous with phase as Z
            x = left
            y = right
            z = np.sin(phase_diff)  # Phase mapped to [-1, 1]
            
        elif self.viz_mode == "attractor":
            # Strange attractor style - uses derivatives
            x = left
            y = right
            # Z is like a "Lyapunov" dimension - rate of change
            z = np.gradient(left) * PHI + np.gradient(right) * PHI_CONJUGATE
            z = np.clip(z * 10, -1, 1)
            
        elif self.viz_mode == "helix":
            # Golden helix
            t = phases / (2 * np.pi)
            spiral_r = PHI_CONJUGATE + left * 0.3
            x = spiral_r * np.cos(phases * PHI)
            y = spiral_r * np.sin(phases * PHI)
            z = (t % 1) * 2 - 1 + right * 0.2
            
        elif self.viz_mode == "torus":
            # Torus (donut) with golden parameters
            R = 1.0  # Major radius
            r = PHI_CONJUGATE  # Minor radius (golden!)
            u = phases  # Around the tube
            v = phases * PHI  # Around the torus
            
            x = (R + r * np.cos(v) + left * 0.1) * np.cos(u)
            y = (R + r * np.cos(v) + left * 0.1) * np.sin(u)
            z = r * np.sin(v) + right * 0.1
        
        else:  # "sphere"
            # Project onto sphere with golden spiral
            theta = phases
            phi_angle = GOLDEN_ANGLE_RAD * np.arange(num_samples) / 10
            r = 0.8 + left * 0.2
            
            x = r * np.sin(theta) * np.cos(phi_angle)
            y = r * np.sin(theta) * np.sin(phi_angle)
            z = r * np.cos(theta) + right * 0.2
        
        # Add to buffers
        for i in range(len(x)):
            self.x_data.append(x[i])
            self.y_data.append(y[i])
            self.z_data.append(z[i])
            # Color based on position in trail (older = darker)
            self.colors.append(len(self.x_data) / self.trail_length)
    
    def init_plot(self):
        """Initialize the 3D plot"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#0a0a15')
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#0a0a15')
        
        # Golden ratio proportions for axes
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5)
        
        self.ax.set_xlabel('Left (L)', color='#ff6b6b', fontsize=12)
        self.ax.set_ylabel('Right (R)', color='#4ecdc4', fontsize=12)
        self.ax.set_zlabel('Phase/Z', color='#ffd700', fontsize=12)
        
        # Style the axes
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#333')
        self.ax.yaxis.pane.set_edgecolor('#333')
        self.ax.zaxis.pane.set_edgecolor('#333')
        self.ax.grid(True, alpha=0.2, color='#ffd700')
        
        # Title
        self.ax.set_title('🌀 GOLDEN WAVE 3D OSCILLOSCOPE 🌀\n'
                         f'φ = {PHI:.6f} | Mode: {self.viz_mode}',
                         color='#ffd700', fontsize=14, fontweight='bold')
        
        # Initialize empty scatter
        self.scatter = self.ax.scatter([], [], [], c=[], cmap='plasma', s=2, alpha=0.8)
        
        # Initialize empty line for trail
        self.line, = self.ax.plot([], [], [], color='#ffd700', alpha=0.3, linewidth=0.5)
        
        # Add parameter text
        self.param_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes,
                                         color='#00ff88', fontsize=10, verticalalignment='top',
                                         fontfamily='monospace')
        
        return self.scatter, self.line
    
    def animate(self, frame):
        """Animation update function"""
        # Generate new data
        self.update_data()
        
        if len(self.x_data) < 10:
            return self.scatter, self.line
        
        # Convert to arrays
        x = np.array(self.x_data)
        y = np.array(self.y_data)
        z = np.array(self.z_data)
        c = np.array(self.colors)
        
        # Update scatter plot
        self.scatter._offsets3d = (x, y, z)
        self.scatter.set_array(c)
        
        # Update line trail
        self.line.set_data(x, y)
        self.line.set_3d_properties(z)
        
        # Auto-rotate
        if self.auto_rotate:
            self.rotation_angle += PHI_CONJUGATE  # Golden rotation speed!
            self.ax.view_init(elev=20, azim=self.rotation_angle)
        
        # Update parameter text
        with self.generator.lock:
            params = (
                f"L: {self.generator.freq_left:.1f} Hz\n"
                f"R: {self.generator.freq_right:.1f} Hz\n"
                f"Beat: {abs(self.generator.freq_right - self.generator.freq_left):.2f} Hz\n"
                f"Phase: {self.generator.phase_angle_deg:.2f}°\n"
                f"Mode: {self.generator.waveform_mode}\n"
                f"Viz: {self.viz_mode}"
            )
        self.param_text.set_text(params)
        
        return self.scatter, self.line
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'q':
            self.running = False
            plt.close()
        elif event.key == 'r':
            self.auto_rotate = not self.auto_rotate
            print(f"Auto-rotate: {self.auto_rotate}")
        elif event.key == ' ':
            if self.playing:
                self.stop_audio()
                print("Audio stopped")
            else:
                self.start_audio()
        elif event.key == '1':
            self.viz_mode = "lissajous_3d"
            self.ax.set_title(f'🌀 GOLDEN WAVE 3D OSCILLOSCOPE 🌀\nMode: {self.viz_mode}',
                             color='#ffd700', fontsize=14)
        elif event.key == '2':
            self.viz_mode = "attractor"
            self.ax.set_title(f'🌀 GOLDEN WAVE 3D OSCILLOSCOPE 🌀\nMode: {self.viz_mode}',
                             color='#ffd700', fontsize=14)
        elif event.key == '3':
            self.viz_mode = "helix"
            self.ax.set_title(f'🌀 GOLDEN WAVE 3D OSCILLOSCOPE 🌀\nMode: {self.viz_mode}',
                             color='#ffd700', fontsize=14)
        elif event.key == '4':
            self.viz_mode = "torus"
            self.ax.set_title(f'🌀 GOLDEN WAVE 3D OSCILLOSCOPE 🌀\nMode: {self.viz_mode}',
                             color='#ffd700', fontsize=14)
        elif event.key == 'up':
            self.generator.freq_left *= PHI_CONJUGATE ** 0.1
            self.generator.freq_right *= PHI_CONJUGATE ** 0.1
        elif event.key == 'down':
            self.generator.freq_left /= PHI_CONJUGATE ** 0.1
            self.generator.freq_right /= PHI_CONJUGATE ** 0.1
        elif event.key == 'left':
            self.generator.phase_angle_deg = (self.generator.phase_angle_deg - 5) % 360
        elif event.key == 'right':
            self.generator.phase_angle_deg = (self.generator.phase_angle_deg + 5) % 360
        elif event.key == 'g':
            # Set to golden angle
            self.generator.phase_angle_deg = 137.5077640500378546
        elif event.key == 'f':
            # Set to fine structure angle
            self.generator.phase_angle_deg = 137.035999084
        elif event.key == 'c':
            # Set to cancellation (180°)
            self.generator.phase_angle_deg = 180.0
        elif event.key == 'w':
            # Toggle waveform
            modes = ["golden_reversed", "golden", "sine"]
            idx = modes.index(self.generator.waveform_mode)
            self.generator.waveform_mode = modes[(idx + 1) % len(modes)]
            print(f"Waveform: {self.generator.waveform_mode}")
    
    def run(self):
        """Run the oscilloscope"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    3D GOLDEN WAVE OSCILLOSCOPE                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONTROLS:                                                                   ║
║    SPACE     - Toggle audio on/off                                          ║
║    1,2,3,4   - Visualization modes (Lissajous, Attractor, Helix, Torus)    ║
║    R         - Toggle auto-rotation                                          ║
║    W         - Toggle waveform (golden_reversed, golden, sine)              ║
║    ←/→       - Adjust phase angle ±5°                                       ║
║    ↑/↓       - Adjust frequency (golden ratio steps)                        ║
║    G         - Set Golden Angle (137.5°)                                    ║
║    F         - Set Fine Structure Angle (137.036°)                          ║
║    C         - Set Cancellation (180°)                                       ║
║    Q         - Quit                                                          ║
║                                                                              ║
║  AXES:                                                                       ║
║    X = Left channel amplitude                                                ║
║    Y = Right channel amplitude                                               ║
║    Z = Phase relationship                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        self.init_plot()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Start audio
        self.start_audio()
        
        # Animation
        ani = FuncAnimation(self.fig, self.animate, interval=30, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
        # Cleanup
        self.stop_audio()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(freq_left=432.0, freq_right=440.0, phase_angle=137.5, 
         amplitude=0.8, waveform="golden_reversed", device_index=None):
    """
    Run the oscilloscope with optional initial parameters.
    
    Args:
        freq_left: Left channel frequency (Hz)
        freq_right: Right channel frequency (Hz)
        phase_angle: Phase angle in degrees
        amplitude: Volume 0-1
        waveform: 'golden_reversed', 'golden', or 'sine'
        device_index: PyAudio device index (None for default)
    """
    if not HAS_MATPLOTLIB:
        print("Please install matplotlib: pip install matplotlib")
        return
    
    scope = Oscilloscope3D()
    
    # Apply initial parameters
    scope.generator.freq_left = freq_left
    scope.generator.freq_right = freq_right
    scope.generator.phase_angle_deg = phase_angle
    scope.generator.amplitude = amplitude
    scope.generator.waveform_mode = waveform
    
    # Store device index for audio output
    if device_index is not None:
        scope.device_index = device_index
    
    scope.run()


def run_from_gui(params: dict):
    """
    Entry point for launching from the main GUI.
    Params dict should contain: freq_left, freq_right, phase_angle, amplitude, waveform, device_index
    """
    main(
        freq_left=params.get('freq_left', 432.0),
        freq_right=params.get('freq_right', 440.0),
        phase_angle=params.get('phase_angle', 137.5),
        amplitude=params.get('amplitude', 0.8),
        waveform=params.get('waveform', 'golden_reversed'),
        device_index=params.get('device_index', None)
    )


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Golden Wave Oscilloscope')
    parser.add_argument('--freq-left', '-fl', type=float, default=432.0, help='Left frequency (Hz)')
    parser.add_argument('--freq-right', '-fr', type=float, default=440.0, help='Right frequency (Hz)')
    parser.add_argument('--phase', '-p', type=float, default=137.5, help='Phase angle (degrees)')
    parser.add_argument('--amplitude', '-a', type=float, default=0.8, help='Amplitude (0-1)')
    parser.add_argument('--waveform', '-w', type=str, default='golden_reversed', 
                       choices=['golden_reversed', 'golden', 'sine'], help='Waveform type')
    parser.add_argument('--device', '-d', type=int, default=None, help='Audio device index')
    
    args = parser.parse_args()
    
    main(
        freq_left=args.freq_left,
        freq_right=args.freq_right,
        phase_angle=args.phase,
        amplitude=args.amplitude,
        waveform=args.waveform,
        device_index=args.device
    )
