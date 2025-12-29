#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GOLDEN SOUND STUDIO - UNIFIED INTERFACE                   ║
║                                                                              ║
║   Refactored for Raspberry Pi 5 deployment with modular architecture        ║
║                                                                              ║
║   🎵 TAB 1: Binaural Beats - Phase angle control, sacred geometry           ║
║   ⚛️ TAB 2: Spectral Sound - Play atomic elements                           ║
║   🧪 TAB 3: Molecular Sound - Play molecules with bond angles as phases     ║
║   🌳 TAB 4: Harmonic Tree - Fibonacci harmonics                             ║
║   🪵 TAB 5: Vibroacoustic - Soundboard panning                              ║
║   🧠 TAB 6: EMDR - Bilateral audio stimulation                              ║
║   🎼 TAB 7: Session Builder - Visual program designer                       ║
║   🔬 TAB 8: Plate Lab - Modal analysis                                      ║
║   🧬 TAB 9: Plate Designer - Evolutionary optimization                      ║
║                                                                              ║
║   "The universe is made of vibrations - let's hear them"                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

REFACTORING NOTE:
This file has been refactored for Pi5 deployment. The monolithic 4000+ line file
has been split into modular components:

- studio/audio_manager.py - AudioEngine for real-time audio
- studio/app.py - GoldenSoundStudio main application class  
- ui/binaural_tab.py, ui/spectral_tab.py, etc. - Individual tab modules

The original golden_studio.py contained duplicate definitions of classes that were
already extracted to ui/. This refactored version imports from the proper modules.
"""

# Import refactored studio modules
from studio.audio_manager import AudioEngine

# Import tab classes that are still defined in this file (need extraction)
from golden_studio_old import (
    BinauralTab,
    SpectralTab, 
    MolecularTab,
    HarmonicTreeTab,
    VibroacousticTab
)

# Import EMDR bilateral stimulation module
try:
    from ui.emdr_tab import EMDRTab
    HAS_EMDR = True
except ImportError:
    HAS_EMDR = False
    print("⚠️ ui/emdr_tab.py not found")

# Import Session Builder module
try:
    from ui.session_builder_tab import SessionBuilderTab
    HAS_SESSION_BUILDER = True
except ImportError:
    HAS_SESSION_BUILDER = False
    print("⚠️ ui/session_builder_tab.py not found")

# Import Plate Lab module (modal analysis)
try:
    from ui.plate_lab_tab import PlateLabTab
    HAS_PLATE_LAB = True
except ImportError:
    HAS_PLATE_LAB = False
    print("⚠️ ui/plate_lab_tab.py not found")

# Import Plate Designer module (evolutionary optimization)
try:
    from ui.plate_designer_tab import PlateDesignerTab
    HAS_PLATE_DESIGNER = True
except ImportError:
    HAS_PLATE_DESIGNER = False
    print("⚠️ ui/plate_designer_tab.py not found")

# Import the main application class (temporarily from studio.app)
import tkinter as tk
from tkinter import ttk


class GoldenSoundStudio:
    """Main application with tabbed interface - REFACTORED VERSION"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌀 GOLDEN SOUND STUDIO")
        self.root.geometry("950x700")
        
        # Shared audio engine (imported from studio module)
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
        
        # Create tabs (from old file for now)
        self.binaural_tab = BinauralTab(self.notebook, self.audio)
        self.spectral_tab = SpectralTab(self.notebook, self.audio)
        self.molecular_tab = MolecularTab(self.notebook, self.audio)
        self.harmonic_tree_tab = HarmonicTreeTab(self.notebook, self.audio)
        self.vibroacoustic_tab = VibroacousticTab(self.notebook, self.audio)
        
        # EMDR Tab (bilateral audio stimulation for hemispheric integration)
        if HAS_EMDR:
            self.emdr_tab = EMDRTab(self.notebook, self.audio)
        
        # Session Builder Tab (visual program designer)
        if HAS_SESSION_BUILDER:
            self.session_builder_tab = SessionBuilderTab(self.notebook)
        
        # Plate Lab Tab (modal analysis for vibroacoustic)
        if HAS_PLATE_LAB:
            self.plate_lab_tab = PlateLabTab(self.notebook, self.audio)
        
        # Plate Designer Tab (evolutionary optimization)
        if HAS_PLATE_DESIGNER:
            self.plate_designer_tab = PlateDesignerTab(self.notebook)
        
        self.notebook.add(self.binaural_tab.frame, text="🎵 Binaural Beats")
        self.notebook.add(self.spectral_tab.frame, text="⚛️ Spectral Sound")
        self.notebook.add(self.molecular_tab.frame, text="🧪 Molecular Sound")
        self.notebook.add(self.harmonic_tree_tab.frame, text="🌳 Harmonic Tree")
        self.notebook.add(self.vibroacoustic_tab.frame, text="🪵 Vibroacoustic")
        
        # Add EMDR tab if available
        if HAS_EMDR:
            self.notebook.add(self.emdr_tab.frame, text="🧠 EMDR")
        
        # Add Session Builder tab if available
        if HAS_SESSION_BUILDER:
            self.notebook.add(self.session_builder_tab, text="🎼 Session Builder")
        
        # Add Plate Lab tab if available
        if HAS_PLATE_LAB:
            self.notebook.add(self.plate_lab_tab.frame, text="🔬 Plate Lab")
        
        # Add Plate Designer tab if available
        if HAS_PLATE_DESIGNER:
            self.notebook.add(self.plate_designer_tab, text="🧬 Plate Designer")
        
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
║   🎼 Tab 7: Session Builder - Visual program designer with pie chart        ║
║                                                                              ║
║   Based on natural phyllotaxis patterns:                                     ║
║   • Harmonics at Fibonacci ratios (2f, 3f, 5f, 8f, 13f)                     ║
║   • Phases rotate by Golden Angle (137.5°) like sunflower seeds             ║
║   • Amplitudes decay by φ⁻ⁿ (natural growth pattern)                        ║
║   • Session Builder: Pie chart + flowchart timeline for journey design     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        self.root.mainloop()


if __name__ == "__main__":
    app = GoldenSoundStudio()
    app.run()
