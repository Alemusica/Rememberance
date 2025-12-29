"""
MolecularTab - Play molecules with bond angles as phases
═════════════════════════════════════════════════════════

Each molecule has a unique structure with bond angles.
These angles become phase relationships in the sound.

Features:
- Molecule presets: H₂O, CO₂, CH₄, NH₃, O₃, H₂S, SO₂, NO₂
- Real spectral lines from atomic elements
- Phase from bond angles (104.5° for water, 180° for CO₂)
- Stereo position from atomic mass
- Molecular structure visualization
- Output modes: molecular, binaural

Based on molecular geometry and VSEPR theory.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from golden_studio import AudioEngine

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI_CONJUGATE = 0.618033988749895  # 1/φ = φ - 1

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from molecular_sound import MolecularSounder
    HAS_MOLECULAR = True
except ImportError:
    HAS_MOLECULAR = False
    MolecularSounder = None


# ═══════════════════════════════════════════════════════════════════════════════
# MOLECULAR TAB
# ═══════════════════════════════════════════════════════════════════════════════

class MolecularTab:
    """Play molecules with bond angles as phases"""
    
    def __init__(self, parent, audio_engine: 'AudioEngine'):
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
    
    def _get_spectral_data(self, mol, use_spectral: bool, mode: str):
        """
        Extract spectral data from molecule.
        
        Returns (frequencies, amplitudes, phases, positions)
        """
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
        
        return all_frequencies, all_amplitudes, all_phases, all_positions
    
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
            
            frequencies, amplitudes, phases, positions = self._get_spectral_data(mol, use_spectral, mode)
            
            # Start continuous streaming with REAL spectral data
            self.audio.start_spectral(frequencies, amplitudes, phases, positions,
                                      master_amplitude=0.7)
            
            self.play_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_var.set(f"🔊 {mol.name}: {len(frequencies)} spectral lines")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def _stop(self):
        """Stop playback"""
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
            # Same precise logic as _play() - refactored to _get_spectral_data()
            # ═══════════════════════════════════════════════════════════════════
            
            frequencies, amplitudes, phases, positions = self._get_spectral_data(mol, use_spectral, mode)
            self.audio.set_spectral_params(frequencies, amplitudes, phases, positions)
        except:
            pass
