#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MOLECULAR GEOMETRY SOUND                                  ║
║                                                                              ║
║   Trasforma la geometria molecolare in suono:                                ║
║   - Angoli di legame → Fasi audio                                           ║
║   - Lunghezze di legame → Frequenze                                         ║
║   - Atomi componenti → Timbro (linee spettrali)                             ║
║   - Simmetria molecolare → Pattern stereo                                    ║
║                                                                              ║
║   "L'acqua canta con l'angolo di 104.5°"                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

# Import spectral data
from spectral_sound import (
    SpectralSounder, PhaseMode, 
    HYDROGEN_BALMER, OXYGEN_ATMOSPHERIC,
    PHI, PHI_CONJUGATE, SAMPLE_RATE, C
)

# ══════════════════════════════════════════════════════════════════════════════
# COSTANTI MOLECOLARI
# ══════════════════════════════════════════════════════════════════════════════

# Angoli di legame in gradi (dati reali)
BOND_ANGLES = {
    # Molecole triatomiche
    "H2O": 104.5,        # Acqua - angolo H-O-H
    "H2S": 92.1,         # Acido solfidrico
    "CO2": 180.0,        # Anidride carbonica (lineare)
    "SO2": 119.0,        # Anidride solforosa
    "NO2": 134.0,        # Biossido di azoto
    "O3": 116.8,         # Ozono
    "H2Se": 91.0,        # Seleniuro di idrogeno
    
    # Molecole tetraedriche (angolo ideale 109.5°)
    "CH4": 109.5,        # Metano
    "NH3": 107.3,        # Ammoniaca (piramidale)
    "PH3": 93.5,         # Fosfina
    "CCl4": 109.5,       # Tetracloruro di carbonio
    
    # Molecole planari
    "BF3": 120.0,        # Trifluoruro di boro
    "BCl3": 120.0,       # Tricloruro di boro
    "C2H4": 121.3,       # Etilene (H-C-H)
    
    # Altre geometrie
    "SF6": 90.0,         # Esafluoruro di zolfo (ottaedrico)
    "PCl5": 90.0,        # Pentacloruro di fosforo (eq-ax)
    "XeF4": 90.0,        # Tetrafluoruro di xeno (planare quadrato)
}

# Lunghezze di legame in Angstrom (Å) - usate per frequenze
BOND_LENGTHS = {
    "O-H": 0.96,         # In H2O
    "H-H": 0.74,         # In H2
    "O=O": 1.21,         # In O2
    "C-H": 1.09,         # In CH4
    "C=O": 1.16,         # In CO2
    "C-C": 1.54,         # Singolo
    "C=C": 1.34,         # Doppio
    "C≡C": 1.20,         # Triplo
    "N-H": 1.01,         # In NH3
    "N≡N": 1.10,         # In N2
    "S-H": 1.34,         # In H2S
    "C-O": 1.43,         # Singolo
    "C-N": 1.47,         # Singolo
    "P-H": 1.42,         # In PH3
}

# Elettronegatività (scala Pauling) - per intensità/ampiezza
ELECTRONEGATIVITY = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Br": 2.96,
    "I": 2.66,
    "B": 2.04,
    "Si": 1.90,
    "Se": 2.55,
    "Xe": 2.60,
}

# Masse atomiche (u) - per frequenze vibrazionali
ATOMIC_MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.904,
    "B": 10.811,
    "Si": 28.086,
    "Se": 78.971,
    "Xe": 131.293,
}


# ══════════════════════════════════════════════════════════════════════════════
# STRUTTURE DATI MOLECOLARI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Atom:
    """Rappresenta un atomo in una molecola"""
    symbol: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Coordinate 3D
    
    @property
    def mass(self) -> float:
        return ATOMIC_MASS.get(self.symbol, 1.0)
    
    @property
    def electronegativity(self) -> float:
        return ELECTRONEGATIVITY.get(self.symbol, 2.5)


@dataclass
class Bond:
    """Rappresenta un legame tra due atomi"""
    atom1_idx: int
    atom2_idx: int
    bond_type: str = "single"  # single, double, triple
    length: float = 1.0  # Angstrom
    
    @property
    def order(self) -> int:
        return {"single": 1, "double": 2, "triple": 3}.get(self.bond_type, 1)


@dataclass 
class Molecule:
    """Rappresenta una molecola completa"""
    name: str
    formula: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    bond_angles: List[float] = field(default_factory=list)  # In gradi
    dihedral_angles: List[float] = field(default_factory=list)  # Angoli torsionali
    symmetry: str = "C1"  # Gruppo di simmetria
    dipole_moment: float = 0.0  # Debye
    
    @property
    def num_atoms(self) -> int:
        return len(self.atoms)
    
    @property
    def total_mass(self) -> float:
        return sum(atom.mass for atom in self.atoms)
    
    @property
    def center_of_mass(self) -> Tuple[float, float, float]:
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        total_mass = self.total_mass
        cx = sum(a.position[0] * a.mass for a in self.atoms) / total_mass
        cy = sum(a.position[1] * a.mass for a in self.atoms) / total_mass
        cz = sum(a.position[2] * a.mass for a in self.atoms) / total_mass
        return (cx, cy, cz)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE MOLECOLE PREDEFINITE
# ══════════════════════════════════════════════════════════════════════════════

def create_water() -> Molecule:
    """Crea molecola di acqua H2O con geometria reale"""
    # Angolo H-O-H = 104.5°
    # Lunghezza O-H = 0.96 Å
    angle_rad = np.radians(104.5 / 2)  # Mezzo angolo
    bond_length = 0.96
    
    atoms = [
        Atom("O", (0.0, 0.0, 0.0)),
        Atom("H", (bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
        Atom("H", (-bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
    ]
    
    bonds = [
        Bond(0, 1, "single", 0.96),
        Bond(0, 2, "single", 0.96),
    ]
    
    return Molecule(
        name="Water",
        formula="H2O",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[104.5],
        symmetry="C2v",
        dipole_moment=1.85
    )


def create_carbon_dioxide() -> Molecule:
    """Crea molecola di CO2 (lineare)"""
    bond_length = 1.16
    
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("O", (-bond_length, 0.0, 0.0)),
        Atom("O", (bond_length, 0.0, 0.0)),
    ]
    
    bonds = [
        Bond(0, 1, "double", 1.16),
        Bond(0, 2, "double", 1.16),
    ]
    
    return Molecule(
        name="Carbon Dioxide",
        formula="CO2",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[180.0],
        symmetry="D∞h",
        dipole_moment=0.0  # Apolare!
    )


def create_methane() -> Molecule:
    """Crea molecola di metano CH4 (tetraedrica)"""
    # Angolo tetraedrico = 109.5°
    bond_length = 1.09
    
    # Coordinate tetraedriche
    t = np.radians(109.5)
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("H", (bond_length, 0.0, 0.0)),
        Atom("H", (bond_length * np.cos(t), bond_length * np.sin(t), 0.0)),
        Atom("H", (bond_length * np.cos(t), bond_length * np.sin(t) * np.cos(2*np.pi/3), 
                   bond_length * np.sin(t) * np.sin(2*np.pi/3))),
        Atom("H", (bond_length * np.cos(t), bond_length * np.sin(t) * np.cos(4*np.pi/3), 
                   bond_length * np.sin(t) * np.sin(4*np.pi/3))),
    ]
    
    bonds = [Bond(0, i, "single", 1.09) for i in range(1, 5)]
    
    return Molecule(
        name="Methane",
        formula="CH4",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[109.5, 109.5, 109.5, 109.5, 109.5, 109.5],  # 6 angoli H-C-H
        symmetry="Td",
        dipole_moment=0.0
    )


def create_ammonia() -> Molecule:
    """Crea molecola di ammoniaca NH3 (piramidale)"""
    angle = 107.3  # Angolo H-N-H
    bond_length = 1.01
    
    angle_rad = np.radians(angle / 2)
    h_z = bond_length * np.cos(np.radians(90 - angle/2))
    h_r = bond_length * np.sin(np.radians(90 - angle/2))
    
    atoms = [
        Atom("N", (0.0, 0.0, 0.0)),
        Atom("H", (h_r, 0.0, -h_z)),
        Atom("H", (h_r * np.cos(2*np.pi/3), h_r * np.sin(2*np.pi/3), -h_z)),
        Atom("H", (h_r * np.cos(4*np.pi/3), h_r * np.sin(4*np.pi/3), -h_z)),
    ]
    
    bonds = [Bond(0, i, "single", 1.01) for i in range(1, 4)]
    
    return Molecule(
        name="Ammonia",
        formula="NH3",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[107.3, 107.3, 107.3],
        symmetry="C3v",
        dipole_moment=1.47
    )


def create_ozone() -> Molecule:
    """Crea molecola di ozono O3"""
    angle = 116.8
    bond_length = 1.28
    
    angle_rad = np.radians(angle / 2)
    
    atoms = [
        Atom("O", (0.0, 0.0, 0.0)),
        Atom("O", (bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
        Atom("O", (-bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
    ]
    
    bonds = [
        Bond(0, 1, "single", 1.28),  # Risonanza
        Bond(0, 2, "single", 1.28),
    ]
    
    return Molecule(
        name="Ozone",
        formula="O3",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[116.8],
        symmetry="C2v",
        dipole_moment=0.53
    )


def create_hydrogen_sulfide() -> Molecule:
    """Crea molecola di H2S"""
    angle = 92.1
    bond_length = 1.34
    
    angle_rad = np.radians(angle / 2)
    
    atoms = [
        Atom("S", (0.0, 0.0, 0.0)),
        Atom("H", (bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
        Atom("H", (-bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
    ]
    
    bonds = [
        Bond(0, 1, "single", 1.34),
        Bond(0, 2, "single", 1.34),
    ]
    
    return Molecule(
        name="Hydrogen Sulfide",
        formula="H2S",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[92.1],
        symmetry="C2v",
        dipole_moment=0.97
    )


def create_sulfur_dioxide() -> Molecule:
    """Crea molecola di SO2"""
    angle = 119.0
    bond_length = 1.43
    
    angle_rad = np.radians(angle / 2)
    
    atoms = [
        Atom("S", (0.0, 0.0, 0.0)),
        Atom("O", (bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
        Atom("O", (-bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
    ]
    
    bonds = [
        Bond(0, 1, "double", 1.43),
        Bond(0, 2, "double", 1.43),
    ]
    
    return Molecule(
        name="Sulfur Dioxide",
        formula="SO2",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[119.0],
        symmetry="C2v",
        dipole_moment=1.63
    )


def create_nitrogen_dioxide() -> Molecule:
    """Crea molecola di NO2"""
    angle = 134.0
    bond_length = 1.20
    
    angle_rad = np.radians(angle / 2)
    
    atoms = [
        Atom("N", (0.0, 0.0, 0.0)),
        Atom("O", (bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
        Atom("O", (-bond_length * np.sin(angle_rad), bond_length * np.cos(angle_rad), 0.0)),
    ]
    
    bonds = [
        Bond(0, 1, "double", 1.20),
        Bond(0, 2, "single", 1.20),
    ]
    
    return Molecule(
        name="Nitrogen Dioxide",
        formula="NO2",
        atoms=atoms,
        bonds=bonds,
        bond_angles=[134.0],
        symmetry="C2v",
        dipole_moment=0.32
    )


# Database delle molecole
MOLECULES_DB: Dict[str, Molecule] = {
    "H2O": create_water(),
    "CO2": create_carbon_dioxide(),
    "CH4": create_methane(),
    "NH3": create_ammonia(),
    "O3": create_ozone(),
    "H2S": create_hydrogen_sulfide(),
    "SO2": create_sulfur_dioxide(),
    "NO2": create_nitrogen_dioxide(),
}


# ══════════════════════════════════════════════════════════════════════════════
# GENERATORE SUONO MOLECOLARE
# ══════════════════════════════════════════════════════════════════════════════

class MolecularSounder:
    """
    Genera suono dalla geometria molecolare.
    
    Mapping:
    - Angoli di legame → Fasi audio (gradi → radianti)
    - Lunghezze di legame → Frequenze (inversamente proporzionali)
    - Elettronegatività → Ampiezze
    - Massa atomica → Posizione stereo (pesanti al centro)
    - Simmetria → Pattern
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.spectral_sounder = SpectralSounder(sample_rate)
        
        # Range frequenze audio
        self.freq_min = 80.0
        self.freq_max = 2000.0
        
        # Range lunghezze legame (Å)
        self.bond_min = 0.7
        self.bond_max = 2.0
    
    def get_molecule(self, formula: str) -> Optional[Molecule]:
        """Ottiene una molecola dal database"""
        return MOLECULES_DB.get(formula)
    
    def get_available_molecules(self) -> List[str]:
        """Lista delle molecole disponibili"""
        return list(MOLECULES_DB.keys())
    
    def bond_length_to_frequency(self, length: float) -> float:
        """
        Converte lunghezza di legame in frequenza.
        Legami corti → frequenze alte (più rigidi, vibrano più veloce)
        """
        # Normalizza e inverti
        length_clamped = np.clip(length, self.bond_min, self.bond_max)
        t = (length_clamped - self.bond_min) / (self.bond_max - self.bond_min)
        
        # Inverti: corto → alto
        t_inv = 1.0 - t
        
        # Scala a range audio con curva golden
        freq = self.freq_min + t_inv * (self.freq_max - self.freq_min)
        
        return freq
    
    def angle_to_phase(self, angle_deg: float) -> float:
        """
        Converte angolo di legame in fase audio.
        L'angolo molecolare diventa la fase dell'onda.
        """
        return np.radians(angle_deg)
    
    def electronegativity_to_amplitude(self, electroneg: float) -> float:
        """
        Converte elettronegatività in ampiezza.
        Più elettronegativo → più "presente" nel suono.
        """
        # Scala Pauling: ~2.0 a ~4.0
        return np.clip((electroneg - 1.5) / 2.5, 0.2, 1.0)
    
    def mass_to_pan(self, mass: float, total_mass: float) -> float:
        """
        Converte massa atomica in posizione stereo.
        Atomi pesanti → centro, leggeri → lati.
        """
        if total_mass == 0:
            return 0.5
        
        # Normalizza massa
        ratio = mass / total_mass
        
        # Pesanti al centro (pan=0.5), leggeri ai lati
        return 0.5 - (0.5 - ratio) * 0.8
    
    def generate_atom_sound(self, 
                           atom: Atom, 
                           duration: float,
                           base_freq: float,
                           phase: float) -> np.ndarray:
        """Genera il suono di un singolo atomo"""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Ampiezza da elettronegatività
        amp = self.electronegativity_to_amplitude(atom.electronegativity)
        
        # Frequenza base modificata dalla massa (leggeri = più alto)
        freq = base_freq * (16.0 / atom.mass) ** 0.25  # Scala con radice quarta
        
        # Genera onda con fase specificata
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        
        return signal
    
    def generate_molecule_sound(self,
                                molecule: Molecule,
                                duration: float = 3.0,
                                use_spectral: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera il suono completo di una molecola.
        
        Args:
            molecule: Molecola da sonificare
            duration: Durata in secondi
            use_spectral: Se True, usa linee spettrali reali per ogni elemento
        
        Returns:
            Tuple (left, right) dei canali stereo
        """
        num_samples = int(self.sample_rate * duration)
        left = np.zeros(num_samples)
        right = np.zeros(num_samples)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        total_mass = molecule.total_mass
        
        # Calcola la fase base dagli angoli di legame
        base_phases = []
        if molecule.bond_angles:
            # Usa il primo angolo come fase principale
            main_angle = molecule.bond_angles[0]
            base_phases = [self.angle_to_phase(main_angle)]
            
            # Aggiungi fasi per ogni angolo aggiuntivo
            for angle in molecule.bond_angles[1:]:
                base_phases.append(self.angle_to_phase(angle))
        else:
            base_phases = [0.0]
        
        # Genera suono per ogni atomo
        for i, atom in enumerate(molecule.atoms):
            # Fase: combina angolo di legame con indice atomo
            phase_idx = i % len(base_phases)
            phase = base_phases[phase_idx] + (2 * np.pi * PHI_CONJUGATE * i)
            
            # Frequenza dal legame (se presente)
            if molecule.bonds and i < len(molecule.bonds):
                bond = molecule.bonds[min(i, len(molecule.bonds)-1)]
                base_freq = self.bond_length_to_frequency(bond.length)
            else:
                base_freq = 300.0  # Default
            
            # Genera suono atomo
            if use_spectral:
                # Usa linee spettrali reali se disponibili
                element_name = self._get_element_name(atom.symbol)
                if element_name:
                    try:
                        atom_signal = self.spectral_sounder.generate_element_sound(
                            element_name, duration, PhaseMode.GOLDEN, amplitude=0.8
                        )
                        # Applica fase molecolare
                        atom_signal = np.roll(atom_signal, int(phase * num_samples / (2 * np.pi)))
                    except:
                        atom_signal = self.generate_atom_sound(atom, duration, base_freq, phase)
                else:
                    atom_signal = self.generate_atom_sound(atom, duration, base_freq, phase)
            else:
                atom_signal = self.generate_atom_sound(atom, duration, base_freq, phase)
            
            # Calcola pan stereo
            pan = self.mass_to_pan(atom.mass, total_mass)
            
            # Distribuisci nei canali
            left += atom_signal * np.cos(pan * np.pi / 2)
            right += atom_signal * np.sin(pan * np.pi / 2)
        
        # Normalizza
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val
        
        # Applica envelope
        left = self._apply_envelope(left)
        right = self._apply_envelope(right)
        
        return left * 0.8, right * 0.8
    
    def generate_molecule_binaural(self,
                                   molecule: Molecule,
                                   beat_freq: float = 7.83,
                                   duration: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera battimenti binaurali basati sulla geometria molecolare.
        L'angolo di legame determina la differenza di fase tra i canali.
        """
        num_samples = int(self.sample_rate * duration)
        left = np.zeros(num_samples)
        right = np.zeros(num_samples)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Angolo principale della molecola → fase binaural
        main_angle = molecule.bond_angles[0] if molecule.bond_angles else 104.5
        phase_diff = self.angle_to_phase(main_angle)
        
        for i, atom in enumerate(molecule.atoms):
            # Frequenza base
            if molecule.bonds and i < len(molecule.bonds):
                bond = molecule.bonds[min(i, len(molecule.bonds)-1)]
                base_freq = self.bond_length_to_frequency(bond.length)
            else:
                base_freq = 200.0 + i * 100
            
            # Ampiezza
            amp = self.electronegativity_to_amplitude(atom.electronegativity)
            
            # Beat frequency scala con massa (leggeri = beat più veloce)
            current_beat = beat_freq * (1.0 / atom.mass) ** 0.5 * 4.0
            
            # Left: frequenza base con fase da indice
            atom_phase = 2 * np.pi * PHI_CONJUGATE * i
            left += amp * np.sin(2 * np.pi * base_freq * t + atom_phase)
            
            # Right: frequenza + beat, con fase molecolare
            right += amp * np.sin(2 * np.pi * (base_freq + current_beat) * t + atom_phase + phase_diff)
        
        # Normalizza
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val
        
        left = self._apply_envelope(left)
        right = self._apply_envelope(right)
        
        return left * 0.8, right * 0.8
    
    def _get_element_name(self, symbol: str) -> Optional[str]:
        """Mappa simbolo atomico a nome nel database spettrale"""
        mapping = {
            "H": "Hydrogen-Balmer",
            "O": "Oxygen",
            "N": "Nitrogen",
            "C": None,  # Non abbiamo carbonio puro
            "S": None,
            "He": "Helium",
            "Na": "Sodium",
            "Ne": "Neon",
            "Fe": "Iron",
            "Ca": "Calcium",
            "Mg": "Magnesium",
        }
        return mapping.get(symbol)
    
    def _apply_envelope(self, signal: np.ndarray) -> np.ndarray:
        """Applica envelope con proporzioni auree"""
        length = len(signal)
        attack = int(length * PHI_CONJUGATE * PHI_CONJUGATE * 0.2)
        release = int(length * PHI_CONJUGATE * 0.3)
        
        envelope = np.ones(length)
        
        # Attack
        for i in range(attack):
            t = i / attack
            envelope[i] = (1 - np.cos(t * np.pi * PHI * PHI_CONJUGATE)) / 2
        
        # Release
        for i in range(release):
            t = i / release
            envelope[length - 1 - i] = (1 - np.cos(t * np.pi * PHI * PHI_CONJUGATE)) / 2
        
        return signal * envelope
    
    def print_molecule_info(self, molecule: Molecule):
        """Stampa informazioni sulla molecola"""
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║  MOLECOLA: {molecule.name:^48} ║")
        print(f"║  Formula: {molecule.formula:^49} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Simmetria: {molecule.symmetry:<15} Dipolo: {molecule.dipole_moment:.2f} D{' '*18} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  {'Atomo':<6} {'Massa':<8} {'Electroneg':<12} {'Posizione':<25} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        
        for atom in molecule.atoms:
            pos_str = f"({atom.position[0]:.2f}, {atom.position[1]:.2f}, {atom.position[2]:.2f})"
            print(f"║  {atom.symbol:<6} {atom.mass:<8.3f} {atom.electronegativity:<12.2f} {pos_str:<25} ║")
        
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  ANGOLI DI LEGAME (→ FASI AUDIO):                            ║")
        for i, angle in enumerate(molecule.bond_angles):
            phase_rad = self.angle_to_phase(angle)
            print(f"║    Angolo {i+1}: {angle:>7.2f}° → Fase: {phase_rad:>6.3f} rad{' '*15} ║")
        
        print(f"╚══════════════════════════════════════════════════════════════╝\n")
    
    def save_wav(self, left: np.ndarray, right: np.ndarray, filename: str):
        """Salva audio stereo in WAV"""
        from scipy.io import wavfile
        
        stereo = np.column_stack((left, right))
        stereo = (stereo * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, stereo)
        print(f"💾 Salvato: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    """Dimostra le capacità del modulo"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MOLECULAR GEOMETRY SOUND - DEMO                           ║
║                    Suonare le molecole attraverso la loro geometria          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    sounder = MolecularSounder()
    
    # Mostra molecole disponibili
    print("🧪 Molecole disponibili:")
    for formula in sounder.get_available_molecules():
        mol = sounder.get_molecule(formula)
        if mol:
            print(f"   • {formula}: {mol.name} (angolo: {mol.bond_angles[0] if mol.bond_angles else 'N/A'}°)")
    print()
    
    # Genera suono acqua
    water = sounder.get_molecule("H2O")
    if water:
        sounder.print_molecule_info(water)
        
        print("🎵 Generazione suono H2O...")
        left, right = sounder.generate_molecule_sound(water, duration=4.0)
        sounder.save_wav(left, right, "water_molecular.wav")
        
        print("🎵 Generazione binaural H2O (angolo 104.5° = fase)...")
        left, right = sounder.generate_molecule_binaural(water, beat_freq=7.83, duration=5.0)
        sounder.save_wav(left, right, "water_binaural.wav")
    
    # Genera suono CO2
    co2 = sounder.get_molecule("CO2")
    if co2:
        sounder.print_molecule_info(co2)
        
        print("🎵 Generazione suono CO2 (lineare, 180°)...")
        left, right = sounder.generate_molecule_sound(co2, duration=4.0)
        sounder.save_wav(left, right, "co2_molecular.wav")
    
    print("\n✅ Demo completata!")


if __name__ == "__main__":
    demo()
