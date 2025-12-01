#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPECTRAL SOUND - SUONARE GLI ELEMENTI                     ║
║                                                                              ║
║   Trasforma le linee spettrali atomiche in suono                            ║
║   - Frequenze ottiche → frequenze audio (scaling lineare)                    ║
║   - Intensità relative → ampiezze (da dati reali)                           ║
║   - Fasi quantistiche → incoerenti (random) o coerenti                      ║
║                                                                              ║
║   "Ogni elemento ha la sua voce unica - il suo timbro atomico"              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json

# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED CONSTANTS (from golden_constants module)
# ══════════════════════════════════════════════════════════════════════════════

from golden_constants import (
    PHI, PHI_CONJUGATE, SAMPLE_RATE,
    C, RYDBERG, PLANCK, FINE_STRUCTURE_INVERSE,
    AUDIO_FREQ_MIN, AUDIO_FREQ_MAX,
    golden_ease, apply_golden_envelope,
    FIBONACCI,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATI SPETTRALI REALI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralLine:
    """Una linea spettrale atomica"""
    name: str                    # Nome (es. "H-α", "H-β")
    wavelength_nm: float         # Lunghezza d'onda in nanometri
    frequency_optical: float     # Frequenza ottica in Hz
    intensity_relative: float    # Intensità relativa (0-1 normalizzata)
    transition: str              # Descrizione transizione (es. "3→2")
    n_upper: int                 # Livello energetico superiore
    n_lower: int                 # Livello energetico inferiore
    
    @property
    def frequency_audio(self) -> float:
        """Frequenza audio scalata (calcolata dinamicamente)"""
        return 0.0  # Placeholder, calcolata nel contesto
    
    @property
    def energy_ev(self) -> float:
        """Energia del fotone in eV"""
        return (PLANCK * self.frequency_optical) / 1.602e-19


class PhaseMode(Enum):
    """Modalità di fase per le onde"""
    INCOHERENT = "incoherent"    # Fasi random (realistico quantistico)
    COHERENT = "coherent"        # Fasi = 0 (armonica perfetta)
    GOLDEN = "golden"            # Fasi in rapporto aureo
    FIBONACCI = "fibonacci"      # Fasi da sequenza Fibonacci


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE ELEMENTI - SERIE SPETTRALI
# ══════════════════════════════════════════════════════════════════════════════

# IDROGENO - Serie di Balmer (transizioni verso n=2, visibile)
HYDROGEN_BALMER = [
    # (nome, λ nm, intensità relativa su 180)
    ("H-α", 656.281, 180),   # 3→2, rosso intenso
    ("H-β", 486.135, 80),    # 4→2, ciano
    ("H-γ", 434.047, 30),    # 5→2, blu
    ("H-δ", 410.174, 15),    # 6→2, viola
    ("H-ε", 397.007, 8),     # 7→2, viola profondo
    ("H-ζ", 388.905, 6),     # 8→2, UV vicino
    ("H-η", 383.539, 5),     # 9→2, UV
]

# IDROGENO - Serie di Lyman (transizioni verso n=1, UV)
HYDROGEN_LYMAN = [
    ("Ly-α", 121.567, 200),  # 2→1, UV forte
    ("Ly-β", 102.572, 50),   # 3→1
    ("Ly-γ", 97.254, 20),    # 4→1
    ("Ly-δ", 94.974, 10),    # 5→1
]

# IDROGENO - Serie di Paschen (transizioni verso n=3, infrarosso)
HYDROGEN_PASCHEN = [
    ("Pa-α", 1875.1, 100),   # 4→3, IR
    ("Pa-β", 1282.2, 40),    # 5→3
    ("Pa-γ", 1093.8, 20),    # 6→3
]

# ELIO - Linee principali visibili
HELIUM_VISIBLE = [
    ("He-D3", 587.562, 100),  # Giallo (D3 line)
    ("He", 501.567, 60),      # Verde
    ("He", 492.193, 50),      # Ciano
    ("He", 471.314, 30),      # Blu
    ("He", 447.148, 80),      # Blu profondo (forte!)
    ("He", 402.619, 25),      # Viola
    ("He", 388.865, 40),      # Viola
]

# SODIO - Doppietto D (giallo caratteristico)
SODIUM_D = [
    ("Na-D1", 589.592, 100),  # D1
    ("Na-D2", 588.995, 100),  # D2 (doppietto)
]

# NEON - Linee visibili (caratteristico rosso-arancio)
NEON_VISIBLE = [
    ("Ne", 640.225, 100),    # Rosso
    ("Ne", 633.443, 80),     # Rosso (laser HeNe!)
    ("Ne", 621.728, 50),     # Arancio-rosso
    ("Ne", 614.306, 40),     # Arancio
    ("Ne", 609.616, 35),     # Arancio
    ("Ne", 585.249, 30),     # Giallo
]

# MERCURIO - Linee visibili
MERCURY_VISIBLE = [
    ("Hg", 579.066, 80),     # Giallo
    ("Hg", 576.960, 80),     # Giallo (doppietto)
    ("Hg", 546.074, 100),    # Verde brillante
    ("Hg", 435.833, 70),     # Blu
    ("Hg", 404.656, 50),     # Viola
]

# OSSIGENO - Linee atmosferiche
OXYGEN_ATMOSPHERIC = [
    ("O-I", 777.194, 100),   # IR triplet (forte)
    ("O-I", 777.417, 100),
    ("O-I", 777.539, 100),
    ("O-I", 844.636, 50),    # IR
    ("O-I", 630.030, 30),    # Rosso (aurora!)
    ("O-I", 557.734, 80),    # Verde (aurora!)
]

# AZOTO - Linee atmosferiche
NITROGEN_ATMOSPHERIC = [
    ("N-I", 746.831, 60),
    ("N-I", 744.229, 60),
    ("N-I", 742.364, 60),
    ("N-II", 500.515, 40),   # Blu-verde
]

# FERRO - Linee (spettro solare)
IRON_VISIBLE = [
    ("Fe-I", 527.039, 50),
    ("Fe-I", 526.954, 100),  # Forte nel sole
    ("Fe-I", 495.761, 40),
    ("Fe-I", 466.814, 30),
    ("Fe-I", 438.355, 60),
]

# CALCIO - Linee H e K (Fraunhofer)
CALCIUM_HK = [
    ("Ca-K", 393.366, 100),  # K line (viola)
    ("Ca-H", 396.847, 100),  # H line (viola)
]

# MAGNESIO - Triplett b
MAGNESIUM_B = [
    ("Mg-b1", 518.362, 100),
    ("Mg-b2", 517.270, 80),
    ("Mg-b4", 516.732, 60),
]


# ══════════════════════════════════════════════════════════════════════════════
# MOLECOLE - Bande Spettrali
# ══════════════════════════════════════════════════════════════════════════════

# ACQUA (H2O) - Bande di assorbimento IR
WATER_IR_BANDS = [
    ("H2O-ν1", 2734, 100),   # Symmetric stretch
    ("H2O-ν2", 6270, 60),    # Bending
    ("H2O-ν3", 2662, 80),    # Asymmetric stretch
]

# ANIDRIDE CARBONICA (CO2)
CO2_IR_BANDS = [
    ("CO2-ν1", 7200, 30),    # Symmetric stretch (Raman active)
    ("CO2-ν2", 15000, 100),  # Bending (IR active)
    ("CO2-ν3", 4300, 80),    # Asymmetric stretch
]

# METANO (CH4)
METHANE_IR = [
    ("CH4-ν1", 3020, 40),
    ("CH4-ν2", 6000, 30),
    ("CH4-ν3", 3300, 100),
    ("CH4-ν4", 7700, 60),
]

# OZONO (O3)
OZONE_BANDS = [
    ("O3-Hartley", 255, 100),   # UV (assorbimento)
    ("O3-Huggins", 320, 50),    # UV
    ("O3-Chappuis", 600, 30),   # Visibile
]


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE: SPECTRAL SOUNDER
# ══════════════════════════════════════════════════════════════════════════════

class SpectralSounder:
    """
    Genera suono dalle linee spettrali degli elementi.
    Trasforma lo spettro elettromagnetico in spettro audio.
    """
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 audio_min: float = AUDIO_FREQ_MIN,
                 audio_max: float = AUDIO_FREQ_MAX):
        self.sample_rate = sample_rate
        self.audio_min = audio_min
        self.audio_max = audio_max
        
        # Cache delle linee spettrali processate
        self._elements: Dict[str, List[SpectralLine]] = {}
        
        # Carica elementi predefiniti
        self._load_default_elements()
    
    def _load_default_elements(self):
        """Carica tutti gli elementi nel database"""
        self._add_element("Hydrogen-Balmer", HYDROGEN_BALMER, n_lower=2)
        self._add_element("Hydrogen-Lyman", HYDROGEN_LYMAN, n_lower=1)
        self._add_element("Hydrogen-Paschen", HYDROGEN_PASCHEN, n_lower=3)
        self._add_element("Helium", HELIUM_VISIBLE)
        self._add_element("Sodium", SODIUM_D)
        self._add_element("Neon", NEON_VISIBLE)
        self._add_element("Mercury", MERCURY_VISIBLE)
        self._add_element("Oxygen", OXYGEN_ATMOSPHERIC)
        self._add_element("Nitrogen", NITROGEN_ATMOSPHERIC)
        self._add_element("Iron", IRON_VISIBLE)
        self._add_element("Calcium", CALCIUM_HK)
        self._add_element("Magnesium", MAGNESIUM_B)
        
        # Molecole
        self._add_element("Water", WATER_IR_BANDS, is_molecule=True)
        self._add_element("CO2", CO2_IR_BANDS, is_molecule=True)
        self._add_element("Methane", METHANE_IR, is_molecule=True)
        self._add_element("Ozone", OZONE_BANDS, is_molecule=True)
    
    def _add_element(self, name: str, lines_data: list, 
                     n_lower: int = 0, is_molecule: bool = False):
        """Aggiunge un elemento al database"""
        lines = []
        max_intensity = max(line[2] for line in lines_data)
        
        for i, (line_name, wavelength, intensity) in enumerate(lines_data):
            freq_optical = C / (wavelength * 1e-9)  # nm → m → Hz
            
            # Per idrogeno, calcola n_upper dalla formula di Balmer
            if "Hydrogen" in name and n_lower > 0:
                n_upper = n_lower + i + 1
                transition = f"{n_upper}→{n_lower}"
            else:
                n_upper = 0
                transition = line_name
            
            lines.append(SpectralLine(
                name=line_name,
                wavelength_nm=wavelength,
                frequency_optical=freq_optical,
                intensity_relative=intensity / max_intensity,
                transition=transition,
                n_upper=n_upper,
                n_lower=n_lower
            ))
        
        self._elements[name] = lines
    
    def get_element_names(self) -> List[str]:
        """Restituisce i nomi degli elementi disponibili"""
        return list(self._elements.keys())
    
    def get_spectral_lines(self, element: str) -> List[SpectralLine]:
        """Ottiene le linee spettrali di un elemento"""
        return self._elements.get(element, [])
    
    def scale_to_audio(self, lines: List[SpectralLine]) -> List[Tuple[float, float]]:
        """
        Scala le frequenze ottiche a frequenze audio.
        Ritorna lista di (freq_audio, amplitude)
        """
        if not lines:
            return []
        
        # Trova range ottico
        optical_freqs = [line.frequency_optical for line in lines]
        opt_min, opt_max = min(optical_freqs), max(optical_freqs)
        
        # Scaling lineare (può essere modificato per log, golden, etc.)
        scaled = []
        for line in lines:
            if opt_max > opt_min:
                # Normalizza a [0, 1]
                t = (line.frequency_optical - opt_min) / (opt_max - opt_min)
                # Scala a range audio
                freq_audio = self.audio_min + t * (self.audio_max - self.audio_min)
            else:
                freq_audio = (self.audio_min + self.audio_max) / 2
            
            scaled.append((freq_audio, line.intensity_relative))
        
        return scaled
    
    def generate_phases(self, num_lines: int, mode: PhaseMode) -> np.ndarray:
        """Genera fasi secondo la modalità scelta"""
        if mode == PhaseMode.INCOHERENT:
            return np.random.uniform(0, 2 * np.pi, num_lines)
        
        elif mode == PhaseMode.COHERENT:
            return np.zeros(num_lines)
        
        elif mode == PhaseMode.GOLDEN:
            # Fasi in rapporto aureo: φ^0, φ^1, φ^2, ... mod 2π
            return np.array([
                (2 * np.pi * PHI_CONJUGATE ** i) % (2 * np.pi)
                for i in range(num_lines)
            ])
        
        elif mode == PhaseMode.FIBONACCI:
            # Fasi dalla sequenza di Fibonacci normalizzata
            fib = [1, 1]
            while len(fib) < num_lines:
                fib.append(fib[-1] + fib[-2])
            fib = fib[:num_lines]
            fib_max = max(fib)
            return np.array([2 * np.pi * f / fib_max for f in fib])
        
        return np.zeros(num_lines)
    
    def generate_element_sound(self,
                               element: str,
                               duration: float = 2.0,
                               phase_mode: PhaseMode = PhaseMode.INCOHERENT,
                               amplitude: float = 0.8,
                               envelope: bool = True) -> np.ndarray:
        """
        Genera il suono di un elemento dalle sue linee spettrali.
        
        Args:
            element: Nome dell'elemento
            duration: Durata in secondi
            phase_mode: Modalità delle fasi
            amplitude: Ampiezza massima [0, 1]
            envelope: Applica envelope attack/decay
        
        Returns:
            Array numpy del segnale audio
        """
        lines = self.get_spectral_lines(element)
        if not lines:
            raise ValueError(f"Elemento '{element}' non trovato")
        
        # Scala a frequenze audio
        scaled = self.scale_to_audio(lines)
        
        # Genera fasi
        phases = self.generate_phases(len(lines), phase_mode)
        
        # Genera segnale
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        signal = np.zeros(num_samples)
        for i, (freq, amp) in enumerate(scaled):
            signal += amp * np.sin(2 * np.pi * freq * t + phases[i])
        
        # Normalizza
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))
        
        # Applica envelope
        if envelope:
            signal = self._apply_golden_envelope(signal)
        
        # Scala ampiezza finale
        signal *= amplitude
        
        return signal
    
    def generate_element_stereo(self,
                                element: str,
                                duration: float = 2.0,
                                phase_mode: PhaseMode = PhaseMode.GOLDEN,
                                stereo_spread: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera suono stereo con linee distribuite nel campo stereo.
        Le linee più basse tendono al centro, quelle alte si allargano.
        """
        lines = self.get_spectral_lines(element)
        if not lines:
            raise ValueError(f"Elemento '{element}' non trovato")
        
        scaled = self.scale_to_audio(lines)
        phases = self.generate_phases(len(lines), phase_mode)
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        left = np.zeros(num_samples)
        right = np.zeros(num_samples)
        
        for i, (freq, amp) in enumerate(scaled):
            wave = amp * np.sin(2 * np.pi * freq * t + phases[i])
            
            # Pan: frequenze basse al centro, alte distribuite
            pan_position = (i / len(scaled)) * stereo_spread
            pan_l = np.cos(pan_position * np.pi / 2)
            pan_r = np.sin(pan_position * np.pi / 2) + (1 - stereo_spread)
            
            left += wave * pan_l
            right += wave * pan_r
        
        # Normalizza
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val
        
        # Envelope
        left = self._apply_golden_envelope(left)
        right = self._apply_golden_envelope(right)
        
        return left * 0.8, right * 0.8
    
    def generate_binaural_element(self,
                                  element: str,
                                  beat_freq: float = 7.83,  # Schumann resonance
                                  duration: float = 5.0,
                                  phase_mode: PhaseMode = PhaseMode.GOLDEN) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera battimenti binaurali dalle linee spettrali.
        Ogni linea crea un battimento con frequenza proporzionale.
        """
        lines = self.get_spectral_lines(element)
        if not lines:
            raise ValueError(f"Elemento '{element}' non trovato")
        
        scaled = self.scale_to_audio(lines)
        phases = self.generate_phases(len(lines), phase_mode)
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        left = np.zeros(num_samples)
        right = np.zeros(num_samples)
        
        for i, (freq, amp) in enumerate(scaled):
            # Beat frequency scala con intensità (più forte = beat più lento)
            current_beat = beat_freq * amp
            
            # Left: frequenza base
            left += amp * np.sin(2 * np.pi * freq * t + phases[i])
            
            # Right: frequenza + beat, con fase golden angle
            phase_right = phases[i] + (137.5 * np.pi / 180)  # Golden angle
            right += amp * np.sin(2 * np.pi * (freq + current_beat) * t + phase_right)
        
        # Normalizza
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val
        
        left = self._apply_golden_envelope(left)
        right = self._apply_golden_envelope(right)
        
        return left * 0.8, right * 0.8
    
    def _apply_golden_envelope(self, signal: np.ndarray) -> np.ndarray:
        """Applica envelope con proporzioni auree (usa funzione centralizzata)"""
        return apply_golden_envelope(signal)
    
    def _golden_ease(self, t: float) -> float:
        """Curva di easing basata su golden ratio (usa funzione centralizzata)"""
        return golden_ease(t)
    
    def save_wav(self, signal: np.ndarray, filename: str, 
                 stereo: bool = False, right_channel: np.ndarray = None):
        """Salva il segnale come file WAV"""
        from scipy.io import wavfile
        
        if stereo and right_channel is not None:
            # Interleave L/R
            stereo_signal = np.column_stack((signal, right_channel))
            stereo_signal = (stereo_signal * 32767).astype(np.int16)
            wavfile.write(filename, self.sample_rate, stereo_signal)
        else:
            mono = (signal * 32767).astype(np.int16)
            wavfile.write(filename, self.sample_rate, mono)
        
        print(f"💾 Salvato: {filename}")
    
    def print_element_info(self, element: str):
        """Stampa informazioni dettagliate su un elemento"""
        lines = self.get_spectral_lines(element)
        if not lines:
            print(f"❌ Elemento '{element}' non trovato")
            return
        
        scaled = self.scale_to_audio(lines)
        
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║  ELEMENTO: {element:^48} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  {'Linea':<10} {'λ (nm)':<10} {'ν ottica (THz)':<15} {'f audio (Hz)':<12} {'Amp':<6} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        
        for line, (f_audio, amp) in zip(lines, scaled):
            freq_thz = line.frequency_optical / 1e12
            print(f"║  {line.name:<10} {line.wavelength_nm:<10.2f} {freq_thz:<15.3f} {f_audio:<12.1f} {amp:<6.2f} ║")
        
        print(f"╚══════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONI DI UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def calculate_hydrogen_wavelength(n_upper: int, n_lower: int) -> float:
    """
    Calcola la lunghezza d'onda per una transizione dell'idrogeno.
    Formula di Rydberg: 1/λ = R_H * (1/n_lower² - 1/n_upper²)
    """
    inv_lambda = RYDBERG * (1/n_lower**2 - 1/n_upper**2)
    return 1e9 / inv_lambda  # Ritorna in nm


def generate_hydrogen_series(n_lower: int, n_max: int = 10) -> List[Tuple[str, float, float]]:
    """
    Genera una serie di linee dell'idrogeno.
    
    Args:
        n_lower: Livello inferiore (1=Lyman, 2=Balmer, 3=Paschen)
        n_max: Livello superiore massimo
    
    Returns:
        Lista di (nome, lunghezza d'onda nm, intensità relativa)
    """
    series_names = {1: "Ly", 2: "H", 3: "Pa", 4: "Br", 5: "Pf"}
    greek = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
    
    lines = []
    for i, n_upper in enumerate(range(n_lower + 1, n_max + 1)):
        wavelength = calculate_hydrogen_wavelength(n_upper, n_lower)
        
        # Intensità scala approssimativamente con 1/n³
        intensity = 1.0 / (n_upper ** 3)
        
        prefix = series_names.get(n_lower, "X")
        suffix = greek[i] if i < len(greek) else str(i)
        name = f"{prefix}-{suffix}"
        
        lines.append((name, wavelength, intensity * 100))  # Scala a 100 max
    
    # Normalizza
    max_int = max(line[2] for line in lines)
    lines = [(n, w, i/max_int * 100) for n, w, i in lines]
    
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# MAIN / DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    """Dimostra le capacità del modulo"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPECTRAL SOUND - DEMO                                     ║
║                    Suonare gli elementi della tavola periodica               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    sounder = SpectralSounder()
    
    # Mostra elementi disponibili
    print("📊 Elementi disponibili:")
    for name in sounder.get_element_names():
        print(f"   • {name}")
    print()
    
    # Info dettagliate su Idrogeno-Balmer
    sounder.print_element_info("Hydrogen-Balmer")
    
    # Genera suono idrogeno
    print("🎵 Generazione suono Idrogeno (Balmer)...")
    signal = sounder.generate_element_sound(
        "Hydrogen-Balmer",
        duration=3.0,
        phase_mode=PhaseMode.GOLDEN
    )
    sounder.save_wav(signal, "hydrogen_balmer.wav")
    
    # Genera stereo elio
    print("🎵 Generazione stereo Elio...")
    left, right = sounder.generate_element_stereo(
        "Helium",
        duration=3.0,
        phase_mode=PhaseMode.GOLDEN
    )
    sounder.save_wav(left, "helium_stereo.wav", stereo=True, right_channel=right)
    
    # Genera binaural sodio
    print("🎵 Generazione binaural Sodio (doppietto D)...")
    left, right = sounder.generate_binaural_element(
        "Sodium",
        beat_freq=10.0,  # Alpha waves
        duration=5.0
    )
    sounder.save_wav(left, "sodium_binaural.wav", stereo=True, right_channel=right)
    
    print("\n✅ Demo completata! Ascolta i file WAV generati.")


if __name__ == "__main__":
    demo()
