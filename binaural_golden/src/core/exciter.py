"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EXCITER SPECIFICATIONS                               ║
║                                                                              ║
║   Database of audio exciters (Visaton, Dayton, etc.) for vibroacoustic      ║
║   plate design with power handling, frequency response, and impedance.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ExciterType(Enum):
    """Type of exciter mechanism."""
    VOICE_COIL = "voice_coil"      # Standard voice coil
    PIEZO = "piezo"                 # Piezoelectric
    TACTILE = "tactile"            # Tactile transducer
    BASS_SHAKER = "bass_shaker"    # Low frequency


@dataclass
class Exciter:
    """
    Audio exciter/transducer specifications.
    
    Used for driving vibroacoustic plates.
    """
    name: str
    manufacturer: str
    model: str
    
    # Power specifications
    power_rms_w: float          # RMS power in watts
    power_peak_w: float         # Peak power in watts
    
    # Electrical specifications  
    impedance_ohm: float        # Nominal impedance (Ω)
    
    # Frequency response
    freq_min_hz: float          # Lower cutoff (-10dB)
    freq_max_hz: float          # Upper cutoff (-10dB)
    freq_resonance_hz: float    # Resonance frequency (Fs)
    
    # Physical specifications
    diameter_mm: float          # Outer diameter
    height_mm: float            # Total height including mounting
    weight_g: float             # Weight in grams
    
    # Type
    exciter_type: ExciterType = ExciterType.VOICE_COIL
    
    # Force output
    bl_factor: float = 0.0      # BL factor (Tm) if known
    
    # Notes
    notes: str = ""
    
    @property
    def power_rms_per_area(self) -> float:
        """Power per area (W/mm²)."""
        area = np.pi * (self.diameter_mm / 2) ** 2
        return self.power_rms_w / area if area > 0 else 0
    
    @property
    def frequency_range(self) -> Tuple[float, float]:
        """Get usable frequency range."""
        return (self.freq_min_hz, self.freq_max_hz)
    
    @property
    def octaves(self) -> float:
        """Number of octaves covered."""
        if self.freq_min_hz <= 0:
            return 0
        return np.log2(self.freq_max_hz / self.freq_min_hz)
    
    def efficiency_at_frequency(self, freq: float) -> float:
        """
        Estimate relative efficiency at a given frequency.
        Peak efficiency near resonance, rolls off at extremes.
        """
        if freq < self.freq_min_hz or freq > self.freq_max_hz:
            return 0.0
        
        # Log-distance from resonance
        log_ratio = abs(np.log2(freq / self.freq_resonance_hz))
        
        # Gaussian-like falloff
        sigma = 1.5  # octaves
        return np.exp(-(log_ratio ** 2) / (2 * sigma ** 2))
    
    def max_plate_area_m2(self, material_efficiency: float = 0.5) -> float:
        """
        Estimate maximum effective plate area this exciter can drive.
        
        Rule of thumb: ~1W RMS per 0.05 m² for wood plates.
        
        Args:
            material_efficiency: 0-1 factor for material coupling
        """
        # Base: 1W drives 0.05 m² of medium-density wood
        base_area_per_watt = 0.05 * material_efficiency
        return self.power_rms_w * base_area_per_watt
    
    def is_suitable_for(self, 
                        freq_target: float,
                        plate_area_m2: float,
                        efficiency_threshold: float = 0.3) -> bool:
        """
        Check if this exciter is suitable for given frequency and plate size.
        """
        freq_ok = self.freq_min_hz <= freq_target <= self.freq_max_hz
        power_ok = plate_area_m2 <= self.max_plate_area_m2()
        efficiency_ok = self.efficiency_at_frequency(freq_target) >= efficiency_threshold
        
        return freq_ok and power_ok and efficiency_ok


# ══════════════════════════════════════════════════════════════════════════════
# EXCITER DATABASE
# ══════════════════════════════════════════════════════════════════════════════

EXCITERS: Dict[str, Exciter] = {
    
    # ─────────────────────────────────────────────────────────────────────────
    # VISATON - German manufacturer, high quality
    # ─────────────────────────────────────────────────────────────────────────
    
    "visaton_ex_45_s": Exciter(
        name="Visaton EX 45 S",
        manufacturer="Visaton",
        model="EX 45 S",
        power_rms_w=10,
        power_peak_w=15,
        impedance_ohm=8,
        freq_min_hz=100,
        freq_max_hz=20000,
        freq_resonance_hz=220,
        diameter_mm=45,
        height_mm=13,
        weight_g=35,
        exciter_type=ExciterType.VOICE_COIL,
        notes="Compact, good for small plates"
    ),
    
    "visaton_ex_60_s": Exciter(
        name="Visaton EX 60 S", 
        manufacturer="Visaton",
        model="EX 60 S",
        power_rms_w=25,
        power_peak_w=40,
        impedance_ohm=8,
        freq_min_hz=80,
        freq_max_hz=20000,
        freq_resonance_hz=170,
        diameter_mm=60,
        height_mm=20,
        weight_g=82,
        exciter_type=ExciterType.VOICE_COIL,
        notes="Medium size, versatile"
    ),
    
    "visaton_ex_80_s": Exciter(
        name="Visaton EX 80 S",
        manufacturer="Visaton", 
        model="EX 80 S",
        power_rms_w=30,
        power_peak_w=50,
        impedance_ohm=8,
        freq_min_hz=50,
        freq_max_hz=20000,
        freq_resonance_hz=120,
        diameter_mm=80,
        height_mm=24,
        weight_g=150,
        exciter_type=ExciterType.VOICE_COIL,
        bl_factor=3.2,
        notes="Good bass, suitable for body-sized plates"
    ),
    
    "visaton_ex_60_r": Exciter(
        name="Visaton EX 60 R",
        manufacturer="Visaton",
        model="EX 60 R",
        power_rms_w=25,
        power_peak_w=40,
        impedance_ohm=4,
        freq_min_hz=80,
        freq_max_hz=20000,
        freq_resonance_hz=180,
        diameter_mm=60,
        height_mm=20,
        weight_g=80,
        exciter_type=ExciterType.VOICE_COIL,
        notes="4 ohm version, higher current"
    ),
    
    "visaton_bs_76": Exciter(
        name="Visaton BS 76",
        manufacturer="Visaton",
        model="BS 76",
        power_rms_w=10,
        power_peak_w=20,
        impedance_ohm=4,
        freq_min_hz=40,
        freq_max_hz=500,
        freq_resonance_hz=50,
        diameter_mm=76,
        height_mm=36,
        weight_g=280,
        exciter_type=ExciterType.BASS_SHAKER,
        notes="Bass shaker, low frequency focus"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # DAYTON AUDIO - American manufacturer, good value
    # ─────────────────────────────────────────────────────────────────────────
    
    "dayton_daex25": Exciter(
        name="Dayton DAEX25",
        manufacturer="Dayton Audio",
        model="DAEX25",
        power_rms_w=10,
        power_peak_w=20,
        impedance_ohm=4,
        freq_min_hz=300,
        freq_max_hz=20000,
        freq_resonance_hz=800,
        diameter_mm=25,
        height_mm=7,
        weight_g=12,
        exciter_type=ExciterType.VOICE_COIL,
        notes="Tiny, for high frequencies"
    ),
    
    "dayton_daex32": Exciter(
        name="Dayton DAEX32",
        manufacturer="Dayton Audio",
        model="DAEX32",
        power_rms_w=20,
        power_peak_w=40,
        impedance_ohm=4,
        freq_min_hz=200,
        freq_max_hz=20000,
        freq_resonance_hz=450,
        diameter_mm=32,
        height_mm=12,
        weight_g=30,
        exciter_type=ExciterType.VOICE_COIL,
        notes="Small, good mid-high"
    ),
    
    "dayton_daex58fp": Exciter(
        name="Dayton DAEX58FP",
        manufacturer="Dayton Audio", 
        model="DAEX58FP",
        power_rms_w=25,
        power_peak_w=50,
        impedance_ohm=4,
        freq_min_hz=100,
        freq_max_hz=20000,
        freq_resonance_hz=280,
        diameter_mm=58,
        height_mm=16,
        weight_g=85,
        exciter_type=ExciterType.VOICE_COIL,
        notes="Flat puck design"
    ),
    
    "dayton_bst_1": Exciter(
        name="Dayton BST-1",
        manufacturer="Dayton Audio",
        model="BST-1",
        power_rms_w=30,
        power_peak_w=60,
        impedance_ohm=4,
        freq_min_hz=20,
        freq_max_hz=200,
        freq_resonance_hz=40,
        diameter_mm=89,
        height_mm=46,
        weight_g=500,
        exciter_type=ExciterType.BASS_SHAKER,
        notes="Heavy bass shaker, needs sturdy mounting"
    ),
    
    "dayton_tt25-8": Exciter(
        name="Dayton TT25-8",
        manufacturer="Dayton Audio",
        model="TT25-8",
        power_rms_w=10,
        power_peak_w=15,
        impedance_ohm=8,
        freq_min_hz=250,
        freq_max_hz=20000,
        freq_resonance_hz=500,
        diameter_mm=25,
        height_mm=7,
        weight_g=12,
        exciter_type=ExciterType.VOICE_COIL,
        notes="8 ohm version of DAEX25"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # TACTILE TRANSDUCERS / BASS SHAKERS
    # ─────────────────────────────────────────────────────────────────────────
    
    "aura_ast_2b_4": Exciter(
        name="Aura AST-2B-4",
        manufacturer="Aura Sound",
        model="AST-2B-4",
        power_rms_w=25,
        power_peak_w=50,
        impedance_ohm=4,
        freq_min_hz=40,
        freq_max_hz=200,
        freq_resonance_hz=60,
        diameter_mm=51,
        height_mm=25,
        weight_g=180,
        exciter_type=ExciterType.TACTILE,
        notes="Tactile transducer, good for body resonance"
    ),
    
    "reckhorn_bs_200": Exciter(
        name="Reckhorn BS-200",
        manufacturer="Reckhorn",
        model="BS-200",
        power_rms_w=100,
        power_peak_w=200,
        impedance_ohm=4,
        freq_min_hz=15,
        freq_max_hz=150,
        freq_resonance_hz=30,
        diameter_mm=120,
        height_mm=65,
        weight_g=1800,
        exciter_type=ExciterType.BASS_SHAKER,
        notes="Professional grade, very powerful"
    ),
    
    "clark_synthesis_tst329": Exciter(
        name="Clark Synthesis TST 329",
        manufacturer="Clark Synthesis",
        model="TST 329",
        power_rms_w=50,
        power_peak_w=100,
        impedance_ohm=4,
        freq_min_hz=5,
        freq_max_hz=17000,
        freq_resonance_hz=25,
        diameter_mm=100,
        height_mm=55,
        weight_g=900,
        exciter_type=ExciterType.TACTILE,
        notes="Wide range, premium quality"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def find_exciters_for_frequency(freq_hz: float, 
                                 min_power_w: float = 10) -> List[Exciter]:
    """
    Find exciters suitable for a target frequency.
    """
    suitable = []
    for exciter in EXCITERS.values():
        if (exciter.freq_min_hz <= freq_hz <= exciter.freq_max_hz and
            exciter.power_rms_w >= min_power_w):
            suitable.append(exciter)
    
    # Sort by efficiency at target frequency
    suitable.sort(key=lambda e: e.efficiency_at_frequency(freq_hz), reverse=True)
    return suitable


def find_exciters_for_plate(plate_area_m2: float,
                            freq_range: Tuple[float, float] = (20, 500)) -> List[Exciter]:
    """
    Find exciters suitable for a plate of given area and frequency range.
    """
    suitable = []
    freq_min, freq_max = freq_range
    
    for exciter in EXCITERS.values():
        # Check if exciter can drive the plate area
        if exciter.max_plate_area_m2() >= plate_area_m2:
            # Check frequency overlap
            if (exciter.freq_min_hz <= freq_max and 
                exciter.freq_max_hz >= freq_min):
                suitable.append(exciter)
    
    # Sort by power
    suitable.sort(key=lambda e: e.power_rms_w, reverse=True)
    return suitable


def recommend_exciter_config(plate_area_m2: float,
                             target_frequencies: List[float],
                             single_exciter: bool = False) -> Dict:
    """
    Recommend exciter configuration for a vibroacoustic plate.
    
    Returns dict with:
        - exciters: list of recommended Exciter objects
        - positions: suggested mounting positions (normalized)
        - total_power: combined RMS power
        - coverage: frequency coverage analysis
    """
    if single_exciter:
        # Find single exciter that covers most frequencies
        best_exciter = None
        best_coverage = 0
        
        for exciter in EXCITERS.values():
            if exciter.max_plate_area_m2() >= plate_area_m2:
                coverage = sum(
                    1 for f in target_frequencies
                    if exciter.freq_min_hz <= f <= exciter.freq_max_hz
                ) / len(target_frequencies)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_exciter = exciter
        
        if best_exciter:
            return {
                "exciters": [best_exciter],
                "positions": [(0.5, 0.5)],  # Center
                "total_power": best_exciter.power_rms_w,
                "coverage": best_coverage,
                "notes": "Single exciter at plate center"
            }
    else:
        # Multi-exciter setup
        exciters = []
        positions = []
        
        # Group frequencies into bass, mid, high
        bass_freqs = [f for f in target_frequencies if f < 100]
        mid_freqs = [f for f in target_frequencies if 100 <= f < 1000]
        high_freqs = [f for f in target_frequencies if f >= 1000]
        
        # Find exciters for each range
        if bass_freqs:
            bass_exciters = find_exciters_for_frequency(min(bass_freqs), min_power_w=20)
            if bass_exciters:
                exciters.append(bass_exciters[0])
                positions.append((0.5, 0.7))  # Toward one end
        
        if mid_freqs:
            mid_exciters = find_exciters_for_frequency(np.mean(mid_freqs))
            if mid_exciters:
                exciters.append(mid_exciters[0])
                positions.append((0.5, 0.5))  # Center
        
        if high_freqs:
            high_exciters = find_exciters_for_frequency(min(high_freqs))
            if high_exciters:
                exciters.append(high_exciters[0])
                positions.append((0.5, 0.3))  # Toward other end
        
        if exciters:
            return {
                "exciters": exciters,
                "positions": positions,
                "total_power": sum(e.power_rms_w for e in exciters),
                "coverage": len([e for e in exciters if e]) / 3,
                "notes": f"Multi-exciter setup: {len(exciters)} units"
            }
    
    return {
        "exciters": [],
        "positions": [],
        "total_power": 0,
        "coverage": 0,
        "notes": "No suitable exciters found"
    }


def get_exciter_by_name(name: str) -> Optional[Exciter]:
    """Get exciter by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for key, exciter in EXCITERS.items():
        if name_lower in key.lower() or name_lower in exciter.name.lower():
            return exciter
    return None


def list_visaton_exciters() -> List[Exciter]:
    """Get all Visaton exciters."""
    return [e for e in EXCITERS.values() if e.manufacturer == "Visaton"]


def list_bass_shakers() -> List[Exciter]:
    """Get all bass shakers."""
    return [e for e in EXCITERS.values() if e.exciter_type == ExciterType.BASS_SHAKER]


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("EXCITER DATABASE")
    print("=" * 60)
    
    print("\n🔊 VISATON EXCITERS:")
    for exciter in list_visaton_exciters():
        print(f"   {exciter.name}")
        print(f"      Power: {exciter.power_rms_w}W RMS / {exciter.power_peak_w}W peak")
        print(f"      Freq: {exciter.freq_min_hz:.0f} - {exciter.freq_max_hz:.0f} Hz")
        print(f"      Size: Ø{exciter.diameter_mm}mm × {exciter.height_mm}mm")
        print()
    
    print("\n📐 RECOMMENDATION FOR 1.7m × 0.5m PLATE (chakra frequencies):")
    plate_area = 1.7 * 0.5
    chakra_freqs = [256, 288, 320, 341.3, 384, 426.7, 480]
    
    config = recommend_exciter_config(plate_area, chakra_freqs)
    print(f"   Exciters: {[e.name for e in config['exciters']]}")
    print(f"   Total power: {config['total_power']}W")
    print(f"   Coverage: {config['coverage']*100:.0f}%")
