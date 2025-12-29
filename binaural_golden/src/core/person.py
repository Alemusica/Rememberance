"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PERSON - Anthropometric Model                        ║
║                                                                              ║
║   Modello antropometrico di persona distesa su tavola vibroacustica.         ║
║                                                                              ║
║   Riferimenti:                                                               ║
║   • NASA-STD-3001 Vol.2 - Anthropometry                                      ║
║   • Griffin (1990) - Handbook of Human Vibration                             ║
║   • Dreyfuss - The Measure of Man                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class BodySegment:
    """Segmento corporeo con massa e posizione."""
    name: str
    mass_fraction: float      # Frazione della massa totale
    position_start: float     # Posizione inizio (0=piedi, 1=testa)
    position_end: float       # Posizione fine
    width_fraction: float     # Larghezza relativa alle spalle
    contact_fraction: float   # Frazione di area in contatto con tavola
    
    @property
    def center_position(self) -> float:
        """Posizione centro del segmento."""
        return (self.position_start + self.position_end) / 2
    
    @property
    def length_fraction(self) -> float:
        """Frazione della lunghezza totale."""
        return self.position_end - self.position_start


# Segmenti corporei standard (persona distesa supina)
BODY_SEGMENTS: Dict[str, BodySegment] = {
    "feet": BodySegment("feet", 0.03, 0.00, 0.05, 0.25, 0.8),
    "lower_legs": BodySegment("lower_legs", 0.09, 0.05, 0.25, 0.30, 0.9),
    "upper_legs": BodySegment("upper_legs", 0.20, 0.25, 0.45, 0.40, 0.95),
    "pelvis": BodySegment("pelvis", 0.15, 0.45, 0.55, 0.55, 1.0),
    "abdomen": BodySegment("abdomen", 0.10, 0.55, 0.65, 0.45, 0.7),
    "chest": BodySegment("chest", 0.20, 0.65, 0.80, 0.60, 0.8),
    "shoulders": BodySegment("shoulders", 0.08, 0.80, 0.87, 1.00, 0.9),
    "neck": BodySegment("neck", 0.03, 0.87, 0.92, 0.30, 0.6),
    "head": BodySegment("head", 0.08, 0.92, 1.00, 0.35, 0.5),
    "arms": BodySegment("arms", 0.04, 0.50, 0.85, 1.20, 0.7),  # Braccia lungo i fianchi
}

# Zone della spina dorsale
SPINE_ZONES: Dict[str, Tuple[float, float]] = {
    "lumbar": (0.45, 0.55),      # Lombare (L1-L5)
    "thoracic": (0.55, 0.75),    # Toracica (T1-T12)
    "cervical": (0.80, 0.92),    # Cervicale (C1-C7)
}


@dataclass
class Person:
    """
    Modello antropometrico di persona per ottimizzazione tavola.
    
    Coordinate: 
    - x: lungo il corpo (0=piedi, 1=testa normalizzato)
    - y: trasversale (0=sinistra, 1=destra, 0.5=centro)
    
    Attributes:
        height_m: Altezza in metri (1.50 - 2.10)
        weight_kg: Peso in kg (45 - 120)
    """
    height_m: float = 1.75
    weight_kg: float = 70.0
    
    # Parametri opzionali per personalizzazione
    shoulder_width_ratio: float = 0.26  # Spalle / altezza
    
    def __post_init__(self):
        """Valida parametri."""
        if not 1.40 <= self.height_m <= 2.20:
            raise ValueError(f"Altezza {self.height_m}m fuori range [1.40, 2.20]")
        if not 35.0 <= self.weight_kg <= 150.0:
            raise ValueError(f"Peso {self.weight_kg}kg fuori range [35, 150]")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Proprietà Geometriche
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def lying_length(self) -> float:
        """Lunghezza da disteso (≈ altezza)."""
        return self.height_m
    
    @property
    def shoulder_width(self) -> float:
        """Larghezza spalle [m]."""
        return self.height_m * self.shoulder_width_ratio
    
    @property
    def recommended_plate_length(self) -> float:
        """Lunghezza tavola raccomandata [m]."""
        return self.height_m + 0.10  # 10cm margine
    
    @property
    def recommended_plate_width(self) -> float:
        """Larghezza tavola raccomandata [m]."""
        # Larghezza = spalle + braccia + margine
        return self.shoulder_width * 1.4
    
    @property
    def navel_position(self) -> float:
        """Posizione ombelico (golden ratio)."""
        return 1.0 / PHI  # ≈ 0.618 dalla testa, 0.382 dai piedi
    
    # ─────────────────────────────────────────────────────────────────────────
    # Distribuzione Massa
    # ─────────────────────────────────────────────────────────────────────────
    
    def segment_mass(self, segment_name: str) -> float:
        """Massa di un segmento corporeo [kg]."""
        if segment_name not in BODY_SEGMENTS:
            raise ValueError(f"Segmento sconosciuto: {segment_name}")
        return self.weight_kg * BODY_SEGMENTS[segment_name].mass_fraction
    
    def mass_at_position(self, x_norm: float) -> float:
        """
        Densità di massa lineare a posizione x [kg/m].
        
        Args:
            x_norm: Posizione normalizzata (0=piedi, 1=testa)
        
        Returns:
            Densità massa lineare [kg/m]
        """
        density = 0.0
        for segment in BODY_SEGMENTS.values():
            if segment.position_start <= x_norm <= segment.position_end:
                # Distribuzione uniforme nel segmento
                segment_length = (segment.position_end - segment.position_start) * self.lying_length
                segment_mass = self.weight_kg * segment.mass_fraction
                density += segment_mass / segment_length
        return density
    
    def mass_distribution_grid(self, nx: int, ny: int) -> np.ndarray:
        """
        Griglia 2D di distribuzione massa per FEM [kg/m²].
        
        Args:
            nx: Numero celle lungo x (lunghezza)
            ny: Numero celle lungo y (larghezza)
        
        Returns:
            Array (nx, ny) con densità superficiale massa
        """
        mass_grid = np.zeros((nx, ny))
        
        plate_length = self.recommended_plate_length
        plate_width = self.recommended_plate_width
        
        dx = plate_length / nx
        dy = plate_width / ny
        cell_area = dx * dy
        
        for i in range(nx):
            x_norm = (i + 0.5) / nx
            
            for j in range(ny):
                y_norm = (j + 0.5) / ny
                
                # Trova segmento a questa posizione
                for segment in BODY_SEGMENTS.values():
                    if segment.position_start <= x_norm <= segment.position_end:
                        # Check se dentro la larghezza del segmento
                        segment_half_width = segment.width_fraction * 0.5
                        center_y = 0.5
                        
                        if abs(y_norm - center_y) <= segment_half_width:
                            # Massa distribuita in questo segmento
                            segment_length = segment.position_end - segment.position_start
                            segment_width = segment.width_fraction
                            segment_area = segment_length * segment_width
                            
                            segment_mass = self.weight_kg * segment.mass_fraction
                            segment_contact = segment.contact_fraction
                            
                            # Densità superficiale [kg/m²]
                            if segment_area > 0:
                                mass_grid[i, j] += (
                                    segment_mass * segment_contact / 
                                    (segment_area * plate_length * plate_width)
                                )
        
        return mass_grid
    
    # ─────────────────────────────────────────────────────────────────────────
    # Zona Spina Dorsale
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def spine_positions(self) -> Dict[str, Tuple[float, float]]:
        """Posizioni zone spina dorsale [m dal piede]."""
        return {
            name: (
                bounds[0] * self.lying_length,
                bounds[1] * self.lying_length
            )
            for name, bounds in SPINE_ZONES.items()
        }
    
    def spine_contact_line(self, n_points: int = 50) -> np.ndarray:
        """
        Linea di contatto spina dorsale.
        
        Returns:
            Array (n_points, 2) con coordinate (x, y) normalizzate
        """
        # Spina al centro del corpo (y=0.5)
        spine_start = 0.45  # Inizia al sacro
        spine_end = 0.92    # Finisce alla base cranio
        
        x = np.linspace(spine_start, spine_end, n_points)
        y = np.full(n_points, 0.5)  # Centro
        
        return np.column_stack([x, y])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Risonanze Corporee
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def body_resonance_frequencies(self) -> Dict[str, float]:
        """
        Frequenze di risonanza corporea [Hz].
        
        Basato su Griffin (1990) - scala con peso/rigidezza.
        """
        # Frequenze base per persona 70kg
        base_freqs = {
            "whole_body": 5.0,       # Traslazione verticale
            "spine_axial": 10.0,     # Compressione assiale
            "spine_bending": 8.0,    # Flessione spinale
            "abdomen": 6.0,          # Cavità addominale
            "chest": 55.0,           # Parete toracica
            "head": 25.0,            # Testa su collo
        }
        
        # Scala con massa (f ∝ √(k/m), k ∝ m^0.7 circa)
        mass_ratio = self.weight_kg / 70.0
        scale_factor = mass_ratio ** (-0.15)  # Leggera diminuzione con peso
        
        return {name: freq * scale_factor for name, freq in base_freqs.items()}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Serializzazione
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict:
        """Converte in dizionario per JSON."""
        return {
            "height_m": self.height_m,
            "weight_kg": self.weight_kg,
            "shoulder_width_ratio": self.shoulder_width_ratio,
            "lying_length": self.lying_length,
            "shoulder_width": self.shoulder_width,
            "recommended_plate": {
                "length": self.recommended_plate_length,
                "width": self.recommended_plate_width,
            },
            "body_resonances": self.body_resonance_frequencies,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Person':
        """Crea da dizionario."""
        return cls(
            height_m=data["height_m"],
            weight_kg=data["weight_kg"],
            shoulder_width_ratio=data.get("shoulder_width_ratio", 0.26),
        )
    
    def __repr__(self) -> str:
        return (
            f"Person(height={self.height_m:.2f}m, weight={self.weight_kg:.1f}kg, "
            f"plate={self.recommended_plate_length:.2f}×{self.recommended_plate_width:.2f}m)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRESET PERSONE
# ══════════════════════════════════════════════════════════════════════════════

PERSON_PRESETS: Dict[str, Person] = {
    "child": Person(height_m=1.40, weight_kg=40.0),
    "small_adult": Person(height_m=1.55, weight_kg=50.0),
    "average_female": Person(height_m=1.65, weight_kg=60.0),
    "average_male": Person(height_m=1.75, weight_kg=75.0),
    "tall_adult": Person(height_m=1.90, weight_kg=90.0),
    "large_adult": Person(height_m=1.85, weight_kg=110.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PERSON MODEL TEST")
    print("=" * 60)
    
    person = Person(height_m=1.75, weight_kg=75.0)
    print(f"\n{person}")
    print(f"\nNavel position: {person.navel_position:.3f} (golden ratio)")
    print(f"Spine positions: {person.spine_positions}")
    print(f"\nBody resonances:")
    for name, freq in person.body_resonance_frequencies.items():
        print(f"  {name}: {freq:.1f} Hz")
    
    print("\nMass distribution grid (10x6):")
    grid = person.mass_distribution_grid(10, 6)
    print(f"  Shape: {grid.shape}")
    print(f"  Total mass on grid: {grid.sum():.1f} kg/m² (normalized)")
    
    print("\nPresets:")
    for name, p in PERSON_PRESETS.items():
        print(f"  {name}: {p}")
