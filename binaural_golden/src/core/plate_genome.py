"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE GENOME - Genetic Plate Representation               ║
║                                                                              ║
║   Codifica genetica della forma tavola per ottimizzazione evolutiva.         ║
║                                                                              ║
║   La tavola è rappresentata come:                                            ║
║   • Contorno: spline con N punti di controllo                               ║
║   • Spessore: campo scalare opzionale (nx × ny)                             ║
║   • Tagli: lista di cutout (cerchi, ellissi)                                ║
║                                                                              ║
║   Operatori genetici:                                                        ║
║   • mutate(): rumore gaussiano sui parametri                                 ║
║   • crossover(): blend tra due genomi                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import copy
import logging

logger = logging.getLogger(__name__)

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


class ContourType(Enum):
    """Tipi di contorno base."""
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    GOLDEN_RECT = "golden_rectangle"
    OVOID = "ovoid"
    VESICA_PISCIS = "vesica_piscis"  # Sacred geometry: two overlapping circles
    SUPERELLIPSE = "superellipse"    # Squircle (rounded rectangle smooth)
    ORGANIC = "organic"              # Blob-like organic shape (Fourier)
    ERGONOMIC = "ergonomic"          # Body-conforming shape
    FREEFORM = "freeform"            # Fully evolvable spline
    PHI_ROUNDED = "phi_rounded"      # Rectangle with PHI-based rounded corners


# Cutout shapes available for optimization
# L'optimizer può usare qualsiasi forma, incluso FREEFORM con poligoni arbitrari
# Tutte le forme sono lavorabili con CNC/fresa
# Cutout shapes - ONLY ORGANIC/CURVED shapes
# Nature doesn't make straight lines! Good acoustic design follows nature.
# Reference: Stradivari f-holes, Schleske violin acoustics
CUTOUT_SHAPES = [
    # === CNC-REALISTIC SHAPES (lavorabili con fresa) ===
    "slot",         # Scanalatura dritta (2-8mm width, length variabile)
    "arc_slot",     # Scanalatura ad arco (raggio + angolo)
    "circle",       # Cerchio (fresa a tuffo o elicoidale)
    "ellipse",      # Ellisse (percorso fresa)
    "f_hole",       # F-hole stilizzato (percorso CNC)
    # === FORME ORGANICHE (richiede fresa manuale) ===
    "crescent",     # Mezzaluna
    "tear",         # Goccia
    "kidney",       # Forma a rene
    "s_curve",      # Curva a S
    "vesica",       # Vesica piscis
    "spiral",       # Spirale logaritmica
    "leaf",         # Forma a foglia
    "wave",         # Onda sinusoidale
    "freeform",     # Poligono arbitrario
]

# CNC tool diameters in mm (standard end mills)
CNC_TOOL_DIAMETERS_MM = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


@dataclass
class CutoutGene:
    """
    Gene per un taglio interno (foro passante) CNC-compatibile.
    
    VINCOLI CNC:
    - width: deve corrispondere al diametro fresa (2-8mm) o essere maggiore
    - Per SLOT: width = diametro fresa, height = lunghezza slot
    - Per ARC_SLOT: segue un arco, width = diametro fresa
    
    Per forme organiche/freeform richiede fresa manuale.
    
    Posizione e dimensioni normalizzate [0, 1].
    """
    x: float              # Centro X normalizzato (0.05-0.95 per stare dentro i bordi)
    y: float              # Centro Y normalizzato (0.05-0.95)
    width: float          # Larghezza normalizzata - per SLOT = tool width
    height: float         # Altezza normalizzata - per SLOT = lunghezza slot
    rotation: float = 0.0 # Rotazione in radianti
    shape: str = "slot"   # Default: slot CNC (forma più realistica)
    corner_radius: float = 0.3  # Per forme arrotondate
    aspect_bias: float = 1.0    # Bias aspetto
    # CNC-specific parameters
    tool_diameter_mm: float = 6.0  # Diametro fresa (standard 6mm)
    arc_radius: float = 0.0   # Per arc_slot: raggio arco normalizzato
    arc_angle: float = 0.0    # Per arc_slot: angolo arco in radianti
    # FREEFORM: punti di controllo per poligono arbitrario (fresa manuale)
    control_points: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize control points for freeform shape."""
        if self.shape == "freeform" and self.control_points is None:
            # Genera poligono random con 5-8 vertici
            n_points = np.random.randint(5, 9)
            angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
            radii = np.random.uniform(0.3, 1.0, n_points)
            self.control_points = np.column_stack([
                radii * np.cos(angles),
                radii * np.sin(angles)
            ])
        
        # Ensure position is within valid bounds
        self.x = np.clip(self.x, 0.08, 0.92)
        self.y = np.clip(self.y, 0.08, 0.92)
    
    def mutate(self, sigma: float = 0.05) -> 'CutoutGene':
        """
        Muta questo cutout mantenendo vincoli CNC-realistici.
        
        Per SLOT/ARC_SLOT:
        - Width resta fisso (= diametro fresa)
        - Length (height) può variare
        - Posizione si muove seguendo modalità
        """
        new_shape = self.shape
        
        # 10% chance di cambiare forma, ma preferisci forme CNC
        if np.random.random() < 0.10:
            # 80% probabilità forme CNC-native
            if np.random.random() < 0.80:
                new_shape = np.random.choice(["slot", "arc_slot", "circle", "ellipse"])
            else:
                new_shape = np.random.choice(CUTOUT_SHAPES)
        
        # Per SLOT: width è fisso (tool diameter), muta solo height (length)
        # CRITICAL: Max height 0.12 (12%) to prevent plate-dividing cuts!
        # On 670mm plate: 0.12 = 80mm max slot length (safe for structural integrity)
        if new_shape in ["slot", "arc_slot"]:
            new_width = self.width  # Keep tool width constant
            new_height = np.clip(self.height + np.random.normal(0, sigma), 0.03, 0.12)  # Max 12%!
            new_tool_diameter = self.tool_diameter_mm  # Keep tool
        else:
            new_width = np.clip(self.width + np.random.normal(0, sigma * 0.5), 0.01, 0.10)
            new_height = np.clip(self.height + np.random.normal(0, sigma * 0.5), 0.01, 0.10)  # Max 10%
            new_tool_diameter = self.tool_diameter_mm
        
        # Muta arc parameters
        new_arc_radius = self.arc_radius
        new_arc_angle = self.arc_angle
        if new_shape == "arc_slot":
            new_arc_radius = np.clip(self.arc_radius + np.random.normal(0, sigma), 0.05, 0.3)
            new_arc_angle = np.clip(self.arc_angle + np.random.normal(0, 0.3), 0.3, np.pi)
        
        # Muta control points per freeform
        new_control_points = None
        if new_shape == "freeform":
            if self.control_points is not None:
                noise = np.random.normal(0, sigma * 0.5, self.control_points.shape)
                new_control_points = np.clip(self.control_points + noise, -1.0, 1.0)
            else:
                n_points = np.random.randint(5, 9)
                angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
                radii = np.random.uniform(0.3, 1.0, n_points)
                new_control_points = np.column_stack([
                    radii * np.cos(angles),
                    radii * np.sin(angles)
                ])
        
        return CutoutGene(
            x=np.clip(self.x + np.random.normal(0, sigma), 0.08, 0.92),
            y=np.clip(self.y + np.random.normal(0, sigma), 0.08, 0.92),
            width=new_width,
            height=new_height,
            rotation=(self.rotation + np.random.normal(0, 0.2)) % (2 * np.pi),
            shape=new_shape,
            corner_radius=np.clip(self.corner_radius + np.random.normal(0, 0.1), 0.0, 1.0),
            aspect_bias=np.clip(self.aspect_bias + np.random.normal(0, 0.2), 0.3, 3.0),
            tool_diameter_mm=new_tool_diameter,
            arc_radius=new_arc_radius,
            arc_angle=new_arc_angle,
            control_points=new_control_points,
        )
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """
        Converti in coordinate assolute [m].
        
        COORDINATE CONVENTION (aligned with ExciterPosition and person.py):
            x: lateral position (0=left edge, 1=right edge) → maps to plate_width
            y: longitudinal position (0=feet, 1=head) → maps to plate_length
        
        FEM CONVENTION:
            FEM x-axis = plate length (longitudinal)
            FEM y-axis = plate width (lateral)
        
        So we swap: (self.y * plate_length, self.x * plate_width)
        """
        return {
            "center": (self.y * plate_length, self.x * plate_width),  # SWAPPED: y→length, x→width
            "size": (self.height * plate_length, self.width * plate_width),  # SWAPPED for consistency
            "rotation": self.rotation,
            "shape": self.shape,
            "tool_diameter_mm": self.tool_diameter_mm,
            "arc_radius": self.arc_radius * min(plate_length, plate_width) if self.arc_radius > 0 else 0,
            "arc_angle": self.arc_angle,
        }


@dataclass
class GrooveGene:
    """
    Gene per una scanalatura di accordatura (LUTHERIE).
    
    Le scanalature sono incisioni superficiali (non passanti) che:
    - Riducono localmente la rigidezza flessionale
    - Permettono accordatura fine dei modi propri
    - Come il "thinning" delle tavole armoniche di violino
    
    Reference: Schleske (2002) - violin plate graduations
    
    Attributes:
        x, y: Posizione centro (normalizzata 0-1)
        length: Lunghezza scanalatura (normalizzata)
        angle: Angolo in radianti (0 = orizzontale)
        depth: Profondità come frazione dello spessore (0.1-0.5)
        width: Larghezza scanalatura in mm (2-10mm tipico)
    """
    x: float              # Centro X normalizzato
    y: float              # Centro Y normalizzato  
    length: float         # Lunghezza normalizzata (0.05-0.3)
    angle: float = 0.0    # Radianti (0 = parallel to width)
    depth: float = 0.3    # Frazione spessore (30% = 4.5mm su 15mm)
    width_mm: float = 5.0 # Larghezza in mm (fresa standard)
    
    def mutate(self, sigma: float = 0.03) -> 'GrooveGene':
        """Muta questa scanalatura."""
        return GrooveGene(
            x=np.clip(self.x + np.random.normal(0, sigma), 0.1, 0.9),
            y=np.clip(self.y + np.random.normal(0, sigma), 0.1, 0.9),
            length=np.clip(self.length + np.random.normal(0, sigma), 0.03, 0.3),
            angle=self.angle + np.random.normal(0, 0.3),
            depth=np.clip(self.depth + np.random.normal(0, 0.05), 0.1, 0.5),
            width_mm=np.clip(self.width_mm + np.random.normal(0, 0.5), 2.0, 10.0),
        )
    
    def to_absolute(self, plate_length: float, plate_width: float, thickness: float) -> Dict:
        """
        Converti in coordinate assolute.
        
        COORDINATE CONVENTION (aligned with ExciterPosition and person.py):
            x: lateral position (0=left, 1=right) → maps to plate_width
            y: longitudinal position (0=feet, 1=head) → maps to plate_length
        """
        return {
            "center": (self.y * plate_length, self.x * plate_width),  # SWAPPED: y→length, x→width
            "length": self.length * plate_length,
            "angle": self.angle,
            "depth": self.depth * thickness,
            "width": self.width_mm / 1000,  # Convert to meters
        }
    
    def stiffness_reduction(self) -> float:
        """
        Calcola riduzione rigidezza locale (0-1).
        
        Per trave rettangolare: I ∝ h³
        Groove riduce h localmente → riduce I
        """
        # Remaining thickness fraction
        h_remaining = 1.0 - self.depth
        # Stiffness scales with h³
        stiffness_remaining = h_remaining ** 3
        return 1.0 - stiffness_remaining  # Reduction factor


@dataclass
class AttachedMassGene:
    """
    Gene per una massa attaccata (tuning modale passivo).
    
    RESEARCH BASIS (Lu 2012, Shen 2016, Zhang 2006):
    - Masse attaccate modificano distribuzione modale SENZA exciters aggiuntivi
    - Posizionate a 1/3 e 2/3 della tavola migliorano flatness
    - Effetto: abbassano frequenze dei modi dove sono posizionate
    - Costo zero in amplificazione (no channel occupato)
    
    FISICA:
    - Massa aggiunta m in posizione (x,y) modifica frequenza modale:
      f'_n ≈ f_n / sqrt(1 + m * φ_n²(x,y) / M_eff)
    - dove φ_n è la forma modale normalizzata, M_eff massa modale effettiva
    - Massa vicino a antinodo → massimo shift frequenza
    - Massa vicino a nodo → effetto minimo
    
    IMPLEMENTAZIONE CNC:
    - Foro filettato M6/M8 + dado/bullone con rondella
    - Oppure: tasca fresata per inserimento massa calibrata
    - Materiali: ottone (denso), acciaio, piombo (max densità)
    
    Attributes:
        x, y: Posizione centro (normalizzata 0-1)
        mass_kg: Massa in kg (0.01-0.5 tipico)
        material: Materiale ('brass', 'steel', 'lead')
        mount_type: Tipo montaggio ('bolt', 'pocket', 'adhesive')
    """
    x: float                    # Centro X normalizzato
    y: float                    # Centro Y normalizzato
    mass_kg: float = 0.05       # 50g default
    material: str = "brass"     # 'brass', 'steel', 'lead'
    mount_type: str = "bolt"    # 'bolt', 'pocket', 'adhesive'
    diameter_mm: float = 20.0   # Diametro massa (per CNC)
    
    # Densità materiali [kg/m³]
    DENSITIES = {
        "brass": 8500,
        "steel": 7850,
        "lead": 11340,
        "aluminum": 2700,
    }
    
    def __post_init__(self):
        """Calcola diametro da massa e materiale se non specificato."""
        if self.diameter_mm == 20.0:  # Default, calcola da massa
            density = self.DENSITIES.get(self.material, 8500)
            # Volume = massa / densità, assumendo cilindro h = d/2
            # V = π * r² * h = π * r² * r = π * r³
            # m = ρ * V = ρ * π * r³
            # r = (m / (ρ * π))^(1/3)
            r_m = (self.mass_kg / (density * np.pi)) ** (1/3)
            self.diameter_mm = r_m * 2000  # Convert to mm
    
    def mutate(self, sigma: float = 0.04) -> 'AttachedMassGene':
        """Muta questa massa attaccata."""
        # Possibilità di cambiare materiale
        new_material = self.material
        if np.random.random() < 0.1:  # 10% chance
            new_material = np.random.choice(["brass", "steel", "lead"])
        
        new_mass = np.clip(self.mass_kg + np.random.normal(0, 0.02), 0.01, 0.5)
        
        return AttachedMassGene(
            x=np.clip(self.x + np.random.normal(0, sigma), 0.1, 0.9),
            y=np.clip(self.y + np.random.normal(0, sigma), 0.1, 0.9),
            mass_kg=new_mass,
            material=new_material,
            mount_type=self.mount_type,
        )
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """
        Converti in coordinate assolute [m].
        
        COORDINATE CONVENTION (aligned with ExciterPosition and person.py):
            x: lateral position (0=left, 1=right) → maps to plate_width
            y: longitudinal position (0=feet, 1=head) → maps to plate_length
        """
        return {
            "center": (self.y * plate_length, self.x * plate_width),  # SWAPPED: y→length, x→width
            "mass_kg": self.mass_kg,
            "material": self.material,
            "diameter_m": self.diameter_mm / 1000,
            "mount_type": self.mount_type,
        }
    
    def frequency_shift_factor(self, mode_amplitude: float, modal_mass_kg: float) -> float:
        """
        Calcola fattore di shift frequenza per un modo.
        
        f'_n / f_n = 1 / sqrt(1 + m * φ²(x,y) / M_eff)
        
        Args:
            mode_amplitude: Ampiezza modo normalizzata in (x,y) [0-1]
            modal_mass_kg: Massa modale effettiva del modo
            
        Returns:
            Fattore moltiplicativo < 1 (frequenza si abbassa)
        """
        if modal_mass_kg <= 0:
            return 1.0
        ratio = self.mass_kg * (mode_amplitude ** 2) / modal_mass_kg
        return 1.0 / np.sqrt(1.0 + ratio)


# Posizioni ottimali per masse (da Shen 2016)
OPTIMAL_MASS_POSITIONS = [
    (0.33, 0.5),   # 1/3 length, center width
    (0.67, 0.5),   # 2/3 length, center width
    (0.5, 0.33),   # Center, 1/3 width
    (0.5, 0.67),   # Center, 2/3 width
]


@dataclass
class SpringSupportGene:
    """
    Gene per un punto di appoggio a molla con FISICA REALE.
    
    ═══════════════════════════════════════════════════════════════════════════
    VIBRATION ISOLATION THEORY (Den Hartog 1956, Harris & Piersol 2002)
    ═══════════════════════════════════════════════════════════════════════════
    
    Trasmissibilità:
        T(ω) = √[(1 + (2ζr)²) / ((1-r²)² + (2ζr)²)]
        
    Phase Shift:
        φ(ω) = -atan2(2ζr, 1 - r²)
        
    dove:
        r = ω/ω_n = frequency ratio
        ω_n = √(k/m) = natural frequency of spring-mass system
        ζ = damping ratio (typical 0.05-0.15 for rubber/spring)
    
    FEM Integration (penalty method):
        K_global[support_dof, support_dof] += k_spring
    
    CONFIGURAZIONE FISICA:
    - 5 punti di appoggio tipici (head, shoulders, hips)
    - Molle alte 150mm, compressione max 80mm → clearance 70mm
    - Spazio sotto per amplificatore, cavi, elettronica
    
    Attributes:
        x, y: Posizione (normalizzata 0-1)
        stiffness_n_m: Rigidezza molla [N/m] (range 3000-50000)
        damping_ratio: Smorzamento ζ (0.05-0.15)
        height_mm: Altezza molla a riposo [mm] (default 150)
        min_height_mm: Compressione massima [mm] (default 70)
        diameter_mm: Diametro base supporto [mm] (default 60)
    """
    x: float
    y: float
    stiffness_n_m: float = 8000.0   # 8 kN/m default (optimized for ~20Hz cutoff)
    damping_ratio: float = 0.1      # ζ typical for rubber/spring combo
    height_mm: float = 150.0        # Altezza a riposo
    min_height_mm: float = 70.0     # Clearance minimo (per hardware)
    diameter_mm: float = 60.0       # Diametro base (zona proibita)
    
    # Range rigidezza per diverse applicazioni
    STIFFNESS_RANGES = {
        "soft": (3000, 6000),       # f_n ~2-3Hz, isola da ~3Hz
        "medium": (6000, 15000),    # f_n ~3-5Hz, isola da ~7Hz
        "hard": (15000, 50000),     # f_n ~5-10Hz, isola da ~14Hz
    }
    
    def natural_frequency_hz(self, supported_mass_kg: float) -> float:
        """
        Frequenza naturale del sistema molla-massa [Hz].
        
        f_n = (1/2π) * √(k/m)
        
        Args:
            supported_mass_kg: Massa supportata da questa molla [kg]
                              (tipicamente peso_persona / n_molle)
        """
        if supported_mass_kg <= 0:
            return 100.0  # Fallback
        omega_n = np.sqrt(self.stiffness_n_m / supported_mass_kg)
        return omega_n / (2 * np.pi)
    
    def transmissibility(self, freq_hz: float, supported_mass_kg: float) -> float:
        """
        Trasmissibilità a una data frequenza.
        
        T(ω) = √[(1 + (2ζr)²) / ((1-r²)² + (2ζr)²)]
        
        - T > 1: AMPLIFICAZIONE (evitare frequenze vicino a f_n!)
        - T ≈ 1: r < 0.5 (basse frequenze passano)
        - T < 1: ISOLAMENTO (r > √2 ≈ 1.41)
        - T → 0: isolamento perfetto (r >> 1)
        
        Args:
            freq_hz: Frequenza di eccitazione [Hz]
            supported_mass_kg: Massa supportata [kg]
        
        Returns:
            Trasmissibilità (rapporto ampiezza output/input)
        """
        f_n = self.natural_frequency_hz(supported_mass_kg)
        if f_n <= 0:
            return 1.0
        
        r = freq_hz / f_n  # Frequency ratio
        zeta = self.damping_ratio
        
        numerator = 1 + (2 * zeta * r) ** 2
        denominator = (1 - r**2)**2 + (2 * zeta * r)**2
        
        if denominator <= 0:
            return 10.0  # Cap at resonance
        
        return np.sqrt(numerator / denominator)
    
    def phase_shift_deg(self, freq_hz: float, supported_mass_kg: float) -> float:
        """
        Sfasamento a una data frequenza [degrees].
        
        φ(ω) = -atan2(2ζr, 1 - r²)
        
        - r < 1: φ → 0° (in fase)
        - r = 1: φ = -90° (risonanza)
        - r > 1: φ → -180° (quasi inversione)
        
        Args:
            freq_hz: Frequenza [Hz]
            supported_mass_kg: Massa supportata [kg]
        
        Returns:
            Phase shift in degrees (negative = lag)
        """
        f_n = self.natural_frequency_hz(supported_mass_kg)
        if f_n <= 0:
            return 0.0
        
        r = freq_hz / f_n
        zeta = self.damping_ratio
        
        phase_rad = -np.arctan2(2 * zeta * r, 1 - r**2)
        return np.degrees(phase_rad)
    
    def isolation_cutoff_hz(self, supported_mass_kg: float) -> float:
        """
        Frequenza di cutoff isolamento (T = 1, r = √2).
        
        Frequenze SOPRA questo valore sono isolate (T < 1).
        Frequenze SOTTO sono trasmesse o amplificate.
        """
        f_n = self.natural_frequency_hz(supported_mass_kg)
        return f_n * np.sqrt(2)
    
    def mutate(self, sigma: float = 0.03) -> 'SpringSupportGene':
        """
        Muta questo supporto RISPETTANDO i vincoli di zona.
        
        CONSTRAINT: Springs must stay in their assigned zone!
        - Corner springs (x < 0.3 or x > 0.7): stay in corners
        - Center springs (0.3 <= x <= 0.7): stay in center
        
        This prevents structural instability from springs migrating
        away from corners during evolution.
        """
        # Determine zone based on current position
        is_left = self.x < 0.3
        is_right = self.x > 0.7
        is_feet = self.y < 0.3
        is_head = self.y > 0.7
        is_center_x = 0.3 <= self.x <= 0.7
        is_center_y = 0.3 <= self.y <= 0.7
        
        # Mutate within zone constraints
        if is_left:
            # Left zone: x must stay in [0.05, 0.30]
            new_x = np.clip(self.x + np.random.normal(0, sigma), 0.05, 0.30)
        elif is_right:
            # Right zone: x must stay in [0.70, 0.95]
            new_x = np.clip(self.x + np.random.normal(0, sigma), 0.70, 0.95)
        else:
            # Center zone: x stays in [0.35, 0.65]
            new_x = np.clip(self.x + np.random.normal(0, sigma), 0.35, 0.65)
        
        if is_feet:
            # Feet zone: y must stay in [0.05, 0.25]
            new_y = np.clip(self.y + np.random.normal(0, sigma), 0.05, 0.25)
        elif is_head:
            # Head zone: y must stay in [0.75, 0.95]
            new_y = np.clip(self.y + np.random.normal(0, sigma), 0.75, 0.95)
        else:
            # Center zone: y stays in [0.35, 0.60] (below person's center of mass)
            new_y = np.clip(self.y + np.random.normal(0, sigma), 0.35, 0.60)
        
        return SpringSupportGene(
            x=new_x,
            y=new_y,
            stiffness_n_m=np.clip(
                self.stiffness_n_m * (1 + np.random.normal(0, 0.2)),
                3000, 50000
            ),
            damping_ratio=np.clip(
                self.damping_ratio + np.random.normal(0, 0.02),
                0.03, 0.25
            ),
            height_mm=self.height_mm,  # Fixed (hardware constraint)
            min_height_mm=self.min_height_mm,  # Fixed
            diameter_mm=self.diameter_mm,  # Fixed
        )
    
    @property
    def exclusion_zone(self) -> Tuple[float, float, float]:
        """
        Zona di esclusione per cutouts/exciters (x, y, radius_normalized).
        
        Returns radius normalized to typical plate dimension (assume 2m plate).
        """
        radius_m = self.diameter_mm / 1000 / 2
        # Normalize to ~2m plate
        radius_norm = radius_m / 2.0 * 1.5  # 50% margin
        return (self.x, self.y, radius_norm)
    
    def deflection_mm(self, load_kg: float) -> float:
        """
        Deflessione statica molla sotto carico [mm].
        
        δ = F/k = mg/k
        
        Args:
            load_kg: Carico in kg (es. peso persona / n_supports)
        
        Returns:
            Deflessione in mm
        """
        force_n = load_kg * 9.81
        deflection_m = force_n / self.stiffness_n_m
        deflection_mm = deflection_m * 1000
        
        # Limita a compressione massima
        max_deflection = self.height_mm - self.min_height_mm
        return min(deflection_mm, max_deflection)
    
    def get_fem_penalty_stiffness(self) -> float:
        """
        Rigidezza da aggiungere alla matrice FEM (penalty method).
        
        Per scikit-fem: K[support_dof, support_dof] += questo valore
        
        Returns:
            Spring stiffness [N/m] per FEM boundary condition
        """
        return self.stiffness_n_m
    
    def is_near_edge(self, margin: float = 0.15) -> bool:
        """
        Verifica se la molla è vicina al bordo della piastra.
        
        Le molle DEVONO essere vicino ai bordi per supportare la struttura!
        Una molla al centro (0.5, 0.5) non può sostenere gli angoli.
        
        Args:
            margin: Distanza dal bordo considerata "vicina" (default 15%)
        
        Returns:
            True se x < margin OR x > 1-margin OR y < margin OR y > 1-margin
        """
        near_left = self.x < margin
        near_right = self.x > (1.0 - margin)
        near_feet = self.y < margin
        near_head = self.y > (1.0 - margin)
        return near_left or near_right or near_feet or near_head
    
    def is_in_corner_region(self, margin: float = 0.25) -> Tuple[bool, str]:
        """
        Verifica se la molla è in una regione d'angolo.
        
        STRUCTURAL PHYSICS: Per sostenere una piastra rettangolare,
        le molle devono essere distribuite in modo da coprire:
        - 4 angoli (o regioni vicine)
        - Opzionalmente: centro (dove poggia il peso della persona)
        
        Returns:
            (is_corner, region_name) dove region è una di:
            'feet_left', 'feet_right', 'head_left', 'head_right', 'center', 'other'
        """
        # Regioni d'angolo (x, y range)
        if self.x < margin and self.y < margin:
            return True, 'feet_left'
        elif self.x > (1.0 - margin) and self.y < margin:
            return True, 'feet_right'
        elif self.x < margin and self.y > (1.0 - margin):
            return True, 'head_left'
        elif self.x > (1.0 - margin) and self.y > (1.0 - margin):
            return True, 'head_right'
        elif 0.35 < self.x < 0.65 and 0.35 < self.y < 0.65:
            return True, 'center'
        else:
            return False, 'other'
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """
        Converti in coordinate assolute [m] per FEM.
        
        COORDINATE CONVENTION (person.py standard):
            x: lateral position (0=left, 1=right) in normalized coords
            y: longitudinal position (0=feet, 1=head) in normalized coords
        
        FEM CONVENTION:
            FEM x-axis = plate length (longitudinal)
            FEM y-axis = plate width (lateral)
        
        Returns dict for fem_modal_analysis spring_supports_abs parameter.
        """
        return {
            "position": (self.y * plate_length, self.x * plate_width),  # y→FEM_x, x→FEM_y
            "stiffness_n_m": self.stiffness_n_m,
            "damping_ratio": self.damping_ratio,
        }


# Default 5-point spring support configuration
# Ottimizzato per tavola vibroacustica standard
# PHYSICS: k=8kN/m con m=15kg → f_n≈3.7Hz → cutoff≈5.2Hz
# Frequenze > 5Hz saranno isolate dal pavimento
#
# STRUCTURAL PRIORITY: Springs must be at CORNERS for structural support!
# The plate cannot stand if springs are only in the center.
# Weight distribution is secondary to structural stability.
DEFAULT_SPRING_SUPPORTS = [
    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL: 4 corners for stability (like table legs)
    # ═══════════════════════════════════════════════════════════════════════
    # Feet-left corner (supports feet end left)
    SpringSupportGene(x=0.15, y=0.10, stiffness_n_m=10000, damping_ratio=0.10),
    # Feet-right corner (supports feet end right)
    SpringSupportGene(x=0.85, y=0.10, stiffness_n_m=10000, damping_ratio=0.10),
    # Head-left corner (supports head end left)
    SpringSupportGene(x=0.15, y=0.90, stiffness_n_m=8000, damping_ratio=0.12),
    # Head-right corner (supports head end right)  
    SpringSupportGene(x=0.85, y=0.90, stiffness_n_m=8000, damping_ratio=0.12),
    # ═══════════════════════════════════════════════════════════════════════
    # CENTER: Support person's weight (heaviest point: hips/lower back)
    # ═══════════════════════════════════════════════════════════════════════
    SpringSupportGene(x=0.50, y=0.45, stiffness_n_m=12000, damping_ratio=0.08),
]


# Hardware clearance zone (amplifier, electronics under plate)
HARDWARE_CLEARANCE = {
    "amplifier": {
        "x_range": (0.35, 0.65),  # Center 30%
        "y_range": (0.40, 0.60),  # Center 20%
        "height_mm": 60,          # Min clearance
    }
}


@dataclass 
class ExciterPosition:
    """
    Posizione e parametri DSP di un eccitatore sulla tavola.
    
    Sistema: 4× Dayton DAEX25 (25mm, 40W, 8Ω)
    - CH1, CH2: Head (stereo left/right)
    - CH3, CH4: Feet (stereo left/right)
    
    DSP PARAMETERS (evolvable in BLOOM phase):
    - phase_deg: Fase (0-360°) per interferenza costruttiva/distruttiva
    - gain_db: Guadagno (-12 to +6 dB) per bilanciare zone
    - delay_ms: Ritardo (0-50 ms) per allineamento temporale
    
    EVOLUTION STRATEGY (SEED → BLOOM):
    - SEED: Ottimizza solo posizione (x, y) per trovare design fisico
    - BLOOM: Aggiunge DSP params per fine-tuning (fase, gain, delay)
    
    Reference: JAB4_DOCUMENTATION.md, Sum & Pan 2000
    """
    x: float              # Posizione X normalizzata (0=left, 1=right)
    y: float              # Posizione Y normalizzata (0=feet, 1=head)
    channel: int          # Canale JAB4 (1-4)
    
    # DSP Parameters (evolvable in BLOOM phase)
    phase_deg: float = 0.0        # Fase in gradi (0-360) - GA può ottimizzare!
    gain_db: float = 0.0          # Guadagno in dB (-12 to +6) - BLOOM phase
    delay_ms: float = 0.0         # Delay in ms (0-50) - BLOOM phase
    
    exciter_model: str = "dayton_daex25"  # Key in EXCITERS database
    
    # Derivati dal modello
    diameter_mm: float = 25.0     # Dayton DAEX25
    power_w: float = 10.0         # RMS power (max 25W per channel)
    impedance_ohm: float = 4.0    # 4Ω version
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """
        Converti in coordinate assolute [m].
        
        COORDINATE CONVENTION (person.py standard):
            x: lateral position (0=left, 1=right) in normalized coords
            y: longitudinal position (0=feet, 1=head) in normalized coords
        
        FEM CONVENTION:
            FEM x-axis = plate length (longitudinal)
            FEM y-axis = plate width (lateral)
        
        Output for FEM: (y * plate_length, x * plate_width)
        """
        return {
            "center": (self.y * plate_length, self.x * plate_width),  # y→FEM_x, x→FEM_y
            "diameter": self.diameter_mm / 1000,
            "channel": self.channel,
        }
    
    @property
    def is_head_zone(self) -> bool:
        """True se eccitatore è nella zona testa (y > 0.7)."""
        return self.y > 0.7
    
    @property
    def is_feet_zone(self) -> bool:
        """True se eccitatore è nella zona piedi (y < 0.3)."""
        return self.y < 0.3
    
    @property
    def gain_linear(self) -> float:
        """Convert gain_db to linear amplitude multiplier."""
        return 10 ** (self.gain_db / 20)
    
    @property
    def delay_samples(self) -> int:
        """Convert delay_ms to samples at 48kHz."""
        return int(self.delay_ms * 48)  # 48000 Hz / 1000 ms
    
    def get_dsp_params(self) -> Dict[str, float]:
        """Return DSP parameters as dict for audio processing."""
        return {
            "channel": self.channel,
            "phase_deg": self.phase_deg,
            "gain_db": self.gain_db,
            "gain_linear": self.gain_linear,
            "delay_ms": self.delay_ms,
            "delay_samples": self.delay_samples,
        }


# Default exciter layout: 4 exciters in standard configuration
DEFAULT_EXCITERS = [
    ExciterPosition(x=0.3, y=0.85, channel=1),   # Head Left (CH1)
    ExciterPosition(x=0.7, y=0.85, channel=2),   # Head Right (CH2)
    ExciterPosition(x=0.3, y=0.15, channel=3),   # Feet Left (CH3)
    ExciterPosition(x=0.7, y=0.15, channel=4),   # Feet Right (CH4)
]


@dataclass
class PlateGenome:
    """
    Rappresentazione genetica di una tavola vibroacustica.
    
    ═══════════════════════════════════════════════════════════════════════════
    LUTHERIE APPROACH (Schleske 2002, Fletcher & Rossing 1998)
    ═══════════════════════════════════════════════════════════════════════════
    
    La tavola vibroacustica è come una tavola armonica di strumento:
    - I cutouts (fori) accordano i modi propri verso frequenze target
    - Le grooves (scanalature) permettono accordatura fine
    - Lo spessore variabile distribuisce la risposta in frequenza
    - Gli exciters (4x Dayton DAEX25) pilotano la tavola
    
    Come nel violino (f-holes) o chitarra (rosetta):
    - Cutouts spostano le frequenze modali (macro tuning)
    - Grooves permettono fine tuning senza fori passanti
    - Posizione determina quali modi vengono alterati
    
    Hardware: 4× Dayton DAEX25 (25mm, 40W, 8Ω) via JAB4 WONDOM
    - CH1/CH2: Head stereo (left/right)
    - CH3/CH4: Feet stereo (left/right)
    
    Attributes:
        length: Lunghezza tavola [m]
        width: Larghezza tavola [m]
        thickness_base: Spessore base [m]
        contour_type: Tipo di contorno
        cutouts: Lista fori passanti (accordatura macro)
        grooves: Lista scanalature (accordatura fine)
        exciters: Posizioni 4 eccitatori
    """
    # Dimensioni base
    length: float = 1.85
    width: float = 0.64
    thickness_base: float = 0.015  # 15mm
    
    # Forma contorno
    contour_type: ContourType = ContourType.GOLDEN_RECT
    control_points: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # FIXED CONTOUR: Se True, il contorno NON può cambiare durante evoluzione
    # Utile quando l'utente ha scelto una forma specifica (es. RECTANGLE)
    fixed_contour: bool = False
    
    # SYMMETRY: Se True, tutti i cutouts/grooves sono specchiati sul centro
    enforce_symmetry: bool = True  # Default True per acoustic balance
    
    # Spessore variabile (opzionale)
    thickness_field: Optional[np.ndarray] = None
    thickness_variation: float = 0.0  # 0 = uniforme, 1 = max variazione
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LUTHERIE: Tuning features
    # NOTA: max_cutouts è solo un valore di init - l'analisi strutturale durante
    # l'evoluzione penalizza genomi con troppi cutouts che compromettono l'integrità.
    # Nessun limite arbitrario! Se 10 cutouts piccoli tengono, vanno bene.
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Fori passanti (macro tuning) - come f-holes violino
    cutouts: List[CutoutGene] = field(default_factory=list)
    max_cutouts: int = 8  # Default 8 - real limit from structural analysis
    
    # Scanalature (fine tuning) - come graduazioni tavola armonica
    grooves: List[GrooveGene] = field(default_factory=list)
    max_grooves: int = 8  # Max 8 scanalature per accordatura fine
    
    # Vincoli liuteria per cutouts
    min_cutout_radius: float = 0.02   # Min 20mm (lavorabile con fresa 40mm)
    max_cutout_radius: float = 0.08   # Max 80mm (struttura OK)
    min_edge_distance: float = 0.05   # Min 50mm dal bordo
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ATTACHED MASSES: Tuning modale passivo (Shen 2016, Lu 2012)
    # ═══════════════════════════════════════════════════════════════════════════
    
    attached_masses: List[AttachedMassGene] = field(default_factory=list)
    max_attached_masses: int = 4  # Max 4 masse per tuning
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPRING SUPPORTS: Sistema di appoggio su molle
    # Le molle agiscono come fulcri parziali, influenzando propagazione fase
    # ═══════════════════════════════════════════════════════════════════════════
    
    spring_supports: List[SpringSupportGene] = field(
        default_factory=lambda: list(DEFAULT_SPRING_SUPPORTS)
    )
    min_support_clearance_mm: float = 70.0  # Clearance per hardware sotto
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXCITERS: 4× Dayton DAEX25 via JAB4
    # ═══════════════════════════════════════════════════════════════════════════
    
    exciters: List[ExciterPosition] = field(default_factory=lambda: list(DEFAULT_EXCITERS))
    
    # Fitness (calcolato dall'evaluator)
    fitness: float = 0.0
    
    def __post_init__(self):
        """Inizializza punti controllo se vuoti."""
        if len(self.control_points) == 0:
            self.control_points = self._generate_base_contour()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Groove helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def total_stiffness_reduction(self) -> float:
        """
        Calcola riduzione totale rigidezza da grooves.
        
        Returns:
            Fattore di riduzione medio (0-1)
        """
        if not self.grooves:
            return 0.0
        
        total_reduction = sum(g.stiffness_reduction() * g.length for g in self.grooves)
        total_length = sum(g.length for g in self.grooves)
        
        # Weight by groove length relative to plate
        plate_fraction = total_length / 1.0  # Normalized
        return total_reduction * plate_fraction * 0.5  # Empirical factor
    
    def local_thickness_at(self, x: float, y: float) -> float:
        """
        Calcola spessore locale considerando grooves.
        
        Args:
            x, y: Coordinate normalizzate (0-1)
        
        Returns:
            Spessore locale in metri
        """
        base = self.thickness_base
        
        # Check if point is within any groove
        for groove in self.grooves:
            # Simple rectangular approximation
            gx, gy = groove.x, groove.y
            half_len = groove.length / 2
            half_width = (groove.width_mm / 1000) / self.width / 2
            
            # Rotate point to groove frame
            dx = x - gx
            dy = y - gy
            cos_a, sin_a = np.cos(groove.angle), np.sin(groove.angle)
            dx_rot = dx * cos_a + dy * sin_a
            dy_rot = -dx * sin_a + dy * cos_a
            
            if abs(dx_rot) < half_len and abs(dy_rot) < half_width:
                return base * (1.0 - groove.depth)
        
        return base
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPRING SUPPORT HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_exclusion_zones(self) -> List[Tuple[float, float, float]]:
        """
        Get all exclusion zones from spring supports.
        
        Returns:
            List of (x, y, radius_normalized) tuples where cutouts/features
            should not be placed.
        """
        zones = []
        for support in self.spring_supports:
            zones.append(support.exclusion_zone)
        return zones
    
    def is_in_exclusion_zone(self, x: float, y: float, margin: float = 0.0) -> bool:
        """
        Check if a point is within any support exclusion zone.
        
        Args:
            x, y: Point coordinates (normalized)
            margin: Additional margin around exclusion zone
        
        Returns:
            True if point should be avoided
        """
        for sx, sy, sr in self.get_exclusion_zones():
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            if dist < (sr + margin):
                return True
        return False
    
    def phase_transmission_at(self, x: float, y: float, freq_hz: float = 100.0, 
                               person_weight_kg: float = 75.0) -> float:
        """
        Estimate phase transmission factor at a point for a given frequency.
        
        Uses REAL PHYSICS: phase shift depends on frequency relative to 
        spring natural frequency.
        
        φ(ω) = -atan2(2ζr, 1 - r²)
        
        Args:
            x, y: Point coordinates (normalized)
            freq_hz: Frequency of interest [Hz]
            person_weight_kg: Person weight for mass calculation
        
        Returns:
            Phase factor: cos(phase_shift) where phase_shift is averaged
            across nearby supports. 1.0 = in phase, -1.0 = inverted
        """
        if not self.spring_supports:
            return 1.0  # Free plate, no phase issues
        
        # Weight distributed across supports
        weight_per_support = person_weight_kg / len(self.spring_supports)
        
        # Weighted average of nearby support phase shifts
        total_weight = 0.0
        weighted_phase = 0.0
        
        for support in self.spring_supports:
            dist = np.sqrt((x - support.x)**2 + (y - support.y)**2)
            # Inverse distance weighting with cutoff
            if dist < 0.01:
                dist = 0.01  # Avoid division by zero
            weight = 1.0 / (dist + 0.1)  # Smooth falloff
            
            # Get physical phase shift at this frequency
            phase_deg = support.phase_shift_deg(freq_hz, weight_per_support)
            # Convert to transmission factor: cos(phase) gives amplitude sign
            phase_rad = np.radians(phase_deg)
            phase_factor = np.cos(phase_rad)
            
            weighted_phase += weight * phase_factor
            total_weight += weight
        
        return weighted_phase / total_weight if total_weight > 0 else 1.0
    
    def spring_transmissibility_spectrum(self, freq_range_hz: Tuple[float, float] = (1, 200),
                                          person_weight_kg: float = 75.0,
                                          n_points: int = 50) -> Dict:
        """
        Calculate transmissibility spectrum for all spring supports.
        
        Useful for understanding which frequencies are isolated vs amplified.
        
        Args:
            freq_range_hz: (min, max) frequency range
            person_weight_kg: Person weight
            n_points: Number of frequency points
        
        Returns:
            Dict with:
                - 'frequencies': array of frequencies [Hz]
                - 'transmissibility': array of T values (averaged across supports)
                - 'phase_deg': array of phase shifts [degrees]
                - 'isolation_start_hz': frequency where isolation begins (T<1)
                - 'resonance_hz': approximate resonance frequency (max T)
        """
        freqs = np.geomspace(freq_range_hz[0], freq_range_hz[1], n_points)
        weight_per_support = person_weight_kg / len(self.spring_supports)
        
        T_avg = np.zeros(n_points)
        phase_avg = np.zeros(n_points)
        
        for i, f in enumerate(freqs):
            T_values = [s.transmissibility(f, weight_per_support) for s in self.spring_supports]
            phase_values = [s.phase_shift_deg(f, weight_per_support) for s in self.spring_supports]
            T_avg[i] = np.mean(T_values)
            phase_avg[i] = np.mean(phase_values)
        
        # Find isolation start (first T < 1 after peak)
        peak_idx = np.argmax(T_avg)
        isolation_idx = peak_idx
        for i in range(peak_idx, len(T_avg)):
            if T_avg[i] < 1.0:
                isolation_idx = i
                break
        
        return {
            'frequencies': freqs,
            'transmissibility': T_avg,
            'phase_deg': phase_avg,
            'isolation_start_hz': freqs[isolation_idx] if isolation_idx < len(freqs) else freqs[-1],
            'resonance_hz': freqs[peak_idx],
            'peak_transmissibility': T_avg[peak_idx],
        }
    
    def total_support_deflection(self, person_weight_kg: float) -> float:
        """
        Calculate average support deflection under person's weight.
        
        Args:
            person_weight_kg: Total weight of person on plate
        
        Returns:
            Average deflection in mm
        """
        if not self.spring_supports:
            return 0.0
        
        # Weight distributed across supports (simplified uniform)
        weight_per_support = person_weight_kg / len(self.spring_supports)
        
        deflections = [s.deflection_mm(weight_per_support) for s in self.spring_supports]
        return np.mean(deflections)
    
    def clearance_ok(self, person_weight_kg: float) -> bool:
        """
        Check if hardware clearance is maintained under load.
        
        Args:
            person_weight_kg: Weight causing deflection
        
        Returns:
            True if min clearance (70mm default) is maintained
        """
        if not self.spring_supports:
            return True
        
        weight_per_support = person_weight_kg / len(self.spring_supports)
        
        for support in self.spring_supports:
            deflection = support.deflection_mm(weight_per_support)
            remaining_height = support.height_mm - deflection
            if remaining_height < self.min_support_clearance_mm:
                return False
        
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # SYMMETRY ENFORCEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def enforce_bilateral_symmetry(self) -> 'PlateGenome':
        """
        Enforce bilateral symmetry around the centerline (x=0.5).
        
        LUTHERIE PRINCIPLE: Musical instrument plates (violins, guitars) 
        are typically symmetric for balanced response and aesthetics.
        
        This method:
        - Mirrors cutouts: for each cutout at x, creates one at (1-x) with same y
        - Mirrors grooves: same principle
        - Enforces symmetric exciter placement (CH1/CH2 mirrored, CH3/CH4 mirrored)
        
        Returns:
            New PlateGenome with enforced symmetry
        """
        new_genome = copy.deepcopy(self)
        
        # ═══════════════════════════════════════════════════════════════════════
        # SYMMETRIC CUTOUTS - PERFECT SYMMETRY (no duplication!)
        # ═══════════════════════════════════════════════════════════════════════
        # For symmetric shapes, pair cutouts by Y position and enforce
        # perfect X symmetry. This ensures L/R cutouts are EXACTLY mirrored.
        
        symmetric_cutouts = self._enforce_perfect_cutout_symmetry(list(self.cutouts))
        new_genome.cutouts = symmetric_cutouts[:self.max_cutouts]
        
        # ═══════════════════════════════════════════════════════════════════════
        # SYMMETRIC GROOVES (only if enabled)
        # ═══════════════════════════════════════════════════════════════════════
        if self.max_grooves > 0:
            symmetric_grooves = []
            for groove in self.grooves:
                symmetric_grooves.append(groove)
                
                if abs(groove.x - 0.5) > 0.05:
                    mirrored = GrooveGene(
                        x=1.0 - groove.x,
                        y=groove.y,
                        length=groove.length,
                        angle=-groove.angle,  # Mirror angle
                        depth=groove.depth,
                        width_mm=groove.width_mm,
                    )
                    if len(symmetric_grooves) < self.max_grooves:
                        symmetric_grooves.append(mirrored)
            
            new_genome.grooves = symmetric_grooves[:self.max_grooves]
        else:
            # max_grooves=0 → CLEAR all grooves
            new_genome.grooves = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # SYMMETRIC EXCITERS
        # ═══════════════════════════════════════════════════════════════════════
        # Pair CH1 with CH2 (head), CH3 with CH4 (feet)
        # Use same distance from center for paired channels
        # PRESERVE phase AND DSP params for acoustic coherence
        
        for i, exc in enumerate(new_genome.exciters):
            if exc.channel == 1:
                # Find CH2 and mirror
                for j, exc2 in enumerate(new_genome.exciters):
                    if exc2.channel == 2:
                        # Make CH2 mirror of CH1 (keep DSP params for stereo coherence)
                        new_genome.exciters[j] = ExciterPosition(
                            x=1.0 - exc.x,
                            y=exc.y,  # Same Y
                            channel=2,
                            phase_deg=getattr(exc, 'phase_deg', 0.0),  # Same phase for stereo
                            gain_db=getattr(exc, 'gain_db', 0.0),      # Same gain for stereo
                            delay_ms=getattr(exc, 'delay_ms', 0.0),    # Same delay for stereo
                            exciter_model=exc.exciter_model,
                        )
                        break
            elif exc.channel == 3:
                # Find CH4 and mirror
                for j, exc2 in enumerate(new_genome.exciters):
                    if exc2.channel == 4:
                        new_genome.exciters[j] = ExciterPosition(
                            x=1.0 - exc.x,
                            y=exc.y,
                            channel=4,
                            phase_deg=getattr(exc, 'phase_deg', 0.0),  # Same phase
                            gain_db=getattr(exc, 'gain_db', 0.0),      # Same gain
                            delay_ms=getattr(exc, 'delay_ms', 0.0),    # Same delay
                            exciter_model=exc.exciter_model,
                        )
                        break
        
        # ═══════════════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════════════
        # SYMMETRIC SPRINGS - PRESERVE COUNT AND STRUCTURAL POSITIONS!
        # ═══════════════════════════════════════════════════════════════════════
        # Springs must be symmetric for balanced support and phase rotation.
        # CRITICAL: 
        # 1. MUST NOT increase the number of springs!
        # 2. MUST PRESERVE Y positions (structural support!)
        # 3. Only mirror X positions for left/right symmetry
        #
        # BUG FIX: Previous version averaged Y positions, destroying structural
        # support by moving corner springs to center. Now we PRESERVE Y positions
        # and only make X symmetric.
        
        original_count = len(new_genome.spring_supports)
        
        if original_count == 0:
            return new_genome
        
        # Group springs by Y position into "rows" (corner vs center regions)
        # A row is a group of springs at similar Y heights
        springs = list(new_genome.spring_supports)
        symmetric_springs = []
        
        # Define Y regions for structural support
        # CORNER_LOW: y < 0.30 (feet end)
        # CENTER: 0.30 <= y <= 0.70
        # CORNER_HIGH: y > 0.70 (head end)
        
        corner_low = [s for s in springs if s.y < 0.30]
        corner_high = [s for s in springs if s.y > 0.70]
        center = [s for s in springs if 0.30 <= s.y <= 0.70]
        
        def make_symmetric_pair(spring_list: list, region_name: str) -> list:
            """Make springs symmetric within a Y region, preserving Y positions."""
            result = []
            if not spring_list:
                return result
            
            # Sort by X to identify left/right
            sorted_springs = sorted(spring_list, key=lambda s: s.x)
            
            if len(sorted_springs) == 1:
                # Single spring in region: center it
                s = sorted_springs[0]
                result.append(SpringSupportGene(
                    x=0.5, y=s.y,  # PRESERVE Y!
                    stiffness_n_m=s.stiffness_n_m,
                    damping_ratio=s.damping_ratio,
                ))
            elif len(sorted_springs) == 2:
                # Pair: make symmetric around x=0.5, PRESERVE Y!
                left, right = sorted_springs[0], sorted_springs[1]
                left_x = min(left.x, 0.45)  # Keep on left side
                if left_x > 0.35:
                    left_x = 0.15  # Force to edge for corners
                
                # PERFECT SYMMETRY: Use same Y for both springs (average)
                y_symmetric = (left.y + right.y) / 2
                
                result.append(SpringSupportGene(
                    x=left_x, y=y_symmetric,  # SAME Y for both!
                    stiffness_n_m=left.stiffness_n_m,
                    damping_ratio=left.damping_ratio,
                ))
                result.append(SpringSupportGene(
                    x=1.0 - left_x, y=y_symmetric,  # SAME Y for both!
                    stiffness_n_m=right.stiffness_n_m,
                    damping_ratio=right.damping_ratio,
                ))
            else:
                # Multiple springs: pair them by X position
                mid = len(sorted_springs) // 2
                for i in range(mid):
                    left = sorted_springs[i]
                    right = sorted_springs[-(i+1)]
                    left_x = min(left.x, 0.45)
                    
                    # PERFECT SYMMETRY: Use same Y for both springs (average)
                    y_symmetric = (left.y + right.y) / 2
                    
                    result.append(SpringSupportGene(
                        x=left_x, y=y_symmetric,  # SAME Y for both!
                        stiffness_n_m=left.stiffness_n_m,
                        damping_ratio=left.damping_ratio,
                    ))
                    result.append(SpringSupportGene(
                        x=1.0 - left_x, y=y_symmetric,  # SAME Y for both!
                        stiffness_n_m=right.stiffness_n_m,
                        damping_ratio=right.damping_ratio,
                    ))
                
                # Handle odd spring in middle
                if len(sorted_springs) % 2 == 1:
                    mid_spring = sorted_springs[mid]
                    result.append(SpringSupportGene(
                        x=0.5, y=mid_spring.y,  # PRESERVE Y!
                        stiffness_n_m=mid_spring.stiffness_n_m,
                        damping_ratio=mid_spring.damping_ratio,
                    ))
            
            return result
        
        # Process each region
        symmetric_springs.extend(make_symmetric_pair(corner_low, "feet"))
        symmetric_springs.extend(make_symmetric_pair(center, "center"))
        symmetric_springs.extend(make_symmetric_pair(corner_high, "head"))
        
        # SAFETY: Verify count is preserved
        if len(symmetric_springs) != original_count:
            # Fallback: just mirror X positions of original springs
            symmetric_springs = []
            for s in springs:
                if s.x < 0.5:
                    symmetric_springs.append(s)
                elif s.x > 0.5:
                    symmetric_springs.append(SpringSupportGene(
                        x=1.0 - s.x, y=s.y,
                        stiffness_n_m=s.stiffness_n_m,
                        damping_ratio=s.damping_ratio,
                    ))
                else:
                    symmetric_springs.append(s)  # Center spring unchanged
        
        new_genome.spring_supports = symmetric_springs[:original_count]
        
        return new_genome
    
    def _mirror_control_points(self, points: np.ndarray) -> np.ndarray:
        """Mirror control points around x=0 (for freeform cutouts)."""
        if points is None:
            return None
        mirrored = points.copy()
        mirrored[:, 0] = -mirrored[:, 0]  # Flip x coordinates
        return mirrored

    def _enforce_perfect_cutout_symmetry(self, cutouts: list) -> list:
        """
        Enforce PERFECT bilateral symmetry for cutouts.
        
        For symmetric plate shapes (rectangle, ellipse), cutouts must be
        EXACTLY mirrored around x=0.5. This is physically necessary because:
        - 1 million generations will converge to symmetric anyway (optimal)
        - Symmetric cutouts = symmetric modal response = better ear_uniformity
        - Manufacturing: symmetric CNC paths are more efficient
        
        Algorithm:
        1. Separate cutouts into left (x < 0.5), center, and right (x > 0.5)
        2. Pair left/right by similar Y position
        3. For each pair, make X perfectly symmetric: x_right = 1.0 - x_left
        4. Also synchronize width, height, rotation (mirrored)
        
        Returns:
            List of cutouts with perfect bilateral symmetry
        """
        if not cutouts:
            return []
        
        # Separate by position
        left_cuts = [c for c in cutouts if c.x < 0.45]
        center_cuts = [c for c in cutouts if 0.45 <= c.x <= 0.55]
        right_cuts = [c for c in cutouts if c.x > 0.55]
        
        # Sort by Y for pairing
        left_cuts.sort(key=lambda c: c.y)
        right_cuts.sort(key=lambda c: c.y)
        
        result = []
        
        # Pair left and right by Y position
        n_pairs = min(len(left_cuts), len(right_cuts))
        for i in range(n_pairs):
            left = left_cuts[i]
            right = right_cuts[i]
            
            # Use left cutout's properties, mirror for right
            # PERFECT symmetry: x_right = 1.0 - x_left
            x_sym = left.x  # Use left's X as reference
            y_avg = (left.y + right.y) / 2  # Average Y for both
            
            # Left cutout (keep original X, use averaged Y)
            result.append(CutoutGene(
                x=x_sym,
                y=y_avg,
                width=left.width,
                height=left.height,
                rotation=left.rotation,
                shape=left.shape,
                corner_radius=left.corner_radius,
                aspect_bias=left.aspect_bias,
                control_points=left.control_points,
                tool_diameter_mm=left.tool_diameter_mm,
                arc_radius=left.arc_radius,
                arc_angle=left.arc_angle,
            ))
            
            # Right cutout: EXACT mirror
            result.append(CutoutGene(
                x=1.0 - x_sym,  # PERFECT mirror
                y=y_avg,        # Same Y as left
                width=left.width,  # Same dimensions
                height=left.height,
                rotation=-left.rotation,  # Mirror rotation
                shape=left.shape,
                corner_radius=left.corner_radius,
                aspect_bias=left.aspect_bias,
                control_points=self._mirror_control_points(left.control_points) if left.control_points is not None else None,
                tool_diameter_mm=left.tool_diameter_mm,
                arc_radius=left.arc_radius,
                arc_angle=left.arc_angle,
            ))
        
        # Handle unpaired cuts (odd numbers): center them
        for c in left_cuts[n_pairs:]:
            result.append(CutoutGene(
                x=0.5, y=c.y, width=c.width, height=c.height,
                rotation=np.pi/2,  # Perpendicular for centered
                shape=c.shape, corner_radius=c.corner_radius,
                aspect_bias=c.aspect_bias, tool_diameter_mm=c.tool_diameter_mm,
            ))
        for c in right_cuts[n_pairs:]:
            result.append(CutoutGene(
                x=0.5, y=c.y, width=c.width, height=c.height,
                rotation=np.pi/2,
                shape=c.shape, corner_radius=c.corner_radius,
                aspect_bias=c.aspect_bias, tool_diameter_mm=c.tool_diameter_mm,
            ))
        
        # Add center cuts unchanged
        result.extend(center_cuts)
        
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # STRUCTURAL SUPPORT ENFORCEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def enforce_structural_support(self, person_weight_kg: float = 75.0) -> 'PlateGenome':
        """
        Enforce valid structural support positions for springs.
        
        PHYSICS PRIORITY: Springs MUST support the plate structurally before
        being optimized for acoustic properties! A plate that can't stand
        is useless regardless of its acoustic performance.
        
        STRUCTURAL REQUIREMENTS:
        1. At least 4 springs must be near the corners (within 25% from edges)
        2. Springs must be distributed to cover all 4 quadrants
        3. If 5+ springs, one should be near center (person weight distribution)
        
        CURRICULUM METAPHOR: "Il petalo non sboccia se il gambo non regge"
        (The petal doesn't bloom if the stem can't support it)
        
        This is the SEED-phase constraint: structural validity comes BEFORE
        acoustic optimization (BLOOM phase).
        
        Args:
            person_weight_kg: Expected weight to support
        
        Returns:
            New PlateGenome with structurally valid spring positions
        """
        new_genome = copy.deepcopy(self)
        n_springs = len(new_genome.spring_supports)
        
        if n_springs == 0:
            # No springs - create default structural support
            new_genome.spring_supports = list(DEFAULT_SPRING_SUPPORTS)
            return new_genome
        
        # Check current structural coverage
        coverage = self._get_spring_coverage()
        
        # If coverage is good, don't modify
        if coverage['score'] >= 0.8:
            return new_genome
        
        # ═══════════════════════════════════════════════════════════════════════
        # STRUCTURAL CORRECTION: Force springs to valid support positions
        # ═══════════════════════════════════════════════════════════════════════
        
        # Define required support regions (corners + optional center)
        required_regions = [
            ('feet_left', 0.15, 0.15),     # Corner 1
            ('feet_right', 0.85, 0.15),    # Corner 2  
            ('head_left', 0.15, 0.85),     # Corner 3
            ('head_right', 0.85, 0.85),    # Corner 4
        ]
        
        if n_springs >= 5:
            required_regions.append(('center', 0.50, 0.50))
        
        # Calculate stiffness for even load distribution
        load_per_spring_N = (person_weight_kg / n_springs) * 9.81
        target_deflection_m = 0.005  # 5mm
        default_k = load_per_spring_N / target_deflection_m
        
        # Assign springs to required regions
        assigned_springs = []
        remaining_springs = list(new_genome.spring_supports)
        
        for region_name, target_x, target_y in required_regions:
            if not remaining_springs:
                break
            
            # Find closest spring to this region
            best_idx = 0
            best_dist = float('inf')
            for i, spring in enumerate(remaining_springs):
                dist = np.sqrt((spring.x - target_x)**2 + (spring.y - target_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            # Move spring to valid region (with small random offset for variation)
            spring = remaining_springs.pop(best_idx)
            offset_x = np.random.uniform(-0.08, 0.08)
            offset_y = np.random.uniform(-0.08, 0.08)
            
            assigned_springs.append(SpringSupportGene(
                x=np.clip(target_x + offset_x, 0.05, 0.95),
                y=np.clip(target_y + offset_y, 0.05, 0.95),
                stiffness_n_m=spring.stiffness_n_m if spring.stiffness_n_m > 3000 else default_k,
                damping_ratio=spring.damping_ratio,
            ))
        
        # Add any extra springs (can be more freely positioned)
        for spring in remaining_springs:
            assigned_springs.append(spring)
        
        new_genome.spring_supports = assigned_springs
        
        return new_genome
    
    def _get_spring_coverage(self) -> Dict:
        """
        Evaluate how well springs cover required structural regions.
        
        Returns dict with:
        - regions_covered: list of covered regions
        - score: 0-1 coverage score (1 = all corners covered)
        - missing: list of uncovered regions
        """
        # Corner regions (x_range, y_range, name)
        corner_regions = [
            ((0.0, 0.30), (0.0, 0.30), 'feet_left'),
            ((0.70, 1.0), (0.0, 0.30), 'feet_right'),
            ((0.0, 0.30), (0.70, 1.0), 'head_left'),
            ((0.70, 1.0), (0.70, 1.0), 'head_right'),
        ]
        
        covered = []
        missing = []
        
        for (x_min, x_max), (y_min, y_max), name in corner_regions:
            region_covered = False
            for spring in self.spring_supports:
                if x_min <= spring.x <= x_max and y_min <= spring.y <= y_max:
                    region_covered = True
                    break
            
            if region_covered:
                covered.append(name)
            else:
                missing.append(name)
        
        score = len(covered) / len(corner_regions) if corner_regions else 0.0
        
        return {
            'regions_covered': covered,
            'missing': missing,
            'score': score,
            'n_springs': len(self.spring_supports),
        }
    
    def structural_support_penalty(self) -> float:
        """
        Calculate penalty for poor structural spring placement.
        
        Returns:
            Penalty value 0-1 (0 = perfect, 1 = completely invalid)
        """
        coverage = self._get_spring_coverage()
        
        # Base penalty: uncovered corners
        corner_penalty = 1.0 - coverage['score']
        
        # Additional penalty if springs are too clustered
        if len(self.spring_supports) >= 2:
            positions = [(s.x, s.y) for s in self.spring_supports]
            
            # Calculate average distance between springs
            total_dist = 0
            n_pairs = 0
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(positions[i+1:], i+1):
                    total_dist += np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    n_pairs += 1
            
            if n_pairs > 0:
                avg_dist = total_dist / n_pairs
                # Ideal distance ~0.5 (spread across plate)
                # If avg_dist < 0.2, springs are too clustered
                if avg_dist < 0.25:
                    cluster_penalty = (0.25 - avg_dist) / 0.25 * 0.5  # Up to 50% penalty
                    corner_penalty = min(1.0, corner_penalty + cluster_penalty)
        
        return corner_penalty

    # ─────────────────────────────────────────────────────────────────────────
    # Generazione Contorno Base
    # ─────────────────────────────────────────────────────────────────────────
    
    def _generate_base_contour(self, n_points: int = 32) -> np.ndarray:
        """Genera contorno base in base al tipo."""
        if self.contour_type == ContourType.RECTANGLE:
            return self._rectangle_points(n_points)
        elif self.contour_type == ContourType.GOLDEN_RECT:
            return self._golden_rect_points(n_points)
        elif self.contour_type == ContourType.ELLIPSE:
            return self._ellipse_points(n_points)
        elif self.contour_type == ContourType.OVOID:
            return self._ovoid_points(n_points)
        elif self.contour_type == ContourType.SUPERELLIPSE:
            return self._superellipse_points(n_points)
        elif self.contour_type == ContourType.ORGANIC:
            return self._organic_points(n_points)
        elif self.contour_type == ContourType.ERGONOMIC:
            return self._ergonomic_points(n_points)
        elif self.contour_type == ContourType.FREEFORM:
            return self._freeform_points(n_points)
        elif self.contour_type == ContourType.PHI_ROUNDED:
            return self._phi_rounded_points(n_points)
        else:
            return self._ellipse_points(n_points)
    
    def _rectangle_points(self, n: int) -> np.ndarray:
        """Rettangolo con angoli arrotondati."""
        points = []
        n_side = n // 4
        
        # Top
        for i in range(n_side):
            points.append([i / n_side, 1.0])
        # Right
        for i in range(n_side):
            points.append([1.0, 1.0 - i / n_side])
        # Bottom
        for i in range(n_side):
            points.append([1.0 - i / n_side, 0.0])
        # Left
        for i in range(n_side):
            points.append([0.0, i / n_side])
        
        return np.array(points)
    
    def _golden_rect_points(self, n: int) -> np.ndarray:
        """Rettangolo aureo (L/W = φ)."""
        # Il rettangolo è già definito da length/width
        # Qui generiamo i punti
        return self._rectangle_points(n)
    
    def _phi_rounded_points(self, n: int) -> np.ndarray:
        """
        Rectangle with PHI-based rounded corners.
        
        Uses golden ratio in multiple ways:
        - Corner radius = width / (2 × PHI²) ≈ 19% of width
        - Corner arc uses PHI spiral approximation
        - Smooth transition preserves acoustic energy flow
        
        References:
        - Livio, "The Golden Ratio" (Broadway Books, 2002)
        - Huntley, "The Divine Proportion" (Dover, 1970)
        - Phyllotaxis patterns in acoustic diffusers (Cox & D'Antonio, 2009)
        """
        # Corner radius based on PHI
        # PHI² ≈ 2.618, so radius ≈ 19% of smaller dimension
        corner_radius = 0.5 / (PHI * PHI)  # In normalized coords
        corner_radius = min(corner_radius, 0.15)  # Cap at 15%
        
        points = []
        n_corner = max(4, n // 8)  # Points per corner arc
        n_edge = max(4, n // 8)    # Points per straight edge
        
        # Helper to create PHI-smooth corner arc
        def phi_corner_arc(cx, cy, start_angle, end_angle, r):
            """Create corner arc with PHI-based smoothing."""
            arc_points = []
            # Use golden angle for natural arc subdivision
            golden_angle = np.pi / PHI  # ~111.25°
            
            # Linear interpolation for consistent spacing
            for i in range(n_corner):
                t = i / max(n_corner - 1, 1)
                # PHI-weighted easing for smoother transition
                t_smooth = t * t * (3 - 2 * t)  # Smoothstep
                angle = start_angle + t_smooth * (end_angle - start_angle)
                
                px = cx + r * np.cos(angle)
                py = cy + r * np.sin(angle)
                arc_points.append([px, py])
            return arc_points
        
        r = corner_radius
        margin = 0.02  # Small margin from edges
        
        # Corners: TL, TR, BR, BL (counterclockwise from top-left)
        corners = [
            (margin + r, 1.0 - margin - r, np.pi/2, np.pi),       # Top-left
            (1.0 - margin - r, 1.0 - margin - r, 0, np.pi/2),     # Top-right
            (1.0 - margin - r, margin + r, -np.pi/2, 0),          # Bottom-right
            (margin + r, margin + r, np.pi, 3*np.pi/2),           # Bottom-left
        ]
        
        # Build contour: edge-corner-edge-corner...
        for i, (cx, cy, start_a, end_a) in enumerate(corners):
            # Add corner arc
            arc = phi_corner_arc(cx, cy, start_a, end_a, r)
            points.extend(arc)
            
            # Add straight edge to next corner
            next_i = (i + 1) % 4
            next_cx, next_cy, next_start, _ = corners[next_i]
            
            # Edge endpoints
            if i == 0:  # Top edge (going right)
                x1, y1 = cx + r, cy + r
                x2, y2 = next_cx - r, next_cy + r
            elif i == 1:  # Right edge (going down)
                x1, y1 = cx + r, cy - r
                x2, y2 = next_cx + r, next_cy + r
            elif i == 2:  # Bottom edge (going left)
                x1, y1 = cx - r, cy - r
                x2, y2 = next_cx + r, next_cy - r
            else:  # Left edge (going up)
                x1, y1 = cx - r, cy + r
                x2, y2 = next_cx - r, next_cy - r
            
            # Add edge points
            for j in range(1, n_edge):
                t = j / n_edge
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                points.append([px, py])
        
        return np.array(points)
    
    def _ellipse_points(self, n: int) -> np.ndarray:
        """Ellisse."""
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 0.5 + 0.5 * np.cos(theta)
        y = 0.5 + 0.5 * np.sin(theta)
        return np.column_stack([x, y])
    
    def _ovoid_points(self, n: int) -> np.ndarray:
        """Ovoide (uovo) - più stretto a un'estremità."""
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Raggio variabile - più grande verso testa (persona distesa)
        k = 0.15  # Fattore asimmetria
        r = 0.5 * (1 + k * np.cos(theta))
        
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + 0.5 * np.sin(theta)  # Y simmetrico
        
        return np.column_stack([x, y])
    
    def _superellipse_points(self, n: int, exponent: float = 2.5) -> np.ndarray:
        """
        Superellipse (squircle) - smooth transition between rectangle and ellipse.
        
        |x/a|^n + |y/b|^n = 1
        - n=2: ellipse
        - n=4: squircle (rounded rectangle)
        - n→∞: rectangle
        
        Great for CNC milling (smooth curves, no sharp corners).
        """
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Use sign-preserving power for smooth curve
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        x = 0.5 + 0.48 * np.sign(cos_t) * np.abs(cos_t) ** (2 / exponent)
        y = 0.5 + 0.48 * np.sign(sin_t) * np.abs(sin_t) ** (2 / exponent)
        
        return np.column_stack([x, y])
    
    def _organic_points(self, n: int) -> np.ndarray:
        """
        Organic blob shape using Fourier harmonics.
        
        Creates smooth, natural-looking shapes like:
        - Guitar bodies
        - Violin plates
        - Amoeba-like forms
        
        The shape is generated using low-frequency Fourier components
        to ensure smooth, CNC-friendly curves.
        """
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Base ellipse
        r = 0.45 * np.ones(n)
        
        # Add random Fourier harmonics (only low frequencies for smoothness)
        np.random.seed(hash(id(self)) % 2**32)  # Reproducible per instance
        n_harmonics = 4
        for k in range(2, n_harmonics + 2):
            amp = 0.08 / k  # Decreasing amplitude for higher harmonics
            phase = np.random.uniform(0, 2 * np.pi)
            r += amp * np.cos(k * theta + phase)
        
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta) * 1.3  # Slightly elongated
        
        # Normalize to fit in [0,1] x [0,1]
        x = (x - x.min()) / (x.max() - x.min()) * 0.96 + 0.02
        y = (y - y.min()) / (y.max() - y.min()) * 0.96 + 0.02
        
        return np.column_stack([x, y])
    
    def _ergonomic_points(self, n: int) -> np.ndarray:
        """
        Body-conforming ergonomic shape for vibroacoustic therapy.
        
        Wider at shoulders, narrower at waist, follows human torso outline.
        Optimized for lying-down position (head at top).
        """
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Human torso-inspired profile
        # Shoulder region (top): wider
        # Waist region (middle): narrower  
        # Hip region (bottom): medium width
        
        # Vertical profile modulation
        y_norm = (np.sin(theta) + 1) / 2  # 0=bottom, 1=top
        
        # Width varies with y (anatomical profile)
        width_profile = 0.35 + 0.15 * np.cos(2 * np.pi * y_norm)  # Narrower at waist
        width_profile += 0.05 * np.cos(4 * np.pi * y_norm)  # Subtle hip curve
        
        x = 0.5 + width_profile * np.cos(theta)
        y = 0.5 + 0.48 * np.sin(theta)
        
        return np.column_stack([x, y])
    
    def _freeform_points(self, n: int) -> np.ndarray:
        """
        Fully evolvable freeform shape using control points.
        
        If control_points already exist (from evolution), interpolate them.
        Otherwise, start from a slightly perturbed ellipse.
        
        Uses cubic spline interpolation for smooth CNC-compatible curves.
        """
        if len(self.control_points) >= 4:
            # Interpolate existing control points with cubic spline
            from scipy.interpolate import splprep, splev
            
            try:
                # Close the curve
                cp = np.vstack([self.control_points, self.control_points[0]])
                
                # Fit spline
                tck, u = splprep([cp[:, 0], cp[:, 1]], s=0, per=True, k=3)
                
                # Evaluate at n points
                u_new = np.linspace(0, 1, n)
                x, y = splev(u_new, tck)
                
                return np.column_stack([x, y])
            except Exception:
                # Fallback to linear interpolation
                pass
        
        # Generate initial freeform from perturbed ellipse
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Start with ellipse + random smooth perturbations
        r_base = 0.45
        r_perturbation = 0.05 * np.sin(3 * theta) + 0.03 * np.cos(5 * theta)
        r = r_base + r_perturbation
        
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta) * 1.2  # Slightly elongated
        
        return np.column_stack([x, y])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Operatori Genetici
    # ─────────────────────────────────────────────────────────────────────────
    
    def mutate(
        self,
        sigma_contour: float = 0.02,
        sigma_thickness: float = 0.1,
        sigma_dimensions: float = 0.02,
        p_add_cutout: float = 0.1,
        p_remove_cutout: float = 0.1,
    ) -> 'PlateGenome':
        """
        Crea una copia mutata del genoma.
        
        Args:
            sigma_contour: Deviazione std per punti contorno
            sigma_thickness: Deviazione std per campo spessore
            sigma_dimensions: Deviazione std per dimensioni
            p_add_cutout: Probabilità di aggiungere un taglio
            p_remove_cutout: Probabilità di rimuovere un taglio
        
        Returns:
            Nuovo PlateGenome mutato
        """
        # Copia profonda
        new_genome = copy.deepcopy(self)
        
        # Muta dimensioni (piccole variazioni)
        # CRITICAL: min_length must accommodate person!
        # Default min 1.85m (for 1.75m person + 10cm margin)
        min_length = getattr(self, 'min_length', 1.85)
        new_genome.length = np.clip(
            self.length * (1 + np.random.normal(0, sigma_dimensions)),
            min_length, 2.2
        )
        new_genome.width = np.clip(
            self.width * (1 + np.random.normal(0, sigma_dimensions)),
            0.4, 0.9
        )
        new_genome.thickness_base = np.clip(
            self.thickness_base * (1 + np.random.normal(0, sigma_dimensions)),
            0.008, 0.030
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONTORNO TAVOLA: può mutare tipo e forma SOLO SE NON FIXED
        # ═══════════════════════════════════════════════════════════════════════
        
        # 10% chance di cambiare tipo contorno, BUT ONLY IF NOT FIXED
        if not self.fixed_contour and np.random.random() < 0.10:
            all_types = [
                ContourType.RECTANGLE, ContourType.GOLDEN_RECT, 
                ContourType.ELLIPSE, ContourType.OVOID,
                ContourType.VESICA_PISCIS, ContourType.SUPERELLIPSE, 
                ContourType.ORGANIC, ContourType.ERGONOMIC, ContourType.FREEFORM
            ]
            new_genome.contour_type = np.random.choice(all_types)
            
            # Se passa a FREEFORM, genera control points iniziali
            if new_genome.contour_type == ContourType.FREEFORM and len(new_genome.control_points) == 0:
                # Genera spline iniziale basata su ellisse
                n_pts = np.random.randint(8, 16)
                angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
                radii = 0.5 + 0.1 * np.random.randn(n_pts)  # Variazione rispetto ellisse
                new_genome.control_points = np.column_stack([
                    0.5 + radii * 0.4 * np.cos(angles),  # x
                    0.5 + radii * 0.5 * np.sin(angles)   # y
                ])
        
        # Muta punti contorno per FREEFORM
        if len(self.control_points) > 0 and self.contour_type == ContourType.FREEFORM:
            noise = np.random.normal(0, sigma_contour, self.control_points.shape)
            new_genome.control_points = np.clip(
                self.control_points + noise, 0.05, 0.95
            )
            
            # Possibilità di aggiungere/rimuovere control point
            if np.random.random() < 0.1 and len(new_genome.control_points) > 6:
                idx = np.random.randint(len(new_genome.control_points))
                new_genome.control_points = np.delete(new_genome.control_points, idx, axis=0)
            elif np.random.random() < 0.1 and len(new_genome.control_points) < 20:
                # Inserisci punto tra due esistenti
                idx = np.random.randint(len(new_genome.control_points))
                pt1 = new_genome.control_points[idx]
                pt2 = new_genome.control_points[(idx + 1) % len(new_genome.control_points)]
                new_pt = (pt1 + pt2) / 2 + np.random.normal(0, 0.02, 2)
                new_genome.control_points = np.insert(new_genome.control_points, idx + 1, new_pt, axis=0)
        
        # Muta campo spessore
        if self.thickness_field is not None:
            noise = np.random.normal(0, sigma_thickness, self.thickness_field.shape)
            new_genome.thickness_field = np.clip(
                self.thickness_field + noise, 0.5, 1.5
            )
        
        # Muta cutouts esistenti
        new_genome.cutouts = [c.mutate(0.03) for c in self.cutouts]
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHYSICS-GUIDED CUTOUT PLACEMENT (Schleske 2002, Fletcher & Rossing 1998)
        # Instead of random placement, use modal analysis to find optimal positions
        # ═══════════════════════════════════════════════════════════════════════
        if self.max_cutouts > 0:
            if np.random.random() < p_add_cutout and len(new_genome.cutouts) < self.max_cutouts:
                # Try physics-guided placement first (80% of the time)
                cutout_params = None
                if np.random.random() < 0.8:
                    cutout_params = self._get_physics_guided_cutout(new_genome.cutouts)
                
                if cutout_params:
                    # Use physics-guided position and shape
                    new_genome.cutouts.append(CutoutGene(
                        x=cutout_params["x"],
                        y=cutout_params["y"],
                        width=cutout_params["width"],
                        height=cutout_params["height"],
                        rotation=cutout_params["rotation"],
                        shape=cutout_params["shape"],
                        corner_radius=cutout_params.get("corner_radius", 0.3),
                        aspect_bias=cutout_params.get("aspect_bias", 1.0),
                    ))
                else:
                    # Fallback to random (still organic shapes only)
                    # STRUCTURAL SAFETY: Max 10% of plate dimension to avoid dividing cuts!
                    new_genome.cutouts.append(CutoutGene(
                        x=np.random.uniform(0.1, 0.9),
                        y=np.random.uniform(0.15, 0.85),
                        width=np.random.uniform(0.02, 0.08),   # Max 8% width
                        height=np.random.uniform(0.02, 0.10),  # Max 10% height
                        rotation=np.random.uniform(0, 2 * np.pi),
                        shape=np.random.choice(CUTOUT_SHAPES),
                        corner_radius=np.random.uniform(0.0, 0.8),
                        aspect_bias=np.random.uniform(0.5, 2.0),
                    ))
            
            if np.random.random() < p_remove_cutout and len(new_genome.cutouts) > 0:
                idx = np.random.randint(len(new_genome.cutouts))
                new_genome.cutouts.pop(idx)
        
        # ═══════════════════════════════════════════════════════════════════════
        # LUTHERIE: Mutate grooves (scanalature per accordatura fine)
        # ═══════════════════════════════════════════════════════════════════════
        
        # ONLY keep/mutate grooves if max_grooves > 0
        if self.max_grooves > 0:
            new_genome.grooves = [g.mutate(0.02) for g in self.grooves]
            
            # Probabilità di aggiungere/rimuovere groove
            p_add_groove = 0.08
            p_remove_groove = 0.08
            
            if np.random.random() < p_add_groove and len(new_genome.grooves) < self.max_grooves:
                # Aggiungi groove in zona spinale (y=0.3-0.7) per massimo effetto
                new_genome.grooves.append(GrooveGene(
                    x=np.random.uniform(0.2, 0.8),
                    y=np.random.uniform(0.3, 0.7),
                    length=np.random.uniform(0.05, 0.15),
                    angle=np.random.uniform(-np.pi/4, np.pi/4),
                    depth=np.random.uniform(0.2, 0.4),
                    width_mm=np.random.choice([3.0, 5.0, 6.0, 8.0]),  # Standard fresa sizes
                ))
            
            if np.random.random() < p_remove_groove and len(new_genome.grooves) > 0:
                idx = np.random.randint(len(new_genome.grooves))
                new_genome.grooves.pop(idx)
        else:
            # max_grooves=0 → CLEAR all grooves
            new_genome.grooves = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # EXCITERS: Mutate positions AND DSP params for optimal modal coupling
        # Reference: Multi-exciter optimization (Lu 2012, Bai & Liu 2004)
        # DSP params (gain, delay, phase) activated in BLOOM phase
        # 
        # BOOSTED rates for faster DSP evolution (2025-01-04):
        # - Phase/gain/delay need higher mutation to escape local optima
        # - JAB optimization requires rapid phase exploration
        # ═══════════════════════════════════════════════════════════════════════
        
        p_mutate_exciter = 0.40  # 40% probability per exciter per generation
        p_large_move = 0.10      # 10% chance of large repositioning
        p_mutate_phase = 0.50    # 50% chance to mutate phase (was 30%) - CRITICAL for JAB!
        p_mutate_gain = 0.40     # 40% chance to mutate gain (was 25%) - L/R balance
        p_mutate_delay = 0.35    # 35% chance to mutate delay (was 20%) - time alignment
        
        for i, exc in enumerate(new_genome.exciters):
            new_x, new_y = exc.x, exc.y
            new_phase = getattr(exc, 'phase_deg', 0.0)
            new_gain = getattr(exc, 'gain_db', 0.0)
            new_delay = getattr(exc, 'delay_ms', 0.0)
            
            # Position mutation
            if np.random.random() < p_mutate_exciter:
                if np.random.random() < p_large_move:
                    # Large repositioning: jump to new location
                    # Keep channel constraints: CH1/2 upper (head), CH3/4 lower (feet)
                    if exc.channel in [1, 2]:
                        new_y = np.random.uniform(0.60, 0.95)  # Upper region
                    else:
                        new_y = np.random.uniform(0.05, 0.40)  # Lower region
                    new_x = np.random.uniform(0.15, 0.85)
                else:
                    # Small position adjustment with LARGER sigma
                    new_x = np.clip(exc.x + np.random.normal(0, 0.12), 0.10, 0.90)
                    # Keep y in appropriate region based on channel
                    if exc.channel in [1, 2]:
                        new_y = np.clip(exc.y + np.random.normal(0, 0.08), 0.55, 0.95)
                    else:
                        new_y = np.clip(exc.y + np.random.normal(0, 0.08), 0.05, 0.45)
            
            # Phase mutation (independent of position)
            if np.random.random() < p_mutate_phase:
                # Either small adjustment or random jump
                if np.random.random() < 0.3:
                    # Jump to common phase values (0, 90, 180, 270)
                    new_phase = np.random.choice([0.0, 90.0, 180.0, 270.0])
                else:
                    # Small adjustment
                    new_phase = (new_phase + np.random.normal(0, 30)) % 360
            
            # Gain mutation (DSP - BLOOM phase feature)
            # Range: -12 dB to +6 dB (max power 25W per channel)
            if np.random.random() < p_mutate_gain:
                if np.random.random() < 0.2:
                    # Jump to common gain values (0, -3, -6, +3)
                    new_gain = np.random.choice([0.0, -3.0, -6.0, 3.0])
                else:
                    # Small adjustment (σ=2 dB)
                    new_gain = np.clip(new_gain + np.random.normal(0, 2.0), -12.0, 6.0)
            
            # Delay mutation (DSP - BLOOM phase feature)
            # Range: 0-50 ms for time alignment
            if np.random.random() < p_mutate_delay:
                if np.random.random() < 0.3:
                    # Jump to common delay values (0, 5, 10, 20 ms)
                    new_delay = np.random.choice([0.0, 5.0, 10.0, 20.0])
                else:
                    # Small adjustment (σ=3 ms)
                    new_delay = np.clip(new_delay + np.random.normal(0, 3.0), 0.0, 50.0)
            
            new_genome.exciters[i] = ExciterPosition(
                x=new_x, y=new_y, channel=exc.channel, 
                phase_deg=new_phase, 
                gain_db=new_gain,
                delay_ms=new_delay,
                exciter_model=exc.exciter_model
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # SPRING SUPPORTS: Mutate positions and stiffness for optimal phase rotation
        # Reference: Phase rotation modes (PHASE_ROTATION_MODES.md)
        # ═══════════════════════════════════════════════════════════════════════
        p_mutate_spring = 0.30  # 30% probability per spring per generation (was 15%)
        
        for i, spring in enumerate(new_genome.spring_supports):
            if np.random.random() < p_mutate_spring:
                new_genome.spring_supports[i] = spring.mutate(sigma=0.06)  # Larger sigma
        
        # ═══════════════════════════════════════════════════════════════════════
        # PRIORITY 1: ENFORCE STRUCTURAL SUPPORT (before symmetry!)
        # "Il petalo non sboccia se il gambo non regge"
        # Springs MUST support the plate structurally FIRST
        # ═══════════════════════════════════════════════════════════════════════
        new_genome = new_genome.enforce_structural_support()
        
        # ═══════════════════════════════════════════════════════════════════════
        # PRIORITY 2: ENFORCE SYMMETRY (if configured)
        # ═══════════════════════════════════════════════════════════════════════
        # After structural support is valid, apply bilateral symmetry
        if self.enforce_symmetry:
            new_genome = new_genome.enforce_bilateral_symmetry()
        
        return new_genome
    
    def mutate_symmetric(
        self,
        sigma_contour: float = 0.02,
        sigma_thickness: float = 0.1,
        sigma_dimensions: float = 0.02,
        p_add_cutout: float = 0.1,
        p_remove_cutout: float = 0.1,
    ) -> 'PlateGenome':
        """
        Create a mutated copy with enforced bilateral symmetry.
        
        LUTHERIE: Like violin/guitar plates, vibroacoustic tables should
        be symmetric for balanced response and aesthetics.
        
        Args:
            sigma_contour: Std dev for contour points
            sigma_thickness: Std dev for thickness field
            sigma_dimensions: Std dev for dimensions
            p_add_cutout: Probability of adding a cutout
            p_remove_cutout: Probability of removing a cutout
        
        Returns:
            New PlateGenome with enforced bilateral symmetry
        """
        # First apply normal mutation
        mutated = self.mutate(
            sigma_contour=sigma_contour,
            sigma_thickness=sigma_thickness,
            sigma_dimensions=sigma_dimensions,
            p_add_cutout=p_add_cutout,
            p_remove_cutout=p_remove_cutout,
        )
        
        # Then enforce symmetry
        return mutated.enforce_bilateral_symmetry()
    
    def _get_physics_guided_cutout(
        self,
        existing_cutouts: List['CutoutGene'],
    ) -> Optional[Dict]:
        """
        Get physics-guided cutout parameters using modal analysis and ABH theory.
        
        PHYSICS PRINCIPLE (Schleske 2002, Fletcher & Rossing 1998):
        - Cutout at ANTINODE → maximum frequency shift
        - Cutout at NODE → minimal effect on that mode
        - F-holes in violins are positioned to tune specific modes
        
        ABH PRINCIPLE (Krylov 2014, Deng 2019):
        - Edge/corner cutouts can focus acoustic energy
        - Tapered profiles create "black hole" effect
        
        Args:
            existing_cutouts: List of existing CutoutGene objects
        
        Returns:
            Dict with cutout parameters, or None if no good position found
        """
        try:
            # 70% use standard modal guidance, 30% use ABH optimizer
            use_abh = np.random.random() < 0.30
            
            if use_abh:
                # Try ABH-based placement (Krylov 2014)
                from .cutout_placement import CutoutPlacementOptimizer, CutoutPurpose as ABHPurpose
                
                optimizer = CutoutPlacementOptimizer(
                    plate_length=self.length,
                    plate_width=self.width,
                )
                
                # Get ABH suggestion for edge energy focusing
                # Randomly choose target zone for variety
                target_zone = np.random.choice(["spine", "ear", "both"])
                suggestions = optimizer.suggest_for_abh_focusing(
                    target_zone=target_zone,
                    use_spirals=np.random.random() < 0.5,  # 50% spiral shapes
                )
                
                if suggestions:
                    suggestion = suggestions[0]
                    return {
                        "x": suggestion.x,
                        "y": suggestion.y,
                        "width": suggestion.recommended_width,
                        "height": suggestion.recommended_height,
                        "rotation": 0.0,
                        "shape": suggestion.recommended_shape,
                        "corner_radius": 0.3,
                        "aspect_bias": suggestion.recommended_width / max(suggestion.recommended_height, 0.01),
                        "purpose": "abh_focus",
                        "confidence": suggestion.confidence,
                    }
            
            # Standard modal guidance (Schleske 2002)
            from .modal_guidance import ModalAnalyzer, create_physics_guided_cutout
            
            # Create analyzer for current plate dimensions
            analyzer = ModalAnalyzer(
                length=self.length,
                width=self.width,
                thickness=self.thickness_base,
            )
            
            # Compute modes
            analyzer.compute_modes(n_modes=10)
            
            # Get physics-guided suggestion - pass n_cutout for variety!
            n_cutout = len(existing_cutouts) + 1 if existing_cutouts else 1
            return create_physics_guided_cutout(
                genome=self,
                analyzer=analyzer,
                existing_cutouts=existing_cutouts,
                n_cutout=n_cutout,  # Different cutouts target different modes!
            )
        except Exception as e:
            # Fallback to None (caller will use random placement)
            import logging
            logging.getLogger(__name__).debug(f"Physics-guided cutout failed: {e}")
            return None
    
    def crossover(self, other: 'PlateGenome', alpha: float = 0.5) -> 'PlateGenome':
        """
        Crossover con un altro genoma.
        
        Args:
            other: Altro genoma genitore
            alpha: Peso blend (0.5 = media)
        
        Returns:
            Nuovo PlateGenome figlio
        """
        # Use MINIMUM of both parents - if either has 0, child has 0
        # This ensures config max_grooves=0 propagates correctly
        child_max_cutouts = min(self.max_cutouts, other.max_cutouts)
        child_max_grooves = min(self.max_grooves, other.max_grooves)
        
        # CONTOUR TYPE: If either parent has fixed_contour, inherit that contour
        # This ensures user's shape choice is preserved through evolution
        if self.fixed_contour:
            child_contour = self.contour_type
            child_fixed_contour = True
        elif other.fixed_contour:
            child_contour = other.contour_type
            child_fixed_contour = True
        else:
            # Neither parent is fixed, randomly choose
            child_contour = self.contour_type if np.random.random() < 0.5 else other.contour_type
            child_fixed_contour = False
        
        # SYMMETRY: Inherit symmetry setting (default True)
        child_enforce_symmetry = self.enforce_symmetry or other.enforce_symmetry
        
        child = PlateGenome(
            length=alpha * self.length + (1 - alpha) * other.length,
            width=alpha * self.width + (1 - alpha) * other.width,
            thickness_base=alpha * self.thickness_base + (1 - alpha) * other.thickness_base,
            contour_type=child_contour,
            fixed_contour=child_fixed_contour,
            enforce_symmetry=child_enforce_symmetry,
            max_cutouts=child_max_cutouts,
            max_grooves=child_max_grooves,
        )
        
        # Blend control points se stesso numero
        if len(self.control_points) == len(other.control_points):
            child.control_points = (
                alpha * self.control_points + 
                (1 - alpha) * other.control_points
            )
        else:
            child.control_points = self.control_points.copy()
        
        # Blend thickness field
        if self.thickness_field is not None and other.thickness_field is not None:
            if self.thickness_field.shape == other.thickness_field.shape:
                child.thickness_field = (
                    alpha * self.thickness_field + 
                    (1 - alpha) * other.thickness_field
                )
        
        # Cutouts: prendi un subset casuale da entrambi (only if enabled)
        # Use CHILD'S max_cutouts (which is min of both parents)
        if child.max_cutouts > 0:
            all_cutouts = self.cutouts + other.cutouts
            if len(all_cutouts) > 0:
                n_keep = min(len(all_cutouts), child.max_cutouts)
                indices = np.random.choice(len(all_cutouts), n_keep, replace=False)
                child.cutouts = [all_cutouts[i] for i in indices]
        else:
            child.cutouts = []
        
        # Grooves: prendi un subset casuale da entrambi (only if enabled)
        # Use CHILD'S max_grooves (which is min of both parents)
        if child.max_grooves > 0:
            all_grooves = self.grooves + other.grooves
            if len(all_grooves) > 0:
                n_keep = min(len(all_grooves), child.max_grooves)
                indices = np.random.choice(len(all_grooves), n_keep, replace=False)
                child.grooves = [all_grooves[i] for i in indices]
        else:
            child.grooves = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # SPRING SUPPORTS: Blend stiffness/damping, keep positions from fitter parent
        # ═══════════════════════════════════════════════════════════════════════
        # Springs are critical for phase rotation - blend their properties
        if len(self.spring_supports) == len(other.spring_supports):
            child.spring_supports = []
            for s1, s2 in zip(self.spring_supports, other.spring_supports):
                # Blend properties
                child.spring_supports.append(SpringSupportGene(
                    x=alpha * s1.x + (1 - alpha) * s2.x,
                    y=alpha * s1.y + (1 - alpha) * s2.y,
                    stiffness_n_m=alpha * s1.stiffness_n_m + (1 - alpha) * s2.stiffness_n_m,
                    damping_ratio=alpha * s1.damping_ratio + (1 - alpha) * s2.damping_ratio,
                ))
        else:
            # Different number of supports - keep from first parent
            child.spring_supports = [copy.deepcopy(s) for s in self.spring_supports]
        
        # ═══════════════════════════════════════════════════════════════════════
        # EXCITERS: Blend positions AND DSP params, keep channel assignments
        # ═══════════════════════════════════════════════════════════════════════
        # Exciters are matched by channel
        child.exciters = []
        for exc1 in self.exciters:
            # Find matching exciter in other parent by channel
            exc2 = next((e for e in other.exciters if e.channel == exc1.channel), exc1)
            # Blend phases (circular average for angles near 0/360)
            phase1 = getattr(exc1, 'phase_deg', 0.0)
            phase2 = getattr(exc2, 'phase_deg', 0.0)
            blended_phase = (alpha * phase1 + (1 - alpha) * phase2) % 360
            
            # Blend DSP params (gain and delay)
            gain1 = getattr(exc1, 'gain_db', 0.0)
            gain2 = getattr(exc2, 'gain_db', 0.0)
            blended_gain = alpha * gain1 + (1 - alpha) * gain2
            
            delay1 = getattr(exc1, 'delay_ms', 0.0)
            delay2 = getattr(exc2, 'delay_ms', 0.0)
            blended_delay = alpha * delay1 + (1 - alpha) * delay2
            
            child.exciters.append(ExciterPosition(
                x=alpha * exc1.x + (1 - alpha) * exc2.x,
                y=alpha * exc1.y + (1 - alpha) * exc2.y,
                channel=exc1.channel,
                phase_deg=blended_phase,
                gain_db=blended_gain,
                delay_ms=blended_delay,
                exciter_model=exc1.exciter_model,
            ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # ENFORCE SYMMETRY (if enabled)
        # ═══════════════════════════════════════════════════════════════════════
        if child_enforce_symmetry:
            child = child.enforce_bilateral_symmetry()
        
        return child
    
    # ─────────────────────────────────────────────────────────────────────────
    # Conversione a Mesh
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_vertices(self, n_points: int = 64) -> np.ndarray:
        """
        Genera vertici contorno in coordinate assolute.
        
        Returns:
            Array (n, 2) con coordinate [m]
        """
        # Prendi punti controllo o genera da tipo
        if len(self.control_points) >= 4:
            base_points = self.control_points
        else:
            base_points = self._generate_base_contour(n_points)
        
        # Interpola se necessario
        if len(base_points) < n_points:
            # Spline interpolation
            from scipy.interpolate import splprep, splev
            try:
                # Chiudi il contorno
                pts = np.vstack([base_points, base_points[0]])
                tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
                u_new = np.linspace(0, 1, n_points)
                x_new, y_new = splev(u_new, tck)
                base_points = np.column_stack([x_new, y_new])
            except (ValueError, TypeError) as e:
                # Spline fitting failed, keep original points
                logger.debug(f"Spline interpolation failed: {e}, keeping original points")
                pass
        
        # Scala a dimensioni reali
        vertices = base_points.copy()
        vertices[:, 0] = (vertices[:, 0] - 0.5) * self.length
        vertices[:, 1] = (vertices[:, 1] - 0.5) * self.width
        
        # Centra su origine
        vertices[:, 0] += self.length / 2
        vertices[:, 1] += self.width / 2
        
        return vertices
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Bounding box del contorno.
        
        Returns:
            (x_min, y_min, x_max, y_max) in metri
        """
        return (0, 0, self.length, self.width)
    
    def get_area(self) -> float:
        """Area approssimata [m²]."""
        if self.contour_type in [ContourType.RECTANGLE, ContourType.GOLDEN_RECT]:
            return self.length * self.width
        elif self.contour_type == ContourType.ELLIPSE:
            return np.pi * (self.length / 2) * (self.width / 2)
        else:
            # Shoelace formula per poligono
            v = self.to_vertices(64)
            n = len(v)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += v[i, 0] * v[j, 1]
                area -= v[j, 0] * v[i, 1]
            return abs(area) / 2
    
    def get_mass(self, density: float = 680.0) -> float:
        """
        Massa tavola [kg].
        
        Args:
            density: Densità materiale [kg/m³]
        """
        area = self.get_area()
        
        # Sottrai area cutouts
        for cutout in self.cutouts:
            cut_area = np.pi * (cutout.width * self.length / 2) * (cutout.height * self.width / 2)
            area -= cut_area
        
        return max(0, area * self.thickness_base * density)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Serializzazione
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict:
        """Converte in dizionario."""
        return {
            "length": self.length,
            "width": self.width,
            "thickness_base": self.thickness_base,
            "contour_type": self.contour_type.value,
            "control_points": self.control_points.tolist() if len(self.control_points) > 0 else [],
            "cutouts": [
                {"x": c.x, "y": c.y, "width": c.width, "height": c.height, "rotation": c.rotation}
                for c in self.cutouts
            ],
            "fitness": self.fitness,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlateGenome':
        """Crea da dizionario."""
        genome = cls(
            length=data["length"],
            width=data["width"],
            thickness_base=data["thickness_base"],
            contour_type=ContourType(data["contour_type"]),
        )
        if data.get("control_points"):
            genome.control_points = np.array(data["control_points"])
        if data.get("cutouts"):
            genome.cutouts = [
                CutoutGene(**c) for c in data["cutouts"]
            ]
        genome.fitness = data.get("fitness", 0.0)
        return genome
    
    def __repr__(self) -> str:
        # Handle case where contour_type might not be a ContourType enum
        if hasattr(self.contour_type, 'value'):
            contour_str = self.contour_type.value
        else:
            contour_str = str(self.contour_type)
        return (
            f"PlateGenome({contour_str}, "
            f"{self.length:.2f}×{self.width:.2f}×{self.thickness_base*1000:.1f}mm, "
            f"cuts={len(self.cutouts)}, fitness={self.fitness:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_genome_for_person(
    person: 'Person',
    contour_type: ContourType = ContourType.GOLDEN_RECT,
    material_density: float = 680.0,
    allow_cutouts: bool = True,  # Default TRUE (liuteria approach)
    max_cutouts: int = 4,  # Number of cutouts allowed
    max_grooves: int = 0,  # Number of grooves allowed (default OFF)
) -> PlateGenome:
    """
    Crea un genoma iniziale ottimale per una persona.
    
    LUTHERIE APPROACH:
    I cutouts (fori) sono abilitati di default perché sono il principale
    strumento di accordatura nella liuteria tradizionale. Come le f-holes
    del violino accordano la risonanza di Helmholtz, i cutouts spostano
    le frequenze modali verso i target vibroacustici.
    
    Reference: Schleske (2002) "Empirical Tools in Contemporary Violin Making"
    
    Args:
        person: Modello persona
        contour_type: Tipo forma
        material_density: Densità materiale [kg/m³]
        allow_cutouts: Permetti tagli interni (default True per liuteria)
        max_cutouts: Numero massimo cutouts (passato da UI/pipeline)
        max_grooves: Numero massimo grooves (passato da UI/pipeline)
    
    Returns:
        PlateGenome inizializzato
    """
    # Initial cutouts: diverse shapes for optimal modal tuning
    # L'ottimizzatore parte da forme diverse e può mutarle liberamente
    initial_cutouts = []
    if allow_cutouts and max_cutouts > 0:
        # Pattern iniziale con forme diverse (l'optimizer evolverà quelle migliori):
        # - f-hole simmetrici nella zona toracica
        # - slot/slot nella zona lombare
        # - forme libere che l'optimizer può mutare
        initial_cutouts = [
            # Zona toracica: f-holes per risonanza
            CutoutGene(x=0.28, y=0.42, width=0.035, height=0.08, 
                       shape="f_hole", rotation=0.1),
            CutoutGene(x=0.72, y=0.42, width=0.035, height=0.08, 
                       shape="f_hole", rotation=-0.1),
            # Zona lombare: slot per bassi
            CutoutGene(x=0.35, y=0.62, width=0.06, height=0.025, 
                       shape="slot", rotation=0.0),
            CutoutGene(x=0.65, y=0.62, width=0.06, height=0.025, 
                       shape="slot", rotation=0.0),
        ][:max_cutouts]  # Limit to max_cutouts
    
    # Initial grooves: subtle thinning for fine tuning
    # Like violin plate graduations (thinner in center for lower modes)
    initial_grooves = []
    if allow_cutouts and max_grooves > 0:  # Enable grooves when requested
        initial_grooves = [
            # Transverse grooves across spine zone for mode 1-2 tuning
            GrooveGene(x=0.5, y=0.50, length=0.12, angle=0.0, depth=0.25, width_mm=5.0),
            # Angled grooves for torsional mode tuning
            GrooveGene(x=0.35, y=0.55, length=0.08, angle=0.4, depth=0.20, width_mm=4.0),
            GrooveGene(x=0.65, y=0.55, length=0.08, angle=-0.4, depth=0.20, width_mm=4.0),
        ][:max_grooves]  # Limit to max_grooves
    
    return PlateGenome(
        length=person.recommended_plate_length,
        width=person.recommended_plate_width,
        thickness_base=0.015,  # 15mm default
        contour_type=contour_type,
        max_cutouts=max_cutouts,
        max_grooves=max_grooves,
        cutouts=initial_cutouts,
        grooves=initial_grooves,
    )


def create_random_population(
    n: int,
    person: 'Person',
    contour_types: Optional[List[ContourType]] = None,
    max_cutouts: int = 4,
    max_grooves: int = 0,
    fixed_contour: bool = False,
) -> List[PlateGenome]:
    """
    Crea popolazione iniziale per algoritmo genetico.
    
    Args:
        n: Dimensione popolazione
        person: Modello persona
        contour_types: Tipi contorno da usare (None = tutti)
        max_cutouts: Numero massimo cutouts per genoma
        max_grooves: Numero massimo grooves per genoma
        fixed_contour: Se True, il contour non può cambiare durante evoluzione
    
    Returns:
        Lista di PlateGenome
    """
    if contour_types is None:
        contour_types = [
            ContourType.RECTANGLE,
            ContourType.GOLDEN_RECT,
            ContourType.ELLIPSE,
            ContourType.OVOID,
        ]
    
    population = []
    for i in range(n):
        ct = contour_types[i % len(contour_types)]
        genome = create_genome_for_person(
            person, ct,
            max_cutouts=max_cutouts,
            max_grooves=max_grooves,
        )
        
        # Set fixed_contour BEFORE mutation
        # This prevents contour type from changing during initial mutation
        genome.fixed_contour = fixed_contour
        genome.enforce_symmetry = True  # Default: symmetric for acoustic balance
        
        # Aggiungi variazione iniziale (won't change contour if fixed)
        genome = genome.mutate(sigma_dimensions=0.05)
        population.append(genome)
    
    return population


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PLATE GENOME TEST")
    print("=" * 60)
    
    # Test base
    genome = PlateGenome(
        length=1.85,
        width=0.64,
        contour_type=ContourType.GOLDEN_RECT,
    )
    print(f"\nBase genome: {genome}")
    print(f"Area: {genome.get_area():.3f} m²")
    print(f"Mass (birch): {genome.get_mass():.2f} kg")
    
    # Test mutazione
    print("\nMutation test:")
    for i in range(5):
        mutated = genome.mutate()
        print(f"  {i+1}: {mutated}")
    
    # Test crossover
    print("\nCrossover test:")
    parent1 = PlateGenome(length=1.80, width=0.60, contour_type=ContourType.ELLIPSE)
    parent2 = PlateGenome(length=1.90, width=0.70, contour_type=ContourType.OVOID)
    child = parent1.crossover(parent2)
    print(f"  Parent 1: {parent1}")
    print(f"  Parent 2: {parent2}")
    print(f"  Child:    {child}")
    
    # Test vertices
    print("\nVertices test:")
    vertices = genome.to_vertices(16)
    print(f"  Shape: {vertices.shape}")
    print(f"  Bounds: x=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], "
          f"y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
