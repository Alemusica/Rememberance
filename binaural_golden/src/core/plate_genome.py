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


# Cutout shapes available for optimization
# L'optimizer può usare qualsiasi forma, incluso FREEFORM con poligoni arbitrari
# Tutte le forme sono lavorabili con CNC/fresa
# Cutout shapes - ONLY ORGANIC/CURVED shapes
# Nature doesn't make straight lines! Good acoustic design follows nature.
# Reference: Stradivari f-holes, Schleske violin acoustics
CUTOUT_SHAPES = [
    # === FORME ORGANICHE (come in natura) ===
    "ellipse",      # Standard f-hole style (violin/cello)
    "circle",       # Cerchio perfetto
    "crescent",     # Mezzaluna - bilanciamento modi asimmetrici
    "tear",         # Goccia - per zone di stress direzionale
    "f_hole",       # Stylized f-hole (double curve like violin)
    "kidney",       # Forma a rene (sound port style)
    "s_curve",      # Curva a S (per modo torsionale)
    "vesica",       # Vesica piscis (sacred geometry)
    "spiral",       # Spirale logaritmica (golden ratio)
    "leaf",         # Forma a foglia (naturale)
    "wave",         # Onda sinusoidale chiusa
    # === FORMA LIBERA (fresa manuale) ===
    "freeform",     # Poligono arbitrario con curve smooth
]

@dataclass
class CutoutGene:
    """
    Gene per un taglio interno (foro passante).
    
    LIBERTÀ TOTALE per l'ottimizzatore:
    - Qualsiasi forma predefinita O freeform (poligono arbitrario)
    - Qualsiasi dimensione (width/height indipendenti)
    - Qualsiasi rotazione (0-2π)
    - Qualsiasi posizione (evitando bordi)
    - Control points per forme freeform (fresa manuale)
    
    Per FREEFORM:
    - control_points: Lista di (x, y) normalizzati [0-1] relativi al centro
    - La fresa manuale può creare qualsiasi forma
    
    Posizione e dimensioni normalizzate [0, 1].
    """
    x: float              # Centro X normalizzato
    y: float              # Centro Y normalizzato
    width: float          # Larghezza normalizzata (libera: 0.01-0.3)
    height: float         # Altezza normalizzata (libera: 0.01-0.3)
    rotation: float = 0.0 # Rotazione in radianti (libera: 0-2π)
    shape: str = "ellipse"  # Forma (libera scelta tra CUTOUT_SHAPES)
    corner_radius: float = 0.3  # Per rounded_rect: raggio angoli (0-1)
    aspect_bias: float = 1.0    # Bias aspetto (0.3-3.0): <1 = wide, >1 = tall
    # FREEFORM: punti di controllo per poligono arbitrario (fresa manuale)
    control_points: Optional[np.ndarray] = None  # Shape (N, 2), punti relativi al centro
    
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
    
    def mutate(self, sigma: float = 0.05) -> 'CutoutGene':
        """Muta questo cutout con libertà totale."""
        # Possibilità di cambiare forma durante mutazione
        new_shape = self.shape
        if np.random.random() < 0.15:  # 15% chance di cambio forma
            new_shape = np.random.choice(CUTOUT_SHAPES)
        
        # Width e height indipendenti (nessun vincolo aspect ratio)
        new_width = np.clip(self.width + np.random.normal(0, sigma * 0.8), 0.01, 0.3)
        new_height = np.clip(self.height + np.random.normal(0, sigma * 0.8), 0.01, 0.3)
        
        # Muta control points per freeform
        new_control_points = None
        if new_shape == "freeform":
            if self.control_points is not None:
                # Muta punti esistenti
                noise = np.random.normal(0, sigma * 0.5, self.control_points.shape)
                new_control_points = np.clip(self.control_points + noise, -1.0, 1.0)
                # Possibilità di aggiungere/rimuovere punto
                if np.random.random() < 0.1 and len(new_control_points) > 4:
                    idx = np.random.randint(len(new_control_points))
                    new_control_points = np.delete(new_control_points, idx, axis=0)
                elif np.random.random() < 0.1 and len(new_control_points) < 12:
                    new_pt = np.random.uniform(-1, 1, (1, 2))
                    new_control_points = np.vstack([new_control_points, new_pt])
            else:
                # Genera nuovi control points
                n_points = np.random.randint(5, 9)
                angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
                radii = np.random.uniform(0.3, 1.0, n_points)
                new_control_points = np.column_stack([
                    radii * np.cos(angles),
                    radii * np.sin(angles)
                ])
        
        return CutoutGene(
            x=np.clip(self.x + np.random.normal(0, sigma), 0.05, 0.95),
            y=np.clip(self.y + np.random.normal(0, sigma), 0.05, 0.95),
            width=new_width,
            height=new_height,
            rotation=(self.rotation + np.random.normal(0, 0.3)) % (2 * np.pi),
            shape=new_shape,
            corner_radius=np.clip(self.corner_radius + np.random.normal(0, 0.1), 0.0, 1.0),
            aspect_bias=np.clip(self.aspect_bias + np.random.normal(0, 0.2), 0.3, 3.0),
            control_points=new_control_points,
        )
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """Converti in coordinate assolute [m]."""
        return {
            "center": (self.x * plate_length, self.y * plate_width),
            "size": (self.width * plate_length, self.height * plate_width),
            "rotation": self.rotation,
            "shape": self.shape,
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
        """Converti in coordinate assolute."""
        return {
            "center": (self.x * plate_length, self.y * plate_width),
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
class ExciterPosition:
    """
    Posizione di un eccitatore sulla tavola.
    
    Sistema: 4× Dayton DAEX25 (25mm, 40W, 8Ω)
    - CH1, CH2: Head (stereo left/right)
    - CH3, CH4: Feet (stereo left/right)
    
    Reference: JAB4_DOCUMENTATION.md
    """
    x: float              # Posizione X normalizzata (0=left, 1=right)
    y: float              # Posizione Y normalizzata (0=feet, 1=head)
    channel: int          # Canale JAB4 (1-4)
    exciter_model: str = "dayton_daex25"  # Key in EXCITERS database
    
    # Derivati dal modello
    diameter_mm: float = 25.0     # Dayton DAEX25
    power_w: float = 10.0         # RMS power
    impedance_ohm: float = 4.0    # 4Ω version
    
    def to_absolute(self, plate_length: float, plate_width: float) -> Dict:
        """Converti in coordinate assolute [m]."""
        return {
            "center": (self.x * plate_width, self.y * plate_length),  # Note: x=width, y=length
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
        # SYMMETRIC CUTOUTS
        # ═══════════════════════════════════════════════════════════════════════
        symmetric_cutouts = []
        for cutout in self.cutouts:
            # Keep original
            symmetric_cutouts.append(cutout)
            
            # If not on centerline, add mirrored version
            if abs(cutout.x - 0.5) > 0.05:  # More than 5% from center
                mirrored = CutoutGene(
                    x=1.0 - cutout.x,
                    y=cutout.y,
                    width=cutout.width,
                    height=cutout.height,
                    rotation=-cutout.rotation,  # Mirror rotation
                    shape=cutout.shape,
                    corner_radius=cutout.corner_radius,
                    aspect_bias=cutout.aspect_bias,
                    control_points=self._mirror_control_points(cutout.control_points) if cutout.control_points is not None else None,
                )
                # Only add if we have room
                if len(symmetric_cutouts) < self.max_cutouts:
                    symmetric_cutouts.append(mirrored)
        
        new_genome.cutouts = symmetric_cutouts[:self.max_cutouts]
        
        # ═══════════════════════════════════════════════════════════════════════
        # SYMMETRIC GROOVES
        # ═══════════════════════════════════════════════════════════════════════
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # SYMMETRIC EXCITERS
        # ═══════════════════════════════════════════════════════════════════════
        # Pair CH1 with CH2 (head), CH3 with CH4 (feet)
        # Use same distance from center for paired channels
        
        for i, exc in enumerate(new_genome.exciters):
            if exc.channel == 1:
                # Find CH2 and mirror
                for j, exc2 in enumerate(new_genome.exciters):
                    if exc2.channel == 2:
                        # Make CH2 mirror of CH1
                        new_genome.exciters[j] = ExciterPosition(
                            x=1.0 - exc.x,
                            y=exc.y,  # Same Y
                            channel=2,
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
                            exciter_model=exc.exciter_model,
                        )
                        break
        
        return new_genome
    
    def _mirror_control_points(self, points: np.ndarray) -> np.ndarray:
        """Mirror control points around x=0 (for freeform cutouts)."""
        if points is None:
            return None
        mirrored = points.copy()
        mirrored[:, 0] = -mirrored[:, 0]  # Flip x coordinates
        return mirrored
    
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
        new_genome.length = np.clip(
            self.length * (1 + np.random.normal(0, sigma_dimensions)),
            1.4, 2.2
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
        # CONTORNO TAVOLA: può mutare tipo e forma
        # ═══════════════════════════════════════════════════════════════════════
        
        # 10% chance di cambiare tipo contorno (higher to explore shapes)
        if np.random.random() < 0.10:
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
                    new_genome.cutouts.append(CutoutGene(
                        x=np.random.uniform(0.1, 0.9),
                        y=np.random.uniform(0.15, 0.85),
                        width=np.random.uniform(0.02, 0.15),
                        height=np.random.uniform(0.02, 0.15),
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
        
        new_genome.grooves = [g.mutate(0.02) for g in self.grooves]
        
        # Probabilità di aggiungere/rimuovere groove
        p_add_groove = 0.08
        p_remove_groove = 0.08
        
        if self.max_grooves > 0:
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # EXCITERS: Mutate positions LIBERAMENTE per massimo coupling
        # ═══════════════════════════════════════════════════════════════════════
        # Gli exciters possono muoversi significativamente per ottimizzare
        # l'accoppiamento modale nelle zone target (spine/head)
        
        p_mutate_exciter = 0.20  # 20% probability per exciter per generation
        p_large_move = 0.05      # 5% chance of large repositioning
        
        for i, exc in enumerate(new_genome.exciters):
            if np.random.random() < p_mutate_exciter:
                if np.random.random() < p_large_move:
                    # Large repositioning: jump to new location
                    # Keep channel constraints: CH1/2 upper (head), CH3/4 lower (feet)
                    if exc.channel in [1, 2]:
                        new_y = np.random.uniform(0.65, 0.95)  # Upper region
                    else:
                        new_y = np.random.uniform(0.05, 0.35)  # Lower region
                    new_x = np.random.uniform(0.15, 0.85)
                else:
                    # Small position adjustment
                    new_x = np.clip(exc.x + np.random.normal(0, 0.08), 0.10, 0.90)
                    # Keep y in appropriate region based on channel
                    if exc.channel in [1, 2]:
                        new_y = np.clip(exc.y + np.random.normal(0, 0.05), 0.55, 0.95)
                    else:
                        new_y = np.clip(exc.y + np.random.normal(0, 0.05), 0.05, 0.45)
                
                new_genome.exciters[i] = ExciterPosition(
                    x=new_x, y=new_y, channel=exc.channel, exciter_model=exc.exciter_model
                )
        
        # ═══════════════════════════════════════════════════════════════════════
        # ENFORCE SYMMETRY (if configured)
        # ═══════════════════════════════════════════════════════════════════════
        # After all mutations, apply bilateral symmetry if enabled
        # This ensures acoustically balanced designs like violin plates
        if getattr(self, '_enforce_symmetry', False):
            new_genome = new_genome.enforce_bilateral_symmetry()
            new_genome._enforce_symmetry = True
        
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
        Get physics-guided cutout parameters using modal analysis.
        
        PHYSICS PRINCIPLE (Schleske 2002, Fletcher & Rossing 1998):
        - Cutout at ANTINODE → maximum frequency shift
        - Cutout at NODE → minimal effect on that mode
        - F-holes in violins are positioned to tune specific modes
        
        Args:
            existing_cutouts: List of existing CutoutGene objects
        
        Returns:
            Dict with cutout parameters, or None if no good position found
        """
        try:
            from .modal_guidance import ModalAnalyzer, create_physics_guided_cutout
            
            # Create analyzer for current plate dimensions
            analyzer = ModalAnalyzer(
                length=self.length,
                width=self.width,
                thickness=self.thickness_base,
            )
            
            # Compute modes
            analyzer.compute_modes(n_modes=10)
            
            # Get physics-guided suggestion
            return create_physics_guided_cutout(
                genome=self,
                analyzer=analyzer,
                existing_cutouts=existing_cutouts,
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
        child = PlateGenome(
            length=alpha * self.length + (1 - alpha) * other.length,
            width=alpha * self.width + (1 - alpha) * other.width,
            thickness_base=alpha * self.thickness_base + (1 - alpha) * other.thickness_base,
            contour_type=self.contour_type if np.random.random() < 0.5 else other.contour_type,
            max_cutouts=self.max_cutouts,
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
        
        # Cutouts: prendi un subset casuale da entrambi
        all_cutouts = self.cutouts + other.cutouts
        if len(all_cutouts) > 0:
            n_keep = min(len(all_cutouts), self.max_cutouts)
            indices = np.random.choice(len(all_cutouts), n_keep, replace=False)
            child.cutouts = [all_cutouts[i] for i in indices]
        
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
            except:
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
        return (
            f"PlateGenome({self.contour_type.value}, "
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
    
    Returns:
        PlateGenome inizializzato
    """
    # Initial cutouts: diverse shapes for optimal modal tuning
    # L'ottimizzatore parte da forme diverse e può mutarle liberamente
    initial_cutouts = []
    if allow_cutouts:
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
        ]
    
    # Initial grooves: subtle thinning for fine tuning
    # Like violin plate graduations (thinner in center for lower modes)
    initial_grooves = []
    if allow_cutouts:  # Enable grooves when cutouts enabled
        initial_grooves = [
            # Transverse grooves across spine zone for mode 1-2 tuning
            GrooveGene(x=0.5, y=0.50, length=0.12, angle=0.0, depth=0.25, width_mm=5.0),
            # Angled grooves for torsional mode tuning
            GrooveGene(x=0.35, y=0.55, length=0.08, angle=0.4, depth=0.20, width_mm=4.0),
            GrooveGene(x=0.65, y=0.55, length=0.08, angle=-0.4, depth=0.20, width_mm=4.0),
        ]
    
    return PlateGenome(
        length=person.recommended_plate_length,
        width=person.recommended_plate_width,
        thickness_base=0.015,  # 15mm default
        contour_type=contour_type,
        max_cutouts=4 if allow_cutouts else 0,
        cutouts=initial_cutouts,
        grooves=initial_grooves,
    )


def create_random_population(
    n: int,
    person: 'Person',
    contour_types: Optional[List[ContourType]] = None,
) -> List[PlateGenome]:
    """
    Crea popolazione iniziale per algoritmo genetico.
    
    Args:
        n: Dimensione popolazione
        person: Modello persona
        contour_types: Tipi contorno da usare (None = tutti)
    
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
        genome = create_genome_for_person(person, ct)
        
        # Aggiungi variazione iniziale
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
