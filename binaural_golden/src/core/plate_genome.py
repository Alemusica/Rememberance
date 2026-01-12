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
    FREEFORM = "freeform"  # Spline libera


@dataclass
class CutoutGene:
    """
    Gene per un taglio interno.
    
    Posizione e dimensioni normalizzate [0, 1].
    """
    x: float              # Centro X normalizzato
    y: float              # Centro Y normalizzato
    width: float          # Larghezza normalizzata
    height: float         # Altezza normalizzata
    rotation: float = 0.0 # Rotazione in radianti
    shape: str = "ellipse"  # 'ellipse' o 'rectangle'
    
    def mutate(self, sigma: float = 0.05) -> 'CutoutGene':
        """Muta questo cutout."""
        return CutoutGene(
            x=np.clip(self.x + np.random.normal(0, sigma), 0.1, 0.9),
            y=np.clip(self.y + np.random.normal(0, sigma), 0.1, 0.9),
            width=np.clip(self.width + np.random.normal(0, sigma * 0.5), 0.02, 0.2),
            height=np.clip(self.height + np.random.normal(0, sigma * 0.5), 0.02, 0.2),
            rotation=self.rotation + np.random.normal(0, 0.2),
            shape=self.shape,
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
class PlateGenome:
    """
    Rappresentazione genetica di una tavola vibroacustica.
    
    Attributes:
        length: Lunghezza tavola [m]
        width: Larghezza tavola [m]
        thickness_base: Spessore base [m]
        contour_type: Tipo di contorno
        control_points: Punti controllo spline (N, 2) normalizzati
        thickness_field: Campo spessore opzionale (nx, ny)
        cutouts: Lista tagli interni
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
    
    # Tagli interni
    cutouts: List[CutoutGene] = field(default_factory=list)
    max_cutouts: int = 0  # 0 = nessun taglio
    
    # Fitness (calcolato dall'evaluator)
    fitness: float = 0.0
    
    def __post_init__(self):
        """Inizializza punti controllo se vuoti."""
        if len(self.control_points) == 0:
            self.control_points = self._generate_base_contour()
    
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
        
        # Muta punti contorno
        if len(self.control_points) > 0 and self.contour_type == ContourType.FREEFORM:
            noise = np.random.normal(0, sigma_contour, self.control_points.shape)
            new_genome.control_points = np.clip(
                self.control_points + noise, 0.05, 0.95
            )
        
        # Muta campo spessore
        if self.thickness_field is not None:
            noise = np.random.normal(0, sigma_thickness, self.thickness_field.shape)
            new_genome.thickness_field = np.clip(
                self.thickness_field + noise, 0.5, 1.5
            )
        
        # Muta cutouts esistenti
        new_genome.cutouts = [c.mutate(0.03) for c in self.cutouts]
        
        # Aggiungi/rimuovi cutouts
        if self.max_cutouts > 0:
            if np.random.random() < p_add_cutout and len(new_genome.cutouts) < self.max_cutouts:
                new_genome.cutouts.append(CutoutGene(
                    x=np.random.uniform(0.2, 0.8),
                    y=np.random.uniform(0.3, 0.7),
                    width=np.random.uniform(0.03, 0.08),
                    height=np.random.uniform(0.03, 0.08),
                ))
            
            if np.random.random() < p_remove_cutout and len(new_genome.cutouts) > 0:
                idx = np.random.randint(len(new_genome.cutouts))
                new_genome.cutouts.pop(idx)
        
        return new_genome
    
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
    allow_cutouts: bool = False,
) -> PlateGenome:
    """
    Crea un genoma iniziale ottimale per una persona.
    
    Args:
        person: Modello persona
        contour_type: Tipo forma
        material_density: Densità materiale [kg/m³]
        allow_cutouts: Permetti tagli interni
    
    Returns:
        PlateGenome inizializzato
    """
    return PlateGenome(
        length=person.recommended_plate_length,
        width=person.recommended_plate_width,
        thickness_base=0.015,  # 15mm default
        contour_type=contour_type,
        max_cutouts=5 if allow_cutouts else 0,
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
