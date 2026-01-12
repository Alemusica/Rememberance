"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   FITNESS EVALUATOR - Multi-Objective Scoring                ║
║                                                                              ║
║   Valuta la fitness di un PlateGenome rispetto a obiettivi multipli:         ║
║   • Risposta frequenza piatta (20-200 Hz)                                    ║
║   • Accoppiamento spina dorsale                                              ║
║   • Peso minimo tavola                                                       ║
║   • Producibilità (no forme impossibili)                                     ║
║                                                                              ║
║   La fitness viene calcolata usando FEM semplificato (analitico) o           ║
║   completo (scikit-fem) se disponibile.                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum

# Local imports
from .person import Person, SPINE_ZONES
from .plate_genome import PlateGenome, ContourType
from .plate_physics import Material, MATERIALS


@dataclass
class ObjectiveWeights:
    """
    Pesi per gli obiettivi di ottimizzazione.
    
    Tutti i pesi devono essere >= 0. Verranno normalizzati automaticamente.
    """
    flatness: float = 1.0        # Risposta frequenza piatta
    spine_coupling: float = 2.0  # Accoppiamento spina dorsale (priorità)
    low_mass: float = 0.3        # Peso minimo tavola
    manufacturability: float = 0.5  # Facilità produzione
    
    def normalized(self) -> 'ObjectiveWeights':
        """Restituisce pesi normalizzati (somma = 1)."""
        total = self.flatness + self.spine_coupling + self.low_mass + self.manufacturability
        if total <= 0:
            total = 1.0
        return ObjectiveWeights(
            flatness=self.flatness / total,
            spine_coupling=self.spine_coupling / total,
            low_mass=self.low_mass / total,
            manufacturability=self.manufacturability / total,
        )


@dataclass
class FitnessResult:
    """
    Risultato valutazione fitness.
    
    Contiene score individuali e totale, più dati diagnostici.
    """
    # Score individuali [0, 1]
    flatness_score: float = 0.0
    spine_coupling_score: float = 0.0
    low_mass_score: float = 0.0
    manufacturability_score: float = 0.0
    
    # Score totale pesato
    total_fitness: float = 0.0
    
    # Dati diagnostici
    frequencies: List[float] = field(default_factory=list)
    mode_shapes: Optional[np.ndarray] = None
    frequency_response: Optional[Tuple[np.ndarray, np.ndarray]] = None
    spine_response: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        return (
            f"Fitness({self.total_fitness:.3f}: "
            f"flat={self.flatness_score:.2f}, spine={self.spine_coupling_score:.2f}, "
            f"mass={self.low_mass_score:.2f}, manuf={self.manufacturability_score:.2f})"
        )


class FitnessEvaluator:
    """
    Valutatore di fitness per PlateGenome.
    
    Calcola un punteggio multi-obiettivo basato su:
    1. Risposta in frequenza piatta nella banda target
    2. Accoppiamento vibrazionale sulla spina dorsale
    3. Peso minimo della tavola
    4. Producibilità della forma
    """
    
    def __init__(
        self,
        person: Person,
        objectives: Optional[ObjectiveWeights] = None,
        material: str = "birch_plywood",
        freq_range: Tuple[float, float] = (20.0, 200.0),
        n_freq_points: int = 50,
        n_modes: int = 15,
    ):
        """
        Inizializza evaluator.
        
        Args:
            person: Modello persona
            objectives: Pesi obiettivi (default se None)
            material: Nome materiale da MATERIALS
            freq_range: Range frequenze target [Hz]
            n_freq_points: Punti per calcolo risposta
            n_modes: Numero modi per FEM
        """
        self.person = person
        self.objectives = (objectives or ObjectiveWeights()).normalized()
        self.material = MATERIALS.get(material, MATERIALS["birch_plywood"])
        self.freq_range = freq_range
        self.n_freq_points = n_freq_points
        self.n_modes = n_modes
        
        # Frequenze di test
        self.test_frequencies = np.linspace(
            freq_range[0], freq_range[1], n_freq_points
        )
        
        # Posizioni spina per test coupling
        self.spine_positions = self._compute_spine_positions()
        
        # Cache per evitare ricalcoli
        self._cache: Dict[str, FitnessResult] = {}
    
    def _compute_spine_positions(self) -> np.ndarray:
        """Calcola posizioni di test sulla spina dorsale."""
        positions = []
        
        # Punti lungo la spina (normalizzati)
        for zone_name, (start, end) in SPINE_ZONES.items():
            n_points = 5
            for i in range(n_points):
                x = start + (end - start) * i / (n_points - 1)
                y = 0.5  # Centro (spina)
                positions.append([x, y])
        
        return np.array(positions)
    
    def evaluate(self, genome: PlateGenome) -> FitnessResult:
        """
        Valuta fitness di un genoma.
        
        Args:
            genome: PlateGenome da valutare
        
        Returns:
            FitnessResult con scores e diagnostici
        """
        result = FitnessResult()
        
        # 1. Calcola modi propri (FEM semplificato)
        frequencies, mode_shapes = self._compute_modes(genome)
        result.frequencies = frequencies
        result.mode_shapes = mode_shapes
        
        # 2. Calcola risposta in frequenza
        freq_response = self._compute_frequency_response(genome, frequencies, mode_shapes)
        result.frequency_response = freq_response
        
        # 3. Calcola risposta sulla spina
        spine_response = self._compute_spine_response(genome, frequencies, mode_shapes)
        result.spine_response = spine_response
        
        # 4. Score: flatness
        result.flatness_score = self._score_flatness(freq_response)
        
        # 5. Score: spine coupling
        result.spine_coupling_score = self._score_spine_coupling(spine_response)
        
        # 6. Score: low mass
        result.low_mass_score = self._score_low_mass(genome)
        
        # 7. Score: manufacturability
        result.manufacturability_score = self._score_manufacturability(genome)
        
        # Score totale pesato
        result.total_fitness = (
            self.objectives.flatness * result.flatness_score +
            self.objectives.spine_coupling * result.spine_coupling_score +
            self.objectives.low_mass * result.low_mass_score +
            self.objectives.manufacturability * result.manufacturability_score
        )
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Calcolo Modi Propri (FEM Semplificato)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_modes(
        self,
        genome: PlateGenome
    ) -> Tuple[List[float], np.ndarray]:
        """
        Calcola frequenze e modi propri.
        
        Usa approssimazione analitica per piastra rettangolare:
        f_mn = (π/2) * sqrt(D/(ρh)) * ((m/L)² + (n/W)²)
        
        Con correzione per massa persona distribuita.
        """
        L = genome.length
        W = genome.width
        h = genome.thickness_base
        
        # Rigidezza flessionale
        E = self.material.E_longitudinal
        nu = self.material.poisson_ratio
        D = E * h**3 / (12 * (1 - nu**2))
        
        # Densità superficiale (tavola + persona)
        rho_plate = self.material.density * h
        
        # Aggiungi massa persona distribuita (approssimazione)
        person_mass_per_area = self.person.weight_kg / (L * W)
        # Solo ~60% della massa è in contatto efficace
        rho_total = rho_plate + 0.6 * person_mass_per_area
        
        frequencies = []
        mode_shapes = []
        
        # Calcola primi n_modes modi (m, n = 1, 2, 3, ...)
        modes_mn = []
        for m in range(1, 8):
            for n in range(1, 6):
                f_mn = (np.pi / 2) * np.sqrt(D / rho_total) * (
                    (m / L)**2 + (n / W)**2
                )
                modes_mn.append((f_mn, m, n))
        
        # Ordina per frequenza
        modes_mn.sort(key=lambda x: x[0])
        
        # Prendi primi n_modes
        nx, ny = 20, 12  # Risoluzione griglia per mode shapes
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for i, (f, m, n) in enumerate(modes_mn[:self.n_modes]):
            frequencies.append(f)
            
            # Mode shape: sin(m*π*x/L) * sin(n*π*y/W)
            shape = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
            mode_shapes.append(shape)
        
        return frequencies, np.array(mode_shapes)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Risposta in Frequenza
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_frequency_response(
        self,
        genome: PlateGenome,
        frequencies: List[float],
        mode_shapes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola risposta in frequenza media sulla tavola.
        
        Returns:
            (freq_array, response_dB)
        """
        freq_array = self.test_frequencies
        response = np.zeros(len(freq_array))
        
        # Smorzamento modale
        zeta = self.material.damping_ratio
        
        for f_idx, f in enumerate(freq_array):
            omega = 2 * np.pi * f
            
            total_response = 0.0
            for mode_idx, f_n in enumerate(frequencies):
                omega_n = 2 * np.pi * f_n
                
                # Funzione trasferimento SDOF
                H = 1.0 / np.sqrt(
                    (1 - (omega/omega_n)**2)**2 + 
                    (2 * zeta * omega/omega_n)**2
                )
                
                # Peso per coupling modale (media sulla superficie)
                if mode_idx < len(mode_shapes):
                    modal_weight = np.mean(np.abs(mode_shapes[mode_idx]))
                else:
                    modal_weight = 0.5
                
                total_response += H * modal_weight
            
            response[f_idx] = total_response
        
        # Converti in dB (normalizzato)
        response_db = 20 * np.log10(response / np.max(response) + 1e-10)
        
        return freq_array, response_db
    
    def _compute_spine_response(
        self,
        genome: PlateGenome,
        frequencies: List[float],
        mode_shapes: np.ndarray
    ) -> np.ndarray:
        """
        Calcola risposta sulla linea della spina dorsale.
        
        Returns:
            Array (n_spine_points, n_freq) con risposta dB
        """
        n_spine = len(self.spine_positions)
        n_freq = len(self.test_frequencies)
        response = np.zeros((n_spine, n_freq))
        
        zeta = self.material.damping_ratio
        nx, ny = mode_shapes.shape[1], mode_shapes.shape[2]
        
        for pos_idx, (x_norm, y_norm) in enumerate(self.spine_positions):
            # Indici griglia più vicini
            ix = min(int(x_norm * nx), nx - 1)
            iy = min(int(y_norm * ny), ny - 1)
            
            for f_idx, f in enumerate(self.test_frequencies):
                omega = 2 * np.pi * f
                
                total = 0.0
                for mode_idx, f_n in enumerate(frequencies):
                    omega_n = 2 * np.pi * f_n
                    
                    H = 1.0 / np.sqrt(
                        (1 - (omega/omega_n)**2)**2 + 
                        (2 * zeta * omega/omega_n)**2
                    )
                    
                    # Mode shape a questa posizione
                    if mode_idx < len(mode_shapes):
                        phi = abs(mode_shapes[mode_idx, ix, iy])
                    else:
                        phi = 0.5
                    
                    total += H * phi
                
                response[pos_idx, f_idx] = total
        
        # Normalizza
        response = response / (np.max(response) + 1e-10)
        
        return response
    
    # ─────────────────────────────────────────────────────────────────────────
    # Score Functions
    # ─────────────────────────────────────────────────────────────────────────
    
    def _score_flatness(
        self,
        freq_response: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """
        Score per risposta piatta.
        
        1.0 = perfettamente piatta, 0.0 = variazione > 20 dB
        """
        _, response_db = freq_response
        
        # Deviazione standard della risposta
        std_db = np.std(response_db)
        
        # Variazione picco-picco
        peak_to_peak = np.max(response_db) - np.min(response_db)
        
        # Score basato su variazione (target: < 6 dB)
        target_variation = 6.0
        score = np.clip(1 - peak_to_peak / (2 * target_variation), 0, 1)
        
        return score
    
    def _score_spine_coupling(self, spine_response: np.ndarray) -> float:
        """
        Score per accoppiamento spina dorsale.
        
        1.0 = risposta uniforme e alta su tutta la spina
        """
        # Media risposta su tutti i punti spina
        mean_response = np.mean(spine_response)
        
        # Uniformità (1 - CV)
        cv = np.std(spine_response) / (np.mean(spine_response) + 1e-10)
        uniformity = np.clip(1 - cv, 0, 1)
        
        # Combina livello e uniformità
        level_score = np.clip(mean_response * 2, 0, 1)  # Scala arbitraria
        
        score = 0.6 * level_score + 0.4 * uniformity
        
        return score
    
    def _score_low_mass(self, genome: PlateGenome) -> float:
        """
        Score per massa bassa.
        
        1.0 = massa < 8 kg, 0.0 = massa > 20 kg
        """
        mass = genome.get_mass(self.material.density)
        
        # Range target: 8-20 kg
        mass_min, mass_max = 8.0, 20.0
        
        if mass <= mass_min:
            return 1.0
        elif mass >= mass_max:
            return 0.0
        else:
            return 1 - (mass - mass_min) / (mass_max - mass_min)
    
    def _score_manufacturability(self, genome: PlateGenome) -> float:
        """
        Score per producibilità.
        
        Penalizza:
        - Forme troppo complesse
        - Troppi tagli
        - Spessori estremi
        """
        score = 1.0
        
        # Penalità per cutouts
        n_cuts = len(genome.cutouts)
        if n_cuts > 0:
            score -= 0.1 * n_cuts
        
        # Penalità per forme non standard
        if genome.contour_type == ContourType.FREEFORM:
            score -= 0.2
        
        # Penalità per spessori estremi
        h = genome.thickness_base
        if h < 0.010 or h > 0.025:
            score -= 0.15
        
        # Penalità per aspect ratio estremo
        aspect = genome.length / genome.width
        if aspect < 2.0 or aspect > 4.0:
            score -= 0.1
        
        return max(0, min(1, score))


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FITNESS EVALUATOR TEST")
    print("=" * 60)
    
    # Crea persona e genoma
    person = Person(height_m=1.75, weight_kg=75.0)
    genome = PlateGenome(
        length=person.recommended_plate_length,
        width=person.recommended_plate_width,
        thickness_base=0.015,
        contour_type=ContourType.GOLDEN_RECT,
    )
    
    print(f"\nPerson: {person}")
    print(f"Genome: {genome}")
    
    # Crea evaluator
    evaluator = FitnessEvaluator(
        person=person,
        objectives=ObjectiveWeights(
            flatness=1.0,
            spine_coupling=2.0,
            low_mass=0.3,
            manufacturability=0.5,
        ),
        material="birch_plywood",
    )
    
    # Valuta
    print("\nEvaluating fitness...")
    result = evaluator.evaluate(genome)
    
    print(f"\nResult: {result}")
    print(f"\nMode frequencies: {result.frequencies[:7]} Hz")
    
    # Test con diversi genomi
    print("\n" + "-" * 40)
    print("Comparing different genomes:")
    print("-" * 40)
    
    for ct in [ContourType.RECTANGLE, ContourType.ELLIPSE, ContourType.OVOID]:
        g = PlateGenome(
            length=person.recommended_plate_length,
            width=person.recommended_plate_width,
            contour_type=ct,
        )
        r = evaluator.evaluate(g)
        print(f"  {ct.value:15s}: fitness={r.total_fitness:.3f} "
              f"(flat={r.flatness_score:.2f}, spine={r.spine_coupling_score:.2f})")
