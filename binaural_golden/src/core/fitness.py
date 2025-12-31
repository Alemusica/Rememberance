"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   FITNESS EVALUATOR - Multi-Objective Scoring                ║
║                                                                              ║
║   Valuta la fitness di un PlateGenome rispetto a obiettivi multipli:         ║
║   • Risposta frequenza piatta (20-200 Hz)                                    ║
║   • Accoppiamento spina dorsale                                              ║
║   • Peso minimo tavola                                                       ║
║   • Producibilità (no forme impossibili)                                     ║
║   • STRUCTURAL INTEGRITY (deflection < 10mm under person weight)             ║
║                                                                              ║
║   La fitness viene calcolata usando FEM semplificato (analitico) o           ║
║   completo (scikit-fem) se disponibile.                                       ║
║                                                                              ║
║   ZONE WEIGHTS (GUI slider):                                                 ║
║   • Spine (tactile): 70% default - vibration therapy focus                   ║
║   • Head (audio): 30% default - binaural music reproduction                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import logging

# Local imports
from .person import Person, SPINE_ZONES
from .plate_genome import PlateGenome, ContourType
from .plate_physics import Material, MATERIALS
from .structural_analysis import (
    StructuralAnalyzer, DeflectionResult, StressResult,
    detect_peninsulas, PeninsulaResult
)
from .analysis_config import get_target_spacing_mm, get_default_config

# Evolution logging (physics-driven)
from .evolution_logger import (
    log_zone_priority,
    log_modal_analysis,
    log_fitness_evaluation,
    log_comparison_with_target,
    log_physics_decision,
    setup_evolution_logging,
)

logger = logging.getLogger(__name__)
evo_logger = logging.getLogger("golden_studio.evolution")


@dataclass
class ZoneWeights:
    """
    Pesi per zone corporee nell'ottimizzazione della risposta in frequenza.
    
    La persona è la "corda da accordare" - ottimizziamo per:
    - spine: Risposta piatta sulla spina dorsale (vibrazione tattile)
    - head: Risposta piatta alle orecchie (ascolto binaurale)
    
    Default: 70% spina, 30% testa (priorità feeling vibrazionale)
    """
    spine: float = 0.70  # 70% priorità spina dorsale
    head: float = 0.30   # 30% priorità orecchie/testa
    
    def normalized(self) -> 'ZoneWeights':
        """Restituisce pesi normalizzati (somma = 1)."""
        total = self.spine + self.head
        if total <= 0:
            total = 1.0
        return ZoneWeights(
            spine=self.spine / total,
            head=self.head / total,
        )


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
class ObjectiveVector:
    """
    Multi-dimensional fitness for NSGA-II Pareto optimization.
    
    Each objective is scored independently for proper trade-off analysis.
    No weighted combination - let NSGA-II find the Pareto front!
    
    RESEARCH BASIS:
    - Deb et al. 2002: NSGA-II for multi-objective optimization
    - Bai & Liu 2004: GA for DML exciter placement
    
    All scores are 0-1 where 1.0 = optimal.
    For pymoo minimization: convert via `to_minimize_array()` (negates scores)
    """
    # === ZONE FLATNESS (Hz response uniformity) ===
    spine_flatness: float = 0.0     # Flatness at spine zone (20-200Hz)
    ear_flatness: float = 0.0       # Flatness at ear positions
    
    # === L/R BALANCE (Critical for binaural!) ===
    ear_lr_uniformity: float = 0.0  # L/R response balance [0-1]
    
    # === ENERGY DELIVERY ===
    spine_energy: float = 0.0       # Energy reaching spine zone
    ear_energy: float = 0.0         # Energy reaching ear zone
    
    # === STRUCTURAL ===
    mass_score: float = 0.0         # Lower mass = higher score
    structural_safety: float = 1.0  # Deflection safety [0-1]
    
    # === MANUFACTURABILITY ===
    cnc_simplicity: float = 0.5     # CNC path complexity [0-1]
    cutout_effectiveness: float = 0.0  # ABH/lutherie score
    
    # === LABELS for UI ===
    labels: Dict[str, str] = field(default_factory=lambda: {
        'spine_flatness': 'Spine Flatness (20-200Hz)',
        'ear_flatness': 'Ear Flatness',
        'ear_lr_uniformity': 'L/R Balance',
        'spine_energy': 'Spine Energy',
        'mass_score': 'Mass Efficiency',
        'structural_safety': 'Structural Safety',
    })
    
    def to_minimize_array(self) -> np.ndarray:
        """
        Convert to numpy array for pymoo minimization.
        
        Pymoo MINIMIZES objectives, so we negate scores (which we want to MAXIMIZE).
        Returns array ready for NSGA-II.
        """
        return np.array([
            -self.spine_flatness,      # Maximize → minimize negative
            -self.ear_flatness,
            -self.ear_lr_uniformity,
            -self.spine_energy,
            -self.mass_score,
            -self.structural_safety,
        ])
    
    def to_labeled_dict(self) -> Dict[str, float]:
        """Return dict with human-readable labels and percentages."""
        return {
            'Spine Flatness': self.spine_flatness * 100,
            'Ear Flatness': self.ear_flatness * 100,
            'L/R Balance': self.ear_lr_uniformity * 100,
            'Spine Energy': self.spine_energy * 100,
            'Mass Efficiency': self.mass_score * 100,
            'Structural Safety': self.structural_safety * 100,
        }
    
    def weighted_total(self, spine_weight: float = 0.7, head_weight: float = 0.3) -> float:
        """
        Compute weighted total for backwards compatibility.
        
        NOTE: For true multi-objective optimization, use NSGA-II with 
        to_minimize_array() instead of this scalar combination!
        
        Args:
            spine_weight: Weight for spine-focused objectives
            head_weight: Weight for head/ear-focused objectives
        
        Returns:
            Weighted scalar fitness [0-1]
        """
        spine_combined = (
            0.5 * self.spine_flatness + 
            0.3 * self.spine_energy +
            0.2 * self.structural_safety
        )
        
        head_combined = (
            0.4 * self.ear_flatness +
            0.4 * self.ear_lr_uniformity +
            0.2 * self.ear_energy
        )
        
        return (
            spine_weight * spine_combined +
            head_weight * head_combined +
            0.1 * self.mass_score  # Always consider mass
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
    
    # Structural integrity score (NEW!)
    structural_score: float = 1.0  # 1.0 = safe, 0.0 = dangerous deflection
    
    # Exciter and groove scores (lutherie)
    exciter_coupling_score: float = 0.0  # Exciter placement quality
    groove_tuning_score: float = 0.0     # Groove effectiveness
    cutout_tuning_score: float = 0.0     # Cutout effectiveness (ABH/lutherie)
    
    # Score totale pesato
    total_fitness: float = 0.0
    
    # Zone-specific scores (70/30 split)
    spine_flatness_score: float = 0.0  # Flatness at spine
    head_flatness_score: float = 0.0   # Flatness at ears
    
    # EAR L/R UNIFORMITY - Critical for binaural balance!
    # This is the KEY metric for multi-objective optimization
    ear_uniformity_score: float = 0.0  # L/R balance [0-1], 1=perfect symmetry
    
    # Structural diagnostics (NEW!)
    max_deflection_mm: float = 0.0     # Max deflection under person weight
    deflection_is_safe: bool = True     # Deflection < 10mm limit
    max_stress_MPa: float = 0.0        # Max stress (Von Mises)
    stress_safety_factor: float = 2.0  # Yield stress / max stress
    
    # Peninsula detection (isolated regions from intersecting cutouts)
    # PARADIGM SHIFT: Peninsulas can be beneficial (ABH energy focusing)
    has_peninsulas: bool = False       # True if isolated regions detected
    n_regions: int = 1                 # Number of connected regions (1 = OK)
    peninsula_penalty: float = 0.0     # Structural risk factor [0-1] (legacy)
    peninsula_benefit: float = 0.0     # ABH energy focusing benefit [0-1] (NEW!)
    peninsula_net_score: float = 0.0   # benefit - penalty, can be positive!
    
    # Dati diagnostici
    frequencies: List[float] = field(default_factory=list)
    mode_shapes: Optional[np.ndarray] = None
    frequency_response: Optional[Tuple[np.ndarray, np.ndarray]] = None
    spine_response: Optional[np.ndarray] = None
    head_response: Optional[np.ndarray] = None  # Response at ear positions
    
    def __repr__(self) -> str:
        return (
            f"Fitness({self.total_fitness:.3f}: "
            f"flat={self.flatness_score:.2f}, spine={self.spine_coupling_score:.2f}, "
            f"ear_LR={self.ear_uniformity_score:.2f}, "
            f"mass={self.low_mass_score:.2f}, manuf={self.manufacturability_score:.2f}, "
            f"struct={self.structural_score:.2f}, defl={self.max_deflection_mm:.1f}mm)"
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
        zone_weights: Optional[ZoneWeights] = None,
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
            zone_weights: Pesi zone (spine 70%, head 30% default)
            material: Nome materiale da MATERIALS
            freq_range: Range frequenze target [Hz]
            n_freq_points: Punti per calcolo risposta
            n_modes: Numero modi per FEM
        """
        self.person = person
        self.objectives = (objectives or ObjectiveWeights()).normalized()
        self.zone_weights = (zone_weights or ZoneWeights()).normalized()
        self.material = MATERIALS.get(material, MATERIALS["birch_plywood"])
        self.freq_range = freq_range
        self.n_freq_points = n_freq_points
        self.n_modes = n_modes
        
        # ═══════════════════════════════════════════════════════════════════
        # LOG ZONE PRIORITY - Physics: zone weights determine WHERE 
        # optimization focuses energy (spine 70% vs head 30%)
        # ═══════════════════════════════════════════════════════════════════
        log_zone_priority(
            spine_weight=self.zone_weights.spine,
            head_weight=self.zone_weights.head,
            logger=evo_logger
        )
        
        # Frequenze di test
        self.test_frequencies = np.linspace(
            freq_range[0], freq_range[1], n_freq_points
        )
        
        # Posizioni spina per test coupling
        self.spine_positions = self._compute_spine_positions()
        
        # Posizioni orecchie per test risposta audio
        self.head_positions = self._compute_head_positions()
        
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
    
    def _compute_head_positions(self) -> np.ndarray:
        """
        Calcola posizioni di test per la testa (orecchie).
        
        Le orecchie sono nella zona HEAD (0.88-1.0 normalizzato)
        posizionate lateralmente rispetto al centro.
        """
        positions = []
        
        # Posizione X della testa (normalizzata)
        head_x = 0.94  # Centro della zona HEAD
        
        # Orecchio sinistro e destro (offset Y dal centro)
        ear_offset = 0.15  # Distanza dal centro (normalizzata)
        
        # Orecchio sinistro
        positions.append([head_x, 0.5 - ear_offset])
        # Orecchio destro  
        positions.append([head_x, 0.5 + ear_offset])
        
        # Aggiungi punti intermedi per migliore sampling
        for i in range(3):
            x = 0.90 + i * 0.03
            positions.append([x, 0.5 - ear_offset * 0.8])
            positions.append([x, 0.5 + ear_offset * 0.8])
        
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
        
        # ═══════════════════════════════════════════════════════════════════
        # LOG MODAL ANALYSIS - Physics: reveals natural vibration patterns
        # Resolution must be > distance between cutouts (Schleske 2002)
        # ═══════════════════════════════════════════════════════════════════
        n_modes_computed = len(frequencies)
        # Estimate resolution from mode shapes (typically matches genome grid)
        resolution = mode_shapes.shape[1:] if mode_shapes is not None and len(mode_shapes.shape) > 2 else (40, 24)
        log_modal_analysis(
            plate_length=genome.length,
            plate_width=genome.width,
            thickness=genome.thickness_base,
            n_modes=n_modes_computed,
            frequencies=frequencies[:5] if frequencies else [],
            resolution=resolution,
            logger=evo_logger
        )
        
        # 2. Calcola risposta in frequenza globale
        freq_response = self._compute_frequency_response(genome, frequencies, mode_shapes)
        result.frequency_response = freq_response
        
        # 3. Calcola risposta sulla spina (70% del peso di ottimizzazione)
        spine_response = self._compute_spine_response(genome, frequencies, mode_shapes)
        result.spine_response = spine_response
        
        # 4. Calcola risposta alla testa/orecchie (30% del peso)
        head_response = self._compute_head_response(genome, frequencies, mode_shapes)
        result.head_response = head_response
        
        # 5. Score: flatness per zona (con zone_weights 70/30)
        result.spine_flatness_score = self._score_zone_flatness(spine_response)
        result.head_flatness_score = self._score_zone_flatness(head_response)
        
        # 5b. EAR L/R UNIFORMITY - Critical metric for binaural balance!
        result.ear_uniformity_score = self._score_ear_uniformity(head_response)
        
        # Flatness combinato con zone_weights (70% spine, 30% head)
        result.flatness_score = (
            self.zone_weights.spine * result.spine_flatness_score +
            self.zone_weights.head * result.head_flatness_score
        )
        
        # 6. Score: spine coupling (vibrazione tattile)
        result.spine_coupling_score = self._score_spine_coupling(spine_response)
        
        # 7. Score: low mass
        result.low_mass_score = self._score_low_mass(genome)
        
        # 8. Score: manufacturability
        result.manufacturability_score = self._score_manufacturability(genome)
        
        # 9. Score: exciter placement (optimal coupling to modes)
        result.exciter_coupling_score = self._score_exciter_placement(genome, frequencies, mode_shapes)
        
        # 10. Score: groove effectiveness (lutherie tuning)
        result.groove_tuning_score = self._score_groove_tuning(genome)
        
        # 11. Score: cutout effectiveness (acoustic black hole / lutherie)
        result.cutout_tuning_score = self._score_cutout_effectiveness(genome, frequencies, mode_shapes)
        
        # 12. Score: structural integrity (deflection under person weight)
        result.structural_score = self._score_structural_integrity(genome, result)

        # Score totale pesato
        # CRITICAL: Include ear_uniformity with HIGH weight for L/R balance!
        result.total_fitness = (
            self.objectives.flatness * result.flatness_score +
            self.objectives.spine_coupling * result.spine_coupling_score +
            self.objectives.low_mass * result.low_mass_score +
            self.objectives.manufacturability * result.manufacturability_score +
            0.1 * result.exciter_coupling_score +  # Bonus for good exciter placement
            0.05 * result.groove_tuning_score +    # Bonus for effective grooves
            0.08 * result.cutout_tuning_score +    # Bonus for effective cutouts (ABH/lutherie)
            0.4 * result.ear_uniformity_score      # HIGH weight for L/R balance!
        )
        
        # === CRITICAL PENALTY: Plate too short for person ===
        # A plate shorter than person + 15cm is UNUSABLE - apply severe penalty!
        min_length = self.person.recommended_plate_length
        if genome.length < min_length:
            length_deficit = (min_length - genome.length) / min_length
            # Apply multiplicative penalty: 50% reduction for 25% deficit
            length_penalty = min(0.6, length_deficit * 2.0)
            result.total_fitness *= (1.0 - length_penalty)
            logger.warning(
                f"Length penalty: plate {genome.length:.2f}m < required {min_length:.2f}m "
                f"(-{length_penalty*100:.0f}% fitness)"
            )
        
        # STRUCTURAL PENALTY: Reduce fitness if deflection exceeds limit
        # This ensures the person won't "fall through" the plate!
        if not result.deflection_is_safe:
            structural_penalty = 1.0 - result.structural_score
            result.total_fitness *= (1.0 - structural_penalty * 0.5)  # Up to 50% reduction
            logger.warning(
                f"Structural penalty: deflection={result.max_deflection_mm:.1f}mm > 10mm limit"
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # LOG FITNESS EVALUATION - Complete assessment
        # ═══════════════════════════════════════════════════════════════════
        n_cutouts = len(genome.cutouts) if genome.cutouts else 0
        log_fitness_evaluation(
            genome_id=id(genome) % 10000,  # Short ID
            flatness_score=result.flatness_score,
            spine_flatness=result.spine_flatness_score,
            head_flatness=result.head_flatness_score,
            ear_uniformity=result.ear_uniformity_score,
            spine_coupling=result.spine_coupling_score,
            structural_score=result.structural_score,
            total_fitness=result.total_fitness,
            zone_weights=(self.zone_weights.spine, self.zone_weights.head),
            n_cutouts=n_cutouts,
            logger=evo_logger
        )
        
        return result
    
    def evaluate_multi(self, genome: PlateGenome) -> Tuple[FitnessResult, ObjectiveVector]:
        """
        Multi-objective evaluation for NSGA-II Pareto optimization.
        
        Returns both legacy FitnessResult (for backwards compatibility)
        and ObjectiveVector (for proper multi-objective optimization).
        
        This enables:
        1. Pareto front exploration without weighted combination
        2. Trade-off analysis between conflicting objectives
        3. UI display of labeled individual scores
        
        Reference: Deb et al. 2002 "NSGA-II"
        """
        # Get standard evaluation
        result = self.evaluate(genome)
        
        # Build ObjectiveVector from individual scores
        obj_vec = ObjectiveVector(
            spine_flatness=result.spine_flatness_score,
            ear_flatness=result.head_flatness_score,
            ear_lr_uniformity=result.ear_uniformity_score,
            spine_energy=result.spine_coupling_score,
            ear_energy=min(result.head_flatness_score, result.ear_uniformity_score),  # Conservative
            mass_score=result.low_mass_score,
            structural_safety=result.structural_score,
            cnc_simplicity=result.manufacturability_score,
            cutout_effectiveness=result.cutout_tuning_score,
        )
        
        return result, obj_vec
    
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # LUTHERIE: Cutout correction for mode frequencies
        # Reference: Schleske (2002) - violin f-holes shift modes by ~5-15%
        # ═══════════════════════════════════════════════════════════════════════
        cutout_area_fraction = 0.0
        if genome.cutouts:
            # Calculate total cutout area (normalized)
            for cutout in genome.cutouts:
                # Ellipse area: π * a * b
                cutout_area = np.pi * cutout.width * cutout.height / 4
                cutout_area_fraction += cutout_area
            # Limit to reasonable range
            cutout_area_fraction = min(cutout_area_fraction, 0.15)  # Max 15%
        
        # Cutouts reduce stiffness → lower frequencies
        # Empirical: Δf/f ≈ -0.5 * (cutout_area / plate_area)
        # But they also reduce mass → slight increase
        # Net effect: frequency shift depends on cutout position
        cutout_frequency_shift = 1.0 - 0.3 * cutout_area_fraction
        
        # ═══════════════════════════════════════════════════════════════════════
        # ADAPTIVE RESOLUTION - Grid spacing must be < typical cutout distance
        # Reference: Schleske (2002) - resolution finer than feature size
        # Uses centralized config from analysis_config.py
        # ═══════════════════════════════════════════════════════════════════════
        target_spacing_mm = get_target_spacing_mm(L, W)  # Adaptive based on plate size
        L_mm = L * 1000  # Convert to mm
        W_mm = W * 1000
        
        # Calculate required resolution (ensure ODD for symmetric grids)
        nx_min = int(np.ceil(L_mm / target_spacing_mm))
        ny_min = int(np.ceil(W_mm / target_spacing_mm))
        
        # Ensure odd numbers for exact center point at 0.5
        nx = nx_min + 1 if nx_min % 2 == 0 else nx_min
        ny = ny_min + 1 if ny_min % 2 == 0 else ny_min
        
        # Clamp to reasonable range (performance vs accuracy tradeoff)
        nx = max(21, min(nx, 101))  # 21-101 points
        ny = max(13, min(ny, 51))   # 13-51 points
        
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for i, (f, m, n) in enumerate(modes_mn[:self.n_modes]):
            # Apply lutherie cutout correction
            f_corrected = f * cutout_frequency_shift
            
            # Additional mode-specific correction based on cutout positions
            # Cutouts near antinodes affect those modes more
            if genome.cutouts:
                mode_correction = self._compute_mode_cutout_correction(
                    cutout=genome.cutouts, m=m, n=n, L=L, W=W
                )
                f_corrected *= mode_correction
            
            frequencies.append(f_corrected)
            
            # Mode shape: sin(m*π*x/L) * sin(n*π*y/W)
            shape = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
            mode_shapes.append(shape)
        
        return frequencies, np.array(mode_shapes)
    
    def _compute_mode_cutout_correction(
        self,
        cutout: list,
        m: int, n: int,
        L: float, W: float
    ) -> float:
        """
        Compute mode-specific frequency correction for cutouts.
        
        LUTHERIE PRINCIPLE (Schleske 2002, Fletcher & Rossing 1998):
        Cutouts near modal antinodes have maximum effect on that mode.
        Cutouts near nodal lines have minimal effect.
        
        f-holes in violins are positioned to tune specific modes:
        - Main air resonance (Helmholtz ~280 Hz)
        - First corpus mode (~450 Hz)
        
        Args:
            cutout: List of CutoutGene
            m, n: Mode numbers
            L, W: Plate dimensions
        
        Returns:
            Frequency correction factor (0.9 - 1.0 typically)
        """
        if not cutout:
            return 1.0
        
        total_correction = 0.0
        total_weight = 0.0
        
        for c in cutout:
            # Cutout position (normalized 0-1)
            cx, cy = c.x, c.y
            
            # Mode shape value at cutout position
            # Higher |φ| = antinode = more effect
            mode_value = np.sin(m * np.pi * cx) * np.sin(n * np.pi * cy)
            mode_amplitude = np.abs(mode_value)
            
            # Cutout area (normalized)
            cutout_area = np.pi * c.width * c.height / 4
            
            # Effect is proportional to:
            # - Cutout area
            # - How close to antinode (mode_amplitude)
            effect = cutout_area * mode_amplitude
            
            # Frequency shift: negative for stiffness reduction
            # Cutout at antinode reduces stiffness → lower frequency
            # Empirical factor from lutherie
            freq_shift = -0.5 * effect
            
            total_correction += freq_shift
            total_weight += cutout_area
        
        # Return correction factor (typically 0.85 - 1.0)
        return max(0.85, 1.0 + total_correction)
    
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
    
    def _compute_head_response(
        self,
        genome: PlateGenome,
        frequencies: List[float],
        mode_shapes: np.ndarray
    ) -> np.ndarray:
        """
        Calcola risposta alle posizioni della testa (orecchie).
        
        Per l'ascolto binaurale, vogliamo risposta piatta alle orecchie.
        
        CRITICAL FOR L/R BALANCE:
        For symmetric plates, L and R ears should have identical responses
        because they are symmetric about y=0.5 centerline.
        
        Returns:
            Array (n_head_points, n_freq) con risposta normalizzata
        """
        n_head = len(self.head_positions)
        n_freq = len(self.test_frequencies)
        response = np.zeros((n_head, n_freq))
        
        zeta = self.material.damping_ratio
        
        # Grid dimensions from mode shapes
        if len(mode_shapes) > 0:
            nx, ny = mode_shapes.shape[1], mode_shapes.shape[2]
        else:
            nx, ny = 20, 12
        
        for pos_idx, (x_norm, y_norm) in enumerate(self.head_positions):
            # Use bilinear interpolation instead of nearest neighbor
            # This gives smoother, more symmetric responses
            x_grid = x_norm * (nx - 1)
            y_grid = y_norm * (ny - 1)
            
            # Integer and fractional parts
            ix0 = int(np.floor(x_grid))
            iy0 = int(np.floor(y_grid))
            ix1 = min(ix0 + 1, nx - 1)
            iy1 = min(iy0 + 1, ny - 1)
            
            fx = x_grid - ix0
            fy = y_grid - iy0
            
            # Bilinear weights
            w00 = (1 - fx) * (1 - fy)
            w01 = (1 - fx) * fy
            w10 = fx * (1 - fy)
            w11 = fx * fy
            
            for f_idx, f in enumerate(self.test_frequencies):
                omega = 2 * np.pi * f
                
                total = 0.0
                for mode_idx, f_n in enumerate(frequencies):
                    omega_n = 2 * np.pi * f_n
                    
                    # Frequency response function
                    H = 1.0 / np.sqrt(
                        (1 - (omega/omega_n)**2)**2 + 
                        (2 * zeta * omega/omega_n)**2
                    )
                    
                    # Mode shape value - use bilinear interpolation
                    if mode_idx < len(mode_shapes):
                        ms = mode_shapes[mode_idx]
                        # Bilinear interpolation for smooth symmetric response
                        phi = (w00 * ms[ix0, iy0] + 
                               w01 * ms[ix0, iy1] + 
                               w10 * ms[ix1, iy0] + 
                               w11 * ms[ix1, iy1])
                        # Use absolute value for modal amplitude
                        phi = abs(phi)
                    else:
                        phi = 0.5
                    
                    total += H * phi
                
                response[pos_idx, f_idx] = total
        
        # Normalizza
        max_val = np.max(response)
        if max_val > 1e-10:
            response = response / max_val
        
        return response
    
    # ─────────────────────────────────────────────────────────────────────────
    # Score Functions
    # ─────────────────────────────────────────────────────────────────────────
    
    def _score_zone_flatness(self, zone_response: np.ndarray) -> float:
        """
        Score per risposta piatta in una zona specifica (spina o testa).
        
        Args:
            zone_response: Array (n_points, n_freq) con risposta normalizzata
            
        Returns:
            Score [0, 1] dove 1.0 = perfettamente piatta
        """
        # Converti in dB
        response_db = 20 * np.log10(zone_response + 1e-10)
        
        # Media la risposta su tutti i punti della zona
        mean_response_db = np.mean(response_db, axis=0)
        
        # Variazione picco-picco nella banda di frequenza
        peak_to_peak = np.max(mean_response_db) - np.min(mean_response_db)
        
        # Score basato su variazione (target: < 6 dB = ottimo)
        target_variation = 6.0
        score = np.clip(1 - peak_to_peak / (2 * target_variation), 0, 1)
        
        # Bonus per uniformità spaziale (tutti i punti rispondono uguale)
        spatial_std = np.std(response_db, axis=0).mean()
        uniformity_bonus = np.clip(1 - spatial_std / 10, 0, 0.2)
        
        return min(score + uniformity_bonus, 1.0)
    
    def _score_ear_uniformity(self, head_response: np.ndarray) -> float:
        """
        Score for Left/Right ear uniformity (binaural balance).
        
        This is CRITICAL for proper binaural audio reproduction.
        L/R imbalance causes localization errors and reduces therapy effectiveness.
        
        Args:
            head_response: Array (n_positions, n_freq) where first 2 positions
                          are left and right ears
        
        Returns:
            Score [0, 1] where 1.0 = perfect L/R symmetry
        """
        if head_response is None or len(head_response) < 2:
            return 0.0
        
        # Get left and right ear responses (first two positions)
        left_ear = head_response[0]
        right_ear = head_response[1]
        
        # Ensure both are 1D arrays
        if len(left_ear.shape) > 1:
            left_ear = np.mean(left_ear, axis=0)
        if len(right_ear.shape) > 1:
            right_ear = np.mean(right_ear, axis=0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # METRIC 1: RMS Level Balance (60% weight)
        # Perfect: L and R have same overall energy
        # ═══════════════════════════════════════════════════════════════════════
        left_rms = np.sqrt(np.mean(left_ear**2))
        right_rms = np.sqrt(np.mean(right_ear**2))
        
        if left_rms + right_rms < 1e-10:
            return 0.0
        
        min_rms = min(left_rms, right_rms)
        max_rms = max(left_rms, right_rms)
        
        if max_rms < 1e-10:
            return 0.0
        
        level_balance = min_rms / max_rms  # 1.0 = perfect balance
        
        # ═══════════════════════════════════════════════════════════════════════
        # METRIC 2: Frequency-by-Frequency Correlation (40% weight)
        # Perfect: L and R have same spectral shape
        # ═══════════════════════════════════════════════════════════════════════
        if len(left_ear) == len(right_ear) and len(left_ear) > 2:
            # Pearson correlation
            correlation = np.corrcoef(left_ear, right_ear)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            correlation = max(0.0, correlation)  # Clamp negative correlations
            
            # Also check frequency-by-frequency difference
            diff_db = 20 * np.log10((np.abs(left_ear - right_ear) + 1e-10) / 
                                    (np.maximum(left_ear, right_ear) + 1e-10))
            mean_diff_db = np.mean(np.abs(diff_db))
            
            # Target: < 3 dB difference across all frequencies
            spectral_match = np.clip(1 - mean_diff_db / 6.0, 0, 1)
        else:
            correlation = 0.5
            spectral_match = 0.5
        
        # ═══════════════════════════════════════════════════════════════════════
        # Combined Score: 50% level + 25% correlation + 25% spectral
        # ═══════════════════════════════════════════════════════════════════════
        uniformity = 0.50 * level_balance + 0.25 * correlation + 0.25 * spectral_match
        
        return float(np.clip(uniformity, 0.0, 1.0))
    
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
        Score per massa bassa e superficie controllata.
        
        1.0 = massa < 8 kg, 0.0 = massa > 20 kg
        
        VINCOLO SUPERFICIE: max 2.0 m² (standard 210x80 + 20%)
        Penalizza fortemente tavole troppo grandi.
        """
        mass = genome.get_mass(self.material.density)
        
        # Range target: 8-20 kg
        mass_min, mass_max = 8.0, 20.0
        
        if mass <= mass_min:
            mass_score = 1.0
        elif mass >= mass_max:
            mass_score = 0.0
        else:
            mass_score = 1 - (mass - mass_min) / (mass_max - mass_min)
        
        # SURFACE AREA CONSTRAINT (standard: 210x80 = 1.68 m²)
        surface_area = genome.length * genome.width
        max_surface = 2.0  # m² (20% over standard)
        
        if surface_area > max_surface:
            # Heavy penalty for oversized plates
            oversize_ratio = surface_area / max_surface
            surface_penalty = min(1.0, (oversize_ratio - 1.0) * 2.0)  # 50% over = full penalty
            mass_score *= (1.0 - surface_penalty * 0.5)
        
        return mass_score
    
    def _score_manufacturability(self, genome: PlateGenome) -> float:
        """
        Score per producibilità.
        
        Penalizza:
        - Forme troppo complesse
        - Troppi tagli
        - Spessori estremi
        - Tavola troppo corta per la persona (CRITICAL!)
        """
        score = 1.0
        
        # === CRITICAL: Penalità per tavola troppo corta ===
        # La tavola DEVE essere >= person.height + 0.15m
        min_length = self.person.recommended_plate_length
        if genome.length < min_length:
            # Penalità proporzionale al deficit
            deficit = min_length - genome.length
            deficit_ratio = deficit / min_length
            # Penalità pesante: tavola inutilizzabile se troppo corta!
            score -= min(0.8, deficit_ratio * 2.0)
        
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
    
    def _score_exciter_placement(
        self,
        genome: PlateGenome,
        frequencies: List[float],
        mode_shapes: np.ndarray
    ) -> float:
        """
        Score for exciter placement quality.
        
        OPTIMAL PLACEMENT:
        - Exciters at mode antinodes maximize energy transfer
        - Head exciters (CH1/CH2) should couple well to head-zone modes
        - Feet exciters (CH3/CH4) should couple well to feet-zone modes
        - Avoid nodal lines (zero coupling)
        
        Hardware: 4× Dayton DAEX25 (25mm, 40W, 8Ω) via JAB4 WONDOM
        
        Returns:
            Score 0-1 (1.0 = optimal placement for all modes)
        """
        if not hasattr(genome, 'exciters') or not genome.exciters:
            return 0.5  # Default score if no exciters defined
        
        total_coupling = 0.0
        n_modes = min(len(frequencies), len(mode_shapes))
        
        for mode_idx in range(n_modes):
            mode_shape = mode_shapes[mode_idx]
            nx, ny = mode_shape.shape
            
            mode_coupling = 0.0
            for exciter in genome.exciters:
                # Convert exciter position to grid indices
                ix = int(np.clip(exciter.x * nx, 0, nx - 1))
                iy = int(np.clip(exciter.y * ny, 0, ny - 1))
                
                # Mode amplitude at exciter position
                amplitude = np.abs(mode_shape[ix, iy])
                
                # Weight by mode importance (lower modes more important)
                mode_weight = 1.0 / (mode_idx + 1)
                
                mode_coupling += amplitude * mode_weight
            
            total_coupling += mode_coupling / len(genome.exciters)
        
        # Normalize to 0-1
        max_possible = sum(1.0 / (i + 1) for i in range(n_modes))
        score = total_coupling / max_possible if max_possible > 0 else 0.5
        
        return np.clip(score, 0, 1)
    
    def _score_groove_tuning(self, genome: PlateGenome) -> float:
        """
        Score for groove tuning effectiveness.
        
        LUTHERIE PRINCIPLE:
        - Grooves near mode antinodes shift those frequencies more
        - Well-placed grooves can tune specific modes toward targets
        - Too many/deep grooves weaken the plate
        
        Returns:
            Score 0-1 (1.0 = effective groove placement)
        """
        if not hasattr(genome, 'grooves') or not genome.grooves:
            return 0.5  # Neutral score if no grooves
        
        score = 0.7  # Base score for having grooves (lutherie approach)
        
        # Bonus for grooves in spine zone (y=0.3-0.7)
        spine_grooves = sum(1 for g in genome.grooves if 0.3 <= g.y <= 0.7)
        score += 0.05 * min(spine_grooves, 4)  # Up to +0.2 for spine grooves
        
        # Penalty for too many grooves (weakens plate)
        if len(genome.grooves) > 6:
            score -= 0.1 * (len(genome.grooves) - 6)
        
        # Penalty for grooves too deep (structural risk)
        deep_grooves = sum(1 for g in genome.grooves if g.depth > 0.4)
        score -= 0.05 * deep_grooves
        
        # Bonus for varied angles (affects different modes)
        angles = [g.angle for g in genome.grooves]
        if len(angles) > 1:
            angle_variance = np.var(angles)
            score += 0.1 * min(angle_variance / 0.5, 1)  # Encourage variety
        
        return np.clip(score, 0, 1)
    
    def _score_cutout_effectiveness(
        self, 
        genome: PlateGenome,
        frequencies: np.ndarray,
        mode_shapes: np.ndarray
    ) -> float:
        """
        Score for cutout effectiveness based on lutherie and ABH principles.
        
        RESEARCH BASIS:
        - Schleske 2002: Cutouts/holes at antinodes shift frequencies maximally
        - Krylov 2014, Deng 2019: ABH (Acoustic Black Holes) focus energy
        - Bai 2004: GA-based optimization for DML
        
        SCORING:
        1. Cutout at mode antinode → shifts that frequency (useful for tuning)
        2. Cutout near edges → ABH-like energy focusing (bonus)
        3. Cutout in ear zone → improves L/R uniformity potential
        4. Too large cutouts → structural penalty (handled elsewhere)
        
        Returns:
            Score 0-1 (1.0 = very effective cutout placement)
        """
        if not hasattr(genome, 'cutouts') or not genome.cutouts:
            return 0.5  # Neutral score if no cutouts
        
        score = 0.6  # Base score for having cutouts (encourages exploration)
        n_cutouts = len(genome.cutouts)
        
        # --- 1. Antinode placement bonus ---
        # Cutouts near antinodes have more acoustic effect
        antinode_bonus = 0.0
        for cut in genome.cutouts:
            # Check each mode's shape at cutout location
            for i, mode_shape in enumerate(mode_shapes[:6]):  # First 6 modes
                if mode_shape is None:
                    continue
                # Sample mode amplitude at cutout (normalized coords)
                nx, ny = int(cut.x * 10), int(cut.y * 10)  # Coarse grid
                try:
                    if hasattr(mode_shape, 'shape') and len(mode_shape.shape) >= 2:
                        ny_max, nx_max = mode_shape.shape[:2]
                        nx = min(nx, nx_max - 1)
                        ny = min(ny, ny_max - 1)
                        amplitude = abs(mode_shape[ny, nx])
                        if amplitude > 0.5:  # Near antinode
                            antinode_bonus += 0.02  # Small bonus per mode
                except (IndexError, TypeError):
                    pass
        
        score += min(antinode_bonus, 0.15)  # Cap at +0.15
        
        # --- 2. ABH edge placement bonus ---
        # Cutouts near edges can focus energy (Krylov, Deng)
        abh_bonus = 0.0
        for cut in genome.cutouts:
            dist_to_edge = min(cut.x, 1 - cut.x, cut.y, 1 - cut.y)
            if dist_to_edge < 0.15:  # Near edge (within 15% of dimension)
                abh_bonus += 0.04  # ABH-like behavior
        
        score += min(abh_bonus, 0.12)  # Cap at +0.12
        
        # --- 3. Ear zone cutouts bonus ---
        # Cutouts near head end (y > 0.8) can improve ear response
        ear_cutouts = sum(1 for c in genome.cutouts if c.y > 0.8)
        score += 0.04 * min(ear_cutouts, 2)  # Up to +0.08
        
        # --- 4. Spine zone avoidance ---
        # Don't cut holes in main load-bearing spine zone (y=0.35-0.65)
        spine_cutouts = sum(1 for c in genome.cutouts if 0.35 <= c.y <= 0.65 and 0.3 <= c.x <= 0.7)
        score -= 0.06 * spine_cutouts  # Penalty for spine holes
        
        # --- 5. Size appropriateness ---
        # Small cutouts (< 5% of plate area) are better for tuning
        total_cutout_area = sum(c.width * c.height for c in genome.cutouts)
        if total_cutout_area < 0.03:  # < 3% total
            score += 0.05  # Good restraint
        elif total_cutout_area > 0.10:  # > 10% total
            score -= 0.1  # Too much material removed
        
        # --- 6. Distribution bonus ---
        # Asymmetric placement (different x positions) can break unwanted modes
        if n_cutouts >= 2:
            x_positions = [c.x for c in genome.cutouts]
            x_variance = np.var(x_positions)
            if x_variance > 0.05:  # Well distributed
                score += 0.05
        
        return np.clip(score, 0, 1)
    
    def _score_structural_integrity(
        self, 
        genome: PlateGenome, 
        result: FitnessResult
    ) -> float:
        """
        Score for structural integrity under person weight.
        
        CRITICAL SAFETY CHECK:
        The plate MUST support the person without excessive deflection!
        
        Uses StructuralAnalyzer to calculate:
        - Max deflection under distributed body load (pelvis 35%, thorax 25%, etc.)
        - Stress concentration at cutout edges
        - Safety factor vs yield stress
        
        LIMITS:
        - Max deflection: 10mm (comfort + stability)
        - Min safety factor: 2.0 (engineering standard)
        
        CUTOUT IMPACT:
        - Cutouts in load-bearing zones (pelvis, thorax) reduce stiffness
        - Deep grooves also reduce local bending stiffness
        
        Returns:
            Score 0-1 (1.0 = safe, 0.0 = dangerous)
        """
        try:
            # Create structural analyzer with plate dimensions
            analyzer = StructuralAnalyzer(
                length=genome.length,
                width=genome.width,
                thickness=genome.thickness_base,
                material=self.material.name if hasattr(self.material, 'name') else "birch_plywood",
                E_modulus=self.material.E_longitudinal,
                poisson=0.33,  # Typical for plywood
            )
            
            # Set grooves for local stiffness reduction
            if hasattr(genome, 'grooves') and genome.grooves:
                analyzer.set_grooves(genome.grooves)
            
            # Prepare cutout list for FEM (x, y, equivalent_radius)
            cutouts_for_fem = []
            if hasattr(genome, 'cutouts') and genome.cutouts:
                for cut in genome.cutouts:
                    # Convert normalized coords to absolute
                    cx = cut.x * genome.length
                    cy = cut.y * genome.width
                    # Equivalent radius from width/height
                    r_equiv = (cut.width * genome.length + cut.height * genome.width) / 4
                    cutouts_for_fem.append((cx, cy, r_equiv))
            
            # Calculate deflection under person weight
            person_weight = self.person.weight_kg if hasattr(self.person, 'weight_kg') else 80.0
            
            defl_result = analyzer.calculate_deflection(
                person_weight_kg=person_weight,
                cutouts=cutouts_for_fem if cutouts_for_fem else None,
                resolution=30,  # Fast for evolution
                use_fem=False,  # Analytical for speed
            )
            
            # Store results in FitnessResult
            result.max_deflection_mm = defl_result.max_deflection_mm
            result.deflection_is_safe = defl_result.is_acceptable
            
            # Calculate stress if cutouts present (concentration risk)
            if cutouts_for_fem:
                try:
                    stress_result = analyzer.calculate_stress(
                        person_weight_kg=person_weight,
                        cutouts=cutouts_for_fem,
                        resolution=30,
                    )
                    result.max_stress_MPa = stress_result.max_stress_MPa
                    result.stress_safety_factor = stress_result.safety_factor
                except Exception as e:
                    logger.debug(f"Stress calculation failed: {e}")
                    result.stress_safety_factor = 2.0  # Assume safe
            
            # Calculate score based on deflection and safety factor
            MAX_DEFLECTION_MM = 10.0
            MIN_SAFETY_FACTOR = 2.0
            
            # Deflection score: linear ramp from 1.0 (0mm) to 0.0 (>15mm)
            if defl_result.max_deflection_mm <= MAX_DEFLECTION_MM:
                defl_score = 1.0 - (defl_result.max_deflection_mm / MAX_DEFLECTION_MM) * 0.3
            else:
                # Penalize heavily beyond limit
                excess = defl_result.max_deflection_mm - MAX_DEFLECTION_MM
                defl_score = 0.7 - min(excess / 5.0, 0.7)  # Can reach 0
            
            # Safety factor score
            sf = result.stress_safety_factor
            if sf >= MIN_SAFETY_FACTOR:
                sf_score = 1.0
            else:
                sf_score = sf / MIN_SAFETY_FACTOR
            
            # ══════════════════════════════════════════════════════════════════
            # PENINSULA DETECTION - Evaluate as potential ABH resonator
            # Ref: Krylov 2014, Deng 2019, Zhao 2014 (ABH research)
            # ══════════════════════════════════════════════════════════════════
            peninsula_score = 1.0
            if hasattr(genome, 'cutouts') and genome.cutouts and len(genome.cutouts) > 1:
                # Build cutout list for peninsula detection
                cutout_dicts = []
                for cut in genome.cutouts:
                    cutout_dicts.append({
                        'x': cut.x,
                        'y': cut.y,
                        'size': (cut.width + cut.height) / 2,  # Average size
                        'shape': getattr(cut, 'shape', 'ellipse'),
                        'rotation': getattr(cut, 'rotation', 0),
                        'aspect': cut.height / max(cut.width, 0.001) if cut.width > 0 else 1.0
                    })
                
                try:
                    peninsula_result = detect_peninsulas(
                        length=genome.length,
                        width=genome.width,
                        cutouts=cutout_dicts,
                        resolution=60  # Fast enough for evolution
                    )
                    
                    # Store result for debugging
                    result.has_peninsulas = peninsula_result.has_peninsulas
                    result.n_regions = peninsula_result.n_regions
                    result.peninsula_penalty = peninsula_result.structural_penalty
                    
                    # NEW: Calculate net benefit (ABH benefit - structural risk)
                    # Ref: ABH research shows isolated regions can HELP
                    result.peninsula_benefit = peninsula_result.abh_benefit
                    result.peninsula_net_score = (
                        peninsula_result.abh_benefit * 0.6 +
                        peninsula_result.resonator_potential * 0.3 -
                        peninsula_result.structural_penalty * 0.5
                    )
                    
                    # Apply peninsula effect (now considers benefit!)
                    if peninsula_result.has_peninsulas:
                        # Net effect: benefit can offset penalty
                        # If net_score > 0, peninsula HELPS the design
                        net_effect = result.peninsula_net_score
                        
                        if net_effect >= 0:
                            # Peninsula is beneficial! Small bonus
                            peninsula_score = 1.0 + net_effect * 0.1  # Up to 10% bonus
                            logger.info(
                                f"Peninsula BENEFIT! {peninsula_result.n_regions} regions, "
                                f"ABH={peninsula_result.abh_benefit:.2f}, "
                                f"resonator={peninsula_result.resonator_potential:.2f}, "
                                f"net={net_effect:+.2f}"
                            )
                        else:
                            # Still a penalty, but reduced
                            peninsula_score = 1.0 + net_effect * 0.3  # Reduced penalty
                            logger.warning(
                                f"Peninsula detected: {peninsula_result.n_regions} regions, "
                                f"net={net_effect:.2f} (penalty reduced by ABH potential)"
                            )
                        
                except Exception as e:
                    logger.debug(f"Peninsula detection failed: {e}")
                    peninsula_score = 1.0  # Assume OK if detection fails
            else:
                result.has_peninsulas = False
                result.n_regions = 1
                result.peninsula_penalty = 0.0
                result.peninsula_benefit = 0.0
                result.peninsula_net_score = 0.0
            
            # Combined score (deflection 50%, safety 20%, peninsula 30%)
            # NOTE: peninsula_score can now be > 1.0 if ABH benefit outweighs risk!
            score = 0.5 * defl_score + 0.2 * sf_score + 0.3 * min(peninsula_score, 1.2)
            
            logger.debug(
                f"Structural: defl={defl_result.max_deflection_mm:.1f}mm, "
                f"sf={result.stress_safety_factor:.1f}, "
                f"peninsula={peninsula_score:.2f} (net={result.peninsula_net_score:+.2f}), "
                f"score={score:.2f}"
            )
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            logger.warning(f"Structural analysis failed: {e}")
            # Return neutral score if analysis fails
            result.deflection_is_safe = True
            result.max_deflection_mm = 5.0  # Assume moderate
            return 0.7  # Assume reasonably safe


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
