"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               EVOLUTIONARY OPTIMIZER - Genetic Algorithm for Plates          ║
║                                                                              ║
║   Ottimizzazione evolutiva della forma tavola vibroacustica.                 ║
║   Usa algoritmo genetico con:                                                 ║
║   • Selezione torneo                                                          ║
║   • Crossover multi-punto                                                     ║
║   • Mutazione adattiva                                                        ║
║   • Elitismo                                                                  ║
║   • Callback per visualizzazione real-time                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import copy
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

# Local imports
from .person import Person
from .plate_genome import PlateGenome, ContourType, SpringSupportGene
from .fitness import FitnessEvaluator, FitnessResult, ObjectiveWeights, ZoneWeights

# Evolution logging (physics-driven)
from .evolution_logger import (
    log_generation_summary,
    log_cutout_placement,
    log_comparison_with_target,
    log_physics_decision,
    setup_evolution_logging,
)

evo_logger = logging.getLogger("golden_studio.evolution")


class SelectionMethod(Enum):
    """Metodo di selezione genitori."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


@dataclass
class EvolutionConfig:
    """
    Configurazione algoritmo evolutivo.
    
    Presets disponibili:
    - QUICK: Test rapido (pop=20, gen=30) 
    - STANDARD: Bilanciato (pop=50, gen=100)
    - INTENSE: Massimizzato per M1 Max (pop=100, gen=300)
    - EXHAUSTIVE: Ricerca esaustiva (pop=200, gen=500)
    """
    # Popolazione
    population_size: int = 50       # Increased from 30
    n_generations: int = 100        # Increased from 50
    elite_count: int = 3            # Keep top 3
    
    # Selezione
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 4        # Increased from 3
    
    # Operatori genetici
    crossover_rate: float = 0.85    # Slightly higher
    mutation_rate: float = 0.25     # Slightly lower for stability
    mutation_sigma: float = 0.05    # Deviazione standard mutazione
    
    # Adattamento mutazione
    adaptive_mutation: bool = True
    mutation_decay: float = 0.97    # Slower decay (was 0.95)
    min_mutation_sigma: float = 0.008  # Lower floor (was 0.01)
    
    # Diversità
    diversity_threshold: float = 0.08
    diversity_injection_rate: float = 0.12
    
    # Vincoli dimensionali (STANDARD: 210cm x 80cm = 1.68 m²)
    # La superficie non può superare il +20% dello standard
    max_surface_area: float = 2.0  # m² (210x80 + 20% margin)
    standard_length: float = 2.10  # m (reference board)
    standard_width: float = 0.80   # m (reference board)
    
    # Vincoli contorno
    allowed_contours: List[ContourType] = field(default_factory=lambda: [
        ContourType.PHI_ROUNDED,  # New default with golden corners
        ContourType.RECTANGLE,
        ContourType.GOLDEN_RECT,
        ContourType.ELLIPSE,
        ContourType.OVOID,
        ContourType.VESICA_PISCIS,
        ContourType.SUPERELLIPSE,
        ContourType.ORGANIC,
        ContourType.ERGONOMIC,
        ContourType.FREEFORM,
    ])
    fixed_contour: Optional[ContourType] = None  # If set, only use this contour type
    max_cutouts: int = 6   # Increased from 4 for more modal tuning options
    max_grooves: int = 4   # Enable grooves by default for fine tuning
    
    # Symmetry enforcement (LUTHERIE: like violin/guitar plates)
    enforce_symmetry: bool = True  # Default ON for balanced vibroacoustic response
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPRING SUPPORTS (physics-based vibration isolation)
    # ═══════════════════════════════════════════════════════════════════════════
    # Reference: Den Hartog "Mechanical Vibrations", Harris & Piersol "Shock and Vibration"
    # Natural frequency: f_n = √(k/m) / (2π)
    # Isolation starts above f_n × √2 (transmissibility T < 1)
    spring_count: int = 5  # Number of spring supports (3-8)
    spring_stiffness_kn_m: float = 10.0  # Default stiffness in kN/m
    spring_damping_ratio: float = 0.10  # ζ = damping ratio (0.02-0.30)
    spring_clearance_mm: float = 70.0  # Min clearance under plate for hardware
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GPU/ACCELERATION (for M1 Max with Metal/JAX)
    # ═══════════════════════════════════════════════════════════════════════════
    use_jax_fem: bool = True       # Use JAX FEM if available
    batch_size: int = 10           # Evaluate this many genomes in parallel on GPU
    cache_modal_analysis: bool = True  # Cache FEM results for similar geometries
    
    # Stopping
    convergence_threshold: float = 0.00005  # Even stricter: 0.005% improvement
    patience: int = 30              # Generations without improvement before checking
    min_generations_before_stop: int = 60  # Minimum generations before allowing early stop


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION CONFIG PRESETS
# ══════════════════════════════════════════════════════════════════════════════

def get_evolution_preset(name: str) -> EvolutionConfig:
    """
    Get predefined evolution configuration preset.
    
    Args:
        name: One of 'quick', 'standard', 'intense', 'exhaustive'
    
    Returns:
        EvolutionConfig with preset values
    """
    presets = {
        'quick': EvolutionConfig(
            population_size=20,
            n_generations=30,
            elite_count=2,
            max_cutouts=4,
            max_grooves=0,
            min_generations_before_stop=20,
        ),
        'standard': EvolutionConfig(
            population_size=50,
            n_generations=100,
            elite_count=3,
            max_cutouts=6,
            max_grooves=4,
            min_generations_before_stop=60,
        ),
        'intense': EvolutionConfig(
            population_size=100,
            n_generations=300,
            elite_count=5,
            tournament_size=5,
            crossover_rate=0.9,
            mutation_rate=0.2,
            mutation_decay=0.98,
            max_cutouts=8,
            max_grooves=6,
            batch_size=20,  # More parallel on GPU
            min_generations_before_stop=150,
            patience=50,
        ),
        'exhaustive': EvolutionConfig(
            population_size=200,
            n_generations=500,
            elite_count=8,
            tournament_size=6,
            crossover_rate=0.92,
            mutation_rate=0.15,
            mutation_decay=0.99,
            max_cutouts=10,
            max_grooves=8,
            batch_size=30,  # Maximum GPU utilization
            min_generations_before_stop=300,
            patience=80,
            convergence_threshold=0.00001,  # Ultra strict
        ),
    }
    
    if name.lower() not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    
    return presets[name.lower()]


@dataclass
class EvolutionState:
    """
    Stato corrente dell'evoluzione.
    
    Passato al callback per visualizzazione.
    """
    generation: int
    best_genome: PlateGenome
    best_fitness: FitnessResult
    population: List[PlateGenome]
    fitness_history: List[float]
    diversity: float
    elapsed_time: float
    is_final: bool = False


# Type alias per callback
EvolutionCallback = Callable[[EvolutionState], bool]


class EvolutionaryOptimizer:
    """
    Ottimizzatore evolutivo per forma tavola.
    
    Trova la forma ottimale che massimizza fitness multi-obiettivo
    per una specifica persona.
    
    Usage:
        optimizer = EvolutionaryOptimizer(person)
        best = optimizer.run(callback=visualize_progress)
    """
    
    def __init__(
        self,
        person: Person,
        config: Optional[EvolutionConfig] = None,
        objectives: Optional[ObjectiveWeights] = None,
        zone_weights: Optional[ZoneWeights] = None,
        material: str = "birch_plywood",
        seed: Optional[int] = None,
    ):
        """
        Inizializza optimizer.
        
        Args:
            person: Modello persona per fitness
            config: Configurazione GA
            objectives: Pesi obiettivi fitness
            zone_weights: Pesi zone corporee (spine 70%, head 30% default)
            material: Materiale tavola
            seed: Seed random per riproducibilità
        """
        self.person = person
        self.config = config or EvolutionConfig()
        self.material = material
        
        # Evaluator con zone weights
        self.evaluator = FitnessEvaluator(
            person=person,
            objectives=objectives,
            zone_weights=zone_weights,
            material=material,
        )
        
        # Random state
        if seed is not None:
            np.random.seed(seed)
        
        # Stato interno
        self._population: List[PlateGenome] = []
        self._fitness_cache: Dict[int, FitnessResult] = {}
        self._best_genome: Optional[PlateGenome] = None
        self._best_fitness: Optional[FitnessResult] = None
        self._fitness_history: List[float] = []
        self._generation: int = 0
        self._stagnation_count: int = 0
        self._start_time: float = 0.0
        self._current_mutation_sigma: float = self.config.mutation_sigma
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────
    
    def run(
        self,
        callback: Optional[EvolutionCallback] = None,
        verbose: bool = True,
    ) -> PlateGenome:
        """
        Esegue ottimizzazione evolutiva.
        
        Args:
            callback: Funzione chiamata ogni generazione.
                      Ritorna False per interrompere.
            verbose: Stampa progresso su console
        
        Returns:
            Miglior PlateGenome trovato
        """
        self._start_time = time.time()
        
        # 1. Inizializza popolazione
        self._initialize_population()
        
        # 2. Valuta popolazione iniziale
        self._evaluate_population()
        
        if verbose:
            print(f"Initial best fitness: {self._best_fitness.total_fitness:.4f}")
        
        # 3. Loop evolutivo
        for gen in range(self.config.n_generations):
            self._generation = gen + 1
            
            # Evolvi
            self._evolve_generation()
            
            # Valuta
            self._evaluate_population()
            
            # Aggiorna sigma mutazione
            if self.config.adaptive_mutation:
                self._adapt_mutation()
            
            # Calcola diversità
            diversity = self._compute_diversity()
            
            # Callback
            if callback is not None:
                state = EvolutionState(
                    generation=self._generation,
                    best_genome=copy.deepcopy(self._best_genome),
                    best_fitness=copy.deepcopy(self._best_fitness),
                    population=[copy.deepcopy(g) for g in self._population],
                    fitness_history=self._fitness_history.copy(),
                    diversity=diversity,
                    elapsed_time=time.time() - self._start_time,
                    is_final=False,
                )
                if not callback(state):
                    if verbose:
                        print(f"Interrupted at generation {self._generation}")
                    break
            
            if verbose and (gen + 1) % 5 == 0:
                print(f"Gen {gen+1:3d}: best={self._best_fitness.total_fitness:.4f}, "
                      f"div={diversity:.3f}, σ={self._current_mutation_sigma:.3f}")
            
            # Check convergenza
            if self._check_convergence():
                if verbose:
                    print(f"Converged at generation {self._generation}")
                break
        
        # Callback finale
        if callback is not None:
            final_state = EvolutionState(
                generation=self._generation,
                best_genome=copy.deepcopy(self._best_genome),
                best_fitness=copy.deepcopy(self._best_fitness),
                population=[copy.deepcopy(g) for g in self._population],
                fitness_history=self._fitness_history.copy(),
                diversity=self._compute_diversity(),
                elapsed_time=time.time() - self._start_time,
                is_final=True,
            )
            callback(final_state)
        
        return self._best_genome
    
    # ─────────────────────────────────────────────────────────────────────────
    # Population Initialization
    # ─────────────────────────────────────────────────────────────────────────
    
    def _initialize_population(self):
        """Crea popolazione iniziale diversificata."""
        self._population = []
        
        n = self.config.population_size
        
        # Use fixed contour or all allowed contours
        if self.config.fixed_contour is not None:
            contours = [self.config.fixed_contour]
        else:
            contours = self.config.allowed_contours
        n_contours = len(contours)
        
        # Dimensioni base - USA STANDARD 210x80 cm come riferimento
        # ma adatta alla persona se necessario
        base_length = min(self.person.recommended_plate_length, self.config.standard_length)
        base_width = min(self.person.recommended_plate_width, self.config.standard_width)
        
        # Assicura che la superficie non ecceda il massimo
        max_surface = self.config.max_surface_area
        current_surface = base_length * base_width
        if current_surface > max_surface:
            scale = np.sqrt(max_surface / current_surface)
            base_length *= scale
            base_width *= scale
        
        for i in range(n):
            # Varia contour type (cycling through allowed types)
            contour = contours[i % n_contours]
            
            # Varia dimensioni (±10% ma controlla superficie)
            length_var = np.random.uniform(0.9, 1.1)
            width_var = np.random.uniform(0.9, 1.1)
            
            # Check superficie non ecceda max
            proposed_length = base_length * length_var
            proposed_width = base_width * width_var
            if proposed_length * proposed_width > max_surface:
                # Scale down to fit
                scale = np.sqrt(max_surface / (proposed_length * proposed_width))
                length_var *= scale
                width_var *= scale
            
            # Varia spessore
            thickness = np.random.uniform(0.012, 0.020)
            
            # ═══════════════════════════════════════════════════════════════════
            # Generate spring supports with physics-based stiffness distribution
            # References: Den Hartog "Mechanical Vibrations", Beranek "Noise Control"
            # ═══════════════════════════════════════════════════════════════════
            spring_supports = self._generate_spring_supports()
            
            genome = PlateGenome(
                length=base_length * length_var,
                width=base_width * width_var,
                thickness_base=thickness,
                contour_type=contour,
                max_cutouts=self.config.max_cutouts,
                max_grooves=self.config.max_grooves,
                spring_supports=spring_supports,
                min_support_clearance_mm=self.config.spring_clearance_mm,
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # LUTHERIE INIT: Ensure diverse population with cutouts
            # Like violin makers, start with different f-hole configurations
            # ═══════════════════════════════════════════════════════════════════
            if self.config.max_cutouts > 0:
                # 70% start with cutouts (was 50%), higher p_add for diversity
                if np.random.random() < 0.7:
                    # Multiple mutations to ensure cutouts get added
                    for _ in range(3):  # Try 3 times to add cutouts
                        if self.config.enforce_symmetry:
                            genome = genome.mutate_symmetric(p_add_cutout=0.9)
                        else:
                            genome = genome.mutate(p_add_cutout=0.9)
                        if genome.cutouts:  # Stop if we got cutouts
                            break
            
            self._population.append(genome)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    
    # ─────────────────────────────────────────────────────────────────────────
    # Spring Support Generation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _generate_spring_supports(self) -> List[SpringSupportGene]:
        """
        Generate spring supports with physics-based distribution.
        
        Physics rationale (Den Hartog, Harris & Piersol):
        - Stiffer springs (higher k) near structural loads (head end)
        - Softer springs (lower k) near bass zone (foot end) for isolation
        - Damping ratio ζ = 0.08-0.12 for good isolation with some damping
        
        Returns:
            List of SpringSupportGene with evolvable positions
        """
        n = self.config.spring_count
        k_base = self.config.spring_stiffness_kn_m * 1000  # kN/m → N/m
        zeta = self.config.spring_damping_ratio
        
        springs = []
        
        # Distribute springs along Y axis (body length)
        # Symmetric X positions with center spring
        for i in range(n):
            # Y position: from 0.2 (head) to 0.9 (foot)
            y = 0.2 + 0.7 * i / max(n - 1, 1)
            
            # X position: alternate sides with center spring for odd n
            if n == 1:
                x = 0.5
            elif i == n - 1 and n % 2 == 1:  # Center spring at foot
                x = 0.5
            elif i % 2 == 0:
                x = 0.2 + np.random.uniform(-0.05, 0.05)  # Left side
            else:
                x = 0.8 + np.random.uniform(-0.05, 0.05)  # Right side
            
            # Stiffness: stiffer at head (y < 0.5), softer at foot (y > 0.5)
            # This creates bass isolation at foot where spine zone is
            k_factor = 1.2 - 0.4 * y  # 1.2 at y=0, 0.84 at y=0.9
            k = k_base * k_factor * np.random.uniform(0.9, 1.1)  # ±10% variation
            
            # Slight damping variation
            zeta_i = zeta * np.random.uniform(0.9, 1.1)
            
            springs.append(SpringSupportGene(
                x=max(0.1, min(0.9, x)),  # Clamp to valid range
                y=y,
                stiffness_n_m=k,
                damping_ratio=max(0.02, min(0.30, zeta_i)),  # Clamp damping
            ))
        
        return springs
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _evaluate_population(self):
        """Valuta fitness di tutta la popolazione."""
        for genome in self._population:
            genome_hash = hash(str(genome))
            
            if genome_hash not in self._fitness_cache:
                result = self.evaluator.evaluate(genome)
                self._fitness_cache[genome_hash] = result
                genome.fitness = result.total_fitness
            else:
                result = self._fitness_cache[genome_hash]
                genome.fitness = result.total_fitness
            
            # Aggiorna best
            if self._best_genome is None or result.total_fitness > self._best_fitness.total_fitness:
                self._best_genome = copy.deepcopy(genome)
                self._best_fitness = copy.deepcopy(result)
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1
        
        # Registra storia
        self._fitness_history.append(self._best_fitness.total_fitness)
        
        # ═══════════════════════════════════════════════════════════════════
        # LOG GENERATION SUMMARY - Track evolution progress
        # ═══════════════════════════════════════════════════════════════════
        if self._generation > 0:
            # Calculate average fitness
            avg_fitness = np.mean([g.fitness for g in self._population])
            
            # Calculate improvement from previous generation
            improvement = 0.0
            if len(self._fitness_history) >= 2:
                improvement = self._fitness_history[-1] - self._fitness_history[-2]
            
            # Get best genome info
            best_n_cutouts = len(self._best_genome.cutouts) if self._best_genome.cutouts else 0
            best_n_grooves = len(self._best_genome.grooves) if self._best_genome.grooves else 0
            # Handle case where contour_type might be bool instead of enum
            if hasattr(self._best_genome.contour_type, 'name'):
                best_contour = self._best_genome.contour_type.name
            else:
                best_contour = str(self._best_genome.contour_type)
            
            log_generation_summary(
                generation=self._generation,
                best_fitness=self._best_fitness.total_fitness,
                avg_fitness=avg_fitness,
                diversity=self._compute_diversity(),
                mutation_sigma=self._current_mutation_sigma,
                best_n_cutouts=best_n_cutouts,
                best_n_grooves=best_n_grooves,
                best_contour=best_contour,
                improvement=improvement,
                logger=evo_logger
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evolution
    # ─────────────────────────────────────────────────────────────────────────
    
    def _evolve_generation(self):
        """Crea nuova generazione."""
        new_population = []
        
        # Elitismo: mantieni i migliori
        sorted_pop = sorted(
            self._population,
            key=lambda g: g.fitness,
            reverse=True
        )
        elite = sorted_pop[:self.config.elite_count]
        for g in elite:
            elite_copy = copy.deepcopy(g)
            # FORCE config limits on elite too
            elite_copy.max_cutouts = self.config.max_cutouts
            elite_copy.max_grooves = self.config.max_grooves
            if self.config.max_grooves == 0:
                elite_copy.grooves = []
            if self.config.max_cutouts == 0:
                elite_copy.cutouts = []
            new_population.append(elite_copy)
        
        # Genera resto della popolazione
        while len(new_population) < self.config.population_size:
            # Selezione genitori
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover (genera 2 figli con alpha diversi)
            if np.random.random() < self.config.crossover_rate:
                child1 = parent1.crossover(parent2, alpha=0.5 + np.random.uniform(-0.2, 0.2))
                child2 = parent2.crossover(parent1, alpha=0.5 + np.random.uniform(-0.2, 0.2))
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
            
            # FORCE config limits on children (prevents legacy genomes with wrong max values)
            child1.max_cutouts = self.config.max_cutouts
            child1.max_grooves = self.config.max_grooves
            child2.max_cutouts = self.config.max_cutouts
            child2.max_grooves = self.config.max_grooves
            # Clear features if disabled
            if self.config.max_grooves == 0:
                child1.grooves = []
                child2.grooves = []
            if self.config.max_cutouts == 0:
                child1.cutouts = []
                child2.cutouts = []
            
            # Mutazione
            if np.random.random() < self.config.mutation_rate:
                # LUTHERIE: Higher cutout probability (like violin f-holes)
                # Cutouts are the main tuning tool, not just decoration
                p_cutout = 0.4 if self.config.max_cutouts > 0 else 0.0
                if self.config.enforce_symmetry:
                    child1 = child1.mutate_symmetric(
                        sigma_contour=self._current_mutation_sigma,
                        p_add_cutout=p_cutout
                    )
                else:
                    child1 = child1.mutate(
                        sigma_contour=self._current_mutation_sigma,
                        p_add_cutout=p_cutout
                    )
            if np.random.random() < self.config.mutation_rate:
                p_cutout = 0.4 if self.config.max_cutouts > 0 else 0.0
                if self.config.enforce_symmetry:
                    child2 = child2.mutate_symmetric(
                        sigma_contour=self._current_mutation_sigma,
                        p_add_cutout=p_cutout
                    )
                else:
                    child2 = child2.mutate(
                        sigma_contour=self._current_mutation_sigma,
                        p_add_cutout=p_cutout
                    )
            
            # ENFORCE FIXED CONTOUR (AFTER mutation to override any contour changes)
            if self.config.fixed_contour is not None:
                child1.contour_type = self.config.fixed_contour
                child2.contour_type = self.config.fixed_contour
            
            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)
        
        # Injection di diversità se stagnazione
        if self._stagnation_count > self.config.patience // 2:
            self._inject_diversity(new_population)
        
        self._population = new_population
    
    def _select_parent(self) -> PlateGenome:
        """Seleziona genitore con metodo configurato."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        else:
            return self._rank_selection()
    
    def _tournament_selection(self) -> PlateGenome:
        """Selezione torneo."""
        candidates = np.random.choice(
            len(self._population),
            size=self.config.tournament_size,
            replace=False
        )
        best_idx = max(candidates, key=lambda i: self._population[i].fitness)
        return self._population[best_idx]
    
    def _roulette_selection(self) -> PlateGenome:
        """Selezione roulette (fitness proporzionale)."""
        fitnesses = np.array([g.fitness for g in self._population])
        fitnesses = fitnesses - fitnesses.min() + 1e-10
        probs = fitnesses / fitnesses.sum()
        idx = np.random.choice(len(self._population), p=probs)
        return self._population[idx]
    
    def _rank_selection(self) -> PlateGenome:
        """Selezione per rank."""
        n = len(self._population)
        sorted_indices = sorted(
            range(n),
            key=lambda i: self._population[i].fitness
        )
        ranks = np.zeros(n)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        probs = ranks / ranks.sum()
        idx = np.random.choice(n, p=probs)
        return self._population[idx]
    
    def _inject_diversity(self, population: List[PlateGenome]):
        """Inietta individui casuali per diversità."""
        n_inject = int(len(population) * self.config.diversity_injection_rate)
        
        # Determine which contour to use
        if self.config.fixed_contour is not None:
            contour = self.config.fixed_contour
        else:
            contour = np.random.choice(self.config.allowed_contours)
        
        for i in range(n_inject):
            if i < len(population):
                # Sostituisci individuo con uno nuovo
                new_genome = PlateGenome(
                    length=self.person.recommended_plate_length * np.random.uniform(0.85, 1.15),
                    width=self.person.recommended_plate_width * np.random.uniform(0.85, 1.15),
                    thickness_base=np.random.uniform(0.010, 0.022),
                    contour_type=contour if self.config.fixed_contour else np.random.choice(self.config.allowed_contours),
                )
                # Non sostituire elite
                replace_idx = np.random.randint(self.config.elite_count, len(population))
                population[replace_idx] = new_genome
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adaptation & Convergence
    # ─────────────────────────────────────────────────────────────────────────
    
    def _adapt_mutation(self):
        """Adatta sigma mutazione nel tempo."""
        self._current_mutation_sigma = max(
            self.config.min_mutation_sigma,
            self._current_mutation_sigma * self.config.mutation_decay
        )
    
    def _compute_diversity(self) -> float:
        """Calcola diversità della popolazione."""
        if len(self._population) < 2:
            return 0.0
        
        # Diversità basata su varianza parametri
        lengths = np.array([g.length for g in self._population])
        widths = np.array([g.width for g in self._population])
        thicknesses = np.array([g.thickness_base for g in self._population])
        
        cv_length = np.std(lengths) / (np.mean(lengths) + 1e-10)
        cv_width = np.std(widths) / (np.mean(widths) + 1e-10)
        cv_thick = np.std(thicknesses) / (np.mean(thicknesses) + 1e-10)
        
        # Diversità contour types
        contour_counts = {}
        for g in self._population:
            # Handle case where contour_type might be bool instead of enum
            if hasattr(g.contour_type, 'value'):
                ct = g.contour_type.value
            else:
                ct = str(g.contour_type)
            contour_counts[ct] = contour_counts.get(ct, 0) + 1
        
        n = len(self._population)
        contour_entropy = -sum(
            (c/n) * np.log(c/n + 1e-10)
            for c in contour_counts.values()
        ) / np.log(len(self.config.allowed_contours) + 1e-10)
        
        return (cv_length + cv_width + cv_thick + contour_entropy) / 4
    
    def _check_convergence(self) -> bool:
        """Verifica convergenza."""
        # Don't allow early stop before minimum generations
        if self._generation < self.config.min_generations_before_stop:
            return False
        
        if len(self._fitness_history) < self.config.patience:
            return False
        
        recent = self._fitness_history[-self.config.patience:]
        improvement = max(recent) - min(recent)
        
        # Also require high absolute fitness before allowing convergence
        if self._best_fitness and self._best_fitness.total_fitness < 0.85:
            return False
        
        return improvement < self.config.convergence_threshold


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("EVOLUTIONARY OPTIMIZER TEST")
    print("=" * 70)
    
    # Setup
    person = Person(height_m=1.78, weight_kg=80.0)
    
    config = EvolutionConfig(
        population_size=20,
        n_generations=25,
        elite_count=2,
        mutation_rate=0.4,
    )
    
    print(f"\nPerson: {person}")
    print(f"Config: pop={config.population_size}, gen={config.n_generations}")
    
    # Callback semplice per test
    def print_callback(state: EvolutionState) -> bool:
        if state.generation % 5 == 0 or state.is_final:
            print(f"  Gen {state.generation:2d}: "
                  f"fitness={state.best_fitness.total_fitness:.4f}, "
                  f"div={state.diversity:.3f}")
        return True  # Continua
    
    # Run
    optimizer = EvolutionaryOptimizer(
        person=person,
        config=config,
        seed=42,
    )
    
    print("\nRunning evolution...")
    best = optimizer.run(callback=print_callback, verbose=False)
    
    print("\n" + "-" * 50)
    print("RESULT:")
    print("-" * 50)
    print(f"Best genome: {best}")
    print(f"Fitness: {optimizer._best_fitness}")
    print(f"Frequencies: {optimizer._best_fitness.frequencies[:5]} Hz")
