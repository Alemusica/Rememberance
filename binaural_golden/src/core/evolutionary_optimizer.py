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
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple
from enum import Enum

# Local imports
from .person import Person
from .plate_genome import PlateGenome, ContourType
from .fitness import FitnessEvaluator, FitnessResult, ObjectiveWeights


class SelectionMethod(Enum):
    """Metodo di selezione genitori."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


@dataclass
class EvolutionConfig:
    """
    Configurazione algoritmo evolutivo.
    """
    # Popolazione
    population_size: int = 30
    n_generations: int = 50
    elite_count: int = 2
    
    # Selezione
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    
    # Operatori genetici
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.05  # Deviazione standard mutazione
    
    # Adattamento mutazione
    adaptive_mutation: bool = True
    mutation_decay: float = 0.95  # Riduzione sigma per generazione
    min_mutation_sigma: float = 0.01
    
    # Diversità
    diversity_threshold: float = 0.1
    diversity_injection_rate: float = 0.1
    
    # Vincoli
    allowed_contours: List[ContourType] = field(default_factory=lambda: [
        ContourType.GOLDEN_RECT,
        ContourType.RECTANGLE,
        ContourType.ELLIPSE,
        ContourType.OVOID,
    ])
    max_cutouts: int = 4
    
    # Stopping
    convergence_threshold: float = 0.001
    patience: int = 10  # Generazioni senza miglioramento


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
        material: str = "birch_plywood",
        seed: Optional[int] = None,
    ):
        """
        Inizializza optimizer.
        
        Args:
            person: Modello persona per fitness
            config: Configurazione GA
            objectives: Pesi obiettivi fitness
            material: Materiale tavola
            seed: Seed random per riproducibilità
        """
        self.person = person
        self.config = config or EvolutionConfig()
        self.material = material
        
        # Evaluator
        self.evaluator = FitnessEvaluator(
            person=person,
            objectives=objectives,
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
        contours = self.config.allowed_contours
        n_contours = len(contours)
        
        # Dimensioni base da persona
        base_length = self.person.recommended_plate_length
        base_width = self.person.recommended_plate_width
        
        for i in range(n):
            # Varia contour type
            contour = contours[i % n_contours]
            
            # Varia dimensioni
            length_var = np.random.uniform(0.9, 1.1)
            width_var = np.random.uniform(0.9, 1.1)
            
            # Varia spessore
            thickness = np.random.uniform(0.012, 0.020)
            
            genome = PlateGenome(
                length=base_length * length_var,
                width=base_width * width_var,
                thickness_base=thickness,
                contour_type=contour,
            )
            
            # Qualche individuo con cutouts
            if np.random.random() < 0.2:
                genome = genome.mutate(p_add_cutout=0.8)
            
            self._population.append(genome)
    
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
        new_population.extend([copy.deepcopy(g) for g in elite])
        
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
            
            # Mutazione
            if np.random.random() < self.config.mutation_rate:
                child1 = child1.mutate(sigma_contour=self._current_mutation_sigma)
            if np.random.random() < self.config.mutation_rate:
                child2 = child2.mutate(sigma_contour=self._current_mutation_sigma)
            
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
        
        for i in range(n_inject):
            if i < len(population):
                # Sostituisci individuo con uno nuovo
                new_genome = PlateGenome(
                    length=self.person.recommended_plate_length * np.random.uniform(0.85, 1.15),
                    width=self.person.recommended_plate_width * np.random.uniform(0.85, 1.15),
                    thickness_base=np.random.uniform(0.010, 0.022),
                    contour_type=np.random.choice(self.config.allowed_contours),
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
            ct = g.contour_type.value
            contour_counts[ct] = contour_counts.get(ct, 0) + 1
        
        n = len(self._population)
        contour_entropy = -sum(
            (c/n) * np.log(c/n + 1e-10)
            for c in contour_counts.values()
        ) / np.log(len(self.config.allowed_contours) + 1e-10)
        
        return (cv_length + cv_width + cv_thick + contour_entropy) / 4
    
    def _check_convergence(self) -> bool:
        """Verifica convergenza."""
        if len(self._fitness_history) < self.config.patience:
            return False
        
        recent = self._fitness_history[-self.config.patience:]
        improvement = max(recent) - min(recent)
        
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
