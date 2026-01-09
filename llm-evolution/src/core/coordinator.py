"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EVOLUTION COORDINATOR                                ║
║                        "Il Midollo Spinale"                                  ║
║                                                                              ║
║   Traduce decisioni LLM → azioni GA.                                         ║
║   Coordina tutti gli agenti e gestisce il loop principale dell'evoluzione.  ║
║                                                                              ║
║   Può funzionare anche SENZA LLM (tutti gli agenti sono opzionali):         ║
║   - Con LLM: evoluzione guidata con interventi intelligenti                 ║
║   - Senza LLM: evoluzione classica con GA puro                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import logging
import time
import numpy as np

from .genome import (
    GenomeSchema,
    DictGenome,
    random_genome,
    crossover_uniform,
    mutate_gaussian,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVOLUTION STATE
# =============================================================================

@dataclass
class EvolutionState:
    """
    Stato corrente dell'evoluzione.
    
    Traccia tutto ciò che serve per:
    - Continuare l'evoluzione
    - Decidere interventi LLM
    - Logging/debugging
    """
    generation: int = 0
    population: List[DictGenome] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)  # Best fitness per gen
    best_genome: Optional[DictGenome] = None
    best_fitness: float = float('-inf')
    stall_count: int = 0  # Generazioni senza miglioramento
    
    # Additional tracking
    total_evaluations: int = 0
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza stato per logging/LLM context."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "stall_count": self.stall_count,
            "total_evaluations": self.total_evaluations,
            "fitness_history_last10": self.fitness_history[-10:] if self.fitness_history else [],
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
        }
    
    def to_llm_context(self) -> str:
        """Genera context string per prompt LLM."""
        lines = [
            "## Current Evolution State",
            f"- Generation: {self.generation}",
            f"- Population size: {len(self.population)}",
            f"- Best fitness: {self.best_fitness:.6f}",
            f"- Stall count: {self.stall_count}",
            f"- Total evaluations: {self.total_evaluations}",
            f"- Mutation rate: {self.mutation_rate:.3f}",
            f"- Crossover rate: {self.crossover_rate:.3f}",
        ]
        
        if self.fitness_history:
            recent = self.fitness_history[-10:]
            lines.append(f"- Recent fitness history: {[f'{f:.4f}' for f in recent]}")
            
            # Trend analysis
            if len(recent) >= 3:
                trend = recent[-1] - recent[0]
                trend_str = "improving" if trend > 0 else "stagnating" if abs(trend) < 0.001 else "degrading"
                lines.append(f"- Trend: {trend_str} ({trend:+.4f})")
        
        if self.best_genome:
            lines.append("")
            lines.append("## Best Genome")
            for key, val in self.best_genome.to_dict().items():
                if isinstance(val, float):
                    lines.append(f"- {key}: {val:.4f}")
                else:
                    lines.append(f"- {key}: {val}")
        
        return "\n".join(lines)


# =============================================================================
# EVOLUTION RESULT
# =============================================================================

@dataclass
class EvolutionResult:
    """
    Risultato finale dell'evoluzione.
    
    Contiene tutto ciò che serve per:
    - Usare il risultato
    - Analizzare la run
    - Continuare da dove ci si è fermati
    """
    best_genome: DictGenome
    best_fitness: float
    generations_run: int
    total_evaluations: int
    
    # LLM interventions
    agent_interventions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    elapsed_time_seconds: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    
    # Final state (for continuation)
    final_state: Optional[EvolutionState] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza risultato."""
        return {
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "best_fitness": self.best_fitness,
            "generations_run": self.generations_run,
            "total_evaluations": self.total_evaluations,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "agent_interventions_count": len(self.agent_interventions),
            "fitness_history": self.fitness_history,
        }
    
    def summary(self) -> str:
        """Summary string per logging."""
        return (
            f"Evolution completed: {self.generations_run} generations, "
            f"best_fitness={self.best_fitness:.6f}, "
            f"evaluations={self.total_evaluations}, "
            f"time={self.elapsed_time_seconds:.1f}s, "
            f"llm_interventions={len(self.agent_interventions)}"
        )


# =============================================================================
# EVOLUTION COORDINATOR
# =============================================================================

class EvolutionCoordinator:
    """
    Coordinatore principale dell'evoluzione.
    
    "Il Midollo Spinale": traduce decisioni cognitive (LLM) in azioni motorie (GA).
    
    Responsabilità:
    1. Loop principale evoluzione
    2. Coordinamento agenti LLM (opzionali)
    3. Applicazione operatori GA
    4. Tracking stato e metriche
    
    Può funzionare senza LLM per evoluzione classica.
    """
    
    def __init__(
        self,
        orchestrator=None,  # OrchestratorAgent
        strategy=None,      # StrategyAgent (if exists)
        analysis=None,      # AnalysisAgent
        rag=None,           # RAGAgent
        tools=None,         # ToolRegistry
        rng: np.random.Generator = None,
    ):
        """
        Args:
            orchestrator: Agente orchestratore (32B reasoning)
            strategy: Agente strategia (parametri GA)
            analysis: Agente analisi (interpretazione)
            rag: Agente RAG (knowledge retrieval)
            tools: Registry dei tool disponibili
            rng: Random generator (per riproducibilità)
        """
        self._orchestrator = orchestrator
        self._strategy = strategy
        self._analysis = analysis
        self._rag = rag
        self._tools = tools
        self._rng = rng or np.random.default_rng()
        
        # State
        self._state: Optional[EvolutionState] = None
        self._schema: Optional[GenomeSchema] = None
        
        # Logging
        self._interventions: List[Dict[str, Any]] = []
        
        # Check what's available
        self._has_llm = orchestrator is not None
        
        logger.info(
            f"EvolutionCoordinator initialized "
            f"(LLM={'enabled' if self._has_llm else 'disabled'})"
        )
    
    async def run(
        self,
        schema: GenomeSchema,
        fitness_fn: Callable[[DictGenome], float],
        max_generations: int = 100,
        population_size: int = 50,
        llm_intervention_frequency: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        elitism: int = 2,
        target_fitness: Optional[float] = None,
        max_stall: int = 20,
    ) -> EvolutionResult:
        """
        Esegue l'evoluzione completa.
        
        Args:
            schema: Schema del genome
            fitness_fn: Funzione fitness (genome → float)
            max_generations: Numero massimo generazioni
            population_size: Dimensione popolazione
            llm_intervention_frequency: Ogni N generazioni chiedi all'LLM
            mutation_rate: Probabilità mutazione per gene
            crossover_rate: Probabilità crossover
            tournament_size: Dimensione torneo per selezione
            elitism: Numero individui da preservare
            target_fitness: Fitness target (opzionale, termina se raggiunto)
            max_stall: Max generazioni senza miglioramento
        
        Returns:
            EvolutionResult con miglior genome e metriche
        """
        start_time = time.time()
        self._schema = schema
        self._interventions = []
        
        logger.info(
            f"Starting evolution: {schema.name}, "
            f"pop_size={population_size}, max_gen={max_generations}"
        )
        
        # Initialize state
        self._state = EvolutionState(
            generation=0,
            population=[],
            fitness_history=[],
            best_genome=None,
            best_fitness=float('-inf'),
            stall_count=0,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )
        
        # Generate initial population
        self._state.population = [
            random_genome(schema, self._rng)
            for _ in range(population_size)
        ]
        
        # Main evolution loop
        for gen in range(max_generations):
            self._state.generation = gen
            
            # 1. Evaluate fitness
            fitnesses = await self._evaluate_population(fitness_fn)
            
            # 2. Update best
            best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[best_idx]
            
            if gen_best_fitness > self._state.best_fitness:
                self._state.best_fitness = gen_best_fitness
                self._state.best_genome = self._state.population[best_idx].clone()
                self._state.stall_count = 0
            else:
                self._state.stall_count += 1
            
            self._state.fitness_history.append(gen_best_fitness)
            
            # Log progress
            if gen % 10 == 0 or gen == max_generations - 1:
                logger.info(
                    f"Gen {gen}: best={gen_best_fitness:.6f}, "
                    f"overall_best={self._state.best_fitness:.6f}, "
                    f"stall={self._state.stall_count}"
                )
            
            # 3. Check termination conditions
            if target_fitness is not None and gen_best_fitness >= target_fitness:
                logger.info(f"Target fitness reached at generation {gen}")
                break
            
            if self._state.stall_count >= max_stall:
                logger.info(f"Stall limit reached at generation {gen}")
                break
            
            # 4. LLM intervention (if enabled and it's time)
            if self._has_llm and gen > 0 and gen % llm_intervention_frequency == 0:
                await self._llm_intervention(gen, fitnesses)
            
            # 5. Selection
            parents = self._tournament_select(
                fitnesses,
                n_select=population_size - elitism,
                tournament_size=tournament_size,
            )
            
            # 6. Crossover
            offspring = self._apply_crossover(parents)
            
            # 7. Mutation
            offspring = self._apply_mutation(offspring)
            
            # 8. Elitism: preserve best individuals
            elite = self._select_elite(fitnesses, elitism)
            
            # 9. Create new population
            self._state.population = elite + offspring
        
        # Build result
        elapsed = time.time() - start_time
        
        result = EvolutionResult(
            best_genome=self._state.best_genome,
            best_fitness=self._state.best_fitness,
            generations_run=self._state.generation + 1,
            total_evaluations=self._state.total_evaluations,
            agent_interventions=self._interventions,
            elapsed_time_seconds=elapsed,
            fitness_history=self._state.fitness_history,
            final_state=self._state,
        )
        
        logger.info(result.summary())
        return result
    
    # =========================================================================
    # FITNESS EVALUATION
    # =========================================================================
    
    async def _evaluate_population(
        self,
        fitness_fn: Callable[[DictGenome], float],
    ) -> List[float]:
        """Valuta fitness di tutta la popolazione."""
        fitnesses = []
        
        for genome in self._state.population:
            try:
                # Support both sync and async fitness functions
                import asyncio
                if asyncio.iscoroutinefunction(fitness_fn):
                    fit = await fitness_fn(genome)
                else:
                    fit = fitness_fn(genome)
                fitnesses.append(float(fit))
            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")
                fitnesses.append(float('-inf'))
            
            self._state.total_evaluations += 1
        
        return fitnesses
    
    # =========================================================================
    # SELECTION
    # =========================================================================
    
    def _tournament_select(
        self,
        fitnesses: List[float],
        n_select: int,
        tournament_size: int = 3,
    ) -> List[DictGenome]:
        """
        Selezione a torneo.
        
        Per ogni slot, sceglie tournament_size individui random
        e seleziona il migliore.
        """
        selected = []
        pop_size = len(self._state.population)
        
        for _ in range(n_select):
            # Random tournament
            candidates = self._rng.choice(pop_size, size=tournament_size, replace=False)
            
            # Find best in tournament
            best_idx = candidates[0]
            best_fit = fitnesses[best_idx]
            
            for idx in candidates[1:]:
                if fitnesses[idx] > best_fit:
                    best_idx = idx
                    best_fit = fitnesses[idx]
            
            selected.append(self._state.population[best_idx])
        
        return selected
    
    def _select_elite(
        self,
        fitnesses: List[float],
        n_elite: int,
    ) -> List[DictGenome]:
        """Seleziona i migliori N individui (elitismo)."""
        if n_elite <= 0:
            return []
        
        # Get indices of top N
        indices = np.argsort(fitnesses)[-n_elite:]
        
        return [self._state.population[i].clone() for i in indices]
    
    # =========================================================================
    # CROSSOVER
    # =========================================================================
    
    def _apply_crossover(
        self,
        parents: List[DictGenome],
    ) -> List[DictGenome]:
        """
        Applica crossover ai genitori selezionati.
        
        Ogni coppia di genitori produce un figlio.
        """
        offspring = []
        
        # Shuffle parents
        self._rng.shuffle(parents)
        
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            
            if self._rng.random() < self._state.crossover_rate:
                child = crossover_uniform(p1, p2, self._rng)
            else:
                # No crossover, just clone one parent
                child = p1.clone() if self._rng.random() < 0.5 else p2.clone()
            
            offspring.append(child)
        
        # If odd number of parents, add last one
        if len(parents) % 2 == 1:
            offspring.append(parents[-1].clone())
        
        # Fill remaining slots if needed
        while len(offspring) < len(parents):
            idx = self._rng.integers(len(parents))
            offspring.append(parents[idx].clone())
        
        return offspring
    
    # =========================================================================
    # MUTATION
    # =========================================================================
    
    def _apply_mutation(
        self,
        population: List[DictGenome],
    ) -> List[DictGenome]:
        """Applica mutazione gaussiana alla popolazione."""
        mutated = []
        
        for genome in population:
            mutated.append(
                mutate_gaussian(
                    genome,
                    sigma=0.1,
                    gene_rate=self._state.mutation_rate,
                    rng=self._rng,
                )
            )
        
        return mutated
    
    # =========================================================================
    # LLM INTERVENTION
    # =========================================================================
    
    async def _llm_intervention(
        self,
        generation: int,
        fitnesses: List[float],
    ):
        """
        Chiede all'LLM suggerimenti e li applica.
        
        Questo è il cuore dell'integrazione LLM-GA:
        l'orchestrator analizza lo stato e suggerisce azioni.
        """
        if not self._orchestrator:
            return
        
        logger.info(f"LLM intervention at generation {generation}")
        
        try:
            # Prepare context
            context = {
                "generation": generation,
                "fitness_history": self._state.fitness_history,
                "stall_count": self._state.stall_count,
                "current_mutation_rate": self._state.mutation_rate,
                "current_crossover_rate": self._state.crossover_rate,
                "best_fitness": self._state.best_fitness,
                "avg_fitness": float(np.mean(fitnesses)),
                "fitness_std": float(np.std(fitnesses)),
            }
            
            # Ask orchestrator for strategy
            response = await self._orchestrator.process(
                {"type": "suggest_strategy"},
                context=context,
            )
            
            if response.success and response.result:
                result = response.result
                
                # Apply suggested changes
                intervention = {
                    "generation": generation,
                    "type": "strategy_adjustment",
                    "suggestions": result,
                    "applied": [],
                }
                
                # Mutation rate adjustment
                if "mutation_rate_delta" in result:
                    delta = float(result["mutation_rate_delta"])
                    new_rate = np.clip(self._state.mutation_rate + delta, 0.01, 0.5)
                    if new_rate != self._state.mutation_rate:
                        self._state.mutation_rate = new_rate
                        intervention["applied"].append(f"mutation_rate → {new_rate:.3f}")
                
                # Diversity injection
                if result.get("inject_diversity", False):
                    n_inject = max(1, len(self._state.population) // 10)
                    for i in range(n_inject):
                        idx = self._rng.integers(len(self._state.population))
                        self._state.population[idx] = random_genome(self._schema, self._rng)
                    intervention["applied"].append(f"injected {n_inject} random individuals")
                
                # Log intervention
                intervention["confidence"] = response.confidence
                intervention["reasoning"] = response.reasoning
                self._interventions.append(intervention)
                
                if intervention["applied"]:
                    logger.info(f"Applied: {intervention['applied']}")
        
        except Exception as e:
            logger.error(f"LLM intervention failed: {e}")
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _check_stall(self, threshold: float = 1e-6) -> bool:
        """
        Verifica se l'evoluzione è in stallo.
        
        Stallo = nessun miglioramento significativo nelle ultime N generazioni.
        """
        if len(self._state.fitness_history) < 5:
            return False
        
        recent = self._state.fitness_history[-5:]
        improvement = max(recent) - min(recent)
        
        return improvement < threshold
    
    def _update_state(
        self,
        new_population: List[DictGenome],
        fitnesses: List[float],
    ):
        """Aggiorna stato con nuova popolazione."""
        self._state.population = new_population
        
        # Update best if improved
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self._state.best_fitness:
            self._state.best_fitness = fitnesses[best_idx]
            self._state.best_genome = new_population[best_idx].clone()
            self._state.stall_count = 0
        else:
            self._state.stall_count += 1
        
        self._state.fitness_history.append(fitnesses[best_idx])
    
    @property
    def state(self) -> Optional[EvolutionState]:
        """Stato corrente."""
        return self._state
    
    @property
    def has_llm(self) -> bool:
        """True se LLM è disponibile."""
        return self._has_llm
