"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          EVOLUTION PIPELINE - Unified Integration of All Phases              ║
║                                                                              ║
║   This module integrates all Action Plan 3.0 components into a unified       ║
║   pipeline for intelligent plate evolution:                                  ║
║                                                                              ║
║   Phase 1: PokayokeObserver    → Anomaly detection, PAUSE + ASK USER        ║
║   Phase 2: ExciterGene         → Staged gene activation (position→emission) ║
║   Phase 3: PhysicsRulesEngine  → Hybrid physics + learned rules             ║
║   Phase 4: RDNNMemory          → PyTorch recurrent with warm start          ║
║   Phase 5: LTMDistiller        → Long-term knowledge extraction             ║
║   Phase 6: ScoringTemplates    → Zone-specific fitness configurations       ║
║                                                                              ║
║   KEY PHILOSOPHY:                                                            ║
║   "The pipeline is the conductor - each component is an instrument"         ║
║   Each component can operate independently, but the pipeline orchestrates   ║
║   them into a coherent optimization process.                                ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Bai & Liu 2004: Genetic algorithms for exciter placement                 ║
║   • NSGA-II (Deb 2002): Multi-objective optimization                         ║
║   • Meta-learning: Transfer learning across optimization runs               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class PipelineMode(Enum):
    """Operating mode for the pipeline."""
    HEADLESS = auto()     # No user interaction (automated)
    INTERACTIVE = auto()  # Full user interaction on anomalies
    MONITORED = auto()    # Log anomalies but don't pause


@dataclass
class PipelineConfig:
    """
    Configuration for the evolution pipeline.
    
    Centralizes settings for all integrated components.
    """
    # Operation mode
    mode: PipelineMode = PipelineMode.HEADLESS
    
    # Component enable flags
    enable_pokayoke: bool = True
    enable_physics_rules: bool = True
    enable_rdnn: bool = True
    enable_ltm: bool = True
    enable_templates: bool = True
    
    # LTM settings
    ltm_path: Optional[Path] = None
    distill_on_complete: bool = True
    min_runs_for_distillation: int = 3
    
    # RDNN settings
    rdnn_architecture: str = "gru"  # "gru" or "lstm"
    rdnn_hidden_size: int = 64
    rdnn_checkpoint_interval: int = 10
    
    # Template settings
    template_name: str = "VAT Therapy"
    custom_template: Optional[Any] = None  # ScoringTemplate instance
    
    # Physics rules
    learn_from_experience: bool = True
    min_confidence_for_rules: float = 0.7
    
    # Observer thresholds
    fitness_stall_threshold: int = 10
    anomaly_detection_sensitivity: float = 0.5
    
    # Evolution settings (passed to optimizer)
    population_size: int = 50
    n_generations: int = 100
    
    # Logging
    log_every_n_generations: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: Optional[Path] = None


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineState:
    """
    Mutable state of the evolution pipeline.
    
    Tracks progress, decisions, and component states.
    """
    # Run info
    run_id: str = ""
    start_time: float = 0.0
    current_generation: int = 0
    total_evaluations: int = 0
    
    # Best solution tracking
    best_fitness: float = float('-inf')
    best_genome: Optional[Any] = None
    best_generation: int = 0
    
    # Convergence tracking
    stall_count: int = 0
    last_improvement_gen: int = 0
    convergence_history: List[float] = field(default_factory=list)
    
    # Component states
    rdnn_hidden_state: Optional[Any] = None
    observer_pause_count: int = 0
    rules_applied: List[str] = field(default_factory=list)
    
    # Statistics
    gene_activation_timeline: Dict[int, str] = field(default_factory=dict)
    anomalies_detected: List[Dict] = field(default_factory=list)
    user_decisions: List[Dict] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Result of a complete pipeline run.
    
    Contains optimized genome, statistics, and extracted knowledge.
    """
    # Best solution
    best_genome: Any
    best_fitness: float
    
    # Run stats
    total_generations: int
    total_evaluations: int
    runtime_seconds: float
    converged: bool
    
    # Component outputs
    distilled_knowledge: Optional[Any] = None
    learned_rules: List[Any] = field(default_factory=list)
    rdnn_state: Optional[Any] = None
    
    # History
    fitness_history: List[float] = field(default_factory=list)
    best_per_generation: List[float] = field(default_factory=list)
    
    # Anomaly log
    anomalies: List[Dict] = field(default_factory=list)
    user_interventions: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionPipeline:
    """
    Unified pipeline for intelligent plate evolution.
    
    Orchestrates all Phase 1-6 components into a coherent optimization process.
    
    ARCHITECTURE:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         EVOLUTION PIPELINE                              │
    │                                                                         │
    │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐          │
    │  │ ScoringTemp   │───▶│ FitnessEval   │───▶│ Selection     │          │
    │  │ (Phase 6)     │    │               │    │               │          │
    │  └───────────────┘    └───────────────┘    └───────────────┘          │
    │          │                    │                    │                   │
    │          │                    ▼                    │                   │
    │          │           ┌───────────────┐            │                   │
    │          │           │ PokayokeObs   │────────────┤                   │
    │          │           │ (Phase 1)     │            │                   │
    │          │           └───────────────┘            │                   │
    │          │                    │                    │                   │
    │          ▼                    ▼                    ▼                   │
    │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐          │
    │  │ PhysicsRules  │───▶│ Mutation      │───▶│ ExciterGene   │          │
    │  │ (Phase 3)     │    │               │    │ (Phase 2)     │          │
    │  └───────────────┘    └───────────────┘    └───────────────┘          │
    │          │                    │                    │                   │
    │          ▼                    ▼                    ▼                   │
    │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐          │
    │  │ RDNNMemory    │◀──▶│ LTMDistiller  │◀───│ Archive       │          │
    │  │ (Phase 4)     │    │ (Phase 5)     │    │               │          │
    │  └───────────────┘    └───────────────┘    └───────────────┘          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    USAGE:
        from core.evolution_pipeline import EvolutionPipeline, PipelineConfig
        
        config = PipelineConfig(
            mode=PipelineMode.HEADLESS,
            template_name="VAT Therapy",
            population_size=50,
            n_generations=100,
        )
        
        pipeline = EvolutionPipeline(config, person)
        result = pipeline.run()
        
        print(f"Best fitness: {result.best_fitness}")
        print(f"Runtime: {result.runtime_seconds:.1f}s")
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        person: Optional[Any] = None,
    ):
        """
        Initialize the evolution pipeline.
        
        Args:
            config: Pipeline configuration
            person: Person instance for body zone positioning
        """
        self.config = config
        self.person = person
        self.state = PipelineState()
        
        # Components (lazy-loaded)
        self._observer: Optional[Any] = None
        self._physics_engine: Optional[Any] = None
        self._rdnn: Optional[Any] = None
        self._ltm_distiller: Optional[Any] = None
        self._template: Optional[Any] = None
        self._evaluator: Optional[Any] = None
        
        # Callbacks
        self._generation_callbacks: List[Callable] = []
        self._anomaly_callbacks: List[Callable] = []
        
        # Run ID
        import uuid
        self.state.run_id = str(uuid.uuid4())[:8]
        
        logger.info(f"EvolutionPipeline initialized (run_id={self.state.run_id})")
    
    # ══════════════════════════════════════════════════════════════════════════
    # COMPONENT INITIALIZATION (Lazy Loading)
    # ══════════════════════════════════════════════════════════════════════════
    
    def _init_observer(self):
        """Initialize PokayokeObserver (Phase 1)."""
        if not self.config.enable_pokayoke:
            return None
        
        try:
            from .pokayoke_observer import create_observer
            from .analysis_config import ObserverConfig
            
            observer_config = ObserverConfig(
                stagnation_threshold=self.config.fitness_stall_threshold,
                verbose=True,
            )
            
            # Determine mode based on pipeline mode
            if self.config.mode == PipelineMode.HEADLESS:
                mode = "headless"
            elif self.config.mode == PipelineMode.INTERACTIVE:
                mode = "cli"
            else:  # MONITORED
                mode = "headless"
            
            self._observer = create_observer(
                mode=mode,
                config=observer_config,
            )
            
            logger.debug("PokayokeObserver initialized")
            return self._observer
            
        except ImportError as e:
            logger.warning(f"Could not import PokayokeObserver: {e}")
            return None
    
    def _init_physics_engine(self):
        """Initialize PhysicsRulesEngine (Phase 3)."""
        if not self.config.enable_physics_rules:
            return None
        
        try:
            from .physics_rules import create_physics_engine
            
            self._physics_engine = create_physics_engine()
            
            logger.debug("PhysicsRulesEngine initialized")
            return self._physics_engine
            
        except ImportError as e:
            logger.warning(f"Could not import PhysicsRulesEngine: {e}")
            return None
    
    def _init_rdnn(self):
        """Initialize RDNNMemory (Phase 4)."""
        if not self.config.enable_rdnn:
            return None
        
        try:
            from .rdnn_memory import create_rdnn_memory, RDNNArchitecture
            
            arch = RDNNArchitecture.GRU if self.config.rdnn_architecture == "gru" else RDNNArchitecture.LSTM
            
            self._rdnn = create_rdnn_memory(
                architecture=arch,
                hidden_size=self.config.rdnn_hidden_size,
            )
            
            logger.debug("RDNNMemory initialized")
            return self._rdnn
            
        except ImportError as e:
            logger.warning(f"Could not import RDNNMemory: {e}")
            return None
    
    def _init_ltm_distiller(self):
        """Initialize LTMDistiller (Phase 5)."""
        if not self.config.enable_ltm:
            return None
        
        try:
            from .ltm_distillation import create_distiller
            
            self._ltm_distiller = create_distiller()
            
            logger.debug("LTMDistiller initialized")
            return self._ltm_distiller
            
        except ImportError as e:
            logger.warning(f"Could not import LTMDistiller: {e}")
            return None
    
    def _init_template(self):
        """Initialize ScoringTemplate (Phase 6)."""
        if not self.config.enable_templates:
            return None
        
        try:
            from .scoring_templates import get_template
            
            if self.config.custom_template:
                self._template = self.config.custom_template
            else:
                self._template = get_template(self.config.template_name)
            
            logger.debug(f"ScoringTemplate initialized: {self._template.name}")
            return self._template
            
        except ImportError as e:
            logger.warning(f"Could not import ScoringTemplates: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Template not found: {e}")
            return None
    
    def _init_evaluator(self):
        """Initialize FitnessEvaluator with template."""
        try:
            from .fitness import FitnessEvaluator, ObjectiveWeights, ZoneWeights
            
            if self._template:
                from .scoring_templates import TemplateAdapter
                adapter = TemplateAdapter(self._template)
                obj_weights = adapter.get_objective_weights()
                zone_weights = adapter.get_zone_weights()
                freq_range = adapter.get_freq_range()
                n_freq_points = adapter.get_n_freq_points()
            else:
                obj_weights = ObjectiveWeights()
                zone_weights = ZoneWeights()
                freq_range = (20, 200)
                n_freq_points = 50
            
            self._evaluator = FitnessEvaluator(
                person=self.person,
                objectives=obj_weights,
                zone_weights=zone_weights,
                freq_range=freq_range,
                n_freq_points=n_freq_points,
            )
            
            logger.debug("FitnessEvaluator initialized")
            return self._evaluator
            
        except Exception as e:
            logger.error(f"Could not initialize FitnessEvaluator: {e}")
            return None
    
    def initialize(self):
        """
        Initialize all pipeline components.
        
        Called automatically before run(), but can be called manually
        for pre-flight checks.
        """
        logger.info("Initializing pipeline components...")
        
        self._init_template()
        self._init_observer()
        self._init_physics_engine()
        self._init_rdnn()
        self._init_ltm_distiller()
        self._init_evaluator()
        
        active = []
        if self._observer: active.append("Pokayoke")
        if self._physics_engine: active.append("PhysicsRules")
        if self._rdnn: active.append("RDNN")
        if self._ltm_distiller: active.append("LTM")
        if self._template: active.append("Templates")
        
        logger.info(f"Active components: {', '.join(active) or 'None'}")
        
        return self
    
    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION WITH INTEGRATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def evaluate_genome(self, genome) -> Tuple[float, Any]:
        """
        Evaluate a genome using all integrated components.
        
        Integrates:
        - PhysicsRulesEngine for rule suggestions
        - FitnessEvaluator for objective calculation
        - PokayokeObserver for anomaly detection
        - RDNNMemory for prediction feedback
        
        Args:
            genome: PlateGenome instance
        
        Returns:
            Tuple of (fitness_score, FitnessResult)
        """
        self.state.total_evaluations += 1
        
        # 1. Apply physics rules (if enabled)
        rule_suggestions = []
        if self._physics_engine:
            try:
                from .physics_rules import create_rule_context
                context = create_rule_context(genome)
                eval_result = self._physics_engine.evaluate_all(context)
                rule_suggestions = eval_result.suggestions
                
                # Track applied rules
                for s in rule_suggestions:
                    if s.rule_name not in self.state.rules_applied:
                        self.state.rules_applied.append(s.rule_name)
                        
            except Exception as e:
                logger.debug(f"Physics rules evaluation error: {e}")
        
        # 2. Evaluate fitness
        result = None
        if self._evaluator:
            try:
                result = self._evaluator.evaluate(genome)
                fitness = result.total_score
            except Exception as e:
                logger.error(f"Fitness evaluation error: {e}")
                fitness = float('-inf')
        else:
            # Fallback: simple score
            fitness = 0.0
        
        # 3. Feed to RDNN (if enabled)
        if self._rdnn and result:
            try:
                from .rdnn_memory import ObservationBuilder
                
                builder = ObservationBuilder()
                obs = builder.from_fitness_result(result).build()
                
                prediction = self._rdnn.predict(obs)
                # Could use prediction to adjust mutation rate
                
            except Exception as e:
                logger.debug(f"RDNN prediction error: {e}")
        
        # 4. Check with observer (if enabled)
        if self._observer:
            try:
                anomaly = self._observer.check_fitness(
                    fitness=fitness,
                    generation=self.state.current_generation,
                    genome_id=getattr(genome, 'id', None),
                )
                
                if anomaly:
                    self.state.anomalies_detected.append({
                        "generation": self.state.current_generation,
                        "type": anomaly.anomaly_type.name,
                        "fitness": fitness,
                    })
                    
                    for cb in self._anomaly_callbacks:
                        cb(anomaly)
                        
            except Exception as e:
                logger.debug(f"Observer check error: {e}")
        
        return fitness, result
    
    def evaluate_population(self, population: List) -> List[Tuple[float, Any]]:
        """
        Evaluate entire population.
        
        Args:
            population: List of PlateGenome instances
        
        Returns:
            List of (fitness, result) tuples
        """
        return [self.evaluate_genome(g) for g in population]
    
    # ══════════════════════════════════════════════════════════════════════════
    # GENETIC OPERATORS WITH INTEGRATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def apply_exciter_genes(self, genome, generation: int):
        """
        Apply ExciterGene staged activation (Phase 2).
        
        Early generations: Freeze position, only mutate emission
        Later generations: Allow full position + emission mutation
        """
        if not hasattr(genome, 'exciters') or not genome.exciters:
            return genome
        
        try:
            from .exciter_gene import GenePhase
            from .analysis_config import get_default_config
            
            config = get_default_config()
            
            # Check phase based on generation
            if generation < config.gene_activation.position_freeze_generations:
                phase = GenePhase.POSITION_FROZEN
                self.state.gene_activation_timeline[generation] = "POSITION_FROZEN"
            else:
                phase = GenePhase.FULL
                self.state.gene_activation_timeline[generation] = "FULL"
            
            # Apply phase to exciters
            for exc in genome.exciters:
                if hasattr(exc, 'set_phase'):
                    exc.set_phase(phase)
                    
        except Exception as e:
            logger.debug(f"ExciterGene application error: {e}")
        
        return genome
    
    def get_rdnn_mutation_rate(self) -> Optional[float]:
        """
        Get RDNN-suggested mutation rate.
        
        Returns:
            Suggested mutation rate, or None to use default
        """
        if not self._rdnn or not self.state.rdnn_hidden_state:
            return None
        
        try:
            # Get last prediction
            if hasattr(self._rdnn, 'get_last_prediction'):
                pred = self._rdnn.get_last_prediction()
                if pred and hasattr(pred, 'suggested_mutation_rate'):
                    return pred.suggested_mutation_rate
        except Exception:
            pass
        
        return None
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAIN RUN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    
    def run(
        self,
        initial_population: Optional[List] = None,
        callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Run the complete evolution pipeline.
        
        Args:
            initial_population: Optional starting population
            callback: Optional per-generation callback(generation, best_fitness, best_genome)
        
        Returns:
            PipelineResult with optimized genome and statistics
        """
        self.state.start_time = time.time()
        
        # Initialize components if not done
        if not self._evaluator:
            self.initialize()
        
        # Create initial population if needed
        if initial_population is None:
            population = self._create_initial_population()
        else:
            population = initial_population
        
        logger.info(
            f"Starting evolution: pop={len(population)}, "
            f"gen={self.config.n_generations}, mode={self.config.mode.name}"
        )
        
        # Initialize RDNN warm start
        if self._rdnn:
            self._rdnn.start_run(run_id=self.state.run_id)
        
        # Main evolution loop
        fitness_history = []
        best_per_gen = []
        
        for gen in range(self.config.n_generations):
            self.state.current_generation = gen
            
            # Apply gene activation phase
            for genome in population:
                self.apply_exciter_genes(genome, gen)
            
            # Evaluate population
            eval_results = self.evaluate_population(population)
            fitnesses = [f for f, _ in eval_results]
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_genome = population[gen_best_idx]
            
            best_per_gen.append(gen_best_fitness)
            fitness_history.extend(fitnesses)
            
            # Update global best
            if gen_best_fitness > self.state.best_fitness:
                self.state.best_fitness = gen_best_fitness
                self.state.best_genome = gen_best_genome
                self.state.best_generation = gen
                self.state.last_improvement_gen = gen
                self.state.stall_count = 0
            else:
                self.state.stall_count += 1
            
            # Convergence history
            self.state.convergence_history.append(gen_best_fitness)
            
            # Log progress
            if gen % self.config.log_every_n_generations == 0:
                logger.info(
                    f"Gen {gen:4d}: best={gen_best_fitness:.4f}, "
                    f"global_best={self.state.best_fitness:.4f}, "
                    f"stall={self.state.stall_count}"
                )
            
            # Callbacks
            if callback:
                callback(gen, gen_best_fitness, gen_best_genome)
            
            for cb in self._generation_callbacks:
                cb(gen, gen_best_fitness, gen_best_genome)
            
            # Check early stopping
            if self._should_stop(gen):
                logger.info(f"Early stopping at generation {gen}")
                break
            
            # Create next generation
            population = self._create_next_generation(population, fitnesses)
        
        # Finalize RDNN
        if self._rdnn:
            self._rdnn.end_run(
                best_fitness=self.state.best_fitness,
                generations=self.state.current_generation,
            )
            self.state.rdnn_hidden_state = self._rdnn.get_hidden_state()
        
        # Distill knowledge
        distilled = None
        if self._ltm_distiller and self.config.distill_on_complete:
            try:
                from .evolution_memory import LongTermMemory
                ltm = LongTermMemory()
                distilled = self._ltm_distiller.distill_all(ltm)
            except Exception as e:
                logger.warning(f"Distillation failed: {e}")
        
        # Build result
        runtime = time.time() - self.state.start_time
        
        result = PipelineResult(
            best_genome=self.state.best_genome,
            best_fitness=self.state.best_fitness,
            total_generations=self.state.current_generation + 1,
            total_evaluations=self.state.total_evaluations,
            runtime_seconds=runtime,
            converged=self.state.stall_count == 0,
            distilled_knowledge=distilled,
            rdnn_state=self.state.rdnn_hidden_state,
            fitness_history=fitness_history,
            best_per_generation=best_per_gen,
            anomalies=self.state.anomalies_detected,
            user_interventions=self.state.observer_pause_count,
        )
        
        logger.info(
            f"Evolution complete: fitness={result.best_fitness:.4f}, "
            f"evals={result.total_evaluations}, time={result.runtime_seconds:.1f}s"
        )
        
        return result
    
    def _should_stop(self, generation: int) -> bool:
        """Check if evolution should stop early."""
        # Minimum generations
        if generation < 20:
            return False
        
        # Stall detection
        if self.state.stall_count > self.config.fitness_stall_threshold * 3:
            return True
        
        return False
    
    def _create_initial_population(self) -> List:
        """Create initial population of genomes."""
        try:
            from .plate_genome import PlateGenome, ContourType
            
            population = []
            for _ in range(self.config.population_size):
                genome = PlateGenome.random()
                population.append(genome)
            
            return population
            
        except Exception as e:
            logger.error(f"Could not create initial population: {e}")
            return []
    
    def _create_next_generation(
        self,
        population: List,
        fitnesses: List[float],
    ) -> List:
        """
        Create next generation using genetic operators.
        
        Uses RDNN suggestions for mutation rate if available.
        """
        new_population = []
        
        # Elitism: keep top N
        elite_count = max(1, self.config.population_size // 10)
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending
        
        for i in range(elite_count):
            import copy
            new_population.append(copy.deepcopy(population[sorted_indices[i]]))
        
        # Get mutation rate
        mutation_rate = self.get_rdnn_mutation_rate()
        if mutation_rate is None:
            mutation_rate = 0.25  # Default
        
        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population, fitnesses)
            parent2 = self._tournament_select(population, fitnesses)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child, mutation_rate)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_select(self, population: List, fitnesses: List[float], k: int = 3):
        """Tournament selection."""
        indices = np.random.choice(len(population), size=k, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        import copy
        return copy.deepcopy(population[best_idx])
    
    def _crossover(self, parent1, parent2):
        """Simple crossover between two genomes."""
        import copy
        child = copy.deepcopy(parent1)
        
        # Exchange some genes from parent2
        if hasattr(child, 'width') and hasattr(parent2, 'width'):
            if np.random.random() < 0.5:
                child.width = parent2.width
            if np.random.random() < 0.5:
                child.height = parent2.height
        
        return child
    
    def _mutate(self, genome, rate: float):
        """Apply mutation with given rate."""
        if np.random.random() > rate:
            return genome
        
        # Apply small random changes
        if hasattr(genome, 'width'):
            genome.width *= np.random.uniform(0.95, 1.05)
        if hasattr(genome, 'height'):
            genome.height *= np.random.uniform(0.95, 1.05)
        
        return genome
    
    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════
    
    def add_generation_callback(self, callback: Callable):
        """Add callback for each generation."""
        self._generation_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable):
        """Add callback for anomaly detection."""
        self._anomaly_callbacks.append(callback)
    
    # ══════════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ══════════════════════════════════════════════════════════════════════════
    
    @property
    def observer(self):
        """Get PokayokeObserver instance."""
        return self._observer
    
    @property
    def physics_engine(self):
        """Get PhysicsRulesEngine instance."""
        return self._physics_engine
    
    @property
    def rdnn(self):
        """Get RDNNMemory instance."""
        return self._rdnn
    
    @property
    def ltm_distiller(self):
        """Get LTMDistiller instance."""
        return self._ltm_distiller
    
    @property
    def template(self):
        """Get ScoringTemplate instance."""
        return self._template


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_pipeline(
    person: Optional[Any] = None,
    mode: PipelineMode = PipelineMode.HEADLESS,
    template: str = "VAT Therapy",
    population_size: int = 50,
    n_generations: int = 100,
) -> EvolutionPipeline:
    """
    Create evolution pipeline with common settings.
    
    Args:
        person: Person instance for body zones
        mode: Pipeline operation mode
        template: Scoring template name
        population_size: Population size
        n_generations: Number of generations
    
    Returns:
        Configured EvolutionPipeline
    """
    config = PipelineConfig(
        mode=mode,
        template_name=template,
        population_size=population_size,
        n_generations=n_generations,
    )
    
    return EvolutionPipeline(config, person).initialize()


def run_quick_optimization(
    person: Optional[Any] = None,
    template: str = "VAT Therapy",
) -> PipelineResult:
    """
    Run quick optimization for testing.
    
    Uses reduced population and generations for fast results.
    
    Args:
        person: Person instance
        template: Scoring template name
    
    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        mode=PipelineMode.HEADLESS,
        template_name=template,
        population_size=20,
        n_generations=30,
        enable_rdnn=False,  # Skip for speed
        enable_ltm=False,
    )
    
    pipeline = EvolutionPipeline(config, person)
    return pipeline.run()


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def get_component_summary() -> Dict[str, Dict]:
    """
    Get summary of available pipeline components.
    
    Returns:
        Dict with component status and capabilities
    """
    summary = {}
    
    # Phase 1: Pokayoke
    try:
        from .pokayoke_observer import PokayokeObserver, AnomalyType
        summary["pokayoke"] = {
            "available": True,
            "anomaly_types": [t.name for t in AnomalyType],
        }
    except ImportError:
        summary["pokayoke"] = {"available": False}
    
    # Phase 2: ExciterGene
    try:
        from .exciter_gene import ExciterGene, GenePhase
        summary["exciter_gene"] = {
            "available": True,
            "phases": [p.name for p in GenePhase],
        }
    except ImportError:
        summary["exciter_gene"] = {"available": False}
    
    # Phase 3: PhysicsRules
    try:
        from .physics_rules import PhysicsRulesEngine, RuleCategory
        summary["physics_rules"] = {
            "available": True,
            "categories": [c.name for c in RuleCategory],
        }
    except ImportError:
        summary["physics_rules"] = {"available": False}
    
    # Phase 4: RDNN
    try:
        from .rdnn_memory import RDNNMemory, RDNNArchitecture
        summary["rdnn"] = {
            "available": True,
            "architectures": [a.name for a in RDNNArchitecture],
        }
    except ImportError:
        summary["rdnn"] = {"available": False}
    
    # Phase 5: LTM
    try:
        from .ltm_distillation import LTMDistiller, DistillationType
        summary["ltm"] = {
            "available": True,
            "distillation_types": [t.name for t in DistillationType],
        }
    except ImportError:
        summary["ltm"] = {"available": False}
    
    # Phase 6: Templates
    try:
        from .scoring_templates import list_available_templates
        templates = list_available_templates()
        summary["templates"] = {
            "available": True,
            "count": len(templates),
            "names": [t["name"] for t in templates],
        }
    except ImportError:
        summary["templates"] = {"available": False}
    
    return summary
