"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   POKAYOKE OBSERVER - Intelligent Monitoring                  ║
║                                                                              ║
║   Monitors evolutionary optimization and PAUSES to ask user when anomalies   ║
║   are detected. Named after Poka-Yoke (ポカヨケ) - mistake-proofing.         ║
║                                                                              ║
║   PHILOSOPHY:                                                                ║
║   "pausare e chiedere a user spiegando che succede, sceglierà user cosa fare"║
║                                                                              ║
║   BEHAVIOR:                                                                  ║
║   1. DETECT anomaly (stagnation, diversity collapse, regression)             ║
║   2. PAUSE optimization                                                      ║
║   3. PRESENT options to user with context                                    ║
║   4. AWAIT user decision                                                     ║
║   5. EXECUTE chosen action                                                   ║
║                                                                              ║
║   ALSO HANDLES:                                                              ║
║   • Gene activation triggers (seed→bloom→freeze transitions)                ║
║   • Physics rule suggestions                                                 ║
║   • Memory/learning checkpoints                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Protocol, List, Dict, Any, Optional, Callable, 
    Tuple, TypeVar, Generic, Union, runtime_checkable
)
from enum import Enum, auto
import logging
import time
from datetime import datetime

from .analysis_config import (
    ObserverConfig, GeneActivationConfig, GenePhase, 
    ActivationTrigger, get_default_config
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STAGNATION = auto()           # No fitness improvement for N generations
    DIVERSITY_COLLAPSE = auto()    # Population becoming too similar
    FITNESS_REGRESSION = auto()    # Fitness getting worse
    CONSTRAINT_VIOLATION = auto()  # Physics constraints violated
    GENE_ACTIVATION_READY = auto() # Time to activate emission genes
    PHYSICS_SUGGESTION = auto()    # Physics rules suggest intervention
    MEMORY_CHECKPOINT = auto()     # Good time to save progress


class UserAction(Enum):
    """Actions user can choose when anomaly detected."""
    RETRY = auto()            # Continue with adjusted parameters
    SKIP = auto()             # Skip this generation, continue
    ABORT = auto()            # Stop optimization, save progress
    ADJUST_MANUAL = auto()    # User will manually adjust parameters
    ACTIVATE_GENES = auto()   # Activate emission genes (seed→bloom)
    FREEZE_POSITION = auto()  # Freeze position genes (bloom→freeze)
    INJECT_DIVERSITY = auto() # Add random individuals
    INCREASE_MUTATION = auto() # Increase mutation rate
    DECREASE_MUTATION = auto() # Decrease mutation rate


@dataclass
class AnomalyContext:
    """
    Context information about a detected anomaly.
    
    Provides all info needed for user to make informed decision.
    """
    anomaly_type: AnomalyType
    generation: int
    severity: float  # 0-1, how serious
    
    # Metrics that triggered detection
    current_fitness: float
    best_fitness_ever: float
    fitness_velocity: float  # Recent improvement rate
    
    # Population health
    population_diversity: float
    stagnation_generations: int
    
    # Suggested actions (ordered by recommendation)
    suggested_actions: List[UserAction]
    
    # Human-readable explanation
    explanation: str
    
    # Additional context
    extra: Dict[str, Any] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserDecision:
    """User's response to an anomaly."""
    action: UserAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # Optional: why user chose this
    
    @classmethod
    def default(cls, action: UserAction) -> 'UserDecision':
        return cls(action=action)


@dataclass
class ObserverState:
    """
    Internal state tracking for the observer.
    """
    # Generation tracking
    last_improvement_gen: int = 0
    best_fitness_ever: float = float('-inf')
    
    # Velocity tracking (for trend detection)
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    
    # Gene activation state
    current_phase: GenePhase = GenePhase.SEED
    phase_started_gen: int = 0
    position_convergence_count: int = 0  # Generations of stable position
    
    # User interaction tracking
    pauses_count: int = 0
    user_decisions: List[UserDecision] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS - User Interaction Interface
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class UserInteractionHandler(Protocol):
    """
    Protocol for handling user interaction.
    
    Implementations can be:
    - CLI (terminal prompt)
    - GUI (dialog box)
    - Headless (auto-select based on config)
    """
    
    def present_anomaly(
        self,
        context: AnomalyContext,
        options: List[UserAction]
    ) -> UserDecision:
        """
        Present anomaly to user and get their decision.
        
        This method should BLOCK until user responds or timeout.
        
        Args:
            context: Full context about the anomaly
            options: Available actions user can choose
            
        Returns:
            User's decision
        """
        ...
    
    def notify_progress(
        self,
        generation: int,
        fitness: float,
        message: str
    ) -> None:
        """Non-blocking progress notification."""
        ...
    
    def notify_phase_transition(
        self,
        old_phase: GenePhase,
        new_phase: GenePhase,
        trigger: ActivationTrigger
    ) -> None:
        """Notify user about gene phase transition."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

class HeadlessHandler:
    """
    Auto-select handler for non-interactive mode.
    
    Uses default actions based on config and anomaly type.
    """
    
    def __init__(self, config: ObserverConfig):
        self.config = config
    
    def present_anomaly(
        self,
        context: AnomalyContext,
        options: List[UserAction]
    ) -> UserDecision:
        """Auto-select based on suggested actions."""
        logger.info(f"[HEADLESS] Anomaly: {context.anomaly_type.name}")
        logger.info(f"  Explanation: {context.explanation}")
        
        if context.suggested_actions:
            action = context.suggested_actions[0]
            logger.info(f"  Auto-selecting: {action.name}")
            return UserDecision(action=action, reason="headless_auto")
        
        # Fallback to config default
        default_map = {
            "retry": UserAction.RETRY,
            "skip": UserAction.SKIP,
            "abort": UserAction.ABORT,
        }
        action = default_map.get(self.config.default_action, UserAction.RETRY)
        return UserDecision(action=action, reason="config_default")
    
    def notify_progress(self, generation: int, fitness: float, message: str):
        if self.config.verbose:
            logger.info(f"[Gen {generation}] Fitness: {fitness:.4f} - {message}")
    
    def notify_phase_transition(
        self,
        old_phase: GenePhase,
        new_phase: GenePhase,
        trigger: ActivationTrigger
    ):
        logger.info(f"[PHASE] {old_phase.name} → {new_phase.name} (trigger: {trigger.name})")


class CLIHandler:
    """
    Command-line interaction handler.
    
    Pauses and prompts user in terminal.
    """
    
    def __init__(self, timeout_seconds: float = 300.0):
        self.timeout = timeout_seconds
    
    def present_anomaly(
        self,
        context: AnomalyContext,
        options: List[UserAction]
    ) -> UserDecision:
        """Present anomaly in terminal and wait for user input."""
        print("\n" + "="*70)
        print(f"⚠️  ANOMALY DETECTED: {context.anomaly_type.name}")
        print("="*70)
        print(f"\n{context.explanation}\n")
        print(f"Generation: {context.generation}")
        print(f"Current Fitness: {context.current_fitness:.4f}")
        print(f"Best Ever: {context.best_fitness_ever:.4f}")
        print(f"Velocity: {context.fitness_velocity:+.6f}")
        print(f"Diversity: {context.population_diversity:.2%}")
        print(f"Stagnation: {context.stagnation_generations} generations")
        print(f"Severity: {context.severity:.1%}")
        
        print("\nAvailable actions:")
        for i, action in enumerate(options, 1):
            marker = " ★" if action in context.suggested_actions[:1] else ""
            print(f"  {i}. {action.name}{marker}")
        
        print(f"\nSuggested: {context.suggested_actions[0].name if context.suggested_actions else 'none'}")
        print("-"*70)
        
        try:
            choice = input(f"Enter choice (1-{len(options)}) or press Enter for suggested: ").strip()
            
            if not choice and context.suggested_actions:
                return UserDecision(action=context.suggested_actions[0], reason="user_accepted_suggestion")
            
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return UserDecision(action=options[idx], reason="user_choice")
            
        except (ValueError, EOFError):
            pass
        
        # Fallback
        if context.suggested_actions:
            return UserDecision(action=context.suggested_actions[0], reason="fallback")
        return UserDecision(action=UserAction.RETRY, reason="fallback")
    
    def notify_progress(self, generation: int, fitness: float, message: str):
        print(f"[Gen {generation:4d}] {fitness:.4f} | {message}")
    
    def notify_phase_transition(
        self,
        old_phase: GenePhase,
        new_phase: GenePhase,
        trigger: ActivationTrigger
    ):
        print(f"\n🌱 PHASE TRANSITION: {old_phase.name} → {new_phase.name}")
        print(f"   Trigger: {trigger.name}\n")


# ══════════════════════════════════════════════════════════════════════════════
# POKAYOKE OBSERVER - Main Class
# ══════════════════════════════════════════════════════════════════════════════

class PokayokeObserver:
    """
    Intelligent evolution monitor that pauses for user decisions.
    
    "Poka-Yoke" (ポカヨケ) = mistake-proofing / error prevention
    
    USAGE:
        observer = PokayokeObserver(config, handler=CLIHandler())
        
        # In evolution loop:
        for gen in range(max_generations):
            # ... evolution step ...
            
            decision = observer.on_generation(
                generation=gen,
                population_fitness=fitnesses,
                best_genome=best,
                population_diversity=diversity
            )
            
            if decision:
                # User made a decision, handle it
                if decision.action == UserAction.ABORT:
                    break
                elif decision.action == UserAction.ACTIVATE_GENES:
                    activate_emission_genes()
                # ... etc
    """
    
    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
        gene_config: Optional[GeneActivationConfig] = None,
        handler: Optional[UserInteractionHandler] = None,
    ):
        """
        Initialize observer.
        
        Args:
            config: Observer configuration (thresholds, timeouts)
            gene_config: Gene activation configuration
            handler: User interaction handler (CLI, GUI, Headless)
        """
        default = get_default_config()
        self.config = config or default.observer
        self.gene_config = gene_config or default.gene_activation
        
        # Select handler
        if handler:
            self.handler = handler
        elif self.config.auto_adjust:
            self.handler = HeadlessHandler(self.config)
        else:
            self.handler = CLIHandler(self.config.timeout_seconds)
        
        # Internal state
        self.state = ObserverState(current_phase=self.gene_config.initial_phase)
        
        logger.info(f"PokayokeObserver initialized (mode: {'headless' if self.config.auto_adjust else 'interactive'})")
    
    def on_generation(
        self,
        generation: int,
        population_fitness: List[float],
        best_genome: Any,
        population_diversity: float,
        position_sigma: float = 1.0,  # For gene activation check
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[UserDecision]:
        """
        Called after each generation. Checks for anomalies.
        
        Args:
            generation: Current generation number
            population_fitness: Fitness scores of all individuals
            best_genome: Best genome this generation
            population_diversity: Population diversity metric [0-1]
            position_sigma: Standard deviation of exciter positions
            extra_context: Additional context for decision making
            
        Returns:
            UserDecision if user intervention requested, None otherwise
        """
        best_fitness = max(population_fitness)
        mean_fitness = sum(population_fitness) / len(population_fitness)
        
        # Update history
        self.state.fitness_history.append(best_fitness)
        self.state.diversity_history.append(population_diversity)
        
        # Track improvements
        if best_fitness > self.state.best_fitness_ever:
            self.state.best_fitness_ever = best_fitness
            self.state.last_improvement_gen = generation
        
        # Calculate velocity (recent trend)
        velocity = self._calculate_velocity()
        stagnation_gens = generation - self.state.last_improvement_gen
        
        # === ANOMALY DETECTION ===
        anomaly = self._detect_anomaly(
            generation=generation,
            best_fitness=best_fitness,
            velocity=velocity,
            diversity=population_diversity,
            stagnation_gens=stagnation_gens,
            position_sigma=position_sigma,
        )
        
        if anomaly:
            # PAUSE and ask user
            decision = self._handle_anomaly(anomaly)
            self.state.pauses_count += 1
            self.state.user_decisions.append(decision)
            return decision
        
        # Progress notification (non-blocking)
        if self.config.verbose and generation % 10 == 0:
            self.handler.notify_progress(
                generation, best_fitness,
                f"div={population_diversity:.2f} vel={velocity:+.4f}"
            )
        
        return None
    
    def _calculate_velocity(self, window: int = 5) -> float:
        """Calculate recent fitness improvement velocity."""
        if len(self.state.fitness_history) < 2:
            return 0.0
        
        recent = self.state.fitness_history[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _detect_anomaly(
        self,
        generation: int,
        best_fitness: float,
        velocity: float,
        diversity: float,
        stagnation_gens: int,
        position_sigma: float,
    ) -> Optional[AnomalyContext]:
        """
        Detect anomalies that require user attention.
        
        Returns AnomalyContext if anomaly found, None otherwise.
        """
        # === STAGNATION ===
        if stagnation_gens >= self.config.stagnation_threshold:
            return AnomalyContext(
                anomaly_type=AnomalyType.STAGNATION,
                generation=generation,
                severity=min(1.0, stagnation_gens / (self.config.stagnation_threshold * 2)),
                current_fitness=best_fitness,
                best_fitness_ever=self.state.best_fitness_ever,
                fitness_velocity=velocity,
                population_diversity=diversity,
                stagnation_generations=stagnation_gens,
                suggested_actions=[
                    UserAction.INJECT_DIVERSITY,
                    UserAction.INCREASE_MUTATION,
                    UserAction.ACTIVATE_GENES,
                    UserAction.RETRY,
                ],
                explanation=(
                    f"Optimization has stagnated for {stagnation_gens} generations.\n"
                    f"No fitness improvement detected. Population may be stuck in local optimum.\n"
                    f"Consider injecting diversity or increasing mutation rate."
                ),
            )
        
        # === DIVERSITY COLLAPSE ===
        if diversity < self.config.diversity_collapse_threshold:
            return AnomalyContext(
                anomaly_type=AnomalyType.DIVERSITY_COLLAPSE,
                generation=generation,
                severity=1.0 - (diversity / self.config.diversity_collapse_threshold),
                current_fitness=best_fitness,
                best_fitness_ever=self.state.best_fitness_ever,
                fitness_velocity=velocity,
                population_diversity=diversity,
                stagnation_generations=stagnation_gens,
                suggested_actions=[
                    UserAction.INJECT_DIVERSITY,
                    UserAction.INCREASE_MUTATION,
                    UserAction.RETRY,
                ],
                explanation=(
                    f"Population diversity has collapsed to {diversity:.1%}.\n"
                    f"All individuals are becoming too similar, limiting exploration.\n"
                    f"Strongly recommend injecting random individuals."
                ),
            )
        
        # === FITNESS REGRESSION ===
        if velocity < self.config.regression_threshold:
            return AnomalyContext(
                anomaly_type=AnomalyType.FITNESS_REGRESSION,
                generation=generation,
                severity=min(1.0, abs(velocity) / abs(self.config.regression_threshold)),
                current_fitness=best_fitness,
                best_fitness_ever=self.state.best_fitness_ever,
                fitness_velocity=velocity,
                population_diversity=diversity,
                stagnation_generations=stagnation_gens,
                suggested_actions=[
                    UserAction.DECREASE_MUTATION,
                    UserAction.RETRY,
                    UserAction.SKIP,
                ],
                explanation=(
                    f"Fitness is regressing at rate {velocity:+.4f} per generation.\n"
                    f"This may indicate excessive mutation or selection pressure.\n"
                    f"Consider decreasing mutation rate."
                ),
            )
        
        # === GENE ACTIVATION READY ===
        if self.state.current_phase == GenePhase.SEED:
            # Track position convergence
            if position_sigma < self.gene_config.position_convergence_sigma:
                self.state.position_convergence_count += 1
            else:
                self.state.position_convergence_count = 0
            
            should_activate, trigger = self.gene_config.should_activate_emission(
                position_sigma=position_sigma,
                stable_generations=self.state.position_convergence_count,
                fitness_velocity=velocity,
                plateau_generations=stagnation_gens,
                current_generation=generation,
            )
            
            if should_activate:
                return AnomalyContext(
                    anomaly_type=AnomalyType.GENE_ACTIVATION_READY,
                    generation=generation,
                    severity=0.3,  # Low severity - positive event
                    current_fitness=best_fitness,
                    best_fitness_ever=self.state.best_fitness_ever,
                    fitness_velocity=velocity,
                    population_diversity=diversity,
                    stagnation_generations=stagnation_gens,
                    suggested_actions=[
                        UserAction.ACTIVATE_GENES,
                        UserAction.SKIP,
                    ],
                    explanation=(
                        f"Exciter POSITIONS have converged (σ={position_sigma:.3f}).\n"
                        f"Trigger: {trigger.name}\n"
                        f"Ready to activate EMISSION genes (phase, delay, gain, polarity).\n"
                        f"This transitions from SEED → BLOOM phase."
                    ),
                    extra={"trigger": trigger, "position_sigma": position_sigma},
                )
        
        return None
    
    def _handle_anomaly(self, context: AnomalyContext) -> UserDecision:
        """Handle detected anomaly by asking user."""
        logger.info(f"Anomaly detected: {context.anomaly_type.name} at gen {context.generation}")
        
        # Get available options
        options = list(UserAction)  # All options available
        
        # Get user decision
        decision = self.handler.present_anomaly(context, options)
        
        # Handle phase transitions
        if decision.action == UserAction.ACTIVATE_GENES:
            old_phase = self.state.current_phase
            self.state.current_phase = GenePhase.BLOOM
            self.state.phase_started_gen = context.generation
            self.handler.notify_phase_transition(
                old_phase, GenePhase.BLOOM,
                context.extra.get("trigger", ActivationTrigger.USER_REQUEST)
            )
        
        elif decision.action == UserAction.FREEZE_POSITION:
            old_phase = self.state.current_phase
            self.state.current_phase = GenePhase.FREEZE
            self.state.phase_started_gen = context.generation
            self.handler.notify_phase_transition(
                old_phase, GenePhase.FREEZE,
                ActivationTrigger.USER_REQUEST
            )
        
        return decision
    
    # === QUERY METHODS ===
    
    def get_current_phase(self) -> GenePhase:
        """Get current gene activation phase."""
        return self.state.current_phase
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get observer statistics."""
        return {
            "pauses_count": self.state.pauses_count,
            "current_phase": self.state.current_phase.name,
            "best_fitness_ever": self.state.best_fitness_ever,
            "decisions": [(d.action.name, d.reason) for d in self.state.user_decisions],
        }
    
    def reset(self):
        """Reset observer state for new optimization run."""
        self.state = ObserverState(current_phase=self.gene_config.initial_phase)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def create_observer(
    mode: str = "auto",  # "auto", "cli", "headless", "gui"
    config: Optional[ObserverConfig] = None,
    gene_config: Optional[GeneActivationConfig] = None,
) -> PokayokeObserver:
    """
    Create an observer with specified mode.
    
    Args:
        mode: Interaction mode
            - "auto": CLI if terminal, headless otherwise
            - "cli": Always use CLI prompts
            - "headless": Never prompt, use defaults
            - "gui": For GUI integration (requires custom handler)
        config: Observer configuration
        gene_config: Gene activation configuration
        
    Returns:
        Configured PokayokeObserver
    """
    import sys
    
    if mode == "auto":
        # Detect if we have a terminal
        if sys.stdin.isatty():
            mode = "cli"
        else:
            mode = "headless"
    
    default = get_default_config()
    obs_config = config or default.observer
    
    if mode == "headless":
        # Override auto_adjust to True for headless
        obs_config = ObserverConfig(
            stagnation_threshold=obs_config.stagnation_threshold,
            diversity_collapse_threshold=obs_config.diversity_collapse_threshold,
            regression_threshold=obs_config.regression_threshold,
            auto_adjust=True,  # Force headless
            timeout_seconds=obs_config.timeout_seconds,
            default_action=obs_config.default_action,
            verbose=obs_config.verbose,
        )
        handler = HeadlessHandler(obs_config)
    elif mode == "cli":
        handler = CLIHandler(obs_config.timeout_seconds)
    else:
        # For GUI or other, caller should provide handler
        handler = None
    
    return PokayokeObserver(
        config=obs_config,
        gene_config=gene_config,
        handler=handler
    )
