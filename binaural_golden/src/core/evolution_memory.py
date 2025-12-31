"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               EVOLUTION MEMORY - Short & Long Term Learning System           ║
║                                                                              ║
║   Neural-network inspired memory for evolutionary optimization:              ║
║   • Short-Term Memory: Ring buffer (3-5-10 generations) for trajectory       ║
║   • Long-Term Memory: Distilled patterns, lesson learned, experience archive ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Schaul et al. 2015: Prioritized Experience Replay (DQN)                  ║
║   • Andrychowicz et al. 2018: Hindsight Experience Replay (goal relabeling)  ║
║   • Population-Based Training (PBT) concepts                                 ║
║   • Czinger 21C / Divergent: Topology-aware generative design                ║
║                                                                              ║
║   KEY CONCEPTS:                                                              ║
║   1. SHORT-TERM: Recent generations (3-5-10) for gradient/trajectory         ║
║   2. LONG-TERM: Successful patterns extracted and distilled                  ║
║   3. PRIORITY: TD-error analog = fitness improvement velocity                ║
║   4. HINDSIGHT: Learn from failures by relabeling goals                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Deque
from collections import deque
from datetime import datetime
import json
import hashlib
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY - Ring Buffer per Trajectory Analysis
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationSnapshot:
    """
    Snapshot of a single generation for short-term memory.
    
    Stores essential info for trajectory analysis without full genome data.
    """
    generation: int
    timestamp: float
    
    # Fitness statistics
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    
    # Multi-objective scores (ObjectiveVector summary)
    objective_means: Dict[str, float]  # {spine_flatness: 0.85, ear_lr_uniformity: 0.72, ...}
    objective_bests: Dict[str, float]
    
    # Diversity metrics
    population_diversity: float  # Genetic diversity [0-1]
    phenotype_diversity: float   # Fitness landscape diversity
    
    # Operator effectiveness (what worked this generation?)
    mutation_improvements: int   # How many mutations improved fitness
    crossover_improvements: int  # How many crossovers improved fitness
    total_evaluations: int
    
    # Best genome fingerprint (hash for quick comparison)
    best_genome_hash: str = ""
    
    # Physics decisions made
    physics_decisions: List[str] = field(default_factory=list)
    
    def fitness_velocity(self, previous: 'GenerationSnapshot') -> float:
        """Calculate fitness improvement velocity (TD-error analog)."""
        if previous is None:
            return 0.0
        return self.best_fitness - previous.best_fitness


@dataclass  
class TrajectoryAnalysis:
    """
    Analysis results from short-term memory trajectory.
    
    Used to guide search direction and adapt parameters.
    """
    # Trend detection
    is_improving: bool = False           # Fitness trending upward?
    is_stagnating: bool = False          # No improvement for N generations?
    is_diverging: bool = False           # Diversity collapsing?
    
    # Velocity analysis
    mean_velocity: float = 0.0           # Average fitness improvement per gen
    velocity_trend: str = "stable"       # "accelerating", "decelerating", "stable"
    
    # Objective-specific trends
    objective_velocities: Dict[str, float] = field(default_factory=dict)
    fastest_improving: str = ""          # Which objective improving fastest
    slowest_improving: str = ""          # Which needs attention
    
    # Operator effectiveness
    mutation_success_rate: float = 0.0   # % of mutations that improved
    crossover_success_rate: float = 0.0  # % of crossovers that improved
    recommended_mutation_rate: float = 0.2  # Suggested adaptation
    
    # Prediction
    predicted_generations_to_goal: int = -1  # Estimated generations to target
    confidence: float = 0.5
    
    # Recommended actions
    actions: List[str] = field(default_factory=list)


class ShortTermMemory:
    """
    Ring buffer storing recent generations for trajectory analysis.
    
    Inspired by Experience Replay in DRL but adapted for evolutionary optimization.
    
    USAGE:
        stm = ShortTermMemory(buffer_sizes=[3, 5, 10])
        
        # After each generation
        stm.record(generation_snapshot)
        
        # Analyze trajectory
        analysis = stm.analyze_trajectory(window=5)
        
        # Get recommended parameter adaptations
        if analysis.is_stagnating:
            new_mutation_rate = analysis.recommended_mutation_rate
    """
    
    def __init__(
        self,
        buffer_sizes: List[int] = [3, 5, 10],
        store_genomes: bool = False,
    ):
        """
        Initialize short-term memory with multiple time horizons.
        
        Args:
            buffer_sizes: List of window sizes for multi-scale analysis
                         [3, 5, 10] = analyze last 3, 5, and 10 generations
            store_genomes: Whether to store full genome data (memory intensive)
        """
        self.buffer_sizes = sorted(buffer_sizes)
        self.max_size = max(buffer_sizes)
        self.store_genomes = store_genomes
        
        # Main ring buffer (deque with maxlen auto-evicts old entries)
        self.buffer: Deque[GenerationSnapshot] = deque(maxlen=self.max_size)
        
        # Optional genome storage
        self.genome_archive: Dict[str, Any] = {}  # hash -> genome
        
        # Running statistics for normalization
        self._fitness_sum = 0.0
        self._fitness_sq_sum = 0.0
        self._count = 0
        
        logger.info(f"ShortTermMemory initialized with windows {buffer_sizes}")
    
    def record(
        self,
        generation: int,
        population_fitnesses: np.ndarray,
        objective_vectors: List[Dict[str, float]],
        best_genome: Any = None,
        mutation_improvements: int = 0,
        crossover_improvements: int = 0,
        physics_decisions: List[str] = None,
    ) -> GenerationSnapshot:
        """
        Record a generation snapshot to short-term memory.
        
        Args:
            generation: Generation number
            population_fitnesses: Array of fitness scores
            objective_vectors: List of objective dicts for each individual
            best_genome: Best genome (stored if store_genomes=True)
            mutation_improvements: Count of successful mutations
            crossover_improvements: Count of successful crossovers
            physics_decisions: List of physics decisions made
        
        Returns:
            The recorded GenerationSnapshot
        """
        fitnesses = np.asarray(population_fitnesses)
        
        # Calculate objective statistics
        obj_names = list(objective_vectors[0].keys()) if objective_vectors else []
        obj_means = {}
        obj_bests = {}
        
        for name in obj_names:
            values = [ov.get(name, 0.0) for ov in objective_vectors]
            obj_means[name] = np.mean(values)
            obj_bests[name] = np.max(values)
        
        # Diversity estimation (coefficient of variation)
        pop_diversity = np.std(fitnesses) / (np.mean(fitnesses) + 1e-8)
        
        # Phenotype diversity (unique fitness values)
        unique_fitnesses = len(np.unique(np.round(fitnesses, 4)))
        pheno_diversity = unique_fitnesses / len(fitnesses)
        
        # Genome hash
        genome_hash = ""
        if best_genome is not None:
            genome_hash = self._hash_genome(best_genome)
            if self.store_genomes:
                self.genome_archive[genome_hash] = best_genome
        
        snapshot = GenerationSnapshot(
            generation=generation,
            timestamp=datetime.now().timestamp(),
            best_fitness=float(np.max(fitnesses)),
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            worst_fitness=float(np.min(fitnesses)),
            objective_means=obj_means,
            objective_bests=obj_bests,
            population_diversity=float(pop_diversity),
            phenotype_diversity=float(pheno_diversity),
            mutation_improvements=mutation_improvements,
            crossover_improvements=crossover_improvements,
            total_evaluations=len(fitnesses),
            best_genome_hash=genome_hash,
            physics_decisions=physics_decisions or [],
        )
        
        self.buffer.append(snapshot)
        
        # Update running stats
        self._fitness_sum += snapshot.best_fitness
        self._fitness_sq_sum += snapshot.best_fitness ** 2
        self._count += 1
        
        return snapshot
    
    def analyze_trajectory(self, window: int = None) -> TrajectoryAnalysis:
        """
        Analyze fitness trajectory over recent generations.
        
        This is the KEY method for adaptive parameter control.
        
        Args:
            window: Number of recent generations to analyze.
                   If None, uses the medium buffer size.
        
        Returns:
            TrajectoryAnalysis with trend detection and recommendations
        """
        if window is None:
            window = self.buffer_sizes[len(self.buffer_sizes) // 2]
        
        window = min(window, len(self.buffer))
        if window < 2:
            return TrajectoryAnalysis(actions=["Need more generations for analysis"])
        
        # Get recent snapshots
        recent = list(self.buffer)[-window:]
        
        # === VELOCITY ANALYSIS ===
        velocities = []
        for i in range(1, len(recent)):
            v = recent[i].fitness_velocity(recent[i-1])
            velocities.append(v)
        
        mean_velocity = np.mean(velocities)
        
        # Velocity trend (linear regression slope)
        if len(velocities) >= 3:
            x = np.arange(len(velocities))
            slope = np.polyfit(x, velocities, 1)[0]
            if slope > 0.001:
                velocity_trend = "accelerating"
            elif slope < -0.001:
                velocity_trend = "decelerating"
            else:
                velocity_trend = "stable"
        else:
            velocity_trend = "stable"
        
        # === TREND DETECTION ===
        is_improving = mean_velocity > 0.001
        is_stagnating = abs(mean_velocity) < 0.0001 and len(velocities) >= 5
        
        # Diversity collapse detection
        recent_diversity = [s.population_diversity for s in recent]
        diversity_trend = recent_diversity[-1] - recent_diversity[0]
        is_diverging = diversity_trend < -0.1  # Diversity dropping fast
        
        # === OBJECTIVE-SPECIFIC ANALYSIS ===
        obj_velocities = {}
        obj_names = list(recent[0].objective_means.keys())
        
        for name in obj_names:
            obj_values = [s.objective_means.get(name, 0) for s in recent]
            if len(obj_values) >= 2:
                obj_velocities[name] = (obj_values[-1] - obj_values[0]) / len(obj_values)
        
        fastest = max(obj_velocities, key=obj_velocities.get) if obj_velocities else ""
        slowest = min(obj_velocities, key=obj_velocities.get) if obj_velocities else ""
        
        # === OPERATOR EFFECTIVENESS ===
        total_mutations = sum(s.mutation_improvements for s in recent)
        total_crossovers = sum(s.crossover_improvements for s in recent)
        total_evals = sum(s.total_evaluations for s in recent)
        
        mutation_rate = total_mutations / (total_evals + 1)
        crossover_rate = total_crossovers / (total_evals + 1)
        
        # === RECOMMENDATIONS ===
        actions = []
        recommended_mutation = 0.2  # Default
        
        if is_stagnating:
            actions.append("STAGNATION: Increase mutation rate")
            actions.append("Consider diversity injection")
            recommended_mutation = 0.35
        
        if is_diverging:
            actions.append("DIVERSITY COLLAPSE: Inject random individuals")
            recommended_mutation = 0.4
        
        if velocity_trend == "decelerating":
            actions.append("DECELERATION: May be converging, reduce mutation")
            recommended_mutation = 0.15
        
        if slowest and obj_velocities.get(slowest, 0) < 0:
            actions.append(f"REGRESSION: {slowest} getting worse, focus on it")
        
        # === PREDICTION ===
        # Simple linear extrapolation to goal (fitness=1.0)
        if mean_velocity > 0:
            current_best = recent[-1].best_fitness
            remaining = 1.0 - current_best
            predicted_gens = int(remaining / mean_velocity)
            confidence = 0.3 + 0.4 * (1 - abs(velocity_trend == "stable"))
        else:
            predicted_gens = -1
            confidence = 0.2
        
        return TrajectoryAnalysis(
            is_improving=is_improving,
            is_stagnating=is_stagnating,
            is_diverging=is_diverging,
            mean_velocity=mean_velocity,
            velocity_trend=velocity_trend,
            objective_velocities=obj_velocities,
            fastest_improving=fastest,
            slowest_improving=slowest,
            mutation_success_rate=mutation_rate,
            crossover_success_rate=crossover_rate,
            recommended_mutation_rate=recommended_mutation,
            predicted_generations_to_goal=predicted_gens,
            confidence=confidence,
            actions=actions,
        )
    
    def get_multi_scale_analysis(self) -> Dict[int, TrajectoryAnalysis]:
        """
        Analyze trajectory at multiple time scales.
        
        Returns dict of {window_size: TrajectoryAnalysis}.
        Useful for detecting patterns at different scales:
        - Short (3): Recent micro-trends
        - Medium (5): Local search behavior
        - Long (10): Overall convergence
        """
        analyses = {}
        for window in self.buffer_sizes:
            if len(self.buffer) >= window:
                analyses[window] = self.analyze_trajectory(window)
        return analyses
    
    def get_priority_weights(self) -> np.ndarray:
        """
        Compute prioritized experience replay weights.
        
        Based on Schaul et al. 2015 - TD-error proportional priority.
        Here we use fitness velocity as TD-error analog.
        
        Returns:
            Array of priority weights for each generation in buffer
        """
        if len(self.buffer) < 2:
            return np.ones(len(self.buffer))
        
        velocities = []
        snapshots = list(self.buffer)
        
        for i in range(len(snapshots)):
            if i == 0:
                v = 0.0
            else:
                v = snapshots[i].fitness_velocity(snapshots[i-1])
            velocities.append(abs(v) + 0.01)  # Small epsilon for stability
        
        priorities = np.array(velocities)
        priorities = priorities / priorities.sum()  # Normalize
        
        return priorities
    
    def _hash_genome(self, genome: Any) -> str:
        """Create a hash fingerprint of a genome for quick comparison."""
        try:
            # Try to use genome's own method if available
            if hasattr(genome, 'fingerprint'):
                return genome.fingerprint()
            
            # Otherwise serialize and hash
            data = pickle.dumps(genome)
            return hashlib.md5(data).hexdigest()[:16]
        except Exception:
            return str(id(genome))[:16]
    
    def clear(self):
        """Clear short-term memory."""
        self.buffer.clear()
        self.genome_archive.clear()
        self._fitness_sum = 0.0
        self._fitness_sq_sum = 0.0
        self._count = 0


# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY - Distilled Patterns & Lesson Learned
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedPattern:
    """
    A distilled pattern learned from successful optimizations.
    
    These are like "memories" in a neural network - general rules
    extracted from specific experiences.
    """
    pattern_id: str
    pattern_type: str  # "cutout", "exciter", "contour", "thickness", "general"
    
    # The pattern itself (semantic representation)
    description: str
    conditions: Dict[str, Any]  # When to apply (person height, target_freq, etc.)
    action: Dict[str, Any]      # What to do (place cutout here, use this shape, etc.)
    
    # Confidence and usage stats
    confidence: float = 0.5     # [0-1] how reliable is this pattern
    success_count: int = 0      # Times applied successfully
    failure_count: int = 0      # Times applied unsuccessfully
    
    # Provenance
    source_runs: List[str] = field(default_factory=list)  # Run IDs that contributed
    created_at: str = ""
    last_used: str = ""
    
    def update_success(self, success: bool):
        """Update pattern confidence based on outcome."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        total = self.success_count + self.failure_count
        self.confidence = (self.success_count + 1) / (total + 2)  # Laplace smoothing
        self.last_used = datetime.now().isoformat()
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


@dataclass
class ExperienceArchiveEntry:
    """
    A complete optimization run stored for future reference.
    
    Like hindsight experience replay - even failed runs teach us something.
    """
    run_id: str
    timestamp: str
    
    # Problem specification
    target_person: Dict[str, Any]  # Person parameters (height, weight, etc.)
    target_objectives: Dict[str, float]  # What we were trying to achieve
    zone_weights: Dict[str, float]  # spine/head weights
    
    # Solution found
    final_fitness: float
    final_objectives: Dict[str, float]
    best_genome_summary: Dict[str, Any]  # Key genome features (not full genome)
    
    # Journey statistics
    total_generations: int
    convergence_generation: int  # When did we effectively stop improving
    final_diversity: float
    
    # What worked / what didn't
    successful_mutations: List[str]  # Types of mutations that improved fitness
    failed_mutations: List[str]      # Types that consistently failed
    
    # Outcome classification
    outcome: str = "unknown"  # "success", "partial", "failure", "timeout"
    
    # Hindsight relabeling (HER-style)
    # If we failed goal A but achieved goal B, store what we DID achieve
    achieved_goals: Dict[str, float] = field(default_factory=dict)


class LongTermMemory:
    """
    Persistent memory for cross-run learning.
    
    Stores:
    1. PATTERNS: Distilled rules like "for tall persons, place exciter higher"
    2. ARCHIVE: Complete run histories for hindsight learning
    3. STATISTICS: What works across many runs
    
    PERSISTENCE: Saves to JSON for cross-session learning.
    
    USAGE:
        ltm = LongTermMemory("~/.golden_studio/memory")
        
        # After a successful run
        ltm.archive_run(run_data)
        ltm.distill_patterns()  # Extract new patterns
        
        # Before a new run
        suggestions = ltm.get_suggestions(target_person, objectives)
        prior_knowledge = ltm.get_prior_knowledge(problem_spec)
    """
    
    def __init__(
        self,
        storage_path: str = None,
        max_archive_size: int = 1000,
        min_pattern_confidence: float = 0.6,
    ):
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Directory for persistence. None = memory only.
            max_archive_size: Maximum runs to keep in archive
            min_pattern_confidence: Minimum confidence to use a pattern
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_archive_size = max_archive_size
        self.min_pattern_confidence = min_pattern_confidence
        
        # Memory storage
        self.patterns: Dict[str, LearnedPattern] = {}
        self.archive: List[ExperienceArchiveEntry] = []
        
        # Global statistics
        self.global_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'avg_convergence_gen': 0.0,
            'best_fitness_ever': 0.0,
            'contour_success_rates': {},
            'cutout_shape_effectiveness': {},
            'exciter_position_heatmap': np.zeros((10, 10)),  # 10x10 grid
        }
        
        # Load from disk if path exists
        if self.storage_path:
            self._load()
        
        logger.info(f"LongTermMemory initialized with {len(self.patterns)} patterns, "
                   f"{len(self.archive)} archived runs")
    
    def archive_run(
        self,
        run_id: str,
        target_person: Dict[str, Any],
        target_objectives: Dict[str, float],
        zone_weights: Dict[str, float],
        final_fitness: float,
        final_objectives: Dict[str, float],
        best_genome_summary: Dict[str, Any],
        total_generations: int,
        convergence_generation: int,
        final_diversity: float,
        successful_mutations: List[str],
        failed_mutations: List[str],
        outcome: str = "unknown",
    ):
        """
        Archive a completed optimization run.
        
        This is called at the END of an optimization to store what happened.
        Even failed runs are valuable for hindsight learning!
        """
        # Determine achieved goals (HER-style relabeling)
        achieved_goals = {}
        for obj_name, achieved in final_objectives.items():
            target = target_objectives.get(obj_name, 1.0)
            if achieved >= target * 0.9:  # Within 10% of target
                achieved_goals[obj_name] = achieved
        
        entry = ExperienceArchiveEntry(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            target_person=target_person,
            target_objectives=target_objectives,
            zone_weights=zone_weights,
            final_fitness=final_fitness,
            final_objectives=final_objectives,
            best_genome_summary=best_genome_summary,
            total_generations=total_generations,
            convergence_generation=convergence_generation,
            final_diversity=final_diversity,
            successful_mutations=successful_mutations,
            failed_mutations=failed_mutations,
            outcome=outcome,
            achieved_goals=achieved_goals,
        )
        
        self.archive.append(entry)
        
        # Trim if over max size (keep most recent + best performing)
        if len(self.archive) > self.max_archive_size:
            self._trim_archive()
        
        # Update global stats
        self._update_global_stats(entry)
        
        # Persist
        if self.storage_path:
            self._save()
        
        logger.info(f"Archived run {run_id}: {outcome}, fitness={final_fitness:.4f}")
    
    def distill_patterns(self, min_occurrences: int = 3):
        """
        Extract generalizable patterns from archive.
        
        This is the "learning" step - looking across many runs to find
        what consistently works.
        
        Args:
            min_occurrences: Minimum times a pattern must appear to be considered
        """
        if len(self.archive) < min_occurrences:
            logger.info(f"Not enough runs ({len(self.archive)}) to distill patterns")
            return
        
        successful_runs = [e for e in self.archive if e.outcome == "success"]
        
        if len(successful_runs) < min_occurrences:
            logger.info(f"Not enough successful runs ({len(successful_runs)})")
            return
        
        # Pattern type 1: Person height -> exciter Y position
        self._distill_height_exciter_pattern(successful_runs, min_occurrences)
        
        # Pattern type 2: Zone weights -> cutout placement
        self._distill_zone_cutout_pattern(successful_runs, min_occurrences)
        
        # Pattern type 3: Contour type effectiveness
        self._distill_contour_patterns(successful_runs, min_occurrences)
        
        # Pattern type 4: Mutation type effectiveness
        self._distill_mutation_patterns(successful_runs, min_occurrences)
        
        logger.info(f"Distilled {len(self.patterns)} patterns from {len(successful_runs)} successful runs")
        
        if self.storage_path:
            self._save()
    
    def get_suggestions(
        self,
        target_person: Dict[str, Any],
        target_objectives: Dict[str, float],
        zone_weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Get suggestions based on learned patterns.
        
        Returns a list of suggestions with confidence scores.
        """
        suggestions = []
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.confidence < self.min_pattern_confidence:
                continue
            
            # Check if pattern conditions match
            if self._conditions_match(pattern.conditions, target_person, target_objectives, zone_weights):
                suggestions.append({
                    'pattern_id': pattern_id,
                    'type': pattern.pattern_type,
                    'action': pattern.action,
                    'description': pattern.description,
                    'confidence': pattern.confidence,
                    'success_rate': pattern.success_rate,
                })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: -x['confidence'])
        
        return suggestions
    
    def get_similar_runs(
        self,
        target_person: Dict[str, Any],
        target_objectives: Dict[str, float],
        top_k: int = 5,
    ) -> List[ExperienceArchiveEntry]:
        """
        Find archived runs similar to current problem.
        
        Useful for warm-starting or learning from past experience.
        """
        # Simple similarity based on person height and objective priorities
        similarities = []
        
        for entry in self.archive:
            sim = self._compute_similarity(entry, target_person, target_objectives)
            similarities.append((sim, entry))
        
        similarities.sort(key=lambda x: -x[0])
        
        return [entry for _, entry in similarities[:top_k]]
    
    def update_pattern_outcome(self, pattern_id: str, success: bool):
        """Update a pattern's success/failure count after using it."""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].update_success(success)
            if self.storage_path:
                self._save()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _distill_height_exciter_pattern(self, runs: List[ExperienceArchiveEntry], min_occ: int):
        """Extract pattern: person height -> optimal exciter Y position."""
        # Group by height range
        height_groups = {}  # height_bin -> [(exciter_y, fitness), ...]
        
        for run in runs:
            height = run.target_person.get('height_m', 1.75)
            exciter_y = run.best_genome_summary.get('exciter_y_mean', 0.5)
            
            height_bin = round(height * 10) / 10  # Round to 0.1m
            
            if height_bin not in height_groups:
                height_groups[height_bin] = []
            height_groups[height_bin].append((exciter_y, run.final_fitness))
        
        for height_bin, data in height_groups.items():
            if len(data) < min_occ:
                continue
            
            # Weighted average exciter position by fitness
            total_weight = sum(f for _, f in data)
            avg_y = sum(y * f for y, f in data) / (total_weight + 1e-8)
            
            pattern_id = f"height_{height_bin:.1f}_exciter"
            
            self.patterns[pattern_id] = LearnedPattern(
                pattern_id=pattern_id,
                pattern_type="exciter",
                description=f"For person ~{height_bin:.1f}m tall, place exciters around Y={avg_y:.2f}",
                conditions={'height_m_range': (height_bin - 0.05, height_bin + 0.05)},
                action={'recommended_exciter_y': avg_y},
                confidence=min(0.5 + len(data) * 0.1, 0.9),
                source_runs=[r.run_id for r, _ in zip(runs, data)],
                created_at=datetime.now().isoformat(),
            )
    
    def _distill_zone_cutout_pattern(self, runs: List[ExperienceArchiveEntry], min_occ: int):
        """Extract pattern: zone weights -> cutout positions."""
        # Group by spine/head weight ratio
        zone_groups = {}  # ratio_bin -> [cutout_positions...]
        
        for run in runs:
            spine_w = run.zone_weights.get('spine', 0.7)
            ratio_bin = round(spine_w * 10) / 10
            
            cutouts = run.best_genome_summary.get('cutout_positions', [])
            
            if ratio_bin not in zone_groups:
                zone_groups[ratio_bin] = []
            zone_groups[ratio_bin].extend(cutouts)
        
        for ratio_bin, positions in zone_groups.items():
            if len(positions) < min_occ * 2:  # Need more cutout samples
                continue
            
            # Find common cutout regions
            positions = np.array(positions)
            if len(positions) == 0:
                continue
                
            # Cluster positions (simple histogram)
            x_hist, x_edges = np.histogram(positions[:, 0] if len(positions.shape) > 1 else positions, bins=5)
            common_x = x_edges[np.argmax(x_hist)]
            
            pattern_id = f"zone_{ratio_bin:.1f}_cutout"
            
            self.patterns[pattern_id] = LearnedPattern(
                pattern_id=pattern_id,
                pattern_type="cutout",
                description=f"For spine_weight={ratio_bin:.1f}, cutouts often at X~{common_x:.2f}",
                conditions={'spine_weight_range': (ratio_bin - 0.05, ratio_bin + 0.05)},
                action={'recommended_cutout_x': common_x},
                confidence=0.4 + min(len(positions), 20) * 0.025,
                source_runs=[],  # Too many to track
                created_at=datetime.now().isoformat(),
            )
    
    def _distill_contour_patterns(self, runs: List[ExperienceArchiveEntry], min_occ: int):
        """Track which contour types work best."""
        contour_fitness = {}  # contour_type -> [fitnesses]
        
        for run in runs:
            contour = run.best_genome_summary.get('contour_type', 'rectangle')
            if contour not in contour_fitness:
                contour_fitness[contour] = []
            contour_fitness[contour].append(run.final_fitness)
        
        # Update global stats
        for contour, fitnesses in contour_fitness.items():
            if len(fitnesses) >= min_occ:
                self.global_stats['contour_success_rates'][contour] = {
                    'mean_fitness': np.mean(fitnesses),
                    'count': len(fitnesses),
                }
    
    def _distill_mutation_patterns(self, runs: List[ExperienceArchiveEntry], min_occ: int):
        """Track which mutation types are most effective."""
        mutation_success = {}
        
        for run in runs:
            for mut in run.successful_mutations:
                if mut not in mutation_success:
                    mutation_success[mut] = {'success': 0, 'fail': 0}
                mutation_success[mut]['success'] += 1
            
            for mut in run.failed_mutations:
                if mut not in mutation_success:
                    mutation_success[mut] = {'success': 0, 'fail': 0}
                mutation_success[mut]['fail'] += 1
        
        # Store as pattern
        for mut_type, counts in mutation_success.items():
            total = counts['success'] + counts['fail']
            if total < min_occ:
                continue
            
            rate = counts['success'] / total
            self.global_stats['mutation_effectiveness'] = mutation_success
            
            if rate > 0.3:  # Only patterns with decent success
                pattern_id = f"mutation_{mut_type}"
                self.patterns[pattern_id] = LearnedPattern(
                    pattern_id=pattern_id,
                    pattern_type="general",
                    description=f"Mutation '{mut_type}' has {rate:.0%} success rate",
                    conditions={},  # Always applicable
                    action={'mutation_type_preference': mut_type, 'expected_success': rate},
                    confidence=min(0.3 + total * 0.02, 0.85),
                    success_count=counts['success'],
                    failure_count=counts['fail'],
                    created_at=datetime.now().isoformat(),
                )
    
    def _conditions_match(
        self,
        conditions: Dict[str, Any],
        person: Dict[str, Any],
        objectives: Dict[str, float],
        zone_weights: Dict[str, float],
    ) -> bool:
        """Check if pattern conditions match current problem."""
        for key, value in conditions.items():
            if key == 'height_m_range':
                height = person.get('height_m', 1.75)
                if not (value[0] <= height <= value[1]):
                    return False
            
            elif key == 'spine_weight_range':
                spine_w = zone_weights.get('spine', 0.7)
                if not (value[0] <= spine_w <= value[1]):
                    return False
        
        return True
    
    def _compute_similarity(
        self,
        entry: ExperienceArchiveEntry,
        target_person: Dict[str, Any],
        target_objectives: Dict[str, float],
    ) -> float:
        """Compute similarity between archived run and current problem."""
        sim = 0.0
        
        # Height similarity
        h1 = entry.target_person.get('height_m', 1.75)
        h2 = target_person.get('height_m', 1.75)
        sim += 1.0 - abs(h1 - h2) / 0.5  # Normalize by 0.5m range
        
        # Objective priorities similarity
        for obj in target_objectives:
            v1 = entry.target_objectives.get(obj, 0.5)
            v2 = target_objectives.get(obj, 0.5)
            sim += 1.0 - abs(v1 - v2)
        
        # Normalize
        n_factors = 1 + len(target_objectives)
        return sim / n_factors
    
    def _update_global_stats(self, entry: ExperienceArchiveEntry):
        """Update global statistics with new entry."""
        self.global_stats['total_runs'] += 1
        
        if entry.outcome == "success":
            self.global_stats['successful_runs'] += 1
        
        # Running average of convergence generation
        n = self.global_stats['total_runs']
        old_avg = self.global_stats['avg_convergence_gen']
        self.global_stats['avg_convergence_gen'] = (
            old_avg * (n - 1) / n + entry.convergence_generation / n
        )
        
        if entry.final_fitness > self.global_stats['best_fitness_ever']:
            self.global_stats['best_fitness_ever'] = entry.final_fitness
    
    def _trim_archive(self):
        """Trim archive to max size, keeping best and most recent."""
        if len(self.archive) <= self.max_archive_size:
            return
        
        # Sort by fitness (keep best) and recency
        scored = []
        for i, entry in enumerate(self.archive):
            recency_score = i / len(self.archive)  # Higher = more recent
            fitness_score = entry.final_fitness
            combined = 0.3 * recency_score + 0.7 * fitness_score
            scored.append((combined, entry))
        
        scored.sort(key=lambda x: -x[0])
        self.archive = [entry for _, entry in scored[:self.max_archive_size]]
    
    def _save(self):
        """Save memory to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save patterns
        patterns_data = {
            pid: {
                'pattern_id': p.pattern_id,
                'pattern_type': p.pattern_type,
                'description': p.description,
                'conditions': p.conditions,
                'action': p.action,
                'confidence': p.confidence,
                'success_count': p.success_count,
                'failure_count': p.failure_count,
                'source_runs': p.source_runs,
                'created_at': p.created_at,
                'last_used': p.last_used,
            }
            for pid, p in self.patterns.items()
        }
        
        with open(self.storage_path / "patterns.json", 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        # Save archive (summary only, not full genomes)
        archive_data = []
        for entry in self.archive[-500:]:  # Keep last 500 for JSON
            archive_data.append({
                'run_id': entry.run_id,
                'timestamp': entry.timestamp,
                'target_person': entry.target_person,
                'target_objectives': entry.target_objectives,
                'zone_weights': entry.zone_weights,
                'final_fitness': entry.final_fitness,
                'final_objectives': entry.final_objectives,
                'total_generations': entry.total_generations,
                'convergence_generation': entry.convergence_generation,
                'outcome': entry.outcome,
            })
        
        with open(self.storage_path / "archive.json", 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        # Save global stats
        stats_copy = dict(self.global_stats)
        stats_copy['exciter_position_heatmap'] = self.global_stats['exciter_position_heatmap'].tolist()
        
        with open(self.storage_path / "global_stats.json", 'w') as f:
            json.dump(stats_copy, f, indent=2)
    
    def _load(self):
        """Load memory from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        # Load patterns
        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            for pid, data in patterns_data.items():
                self.patterns[pid] = LearnedPattern(**data)
        
        # Load archive
        archive_file = self.storage_path / "archive.json"
        if archive_file.exists():
            with open(archive_file, 'r') as f:
                archive_data = json.load(f)
            
            for data in archive_data:
                # Fill in missing fields with defaults
                entry = ExperienceArchiveEntry(
                    run_id=data.get('run_id', 'unknown'),
                    timestamp=data.get('timestamp', ''),
                    target_person=data.get('target_person', {}),
                    target_objectives=data.get('target_objectives', {}),
                    zone_weights=data.get('zone_weights', {}),
                    final_fitness=data.get('final_fitness', 0.0),
                    final_objectives=data.get('final_objectives', {}),
                    best_genome_summary=data.get('best_genome_summary', {}),
                    total_generations=data.get('total_generations', 0),
                    convergence_generation=data.get('convergence_generation', 0),
                    final_diversity=data.get('final_diversity', 0.0),
                    successful_mutations=data.get('successful_mutations', []),
                    failed_mutations=data.get('failed_mutations', []),
                    outcome=data.get('outcome', 'unknown'),
                )
                self.archive.append(entry)
        
        # Load global stats
        stats_file = self.storage_path / "global_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            
            # Restore numpy array
            if 'exciter_position_heatmap' in stats_data:
                stats_data['exciter_position_heatmap'] = np.array(
                    stats_data['exciter_position_heatmap']
                )
            
            self.global_stats.update(stats_data)


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionMemory:
    """
    Unified memory system combining short-term and long-term memory.
    
    This is the main interface for memory-augmented evolution.
    
    USAGE:
        memory = EvolutionMemory(
            storage_path="~/.golden_studio/memory",
            short_term_windows=[3, 5, 10],
        )
        
        # During evolution
        for generation in range(max_generations):
            # ... evaluate population ...
            
            memory.record_generation(generation, fitnesses, objectives)
            
            # Get adaptive parameters
            analysis = memory.get_trajectory_analysis()
            if analysis.is_stagnating:
                mutation_rate = analysis.recommended_mutation_rate
            
            # Get suggestions from long-term patterns
            suggestions = memory.get_pattern_suggestions(person, objectives)
        
        # After evolution
        memory.finalize_run(run_id, outcome, final_genome_summary)
    """
    
    def __init__(
        self,
        storage_path: str = None,
        short_term_windows: List[int] = [3, 5, 10],
        store_genomes: bool = False,
    ):
        """
        Initialize unified memory system.
        
        Args:
            storage_path: Path for persistent storage (None = memory only)
            short_term_windows: Time horizons for trajectory analysis
            store_genomes: Whether to store full genomes in short-term memory
        """
        self.stm = ShortTermMemory(
            buffer_sizes=short_term_windows,
            store_genomes=store_genomes,
        )
        
        self.ltm = LongTermMemory(
            storage_path=storage_path,
            max_archive_size=1000,
        )
        
        # Current run tracking
        self._current_run_id: str = None
        self._current_target_person: Dict[str, Any] = {}
        self._current_target_objectives: Dict[str, float] = {}
        self._current_zone_weights: Dict[str, float] = {}
        self._successful_mutations: List[str] = []
        self._failed_mutations: List[str] = []
        
        logger.info("EvolutionMemory initialized")
    
    def start_run(
        self,
        run_id: str,
        target_person: Dict[str, Any],
        target_objectives: Dict[str, float],
        zone_weights: Dict[str, float],
    ):
        """Start tracking a new optimization run."""
        self._current_run_id = run_id
        self._current_target_person = target_person
        self._current_target_objectives = target_objectives
        self._current_zone_weights = zone_weights
        self._successful_mutations = []
        self._failed_mutations = []
        
        self.stm.clear()  # Fresh short-term memory for new run
        
        logger.info(f"Started tracking run {run_id}")
    
    def record_generation(
        self,
        generation: int,
        population_fitnesses: np.ndarray,
        objective_vectors: List[Dict[str, float]],
        best_genome: Any = None,
        mutation_outcomes: Dict[str, int] = None,
        physics_decisions: List[str] = None,
    ) -> GenerationSnapshot:
        """
        Record a generation to short-term memory.
        
        Args:
            generation: Generation number
            population_fitnesses: Array of fitness scores
            objective_vectors: List of objective dicts
            best_genome: Best genome (optional)
            mutation_outcomes: {'mutation_type': improvement_count}
            physics_decisions: List of physics decisions made
        
        Returns:
            The recorded GenerationSnapshot
        """
        # Track mutation outcomes
        mutation_improvements = 0
        crossover_improvements = 0
        
        if mutation_outcomes:
            for mut_type, count in mutation_outcomes.items():
                if count > 0:
                    self._successful_mutations.append(mut_type)
                    mutation_improvements += count
                else:
                    self._failed_mutations.append(mut_type)
        
        return self.stm.record(
            generation=generation,
            population_fitnesses=population_fitnesses,
            objective_vectors=objective_vectors,
            best_genome=best_genome,
            mutation_improvements=mutation_improvements,
            crossover_improvements=crossover_improvements,
            physics_decisions=physics_decisions,
        )
    
    def get_trajectory_analysis(self, window: int = None) -> TrajectoryAnalysis:
        """Get trajectory analysis from short-term memory."""
        return self.stm.analyze_trajectory(window)
    
    def get_multi_scale_analysis(self) -> Dict[int, TrajectoryAnalysis]:
        """Get trajectory analysis at multiple time scales."""
        return self.stm.get_multi_scale_analysis()
    
    def get_pattern_suggestions(
        self,
        target_person: Dict[str, Any] = None,
        target_objectives: Dict[str, float] = None,
        zone_weights: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """Get suggestions from long-term pattern memory."""
        return self.ltm.get_suggestions(
            target_person or self._current_target_person,
            target_objectives or self._current_target_objectives,
            zone_weights or self._current_zone_weights,
        )
    
    def get_similar_runs(self, top_k: int = 5) -> List[ExperienceArchiveEntry]:
        """Get similar archived runs for warm-starting or reference."""
        return self.ltm.get_similar_runs(
            self._current_target_person,
            self._current_target_objectives,
            top_k,
        )
    
    def finalize_run(
        self,
        final_fitness: float,
        final_objectives: Dict[str, float],
        best_genome_summary: Dict[str, Any],
        outcome: str = "unknown",
    ):
        """
        Finalize and archive the current run.
        
        Call this at the END of evolution to store results in long-term memory.
        """
        if not self._current_run_id:
            logger.warning("No run was started, cannot finalize")
            return
        
        # Determine convergence generation
        if len(self.stm.buffer) > 0:
            snapshots = list(self.stm.buffer)
            best_gen = 0
            best_fit = 0.0
            for snap in snapshots:
                if snap.best_fitness > best_fit:
                    best_fit = snap.best_fitness
                    best_gen = snap.generation
            convergence_gen = best_gen
            final_diversity = snapshots[-1].population_diversity if snapshots else 0.0
        else:
            convergence_gen = 0
            final_diversity = 0.0
        
        self.ltm.archive_run(
            run_id=self._current_run_id,
            target_person=self._current_target_person,
            target_objectives=self._current_target_objectives,
            zone_weights=self._current_zone_weights,
            final_fitness=final_fitness,
            final_objectives=final_objectives,
            best_genome_summary=best_genome_summary,
            total_generations=len(self.stm.buffer),
            convergence_generation=convergence_gen,
            final_diversity=final_diversity,
            successful_mutations=list(set(self._successful_mutations)),
            failed_mutations=list(set(self._failed_mutations)),
            outcome=outcome,
        )
        
        # Periodically distill patterns
        if self.ltm.global_stats['total_runs'] % 10 == 0:
            self.ltm.distill_patterns()
        
        logger.info(f"Finalized run {self._current_run_id}: {outcome}")
    
    def update_pattern_outcome(self, pattern_id: str, success: bool):
        """Report outcome of using a pattern suggestion."""
        self.ltm.update_pattern_outcome(pattern_id, success)
    
    @property
    def global_stats(self) -> Dict[str, Any]:
        """Get global statistics from long-term memory."""
        return self.ltm.global_stats
    
    @property
    def patterns(self) -> Dict[str, LearnedPattern]:
        """Get all learned patterns."""
        return self.ltm.patterns
