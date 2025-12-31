"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           LTM DISTILLATION - Long-Term Memory Knowledge Transfer             ║
║                                                                              ║
║   Distills experience from successful optimization runs into:                ║
║   1. Neural priors for RDNN warm start                                       ║
║   2. Rule candidates for PhysicsRulesEngine                                  ║
║   3. Parameter recommendations for ObserverConfig                            ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Yu et al. 2023: Multi-Source Domain Knowledge Transfer                   ║
║   • Andrychowicz et al. 2018: Hindsight Experience Replay                    ║
║   • MDKL (Multi-Domain Knowledge Learning) concepts                          ║
║                                                                              ║
║   KEY INSIGHT:                                                               ║
║   ShortTermMemory (STM) + RDNN handle real-time trajectory.                  ║
║   LongTermMemory (LTM) stores raw experience archive.                        ║
║   THIS MODULE bridges them: distills LTM into actionable knowledge           ║
║   that improves RDNN initialization and physics rules.                       ║
║                                                                              ║
║   INTEGRATION:                                                               ║
║   - Reads from: LongTermMemory archive (evolution_memory.py)                 ║
║   - Writes to: RDNN hidden state priors, PhysicsRulesEngine rules            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import json
import logging
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DISTILLATION TYPES
# ══════════════════════════════════════════════════════════════════════════════

class DistillationType(Enum):
    """Types of knowledge that can be distilled."""
    PARAMETER_PRIOR = auto()      # Initial parameter recommendations
    RULE_CANDIDATE = auto()        # Candidate for PhysicsRulesEngine
    HIDDEN_STATE_PRIOR = auto()    # Prior for RDNN hidden state
    MUTATION_SCHEDULE = auto()     # Adaptive mutation schedule
    ZONE_WEIGHTING = auto()        # Zone importance weights


@dataclass
class DistilledKnowledge:
    """
    A piece of distilled knowledge extracted from LTM.
    
    This is the output of the distillation process - actionable
    knowledge that can improve future optimizations.
    """
    knowledge_id: str
    distillation_type: DistillationType
    
    # What was learned
    description: str
    parameters: Dict[str, Any]  # The actual knowledge (priors, rules, etc.)
    
    # Provenance
    source_runs: int          # How many runs contributed
    source_fitness_mean: float  # Average fitness of source runs
    source_fitness_std: float
    
    # Confidence and applicability
    confidence: float = 0.5
    domain_conditions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Example: {"person_height_m": (1.6, 1.9)} means applies to heights 1.6-1.9m
    
    # Timestamps
    created_at: str = ""
    last_verified: str = ""
    verification_count: int = 0
    verification_successes: int = 0
    
    def matches_domain(
        self,
        person_height_m: float = 1.75,
        spine_weight: float = 0.7,
        target_frequency: float = 100.0,
    ) -> bool:
        """Check if this knowledge applies to given domain."""
        for key, (low, high) in self.domain_conditions.items():
            if key == "person_height_m":
                if not (low <= person_height_m <= high):
                    return False
            elif key == "spine_weight":
                if not (low <= spine_weight <= high):
                    return False
            elif key == "target_frequency":
                if not (low <= target_frequency <= high):
                    return False
        return True
    
    def update_verification(self, success: bool):
        """Update verification statistics."""
        self.verification_count += 1
        if success:
            self.verification_successes += 1
        
        # Update confidence based on verification
        if self.verification_count > 0:
            success_rate = self.verification_successes / self.verification_count
            # Blend with prior confidence
            self.confidence = 0.3 * self.confidence + 0.7 * success_rate
        
        self.last_verified = datetime.now().isoformat()


@dataclass
class ExperienceStatistics:
    """
    Aggregated statistics from LTM archive.
    
    Used as input to distillation algorithms.
    """
    total_runs: int = 0
    successful_runs: int = 0
    
    # Fitness statistics
    fitness_mean: float = 0.0
    fitness_std: float = 0.0
    fitness_best: float = 0.0
    
    # Convergence statistics
    convergence_gen_mean: float = 50
    convergence_gen_std: float = 20
    
    # Zone effectiveness
    zone_fitness_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Example: {"spine_weight_0.7": {"mean": 0.85, "std": 0.05}}
    
    # Exciter placement heatmap (discretized)
    exciter_heatmap: Optional[np.ndarray] = None  # 10x10 normalized success
    
    # Mutation effectiveness
    mutation_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Contour preferences
    contour_fitness: Dict[str, float] = field(default_factory=dict)
    
    # Parameter correlations
    height_exciter_correlation: float = 0.0
    weight_cutout_correlation: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LTM DISTILLER (Main Class)
# ══════════════════════════════════════════════════════════════════════════════

class LTMDistiller:
    """
    Distills Long-Term Memory into actionable knowledge.
    
    This is the "teacher" that extracts patterns from experience
    and packages them for consumption by:
    - RDNN (hidden state priors)
    - PhysicsRulesEngine (rule candidates)
    - ObserverConfig (parameter recommendations)
    
    USAGE:
        # Connect to existing LTM
        distiller = LTMDistiller()
        
        # Analyze experience archive
        stats = distiller.analyze_archive(ltm.archive)
        
        # Distill specific knowledge types
        priors = distiller.distill_parameter_priors(stats)
        rules = distiller.distill_rule_candidates(stats)
        rdnn_init = distiller.distill_rdnn_priors(stats)
        
        # Apply to new optimization
        rdnn.initialize_from_prior(rdnn_init)
        physics_engine.add_learned_rules(rules)
    
    PAPER REFERENCE:
        Yu et al. 2023 - MDKL transfers knowledge across domains
        via learned representations, not just raw patterns.
    """
    
    def __init__(
        self,
        min_runs_for_distillation: int = 5,
        min_confidence_for_rule: float = 0.6,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize LTM Distiller.
        
        Args:
            min_runs_for_distillation: Minimum runs needed to extract patterns
            min_confidence_for_rule: Minimum confidence to create a rule
            storage_path: Path to store distilled knowledge
        """
        self.min_runs = min_runs_for_distillation
        self.min_confidence = min_confidence_for_rule
        self.storage_path = storage_path
        
        # Distilled knowledge cache
        self.knowledge_base: Dict[str, DistilledKnowledge] = {}
        
        # Load existing knowledge
        if storage_path:
            self._load_knowledge()
        
        logger.info(
            f"LTMDistiller initialized: min_runs={min_runs_for_distillation}, "
            f"{len(self.knowledge_base)} existing knowledge items"
        )
    
    def analyze_archive(
        self,
        archive: List[Any],  # List[ExperienceArchiveEntry] from evolution_memory
    ) -> ExperienceStatistics:
        """
        Analyze LTM archive to extract statistics.
        
        This is the first step before distillation.
        
        Args:
            archive: List of ExperienceArchiveEntry from LongTermMemory
        
        Returns:
            ExperienceStatistics with aggregated data
        """
        if not archive:
            return ExperienceStatistics()
        
        stats = ExperienceStatistics()
        stats.total_runs = len(archive)
        
        # Collect data
        fitnesses = []
        convergence_gens = []
        zone_data = defaultdict(list)  # zone_key -> [fitness]
        mutation_success = defaultdict(lambda: {"success": 0, "total": 0})
        contour_fitness = defaultdict(list)
        exciter_positions = []  # [(x, y, fitness)]
        height_exciter = []  # [(height, exciter_y)]
        
        for entry in archive:
            # Basic stats
            fitnesses.append(entry.final_fitness)
            convergence_gens.append(entry.convergence_generation)
            
            if entry.outcome == "success":
                stats.successful_runs += 1
            
            # Zone effectiveness
            spine_w = entry.zone_weights.get("spine", 0.7)
            zone_key = f"spine_{spine_w:.1f}"
            zone_data[zone_key].append(entry.final_fitness)
            
            # Mutation effectiveness
            for mut in entry.successful_mutations:
                mutation_success[mut]["success"] += 1
                mutation_success[mut]["total"] += 1
            for mut in entry.failed_mutations:
                mutation_success[mut]["total"] += 1
            
            # Contour effectiveness
            contour = entry.best_genome_summary.get("contour_type", "rectangle")
            contour_fitness[contour].append(entry.final_fitness)
            
            # Exciter positions
            exciter_y = entry.best_genome_summary.get("exciter_y_mean")
            exciter_x = entry.best_genome_summary.get("exciter_x_mean")
            if exciter_x is not None and exciter_y is not None:
                exciter_positions.append((exciter_x, exciter_y, entry.final_fitness))
            
            # Height-exciter correlation
            height = entry.target_person.get("height_m", 1.75)
            if exciter_y is not None:
                height_exciter.append((height, exciter_y))
        
        # Aggregate statistics
        stats.fitness_mean = float(np.mean(fitnesses))
        stats.fitness_std = float(np.std(fitnesses))
        stats.fitness_best = float(np.max(fitnesses))
        
        stats.convergence_gen_mean = float(np.mean(convergence_gens))
        stats.convergence_gen_std = float(np.std(convergence_gens))
        
        # Zone fitness map
        for zone_key, zone_fitnesses in zone_data.items():
            if len(zone_fitnesses) >= 2:
                stats.zone_fitness_map[zone_key] = {
                    "mean": float(np.mean(zone_fitnesses)),
                    "std": float(np.std(zone_fitnesses)),
                    "count": len(zone_fitnesses),
                }
        
        # Mutation success rates
        for mut, counts in mutation_success.items():
            if counts["total"] > 0:
                stats.mutation_success_rates[mut] = counts["success"] / counts["total"]
        
        # Contour fitness
        for contour, cont_fitnesses in contour_fitness.items():
            stats.contour_fitness[contour] = float(np.mean(cont_fitnesses))
        
        # Exciter heatmap (10x10 grid)
        if exciter_positions:
            heatmap = np.zeros((10, 10))
            weight_map = np.zeros((10, 10))
            
            for x, y, fitness in exciter_positions:
                ix = int(np.clip(x * 10, 0, 9))
                iy = int(np.clip(y * 10, 0, 9))
                heatmap[iy, ix] += fitness
                weight_map[iy, ix] += 1
            
            # Normalize
            with np.errstate(divide='ignore', invalid='ignore'):
                heatmap = np.where(weight_map > 0, heatmap / weight_map, 0)
            
            stats.exciter_heatmap = heatmap
        
        # Height-exciter correlation
        if len(height_exciter) > 5:
            heights, exciters = zip(*height_exciter)
            stats.height_exciter_correlation = float(np.corrcoef(heights, exciters)[0, 1])
        
        return stats
    
    def distill_parameter_priors(
        self,
        stats: ExperienceStatistics,
    ) -> List[DistilledKnowledge]:
        """
        Distill recommended parameter priors from experience.
        
        These are initial values/ranges for optimization parameters.
        
        Args:
            stats: Experience statistics from analyze_archive()
        
        Returns:
            List of DistilledKnowledge with parameter recommendations
        """
        knowledge_items = []
        
        if stats.total_runs < self.min_runs:
            logger.info(f"Not enough runs ({stats.total_runs}) for parameter priors")
            return knowledge_items
        
        # 1. Mutation rate prior based on convergence speed
        if stats.convergence_gen_mean > 0:
            # Faster convergence → can start with lower mutation
            # Slower convergence → need higher initial mutation
            base_mutation = 0.2
            if stats.convergence_gen_mean < 30:
                recommended_mutation = base_mutation * 0.8
            elif stats.convergence_gen_mean > 70:
                recommended_mutation = base_mutation * 1.3
            else:
                recommended_mutation = base_mutation
            
            knowledge_items.append(DistilledKnowledge(
                knowledge_id="param_mutation_rate",
                distillation_type=DistillationType.PARAMETER_PRIOR,
                description=f"Recommended initial mutation rate based on {stats.total_runs} runs",
                parameters={
                    "initial_mutation_rate": recommended_mutation,
                    "mutation_rate_range": (0.1, 0.4),
                },
                source_runs=stats.total_runs,
                source_fitness_mean=stats.fitness_mean,
                source_fitness_std=stats.fitness_std,
                confidence=min(0.5 + stats.successful_runs / stats.total_runs * 0.4, 0.9),
                created_at=datetime.now().isoformat(),
            ))
        
        # 2. Population size prior
        if stats.convergence_gen_std > 0:
            # High variance in convergence → need larger population for robustness
            if stats.convergence_gen_std > 25:
                recommended_pop = 60
            else:
                recommended_pop = 40
            
            knowledge_items.append(DistilledKnowledge(
                knowledge_id="param_population_size",
                distillation_type=DistillationType.PARAMETER_PRIOR,
                description=f"Recommended population size for stability",
                parameters={
                    "population_size": recommended_pop,
                    "elite_ratio": 0.1,
                },
                source_runs=stats.total_runs,
                source_fitness_mean=stats.fitness_mean,
                source_fitness_std=stats.fitness_std,
                confidence=0.6,
                created_at=datetime.now().isoformat(),
            ))
        
        # 3. Best contour prior
        if stats.contour_fitness:
            best_contour = max(stats.contour_fitness, key=stats.contour_fitness.get)
            best_fitness = stats.contour_fitness[best_contour]
            
            knowledge_items.append(DistilledKnowledge(
                knowledge_id="param_contour_preference",
                distillation_type=DistillationType.PARAMETER_PRIOR,
                description=f"Preferred contour type: {best_contour}",
                parameters={
                    "preferred_contour": best_contour,
                    "contour_fitness_map": stats.contour_fitness,
                },
                source_runs=stats.total_runs,
                source_fitness_mean=best_fitness,
                source_fitness_std=0.0,
                confidence=min(0.4 + len(stats.contour_fitness) * 0.1, 0.8),
                created_at=datetime.now().isoformat(),
            ))
        
        # Store in knowledge base
        for k in knowledge_items:
            self.knowledge_base[k.knowledge_id] = k
        
        return knowledge_items
    
    def distill_rule_candidates(
        self,
        stats: ExperienceStatistics,
    ) -> List[DistilledKnowledge]:
        """
        Distill rule candidates for PhysicsRulesEngine.
        
        These are patterns that could become learned rules.
        
        Args:
            stats: Experience statistics from analyze_archive()
        
        Returns:
            List of DistilledKnowledge with rule specifications
        """
        knowledge_items = []
        
        if stats.total_runs < self.min_runs:
            return knowledge_items
        
        # 1. Exciter placement rule from heatmap
        if stats.exciter_heatmap is not None:
            heatmap = stats.exciter_heatmap
            
            # Find hot zones (high success regions)
            threshold = np.percentile(heatmap[heatmap > 0], 75) if np.any(heatmap > 0) else 0
            hot_zones = np.argwhere(heatmap >= threshold)
            
            if len(hot_zones) > 0:
                # Convert to normalized coordinates
                hot_regions = []
                for iy, ix in hot_zones:
                    hot_regions.append({
                        "x_range": (ix / 10, (ix + 1) / 10),
                        "y_range": (iy / 10, (iy + 1) / 10),
                        "fitness": float(heatmap[iy, ix]),
                    })
                
                knowledge_items.append(DistilledKnowledge(
                    knowledge_id="rule_exciter_hot_zones",
                    distillation_type=DistillationType.RULE_CANDIDATE,
                    description="Exciter placement hot zones from experience",
                    parameters={
                        "hot_regions": hot_regions,
                        "heatmap": heatmap.tolist(),
                    },
                    source_runs=stats.total_runs,
                    source_fitness_mean=stats.fitness_mean,
                    source_fitness_std=stats.fitness_std,
                    confidence=min(0.5 + stats.successful_runs / max(stats.total_runs, 1) * 0.4, 0.85),
                    created_at=datetime.now().isoformat(),
                ))
        
        # 2. Height-based exciter rule
        if abs(stats.height_exciter_correlation) > 0.3:
            # Significant correlation between height and exciter position
            direction = "higher" if stats.height_exciter_correlation > 0 else "lower"
            
            knowledge_items.append(DistilledKnowledge(
                knowledge_id="rule_height_exciter",
                distillation_type=DistillationType.RULE_CANDIDATE,
                description=f"Taller persons → {direction} exciter Y (correlation={stats.height_exciter_correlation:.2f})",
                parameters={
                    "correlation": stats.height_exciter_correlation,
                    "adjustment_per_10cm": stats.height_exciter_correlation * 0.05,
                },
                source_runs=stats.total_runs,
                source_fitness_mean=stats.fitness_mean,
                source_fitness_std=stats.fitness_std,
                confidence=min(0.4 + abs(stats.height_exciter_correlation) * 0.5, 0.8),
                created_at=datetime.now().isoformat(),
            ))
        
        # 3. Zone-specific rules
        for zone_key, zone_stats in stats.zone_fitness_map.items():
            if zone_stats["count"] >= self.min_runs:
                knowledge_items.append(DistilledKnowledge(
                    knowledge_id=f"rule_zone_{zone_key}",
                    distillation_type=DistillationType.RULE_CANDIDATE,
                    description=f"Zone configuration {zone_key} → fitness {zone_stats['mean']:.3f}",
                    parameters={
                        "zone_config": zone_key,
                        "expected_fitness": zone_stats["mean"],
                        "fitness_std": zone_stats["std"],
                    },
                    source_runs=zone_stats["count"],
                    source_fitness_mean=zone_stats["mean"],
                    source_fitness_std=zone_stats["std"],
                    confidence=min(0.5 + zone_stats["count"] / 20 * 0.3, 0.8),
                    created_at=datetime.now().isoformat(),
                ))
        
        # Store
        for k in knowledge_items:
            self.knowledge_base[k.knowledge_id] = k
        
        return knowledge_items
    
    def distill_rdnn_priors(
        self,
        stats: ExperienceStatistics,
        hidden_size: int = 64,
    ) -> DistilledKnowledge:
        """
        Distill prior initialization for RDNN hidden state.
        
        Instead of zero-initialization, use experience to create
        meaningful starting point for hidden state.
        
        Args:
            stats: Experience statistics
            hidden_size: RDNN hidden dimension
        
        Returns:
            DistilledKnowledge with hidden state initialization
        """
        # Create prior hidden state from experience
        # This is a simplified version - could be more sophisticated with actual training
        
        # Encode key statistics into prior
        prior_features = [
            stats.fitness_mean,
            stats.fitness_std,
            stats.convergence_gen_mean / 100,  # Normalized
            stats.successful_runs / max(stats.total_runs, 1),  # Success rate
        ]
        
        # Expand to hidden size using learned linear combination
        # (In practice, this could be a trained encoder)
        np.random.seed(42)  # Reproducible
        expansion_matrix = np.random.randn(len(prior_features), hidden_size) * 0.1
        
        prior_hidden = np.tanh(
            np.dot(prior_features, expansion_matrix)
        ).astype(np.float32)
        
        return DistilledKnowledge(
            knowledge_id="rdnn_hidden_prior",
            distillation_type=DistillationType.HIDDEN_STATE_PRIOR,
            description=f"RDNN hidden state prior from {stats.total_runs} runs",
            parameters={
                "hidden_prior": prior_hidden.tolist(),
                "hidden_size": hidden_size,
                "source_stats": {
                    "fitness_mean": stats.fitness_mean,
                    "success_rate": stats.successful_runs / max(stats.total_runs, 1),
                },
            },
            source_runs=stats.total_runs,
            source_fitness_mean=stats.fitness_mean,
            source_fitness_std=stats.fitness_std,
            confidence=min(0.4 + stats.total_runs / 50 * 0.4, 0.8),
            created_at=datetime.now().isoformat(),
        )
    
    def distill_mutation_schedule(
        self,
        stats: ExperienceStatistics,
    ) -> DistilledKnowledge:
        """
        Distill adaptive mutation schedule from experience.
        
        Creates a generation-dependent mutation rate schedule.
        
        Args:
            stats: Experience statistics
        
        Returns:
            DistilledKnowledge with mutation schedule
        """
        # Create schedule based on typical convergence
        conv_gen = stats.convergence_gen_mean
        
        # Phase 1: Exploration (high mutation until ~30% of convergence)
        # Phase 2: Exploitation (lower mutation)
        # Phase 3: Fine-tuning (very low mutation near convergence)
        
        schedule = [
            {"gen_range": (0, int(conv_gen * 0.3)), "mutation_rate": 0.3},
            {"gen_range": (int(conv_gen * 0.3), int(conv_gen * 0.7)), "mutation_rate": 0.2},
            {"gen_range": (int(conv_gen * 0.7), int(conv_gen)), "mutation_rate": 0.1},
            {"gen_range": (int(conv_gen), 200), "mutation_rate": 0.05},
        ]
        
        return DistilledKnowledge(
            knowledge_id="mutation_schedule",
            distillation_type=DistillationType.MUTATION_SCHEDULE,
            description=f"Adaptive mutation schedule based on convergence at gen {conv_gen:.0f}",
            parameters={
                "schedule": schedule,
                "convergence_generation": conv_gen,
            },
            source_runs=stats.total_runs,
            source_fitness_mean=stats.fitness_mean,
            source_fitness_std=stats.fitness_std,
            confidence=min(0.5 + stats.successful_runs / stats.total_runs * 0.3, 0.8),
            created_at=datetime.now().isoformat(),
        )
    
    def get_applicable_knowledge(
        self,
        person_height_m: float = 1.75,
        spine_weight: float = 0.7,
        target_frequency: float = 100.0,
        min_confidence: float = None,
    ) -> List[DistilledKnowledge]:
        """
        Get all knowledge applicable to given domain.
        
        Args:
            person_height_m: Target person height
            spine_weight: Spine zone weight
            target_frequency: Target frequency
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of applicable DistilledKnowledge items
        """
        min_conf = min_confidence or self.min_confidence
        
        applicable = []
        for k in self.knowledge_base.values():
            if k.confidence >= min_conf:
                if k.matches_domain(person_height_m, spine_weight, target_frequency):
                    applicable.append(k)
        
        # Sort by confidence
        applicable.sort(key=lambda x: -x.confidence)
        return applicable
    
    def apply_to_rdnn(
        self,
        rdnn_memory: Any,  # RDNNMemory from rdnn_memory.py
        knowledge: DistilledKnowledge,
    ) -> bool:
        """
        Apply distilled knowledge to RDNN hidden state.
        
        Args:
            rdnn_memory: RDNNMemory instance
            knowledge: HIDDEN_STATE_PRIOR knowledge
        
        Returns:
            True if successfully applied
        """
        if knowledge.distillation_type != DistillationType.HIDDEN_STATE_PRIOR:
            logger.warning(f"Cannot apply {knowledge.distillation_type} to RDNN")
            return False
        
        try:
            prior_hidden = knowledge.parameters.get("hidden_prior")
            if prior_hidden is None:
                return False
            
            # Convert to torch tensor
            hidden_tensor = torch.tensor(
                [prior_hidden],  # Add batch dimension
                dtype=torch.float32,
                device=rdnn_memory.config.device,
            )
            
            # Set as initial hidden state
            # Shape: (num_layers, batch, hidden_size)
            num_layers = rdnn_memory.config.num_layers
            hidden_tensor = hidden_tensor.unsqueeze(0).expand(num_layers, -1, -1)
            
            rdnn_memory._hidden = hidden_tensor
            
            logger.info(f"Applied RDNN prior from {knowledge.source_runs} runs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply RDNN prior: {e}")
            return False
    
    def save_knowledge(self, path: Optional[Path] = None):
        """Save distilled knowledge to disk."""
        save_path = path or self.storage_path
        if save_path is None:
            return
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Serialize knowledge
        data = {}
        for kid, k in self.knowledge_base.items():
            data[kid] = {
                "knowledge_id": k.knowledge_id,
                "distillation_type": k.distillation_type.name,
                "description": k.description,
                "parameters": k.parameters,
                "source_runs": k.source_runs,
                "source_fitness_mean": k.source_fitness_mean,
                "source_fitness_std": k.source_fitness_std,
                "confidence": k.confidence,
                "domain_conditions": k.domain_conditions,
                "created_at": k.created_at,
                "last_verified": k.last_verified,
                "verification_count": k.verification_count,
                "verification_successes": k.verification_successes,
            }
        
        with open(save_path / "distilled_knowledge.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(data)} knowledge items to {save_path}")
    
    def _load_knowledge(self):
        """Load existing knowledge from disk."""
        if self.storage_path is None:
            return
        
        knowledge_file = self.storage_path / "distilled_knowledge.json"
        if not knowledge_file.exists():
            return
        
        try:
            with open(knowledge_file) as f:
                data = json.load(f)
            
            for kid, kdata in data.items():
                self.knowledge_base[kid] = DistilledKnowledge(
                    knowledge_id=kdata["knowledge_id"],
                    distillation_type=DistillationType[kdata["distillation_type"]],
                    description=kdata["description"],
                    parameters=kdata["parameters"],
                    source_runs=kdata["source_runs"],
                    source_fitness_mean=kdata["source_fitness_mean"],
                    source_fitness_std=kdata["source_fitness_std"],
                    confidence=kdata["confidence"],
                    domain_conditions=kdata.get("domain_conditions", {}),
                    created_at=kdata.get("created_at", ""),
                    last_verified=kdata.get("last_verified", ""),
                    verification_count=kdata.get("verification_count", 0),
                    verification_successes=kdata.get("verification_successes", 0),
                )
            
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge items")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_distiller(
    storage_path: Optional[str] = None,
    min_runs: int = 5,
) -> LTMDistiller:
    """Factory function to create LTM Distiller."""
    path = Path(storage_path) if storage_path else None
    return LTMDistiller(
        min_runs_for_distillation=min_runs,
        storage_path=path,
    )


def distill_and_apply(
    ltm_archive: List[Any],
    rdnn_memory: Any = None,
    physics_engine: Any = None,
    storage_path: Optional[str] = None,
) -> Dict[str, List[DistilledKnowledge]]:
    """
    One-shot distillation and application.
    
    Convenience function to distill from archive and apply to components.
    
    Args:
        ltm_archive: Archive from LongTermMemory
        rdnn_memory: Optional RDNNMemory to initialize
        physics_engine: Optional PhysicsRulesEngine to add rules
        storage_path: Optional path to save knowledge
    
    Returns:
        Dict with distilled knowledge by type
    """
    distiller = create_distiller(storage_path)
    
    # Analyze archive
    stats = distiller.analyze_archive(ltm_archive)
    
    # Distill all types
    results = {
        "parameter_priors": distiller.distill_parameter_priors(stats),
        "rule_candidates": distiller.distill_rule_candidates(stats),
        "mutation_schedule": [distiller.distill_mutation_schedule(stats)] if stats.total_runs >= distiller.min_runs else [],
    }
    
    # Apply RDNN prior if available
    if rdnn_memory is not None and stats.total_runs >= distiller.min_runs:
        rdnn_prior = distiller.distill_rdnn_priors(stats, rdnn_memory.config.hidden_size)
        results["rdnn_prior"] = [rdnn_prior]
        distiller.apply_to_rdnn(rdnn_memory, rdnn_prior)
    
    # Apply rules to physics engine if available
    if physics_engine is not None and hasattr(physics_engine, "add_learned_rule"):
        for rule_knowledge in results.get("rule_candidates", []):
            if rule_knowledge.confidence >= distiller.min_confidence:
                # Convert to LearnedRule format
                physics_engine.add_learned_rule(
                    name=rule_knowledge.knowledge_id,
                    description=rule_knowledge.description,
                    weight=rule_knowledge.confidence,
                    paper_ref="distilled_from_experience",
                )
    
    # Save
    distiller.save_knowledge()
    
    return results
