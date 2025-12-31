"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 PHYSICS RULES ENGINE - Hybrid Rule System                     ║
║                                                                              ║
║   Hybrid approach: Manual core rules + Auto-expansion from experience        ║
║                                                                              ║
║   CORE RULES (hardcoded physics - always valid):                            ║
║   • Exciter-at-antinode maximizes coupling                                  ║
║   • Exciter-at-node minimizes coupling                                      ║
║   • Cutout-near-antinode tunes that mode                                    ║
║   • Edge distance affects structural integrity                               ║
║   • Phase between exciters affects beam steering                            ║
║                                                                              ║
║   LEARNED RULES (extracted from successful optimizations):                   ║
║   • Frequency-specific exciter positions                                     ║
║   • Material-dependent cutout placement                                      ║
║   • Zone-specific configurations                                             ║
║                                                                              ║
║   REFERENCES:                                                               ║
║   • Bai & Liu 2004: Genetic algorithm for exciter placement                 ║
║   • Aures 2001: Optimal exciter placement                                   ║
║   • Sum & Pan 2000: Modal cross-coupling                                    ║
║   • Lu 2012: Multi-exciter optimization                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Any, Optional, Tuple, Callable,
    Protocol, runtime_checkable, Union, Set
)
from enum import Enum, auto
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RULE TYPES & PRIORITIES
# ══════════════════════════════════════════════════════════════════════════════

class RuleCategory(Enum):
    """Category of physics rule."""
    HARD_CONSTRAINT = auto()   # Must be satisfied (structural integrity)
    SOFT_CONSTRAINT = auto()   # Should be satisfied (performance)
    GUIDANCE = auto()          # Suggestion for better results
    LEARNED = auto()           # Extracted from experience


class RuleDomain(Enum):
    """Domain the rule applies to."""
    POSITION = auto()          # Exciter position rules
    EMISSION = auto()          # DSP/emission parameter rules
    STRUCTURAL = auto()        # Plate structure rules
    FREQUENCY = auto()         # Frequency response rules
    ZONE = auto()              # Body zone targeting rules


@dataclass
class RuleCondition:
    """
    Condition for rule applicability.
    
    Example: "When targeting ear zones AND frequency < 200Hz"
    """
    attribute: str           # e.g., "target_zone", "frequency", "position"
    operator: str            # "eq", "lt", "gt", "in", "between"
    value: Any               # Comparison value
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if condition is met."""
        if self.attribute not in context:
            return False
        
        actual = context[self.attribute]
        
        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "neq":
            return actual != self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lte":
            return actual <= self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "between":
            return self.value[0] <= actual <= self.value[1]
        elif self.operator == "contains":
            return self.value in actual
        
        return False


@dataclass
class RuleSuggestion:
    """
    Suggestion from a rule evaluation.
    """
    parameter: str           # What to adjust
    direction: str           # "increase", "decrease", "set"
    magnitude: float         # How much (relative or absolute)
    confidence: float        # 0-1 confidence in suggestion
    explanation: str         # Human-readable reason
    source_rule: str         # Which rule generated this


@dataclass
class RuleEvaluationResult:
    """
    Result of evaluating a rule.
    """
    rule_id: str
    satisfied: bool
    score: float             # 0-1, how well satisfied
    suggestions: List[RuleSuggestion] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS RULE BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsRule:
    """
    Base class for physics rules.
    
    Each rule has:
    - id: Unique identifier
    - category: Hard/soft constraint or guidance
    - domain: What aspect of the system it applies to
    - conditions: When the rule applies
    - evaluate(): Check if rule is satisfied
    - suggest(): Generate improvement suggestions
    """
    id: str
    name: str
    description: str
    category: RuleCategory
    domain: RuleDomain
    conditions: List[RuleCondition] = field(default_factory=list)
    priority: int = 50  # 0-100, higher = more important
    enabled: bool = True
    
    # Paper reference for traceability
    paper_ref: Optional[str] = None
    
    def applies_to(self, context: Dict[str, Any]) -> bool:
        """Check if rule applies to current context."""
        if not self.enabled:
            return False
        
        # All conditions must be met
        return all(cond.evaluate(context) for cond in self.conditions)
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """
        Evaluate rule against context.
        
        Override in subclasses for specific rule logic.
        """
        raise NotImplementedError
    
    def suggest(self, context: Dict[str, Any]) -> List[RuleSuggestion]:
        """
        Generate improvement suggestions.
        
        Override in subclasses.
        """
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CORE PHYSICS RULES (Hardcoded - always valid)
# ══════════════════════════════════════════════════════════════════════════════

class ExciterAtAntinodeRule(PhysicsRule):
    """
    CORE RULE: Exciter at modal antinode maximizes coupling.
    
    Physics: Mode shape amplitude is maximum at antinodes.
    Placing exciter there maximizes energy transfer to that mode.
    
    Reference: Bai & Liu 2004, Aures 2001
    """
    
    def __init__(self):
        super().__init__(
            id="core_exciter_antinode",
            name="Exciter at Antinode",
            description="Place exciter near mode antinode for maximum coupling",
            category=RuleCategory.SOFT_CONSTRAINT,
            domain=RuleDomain.POSITION,
            priority=80,
            paper_ref="bai2004genetic"
        )
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """
        Evaluate exciter-antinode coupling.
        
        Context needs:
        - exciter_positions: List of (x, y) normalized
        - mode_shapes: Array of mode shapes
        - target_frequencies: Which modes to optimize
        """
        exciter_pos = context.get("exciter_positions", [])
        mode_shapes = context.get("mode_shapes")
        target_freqs = context.get("target_frequencies", [])
        
        if not exciter_pos or mode_shapes is None:
            return RuleEvaluationResult(
                rule_id=self.id,
                satisfied=True,  # Can't evaluate = assume OK
                score=0.5,
            )
        
        # Calculate coupling scores for each exciter
        coupling_scores = []
        suggestions = []
        
        for idx, (ex, ey) in enumerate(exciter_pos):
            # Sample mode shapes at exciter position
            nx, ny = mode_shapes.shape[1], mode_shapes.shape[2]
            ix = int(ex * (nx - 1))
            iy = int(ey * (ny - 1))
            
            # Average coupling across modes
            mode_values = mode_shapes[:, ix, iy]
            avg_coupling = np.mean(np.abs(mode_values))
            coupling_scores.append(avg_coupling)
            
            # Find better position if coupling is low
            if avg_coupling < 0.5:
                # Find antinode (max amplitude)
                flat_idx = np.argmax(np.abs(mode_shapes[0]))  # First mode
                best_ix, best_iy = np.unravel_index(flat_idx, mode_shapes.shape[1:])
                best_x, best_y = best_ix / (nx - 1), best_iy / (ny - 1)
                
                suggestions.append(RuleSuggestion(
                    parameter=f"exciter_{idx}_position",
                    direction="set",
                    magnitude=0.0,  # Will use explanation
                    confidence=0.7,
                    explanation=f"Move exciter {idx} toward ({best_x:.2f}, {best_y:.2f}) for better coupling",
                    source_rule=self.id
                ))
        
        overall_score = np.mean(coupling_scores) if coupling_scores else 0.5
        
        return RuleEvaluationResult(
            rule_id=self.id,
            satisfied=overall_score > 0.3,
            score=overall_score,
            suggestions=suggestions,
        )


class ExciterAvoidNodeRule(PhysicsRule):
    """
    CORE RULE: Exciter at modal node has zero coupling.
    
    Physics: Mode shape amplitude is zero at nodal lines.
    Exciter there cannot excite that mode at all.
    
    This is a hard constraint for single-frequency targeting.
    """
    
    def __init__(self):
        super().__init__(
            id="core_exciter_avoid_node",
            name="Avoid Nodal Lines",
            description="Do not place exciter on nodal lines of target modes",
            category=RuleCategory.HARD_CONSTRAINT,
            domain=RuleDomain.POSITION,
            priority=90,
            paper_ref="sum2000modal"
        )
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """Check if any exciter is on a nodal line."""
        exciter_pos = context.get("exciter_positions", [])
        mode_shapes = context.get("mode_shapes")
        
        if not exciter_pos or mode_shapes is None:
            return RuleEvaluationResult(rule_id=self.id, satisfied=True, score=1.0)
        
        violations = []
        min_coupling = 1.0
        
        for idx, (ex, ey) in enumerate(exciter_pos):
            nx, ny = mode_shapes.shape[1], mode_shapes.shape[2]
            ix = int(ex * (nx - 1))
            iy = int(ey * (ny - 1))
            
            # Check if on nodal line (near zero for all important modes)
            for mode_idx in range(min(3, len(mode_shapes))):  # Check first 3 modes
                coupling = abs(mode_shapes[mode_idx, ix, iy])
                min_coupling = min(min_coupling, coupling)
                
                if coupling < 0.05:  # Effectively on nodal line
                    violations.append(
                        f"Exciter {idx} is on nodal line of mode {mode_idx+1}"
                    )
        
        return RuleEvaluationResult(
            rule_id=self.id,
            satisfied=len(violations) == 0,
            score=min_coupling,
            violations=violations,
        )


class PhaseSteeringRule(PhysicsRule):
    """
    CORE RULE: Phase difference between exciters affects beam steering.
    
    Physics: Constructive/destructive interference based on phase.
    For stereo zones (ears), phase controls perceived direction.
    
    Reference: Sum & Pan 2000
    """
    
    def __init__(self):
        super().__init__(
            id="core_phase_steering",
            name="Phase Steering",
            description="Phase between exciters controls energy distribution",
            category=RuleCategory.GUIDANCE,
            domain=RuleDomain.EMISSION,
            priority=70,
            paper_ref="sum2000modal"
        )
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """
        Evaluate phase configuration.
        
        For ear zones: L/R should have small phase difference for centered image.
        For feet zones: Can use phase for distribution across spine.
        """
        emission_params = context.get("emission_params", [])
        target_zone = context.get("target_zone", "")
        
        if len(emission_params) < 2:
            return RuleEvaluationResult(rule_id=self.id, satisfied=True, score=1.0)
        
        suggestions = []
        
        # For ear targeting, check L/R phase balance
        if "ear" in target_zone.lower():
            head_exciters = [e for e in emission_params if e.get("zone") == "head"]
            if len(head_exciters) >= 2:
                phase_diff = abs(head_exciters[0].get("phase_deg", 0) - 
                               head_exciters[1].get("phase_deg", 0))
                
                # For centered image, phase diff should be small
                if phase_diff > 30:
                    suggestions.append(RuleSuggestion(
                        parameter="head_phase_difference",
                        direction="decrease",
                        magnitude=phase_diff - 15,
                        confidence=0.8,
                        explanation=f"Reduce L/R phase difference ({phase_diff:.0f}°) for centered stereo image",
                        source_rule=self.id
                    ))
                    return RuleEvaluationResult(
                        rule_id=self.id,
                        satisfied=False,
                        score=1.0 - (phase_diff / 180),
                        suggestions=suggestions
                    )
        
        return RuleEvaluationResult(rule_id=self.id, satisfied=True, score=1.0)


class EdgeDistanceRule(PhysicsRule):
    """
    CORE RULE: Minimum distance from plate edge.
    
    Physics: Structural integrity requires margin from edges.
    Also, edge effects change local stiffness.
    """
    
    def __init__(self, min_distance: float = 0.05):
        super().__init__(
            id="core_edge_distance",
            name="Edge Distance",
            description=f"Maintain at least {min_distance*100:.0f}mm from plate edge",
            category=RuleCategory.HARD_CONSTRAINT,
            domain=RuleDomain.STRUCTURAL,
            priority=95,
        )
        self.min_distance = min_distance
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """Check edge distances for exciters and cutouts."""
        exciter_pos = context.get("exciter_positions", [])
        cutout_pos = context.get("cutout_positions", [])
        
        violations = []
        min_dist = 1.0
        
        # Check exciters
        for idx, (x, y) in enumerate(exciter_pos):
            dist = min(x, 1-x, y, 1-y)
            min_dist = min(min_dist, dist)
            if dist < self.min_distance:
                violations.append(f"Exciter {idx} too close to edge ({dist*100:.0f}mm)")
        
        # Check cutouts
        for idx, cutout in enumerate(cutout_pos):
            x, y = cutout.get("x", 0.5), cutout.get("y", 0.5)
            r = cutout.get("radius", 0.02)
            dist = min(x - r, 1 - x - r, y - r, 1 - y - r)
            min_dist = min(min_dist, dist)
            if dist < self.min_distance:
                violations.append(f"Cutout {idx} too close to edge")
        
        return RuleEvaluationResult(
            rule_id=self.id,
            satisfied=len(violations) == 0,
            score=max(0, min(1, min_dist / self.min_distance)),
            violations=violations,
        )


class CutoutAntinodeTuningRule(PhysicsRule):
    """
    CORE RULE: Cutout near antinode tunes that mode's frequency.
    
    Physics: Removing material near high-amplitude regions
    reduces local stiffness, lowering that mode's frequency.
    
    Reference: ABH research (Krylov 2014)
    """
    
    def __init__(self):
        super().__init__(
            id="core_cutout_tuning",
            name="Cutout Tuning",
            description="Cutout placement affects mode frequencies",
            category=RuleCategory.GUIDANCE,
            domain=RuleDomain.FREQUENCY,
            priority=60,
            paper_ref="krylov2014abh"
        )
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """Evaluate cutout placement effectiveness."""
        cutout_pos = context.get("cutout_positions", [])
        mode_shapes = context.get("mode_shapes")
        target_mode = context.get("target_mode_index", 0)
        
        if not cutout_pos or mode_shapes is None:
            return RuleEvaluationResult(rule_id=self.id, satisfied=True, score=0.5)
        
        # Check if cutouts are near antinodes of target mode
        target_shape = mode_shapes[target_mode] if target_mode < len(mode_shapes) else mode_shapes[0]
        nx, ny = target_shape.shape
        
        effectiveness_scores = []
        for cutout in cutout_pos:
            x, y = cutout.get("x", 0.5), cutout.get("y", 0.5)
            ix, iy = int(x * (nx-1)), int(y * (ny-1))
            local_amplitude = abs(target_shape[ix, iy])
            effectiveness_scores.append(local_amplitude)
        
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0.5
        
        return RuleEvaluationResult(
            rule_id=self.id,
            satisfied=avg_effectiveness > 0.3,
            score=avg_effectiveness,
        )


# ══════════════════════════════════════════════════════════════════════════════
# LEARNED RULE - Created from successful optimizations
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedRule(PhysicsRule):
    """
    Rule extracted from successful optimization experience.
    
    These rules are generated automatically by analyzing what
    configurations led to good fitness scores.
    """
    # Learning metadata
    source_generation: int = 0
    training_samples: int = 0
    success_rate: float = 0.0
    
    # The actual learned pattern
    pattern: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = RuleCategory.LEARNED
    
    def evaluate(self, context: Dict[str, Any]) -> RuleEvaluationResult:
        """Check if pattern is being followed."""
        if not self.pattern:
            return RuleEvaluationResult(rule_id=self.id, satisfied=True, score=0.5)
        
        score = 0.0
        matches = 0
        total = 0
        
        for key, expected in self.pattern.items():
            if key in context:
                total += 1
                actual = context[key]
                
                if isinstance(expected, tuple) and len(expected) == 2:
                    # Range check
                    if expected[0] <= actual <= expected[1]:
                        matches += 1
                elif isinstance(expected, (int, float)):
                    # Numeric similarity
                    diff = abs(actual - expected) / (abs(expected) + 0.001)
                    if diff < 0.2:
                        matches += 1
                elif actual == expected:
                    matches += 1
        
        score = matches / total if total > 0 else 0.5
        
        return RuleEvaluationResult(
            rule_id=self.id,
            satisfied=score > 0.5,
            score=score * self.success_rate,  # Weight by historical success
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS RULES ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsRulesEngine:
    """
    Engine for evaluating and applying physics rules.
    
    USAGE:
        engine = PhysicsRulesEngine()
        
        # Add custom rules
        engine.add_rule(MyCustomRule())
        
        # Evaluate context
        results = engine.evaluate(context={
            "exciter_positions": [(0.3, 0.85), (0.7, 0.85)],
            "mode_shapes": mode_shapes_array,
            "target_zone": "ears",
        })
        
        # Get suggestions
        suggestions = engine.get_suggestions(context)
        
        # Learn new rules from experience
        engine.learn_from_success(context, fitness_score)
    """
    
    def __init__(self):
        """Initialize with core rules."""
        self.rules: Dict[str, PhysicsRule] = {}
        self.learned_rules: List[LearnedRule] = []
        self.experience_buffer: List[Tuple[Dict, float]] = []
        self.max_buffer_size: int = 100
        
        # Register core rules
        self._register_core_rules()
    
    def _register_core_rules(self):
        """Register hardcoded physics rules."""
        core_rules = [
            ExciterAtAntinodeRule(),
            ExciterAvoidNodeRule(),
            PhaseSteeringRule(),
            EdgeDistanceRule(),
            CutoutAntinodeTuningRule(),
        ]
        
        for rule in core_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: PhysicsRule):
        """Add a rule to the engine."""
        self.rules[rule.id] = rule
        logger.debug(f"Registered rule: {rule.id} ({rule.category.name})")
    
    def remove_rule(self, rule_id: str):
        """Remove a rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    def evaluate(
        self,
        context: Dict[str, Any],
        categories: Optional[Set[RuleCategory]] = None,
        domains: Optional[Set[RuleDomain]] = None,
    ) -> Dict[str, RuleEvaluationResult]:
        """
        Evaluate all applicable rules.
        
        Args:
            context: Current state/genome information
            categories: Filter by rule categories (None = all)
            domains: Filter by rule domains (None = all)
            
        Returns:
            Dict mapping rule_id to evaluation result
        """
        results = {}
        
        for rule_id, rule in self.rules.items():
            # Filter by category
            if categories and rule.category not in categories:
                continue
            
            # Filter by domain
            if domains and rule.domain not in domains:
                continue
            
            # Check if rule applies to context
            if not rule.applies_to(context):
                continue
            
            # Evaluate
            try:
                result = rule.evaluate(context)
                results[rule_id] = result
            except Exception as e:
                logger.warning(f"Rule {rule_id} evaluation failed: {e}")
        
        return results
    
    def check_hard_constraints(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check only hard constraints.
        
        Returns:
            (all_satisfied, list_of_violations)
        """
        results = self.evaluate(context, categories={RuleCategory.HARD_CONSTRAINT})
        
        violations = []
        for rule_id, result in results.items():
            if not result.satisfied:
                violations.extend(result.violations)
        
        return len(violations) == 0, violations
    
    def get_suggestions(
        self,
        context: Dict[str, Any],
        max_suggestions: int = 5,
    ) -> List[RuleSuggestion]:
        """
        Get improvement suggestions from all rules.
        
        Returns suggestions sorted by confidence.
        """
        results = self.evaluate(context)
        
        all_suggestions = []
        for rule_id, result in results.items():
            all_suggestions.extend(result.suggestions)
        
        # Sort by confidence
        all_suggestions.sort(key=lambda s: s.confidence, reverse=True)
        
        return all_suggestions[:max_suggestions]
    
    def get_overall_score(self, context: Dict[str, Any]) -> float:
        """
        Get weighted overall physics compliance score.
        """
        results = self.evaluate(context)
        
        if not results:
            return 1.0
        
        total_weight = 0
        weighted_score = 0
        
        for rule_id, result in results.items():
            rule = self.rules[rule_id]
            weight = rule.priority / 100  # Normalize priority
            
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 1.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEARNING FROM EXPERIENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_experience(self, context: Dict[str, Any], fitness: float):
        """
        Record a genome/fitness pair for later learning.
        """
        # Store copy to avoid mutation
        self.experience_buffer.append((dict(context), fitness))
        
        # Trim buffer
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
    
    def learn_from_experience(
        self,
        fitness_threshold: float = 0.7,
        min_samples: int = 10,
    ) -> Optional[LearnedRule]:
        """
        Extract a rule from successful experiences.
        
        Analyzes high-fitness genomes to find common patterns.
        """
        # Filter successful experiences
        successful = [(ctx, f) for ctx, f in self.experience_buffer if f >= fitness_threshold]
        
        if len(successful) < min_samples:
            logger.debug(f"Not enough successful samples ({len(successful)}) to learn")
            return None
        
        # Find common patterns
        pattern = self._extract_pattern(successful)
        
        if not pattern:
            return None
        
        # Create learned rule
        rule = LearnedRule(
            id=f"learned_{len(self.learned_rules)}",
            name=f"Learned Pattern #{len(self.learned_rules)}",
            description="Pattern extracted from successful optimizations",
            category=RuleCategory.LEARNED,
            domain=RuleDomain.POSITION,  # Could infer from pattern
            priority=40,
            source_generation=0,
            training_samples=len(successful),
            success_rate=len(successful) / len(self.experience_buffer),
            pattern=pattern,
        )
        
        self.learned_rules.append(rule)
        self.add_rule(rule)
        
        logger.info(f"Learned new rule: {rule.id} from {len(successful)} samples")
        return rule
    
    def _extract_pattern(
        self,
        successful: List[Tuple[Dict, float]]
    ) -> Dict[str, Any]:
        """
        Extract common patterns from successful contexts.
        """
        if not successful:
            return {}
        
        # Collect values for each key
        key_values: Dict[str, List] = {}
        
        for ctx, _ in successful:
            for key, value in ctx.items():
                if isinstance(value, (int, float)):
                    if key not in key_values:
                        key_values[key] = []
                    key_values[key].append(value)
        
        # Find consistent patterns (low variance)
        pattern = {}
        
        for key, values in key_values.items():
            if len(values) < 5:
                continue
            
            values_arr = np.array(values)
            mean = np.mean(values_arr)
            std = np.std(values_arr)
            cv = std / (abs(mean) + 0.001)  # Coefficient of variation
            
            # Low CV means consistent pattern
            if cv < 0.3:
                # Store as range (mean ± 1.5*std)
                pattern[key] = (mean - 1.5*std, mean + 1.5*std)
        
        return pattern
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state for persistence."""
        return {
            "learned_rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "pattern": r.pattern,
                    "success_rate": r.success_rate,
                    "training_samples": r.training_samples,
                }
                for r in self.learned_rules
            ],
            "experience_buffer_size": len(self.experience_buffer),
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Restore learned rules from saved state."""
        for rule_data in data.get("learned_rules", []):
            rule = LearnedRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description="Restored learned rule",
                category=RuleCategory.LEARNED,
                domain=RuleDomain.POSITION,
                priority=40,
                pattern=rule_data.get("pattern", {}),
                success_rate=rule_data.get("success_rate", 0.5),
                training_samples=rule_data.get("training_samples", 0),
            )
            self.learned_rules.append(rule)
            self.add_rule(rule)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Create context from genome
# ══════════════════════════════════════════════════════════════════════════════

def create_rule_context(
    genome: Any,
    mode_shapes: Optional[np.ndarray] = None,
    frequencies: Optional[np.ndarray] = None,
    target_zone: str = "",
) -> Dict[str, Any]:
    """
    Create rule evaluation context from a PlateGenome.
    
    This bridges the genome structure to rule evaluation.
    """
    context = {
        "target_zone": target_zone,
    }
    
    # Extract exciter positions
    if hasattr(genome, 'exciters'):
        context["exciter_positions"] = [
            (e.x, e.y) for e in genome.exciters
        ]
        
        # Extract emission params if available
        emission_params = []
        for e in genome.exciters:
            ep = {"zone": "head" if e.y > 0.7 else "feet" if e.y < 0.3 else "torso"}
            if hasattr(e, 'emission') and hasattr(e, 'is_emission_active'):
                if e.is_emission_active():
                    ep["phase_deg"] = e.emission.phase_deg
                    ep["delay_samples"] = e.emission.delay_samples
                    ep["gain_db"] = e.emission.gain_db
            emission_params.append(ep)
        context["emission_params"] = emission_params
    
    # Extract cutout positions
    if hasattr(genome, 'cutouts'):
        context["cutout_positions"] = [
            {"x": c.x, "y": c.y, "radius": c.radius}
            for c in genome.cutouts
        ]
    
    # Modal analysis results
    if mode_shapes is not None:
        context["mode_shapes"] = mode_shapes
    if frequencies is not None:
        context["frequencies"] = frequencies
    
    return context


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_physics_engine() -> PhysicsRulesEngine:
    """Create a default physics rules engine."""
    return PhysicsRulesEngine()
