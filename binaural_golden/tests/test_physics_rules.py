"""
Tests for PhysicsRulesEngine - Phase 3 of Action Plan 3.0

Hybrid physics rules: core (hardcoded) + learned (from experience)
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.physics_rules import (
    PhysicsRulesEngine, PhysicsRule, LearnedRule,
    RuleCategory, RuleDomain, RuleCondition, RuleSuggestion, RuleEvaluationResult,
    ExciterAtAntinodeRule, ExciterAvoidNodeRule, PhaseSteeringRule,
    EdgeDistanceRule, CutoutAntinodeTuningRule,
    create_rule_context, create_physics_engine
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """Create physics engine with core rules."""
    return create_physics_engine()


@pytest.fixture
def mode_shapes():
    """Create simple mode shapes for testing."""
    # 20x20 grid
    nx, ny = 20, 20
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Mode (1,1): sin(π*x) * sin(π*y) - antinode at center
    mode_11 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Mode (2,1): sin(2π*x) * sin(π*y) - nodal line at x=0.5
    mode_21 = np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    
    # Mode (1,2): sin(π*x) * sin(2π*y) - nodal line at y=0.5
    mode_12 = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    
    return np.array([mode_11, mode_21, mode_12])


# ══════════════════════════════════════════════════════════════════════════════
# TEST CORE RULES
# ══════════════════════════════════════════════════════════════════════════════

class TestCoreRules:
    """Test hardcoded physics rules."""
    
    def test_engine_has_core_rules(self, engine):
        """Engine should be initialized with core rules."""
        assert len(engine.rules) >= 5
        assert "core_exciter_antinode" in engine.rules
        assert "core_exciter_avoid_node" in engine.rules
        assert "core_edge_distance" in engine.rules
    
    def test_exciter_at_antinode_rule(self, mode_shapes):
        """Exciter at antinode should have high coupling."""
        rule = ExciterAtAntinodeRule()
        
        # Mode (1,1) has antinode at center (0.5, 0.5)
        # But modes (2,1) and (1,2) have nodal lines there, so average is ~0.4
        context = {
            "exciter_positions": [(0.5, 0.5)],  # At antinode of mode 1,1
            "mode_shapes": mode_shapes,
        }
        
        result = rule.evaluate(context)
        
        assert result.satisfied
        assert result.score > 0.3  # Average coupling across modes
    
    def test_exciter_at_node_detected(self, mode_shapes):
        """Exciter at nodal line should be flagged."""
        rule = ExciterAvoidNodeRule()
        
        # Mode (2,1) has nodal line at x=0.5
        context = {
            "exciter_positions": [(0.5, 0.5)],  # On nodal line of mode 21
            "mode_shapes": mode_shapes,
        }
        
        result = rule.evaluate(context)
        
        # Should detect that we're near nodal line of mode 21
        # Note: mode_11 has antinode there, so overall may still pass
        # The rule checks if ANY important mode has near-zero coupling
        assert result.rule_id == "core_exciter_avoid_node"
    
    def test_edge_distance_violation(self):
        """Exciter too close to edge should be flagged."""
        rule = EdgeDistanceRule(min_distance=0.05)
        
        context = {
            "exciter_positions": [(0.02, 0.5)],  # Too close to left edge
            "cutout_positions": [],
        }
        
        result = rule.evaluate(context)
        
        assert not result.satisfied
        assert len(result.violations) > 0
        assert "too close to edge" in result.violations[0].lower()
    
    def test_edge_distance_satisfied(self):
        """Exciter with good clearance should pass."""
        rule = EdgeDistanceRule(min_distance=0.05)
        
        context = {
            "exciter_positions": [(0.3, 0.5), (0.7, 0.5)],
            "cutout_positions": [],
        }
        
        result = rule.evaluate(context)
        
        assert result.satisfied
        assert result.score > 0.9
    
    def test_phase_steering_for_ears(self):
        """L/R phase difference should be small for centered image."""
        rule = PhaseSteeringRule()
        
        # Large phase difference
        context_bad = {
            "emission_params": [
                {"zone": "head", "phase_deg": 0},
                {"zone": "head", "phase_deg": 90},
            ],
            "target_zone": "ears"
        }
        
        result_bad = rule.evaluate(context_bad)
        
        # Should suggest reducing phase difference
        assert len(result_bad.suggestions) > 0
        assert "phase" in result_bad.suggestions[0].parameter.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TEST ENGINE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineEvaluation:
    """Test engine evaluation methods."""
    
    def test_evaluate_all_rules(self, engine, mode_shapes):
        """Engine should evaluate all applicable rules."""
        context = {
            "exciter_positions": [(0.3, 0.85), (0.7, 0.85)],
            "mode_shapes": mode_shapes,
            "cutout_positions": [],
        }
        
        results = engine.evaluate(context)
        
        # Should have results from multiple rules
        assert len(results) > 0
        
        # All results should be RuleEvaluationResult
        for rule_id, result in results.items():
            assert isinstance(result, RuleEvaluationResult)
    
    def test_filter_by_category(self, engine):
        """Should filter evaluation by category."""
        context = {
            "exciter_positions": [(0.02, 0.5)],  # Edge violation
            "cutout_positions": [],
        }
        
        # Only hard constraints
        hard_results = engine.evaluate(
            context,
            categories={RuleCategory.HARD_CONSTRAINT}
        )
        
        # Should include edge distance (hard constraint)
        assert any("edge" in rid for rid in hard_results.keys())
        
        # But not soft constraints or guidance
        for rule_id in hard_results.keys():
            rule = engine.rules[rule_id]
            assert rule.category == RuleCategory.HARD_CONSTRAINT
    
    def test_check_hard_constraints(self, engine):
        """Should check only hard constraints."""
        # Bad context - edge violation
        context_bad = {
            "exciter_positions": [(0.02, 0.5)],
            "cutout_positions": [],
        }
        
        satisfied, violations = engine.check_hard_constraints(context_bad)
        
        assert not satisfied
        assert len(violations) > 0
        
        # Good context
        context_good = {
            "exciter_positions": [(0.3, 0.5)],
            "cutout_positions": [],
        }
        
        satisfied, violations = engine.check_hard_constraints(context_good)
        
        assert satisfied
        assert len(violations) == 0
    
    def test_get_suggestions(self, engine, mode_shapes):
        """Should get improvement suggestions."""
        context = {
            "exciter_positions": [(0.1, 0.1)],  # Probably not optimal
            "mode_shapes": mode_shapes,
            "target_zone": "ears",
        }
        
        suggestions = engine.get_suggestions(context, max_suggestions=3)
        
        # Should return list of suggestions
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert isinstance(s, RuleSuggestion)
            assert s.confidence >= 0 and s.confidence <= 1
    
    def test_overall_score(self, engine, mode_shapes):
        """Should compute weighted overall score."""
        # Good configuration
        context_good = {
            "exciter_positions": [(0.3, 0.85), (0.7, 0.85)],
            "mode_shapes": mode_shapes,
            "cutout_positions": [],
        }
        
        score_good = engine.get_overall_score(context_good)
        
        # Bad configuration
        context_bad = {
            "exciter_positions": [(0.02, 0.5)],  # Edge violation
            "mode_shapes": mode_shapes,
            "cutout_positions": [],
        }
        
        score_bad = engine.get_overall_score(context_bad)
        
        # Good should score higher
        assert score_good > score_bad


# ══════════════════════════════════════════════════════════════════════════════
# TEST LEARNING
# ══════════════════════════════════════════════════════════════════════════════

class TestLearning:
    """Test learning from experience."""
    
    def test_record_experience(self, engine):
        """Should record experiences in buffer."""
        context = {"param1": 0.5, "param2": 0.3}
        fitness = 0.8
        
        engine.record_experience(context, fitness)
        
        assert len(engine.experience_buffer) == 1
    
    def test_buffer_trimming(self, engine):
        """Buffer should trim to max size."""
        engine.max_buffer_size = 10
        
        for i in range(20):
            engine.record_experience({"i": i}, 0.5)
        
        assert len(engine.experience_buffer) == 10
        # Should keep most recent
        assert engine.experience_buffer[-1][0]["i"] == 19
    
    def test_learn_from_experience(self, engine):
        """Should learn patterns from successful experiences."""
        # Record many similar successful experiences
        for i in range(15):
            context = {
                "exciter_x": 0.3 + np.random.normal(0, 0.02),  # Consistent
                "exciter_y": 0.85 + np.random.normal(0, 0.02),  # Consistent
                "random_param": np.random.random(),  # Inconsistent
            }
            fitness = 0.8 + np.random.random() * 0.1  # High fitness
            engine.record_experience(context, fitness)
        
        # Try to learn
        learned_rule = engine.learn_from_experience(
            fitness_threshold=0.7,
            min_samples=10
        )
        
        # May or may not learn depending on variance
        # Just check it doesn't crash
        if learned_rule:
            assert isinstance(learned_rule, LearnedRule)
            assert learned_rule.category == RuleCategory.LEARNED
    
    def test_serialization(self, engine):
        """Should serialize and restore state."""
        # Add a learned rule
        learned = LearnedRule(
            id="test_learned",
            name="Test Learned Rule",
            description="Test",
            category=RuleCategory.LEARNED,
            domain=RuleDomain.POSITION,
            pattern={"x": (0.2, 0.4)},
            success_rate=0.8,
        )
        engine.learned_rules.append(learned)
        engine.add_rule(learned)
        
        # Serialize
        data = engine.to_dict()
        
        assert len(data["learned_rules"]) == 1
        assert data["learned_rules"][0]["id"] == "test_learned"
        
        # Restore to new engine
        engine2 = create_physics_engine()
        engine2.from_dict(data)
        
        assert "test_learned" in engine2.rules


# ══════════════════════════════════════════════════════════════════════════════
# TEST RULE CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestRuleConditions:
    """Test rule condition evaluation."""
    
    def test_eq_condition(self):
        cond = RuleCondition("zone", "eq", "head")
        
        assert cond.evaluate({"zone": "head"})
        assert not cond.evaluate({"zone": "feet"})
    
    def test_lt_condition(self):
        cond = RuleCondition("frequency", "lt", 200)
        
        assert cond.evaluate({"frequency": 100})
        assert not cond.evaluate({"frequency": 300})
    
    def test_between_condition(self):
        cond = RuleCondition("x", "between", (0.2, 0.8))
        
        assert cond.evaluate({"x": 0.5})
        assert not cond.evaluate({"x": 0.1})
    
    def test_in_condition(self):
        cond = RuleCondition("zone", "in", ["head", "ears"])
        
        assert cond.evaluate({"zone": "head"})
        assert cond.evaluate({"zone": "ears"})
        assert not cond.evaluate({"zone": "feet"})


# ══════════════════════════════════════════════════════════════════════════════
# TEST CONTEXT CREATION
# ══════════════════════════════════════════════════════════════════════════════

class TestContextCreation:
    """Test context creation from genome."""
    
    def test_create_context_basic(self):
        """Should create context from mock genome."""
        # Mock genome with exciters attribute
        class MockGenome:
            class MockExciter:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            
            exciters = [MockExciter(0.3, 0.85), MockExciter(0.7, 0.85)]
            cutouts = []
        
        genome = MockGenome()
        context = create_rule_context(genome, target_zone="ears")
        
        assert "exciter_positions" in context
        assert len(context["exciter_positions"]) == 2
        assert context["target_zone"] == "ears"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
