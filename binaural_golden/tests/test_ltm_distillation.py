"""
Tests for LTM Distillation module.

Tests cover:
- ExperienceStatistics aggregation
- Knowledge distillation (parameters, rules, RDNN priors)
- Knowledge persistence
- Domain matching
- Integration with RDNN
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List

from src.core.ltm_distillation import (
    LTMDistiller,
    DistilledKnowledge,
    DistillationType,
    ExperienceStatistics,
    create_distiller,
    distill_and_apply,
)


# ══════════════════════════════════════════════════════════════════════════════
# MOCK CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockArchiveEntry:
    """Mock ExperienceArchiveEntry for testing."""
    run_id: str
    final_fitness: float
    convergence_generation: int
    outcome: str
    target_person: Dict[str, Any]
    zone_weights: Dict[str, float]
    best_genome_summary: Dict[str, Any]
    successful_mutations: List[str]
    failed_mutations: List[str]
    final_objectives: Dict[str, float] = None
    
    def __post_init__(self):
        if self.final_objectives is None:
            self.final_objectives = {}


def create_mock_archive(n_runs: int = 20, success_rate: float = 0.7) -> List[MockArchiveEntry]:
    """Create mock archive for testing."""
    archive = []
    
    for i in range(n_runs):
        is_success = np.random.random() < success_rate
        
        # Vary person height
        height = 1.6 + np.random.random() * 0.3  # 1.6 - 1.9m
        
        # Exciter position correlated with height (for testing)
        exciter_y = 0.3 + (height - 1.6) * 0.5 + np.random.normal(0, 0.05)
        exciter_x = 0.4 + np.random.normal(0, 0.1)
        
        entry = MockArchiveEntry(
            run_id=f"run_{i:03d}",
            final_fitness=0.7 + np.random.random() * 0.25 if is_success else 0.4 + np.random.random() * 0.3,
            convergence_generation=30 + int(np.random.exponential(20)),
            outcome="success" if is_success else "failure",
            target_person={
                "height_m": height,
                "weight_kg": 60 + np.random.random() * 30,
            },
            zone_weights={
                "spine": 0.6 + np.random.random() * 0.2,
                "head": 0.2 + np.random.random() * 0.1,
            },
            best_genome_summary={
                "contour_type": np.random.choice(["rectangle", "ellipse", "superellipse"]),
                "exciter_x_mean": exciter_x,
                "exciter_y_mean": exciter_y,
                "cutout_positions": [(np.random.random(), np.random.random()) for _ in range(2)],
            },
            successful_mutations=["position_shift", "phase_adjust"] if is_success else ["position_shift"],
            failed_mutations=["random_reset"] if not is_success else [],
        )
        archive.append(entry)
    
    return archive


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_storage():
    """Temporary directory for storage tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def mock_archive():
    """Mock archive with 20 runs."""
    np.random.seed(42)  # Reproducible
    return create_mock_archive(20, 0.7)


@pytest.fixture
def small_archive():
    """Small archive (3 runs) - below minimum."""
    np.random.seed(42)
    return create_mock_archive(3, 0.8)


@pytest.fixture
def distiller():
    """Basic distiller instance."""
    return LTMDistiller(min_runs_for_distillation=5)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE STATISTICS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestExperienceStatistics:
    """Test experience statistics aggregation."""
    
    def test_analyze_empty_archive(self, distiller):
        """Test analyzing empty archive."""
        stats = distiller.analyze_archive([])
        
        assert stats.total_runs == 0
        assert stats.fitness_mean == 0.0
    
    def test_analyze_basic_stats(self, distiller, mock_archive):
        """Test basic statistics from archive."""
        stats = distiller.analyze_archive(mock_archive)
        
        assert stats.total_runs == 20
        assert stats.successful_runs > 0
        assert 0 < stats.fitness_mean < 1
        assert stats.fitness_std >= 0
        assert stats.convergence_gen_mean > 0
    
    def test_analyze_zone_map(self, distiller, mock_archive):
        """Test zone fitness map creation."""
        stats = distiller.analyze_archive(mock_archive)
        
        # Should have zone entries
        assert len(stats.zone_fitness_map) > 0
        
        for zone_key, zone_stats in stats.zone_fitness_map.items():
            assert "mean" in zone_stats
            assert "std" in zone_stats
            assert "count" in zone_stats
    
    def test_analyze_exciter_heatmap(self, distiller, mock_archive):
        """Test exciter heatmap creation."""
        stats = distiller.analyze_archive(mock_archive)
        
        assert stats.exciter_heatmap is not None
        assert stats.exciter_heatmap.shape == (10, 10)
        assert stats.exciter_heatmap.sum() > 0  # Has values
    
    def test_analyze_mutation_rates(self, distiller, mock_archive):
        """Test mutation success rate calculation."""
        stats = distiller.analyze_archive(mock_archive)
        
        assert len(stats.mutation_success_rates) > 0
        
        for mut, rate in stats.mutation_success_rates.items():
            assert 0 <= rate <= 1
    
    def test_analyze_contour_fitness(self, distiller, mock_archive):
        """Test contour type effectiveness."""
        stats = distiller.analyze_archive(mock_archive)
        
        assert len(stats.contour_fitness) > 0
        
        for contour, fitness in stats.contour_fitness.items():
            assert 0 < fitness < 1
    
    def test_height_exciter_correlation(self, distiller, mock_archive):
        """Test height-exciter correlation calculation."""
        stats = distiller.analyze_archive(mock_archive)
        
        # Should have some correlation (we designed mock data this way)
        assert stats.height_exciter_correlation != 0


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER DISTILLATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestParameterDistillation:
    """Test parameter prior distillation."""
    
    def test_distill_insufficient_runs(self, distiller, small_archive):
        """Test that insufficient runs returns empty list."""
        stats = distiller.analyze_archive(small_archive)
        priors = distiller.distill_parameter_priors(stats)
        
        assert len(priors) == 0
    
    def test_distill_mutation_rate(self, distiller, mock_archive):
        """Test mutation rate prior extraction."""
        stats = distiller.analyze_archive(mock_archive)
        priors = distiller.distill_parameter_priors(stats)
        
        # Find mutation rate prior
        mutation_prior = next(
            (p for p in priors if p.knowledge_id == "param_mutation_rate"),
            None
        )
        
        assert mutation_prior is not None
        assert "initial_mutation_rate" in mutation_prior.parameters
        assert 0 < mutation_prior.parameters["initial_mutation_rate"] < 1
    
    def test_distill_population_size(self, distiller, mock_archive):
        """Test population size prior."""
        stats = distiller.analyze_archive(mock_archive)
        priors = distiller.distill_parameter_priors(stats)
        
        pop_prior = next(
            (p for p in priors if p.knowledge_id == "param_population_size"),
            None
        )
        
        assert pop_prior is not None
        assert "population_size" in pop_prior.parameters
        assert pop_prior.parameters["population_size"] > 0
    
    def test_distill_contour_preference(self, distiller, mock_archive):
        """Test contour preference extraction."""
        stats = distiller.analyze_archive(mock_archive)
        priors = distiller.distill_parameter_priors(stats)
        
        contour_prior = next(
            (p for p in priors if p.knowledge_id == "param_contour_preference"),
            None
        )
        
        assert contour_prior is not None
        assert "preferred_contour" in contour_prior.parameters
    
    def test_prior_confidence_bounds(self, distiller, mock_archive):
        """Test that confidence is bounded [0, 1]."""
        stats = distiller.analyze_archive(mock_archive)
        priors = distiller.distill_parameter_priors(stats)
        
        for prior in priors:
            assert 0 <= prior.confidence <= 1


# ══════════════════════════════════════════════════════════════════════════════
# RULE DISTILLATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRuleDistillation:
    """Test rule candidate distillation."""
    
    def test_distill_rules_insufficient_runs(self, distiller, small_archive):
        """Test that insufficient runs returns empty."""
        stats = distiller.analyze_archive(small_archive)
        rules = distiller.distill_rule_candidates(stats)
        
        assert len(rules) == 0
    
    def test_distill_exciter_hot_zones(self, distiller, mock_archive):
        """Test exciter hot zone rule extraction."""
        stats = distiller.analyze_archive(mock_archive)
        rules = distiller.distill_rule_candidates(stats)
        
        hot_zone_rule = next(
            (r for r in rules if r.knowledge_id == "rule_exciter_hot_zones"),
            None
        )
        
        assert hot_zone_rule is not None
        assert hot_zone_rule.distillation_type == DistillationType.RULE_CANDIDATE
        assert "hot_regions" in hot_zone_rule.parameters
    
    def test_distill_height_exciter_rule(self, distiller, mock_archive):
        """Test height-exciter correlation rule."""
        stats = distiller.analyze_archive(mock_archive)
        rules = distiller.distill_rule_candidates(stats)
        
        # May or may not exist depending on correlation strength
        height_rule = next(
            (r for r in rules if r.knowledge_id == "rule_height_exciter"),
            None
        )
        
        if height_rule is not None:
            assert "correlation" in height_rule.parameters
    
    def test_distill_zone_rules(self, distiller, mock_archive):
        """Test zone-specific rules."""
        stats = distiller.analyze_archive(mock_archive)
        rules = distiller.distill_rule_candidates(stats)
        
        zone_rules = [r for r in rules if r.knowledge_id.startswith("rule_zone_")]
        
        for rule in zone_rules:
            assert "expected_fitness" in rule.parameters


# ══════════════════════════════════════════════════════════════════════════════
# RDNN PRIOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRDNNPriors:
    """Test RDNN hidden state prior distillation."""
    
    def test_distill_rdnn_prior(self, distiller, mock_archive):
        """Test RDNN prior creation."""
        stats = distiller.analyze_archive(mock_archive)
        prior = distiller.distill_rdnn_priors(stats, hidden_size=32)
        
        assert prior.distillation_type == DistillationType.HIDDEN_STATE_PRIOR
        assert "hidden_prior" in prior.parameters
        assert len(prior.parameters["hidden_prior"]) == 32
    
    def test_rdnn_prior_different_sizes(self, distiller, mock_archive):
        """Test RDNN prior with different hidden sizes."""
        stats = distiller.analyze_archive(mock_archive)
        
        for size in [16, 32, 64, 128]:
            prior = distiller.distill_rdnn_priors(stats, hidden_size=size)
            assert len(prior.parameters["hidden_prior"]) == size
    
    def test_rdnn_prior_values_bounded(self, distiller, mock_archive):
        """Test that prior values are bounded (tanh output)."""
        stats = distiller.analyze_archive(mock_archive)
        prior = distiller.distill_rdnn_priors(stats, hidden_size=64)
        
        values = np.array(prior.parameters["hidden_prior"])
        assert np.all(np.abs(values) <= 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# MUTATION SCHEDULE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMutationSchedule:
    """Test mutation schedule distillation."""
    
    def test_distill_schedule(self, distiller, mock_archive):
        """Test mutation schedule creation."""
        stats = distiller.analyze_archive(mock_archive)
        schedule = distiller.distill_mutation_schedule(stats)
        
        assert schedule.distillation_type == DistillationType.MUTATION_SCHEDULE
        assert "schedule" in schedule.parameters
        assert len(schedule.parameters["schedule"]) > 0
    
    def test_schedule_phases(self, distiller, mock_archive):
        """Test that schedule has decreasing mutation rates."""
        stats = distiller.analyze_archive(mock_archive)
        schedule = distiller.distill_mutation_schedule(stats)
        
        phases = schedule.parameters["schedule"]
        rates = [p["mutation_rate"] for p in phases]
        
        # Should generally decrease
        assert rates[0] >= rates[-1]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN MATCHING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainMatching:
    """Test knowledge domain matching."""
    
    def test_matches_empty_conditions(self):
        """Test that empty conditions always match."""
        knowledge = DistilledKnowledge(
            knowledge_id="test",
            distillation_type=DistillationType.PARAMETER_PRIOR,
            description="test",
            parameters={},
            source_runs=10,
            source_fitness_mean=0.8,
            source_fitness_std=0.1,
            domain_conditions={},
        )
        
        assert knowledge.matches_domain(person_height_m=1.5)
        assert knowledge.matches_domain(person_height_m=2.0)
    
    def test_matches_height_range(self):
        """Test height range matching."""
        knowledge = DistilledKnowledge(
            knowledge_id="test",
            distillation_type=DistillationType.PARAMETER_PRIOR,
            description="test",
            parameters={},
            source_runs=10,
            source_fitness_mean=0.8,
            source_fitness_std=0.1,
            domain_conditions={"person_height_m": (1.6, 1.8)},
        )
        
        assert knowledge.matches_domain(person_height_m=1.7)
        assert not knowledge.matches_domain(person_height_m=1.5)
        assert not knowledge.matches_domain(person_height_m=1.9)
    
    def test_matches_multiple_conditions(self):
        """Test multiple condition matching."""
        knowledge = DistilledKnowledge(
            knowledge_id="test",
            distillation_type=DistillationType.PARAMETER_PRIOR,
            description="test",
            parameters={},
            source_runs=10,
            source_fitness_mean=0.8,
            source_fitness_std=0.1,
            domain_conditions={
                "person_height_m": (1.6, 1.9),
                "spine_weight": (0.5, 0.8),
            },
        )
        
        assert knowledge.matches_domain(person_height_m=1.75, spine_weight=0.6)
        assert not knowledge.matches_domain(person_height_m=1.75, spine_weight=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestVerification:
    """Test knowledge verification updates."""
    
    def test_update_verification_success(self):
        """Test verification success updates."""
        knowledge = DistilledKnowledge(
            knowledge_id="test",
            distillation_type=DistillationType.PARAMETER_PRIOR,
            description="test",
            parameters={},
            source_runs=10,
            source_fitness_mean=0.8,
            source_fitness_std=0.1,
            confidence=0.5,
        )
        
        knowledge.update_verification(success=True)
        
        assert knowledge.verification_count == 1
        assert knowledge.verification_successes == 1
        assert knowledge.confidence > 0.5  # Should increase
    
    def test_update_verification_failure(self):
        """Test verification failure updates."""
        knowledge = DistilledKnowledge(
            knowledge_id="test",
            distillation_type=DistillationType.PARAMETER_PRIOR,
            description="test",
            parameters={},
            source_runs=10,
            source_fitness_mean=0.8,
            source_fitness_std=0.1,
            confidence=0.8,
        )
        
        knowledge.update_verification(success=False)
        
        assert knowledge.verification_count == 1
        assert knowledge.verification_successes == 0
        assert knowledge.confidence < 0.8  # Should decrease


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    """Test knowledge persistence."""
    
    def test_save_knowledge(self, mock_archive, temp_storage):
        """Test saving knowledge to disk."""
        distiller = LTMDistiller(
            min_runs_for_distillation=5,
            storage_path=temp_storage,
        )
        
        stats = distiller.analyze_archive(mock_archive)
        distiller.distill_parameter_priors(stats)
        distiller.save_knowledge()
        
        assert (temp_storage / "distilled_knowledge.json").exists()
    
    def test_load_knowledge(self, mock_archive, temp_storage):
        """Test loading knowledge from disk."""
        # Create and save
        distiller1 = LTMDistiller(
            min_runs_for_distillation=5,
            storage_path=temp_storage,
        )
        stats = distiller1.analyze_archive(mock_archive)
        distiller1.distill_parameter_priors(stats)
        distiller1.save_knowledge()
        
        n_items = len(distiller1.knowledge_base)
        
        # Load in new instance
        distiller2 = LTMDistiller(
            min_runs_for_distillation=5,
            storage_path=temp_storage,
        )
        
        assert len(distiller2.knowledge_base) == n_items


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests."""
    
    def test_full_distillation_workflow(self, mock_archive, temp_storage):
        """Test complete distillation workflow."""
        distiller = LTMDistiller(
            min_runs_for_distillation=5,
            storage_path=temp_storage,
        )
        
        # Analyze
        stats = distiller.analyze_archive(mock_archive)
        
        # Distill all types
        priors = distiller.distill_parameter_priors(stats)
        rules = distiller.distill_rule_candidates(stats)
        rdnn_prior = distiller.distill_rdnn_priors(stats)
        schedule = distiller.distill_mutation_schedule(stats)
        
        # All should produce results
        assert len(priors) > 0
        assert len(rules) > 0
        assert rdnn_prior is not None
        assert schedule is not None
        
        # Save
        distiller.save_knowledge()
        assert (temp_storage / "distilled_knowledge.json").exists()
    
    def test_get_applicable_knowledge(self, mock_archive):
        """Test filtering applicable knowledge."""
        distiller = LTMDistiller(min_runs_for_distillation=5)
        
        stats = distiller.analyze_archive(mock_archive)
        distiller.distill_parameter_priors(stats)
        distiller.distill_rule_candidates(stats)
        
        # Get applicable knowledge
        applicable = distiller.get_applicable_knowledge(
            person_height_m=1.75,
            spine_weight=0.7,
            min_confidence=0.3,  # Lower threshold for testing
        )
        
        assert len(applicable) > 0
        # Should be sorted by confidence
        for i in range(len(applicable) - 1):
            assert applicable[i].confidence >= applicable[i + 1].confidence
    
    def test_factory_function(self, temp_storage):
        """Test create_distiller factory."""
        distiller = create_distiller(
            storage_path=str(temp_storage),
            min_runs=3,
        )
        
        assert distiller.min_runs == 3
        assert distiller.storage_path == temp_storage


# ══════════════════════════════════════════════════════════════════════════════
# DISTILL AND APPLY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDistillAndApply:
    """Test one-shot distill_and_apply function."""
    
    def test_distill_and_apply_basic(self, mock_archive, temp_storage):
        """Test basic distill_and_apply."""
        results = distill_and_apply(
            ltm_archive=mock_archive,
            storage_path=str(temp_storage),
        )
        
        assert "parameter_priors" in results
        assert "rule_candidates" in results
        assert "mutation_schedule" in results
        
        assert len(results["parameter_priors"]) > 0
    
    def test_distill_and_apply_empty_archive(self, temp_storage):
        """Test with empty archive."""
        results = distill_and_apply(
            ltm_archive=[],
            storage_path=str(temp_storage),
        )
        
        # Should not crash, just empty results
        assert len(results["parameter_priors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
