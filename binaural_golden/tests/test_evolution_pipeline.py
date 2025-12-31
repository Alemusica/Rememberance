"""
Test suite for evolution_pipeline.py

Tests Phase 7: Integration of all Phase 1-6 components
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.evolution_pipeline import (
    # Config
    PipelineConfig,
    PipelineMode,
    # State
    PipelineState,
    PipelineResult,
    # Main class
    EvolutionPipeline,
    # Factories
    create_pipeline,
    run_quick_optimization,
    get_component_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ENUMS AND CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.mode == PipelineMode.HEADLESS
        assert config.enable_pokayoke is True
        assert config.enable_physics_rules is True
        assert config.enable_rdnn is True
        assert config.enable_ltm is True
        assert config.enable_templates is True
        assert config.population_size == 50
        assert config.n_generations == 100
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            mode=PipelineMode.INTERACTIVE,
            enable_rdnn=False,
            population_size=20,
            template_name="Binaural Audio",
        )
        
        assert config.mode == PipelineMode.INTERACTIVE
        assert config.enable_rdnn is False
        assert config.population_size == 20
        assert config.template_name == "Binaural Audio"
    
    def test_pipeline_modes(self):
        """Test all pipeline modes exist."""
        assert PipelineMode.HEADLESS
        assert PipelineMode.INTERACTIVE
        assert PipelineMode.MONITORED


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PIPELINE STATE
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineState:
    """Test PipelineState dataclass."""
    
    def test_default_state(self):
        """Test default state initialization."""
        state = PipelineState()
        
        assert state.run_id == ""
        assert state.current_generation == 0
        assert state.total_evaluations == 0
        assert state.best_fitness == float('-inf')
        assert state.best_genome is None
        assert state.stall_count == 0
    
    def test_state_mutation(self):
        """Test state can be updated."""
        state = PipelineState()
        
        state.current_generation = 10
        state.best_fitness = 0.85
        state.stall_count = 5
        
        assert state.current_generation == 10
        assert state.best_fitness == 0.85
        assert state.stall_count == 5


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PIPELINE RESULT
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineResult:
    """Test PipelineResult dataclass."""
    
    def test_create_result(self):
        """Test result creation."""
        result = PipelineResult(
            best_genome="test_genome",
            best_fitness=0.92,
            total_generations=100,
            total_evaluations=5000,
            runtime_seconds=120.5,
            converged=True,
        )
        
        assert result.best_genome == "test_genome"
        assert result.best_fitness == 0.92
        assert result.total_generations == 100
        assert result.total_evaluations == 5000
        assert result.runtime_seconds == 120.5
        assert result.converged is True
    
    def test_result_with_optional_fields(self):
        """Test result with optional knowledge."""
        result = PipelineResult(
            best_genome="genome",
            best_fitness=0.8,
            total_generations=50,
            total_evaluations=2500,
            runtime_seconds=60.0,
            converged=False,
            distilled_knowledge={"param": "value"},
            learned_rules=["rule1", "rule2"],
            fitness_history=[0.1, 0.2, 0.3],
        )
        
        assert result.distilled_knowledge == {"param": "value"}
        assert len(result.learned_rules) == 2
        assert len(result.fitness_history) == 3


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EVOLUTION PIPELINE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TestEvolutionPipelineInit:
    """Test EvolutionPipeline initialization."""
    
    def test_basic_init(self):
        """Test basic pipeline initialization."""
        config = PipelineConfig(
            enable_rdnn=False,
            enable_ltm=False,
        )
        pipeline = EvolutionPipeline(config)
        
        assert pipeline.config == config
        assert pipeline.state is not None
        assert pipeline.state.run_id != ""  # Should have UUID
    
    def test_init_with_person(self):
        """Test initialization with person."""
        mock_person = Mock()
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config, person=mock_person)
        
        assert pipeline.person is mock_person
    
    def test_initialize_method(self):
        """Test explicit initialization."""
        config = PipelineConfig(
            enable_pokayoke=True,
            enable_physics_rules=True,
            enable_rdnn=False,  # Skip for test speed
            enable_ltm=False,
            enable_templates=True,
        )
        pipeline = EvolutionPipeline(config).initialize()
        
        # Template should be loaded
        assert pipeline.template is not None
        
        # Observer should be loaded
        assert pipeline.observer is not None
        
        # Physics engine should be loaded
        assert pipeline.physics_engine is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: COMPONENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

class TestComponentLoading:
    """Test lazy loading of components."""
    
    def test_load_templates(self):
        """Test template loading."""
        config = PipelineConfig(
            enable_templates=True,
            template_name="VAT Therapy",
        )
        pipeline = EvolutionPipeline(config)
        pipeline._init_template()
        
        assert pipeline._template is not None
        assert pipeline._template.name == "VAT Therapy"
    
    def test_load_binaural_template(self):
        """Test loading binaural template."""
        config = PipelineConfig(
            enable_templates=True,
            template_name="Binaural Audio",
        )
        pipeline = EvolutionPipeline(config)
        pipeline._init_template()
        
        assert pipeline._template is not None
        assert pipeline._template.name == "Binaural Audio"
    
    def test_disabled_component_not_loaded(self):
        """Test disabled components are not loaded."""
        config = PipelineConfig(
            enable_pokayoke=False,
            enable_physics_rules=False,
            enable_rdnn=False,
            enable_ltm=False,
            enable_templates=False,
        )
        pipeline = EvolutionPipeline(config).initialize()
        
        assert pipeline._observer is None
        assert pipeline._physics_engine is None
        assert pipeline._rdnn is None
        assert pipeline._ltm_distiller is None
        assert pipeline._template is None
    
    def test_observer_modes(self):
        """Test observer initialization with different modes."""
        # Headless mode
        config = PipelineConfig(mode=PipelineMode.HEADLESS)
        pipeline = EvolutionPipeline(config)
        pipeline._init_observer()
        assert pipeline._observer is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluation:
    """Test genome evaluation."""
    
    def test_evaluation_count(self):
        """Test evaluation counter increments."""
        config = PipelineConfig(
            enable_pokayoke=False,
            enable_physics_rules=False,
            enable_rdnn=False,
            enable_ltm=False,
            enable_templates=False,
        )
        pipeline = EvolutionPipeline(config).initialize()
        
        # Create mock genome
        mock_genome = Mock()
        mock_genome.width = 0.4
        mock_genome.height = 0.6
        
        initial_count = pipeline.state.total_evaluations
        
        pipeline.evaluate_genome(mock_genome)
        
        assert pipeline.state.total_evaluations == initial_count + 1
    
    def test_population_evaluation(self):
        """Test population evaluation."""
        config = PipelineConfig(
            enable_pokayoke=False,
            enable_physics_rules=False,
            enable_rdnn=False,
            enable_ltm=False,
            enable_templates=False,
        )
        pipeline = EvolutionPipeline(config).initialize()
        
        # Create mock population
        population = [Mock(width=0.4, height=0.6) for _ in range(5)]
        
        results = pipeline.evaluate_population(population)
        
        assert len(results) == 5
        assert pipeline.state.total_evaluations == 5


# ══════════════════════════════════════════════════════════════════════════════
# TEST: GENETIC OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneticOperators:
    """Test genetic operators."""
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        population = [Mock(width=i * 0.1) for i in range(10)]
        fitnesses = [i * 0.1 for i in range(10)]
        
        selected = pipeline._tournament_select(population, fitnesses, k=3)
        
        # Should return a copy
        assert selected is not None
    
    def test_crossover(self):
        """Test crossover operation."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        parent1 = Mock(width=0.4, height=0.6)
        parent2 = Mock(width=0.5, height=0.7)
        
        child = pipeline._crossover(parent1, parent2)
        
        assert child is not None
        assert hasattr(child, 'width')
        assert hasattr(child, 'height')
    
    def test_mutation(self):
        """Test mutation operation."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        genome = Mock()
        genome.width = 0.4
        genome.height = 0.6
        
        # With rate 1.0, mutation should always happen
        mutated = pipeline._mutate(genome, rate=1.0)
        
        # Values may or may not change due to random
        assert mutated is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EXCITER GENE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestExciterGeneIntegration:
    """Test ExciterGene integration."""
    
    def test_gene_activation_tracking(self):
        """Test gene activation timeline tracking."""
        config = PipelineConfig(
            enable_pokayoke=False,
            enable_rdnn=False,
            enable_ltm=False,
        )
        pipeline = EvolutionPipeline(config)
        
        # Create mock genome with exciters that have set_phase
        mock_exciter = Mock()
        mock_exciter.set_phase = Mock()
        
        mock_genome = Mock()
        mock_genome.exciters = [mock_exciter]
        
        # Apply at early generation (should freeze positions)
        pipeline.apply_exciter_genes(mock_genome, generation=0)
        
        # Timeline might be tracked if ExciterGene is available
        # This is conditional based on imports
        # Just verify no crash occurred
    
    def test_no_crash_without_exciters(self):
        """Test no crash when genome has no exciters."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        mock_genome = Mock()
        mock_genome.exciters = None
        
        # Should not crash
        result = pipeline.apply_exciter_genes(mock_genome, generation=0)
        assert result == mock_genome


# ══════════════════════════════════════════════════════════════════════════════
# TEST: CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class TestCallbacks:
    """Test callback functionality."""
    
    def test_add_generation_callback(self):
        """Test adding generation callback."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        callback = Mock()
        pipeline.add_generation_callback(callback)
        
        assert callback in pipeline._generation_callbacks
    
    def test_add_anomaly_callback(self):
        """Test adding anomaly callback."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        callback = Mock()
        pipeline.add_anomaly_callback(callback)
        
        assert callback in pipeline._anomaly_callbacks


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_pipeline(self):
        """Test create_pipeline factory."""
        pipeline = create_pipeline(
            mode=PipelineMode.HEADLESS,
            template="VAT Therapy",
            population_size=10,
            n_generations=5,
        )
        
        assert pipeline is not None
        assert pipeline.config.mode == PipelineMode.HEADLESS
        assert pipeline.config.population_size == 10
        assert pipeline.config.n_generations == 5
    
    def test_get_component_summary(self):
        """Test component summary."""
        summary = get_component_summary()
        
        # Should have all phases
        assert "pokayoke" in summary
        assert "exciter_gene" in summary
        assert "physics_rules" in summary
        assert "rdnn" in summary
        assert "ltm" in summary
        assert "templates" in summary
        
        # Check structure
        for component in summary.values():
            assert "available" in component
    
    def test_component_summary_template_details(self):
        """Test template details in summary."""
        summary = get_component_summary()
        
        if summary["templates"]["available"]:
            assert "count" in summary["templates"]
            assert "names" in summary["templates"]
            assert "VAT Therapy" in summary["templates"]["names"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class TestEarlyStopping:
    """Test early stopping logic."""
    
    def test_no_stop_early_generations(self):
        """Test no early stop in first 20 generations."""
        config = PipelineConfig()
        pipeline = EvolutionPipeline(config)
        
        assert pipeline._should_stop(0) is False
        assert pipeline._should_stop(10) is False
        assert pipeline._should_stop(19) is False
    
    def test_stop_on_stall(self):
        """Test stop when stalled too long."""
        config = PipelineConfig(fitness_stall_threshold=5)
        pipeline = EvolutionPipeline(config)
        
        # Simulate long stall
        pipeline.state.stall_count = 50  # 5 * 3 * 3 = 45, so 50 should trigger
        
        assert pipeline._should_stop(50) is True


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests with other modules."""
    
    def test_all_phases_available(self):
        """Verify all Phase 1-6 components can be imported."""
        summary = get_component_summary()
        
        # All should be available in properly configured environment
        expected_available = ["pokayoke", "exciter_gene", "physics_rules", "templates"]
        
        for component in expected_available:
            assert summary[component]["available"], f"{component} should be available"
    
    def test_pipeline_uses_template_weights(self):
        """Test pipeline applies template weights."""
        config = PipelineConfig(
            template_name="VAT Therapy",
            enable_pokayoke=False,
            enable_physics_rules=False,
            enable_rdnn=False,
            enable_ltm=False,
        )
        pipeline = EvolutionPipeline(config).initialize()
        
        # Template should be VAT Therapy
        assert pipeline.template is not None
        assert pipeline.template.name == "VAT Therapy"
        
        # Spine weight should be dominant
        spine_weight = pipeline.template.get_spine_weight()
        ear_weight = pipeline.template.get_ear_weight()
        
        assert spine_weight > ear_weight


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

class TestProperties:
    """Test pipeline properties."""
    
    def test_observer_property(self):
        """Test observer property access."""
        config = PipelineConfig(enable_pokayoke=True)
        pipeline = EvolutionPipeline(config).initialize()
        
        assert pipeline.observer is not None
    
    def test_physics_engine_property(self):
        """Test physics_engine property access."""
        config = PipelineConfig(enable_physics_rules=True)
        pipeline = EvolutionPipeline(config).initialize()
        
        assert pipeline.physics_engine is not None
    
    def test_template_property(self):
        """Test template property access."""
        config = PipelineConfig(enable_templates=True)
        pipeline = EvolutionPipeline(config).initialize()
        
        assert pipeline.template is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: RDNN INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestRDNNIntegration:
    """Test RDNN integration (if available)."""
    
    def test_rdnn_mutation_rate_default(self):
        """Test default mutation rate when RDNN not active."""
        config = PipelineConfig(enable_rdnn=False)
        pipeline = EvolutionPipeline(config)
        
        rate = pipeline.get_rdnn_mutation_rate()
        
        # Should return None when RDNN disabled
        assert rate is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
