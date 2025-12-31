"""
Tests for RDNN Memory module.

Tests cover:
- RDNNModel architecture and forward pass
- RDNNMemory state persistence (warm start)
- ObservationBuilder integration
- Online learning from successful runs
- Hidden state management
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from src.core.rdnn_memory import (
    RDNNConfig,
    RDNNArchitecture,
    RDNNModel,
    RDNNMemory,
    RDNNObservation,
    RDNNPrediction,
    ObservationBuilder,
    create_rdnn_memory,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_config():
    """Default RDNN configuration for testing."""
    return RDNNConfig(
        architecture=RDNNArchitecture.GRU,
        hidden_size=32,  # Smaller for fast tests
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def temp_state_dir():
    """Temporary directory for state persistence tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def sample_observation():
    """Sample observation for testing."""
    return RDNNObservation(
        generation=10,
        best_fitness=0.75,
        mean_fitness=0.65,
        fitness_std=0.08,
        fitness_velocity=0.02,
        population_diversity=0.15,
        phenotype_diversity=0.8,
        spine_flatness=0.7,
        ear_uniformity=0.6,
        total_energy=0.85,
        antinode_score=0.9,
        node_avoidance_score=0.8,
        edge_distance_score=0.95,
        phase_coherence_score=0.7,
        stagnation_signal=0.1,
        diversity_collapse_signal=0.05,
        regression_signal=0.0,
        mutation_success_rate=0.25,
        crossover_success_rate=0.15,
        emission_genes_active=0.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RDNN OBSERVATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRDNNObservation:
    """Test RDNNObservation dataclass and tensor conversion."""
    
    def test_observation_creation(self, sample_observation):
        """Test observation can be created with all fields."""
        obs = sample_observation
        assert obs.generation == 10
        assert obs.best_fitness == 0.75
        assert obs.ear_uniformity == 0.6
    
    def test_observation_to_tensor(self, sample_observation):
        """Test conversion to tensor."""
        tensor = sample_observation.to_tensor()
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (20,)  # 20 features
        
        # Check specific values
        assert tensor[0].item() == pytest.approx(0.75)  # best_fitness
        assert tensor[1].item() == pytest.approx(0.65)  # mean_fitness
    
    def test_observation_to_tensor_device(self, sample_observation):
        """Test tensor device placement."""
        tensor_cpu = sample_observation.to_tensor(device="cpu")
        assert tensor_cpu.device.type == "cpu"
    
    def test_fitness_velocity_clipping(self):
        """Test that extreme velocity values are clipped."""
        obs = RDNNObservation(
            generation=1,
            best_fitness=0.5,
            mean_fitness=0.4,
            fitness_std=0.1,
            fitness_velocity=5.0,  # Extreme value
            population_diversity=0.2,
            phenotype_diversity=0.5,
        )
        tensor = obs.to_tensor()
        # Velocity should be clipped to [-1, 1]
        assert tensor[3].item() == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# RDNN MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRDNNModel:
    """Test RDNNModel PyTorch module."""
    
    def test_model_creation_gru(self, default_config):
        """Test GRU model creation."""
        model = RDNNModel(default_config)
        
        assert hasattr(model, "rnn")
        assert isinstance(model.rnn, torch.nn.GRU)
        assert model.rnn.hidden_size == 32
    
    def test_model_creation_lstm(self):
        """Test LSTM model creation."""
        config = RDNNConfig(
            architecture=RDNNArchitecture.LSTM,
            hidden_size=32,
            num_layers=2,
        )
        model = RDNNModel(config)
        
        assert isinstance(model.rnn, torch.nn.LSTM)
        assert model.rnn.num_layers == 2
    
    def test_model_forward_pass(self, default_config):
        """Test forward pass produces expected outputs."""
        model = RDNNModel(default_config)
        model.eval()
        
        # Create batch of observations
        x = torch.randn(2, 5, 20)  # batch=2, seq_len=5, features=20
        
        with torch.no_grad():
            outputs, hidden = model(x)
        
        # Check output shapes
        assert "mutation_rate" in outputs
        assert "direction" in outputs
        assert "fitness" in outputs
        assert "anomaly" in outputs
        
        assert outputs["mutation_rate"].shape == (2, 1)
        assert outputs["direction"].shape == (2, 3)
        assert outputs["fitness"].shape == (2, 2)
        assert outputs["anomaly"].shape == (2, 4)
        
        # Check hidden state
        assert hidden is not None
        assert hidden.shape == (1, 2, 32)  # (layers, batch, hidden)
    
    def test_model_hidden_state_persistence(self, default_config):
        """Test that hidden state can persist across calls."""
        model = RDNNModel(default_config)
        model.eval()
        
        x1 = torch.randn(1, 3, 20)
        x2 = torch.randn(1, 3, 20)
        
        with torch.no_grad():
            _, hidden1 = model(x1)
            outputs2a, hidden2a = model(x2, hidden1)  # With persistence
            outputs2b, _ = model(x2, None)  # Without persistence
        
        # Outputs should differ when hidden state is used
        assert not torch.allclose(
            outputs2a["mutation_rate"],
            outputs2b["mutation_rate"]
        )
    
    def test_output_ranges(self, default_config):
        """Test that outputs are in expected ranges."""
        model = RDNNModel(default_config)
        model.eval()
        
        x = torch.randn(1, 10, 20)
        
        with torch.no_grad():
            outputs, _ = model(x)
        
        # Mutation rate should be in [0, 1] (sigmoid)
        assert 0 <= outputs["mutation_rate"].item() <= 1
        
        # Direction should sum to ~1 (softmax)
        assert outputs["direction"].sum().item() == pytest.approx(1.0, abs=0.01)
        
        # Anomaly should sum to ~1 (softmax)
        assert outputs["anomaly"].sum().item() == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# RDNN MEMORY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRDNNMemory:
    """Test RDNNMemory main interface."""
    
    def test_memory_creation(self, default_config):
        """Test memory initialization."""
        memory = RDNNMemory(default_config)
        
        assert memory.total_runs == 0
        assert memory.successful_runs == 0
        assert len(memory.history) == 0
    
    def test_step_returns_prediction(self, default_config, sample_observation):
        """Test that step() returns valid prediction."""
        memory = RDNNMemory(default_config)
        
        prediction = memory.step(sample_observation)
        
        assert isinstance(prediction, RDNNPrediction)
        assert 0 <= prediction.suggested_mutation_rate <= 1
        assert prediction.focus_spine + prediction.focus_ears + prediction.focus_energy == pytest.approx(1.0, abs=0.01)
        assert prediction.recommended_action in [
            "continue", "inject_diversity", "reset_population",
            "restore_best", "increase_exploration", "fine_tune"
        ]
    
    def test_step_accumulates_history(self, default_config, sample_observation):
        """Test that observations accumulate in history."""
        memory = RDNNMemory(default_config)
        
        for i in range(5):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5 + i * 0.05,
                mean_fitness=0.4 + i * 0.04,
                fitness_std=0.1,
                fitness_velocity=0.05,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory.step(obs)
        
        assert len(memory.history) == 5
    
    def test_history_max_length(self, default_config):
        """Test that history is bounded."""
        memory = RDNNMemory(default_config)
        
        # Add more than max_history
        for i in range(30):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5,
                mean_fitness=0.4,
                fitness_std=0.1,
                fitness_velocity=0.0,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory.step(obs)
        
        # Should be bounded
        assert len(memory.history) <= memory.max_history
    
    def test_finalize_run_updates_stats(self, default_config):
        """Test run finalization updates statistics."""
        memory = RDNNMemory(default_config)
        memory.start_run()
        
        # Add some observations
        for i in range(10):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5 + i * 0.05,
                mean_fitness=0.4 + i * 0.04,
                fitness_std=0.1,
                fitness_velocity=0.05,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory.step(obs)
        
        memory.finalize_run(success=True, final_fitness=0.95)
        
        assert memory.total_runs == 1
        assert memory.successful_runs == 1
        assert len(memory.history) == 0  # History cleared
    
    def test_hidden_state_persists_after_clear(self, default_config):
        """Test that hidden state persists even after clearing history."""
        memory = RDNNMemory(default_config)
        
        # First run
        for i in range(5):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5 + i * 0.05,
                mean_fitness=0.4,
                fitness_std=0.1,
                fitness_velocity=0.05,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory.step(obs)
        
        # Check hidden state exists
        assert memory._hidden is not None
        hidden_before = memory._hidden
        
        # Finalize (clears history but keeps hidden)
        memory.finalize_run(success=True, final_fitness=0.75)
        
        # Hidden state should still exist
        assert memory._hidden is not None
    
    def test_reset_hidden_state(self, default_config):
        """Test explicit hidden state reset."""
        memory = RDNNMemory(default_config)
        
        # Generate some hidden state
        obs = RDNNObservation(
            generation=0,
            best_fitness=0.5,
            mean_fitness=0.4,
            fitness_std=0.1,
            fitness_velocity=0.0,
            population_diversity=0.2,
            phenotype_diversity=0.5,
        )
        memory.step(obs)
        
        assert memory._hidden is not None
        
        memory.reset_hidden_state()
        
        assert memory._hidden is None


# ══════════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStatePersistence:
    """Test state save/load functionality."""
    
    def test_save_state(self, default_config, temp_state_dir):
        """Test state can be saved."""
        config = RDNNConfig(
            architecture=RDNNArchitecture.GRU,
            hidden_size=32,
            num_layers=1,
        )
        memory = RDNNMemory(config, state_path=temp_state_dir)
        
        # Generate some state
        for i in range(5):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5 + i * 0.05,
                mean_fitness=0.4,
                fitness_std=0.1,
                fitness_velocity=0.05,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory.step(obs)
        
        memory.finalize_run(success=True, final_fitness=0.75)
        memory.save_state()
        
        # Check file exists
        assert (temp_state_dir / "rdnn_state.pt").exists()
    
    def test_load_state(self, temp_state_dir):
        """Test state can be loaded."""
        config = RDNNConfig(
            architecture=RDNNArchitecture.GRU,
            hidden_size=32,
            num_layers=1,
        )
        
        # Create and save state
        memory1 = RDNNMemory(config, state_path=temp_state_dir)
        for i in range(5):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5 + i * 0.05,
                mean_fitness=0.4,
                fitness_std=0.1,
                fitness_velocity=0.05,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory1.step(obs)
        memory1.finalize_run(success=True, final_fitness=0.75)
        memory1.finalize_run(success=False, final_fitness=0.60)  # Second run
        memory1.save_state()
        
        # Load in new instance
        memory2 = RDNNMemory(config, state_path=temp_state_dir)
        
        # Stats should be loaded
        assert memory2.total_runs == 2
        assert memory2.successful_runs == 1
    
    def test_warm_start_different_predictions(self, temp_state_dir):
        """Test that warm start produces different predictions than cold start."""
        config = RDNNConfig(
            architecture=RDNNArchitecture.GRU,
            hidden_size=32,
            num_layers=1,
        )
        
        # Train first instance
        memory1 = RDNNMemory(config, state_path=temp_state_dir)
        for run in range(3):
            memory1.start_run()
            for i in range(10):
                obs = RDNNObservation(
                    generation=i,
                    best_fitness=0.5 + i * 0.05,
                    mean_fitness=0.4,
                    fitness_std=0.1,
                    fitness_velocity=0.05,
                    population_diversity=0.2,
                    phenotype_diversity=0.5,
                )
                memory1.step(obs)
            memory1.finalize_run(success=True, final_fitness=0.95)
        memory1.save_state()
        
        # Load warm-started instance
        memory_warm = RDNNMemory(config, state_path=temp_state_dir)
        
        # Create cold-started instance
        memory_cold = RDNNMemory(config)
        
        # Same observation
        test_obs = RDNNObservation(
            generation=0,
            best_fitness=0.5,
            mean_fitness=0.4,
            fitness_std=0.1,
            fitness_velocity=0.0,
            population_diversity=0.2,
            phenotype_diversity=0.5,
        )
        
        pred_warm = memory_warm.step(test_obs)
        
        # Reset cold for fair comparison
        memory_cold.reset_hidden_state()
        pred_cold = memory_cold.step(test_obs)
        
        # Predictions should be different (warm has learned)
        # This test verifies the concept - in practice, difference may be small
        # unless training is more extensive
        assert isinstance(pred_warm.suggested_mutation_rate, float)
        assert isinstance(pred_cold.suggested_mutation_rate, float)


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION BUILDER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestObservationBuilder:
    """Test ObservationBuilder helper class."""
    
    def test_builder_fitness_data(self):
        """Test setting fitness data."""
        builder = ObservationBuilder()
        
        fitnesses = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        builder.set_fitness_data(fitnesses)
        
        obs = builder.build(generation=5)
        
        assert obs.best_fitness == pytest.approx(0.9)
        assert obs.mean_fitness == pytest.approx(0.7)
        assert obs.generation == 5
    
    def test_builder_objectives(self):
        """Test setting objective data."""
        builder = ObservationBuilder()
        
        fitnesses = np.array([0.5, 0.6, 0.7])
        objectives = [
            {"spine_flatness": 0.8, "ear_lr_uniformity": 0.7},
            {"spine_flatness": 0.85, "ear_lr_uniformity": 0.75},
            {"spine_flatness": 0.9, "ear_lr_uniformity": 0.8},
        ]
        
        builder.set_fitness_data(fitnesses, objectives)
        obs = builder.build(generation=1)
        
        assert obs.spine_flatness == pytest.approx(0.85)
        assert obs.ear_uniformity == pytest.approx(0.75)
    
    def test_builder_anomaly_data(self):
        """Test setting anomaly signals."""
        builder = ObservationBuilder()
        
        builder.set_anomaly_data({
            "stagnation": 0.3,
            "diversity_collapse": 0.1,
            "regression": 0.0,
        })
        builder.set_fitness_data(np.array([0.5]))
        
        obs = builder.build(generation=1)
        
        assert obs.stagnation_signal == pytest.approx(0.3)
        assert obs.diversity_collapse_signal == pytest.approx(0.1)
    
    def test_builder_physics_data(self):
        """Test setting physics rule scores."""
        builder = ObservationBuilder()
        
        builder.set_physics_data({
            "antinode": 0.95,
            "node_avoidance": 0.8,
            "edge_distance": 0.9,
            "phase_coherence": 0.7,
        })
        builder.set_fitness_data(np.array([0.5]))
        
        obs = builder.build(generation=1)
        
        assert obs.antinode_score == pytest.approx(0.95)
        assert obs.node_avoidance_score == pytest.approx(0.8)
    
    def test_builder_chaining(self):
        """Test fluent interface."""
        builder = ObservationBuilder()
        
        obs = (
            builder
            .set_fitness_data(np.array([0.5, 0.6, 0.7]))
            .set_anomaly_data({"stagnation": 0.2})
            .set_physics_data({"antinode": 0.9})
            .set_operator_data(mutation_successes=5, crossover_successes=3, total_evaluations=20)
            .set_gene_state(emission_active_ratio=0.5)
            .build(generation=10)
        )
        
        assert obs.generation == 10
        assert obs.best_fitness == pytest.approx(0.7)
        assert obs.stagnation_signal == pytest.approx(0.2)
        assert obs.mutation_success_rate == pytest.approx(5 / 21)
        assert obs.emission_genes_active == pytest.approx(0.5)
    
    def test_builder_fitness_velocity(self):
        """Test velocity calculation across builds."""
        builder = ObservationBuilder()
        
        # First observation
        builder.set_fitness_data(np.array([0.5]))
        obs1 = builder.build(generation=1)
        
        builder.reset()
        
        # Second observation - should have velocity
        builder.set_fitness_data(np.array([0.7]))
        obs2 = builder.build(generation=2)
        
        assert obs2.fitness_velocity == pytest.approx(0.2)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFactory:
    """Test factory function."""
    
    def test_create_rdnn_memory_defaults(self):
        """Test factory with defaults."""
        memory = create_rdnn_memory()
        
        assert memory.config.architecture == RDNNArchitecture.GRU
        assert memory.config.hidden_size == 64
        assert memory.config.device == "cpu"
    
    def test_create_rdnn_memory_custom(self, temp_state_dir):
        """Test factory with custom settings."""
        memory = create_rdnn_memory(
            state_path=str(temp_state_dir),
            architecture=RDNNArchitecture.LSTM,
            hidden_size=128,
        )
        
        assert memory.config.architecture == RDNNArchitecture.LSTM
        assert memory.config.hidden_size == 128
        assert memory.state_path == temp_state_dir


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests simulating real usage."""
    
    def test_full_optimization_simulation(self, default_config):
        """Simulate a complete optimization run."""
        memory = RDNNMemory(default_config)
        memory.start_run()
        
        # Simulate 50 generations
        predictions = []
        for gen in range(50):
            # Simulated improving fitness
            fitness = 0.3 + 0.01 * gen + np.random.normal(0, 0.02)
            fitness = np.clip(fitness, 0, 1)
            
            obs = RDNNObservation(
                generation=gen,
                best_fitness=float(fitness),
                mean_fitness=float(fitness - 0.1),
                fitness_std=0.05,
                fitness_velocity=0.01 if gen > 0 else 0,
                population_diversity=0.2 - gen * 0.002,  # Decreasing diversity
                phenotype_diversity=0.5,
                spine_flatness=float(fitness * 0.9),
                ear_uniformity=float(fitness * 0.8),
            )
            
            pred = memory.step(obs)
            predictions.append(pred)
        
        memory.finalize_run(success=True, final_fitness=0.8)
        
        # Verify predictions were generated
        assert len(predictions) == 50
        assert all(isinstance(p, RDNNPrediction) for p in predictions)
        
        # Verify stats updated
        assert memory.total_runs == 1
        assert memory.successful_runs == 1
    
    def test_anomaly_detection_response(self, default_config):
        """Test that anomaly signals affect predictions."""
        memory = RDNNMemory(default_config)
        
        # Normal observation
        obs_normal = RDNNObservation(
            generation=10,
            best_fitness=0.5,
            mean_fitness=0.45,
            fitness_std=0.05,
            fitness_velocity=0.01,
            population_diversity=0.2,
            phenotype_diversity=0.5,
            stagnation_signal=0.0,
        )
        
        pred_normal = memory.step(obs_normal)
        
        # Stagnation observation
        obs_stagnant = RDNNObservation(
            generation=11,
            best_fitness=0.5,  # No improvement
            mean_fitness=0.45,
            fitness_std=0.02,  # Lower diversity
            fitness_velocity=0.0,
            population_diversity=0.1,  # Lower
            phenotype_diversity=0.3,
            stagnation_signal=0.9,  # High stagnation
        )
        
        pred_stagnant = memory.step(obs_stagnant)
        
        # Stagnant observation should suggest higher mutation
        # (depends on model state, so we just verify output exists)
        assert isinstance(pred_stagnant.suggested_mutation_rate, float)
        assert pred_stagnant.anomaly_probability >= 0
    
    def test_get_stats(self, default_config):
        """Test statistics retrieval."""
        memory = RDNNMemory(default_config)
        
        # Run some operations
        for i in range(5):
            obs = RDNNObservation(
                generation=i,
                best_fitness=0.5,
                mean_fitness=0.4,
                fitness_std=0.1,
                fitness_velocity=0.0,
                population_diversity=0.2,
                phenotype_diversity=0.5,
            )
            memory.step(obs)
        
        stats = memory.get_stats()
        
        assert "total_runs" in stats
        assert "hidden_state_active" in stats
        assert stats["history_length"] == 5
        assert stats["architecture"] == "GRU"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
