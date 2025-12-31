"""
Tests for PokayokeObserver - Phase 1 of Action Plan 3.0
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.pokayoke_observer import (
    PokayokeObserver, AnomalyType, UserAction, AnomalyContext, UserDecision,
    ObserverState, HeadlessHandler, CLIHandler, create_observer
)
from core.analysis_config import (
    ObserverConfig, GeneActivationConfig, GenePhase, ActivationTrigger
)


class TestAnomalyDetection:
    """Test anomaly detection logic."""
    
    def test_stagnation_detection(self):
        """Observer should detect stagnation after N generations without improvement."""
        config = ObserverConfig(
            stagnation_threshold=5,  # Trigger after 5 gens
            auto_adjust=True,  # Use headless for testing
        )
        observer = PokayokeObserver(config=config, handler=HeadlessHandler(config))
        
        # Simulate 10 generations with no improvement
        result = None
        for gen in range(10):
            result = observer.on_generation(
                generation=gen,
                population_fitness=[0.5, 0.4, 0.3],  # Same fitness
                best_genome=None,
                population_diversity=0.5,
            )
            if result and result.action != UserAction.RETRY:
                break
        
        # Should have detected stagnation at gen 5
        assert observer.state.pauses_count > 0, "Should have detected anomaly"
    
    def test_diversity_collapse_detection(self):
        """Observer should detect when population diversity collapses."""
        config = ObserverConfig(
            diversity_collapse_threshold=0.2,
            auto_adjust=True,
        )
        observer = PokayokeObserver(config=config, handler=HeadlessHandler(config))
        
        # Low diversity should trigger
        result = observer.on_generation(
            generation=10,
            population_fitness=[0.5, 0.5, 0.5, 0.5],
            best_genome=None,
            population_diversity=0.05,  # Very low diversity
        )
        
        assert result is not None, "Should detect diversity collapse"
        # Check it was captured in pauses_count
        assert observer.state.pauses_count > 0
    
    def test_no_anomaly_healthy_evolution(self):
        """No anomaly should be detected when evolution is healthy."""
        config = ObserverConfig(
            stagnation_threshold=10,
            diversity_collapse_threshold=0.1,
            auto_adjust=True,
        )
        observer = PokayokeObserver(config=config, handler=HeadlessHandler(config))
        
        # Simulate healthy improving evolution
        for gen in range(5):
            result = observer.on_generation(
                generation=gen,
                population_fitness=[0.3 + gen * 0.1, 0.2 + gen * 0.1],  # Improving
                best_genome=None,
                population_diversity=0.4,  # Healthy diversity
            )
            assert result is None, f"Should not detect anomaly at gen {gen}"
        
        assert observer.state.pauses_count == 0


class TestGeneActivation:
    """Test gene activation phase transitions."""
    
    def test_seed_to_bloom_activation(self):
        """Observer should suggest ACTIVATE_GENES when positions converge."""
        gene_config = GeneActivationConfig(
            initial_phase=GenePhase.SEED,
            position_convergence_sigma=0.02,
            position_convergence_generations=3,  # Fixed: correct field name
        )
        config = ObserverConfig(
            stagnation_threshold=100,  # Disable stagnation detection
            auto_adjust=True,
        )
        observer = PokayokeObserver(
            config=config,
            gene_config=gene_config,
            handler=HeadlessHandler(config)
        )
        
        # Start in SEED phase
        assert observer.get_current_phase() == GenePhase.SEED
        
        # Simulate converged positions for several generations
        for gen in range(5):
            result = observer.on_generation(
                generation=gen,
                population_fitness=[0.5, 0.4],
                best_genome=None,
                population_diversity=0.3,
                position_sigma=0.01,  # Very converged
            )
            
            if result and result.action == UserAction.ACTIVATE_GENES:
                # Headless mode auto-selects suggested action
                break
        
        # Should have transitioned to BLOOM
        assert observer.get_current_phase() == GenePhase.BLOOM
    
    def test_phase_starts_as_seed(self):
        """Initial phase should be SEED."""
        observer = create_observer(mode="headless")
        assert observer.get_current_phase() == GenePhase.SEED


class TestUserDecision:
    """Test user decision handling."""
    
    def test_headless_auto_select(self):
        """Headless handler should auto-select first suggested action."""
        config = ObserverConfig(auto_adjust=True)
        handler = HeadlessHandler(config)
        
        context = AnomalyContext(
            anomaly_type=AnomalyType.STAGNATION,
            generation=50,
            severity=0.5,
            current_fitness=0.3,
            best_fitness_ever=0.35,
            fitness_velocity=-0.001,
            population_diversity=0.2,
            stagnation_generations=10,
            suggested_actions=[
                UserAction.INJECT_DIVERSITY,
                UserAction.INCREASE_MUTATION,
            ],
            explanation="Test stagnation"
        )
        
        decision = handler.present_anomaly(context, list(UserAction))
        
        assert decision.action == UserAction.INJECT_DIVERSITY
        assert decision.reason == "headless_auto"
    
    def test_decision_records_in_state(self):
        """User decisions should be recorded in state."""
        config = ObserverConfig(
            stagnation_threshold=3,
            auto_adjust=True,
        )
        observer = PokayokeObserver(config=config, handler=HeadlessHandler(config))
        
        # Trigger anomaly
        for gen in range(5):
            observer.on_generation(
                generation=gen,
                population_fitness=[0.3, 0.3, 0.3],
                best_genome=None,
                population_diversity=0.3,
            )
        
        stats = observer.get_statistics()
        assert stats["pauses_count"] > 0
        assert len(stats["decisions"]) > 0


class TestObserverFactory:
    """Test observer factory function."""
    
    def test_create_headless(self):
        """Should create headless observer."""
        observer = create_observer(mode="headless")
        assert observer.config.auto_adjust is True
    
    def test_create_cli(self):
        """Should create CLI observer."""
        observer = create_observer(mode="cli")
        assert isinstance(observer.handler, CLIHandler)
    
    def test_reset_clears_state(self):
        """Reset should clear observer state."""
        observer = create_observer(mode="headless")
        
        # Add some state
        observer.state.pauses_count = 5
        observer.state.best_fitness_ever = 0.9
        observer.state.current_phase = GenePhase.BLOOM
        
        observer.reset()
        
        assert observer.state.pauses_count == 0
        assert observer.state.best_fitness_ever == float('-inf')
        assert observer.state.current_phase == GenePhase.SEED


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_evolution_scenario(self):
        """Simulate realistic evolution with phase transitions."""
        observer = create_observer(mode="headless")
        
        decisions_made = []
        
        # Phase 1: SEED - optimize positions
        for gen in range(20):
            fitness_improving = 0.3 + gen * 0.01
            position_sigma = 0.1 - gen * 0.004  # Converging
            
            result = observer.on_generation(
                generation=gen,
                population_fitness=[fitness_improving, fitness_improving - 0.05],
                best_genome=None,
                population_diversity=0.4 - gen * 0.01,
                position_sigma=max(0.01, position_sigma),
            )
            
            if result:
                decisions_made.append((gen, result.action.name))
        
        # Should have at least one decision
        stats = observer.get_statistics()
        print(f"Decisions made: {stats['decisions']}")
        print(f"Final phase: {stats['current_phase']}")
        
        # Verify observer tracked everything
        assert stats["best_fitness_ever"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
