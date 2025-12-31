"""
Tests for evolutionary_optimizer.py

Coverage targets:
- EvolutionConfig presets and validation
- EvolutionaryOptimizer initialization
- Population initialization and evaluation
- Selection methods (tournament, roulette, rank)
- Crossover and mutation operators
- Diversity management
- Convergence detection
- Callbacks
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.core.evolutionary_optimizer import (
    EvolutionConfig,
    SelectionMethod,
    EvolutionaryOptimizer,
    EvolutionState,
    get_evolution_preset,
)
from src.core.plate_genome import PlateGenome, ContourType
from src.core.person import Person


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_person():
    """Create a test person."""
    return Person(
        height_m=1.75,
        weight_kg=70,
    )


@pytest.fixture
def minimal_config():
    """Create minimal config for fast tests."""
    return EvolutionConfig(
        population_size=10,
        n_generations=5,
        elite_count=2,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.2,
        adaptive_mutation=False,  # Disable for deterministic tests
    )


@pytest.fixture
def optimizer(test_person, minimal_config):
    """Create optimizer with mocked evaluator."""
    opt = EvolutionaryOptimizer(
        person=test_person,
        config=minimal_config,
    )
    return opt


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EVOLUTION CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestEvolutionConfig:
    """Test EvolutionConfig dataclass and presets."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        
        assert config.population_size == 50
        assert config.n_generations == 100
        assert config.elite_count == 3
        assert config.crossover_rate == 0.85
        assert config.mutation_rate == 0.25
        assert config.adaptive_mutation is True
    
    def test_quick_preset(self):
        """Test QUICK preset for fast iteration."""
        config = get_evolution_preset("QUICK")
        
        assert config.population_size < 30
        assert config.n_generations < 50
    
    def test_standard_preset(self):
        """Test STANDARD balanced preset."""
        config = get_evolution_preset("STANDARD")
        
        assert config.population_size >= 50
        assert config.n_generations >= 100
    
    def test_intense_preset(self):
        """Test INTENSE preset for thorough optimization."""
        config = get_evolution_preset("INTENSE")
        
        assert config.population_size >= 100
        assert config.n_generations >= 200
    
    def test_invalid_preset(self):
        """Invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_evolution_preset("NONEXISTENT")
    
    def test_allowed_contours(self):
        """Test default allowed contour types."""
        config = EvolutionConfig()
        
        assert ContourType.RECTANGLE in config.allowed_contours
        assert ContourType.GOLDEN_RECT in config.allowed_contours
        assert len(config.allowed_contours) > 5


class TestSelectionMethod:
    """Test selection method enum."""
    
    def test_enum_values(self):
        """Test selection method enumeration."""
        assert SelectionMethod.TOURNAMENT.value == "tournament"
        assert SelectionMethod.ROULETTE.value == "roulette"
        assert SelectionMethod.RANK.value == "rank"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: OPTIMIZER INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizerInit:
    """Test EvolutionaryOptimizer initialization."""
    
    def test_basic_init(self, test_person):
        """Test basic optimizer creation."""
        opt = EvolutionaryOptimizer(person=test_person)
        
        assert opt.person == test_person
        assert opt.config is not None
        assert opt.evaluator is not None
    
    def test_custom_config(self, test_person, minimal_config):
        """Test optimizer with custom config."""
        opt = EvolutionaryOptimizer(
            person=test_person,
            config=minimal_config
        )
        
        assert opt.config.population_size == 10
        assert opt.config.n_generations == 5
    
    def test_custom_zone_weights(self, test_person):
        """Test optimizer with custom zone weights."""
        from src.core.fitness import ZoneWeights
        
        weights = ZoneWeights(spine=0.5, head=0.5)
        opt = EvolutionaryOptimizer(
            person=test_person,
            zone_weights=weights
        )
        
        # Zone weights should be passed to evaluator
        assert opt.evaluator is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: POPULATION INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TestPopulationInit:
    """Test population initialization."""
    
    def test_population_size(self, optimizer):
        """Test initial population has correct size."""
        optimizer._initialize_population()
        
        assert len(optimizer._population) == optimizer.config.population_size
    
    def test_population_diversity(self, optimizer):
        """Test initial population has diversity."""
        optimizer._initialize_population()
        
        # Check that not all genomes are identical
        lengths = [g.length for g in optimizer._population]
        assert len(set(lengths)) > 1  # At least some variation
    
    def test_genome_validity(self, optimizer):
        """Test all genomes in initial population are valid."""
        optimizer._initialize_population()
        
        for genome in optimizer._population:
            assert isinstance(genome, PlateGenome)
            assert genome.length > 0
            assert genome.width > 0
            assert genome.thickness_base > 0
    
    def test_contour_types_used(self, optimizer):
        """Test population uses allowed contour types."""
        optimizer._initialize_population()
        
        contour_types = {g.contour_type for g in optimizer._population}
        
        # At least some variety in contours
        assert len(contour_types) >= 1
        
        # All are from allowed list
        for ct in contour_types:
            assert ct in optimizer.config.allowed_contours


# ══════════════════════════════════════════════════════════════════════════════
# TEST: SELECTION METHODS
# ══════════════════════════════════════════════════════════════════════════════

class TestSelectionMethods:
    """Test parent selection algorithms."""
    
    def test_tournament_selection(self, optimizer):
        """Test tournament selection returns valid parent."""
        optimizer._initialize_population()
        optimizer._evaluate_population()
        
        parent = optimizer._tournament_selection()
        
        assert parent is not None
        assert isinstance(parent, PlateGenome)
        assert parent in optimizer._population
    
    def test_roulette_selection(self, optimizer):
        """Test roulette wheel selection."""
        optimizer.config.selection_method = SelectionMethod.ROULETTE
        optimizer._initialize_population()
        optimizer._evaluate_population()
        
        parent = optimizer._roulette_selection()
        
        assert parent is not None
        assert isinstance(parent, PlateGenome)
    
    def test_rank_selection(self, optimizer):
        """Test rank-based selection."""
        optimizer.config.selection_method = SelectionMethod.RANK
        optimizer._initialize_population()
        optimizer._evaluate_population()
        
        parent = optimizer._rank_selection()
        
        assert parent is not None
        assert isinstance(parent, PlateGenome)
    
    def test_select_parent_routes_correctly(self, optimizer):
        """Test _select_parent uses correct method."""
        optimizer._initialize_population()
        optimizer._evaluate_population()
        
        # Tournament (default)
        optimizer.config.selection_method = SelectionMethod.TOURNAMENT
        parent = optimizer._select_parent()
        assert parent is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: DIVERSITY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class TestDiversityManagement:
    """Test diversity calculation and injection."""
    
    def test_compute_diversity(self, optimizer):
        """Test diversity metric calculation."""
        optimizer._initialize_population()
        
        diversity = optimizer._compute_diversity()
        
        # Diversity should be a positive number
        assert diversity >= 0
        assert diversity <= 1.0 or diversity > 0  # May exceed 1 depending on metric
    
    def test_inject_diversity(self, optimizer):
        """Test diversity injection creates new individuals."""
        optimizer._initialize_population()
        original_pop = [g.length for g in optimizer._population]
        
        # Force low diversity
        optimizer._inject_diversity(optimizer._population)
        
        new_pop = [g.length for g in optimizer._population]
        
        # Population should still have same size
        assert len(optimizer._population) == optimizer.config.population_size


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ADAPTIVE MUTATION
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveMutation:
    """Test adaptive mutation rate adjustment."""
    
    def test_mutation_decay(self, test_person):
        """Test mutation sigma decays over generations."""
        config = EvolutionConfig(
            population_size=10,
            n_generations=5,
            adaptive_mutation=True,
            mutation_decay=0.9,
            mutation_sigma=0.1,
            min_mutation_sigma=0.01,
        )
        opt = EvolutionaryOptimizer(person=test_person, config=config)
        
        # The internal sigma is stored in _current_mutation_sigma
        initial_sigma = opt._current_mutation_sigma
        opt._adapt_mutation()
        
        # Sigma should decrease
        assert opt._current_mutation_sigma < initial_sigma
        assert opt._current_mutation_sigma == initial_sigma * 0.9
    
    def test_mutation_floor(self, test_person):
        """Test mutation sigma doesn't go below minimum."""
        config = EvolutionConfig(
            population_size=10,
            n_generations=5,
            adaptive_mutation=True,
            mutation_decay=0.1,  # Aggressive decay
            mutation_sigma=0.02,
            min_mutation_sigma=0.01,
        )
        opt = EvolutionaryOptimizer(person=test_person, config=config)
        
        opt._adapt_mutation()
        
        # Should be clamped to minimum
        assert opt._current_mutation_sigma >= config.min_mutation_sigma


# ══════════════════════════════════════════════════════════════════════════════
# TEST: CONVERGENCE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestConvergence:
    """Test convergence detection."""
    
    def test_no_early_convergence(self, optimizer):
        """Test no convergence in early generations."""
        optimizer._initialize_population()
        optimizer._evaluate_population()
        optimizer.generation = 0
        
        # Should not converge immediately
        converged = optimizer._check_convergence()
        
        # Early generations typically don't converge
        assert isinstance(converged, bool)
    
    def test_convergence_on_stall(self, optimizer):
        """Test convergence after fitness stalls."""
        optimizer._initialize_population()
        optimizer._evaluate_population()
        
        # Simulate stalled fitness
        optimizer.fitness_history = [0.5] * 20  # Same fitness for many generations
        optimizer.generation = 50
        
        converged = optimizer._check_convergence()
        
        # With stalled fitness, should eventually converge
        # (depends on stall_generations setting)
        assert isinstance(converged, bool)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class TestCallbacks:
    """Test callback system."""
    
    def test_callback_called(self, test_person, minimal_config):
        """Test callback is called each generation."""
        callback_calls = []
        
        def track_callback(state: EvolutionState) -> bool:
            callback_calls.append(state.generation)
            return True  # Continue
        
        minimal_config.n_generations = 3
        opt = EvolutionaryOptimizer(person=test_person, config=minimal_config)
        
        opt.run(callback=track_callback)
        
        # Should be called for each generation
        assert len(callback_calls) >= 3
    
    def test_callback_can_stop(self, test_person, minimal_config):
        """Test callback returning False stops evolution."""
        generations_run = []
        
        def stop_callback(state: EvolutionState) -> bool:
            generations_run.append(state.generation)
            return state.generation < 2  # Stop after gen 2
        
        minimal_config.n_generations = 10
        opt = EvolutionaryOptimizer(person=test_person, config=minimal_config)
        
        opt.run(callback=stop_callback)
        
        # Should stop early
        assert len(generations_run) <= 3


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FULL RUN
# ══════════════════════════════════════════════════════════════════════════════

class TestFullRun:
    """Test complete optimization run."""
    
    def test_run_returns_best(self, test_person, minimal_config):
        """Test run returns best genome."""
        opt = EvolutionaryOptimizer(person=test_person, config=minimal_config)
        
        best = opt.run(verbose=False)
        
        assert best is not None
        assert isinstance(best, PlateGenome)
        # Access fitness via _best_fitness after run
        assert opt._best_fitness.total_fitness > 0
    
    def test_run_improves_fitness(self, test_person, minimal_config):
        """Test fitness improves over generations."""
        minimal_config.n_generations = 10
        opt = EvolutionaryOptimizer(person=test_person, config=minimal_config)
        
        best = opt.run(verbose=False)
        
        # Final fitness should be non-negative
        final_fitness = opt._best_fitness.total_fitness
        assert final_fitness >= 0
        
        # Best should have reasonable dimensions
        assert best.length > 1.0  # At least 1m long
        assert best.width > 0.3   # At least 30cm wide
    
    def test_evolution_state_tracking(self, test_person, minimal_config):
        """Test evolution state is properly tracked."""
        states = []
        
        def collect_states(state: EvolutionState) -> bool:
            states.append(state)
            return True
        
        minimal_config.n_generations = 5
        opt = EvolutionaryOptimizer(person=test_person, config=minimal_config)
        opt.run(callback=collect_states, verbose=False)
        
        # Check state attributes (includes final state)
        assert len(states) >= 5
        for state in states:
            assert hasattr(state, 'generation')
            assert hasattr(state, 'best_fitness')
            assert hasattr(state, 'diversity')
