"""
Tests for modular scorers (extracted from fitness.py).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from src.core.scorers import (
    Scorer, ScorerResult, ScorerBase,
    ZoneFlatnessScorer,
    EarUniformityScorer,
    SpineCouplingScorer,
    StructuralScorer,
    ManufacturabilityScorer,
    ExciterScorer,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: SCORER PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

class TestScorerProtocol:
    """Test scorer protocol and base class."""
    
    def test_scorer_result_creation(self):
        """Test ScorerResult dataclass."""
        result = ScorerResult(
            score=0.85,
            name="test_scorer",
            weight=1.0,
            details={'metric': 123}
        )
        assert result.score == 0.85
        assert result.name == "test_scorer"
        assert result.weighted_score() == 0.85
    
    def test_scorer_result_weighted(self):
        """Test weighted score calculation."""
        result = ScorerResult(score=0.8, name="test", weight=0.5)
        assert result.weighted_score() == 0.4
    
    def test_scorer_base_utilities(self):
        """Test ScorerBase utility methods."""
        base = ScorerBase()
        
        # Normalize
        assert base._safe_normalize(50, 0, 100) == 0.5
        assert base._safe_normalize(0, 0, 100) == 0.0
        assert base._safe_normalize(100, 0, 100) == 1.0
        assert base._safe_normalize(150, 0, 100) == 1.0  # Clamped
        
        # CV
        data = np.array([1, 1, 1, 1])
        assert base._coefficient_of_variation(data) == 0.0
        
        # dB conversion
        assert base._linear_to_db(1.0) == 0.0
        assert abs(base._db_to_linear(0.0) - 1.0) < 0.001


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ZONE FLATNESS SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneFlatnessScorer:
    """Test zone flatness scoring."""
    
    def test_perfect_flat_response(self):
        """Perfectly flat response should score near 1.0."""
        scorer = ZoneFlatnessScorer()
        
        # Create perfectly flat response (all values = 1.0)
        flat_response = np.ones((5, 50))
        
        context = {
            'spine_response': flat_response,
            'head_response': flat_response,
            'zone_weights': {'spine': 0.7, 'head': 0.3}
        }
        
        result = scorer.score(Mock(), context)
        
        # Flat response should score high (but may not be exactly 1.0 due to dB conversion)
        assert result.score > 0.8
        assert 'spine_flatness' in result.details
        assert 'head_flatness' in result.details
    
    def test_varying_response(self):
        """Response with large variation should score lower."""
        scorer = ZoneFlatnessScorer()
        
        # Create response with 20dB variation
        varying = np.linspace(0.1, 10.0, 50)  # 40dB range
        response = np.tile(varying, (5, 1))
        
        context = {
            'spine_response': response,
            'head_response': response,
            'zone_weights': {'spine': 0.7, 'head': 0.3}
        }
        
        result = scorer.score(Mock(), context)
        
        # Large variation should score low
        assert result.score < 0.5
    
    def test_zone_weights_applied(self):
        """Zone weights should affect final score."""
        scorer = ZoneFlatnessScorer()
        
        # Spine flat, head varying
        flat = np.ones((5, 50))
        varying = np.linspace(0.1, 10.0, 50)
        varying_response = np.tile(varying, (5, 1))
        
        context = {
            'spine_response': flat,
            'head_response': varying_response,
            'zone_weights': {'spine': 0.9, 'head': 0.1}  # Heavy spine weight
        }
        
        result = scorer.score(Mock(), context)
        
        # With heavy spine weight and flat spine, should score well
        assert result.score > 0.6
    
    def test_empty_response(self):
        """Empty response should return 0."""
        scorer = ZoneFlatnessScorer()
        
        context = {
            'spine_response': None,
            'head_response': np.array([]),
            'zone_weights': {'spine': 0.7, 'head': 0.3}
        }
        
        result = scorer.score(Mock(), context)
        assert result.score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EAR UNIFORMITY SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestEarUniformityScorer:
    """Test ear uniformity (L/R balance) scoring."""
    
    def test_perfect_balance(self):
        """Identical L/R should score high (correlation may be 0 for constant signals)."""
        scorer = EarUniformityScorer()
        
        # Identical left and right
        ear_response = np.ones((2, 50))
        
        context = {'head_response': ear_response}
        result = scorer.score(Mock(), context)
        
        # Level balance should be perfect
        assert result.details['level_balance'] == 1.0
        # Note: correlation of constant signals is undefined (returns 0)
        # So total score may be 0.5 (50% level + 0% correlation + 0% spectral)
        assert result.score >= 0.5
    
    def test_level_imbalance(self):
        """Different L/R levels should reduce score."""
        scorer = EarUniformityScorer()
        
        # Left ear 2x louder than right
        left = np.ones(50) * 2.0
        right = np.ones(50) * 1.0
        ear_response = np.array([left, right])
        
        context = {'head_response': ear_response}
        result = scorer.score(Mock(), context)
        
        # Level balance should be 0.5 (1/2)
        assert result.details['level_balance'] == 0.5
        # Final score is weighted: 0.5*0.5 + 0.25*corr + 0.25*spectral
        # Constant signals have 0 correlation, so score ~= 0.25
        assert result.score >= 0.2
        assert result.score < 0.8
    
    def test_uncorrelated_responses(self):
        """Uncorrelated L/R should score lower."""
        scorer = EarUniformityScorer()
        
        np.random.seed(42)
        left = np.random.rand(50)
        right = np.random.rand(50)
        ear_response = np.array([left, right])
        
        context = {'head_response': ear_response}
        result = scorer.score(Mock(), context)
        
        # Random signals have low correlation
        assert result.details['correlation'] < 0.5
        assert result.score < 0.8
    
    def test_insufficient_data(self):
        """Less than 2 positions should return 0."""
        scorer = EarUniformityScorer()
        
        context = {'head_response': np.ones((1, 50))}
        result = scorer.score(Mock(), context)
        
        assert result.score == 0.0
        assert 'error' in result.details


# ══════════════════════════════════════════════════════════════════════════════
# TEST: SPINE COUPLING SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestSpineCouplingScorer:
    """Test spine coupling scoring."""
    
    def test_high_uniform_response(self):
        """High, uniform response should score well."""
        scorer = SpineCouplingScorer()
        
        # High uniform response
        response = np.ones((10, 50)) * 0.8
        
        context = {'spine_response': response}
        result = scorer.score(Mock(), context)
        
        assert result.score > 0.8
        assert result.details['uniformity'] > 0.9
    
    def test_low_response(self):
        """Low response should score poorly."""
        scorer = SpineCouplingScorer()
        
        # Very low response
        response = np.ones((10, 50)) * 0.1
        
        context = {'spine_response': response}
        result = scorer.score(Mock(), context)
        
        # Low level = low score
        assert result.details['level_score'] < 0.3
    
    def test_nonuniform_response(self):
        """Non-uniform response should reduce score."""
        scorer = SpineCouplingScorer()
        
        # High variation
        response = np.random.rand(10, 50) * 2.0
        
        context = {'spine_response': response}
        result = scorer.score(Mock(), context)
        
        assert result.details['uniformity'] < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# TEST: STRUCTURAL SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuralScorer:
    """Test structural integrity scoring."""
    
    def test_deflection_scoring(self):
        """Test deflection score calculation."""
        scorer = StructuralScorer()
        
        # Within limit
        assert scorer._deflection_score(5.0) > 0.8
        
        # At limit
        assert scorer._deflection_score(10.0) > 0.6
        
        # Over limit
        assert scorer._deflection_score(15.0) < 0.5
    
    def test_safety_factor_scoring(self):
        """Test safety factor score calculation."""
        scorer = StructuralScorer()
        
        # Safe
        assert scorer._safety_factor_score(3.0) == 1.0
        
        # At minimum
        assert scorer._safety_factor_score(2.0) == 1.0
        
        # Below minimum
        assert scorer._safety_factor_score(1.0) == 0.5


# ══════════════════════════════════════════════════════════════════════════════
# TEST: MANUFACTURABILITY SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestManufacturabilityScorer:
    """Test manufacturability scoring."""
    
    def test_simple_plate(self):
        """Simple standard plate should score high."""
        scorer = ManufacturabilityScorer()
        
        genome = Mock()
        genome.length = 2.0
        genome.width = 0.6
        genome.thickness_base = 0.015
        genome.cutouts = []
        genome.contour_type = Mock()
        genome.contour_type.name = "RECTANGLE"
        
        person = Mock()
        person.recommended_plate_length = 1.9
        
        context = {'person': person}
        result = scorer.score(genome, context)
        
        assert result.score > 0.8
    
    def test_many_cutouts(self):
        """Many cutouts should reduce score."""
        scorer = ManufacturabilityScorer()
        
        genome = Mock()
        genome.length = 2.0
        genome.width = 0.6
        genome.thickness_base = 0.015
        genome.cutouts = [Mock() for _ in range(5)]  # 5 cutouts
        
        person = Mock()
        person.recommended_plate_length = 1.9
        
        context = {'person': person}
        result = scorer.score(genome, context)
        
        assert result.details['penalties'].get('cutouts', 0) > 0
        assert result.score < 0.7
    
    def test_plate_too_short(self):
        """Plate shorter than person should be penalized."""
        scorer = ManufacturabilityScorer()
        
        genome = Mock()
        genome.length = 1.5  # Too short!
        genome.width = 0.6
        genome.thickness_base = 0.015
        genome.cutouts = []
        
        person = Mock()
        person.recommended_plate_length = 2.0
        
        context = {'person': person}
        result = scorer.score(genome, context)
        
        assert 'length_deficit' in result.details['penalties']
        # Score should be <= 0.5 due to penalty
        assert result.score <= 0.5


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EXCITER SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestExciterScorer:
    """Test exciter placement scoring."""
    
    def test_antinode_placement(self):
        """Exciters at antinodes should score well."""
        scorer = ExciterScorer()
        
        # Mode shape with antinode at center
        mode_shape = np.zeros((10, 10))
        mode_shape[5, 5] = 1.0  # Antinode at center
        
        # Exciter at center (antinode)
        exciter = Mock()
        exciter.x = 0.5  # Normalized position
        exciter.y = 0.5
        
        genome = Mock()
        genome.exciters = [exciter]
        
        context = {
            'frequencies': [100.0],
            'mode_shapes': np.array([mode_shape])
        }
        
        result = scorer.score(genome, context)
        
        # Should have good coupling
        assert result.score > 0.5
    
    def test_nodal_placement(self):
        """Exciters at nodes should score poorly."""
        scorer = ExciterScorer()
        
        # Mode shape with node at corner
        mode_shape = np.ones((10, 10))
        mode_shape[0, 0] = 0.0  # Node at corner
        
        # Exciter at corner (node)
        exciter = Mock()
        exciter.x = 0.0
        exciter.y = 0.0
        
        genome = Mock()
        genome.exciters = [exciter]
        
        context = {
            'frequencies': [100.0],
            'mode_shapes': np.array([mode_shape])
        }
        
        result = scorer.score(genome, context)
        
        # Should have poor coupling at node
        assert result.details['total_coupling'] < 0.5
    
    def test_no_exciters(self):
        """No exciters should return default score."""
        scorer = ExciterScorer()
        
        genome = Mock()
        genome.exciters = []
        
        context = {}
        result = scorer.score(genome, context)
        
        assert result.score == 0.5
        assert 'error' in result.details
