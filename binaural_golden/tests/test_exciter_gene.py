"""
Tests for ExciterGene - Phase 2 of Action Plan 3.0

"Il seme non parla dei petali" - Emission genes activate when needed.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.exciter_gene import (
    ExciterGene, EmissionGenes,
    upgrade_exciters_to_genes, activate_all_emission, freeze_all_positions,
    calculate_position_sigma, get_emission_summary
)
from core.analysis_config import GenePhase, EmissionBounds


class TestEmissionGenes:
    """Test emission gene behavior."""
    
    def test_default_is_neutral(self):
        """Default emission should have no effect."""
        em = EmissionGenes()
        assert em.is_neutral()
        assert em.phase_deg == 0.0
        assert em.delay_samples == 0
        assert em.gain_db == 0.0
        assert not em.polarity_inverted
    
    def test_to_dsp_dict(self):
        """DSP export should include all parameters."""
        em = EmissionGenes(
            phase_deg=90.0,
            delay_samples=10,
            gain_db=-3.0,
            polarity_inverted=True
        )
        dsp = em.to_dsp_dict()
        
        assert dsp["phase_deg"] == 90.0
        assert dsp["delay_samples"] == 10
        assert abs(dsp["delay_ms"] - 10/48.0) < 0.001
        assert dsp["gain_db"] == -3.0
        assert dsp["polarity"] == -1
    
    def test_clone(self):
        """Clone should create independent copy."""
        em1 = EmissionGenes(phase_deg=45.0, gain_db=2.0)
        em2 = em1.clone()
        
        em2.phase_deg = 90.0
        assert em1.phase_deg == 45.0  # Original unchanged


class TestExciterGenePhasingSystem:
    """Test the SEED → BLOOM → FREEZE phase transitions."""
    
    def test_starts_in_seed_phase(self):
        """New gene should start in SEED phase."""
        gene = ExciterGene(x=0.3, y=0.85, channel=1)
        
        assert gene.phase == GenePhase.SEED
        assert not gene.is_emission_active()
        assert gene.is_position_evolvable()
    
    def test_seed_to_bloom_transition(self):
        """activate_emission() should transition SEED → BLOOM."""
        gene = ExciterGene(x=0.3, y=0.85, channel=1)
        gene.activate_emission()
        
        assert gene.phase == GenePhase.BLOOM
        assert gene.is_emission_active()
        assert gene.is_position_evolvable()  # Position still evolvable
    
    def test_bloom_to_freeze_transition(self):
        """freeze_position() should lock position."""
        gene = ExciterGene(x=0.3, y=0.85, channel=1)
        gene.activate_emission()
        gene.freeze_position()
        
        assert gene.phase == GenePhase.FREEZE
        assert gene.is_emission_active()
        assert not gene.is_position_evolvable()  # Position locked
        assert gene.position_locked
    
    def test_seed_emission_dormant(self):
        """In SEED phase, emission should be dormant (not mutated)."""
        gene = ExciterGene(x=0.3, y=0.85, channel=1)
        
        # Store original emission
        original_phase = gene.emission.phase_deg
        
        # Mutate (emission should not change in SEED)
        gene.mutate_emission()
        
        # Emission unchanged
        assert gene.emission.phase_deg == original_phase


class TestExciterGeneMutation:
    """Test mutation behavior respecting phase."""
    
    def test_position_mutation_in_seed(self):
        """Position should mutate in SEED phase."""
        gene = ExciterGene(x=0.5, y=0.5, channel=1)
        original_x, original_y = gene.x, gene.y
        
        # Mutate many times to ensure at least one change
        for _ in range(10):
            gene.mutate_position(sigma_x=0.1, sigma_y=0.1)
        
        # Position should have changed
        assert (gene.x != original_x) or (gene.y != original_y)
    
    def test_position_locked_after_freeze(self):
        """Position should not mutate after FREEZE."""
        gene = ExciterGene(x=0.5, y=0.5, channel=1)
        gene.freeze_position()
        
        original_x, original_y = gene.x, gene.y
        
        # Try to mutate
        for _ in range(10):
            gene.mutate_position(sigma_x=0.2, sigma_y=0.2)
        
        # Position unchanged
        assert gene.x == original_x
        assert gene.y == original_y
    
    def test_emission_mutation_in_bloom(self):
        """Emission should mutate in BLOOM phase."""
        gene = ExciterGene(x=0.5, y=0.5, channel=1)
        gene.activate_emission()
        
        # Store and mutate
        changes_detected = False
        for _ in range(20):
            old_phase = gene.emission.phase_deg
            gene.mutate_emission(mutation_rates={
                "phase": 0.9,  # High rate for testing
                "delay": 0.9,
                "gain": 0.9,
                "polarity": 0.5,
            })
            if gene.emission.phase_deg != old_phase:
                changes_detected = True
                break
        
        assert changes_detected, "Emission should mutate in BLOOM"
    
    def test_bounds_respected(self):
        """Mutation should respect emission bounds."""
        bounds = EmissionBounds(
            phase_min=0.0, phase_max=360.0,
            delay_min=0, delay_max=17,
            gain_min_db=-12.0, gain_max_db=6.0
        )
        
        gene = ExciterGene(x=0.5, y=0.5, channel=1)
        gene.activate_emission()
        
        # Mutate many times
        for _ in range(100):
            gene.mutate_emission(config=bounds)
        
        # Check bounds
        assert 0.0 <= gene.emission.phase_deg <= 360.0
        assert bounds.delay_min <= gene.emission.delay_samples <= bounds.delay_max
        assert bounds.gain_min_db <= gene.emission.gain_db <= bounds.gain_max_db


class TestExciterGeneCrossover:
    """Test crossover operations."""
    
    def test_position_blending(self):
        """Position should blend between parents."""
        parent1 = ExciterGene(x=0.2, y=0.2, channel=1)
        parent2 = ExciterGene(x=0.8, y=0.8, channel=1)
        
        child = parent1.crossover_with(parent2, position_blend=0.5)
        
        # Should be approximately in the middle
        assert 0.4 < child.x < 0.6
        assert 0.4 < child.y < 0.6
    
    def test_emission_crossover_in_bloom(self):
        """Emission should crossover when active."""
        parent1 = ExciterGene(x=0.5, y=0.5, channel=1)
        parent1.activate_emission()
        parent1.emission.phase_deg = 90.0
        parent1.emission.gain_db = 3.0
        
        parent2 = ExciterGene(x=0.5, y=0.5, channel=1)
        parent2.activate_emission()
        parent2.emission.phase_deg = 180.0
        parent2.emission.gain_db = -3.0
        
        child = parent1.crossover_with(parent2, emission_from_other=False)
        child.activate_emission()
        
        # Emission should be blended
        assert child.emission.phase_deg == (90.0 + 180.0) / 2
        assert child.emission.gain_db == (3.0 + (-3.0)) / 2


class TestExciterGeneCompatibility:
    """Test backwards compatibility with ExciterPosition."""
    
    def test_to_absolute_compatible(self):
        """to_absolute() should work like ExciterPosition."""
        gene = ExciterGene(x=0.3, y=0.85, channel=1)
        
        result = gene.to_absolute(plate_length=1.85, plate_width=0.64)
        
        assert "center" in result
        assert "diameter" in result
        assert "channel" in result
        
        # Coordinate mapping: y→FEM_x, x→FEM_y
        expected_center = (0.85 * 1.85, 0.3 * 0.64)
        assert abs(result["center"][0] - expected_center[0]) < 0.001
        assert abs(result["center"][1] - expected_center[1]) < 0.001
    
    def test_from_legacy_dict(self):
        """Should create from legacy dict."""
        legacy = {
            "x": 0.3,
            "y": 0.85,
            "channel": 1,
            "exciter_model": "dayton_daex25",
            "diameter_mm": 25.0
        }
        
        gene = ExciterGene.from_legacy_position(legacy)
        
        assert gene.x == 0.3
        assert gene.y == 0.85
        assert gene.channel == 1
        assert gene.phase == GenePhase.SEED
    
    def test_zone_properties(self):
        """Zone detection should work."""
        head = ExciterGene(x=0.5, y=0.85, channel=1)
        feet = ExciterGene(x=0.5, y=0.15, channel=3)
        torso = ExciterGene(x=0.5, y=0.5, channel=2)
        
        assert head.is_head_zone
        assert head.zone == "head"
        
        assert feet.is_feet_zone
        assert feet.zone == "feet"
        
        assert torso.zone == "torso"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_upgrade_exciters_to_genes(self):
        """Should upgrade legacy list."""
        legacy = [
            {"x": 0.3, "y": 0.85, "channel": 1},
            {"x": 0.7, "y": 0.85, "channel": 2},
        ]
        
        genes = upgrade_exciters_to_genes(legacy)
        
        assert len(genes) == 2
        assert all(isinstance(g, ExciterGene) for g in genes)
        assert all(g.phase == GenePhase.SEED for g in genes)
    
    def test_activate_all_emission(self):
        """Should activate all genes."""
        genes = ExciterGene.create_default_layout()
        assert all(g.phase == GenePhase.SEED for g in genes)
        
        activate_all_emission(genes)
        
        assert all(g.phase == GenePhase.BLOOM for g in genes)
    
    def test_freeze_all_positions(self):
        """Should freeze all positions."""
        genes = ExciterGene.create_default_layout()
        freeze_all_positions(genes)
        
        assert all(g.phase == GenePhase.FREEZE for g in genes)
        assert all(g.position_locked for g in genes)
    
    def test_calculate_position_sigma(self):
        """Should calculate position spread."""
        # Tight cluster
        genes_tight = [
            ExciterGene(x=0.5, y=0.5, channel=i) for i in range(4)
        ]
        sigma_tight = calculate_position_sigma(genes_tight)
        
        # Wide spread
        genes_wide = [
            ExciterGene(x=0.1, y=0.1, channel=1),
            ExciterGene(x=0.9, y=0.1, channel=2),
            ExciterGene(x=0.1, y=0.9, channel=3),
            ExciterGene(x=0.9, y=0.9, channel=4),
        ]
        sigma_wide = calculate_position_sigma(genes_wide)
        
        assert sigma_tight < sigma_wide
    
    def test_get_emission_summary(self):
        """Should summarize emission parameters."""
        genes = ExciterGene.create_default_layout()
        
        # In SEED - no active emission
        summary = get_emission_summary(genes)
        assert summary["active_count"] == 0
        
        # Activate
        activate_all_emission(genes)
        genes[0].emission.phase_deg = 90.0
        genes[1].emission.phase_deg = 180.0
        
        summary = get_emission_summary(genes)
        assert summary["active_count"] == 4
        # phase_spread is max - min: 180 - 0 = 180 (genes[2] and [3] are still at 0)
        assert summary["phase_spread"] == 180.0


class TestDefaultLayout:
    """Test default exciter layout creation."""
    
    def test_create_default_layout(self):
        """Should create 4 exciters in standard configuration."""
        layout = ExciterGene.create_default_layout()
        
        assert len(layout) == 4
        
        # Check channels
        channels = [g.channel for g in layout]
        assert sorted(channels) == [1, 2, 3, 4]
        
        # Check zones
        head_exciters = [g for g in layout if g.is_head_zone]
        feet_exciters = [g for g in layout if g.is_feet_zone]
        
        assert len(head_exciters) == 2
        assert len(feet_exciters) == 2
    
    def test_create_stereo_pairs(self):
        """Should create stereo pairs."""
        head_l, head_r = ExciterGene.create_head_stereo()
        
        assert head_l.channel == 1
        assert head_r.channel == 2
        assert head_l.x < head_r.x  # Left is actually left
        
        feet_l, feet_r = ExciterGene.create_feet_stereo()
        
        assert feet_l.channel == 3
        assert feet_r.channel == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
