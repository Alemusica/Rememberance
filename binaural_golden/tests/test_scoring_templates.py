"""
Test suite for scoring_templates.py

Tests Phase 6: Zone-specific scoring templates for different use cases
"""

import pytest
import numpy as np
import json
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.scoring_templates import (
    # Enums
    TemplateType,
    FrequencyBand,
    # Dataclasses
    FrequencyTarget,
    ZoneScoringConfig,
    ScoringTemplate,
    # Factory functions
    create_vat_therapy_template,
    create_binaural_audio_template,
    create_hybrid_template,
    create_meditation_template,
    create_research_template,
    # Registry
    TemplateRegistry,
    # Adapter
    TemplateAdapter,
    # Helpers
    get_template,
    list_available_templates,
    get_default_registry,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class TestEnums:
    """Test enum definitions."""
    
    def test_template_types_exist(self):
        """Verify all expected template types exist."""
        assert TemplateType.VAT_THERAPY
        assert TemplateType.BINAURAL_AUDIO
        assert TemplateType.HYBRID
        assert TemplateType.RESEARCH
        assert TemplateType.MEDITATION
        assert TemplateType.SOUND_BATH
    
    def test_frequency_band_ranges(self):
        """Verify frequency band definitions are valid."""
        assert FrequencyBand.INFRASONIC.value == (1, 20)
        assert FrequencyBand.SUB_BASS.value == (20, 60)
        assert FrequencyBand.VAT_OPTIMAL.value == (30, 120)
        assert FrequencyBand.SPINE_RESONANCE.value == (8, 15)
    
    def test_frequency_bands_are_ordered(self):
        """Verify band low < high for all bands."""
        for band in FrequencyBand:
            low, high = band.value
            assert low < high, f"{band.name}: {low} should be < {high}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FREQUENCY TARGET
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequencyTarget:
    """Test FrequencyTarget dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        target = FrequencyTarget(band=(20, 200))
        
        assert target.band == (20, 200)
        assert target.response_type == "flat"
        assert target.target_db == 0.0
        assert target.tolerance_db == 6.0
        assert target.weight == 1.0
    
    def test_custom_values(self):
        """Test custom initialization."""
        target = FrequencyTarget(
            band=(30, 120),
            response_type="boost",
            target_db=3.0,
            tolerance_db=4.0,
            weight=0.8,
        )
        
        assert target.band == (30, 120)
        assert target.response_type == "boost"
        assert target.target_db == 3.0
        assert target.tolerance_db == 4.0
        assert target.weight == 0.8
    
    def test_resonant_type(self):
        """Test resonant frequency target."""
        target = FrequencyTarget(
            band=(80, 100),
            response_type="resonant",
            center_hz=90.0,
            q_factor=5.0,
        )
        
        assert target.response_type == "resonant"
        assert target.center_hz == 90.0
        assert target.q_factor == 5.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ZONE SCORING CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneScoringConfig:
    """Test ZoneScoringConfig dataclass."""
    
    def test_default_spine_config(self):
        """Test spine zone default configuration."""
        config = ZoneScoringConfig(zone_name="spine")
        
        assert config.zone_name == "spine"
        assert config.weight == 0.5
        assert config.frequency_targets == []
        assert config.min_energy_db == -20.0
        assert config.max_energy_db == 6.0
        assert config.requires_lr_balance is False
    
    def test_ear_config_with_balance(self):
        """Test ear zone with L/R balance requirement."""
        config = ZoneScoringConfig(
            zone_name="ears",
            weight=0.35,
            requires_lr_balance=True,
            lr_tolerance_db=2.0,
        )
        
        assert config.requires_lr_balance is True
        assert config.lr_tolerance_db == 2.0
    
    def test_with_frequency_targets(self):
        """Test zone config with frequency targets."""
        targets = [
            FrequencyTarget(band=(30, 120), response_type="flat"),
            FrequencyTarget(band=(8, 15), response_type="boost", target_db=3.0),
        ]
        
        config = ZoneScoringConfig(
            zone_name="spine",
            weight=0.6,
            frequency_targets=targets,
        )
        
        assert len(config.frequency_targets) == 2
        assert config.frequency_targets[0].band == (30, 120)
        assert config.frequency_targets[1].target_db == 3.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: SCORING TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringTemplate:
    """Test ScoringTemplate dataclass."""
    
    def test_minimal_template(self):
        """Test minimal template creation."""
        template = ScoringTemplate(
            name="Test",
            template_type=TemplateType.RESEARCH,
            description="Test template",
        )
        
        assert template.name == "Test"
        assert template.template_type == TemplateType.RESEARCH
        assert template.zone_configs == {}
        assert template.freq_range == (20.0, 200.0)
    
    def test_get_zone_weight(self):
        """Test zone weight retrieval."""
        template = ScoringTemplate(
            name="Test",
            template_type=TemplateType.RESEARCH,
            description="Test",
            zone_configs={
                "spine": ZoneScoringConfig(zone_name="spine", weight=0.6),
                "ears": ZoneScoringConfig(zone_name="ears", weight=0.4),
            },
        )
        
        assert template.get_zone_weight("spine") == 0.6
        assert template.get_zone_weight("ears") == 0.4
        assert template.get_zone_weight("nonexistent") == 0.0
    
    def test_get_spine_weight(self):
        """Test combined spine weight calculation."""
        template = ScoringTemplate(
            name="Test",
            template_type=TemplateType.VAT_THERAPY,
            description="Test",
            zone_configs={
                "spine": ZoneScoringConfig(zone_name="spine", weight=0.4),
                "lower_back": ZoneScoringConfig(zone_name="lower_back", weight=0.2),
                "sacrum": ZoneScoringConfig(zone_name="sacrum", weight=0.1),
            },
        )
        
        # Should sum spine-related zones
        assert template.get_spine_weight() == pytest.approx(0.7)
    
    def test_get_ear_weight(self):
        """Test combined ear weight calculation."""
        template = ScoringTemplate(
            name="Test",
            template_type=TemplateType.BINAURAL_AUDIO,
            description="Test",
            zone_configs={
                "left_ear": ZoneScoringConfig(zone_name="left_ear", weight=0.3),
                "right_ear": ZoneScoringConfig(zone_name="right_ear", weight=0.3),
                "head": ZoneScoringConfig(zone_name="head", weight=0.1),
            },
        )
        
        # Should sum ear-related zones
        assert template.get_ear_weight() == pytest.approx(0.7)
    
    def test_to_zone_weights(self):
        """Test conversion to legacy zone weights."""
        template = ScoringTemplate(
            name="Test",
            template_type=TemplateType.HYBRID,
            description="Test",
            zone_configs={
                "spine": ZoneScoringConfig(zone_name="spine", weight=0.7),
                "ears": ZoneScoringConfig(zone_name="ears", weight=0.3),
            },
        )
        
        zone_weights = template.to_zone_weights()
        
        assert zone_weights["spine"] == pytest.approx(0.7)
        assert zone_weights["head"] == pytest.approx(0.3)
    
    def test_to_objective_weights(self):
        """Test conversion to legacy objective weights."""
        template = ScoringTemplate(
            name="Test",
            template_type=TemplateType.VAT_THERAPY,
            description="Test",
            flatness_weight=0.8,
            structural_weight=0.6,
            mass_weight=0.2,
            zone_configs={
                "spine": ZoneScoringConfig(zone_name="spine", weight=0.7),
            },
        )
        
        obj_weights = template.to_objective_weights()
        
        assert obj_weights["flatness"] == 0.8
        assert obj_weights["low_mass"] == 0.2
        assert obj_weights["manufacturability"] == 0.6
        # spine_coupling = spine_weight * 2
        assert obj_weights["spine_coupling"] == pytest.approx(1.4)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: JSON SERIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TestJsonSerialization:
    """Test JSON serialization/deserialization."""
    
    def test_to_json(self):
        """Test template to JSON conversion."""
        template = create_vat_therapy_template()
        json_str = template.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        
        assert data["name"] == "VAT Therapy"
        assert data["template_type"] == "VAT_THERAPY"
        assert "spine" in data["zone_configs"]
    
    def test_from_json(self):
        """Test template from JSON creation."""
        original = create_binaural_audio_template()
        json_str = original.to_json()
        
        restored = ScoringTemplate.from_json(json_str)
        
        assert restored.name == original.name
        assert restored.template_type == original.template_type
        assert len(restored.zone_configs) == len(original.zone_configs)
    
    def test_roundtrip(self):
        """Test JSON roundtrip preserves data."""
        original = create_hybrid_template()
        
        json_str = original.to_json()
        restored = ScoringTemplate.from_json(json_str)
        json_str2 = restored.to_json()
        
        # Should serialize identically
        assert json.loads(json_str) == json.loads(json_str2)
    
    def test_frequency_targets_preserved(self):
        """Test frequency targets survive serialization."""
        original = create_vat_therapy_template()
        spine_config = original.zone_configs["spine"]
        
        json_str = original.to_json()
        restored = ScoringTemplate.from_json(json_str)
        
        restored_spine = restored.zone_configs["spine"]
        assert len(restored_spine.frequency_targets) == len(spine_config.frequency_targets)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PREDEFINED TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

class TestPredefinedTemplates:
    """Test predefined template factory functions."""
    
    def test_vat_therapy_template(self):
        """Test VAT therapy template is properly configured."""
        template = create_vat_therapy_template()
        
        assert template.name == "VAT Therapy"
        assert template.template_type == TemplateType.VAT_THERAPY
        assert "spine" in template.zone_configs
        
        # Spine should be highest priority
        assert template.get_spine_weight() > template.get_ear_weight()
        
        # Frequency range should cover VAT range
        assert template.freq_range[0] <= 30  # Low end
        assert template.freq_range[1] >= 120  # VAT optimal high
        
        # Should have paper references
        assert len(template.paper_references) > 0
    
    def test_binaural_audio_template(self):
        """Test binaural audio template is properly configured."""
        template = create_binaural_audio_template()
        
        assert template.name == "Binaural Audio"
        assert template.template_type == TemplateType.BINAURAL_AUDIO
        
        # Should have L/R ear zones
        assert "left_ear" in template.zone_configs
        assert "right_ear" in template.zone_configs
        
        # Ear zones should require balance
        right_ear = template.zone_configs["right_ear"]
        assert right_ear.requires_lr_balance is True
        
        # Balance weight should be high
        assert template.balance_weight >= 1.0
        
        # Should prefer symmetric design
        assert template.prefer_symmetric is True
    
    def test_hybrid_template(self):
        """Test hybrid template is balanced."""
        template = create_hybrid_template()
        
        assert template.name == "Hybrid"
        assert template.template_type == TemplateType.HYBRID
        
        # Should be reasonably balanced
        spine_w = template.get_spine_weight()
        ear_w = template.get_ear_weight()
        
        # Neither should be < 20% or > 70%
        # Hybrid still favors spine slightly for vibroacoustic benefit
        assert 0.2 <= spine_w <= 0.7
        assert 0.2 <= ear_w <= 0.5
    
    def test_meditation_template(self):
        """Test meditation template focuses on low frequencies."""
        template = create_meditation_template()
        
        assert template.name == "Meditation"
        assert template.template_type == TemplateType.MEDITATION
        
        # Should have very low frequency range
        assert template.freq_range[0] <= 10  # Include theta range
        
        # Spine frequency targets should include theta range
        spine = template.zone_configs["spine"]
        has_theta_target = any(
            ft.band[0] <= 8 for ft in spine.frequency_targets
        )
        assert has_theta_target
    
    def test_research_template(self):
        """Test research template is customizable."""
        custom_zones = {
            "custom_zone": {
                "weight": 0.8,
                "frequency_targets": [
                    {"band": [100, 500], "response_type": "flat"}
                ],
            }
        }
        
        template = create_research_template(
            name="My Research",
            zones=custom_zones,
        )
        
        assert template.name == "My Research"
        assert template.template_type == TemplateType.RESEARCH
        assert "custom_zone" in template.zone_configs
        assert template.zone_configs["custom_zone"].weight == 0.8


# ══════════════════════════════════════════════════════════════════════════════
# TEST: TEMPLATE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class TestTemplateRegistry:
    """Test TemplateRegistry class."""
    
    def test_default_registry_has_templates(self):
        """Test default registry loads predefined templates."""
        registry = TemplateRegistry()
        
        names = registry.list_templates()
        
        assert "VAT Therapy" in names
        assert "Binaural Audio" in names
        assert "Hybrid" in names
        assert "Meditation" in names
    
    def test_get_template_by_name(self):
        """Test retrieving template by name."""
        registry = TemplateRegistry()
        
        template = registry.get("VAT Therapy")
        
        assert template is not None
        assert template.name == "VAT Therapy"
    
    def test_get_nonexistent_returns_none(self):
        """Test retrieving nonexistent template returns None."""
        registry = TemplateRegistry()
        
        template = registry.get("Nonexistent Template")
        
        assert template is None
    
    def test_register_custom_template(self):
        """Test registering custom template."""
        registry = TemplateRegistry()
        
        custom = ScoringTemplate(
            name="Custom",
            template_type=TemplateType.RESEARCH,
            description="Custom test",
        )
        
        registry.register(custom)
        
        assert "Custom" in registry.list_templates()
        assert registry.get("Custom") is custom
    
    def test_get_by_type(self):
        """Test retrieving templates by type."""
        registry = TemplateRegistry()
        
        vat_templates = registry.get_by_type(TemplateType.VAT_THERAPY)
        
        assert len(vat_templates) >= 1
        assert all(t.template_type == TemplateType.VAT_THERAPY for t in vat_templates)
    
    def test_empty_registry(self):
        """Test registry without defaults."""
        registry = TemplateRegistry(load_defaults=False)
        
        assert registry.list_templates() == []
    
    def test_save_and_load_file(self):
        """Test saving and loading registry to file."""
        registry = TemplateRegistry()
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        try:
            registry.save_to_file(path)
            
            # Create new registry and load
            new_registry = TemplateRegistry(load_defaults=False)
            new_registry.load_from_file(path)
            
            assert new_registry.list_templates() == registry.list_templates()
        finally:
            path.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# TEST: TEMPLATE ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class TestTemplateAdapter:
    """Test TemplateAdapter for FitnessEvaluator integration."""
    
    def test_adapter_creation(self):
        """Test adapter can be created."""
        template = create_vat_therapy_template()
        adapter = TemplateAdapter(template)
        
        assert adapter.template is template
    
    def test_get_freq_range(self):
        """Test adapter returns frequency range."""
        template = create_binaural_audio_template()
        adapter = TemplateAdapter(template)
        
        freq_range = adapter.get_freq_range()
        
        assert freq_range == template.freq_range
    
    def test_get_n_freq_points(self):
        """Test adapter returns frequency points."""
        template = create_binaural_audio_template()
        adapter = TemplateAdapter(template)
        
        n_points = adapter.get_n_freq_points()
        
        assert n_points == template.n_freq_points


# ══════════════════════════════════════════════════════════════════════════════
# TEST: HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestHelperFunctions:
    """Test module-level helper functions."""
    
    def test_get_template_by_name(self):
        """Test get_template with name string."""
        template = get_template("VAT Therapy")
        
        assert template.name == "VAT Therapy"
    
    def test_get_template_by_type(self):
        """Test get_template with TemplateType enum."""
        template = get_template(TemplateType.BINAURAL_AUDIO)
        
        assert template.template_type == TemplateType.BINAURAL_AUDIO
    
    def test_get_template_invalid_name_raises(self):
        """Test get_template raises for invalid name."""
        with pytest.raises(ValueError, match="not found"):
            get_template("Invalid Template Name")
    
    def test_list_available_templates(self):
        """Test listing available templates."""
        templates = list_available_templates()
        
        assert len(templates) >= 4  # At least 4 defaults
        
        # Check structure
        for t in templates:
            assert "name" in t
            assert "type" in t
            assert "description" in t
    
    def test_get_default_registry_singleton(self):
        """Test default registry is reused."""
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        
        assert reg1 is reg2


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PHYSICS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysicsValidation:
    """Validate templates against physics constraints."""
    
    def test_vat_frequency_range_is_valid(self):
        """Validate VAT frequency range (Skille 1989: 30-120Hz)."""
        template = create_vat_therapy_template()
        
        spine = template.zone_configs["spine"]
        vat_target = next(
            (t for t in spine.frequency_targets if t.band == (30, 120)),
            None
        )
        
        assert vat_target is not None, "Should have 30-120Hz target"
        assert vat_target.response_type == "flat"
    
    def test_spine_resonance_range_is_valid(self):
        """Validate spine resonance range (Griffin 1990: 8-15Hz)."""
        template = create_vat_therapy_template()
        
        spine = template.zone_configs["spine"]
        resonance_target = next(
            (t for t in spine.frequency_targets if t.band[0] <= 10),
            None
        )
        
        # Should include spine resonance range
        assert resonance_target is not None, "Should include spine resonance frequencies"
    
    def test_tolerance_is_reasonable(self):
        """Validate tolerance values are physically reasonable."""
        templates = [
            create_vat_therapy_template(),
            create_binaural_audio_template(),
            create_hybrid_template(),
        ]
        
        for template in templates:
            for zone in template.zone_configs.values():
                for target in zone.frequency_targets:
                    # Tolerance should be between 1-15 dB
                    assert 1.0 <= target.tolerance_db <= 15.0, \
                        f"Unreasonable tolerance: {target.tolerance_db} dB"
    
    def test_lr_tolerance_is_reasonable(self):
        """Validate L/R tolerance values are physically reasonable."""
        template = create_binaural_audio_template()
        
        for zone in template.zone_configs.values():
            if zone.requires_lr_balance:
                # L/R tolerance should be 1-10 dB
                assert 1.0 <= zone.lr_tolerance_db <= 10.0, \
                    f"Unreasonable L/R tolerance: {zone.lr_tolerance_db} dB"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_zone_configs(self):
        """Test template with no zone configs."""
        template = ScoringTemplate(
            name="Empty",
            template_type=TemplateType.RESEARCH,
            description="Empty zones",
        )
        
        assert template.get_spine_weight() == 0.0
        assert template.get_ear_weight() == 0.0
        
        zone_weights = template.to_zone_weights()
        # Should return defaults
        assert zone_weights["spine"] == 0.7
        assert zone_weights["head"] == 0.3
    
    def test_zero_weight_zones(self):
        """Test zones with zero weight."""
        template = ScoringTemplate(
            name="Zero",
            template_type=TemplateType.RESEARCH,
            description="Zero weight zones",
            zone_configs={
                "spine": ZoneScoringConfig(zone_name="spine", weight=0.0),
                "ears": ZoneScoringConfig(zone_name="ears", weight=0.0),
            },
        )
        
        zone_weights = template.to_zone_weights()
        # Should return defaults when all zero
        assert zone_weights["spine"] == 0.7
        assert zone_weights["head"] == 0.3
    
    def test_very_narrow_frequency_band(self):
        """Test very narrow frequency band."""
        target = FrequencyTarget(
            band=(99, 101),  # 2 Hz band
            response_type="resonant",
            center_hz=100.0,
        )
        
        assert target.band[1] - target.band[0] == 2
    
    def test_json_with_special_characters(self):
        """Test JSON with special characters in description."""
        template = ScoringTemplate(
            name="Special",
            template_type=TemplateType.RESEARCH,
            description='Test with "quotes" and\nnewlines',
        )
        
        json_str = template.to_json()
        restored = ScoringTemplate.from_json(json_str)
        
        assert 'quotes' in restored.description


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests with other modules."""
    
    def test_template_to_zone_weights_format(self):
        """Verify to_zone_weights output matches expected format."""
        template = create_hybrid_template()
        
        zone_weights = template.to_zone_weights()
        
        # Should have exactly these keys
        assert set(zone_weights.keys()) == {"spine", "head"}
        
        # Values should sum to 1.0
        assert zone_weights["spine"] + zone_weights["head"] == pytest.approx(1.0)
    
    def test_template_to_objective_weights_format(self):
        """Verify to_objective_weights output matches expected format."""
        template = create_vat_therapy_template()
        
        obj_weights = template.to_objective_weights()
        
        # Should have expected keys
        expected_keys = {"flatness", "spine_coupling", "low_mass", "manufacturability"}
        assert set(obj_weights.keys()) == expected_keys
        
        # All values should be positive
        for key, value in obj_weights.items():
            assert value >= 0, f"{key} should be non-negative"
    
    def test_all_templates_are_valid(self):
        """Validate all predefined templates."""
        registry = TemplateRegistry()
        
        for name in registry.list_templates():
            template = registry.get(name)
            
            # Should have required fields
            assert template.name
            assert template.description
            assert template.template_type in TemplateType
            
            # Frequency range should be valid
            assert template.freq_range[0] < template.freq_range[1]
            
            # Should be JSON serializable
            json_str = template.to_json()
            restored = ScoringTemplate.from_json(json_str)
            assert restored.name == template.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
