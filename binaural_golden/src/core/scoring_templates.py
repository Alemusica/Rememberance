"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SCORING TEMPLATES - Zone-Specific Fitness Configurations           ║
║                                                                              ║
║   Pre-configured scoring templates for different use cases:                  ║
║   1. VAT Therapy: Maximize spine vibration (10-300Hz)                        ║
║   2. Binaural Audio: Maximize ear L/R balance, flat response                 ║
║   3. Hybrid: Balance between therapy and audio                               ║
║   4. Research: Custom researcher-defined objectives                          ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Skille 1989: VAT founder (30-120Hz optimal for therapy)                  ║
║   • Griffin 1990: Body resonances (spine 10-12Hz fundamental)                ║
║   • Bai & Liu 2004: Multi-exciter for flat frequency response                ║
║                                                                              ║
║   KEY INSIGHT:                                                               ║
║   Different applications need different fitness functions.                   ║
║   Templates make it easy to switch between use cases without                 ║
║   manually configuring weights and frequency ranges.                         ║
║                                                                              ║
║   INTEGRATION:                                                               ║
║   - Used by FitnessEvaluator to configure objective weights                  ║
║   - Can be modified by LTM distillation based on experience                  ║
║   - GUI can offer template selection to users                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
from enum import Enum, auto
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE TYPES
# ══════════════════════════════════════════════════════════════════════════════

class TemplateType(Enum):
    """Predefined template types."""
    VAT_THERAPY = auto()      # Vibroacoustic therapy (spine focus)
    BINAURAL_AUDIO = auto()   # Music reproduction (ear L/R balance)
    HYBRID = auto()           # Balanced therapy + audio
    RESEARCH = auto()         # Custom configuration
    MEDITATION = auto()        # Low frequency (theta/delta waves)
    SOUND_BATH = auto()        # Full body immersion


class FrequencyBand(Enum):
    """Standard frequency bands for vibroacoustic applications."""
    INFRASONIC = (1, 20)       # Below hearing, felt not heard
    SUB_BASS = (20, 60)         # Deep bass, body resonance
    BASS = (60, 250)            # Musical bass, spine coupling
    LOW_MID = (250, 500)        # Warmth, chest resonance
    MID = (500, 2000)           # Clarity, speech
    HIGH_MID = (2000, 4000)     # Presence, detail
    HIGH = (4000, 20000)        # Air, brilliance
    
    # Therapeutic bands
    VAT_OPTIMAL = (30, 120)     # Skille 1989 - VAT core range
    SPINE_RESONANCE = (8, 15)   # Griffin 1990 - spine fundamental
    BODY_HARMONICS = (15, 60)   # Whole body resonances


@dataclass
class FrequencyTarget:
    """
    Target frequency specification for optimization.
    
    Defines what frequency response is desired in a zone.
    """
    band: Tuple[float, float]  # (low_hz, high_hz)
    response_type: str = "flat"  # "flat", "boost", "cut", "resonant"
    target_db: float = 0.0      # Target level relative to reference
    tolerance_db: float = 6.0   # Acceptable deviation
    weight: float = 1.0         # Importance weight
    
    # For resonant type
    center_hz: Optional[float] = None
    q_factor: Optional[float] = None


@dataclass
class ZoneScoringConfig:
    """
    Scoring configuration for a specific body zone.
    
    Defines how fitness is calculated for this zone.
    """
    zone_name: str  # "spine", "left_ear", "right_ear", "chest", etc.
    
    # Weight in overall fitness [0, 1]
    weight: float = 0.5
    
    # Frequency targets
    frequency_targets: List[FrequencyTarget] = field(default_factory=list)
    
    # Energy requirements
    min_energy_db: float = -20.0  # Minimum energy to deliver
    max_energy_db: float = 6.0    # Maximum before clipping concern
    
    # Uniformity requirements (for paired zones like ears)
    requires_lr_balance: bool = False
    lr_tolerance_db: float = 3.0  # Max L/R difference
    
    # Phase requirements
    phase_coherence_required: bool = False
    max_phase_deviation_deg: float = 45.0


@dataclass
class ScoringTemplate:
    """
    Complete scoring template for optimization.
    
    Combines zone configurations, frequency bands, and weighting
    into a complete fitness configuration.
    
    USAGE:
        template = ScoringTemplate.vat_therapy()
        evaluator = FitnessEvaluator(person, template=template)
    """
    name: str
    template_type: TemplateType
    description: str
    
    # Zone configurations
    zone_configs: Dict[str, ZoneScoringConfig] = field(default_factory=dict)
    
    # Global settings
    freq_range: Tuple[float, float] = (20.0, 200.0)
    n_freq_points: int = 50
    
    # Objective weights (for legacy compatibility)
    flatness_weight: float = 1.0
    energy_weight: float = 1.0
    balance_weight: float = 0.5
    structural_weight: float = 0.5
    mass_weight: float = 0.3
    
    # Structural constraints
    max_deflection_mm: float = 10.0
    min_safety_factor: float = 2.0
    
    # Exciter constraints
    min_exciters: int = 1
    max_exciters: int = 6
    prefer_symmetric: bool = False
    
    # Peninsula/cutout handling
    allow_peninsulas: bool = True  # ABH benefit
    cutout_style: str = "lutherie"  # "lutherie", "holes", "slots", "none"
    
    # Metadata
    paper_references: List[str] = field(default_factory=list)
    created_by: str = ""
    version: str = "1.0"
    
    def get_zone_weight(self, zone_name: str) -> float:
        """Get weight for a specific zone."""
        if zone_name in self.zone_configs:
            return self.zone_configs[zone_name].weight
        return 0.0
    
    def get_spine_weight(self) -> float:
        """Get combined spine zone weight."""
        spine_zones = ["spine", "lower_back", "upper_back", "sacrum"]
        return sum(self.get_zone_weight(z) for z in spine_zones)
    
    def get_ear_weight(self) -> float:
        """Get combined ear zone weight."""
        ear_zones = ["left_ear", "right_ear", "ears", "head"]
        return sum(self.get_zone_weight(z) for z in ear_zones)
    
    def to_objective_weights(self) -> Dict[str, float]:
        """Convert to legacy ObjectiveWeights format."""
        return {
            "flatness": self.flatness_weight,
            "spine_coupling": self.get_spine_weight() * 2,  # Scale for compatibility
            "low_mass": self.mass_weight,
            "manufacturability": self.structural_weight,
        }
    
    def to_zone_weights(self) -> Dict[str, float]:
        """Convert to legacy ZoneWeights format."""
        spine = self.get_spine_weight()
        head = self.get_ear_weight()
        total = spine + head
        
        if total > 0:
            return {"spine": spine / total, "head": head / total}
        return {"spine": 0.7, "head": 0.3}
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "name": self.name,
            "template_type": self.template_type.name,
            "description": self.description,
            "zone_configs": {
                name: {
                    "zone_name": cfg.zone_name,
                    "weight": cfg.weight,
                    "frequency_targets": [
                        {
                            "band": ft.band,
                            "response_type": ft.response_type,
                            "target_db": ft.target_db,
                            "tolerance_db": ft.tolerance_db,
                            "weight": ft.weight,
                        }
                        for ft in cfg.frequency_targets
                    ],
                    "min_energy_db": cfg.min_energy_db,
                    "max_energy_db": cfg.max_energy_db,
                    "requires_lr_balance": cfg.requires_lr_balance,
                    "lr_tolerance_db": cfg.lr_tolerance_db,
                }
                for name, cfg in self.zone_configs.items()
            },
            "freq_range": self.freq_range,
            "flatness_weight": self.flatness_weight,
            "energy_weight": self.energy_weight,
            "balance_weight": self.balance_weight,
            "structural_weight": self.structural_weight,
            "mass_weight": self.mass_weight,
            "max_deflection_mm": self.max_deflection_mm,
            "min_safety_factor": self.min_safety_factor,
            "allow_peninsulas": self.allow_peninsulas,
            "cutout_style": self.cutout_style,
            "paper_references": self.paper_references,
            "version": self.version,
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ScoringTemplate":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        
        zone_configs = {}
        for name, cfg_data in data.get("zone_configs", {}).items():
            freq_targets = [
                FrequencyTarget(
                    band=tuple(ft["band"]),
                    response_type=ft["response_type"],
                    target_db=ft["target_db"],
                    tolerance_db=ft["tolerance_db"],
                    weight=ft["weight"],
                )
                for ft in cfg_data.get("frequency_targets", [])
            ]
            
            zone_configs[name] = ZoneScoringConfig(
                zone_name=cfg_data["zone_name"],
                weight=cfg_data["weight"],
                frequency_targets=freq_targets,
                min_energy_db=cfg_data.get("min_energy_db", -20),
                max_energy_db=cfg_data.get("max_energy_db", 6),
                requires_lr_balance=cfg_data.get("requires_lr_balance", False),
                lr_tolerance_db=cfg_data.get("lr_tolerance_db", 3),
            )
        
        return cls(
            name=data["name"],
            template_type=TemplateType[data["template_type"]],
            description=data["description"],
            zone_configs=zone_configs,
            freq_range=tuple(data.get("freq_range", (20, 200))),
            flatness_weight=data.get("flatness_weight", 1.0),
            energy_weight=data.get("energy_weight", 1.0),
            balance_weight=data.get("balance_weight", 0.5),
            structural_weight=data.get("structural_weight", 0.5),
            mass_weight=data.get("mass_weight", 0.3),
            max_deflection_mm=data.get("max_deflection_mm", 10),
            min_safety_factor=data.get("min_safety_factor", 2),
            allow_peninsulas=data.get("allow_peninsulas", True),
            cutout_style=data.get("cutout_style", "lutherie"),
            paper_references=data.get("paper_references", []),
            version=data.get("version", "1.0"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PREDEFINED TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def create_vat_therapy_template() -> ScoringTemplate:
    """
    Create VAT (Vibroacoustic Therapy) focused template.
    
    Based on Skille 1989 and Griffin 1990:
    - Primary focus on spine vibration delivery
    - Optimal range 30-120Hz
    - Sub-bass for deep relaxation
    
    Use case: Therapeutic relaxation, pain management, stress relief
    """
    return ScoringTemplate(
        name="VAT Therapy",
        template_type=TemplateType.VAT_THERAPY,
        description="Vibroacoustic therapy focus - maximize spine vibration delivery",
        zone_configs={
            "spine": ZoneScoringConfig(
                zone_name="spine",
                weight=0.6,
                frequency_targets=[
                    FrequencyTarget(
                        band=(30, 120),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=6.0,
                        weight=1.0,
                    ),
                    FrequencyTarget(
                        band=(8, 15),
                        response_type="boost",
                        target_db=3.0,  # Slight boost at spine resonance
                        tolerance_db=3.0,
                        weight=0.5,
                    ),
                ],
                min_energy_db=-10,
                max_energy_db=6,
            ),
            "sacrum": ZoneScoringConfig(
                zone_name="sacrum",
                weight=0.2,
                frequency_targets=[
                    FrequencyTarget(
                        band=(20, 60),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=8.0,
                        weight=1.0,
                    ),
                ],
                min_energy_db=-15,
            ),
            "ears": ZoneScoringConfig(
                zone_name="ears",
                weight=0.2,
                frequency_targets=[
                    FrequencyTarget(
                        band=(60, 200),
                        response_type="flat",
                        target_db=-6.0,  # Reduced level at ears
                        tolerance_db=10.0,
                        weight=0.5,
                    ),
                ],
                requires_lr_balance=True,
                lr_tolerance_db=6.0,  # More tolerant for therapy
            ),
        },
        freq_range=(20, 200),
        flatness_weight=0.8,
        energy_weight=1.5,  # Energy delivery more important
        balance_weight=0.3,
        structural_weight=0.6,
        mass_weight=0.2,
        allow_peninsulas=True,
        cutout_style="lutherie",
        paper_references=[
            "skille1989vibroacoustic",
            "griffin1990handbook",
            "bartel2015vat",
        ],
        version="1.0",
    )


def create_binaural_audio_template() -> ScoringTemplate:
    """
    Create binaural audio focused template.
    
    Based on DML research (Harris 2010, Aures 2001):
    - Priority on ear L/R balance for stereo imaging
    - Flat frequency response for music reproduction
    - Multi-exciter for extended bandwidth
    
    Use case: Music listening, meditation with binaural beats
    """
    return ScoringTemplate(
        name="Binaural Audio",
        template_type=TemplateType.BINAURAL_AUDIO,
        description="Binaural audio focus - maximize ear L/R balance and flatness",
        zone_configs={
            "left_ear": ZoneScoringConfig(
                zone_name="left_ear",
                weight=0.35,
                frequency_targets=[
                    FrequencyTarget(
                        band=(40, 500),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=4.0,  # Tight tolerance for audio
                        weight=1.0,
                    ),
                ],
                min_energy_db=-15,
                max_energy_db=3,
            ),
            "right_ear": ZoneScoringConfig(
                zone_name="right_ear",
                weight=0.35,
                frequency_targets=[
                    FrequencyTarget(
                        band=(40, 500),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=4.0,
                        weight=1.0,
                    ),
                ],
                min_energy_db=-15,
                max_energy_db=3,
                requires_lr_balance=True,
                lr_tolerance_db=2.0,  # Very tight L/R tolerance
            ),
            "spine": ZoneScoringConfig(
                zone_name="spine",
                weight=0.3,
                frequency_targets=[
                    FrequencyTarget(
                        band=(20, 120),
                        response_type="flat",
                        target_db=-3.0,  # Slightly reduced
                        tolerance_db=8.0,
                        weight=0.7,
                    ),
                ],
                min_energy_db=-20,
            ),
        },
        freq_range=(30, 500),
        n_freq_points=80,  # More resolution for audio
        flatness_weight=1.5,  # Flatness more important
        energy_weight=0.8,
        balance_weight=2.0,  # L/R balance critical!
        structural_weight=0.4,
        mass_weight=0.3,
        prefer_symmetric=True,  # Symmetric for L/R
        allow_peninsulas=True,
        cutout_style="lutherie",
        paper_references=[
            "harris2010dml",
            "aures2001dml",
            "bai2004genetic",
            "bank2010modal",
        ],
        version="1.0",
    )


def create_hybrid_template() -> ScoringTemplate:
    """
    Create hybrid therapy + audio template.
    
    Balanced approach for applications needing both:
    - Good spine vibration for relaxation
    - Reasonable ear balance for music enjoyment
    
    Use case: Sound baths, relaxation with music
    """
    return ScoringTemplate(
        name="Hybrid",
        template_type=TemplateType.HYBRID,
        description="Balanced therapy and audio - good for sound baths",
        zone_configs={
            "spine": ZoneScoringConfig(
                zone_name="spine",
                weight=0.45,
                frequency_targets=[
                    FrequencyTarget(
                        band=(30, 120),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=6.0,
                        weight=1.0,
                    ),
                ],
                min_energy_db=-12,
            ),
            "ears": ZoneScoringConfig(
                zone_name="ears",
                weight=0.35,
                frequency_targets=[
                    FrequencyTarget(
                        band=(40, 300),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=6.0,
                        weight=1.0,
                    ),
                ],
                requires_lr_balance=True,
                lr_tolerance_db=4.0,
            ),
            "sacrum": ZoneScoringConfig(
                zone_name="sacrum",
                weight=0.2,
                frequency_targets=[
                    FrequencyTarget(
                        band=(15, 60),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=8.0,
                        weight=0.8,
                    ),
                ],
            ),
        },
        freq_range=(20, 300),
        flatness_weight=1.0,
        energy_weight=1.0,
        balance_weight=1.0,
        structural_weight=0.5,
        mass_weight=0.3,
        allow_peninsulas=True,
        cutout_style="lutherie",
        paper_references=[
            "skille1989vibroacoustic",
            "harris2010dml",
        ],
        version="1.0",
    )


def create_meditation_template() -> ScoringTemplate:
    """
    Create meditation-focused template.
    
    Optimized for very low frequencies:
    - Theta wave enhancement (4-8Hz)
    - Deep bass immersion
    - Minimal audio distraction
    
    Use case: Deep meditation, sleep induction
    """
    return ScoringTemplate(
        name="Meditation",
        template_type=TemplateType.MEDITATION,
        description="Low frequency focus for deep meditation and theta enhancement",
        zone_configs={
            "spine": ZoneScoringConfig(
                zone_name="spine",
                weight=0.5,
                frequency_targets=[
                    FrequencyTarget(
                        band=(4, 15),
                        response_type="boost",
                        target_db=3.0,
                        tolerance_db=4.0,
                        weight=1.0,
                    ),
                    FrequencyTarget(
                        band=(15, 60),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=6.0,
                        weight=0.8,
                    ),
                ],
                min_energy_db=-8,
            ),
            "sacrum": ZoneScoringConfig(
                zone_name="sacrum",
                weight=0.3,
                frequency_targets=[
                    FrequencyTarget(
                        band=(8, 40),
                        response_type="flat",
                        target_db=0.0,
                        tolerance_db=6.0,
                        weight=1.0,
                    ),
                ],
            ),
            "ears": ZoneScoringConfig(
                zone_name="ears",
                weight=0.2,
                frequency_targets=[
                    FrequencyTarget(
                        band=(30, 100),
                        response_type="flat",
                        target_db=-6.0,  # Reduced for less distraction
                        tolerance_db=10.0,
                        weight=0.5,
                    ),
                ],
                requires_lr_balance=False,  # Less critical for meditation
            ),
        },
        freq_range=(4, 100),
        flatness_weight=0.6,
        energy_weight=1.2,
        balance_weight=0.2,
        structural_weight=0.6,
        mass_weight=0.2,
        allow_peninsulas=True,
        cutout_style="lutherie",
        paper_references=[
            "skille1989vibroacoustic",
            "griffin1990handbook",
        ],
        version="1.0",
    )


def create_research_template(
    name: str = "Custom Research",
    zones: Optional[Dict[str, Dict]] = None,
) -> ScoringTemplate:
    """
    Create customizable research template.
    
    For researchers who want full control over scoring.
    
    Args:
        name: Template name
        zones: Optional zone configuration dict
    
    Returns:
        Customizable ScoringTemplate
    """
    zone_configs = {}
    
    if zones:
        for zone_name, cfg in zones.items():
            freq_targets = []
            for ft in cfg.get("frequency_targets", []):
                freq_targets.append(FrequencyTarget(
                    band=tuple(ft.get("band", (20, 200))),
                    response_type=ft.get("response_type", "flat"),
                    target_db=ft.get("target_db", 0.0),
                    tolerance_db=ft.get("tolerance_db", 6.0),
                    weight=ft.get("weight", 1.0),
                ))
            
            zone_configs[zone_name] = ZoneScoringConfig(
                zone_name=zone_name,
                weight=cfg.get("weight", 0.5),
                frequency_targets=freq_targets,
                min_energy_db=cfg.get("min_energy_db", -20),
                max_energy_db=cfg.get("max_energy_db", 6),
                requires_lr_balance=cfg.get("requires_lr_balance", False),
                lr_tolerance_db=cfg.get("lr_tolerance_db", 3),
            )
    
    return ScoringTemplate(
        name=name,
        template_type=TemplateType.RESEARCH,
        description="Custom research configuration",
        zone_configs=zone_configs,
        freq_range=(20, 500),
        flatness_weight=1.0,
        energy_weight=1.0,
        balance_weight=1.0,
        structural_weight=0.5,
        mass_weight=0.3,
        allow_peninsulas=True,
        cutout_style="lutherie",
        version="1.0",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class TemplateRegistry:
    """
    Registry for scoring templates.
    
    Manages predefined and custom templates.
    
    USAGE:
        registry = TemplateRegistry()
        
        # Get predefined template
        vat = registry.get("VAT Therapy")
        
        # Register custom template
        registry.register(my_template)
        
        # List available templates
        names = registry.list_templates()
    """
    
    def __init__(self, load_defaults: bool = True):
        """
        Initialize template registry.
        
        Args:
            load_defaults: Whether to load predefined templates
        """
        self._templates: Dict[str, ScoringTemplate] = {}
        
        if load_defaults:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load predefined templates."""
        self.register(create_vat_therapy_template())
        self.register(create_binaural_audio_template())
        self.register(create_hybrid_template())
        self.register(create_meditation_template())
        
        logger.info(f"Loaded {len(self._templates)} default templates")
    
    def register(self, template: ScoringTemplate):
        """Register a template."""
        self._templates[template.name] = template
    
    def get(self, name: str) -> Optional[ScoringTemplate]:
        """Get template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List registered template names."""
        return list(self._templates.keys())
    
    def get_by_type(self, template_type: TemplateType) -> List[ScoringTemplate]:
        """Get all templates of a specific type."""
        return [t for t in self._templates.values() if t.template_type == template_type]
    
    def save_to_file(self, path: Path):
        """Save all templates to JSON file."""
        data = {name: t.to_json() for name, t in self._templates.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, path: Path):
        """Load templates from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        for name, json_str in data.items():
            template = ScoringTemplate.from_json(json_str)
            self.register(template)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE ADAPTER (For FitnessEvaluator Integration)
# ══════════════════════════════════════════════════════════════════════════════

class TemplateAdapter:
    """
    Adapts ScoringTemplate for use with FitnessEvaluator.
    
    Bridges the new template system with existing fitness code.
    
    USAGE:
        template = create_vat_therapy_template()
        adapter = TemplateAdapter(template)
        
        # Get legacy format
        objective_weights = adapter.get_objective_weights()
        zone_weights = adapter.get_zone_weights()
        
        # Or create evaluator directly
        evaluator = adapter.create_evaluator(person)
    """
    
    def __init__(self, template: ScoringTemplate):
        self.template = template
    
    def get_objective_weights(self):
        """Get ObjectiveWeights for legacy evaluator."""
        # Import here to avoid circular dependency
        from .fitness import ObjectiveWeights
        
        weights = self.template.to_objective_weights()
        return ObjectiveWeights(
            flatness=weights["flatness"],
            spine_coupling=weights["spine_coupling"],
            low_mass=weights["low_mass"],
            manufacturability=weights["manufacturability"],
        )
    
    def get_zone_weights(self):
        """Get ZoneWeights for legacy evaluator."""
        from .fitness import ZoneWeights
        
        weights = self.template.to_zone_weights()
        return ZoneWeights(
            spine=weights["spine"],
            head=weights["head"],
        )
    
    def get_freq_range(self) -> Tuple[float, float]:
        """Get frequency range."""
        return self.template.freq_range
    
    def get_n_freq_points(self) -> int:
        """Get number of frequency points."""
        return self.template.n_freq_points
    
    def create_evaluator(self, person, material: str = "birch_plywood"):
        """
        Create FitnessEvaluator with template settings.
        
        Args:
            person: Person instance
            material: Material name
        
        Returns:
            Configured FitnessEvaluator
        """
        from .fitness import FitnessEvaluator
        
        return FitnessEvaluator(
            person=person,
            objectives=self.get_objective_weights(),
            zone_weights=self.get_zone_weights(),
            material=material,
            freq_range=self.get_freq_range(),
            n_freq_points=self.get_n_freq_points(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_template(name_or_type: Union[str, TemplateType]) -> ScoringTemplate:
    """
    Get a scoring template by name or type.
    
    Args:
        name_or_type: Template name string or TemplateType enum
    
    Returns:
        ScoringTemplate instance
    
    Examples:
        template = get_template("VAT Therapy")
        template = get_template(TemplateType.BINAURAL_AUDIO)
    """
    registry = TemplateRegistry()
    
    if isinstance(name_or_type, TemplateType):
        templates = registry.get_by_type(name_or_type)
        if templates:
            return templates[0]
        raise ValueError(f"No template found for type {name_or_type}")
    
    template = registry.get(name_or_type)
    if template is None:
        raise ValueError(f"Template '{name_or_type}' not found")
    return template


def list_available_templates() -> List[Dict[str, str]]:
    """
    List all available templates with descriptions.
    
    Returns:
        List of dicts with 'name', 'type', and 'description'
    """
    registry = TemplateRegistry()
    return [
        {
            "name": t.name,
            "type": t.template_type.name,
            "description": t.description,
        }
        for t in registry._templates.values()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

# Global registry instance
_default_registry: Optional[TemplateRegistry] = None


def get_default_registry() -> TemplateRegistry:
    """Get or create default template registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TemplateRegistry()
    return _default_registry
