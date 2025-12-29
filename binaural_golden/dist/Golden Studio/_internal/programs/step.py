"""
Step - Single phase in a program sequence
==========================================

A Step represents one phase of a sound journey with:
- Duration
- Frequencies with amplitudes
- Spatial positions (for vibroacoustic)
- Fade in/out transitions
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum


class FadeCurve(Enum):
    """Fade curve types"""
    LINEAR = "linear"
    GOLDEN = "golden"      # φ-based S-curve
    EXPONENTIAL = "exponential"
    INSTANT = "instant"    # No fade


@dataclass
class FrequencyConfig:
    """Configuration for a single frequency component"""
    frequency_hz: float
    amplitude: float = 1.0           # 0.0 to 1.0
    phase_offset_rad: float = 0.0    # Phase offset in radians
    label: str = ""                  # Optional name (e.g., "Root", "4th")
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FrequencyConfig':
        return cls(**data)


@dataclass
class PositionConfig:
    """Configuration for spatial position (vibroacoustic)"""
    position_mm: float              # Position on board in mm (0=HEAD, 1950=FEET)
    label: str = ""                 # Optional name (e.g., "SOLAR_PLEXUS")
    
    # Derived pan value (-1 to +1)
    @property
    def pan(self) -> float:
        """Convert mm position to pan value"""
        return (self.position_mm / 975.0) - 1.0
    
    def to_dict(self) -> dict:
        return {'position_mm': self.position_mm, 'label': self.label}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PositionConfig':
        return cls(**data)


# Standard body positions
BODY_POSITIONS = {
    'HEAD': PositionConfig(0.0, 'HEAD'),
    'THROAT': PositionConfig(200.0, 'THROAT'),
    'HEART': PositionConfig(450.0, 'HEART'),
    'SOLAR_PLEXUS': PositionConfig(600.0, 'SOLAR_PLEXUS'),
    'SACRAL': PositionConfig(800.0, 'SACRAL'),
    'ROOT': PositionConfig(1000.0, 'ROOT'),
    'KNEES': PositionConfig(1400.0, 'KNEES'),
    'FEET': PositionConfig(1750.0, 'FEET'),
}


@dataclass
class Step:
    """
    A single step/phase in a program.
    
    Example: Phase 1 of Chakra Sunrise
        Step(
            name="4th Emerging",
            duration_sec=17.5,  # 14.6% of 120s
            frequencies=[FrequencyConfig(144.0, 1.0, 0.0, "Perfect 4th")],
            positions=[PositionConfig(600.0, "SOLAR_PLEXUS")],
            fade_in_sec=17.5,
            fade_curve=FadeCurve.GOLDEN
        )
    """
    name: str
    duration_sec: float
    frequencies: List[FrequencyConfig] = field(default_factory=list)
    positions: List[PositionConfig] = field(default_factory=list)
    
    # Fade settings
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    fade_curve: FadeCurve = FadeCurve.GOLDEN
    
    # Movement (for position transitions)
    position_start: Optional[List[PositionConfig]] = None  # Start positions if different
    
    # Metadata
    description: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'name': self.name,
            'duration_sec': self.duration_sec,
            'frequencies': [f.to_dict() for f in self.frequencies],
            'positions': [p.to_dict() for p in self.positions],
            'fade_in_sec': self.fade_in_sec,
            'fade_out_sec': self.fade_out_sec,
            'fade_curve': self.fade_curve.value,
            'position_start': [p.to_dict() for p in self.position_start] if self.position_start else None,
            'description': self.description,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Step':
        """Deserialize from dictionary"""
        return cls(
            name=data['name'],
            duration_sec=data['duration_sec'],
            frequencies=[FrequencyConfig.from_dict(f) for f in data.get('frequencies', [])],
            positions=[PositionConfig.from_dict(p) for p in data.get('positions', [])],
            fade_in_sec=data.get('fade_in_sec', 0.0),
            fade_out_sec=data.get('fade_out_sec', 0.0),
            fade_curve=FadeCurve(data.get('fade_curve', 'golden')),
            position_start=[PositionConfig.from_dict(p) for p in data['position_start']] if data.get('position_start') else None,
            description=data.get('description', ''),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Step':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))
