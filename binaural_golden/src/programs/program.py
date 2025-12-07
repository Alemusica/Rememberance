"""
Program - Complete sequence of steps
=====================================

A Program is a complete sound journey composed of multiple Steps.
Programs can be saved/loaded as JSON and played by the audio engine.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from datetime import datetime

from .step import Step, FrequencyConfig, PositionConfig, FadeCurve, BODY_POSITIONS

# Import golden math - try relative first, then absolute
try:
    from core.golden_math import golden_phase_boundaries, PHI
except ImportError:
    # Fallback: define locally
    PHI = 1.618033988749895
    def golden_phase_boundaries(num_phases):
        if num_phases == 5:
            unit = 1.0 / (2.0 + 3.0 * PHI)
            phi_unit = PHI * unit
            return [0.0, unit, unit + phi_unit, unit + 2*phi_unit, unit + 3*phi_unit, 1.0]
        return [i/num_phases for i in range(num_phases+1)]


@dataclass
class Program:
    """
    A complete sound program/journey.
    
    Example: Chakra Sunrise Journey
        Program(
            name="Chakra Sunrise",
            description="Three frequencies converging at solar plexus",
            base_frequency=108.0,
            steps=[...],
        )
    """
    name: str
    steps: List[Step] = field(default_factory=list)
    description: str = ""
    base_frequency: float = 108.0
    
    # Metadata
    author: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    @property
    def total_duration(self) -> float:
        """Total duration in seconds"""
        return sum(step.duration_sec for step in self.steps)
    
    @property
    def num_steps(self) -> int:
        """Number of steps"""
        return len(self.steps)
    
    def get_step_at_time(self, time_sec: float) -> tuple[int, Step, float]:
        """
        Get the step active at a given time.
        
        Args:
            time_sec: Time in seconds from start
            
        Returns:
            Tuple of (step_index, step, progress_within_step)
        """
        elapsed = 0.0
        for i, step in enumerate(self.steps):
            if elapsed + step.duration_sec > time_sec:
                progress = (time_sec - elapsed) / step.duration_sec
                return (i, step, progress)
            elapsed += step.duration_sec
        
        # Past end - return last step at 100%
        return (len(self.steps) - 1, self.steps[-1], 1.0)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'base_frequency': self.base_frequency,
            'steps': [s.to_dict() for s in self.steps],
            'author': self.author,
            'created': self.created,
            'version': self.version,
            'tags': self.tags,
            'total_duration': self.total_duration,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Program':
        """Deserialize from dictionary"""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            base_frequency=data.get('base_frequency', 108.0),
            steps=[Step.from_dict(s) for s in data.get('steps', [])],
            author=data.get('author', ''),
            created=data.get('created', datetime.now().isoformat()),
            version=data.get('version', '1.0'),
            tags=data.get('tags', []),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Program':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, filepath: str):
        """Save program to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'Program':
        """Load program from JSON file"""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())


# ══════════════════════════════════════════════════════════════════════════════
# PRESET PROGRAMS
# ══════════════════════════════════════════════════════════════════════════════

def create_chakra_sunrise(base_freq: float = 108.0, 
                          duration_sec: float = 120.0) -> Program:
    """
    Create the Chakra Sunrise Journey program.
    
    Three frequencies converging at solar plexus:
    - Perfect 4th (4/3 × base) - always at solar plexus
    - Root (base) - rises from feet to solar plexus
    - Octave (2 × base) - descends from head to solar plexus
    
    Timeline uses golden proportions (1:φ:φ:φ:1):
    - Phase 1 (14.6%): 4th fades in at solar plexus
    - Phase 2 (23.6%): Root fades in at feet, rises to sacral
    - Phase 3 (23.6%): Root continues to solar plexus
    - Phase 4 (23.6%): Octave fades in at head, descends to solar plexus
    - Phase 5 (14.6%): Convergence - all at solar plexus
    """
    # Calculate frequencies
    freq_fourth = base_freq * 4 / 3
    freq_root = base_freq
    freq_octave = base_freq * 2
    
    # Golden phase boundaries
    boundaries = golden_phase_boundaries(5)
    durations = [
        (boundaries[i+1] - boundaries[i]) * duration_sec 
        for i in range(5)
    ]
    
    steps = [
        # Phase 1: 4th emerges at solar plexus
        Step(
            name="4th Emerging",
            duration_sec=durations[0],
            frequencies=[
                FrequencyConfig(freq_fourth, 1.0, 0.0, "Perfect 4th"),
                FrequencyConfig(freq_root, 0.0, 0.0, "Root"),
                FrequencyConfig(freq_octave, 0.0, 0.0, "Octave"),
            ],
            positions=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['FEET'],
                BODY_POSITIONS['HEAD'],
            ],
            fade_in_sec=durations[0],
            fade_curve=FadeCurve.GOLDEN,
            description="Perfect 4th fades in at solar plexus",
        ),
        
        # Phase 2: Root rises from feet
        Step(
            name="Root Rising",
            duration_sec=durations[1],
            frequencies=[
                FrequencyConfig(freq_fourth, 1.0, 0.0, "Perfect 4th"),
                FrequencyConfig(freq_root, 1.0, 0.0, "Root"),
                FrequencyConfig(freq_octave, 0.0, 0.0, "Octave"),
            ],
            positions=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SACRAL'],  # End position
                BODY_POSITIONS['HEAD'],
            ],
            position_start=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['FEET'],    # Start position
                BODY_POSITIONS['HEAD'],
            ],
            fade_in_sec=durations[1],
            fade_curve=FadeCurve.GOLDEN,
            description="Root fades in at feet, rises toward sacral",
        ),
        
        # Phase 3: Root continues to solar plexus
        Step(
            name="Root Through Sacral",
            duration_sec=durations[2],
            frequencies=[
                FrequencyConfig(freq_fourth, 1.0, 0.0, "Perfect 4th"),
                FrequencyConfig(freq_root, 1.0, 0.0, "Root"),
                FrequencyConfig(freq_octave, 0.0, 0.0, "Octave"),
            ],
            positions=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SOLAR_PLEXUS'],  # End
                BODY_POSITIONS['HEAD'],
            ],
            position_start=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SACRAL'],        # Start
                BODY_POSITIONS['HEAD'],
            ],
            fade_curve=FadeCurve.GOLDEN,
            description="Root continues through sacral to solar plexus",
        ),
        
        # Phase 4: Octave descends from head
        Step(
            name="Octave Descending",
            duration_sec=durations[3],
            frequencies=[
                FrequencyConfig(freq_fourth, 1.0, 0.0, "Perfect 4th"),
                FrequencyConfig(freq_root, 1.0, 0.0, "Root"),
                FrequencyConfig(freq_octave, 1.0, 0.0, "Octave"),
            ],
            positions=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SOLAR_PLEXUS'],  # End
            ],
            position_start=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['HEAD'],          # Start
            ],
            fade_in_sec=durations[3],
            fade_curve=FadeCurve.GOLDEN,
            description="Octave fades in at head, descends to solar plexus",
        ),
        
        # Phase 5: Convergence
        Step(
            name="Convergence",
            duration_sec=durations[4],
            frequencies=[
                FrequencyConfig(freq_fourth, 1.0, 0.0, "Perfect 4th"),
                FrequencyConfig(freq_root, 1.0, 0.0, "Root"),
                FrequencyConfig(freq_octave, 1.0, 0.0, "Octave"),
            ],
            positions=[
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SOLAR_PLEXUS'],
                BODY_POSITIONS['SOLAR_PLEXUS'],
            ],
            fade_curve=FadeCurve.GOLDEN,
            description="All three frequencies united at solar plexus (φ point)",
        ),
    ]
    
    return Program(
        name="Chakra Sunrise",
        description="Three frequencies (Root, Perfect 4th, Octave) converging at solar plexus using golden ratio proportions",
        base_frequency=base_freq,
        steps=steps,
        author="Golden Sound Studio",
        tags=["chakra", "vibroacoustic", "golden ratio", "convergence"],
    )
