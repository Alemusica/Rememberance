"""
Program Generators - Create sequences programmatically
========================================================

Instead of writing hundreds of lines of JSON, define sequences
with simple parameters and let the generator do the work.

Usage:
    from programs.generators import orchestra_tuning, chakra_journey
    
    # Generate orchestra tuning at 432 Hz, 4 minutes
    program = orchestra_tuning(432.0, duration_min=4)
    
    # Generate chakra sequence
    program = chakra_journey(base_freq=108.0, duration_min=10)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import math

from .step import Step, FrequencyConfig, PositionConfig, FadeCurve, BODY_POSITIONS
from .program import Program

# Golden ratio
PHI = 1.618033988749895


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def harmonic_series(fundamental: float, num_harmonics: int = 8) -> List[float]:
    """Generate harmonic series from fundamental"""
    return [fundamental * (n + 1) for n in range(num_harmonics)]


def octave_series(frequency: float, octaves_down: int = 3, octaves_up: int = 2) -> List[float]:
    """Generate octave series (A1, A2, A3, A4, A5, A6...)"""
    result = []
    for i in range(-octaves_down, octaves_up + 1):
        result.append(frequency * (2 ** i))
    return sorted(result)


def fifth_above(freq: float) -> float:
    """Perfect fifth (3:2 ratio)"""
    return freq * 3 / 2


def fourth_above(freq: float) -> float:
    """Perfect fourth (4:3 ratio)"""
    return freq * 4 / 3


def major_third_above(freq: float) -> float:
    """Major third (5:4 ratio in just intonation)"""
    return freq * 5 / 4


def golden_distribute(total: float, num_parts: int) -> List[float]:
    """Distribute time using golden ratio proportions"""
    if num_parts <= 1:
        return [total]
    
    # Use golden ratio to create pleasant proportions
    weights = []
    for i in range(num_parts):
        # Alternate between 1 and φ for balance
        weights.append(PHI if i % 2 == 1 else 1.0)
    
    total_weight = sum(weights)
    return [w / total_weight * total for w in weights]


def crescendo_amplitudes(num_steps: int, start: float = 0.1, peak: float = 1.0) -> List[float]:
    """Generate crescendo amplitude curve"""
    if num_steps <= 1:
        return [peak]
    return [start + (peak - start) * (i / (num_steps - 1)) ** 0.7 for i in range(num_steps)]


def decrescendo_amplitudes(num_steps: int, start: float = 1.0, end: float = 0.0) -> List[float]:
    """Generate decrescendo amplitude curve"""
    if num_steps <= 1:
        return [start]
    return [start - (start - end) * (i / (num_steps - 1)) ** 0.7 for i in range(num_steps)]


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRA TUNING GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestraSection:
    """Define an orchestra section for tuning sequence"""
    name: str
    frequencies: List[float]
    amplitudes: List[float]
    positions: List[str]
    duration_ratio: float  # Relative duration (1.0 = base unit)
    fade_in_ratio: float = 0.2  # Portion of duration for fade in


# Standard orchestra sections for tuning
ORCHESTRA_SECTIONS = {
    'silence': OrchestraSection(
        name="Silence",
        frequencies=[],
        amplitudes=[],
        positions=['SOLAR_PLEXUS'],
        duration_ratio=0.5,
        fade_in_ratio=0.0
    ),
    'oboe': OrchestraSection(
        name="Oboe - Reference A",
        frequencies=[1.0, 2.0],  # Ratios to fundamental
        amplitudes=[0.85, 0.15],
        positions=['SOLAR_PLEXUS'],
        duration_ratio=0.8,
        fade_in_ratio=0.1
    ),
    'concertmaster': OrchestraSection(
        name="Concertmaster",
        frequencies=[1.0, 2.0, 3.0],
        amplitudes=[0.7, 0.2, 0.1],
        positions=['HEART', 'SOLAR_PLEXUS'],
        duration_ratio=1.2,
        fade_in_ratio=0.2
    ),
    'first_violins': OrchestraSection(
        name="First Violins",
        frequencies=[1.0, 1.001, 0.999, 2.0],  # Slight detuning for realism
        amplitudes=[0.65, 0.15, 0.15, 0.15],
        positions=['THROAT', 'HEART', 'SOLAR_PLEXUS'],
        duration_ratio=1.5,
        fade_in_ratio=0.2
    ),
    'second_violins': OrchestraSection(
        name="Second Violins",
        frequencies=[1.0, 2.0, 3.0, 4.0],
        amplitudes=[0.6, 0.2, 0.1, 0.05],
        positions=['THROAT', 'HEART', 'SOLAR_PLEXUS'],
        duration_ratio=1.2,
        fade_in_ratio=0.2
    ),
    'violas': OrchestraSection(
        name="Violas",
        frequencies=[1.0, 0.667, 2.0, 1.333],  # A, D, A5, D5
        amplitudes=[0.5, 0.25, 0.15, 0.1],
        positions=['HEART', 'SOLAR_PLEXUS', 'SACRAL'],
        duration_ratio=1.0,
        fade_in_ratio=0.2
    ),
    'cellos': OrchestraSection(
        name="Cellos",
        frequencies=[0.5, 0.333, 1.0, 0.667],  # A3, D3, A4, D4
        amplitudes=[0.5, 0.25, 0.3, 0.15],
        positions=['SOLAR_PLEXUS', 'SACRAL', 'ROOT'],
        duration_ratio=1.2,
        fade_in_ratio=0.25
    ),
    'basses': OrchestraSection(
        name="Double Basses",
        frequencies=[0.25, 0.167, 0.5, 1.0],  # A2, D2, A3, A4
        amplitudes=[0.45, 0.3, 0.35, 0.2],
        positions=['SACRAL', 'ROOT', 'KNEES', 'FEET'],
        duration_ratio=1.0,
        fade_in_ratio=0.3
    ),
    'woodwinds': OrchestraSection(
        name="Woodwinds",
        frequencies=[1.0, 2.0, 0.5, 1.5],  # A4, A5, A3, E5
        amplitudes=[0.4, 0.3, 0.25, 0.15],
        positions=['THROAT', 'HEART', 'SOLAR_PLEXUS'],
        duration_ratio=1.2,
        fade_in_ratio=0.2
    ),
    'brass_high': OrchestraSection(
        name="Horns & Trumpets",
        frequencies=[1.0, 0.5, 2.0, 1.5, 2.5],
        amplitudes=[0.35, 0.25, 0.25, 0.15, 0.1],
        positions=['THROAT', 'HEART', 'SOLAR_PLEXUS'],
        duration_ratio=1.2,
        fade_in_ratio=0.25
    ),
    'brass_low': OrchestraSection(
        name="Trombones & Tuba",
        frequencies=[1.0, 0.5, 0.25, 0.125],  # A4, A3, A2, A1
        amplitudes=[0.3, 0.3, 0.25, 0.15],
        positions=['SOLAR_PLEXUS', 'SACRAL', 'ROOT', 'KNEES'],
        duration_ratio=1.0,
        fade_in_ratio=0.25
    ),
    'timpani': OrchestraSection(
        name="Timpani",
        frequencies=[1.0, 0.333, 0.25, 0.5],  # A4, D3, A2, A3
        amplitudes=[0.25, 0.35, 0.25, 0.15],
        positions=['SACRAL', 'ROOT', 'KNEES', 'FEET'],
        duration_ratio=0.8,
        fade_in_ratio=0.2
    ),
    'tutti': OrchestraSection(
        name="Full Orchestra",
        frequencies=[0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],  # A0 to A6+
        amplitudes=[0.1, 0.2, 0.35, 0.5, 1.0, 0.5, 0.25, 0.15, 0.08],
        positions=['HEAD', 'THROAT', 'HEART', 'SOLAR_PLEXUS', 'SACRAL', 'ROOT', 'FEET'],
        duration_ratio=2.0,
        fade_in_ratio=0.25
    ),
}


def orchestra_tuning(
    fundamental: float = 432.0,
    duration_min: float = 4.0,
    sections: Optional[List[str]] = None,
    include_crescendo: bool = True,
    include_decrescendo: bool = True,
) -> Program:
    """
    Generate an orchestra tuning sequence.
    
    Args:
        fundamental: Base frequency (default 432 Hz for Verdi pitch)
        duration_min: Total duration in minutes
        sections: List of section names to include (None = all)
        include_crescendo: Add tutti crescendo at end
        include_decrescendo: Add fade out at end
    
    Returns:
        Program object ready to play or save
    
    Example:
        program = orchestra_tuning(432.0, duration_min=5)
        program.save('my_tuning.json')
    """
    duration_sec = duration_min * 60
    
    # Default section order
    if sections is None:
        sections = [
            'silence', 'oboe', 'concertmaster', 'first_violins',
            'second_violins', 'violas', 'cellos', 'basses',
            'woodwinds', 'brass_high', 'brass_low', 'timpani'
        ]
    
    # Calculate total duration ratio
    total_ratio = sum(ORCHESTRA_SECTIONS[s].duration_ratio for s in sections if s in ORCHESTRA_SECTIONS)
    
    # Add crescendo/decrescendo portions
    if include_crescendo:
        total_ratio += 4.0  # Tutti + peak
    if include_decrescendo:
        total_ratio += 3.0  # Fade + resonance + silence
    
    # Duration per unit ratio
    unit_duration = duration_sec / total_ratio
    
    steps = []
    
    # Generate steps for each section
    for section_name in sections:
        if section_name not in ORCHESTRA_SECTIONS:
            continue
            
        section = ORCHESTRA_SECTIONS[section_name]
        step_duration = section.duration_ratio * unit_duration
        
        # Build frequency configs
        freqs = []
        for i, ratio in enumerate(section.frequencies):
            if ratio > 0:
                amp = section.amplitudes[i] if i < len(section.amplitudes) else 0.5
                freqs.append(FrequencyConfig(
                    frequency_hz=fundamental * ratio,
                    amplitude=amp,
                    phase_offset_rad=0.0,
                    label=f"{fundamental * ratio:.1f} Hz"
                ))
        
        # Build position configs
        positions = [
            PositionConfig(BODY_POSITIONS[p].position_mm, p)
            for p in section.positions if p in BODY_POSITIONS
        ]
        
        # Handle silence step specially
        if not freqs:
            freqs = [FrequencyConfig(fundamental, 0.0, 0.0, "Silence")]
        
        steps.append(Step(
            name=section.name,
            duration_sec=step_duration,
            frequencies=freqs,
            positions=positions or [PositionConfig(600.0, 'CENTER')],
            fade_in_sec=step_duration * section.fade_in_ratio,
            fade_out_sec=step_duration * 0.15,
            fade_curve=FadeCurve.GOLDEN,
            description=f"{section.name} tunes to {fundamental} Hz"
        ))
    
    # Add tutti crescendo
    if include_crescendo:
        tutti = ORCHESTRA_SECTIONS['tutti']
        
        # Tutti entrance
        tutti_freqs = [
            FrequencyConfig(fundamental * r, a, 0.0, f"A×{r}")
            for r, a in zip(tutti.frequencies, tutti.amplitudes)
        ]
        tutti_positions = [
            PositionConfig(BODY_POSITIONS[p].position_mm, p)
            for p in tutti.positions if p in BODY_POSITIONS
        ]
        
        steps.append(Step(
            name="Tutti - Full Orchestra",
            duration_sec=2.0 * unit_duration,
            frequencies=tutti_freqs,
            positions=tutti_positions,
            fade_in_sec=0.5 * unit_duration,
            fade_out_sec=0.0,
            fade_curve=FadeCurve.GOLDEN,
            description="Complete orchestra sounds unified A"
        ))
        
        # Peak
        steps.append(Step(
            name="Tutti - Peak",
            duration_sec=2.0 * unit_duration,
            frequencies=tutti_freqs,
            positions=tutti_positions,
            fade_in_sec=0.8 * unit_duration,
            fade_out_sec=0.0,
            fade_curve=FadeCurve.GOLDEN,
            description="Maximum resonance"
        ))
    
    # Add decrescendo
    if include_decrescendo:
        # Fade out
        fade_freqs = [
            FrequencyConfig(fundamental * 0.125, 0.15, 0.0, "A1"),
            FrequencyConfig(fundamental * 0.25, 0.25, 0.0, "A2"),
            FrequencyConfig(fundamental * 0.5, 0.35, 0.0, "A3"),
            FrequencyConfig(fundamental, 0.8, 0.0, "A4"),
            FrequencyConfig(fundamental * 2, 0.3, 0.0, "A5"),
        ]
        
        steps.append(Step(
            name="Decrescendo",
            duration_sec=1.5 * unit_duration,
            frequencies=fade_freqs,
            positions=[
                PositionConfig(450.0, 'HEART'),
                PositionConfig(600.0, 'SOLAR_PLEXUS'),
                PositionConfig(800.0, 'SACRAL'),
            ],
            fade_in_sec=0.0,
            fade_out_sec=1.2 * unit_duration,
            fade_curve=FadeCurve.GOLDEN,
            description="Orchestra fades"
        ))
        
        # Resonance
        steps.append(Step(
            name="Resonance",
            duration_sec=1.0 * unit_duration,
            frequencies=[
                FrequencyConfig(fundamental, 0.2, 0.0, "Echo"),
                FrequencyConfig(fundamental * 0.5, 0.1, 0.0, "Memory"),
            ],
            positions=[PositionConfig(600.0, 'SOLAR_PLEXUS')],
            fade_in_sec=0.0,
            fade_out_sec=0.8 * unit_duration,
            fade_curve=FadeCurve.GOLDEN,
            description="Hall resonates"
        ))
        
        # Final silence
        steps.append(Step(
            name="Silence",
            duration_sec=0.5 * unit_duration,
            frequencies=[FrequencyConfig(fundamental, 0.0, 0.0, "Silence")],
            positions=[PositionConfig(600.0, 'CENTER')],
            fade_in_sec=0.0,
            fade_out_sec=0.0,
            fade_curve=FadeCurve.INSTANT,
            description="Ready to perform"
        ))
    
    return Program(
        name=f"Orchestra Tuning {fundamental:.0f} Hz",
        description=f"Authentic orchestral tuning sequence at A={fundamental:.0f} Hz",
        base_frequency=fundamental,
        steps=steps,
        author="Rememberance Generator",
        tags=["orchestra", f"{fundamental:.0f}hz", "tuning", "generated"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# HARMONIC MEDITATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def harmonic_meditation(
    fundamental: float = 432.0,
    duration_min: float = 10.0,
    num_harmonics: int = 7,
    mode: str = 'ascending',  # 'ascending', 'descending', 'converging'
) -> Program:
    """
    Generate a harmonic meditation sequence.
    
    Args:
        fundamental: Base frequency
        duration_min: Total duration in minutes
        num_harmonics: How many harmonics to include (2-12)
        mode: How harmonics appear
            'ascending' - from fundamental up
            'descending' - from highest down
            'converging' - from extremes to middle
    
    Returns:
        Program with smooth harmonic transitions
    """
    duration_sec = duration_min * 60
    harmonics = harmonic_series(fundamental, num_harmonics)
    
    # Calculate step durations using golden distribution
    num_steps = num_harmonics + 2  # +2 for intro and outro
    durations = golden_distribute(duration_sec, num_steps)
    
    # Determine order
    if mode == 'descending':
        harmonics = list(reversed(harmonics))
    elif mode == 'converging':
        # Interleave from both ends
        reordered = []
        left, right = 0, len(harmonics) - 1
        while left <= right:
            if left == right:
                reordered.append(harmonics[left])
            else:
                reordered.append(harmonics[left])
                reordered.append(harmonics[right])
            left += 1
            right -= 1
        harmonics = reordered
    
    steps = []
    current_freqs = []
    
    # Intro - silence with anticipation
    steps.append(Step(
        name="Opening Breath",
        duration_sec=durations[0],
        frequencies=[FrequencyConfig(fundamental, 0.05, 0.0, "Whisper")],
        positions=[PositionConfig(600.0, 'SOLAR_PLEXUS')],
        fade_in_sec=durations[0] * 0.8,
        fade_out_sec=0.0,
        fade_curve=FadeCurve.GOLDEN,
        description="The journey begins"
    ))
    
    # Add each harmonic progressively
    for i, freq in enumerate(harmonics):
        current_freqs.append(
            FrequencyConfig(freq, 1.0 / (i + 1), 0.0, f"Harmonic {i+1}")
        )
        
        # Map frequency to body position (low=feet, high=head)
        pos_ratio = (freq - fundamental) / (harmonics[-1] - fundamental + 0.001)
        pos_mm = 1750 - (pos_ratio * 1750)  # 0=HEAD, 1750=FEET
        
        steps.append(Step(
            name=f"Harmonic {i+1}: {freq:.1f} Hz",
            duration_sec=durations[i + 1],
            frequencies=list(current_freqs),  # Copy current state
            positions=[
                PositionConfig(pos_mm, f"Position {i+1}"),
                PositionConfig(600.0, 'CENTER'),
            ],
            fade_in_sec=durations[i + 1] * 0.3,
            fade_out_sec=durations[i + 1] * 0.1,
            fade_curve=FadeCurve.GOLDEN,
            description=f"Harmonic {i+1} emerges at {freq:.1f} Hz"
        ))
    
    # Outro - all harmonics fade together
    steps.append(Step(
        name="Integration",
        duration_sec=durations[-1],
        frequencies=current_freqs,
        positions=[
            PositionConfig(0.0, 'HEAD'),
            PositionConfig(600.0, 'SOLAR_PLEXUS'),
            PositionConfig(1750.0, 'FEET'),
        ],
        fade_in_sec=0.0,
        fade_out_sec=durations[-1] * 0.9,
        fade_curve=FadeCurve.GOLDEN,
        description="All harmonics resonate as one"
    ))
    
    return Program(
        name=f"Harmonic Meditation ({mode})",
        description=f"{num_harmonics} harmonics from {fundamental} Hz, {mode} mode",
        base_frequency=fundamental,
        steps=steps,
        author="Rememberance Generator",
        tags=["meditation", "harmonics", mode, f"{fundamental:.0f}hz"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# BINAURAL SWEEP GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def binaural_sweep(
    carrier: float = 200.0,
    start_beat: float = 10.0,
    end_beat: float = 4.0,
    duration_min: float = 15.0,
    num_steps: int = 5,
) -> Program:
    """
    Generate a binaural beat sweep (e.g., alpha to theta).
    
    Args:
        carrier: Carrier frequency (Hz)
        start_beat: Starting binaural beat frequency
        end_beat: Ending binaural beat frequency
        duration_min: Total duration in minutes
        num_steps: Number of transition steps
    
    Returns:
        Program that smoothly transitions between brain states
    """
    duration_sec = duration_min * 60
    durations = golden_distribute(duration_sec, num_steps)
    
    # Calculate beat frequencies for each step
    beats = [
        start_beat - (start_beat - end_beat) * (i / (num_steps - 1))
        for i in range(num_steps)
    ]
    
    # Brainwave names
    def beat_name(freq):
        if freq >= 12:
            return "Beta (Alert)"
        elif freq >= 8:
            return "Alpha (Relaxed)"
        elif freq >= 4:
            return "Theta (Meditative)"
        else:
            return "Delta (Deep)"
    
    steps = []
    
    for i, (beat, dur) in enumerate(zip(beats, durations)):
        left_freq = carrier - beat / 2
        right_freq = carrier + beat / 2
        
        steps.append(Step(
            name=f"{beat_name(beat)} - {beat:.1f} Hz",
            duration_sec=dur,
            frequencies=[
                FrequencyConfig(left_freq, 1.0, 0.0, "Left"),
                FrequencyConfig(right_freq, 1.0, 0.0, "Right"),
                FrequencyConfig(carrier, 0.3, 0.0, "Center"),
            ],
            positions=[
                PositionConfig(100.0, 'LEFT_EAR'),
                PositionConfig(100.0, 'RIGHT_EAR'),
            ],
            fade_in_sec=dur * 0.2 if i == 0 else dur * 0.1,
            fade_out_sec=dur * 0.2 if i == num_steps - 1 else dur * 0.1,
            fade_curve=FadeCurve.GOLDEN,
            description=f"Binaural beat at {beat:.1f} Hz ({beat_name(beat)})"
        ))
    
    return Program(
        name=f"Binaural Sweep {start_beat:.0f}→{end_beat:.0f} Hz",
        description=f"Transition from {beat_name(start_beat)} to {beat_name(end_beat)}",
        base_frequency=carrier,
        steps=steps,
        author="Rememberance Generator",
        tags=["binaural", "brainwave", "sweep", "meditation"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHAKRA SEQUENCE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

# Traditional chakra frequencies (various systems exist)
CHAKRA_FREQUENCIES = {
    'root': 256.0,      # C4 (or 396 Hz Solfeggio)
    'sacral': 288.0,    # D4 (or 417 Hz)
    'solar': 320.0,     # E4 (or 528 Hz)
    'heart': 341.3,     # F4 (or 639 Hz)
    'throat': 384.0,    # G4 (or 741 Hz)
    'third_eye': 426.7, # A4 (or 852 Hz)
    'crown': 480.0,     # B4 (or 963 Hz)
}

CHAKRA_POSITIONS = {
    'root': 'ROOT',
    'sacral': 'SACRAL',
    'solar': 'SOLAR_PLEXUS',
    'heart': 'HEART',
    'throat': 'THROAT',
    'third_eye': 'HEAD',
    'crown': 'HEAD',
}


def chakra_journey(
    duration_min: float = 20.0,
    direction: str = 'ascending',  # 'ascending', 'descending', 'heart_centered'
    frequency_system: str = 'pythagorean',  # 'pythagorean', 'solfeggio', '432'
) -> Program:
    """
    Generate a chakra activation sequence.
    
    Args:
        duration_min: Total duration in minutes
        direction: Order of chakras
        frequency_system: Which frequency set to use
    
    Returns:
        Program activating each chakra in sequence
    """
    duration_sec = duration_min * 60
    
    # Select frequency system
    if frequency_system == 'solfeggio':
        freqs = {
            'root': 396.0, 'sacral': 417.0, 'solar': 528.0,
            'heart': 639.0, 'throat': 741.0, 'third_eye': 852.0, 'crown': 963.0
        }
    elif frequency_system == '432':
        # Based on A=432 Hz scale
        freqs = {
            'root': 256.87, 'sacral': 288.33, 'solar': 324.0,
            'heart': 342.88, 'throat': 384.87, 'third_eye': 432.0, 'crown': 484.9
        }
    else:  # pythagorean
        freqs = CHAKRA_FREQUENCIES.copy()
    
    # Order chakras
    chakra_order = list(freqs.keys())
    if direction == 'descending':
        chakra_order = list(reversed(chakra_order))
    elif direction == 'heart_centered':
        # heart first, then alternate above/below
        chakra_order = ['heart', 'throat', 'solar', 'third_eye', 'sacral', 'crown', 'root']
    
    # Distribute time (more time for heart)
    weights = [1.5 if c == 'heart' else 1.0 for c in chakra_order]
    total_weight = sum(weights)
    durations = [w / total_weight * duration_sec for w in weights]
    
    steps = []
    active_freqs = []
    
    for chakra, dur in zip(chakra_order, durations):
        freq = freqs[chakra]
        pos = CHAKRA_POSITIONS[chakra]
        
        # Add this chakra's frequency
        active_freqs.append(FrequencyConfig(freq, 0.8, 0.0, chakra.title()))
        
        # Add subtle harmonics
        step_freqs = list(active_freqs)
        step_freqs.append(FrequencyConfig(freq * 2, 0.2, 0.0, f"{chakra.title()} overtone"))
        
        steps.append(Step(
            name=f"{chakra.title().replace('_', ' ')} Chakra",
            duration_sec=dur,
            frequencies=step_freqs,
            positions=[
                PositionConfig(BODY_POSITIONS[pos].position_mm, pos),
            ],
            fade_in_sec=dur * 0.25,
            fade_out_sec=dur * 0.1,
            fade_curve=FadeCurve.GOLDEN,
            description=f"Activating {chakra.replace('_', ' ')} at {freq:.1f} Hz"
        ))
    
    # Integration step - all chakras together
    all_freqs = [
        FrequencyConfig(freqs[c], 0.5, 0.0, c.title())
        for c in freqs.keys()
    ]
    all_positions = [
        PositionConfig(BODY_POSITIONS[CHAKRA_POSITIONS[c]].position_mm, CHAKRA_POSITIONS[c])
        for c in freqs.keys()
    ]
    
    integration_dur = duration_sec * 0.1
    steps.append(Step(
        name="Integration - All Chakras",
        duration_sec=integration_dur,
        frequencies=all_freqs,
        positions=all_positions,
        fade_in_sec=integration_dur * 0.2,
        fade_out_sec=integration_dur * 0.6,
        fade_curve=FadeCurve.GOLDEN,
        description="All chakras resonate together"
    ))
    
    return Program(
        name=f"Chakra Journey ({direction})",
        description=f"7 chakra activation using {frequency_system} frequencies",
        base_frequency=freqs['heart'],  # Heart as base
        steps=steps,
        author="Rememberance Generator",
        tags=["chakra", "meditation", direction, frequency_system],
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def quick_program(
    name: str,
    fundamental: float,
    duration_min: float,
    pattern: str = 'harmonic',
) -> Program:
    """
    Quick way to generate common programs.
    
    Args:
        name: Program name
        fundamental: Base frequency
        duration_min: Duration in minutes
        pattern: 'harmonic', 'binaural', 'chakra', 'orchestra'
    
    Returns:
        Generated Program
    """
    if pattern == 'orchestra':
        return orchestra_tuning(fundamental, duration_min)
    elif pattern == 'binaural':
        return binaural_sweep(fundamental, duration_min=duration_min)
    elif pattern == 'chakra':
        return chakra_journey(duration_min)
    else:  # harmonic
        return harmonic_meditation(fundamental, duration_min)


# Export all generators
__all__ = [
    'orchestra_tuning',
    'harmonic_meditation', 
    'binaural_sweep',
    'chakra_journey',
    'quick_program',
    'harmonic_series',
    'octave_series',
    'ORCHESTRA_SECTIONS',
    'CHAKRA_FREQUENCIES',
]
