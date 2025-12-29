"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 DSP EXPORT - Simulation Results for DSP Agent                 ║
║                                                                              ║
║   Exports plate simulation data in a structured format for DSP processing.   ║
║   The DSP agent can use this data to:                                        ║
║   • Compensate for material limitations via EQ/filtering                     ║
║   • Take advantage of plate resonances                                       ║
║   • Optimize per-channel audio processing based on exciter coupling          ║
║   • Apply zone-specific processing for spine vs head frequencies             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Local imports
from .plate_genome import PlateGenome, ContourType
from .fitness import FitnessResult
from .plate_physics import Material, MATERIALS


@dataclass
class ExciterData:
    """Data for a single exciter channel."""
    channel: int                    # 1-4
    position_x: float               # Normalized 0-1 (0=feet, 1=head)
    position_y: float               # Normalized 0-1 (0=left, 1=right)
    zone: str                       # 'head' or 'feet'
    coupling_efficiency: float      # 0-1, how well it couples to body
    frequency_response_db: List[float]  # dB response at test frequencies
    recommended_gain_db: float      # Suggested gain adjustment
    recommended_eq: Dict[str, float]  # freq_hz -> gain_db suggestions


@dataclass
class ModalResonance:
    """Data for a single modal resonance."""
    mode_index: int                 # 0-based mode number
    mode_mn: Tuple[int, int]        # (m, n) mode numbers
    frequency_hz: float             # Resonant frequency
    q_factor: float                 # Quality factor (sharpness)
    amplitude_relative: float       # Relative amplitude at excitation
    damping_ratio: float            # Damping (from material)
    zone_coupling: Dict[str, float] # How much this mode couples to each zone
    dsp_recommendation: str         # Suggested DSP action


@dataclass 
class TransferFunctionData:
    """Frequency response / transfer function data."""
    frequencies_hz: List[float]     # Frequency points
    magnitude_db: List[float]       # Magnitude in dB
    phase_deg: List[float]          # Phase in degrees
    group_delay_ms: List[float]     # Group delay


@dataclass
class MaterialData:
    """Material properties for DSP agent."""
    name: str
    material_type: str              # 'wood_orthotropic', 'wood_isotropic', 'metal', 'composite'
    density_kg_m3: float
    E_longitudinal_GPa: float       # Young's modulus along grain
    E_transverse_GPa: float         # Young's modulus across grain
    poisson_ratio: float
    damping_ratio: float
    fiber_direction_deg: float      # 0 = along plate length
    speed_of_sound_longitudinal_m_s: float  # c = sqrt(E/rho)
    speed_of_sound_transverse_m_s: float
    acoustic_impedance: float       # Z = rho * c
    
    # DSP-relevant characteristics
    low_frequency_rolloff_hz: float # Where response drops off
    high_frequency_limit_hz: float  # Upper limit for vibration transmission
    resonance_character: str        # 'bright', 'warm', 'neutral', 'damped'
    sustain_character: str          # 'long', 'medium', 'short'


@dataclass
class ZoneData:
    """Data for a body zone (spine or head)."""
    name: str                       # 'spine', 'head'
    weight_in_optimization: float   # 0-1, how much priority
    target_frequencies_hz: List[float]  # Target frequencies for this zone
    achieved_response_db: List[float]   # Actual response achieved
    flatness_deviation_db: float    # How flat the response is (lower = better)
    coupling_score: float           # 0-1, overall coupling quality
    recommended_eq_bands: List[Dict[str, float]]  # EQ suggestions


@dataclass
class PlateGeometryData:
    """Plate geometry for DSP agent reference."""
    length_m: float
    width_m: float
    thickness_mm: float
    contour_type: str
    area_m2: float
    weight_kg: float
    cutout_count: int
    cutout_area_fraction: float
    groove_count: int
    
    # Lutherie-derived characteristics
    stiffness_effective: float      # Effective bending stiffness
    mass_ratio_to_body: float       # Plate mass / typical body mass


@dataclass
class DSPExportResult:
    """
    Complete simulation results package for DSP agent.
    
    This is the handoff data structure from the Plate Designer to the DSP Engineer.
    Contains everything needed to optimize the audio processing chain.
    
    Usage:
        result = DSPExportResult.from_simulation(genome, fitness, material)
        json_str = result.to_json()
        result.save("plate_optimization_results.json")
    """
    
    # Metadata
    export_version: str = "1.0.0"
    export_timestamp: str = ""
    optimization_id: str = ""
    
    # Fitness summary
    total_fitness_score: float = 0.0
    flatness_score: float = 0.0
    spine_coupling_score: float = 0.0
    manufacturability_score: float = 0.0
    
    # Core simulation data
    material: Optional[MaterialData] = None
    geometry: Optional[PlateGeometryData] = None
    exciters: List[ExciterData] = field(default_factory=list)
    modal_resonances: List[ModalResonance] = field(default_factory=list)
    
    # Transfer functions
    plate_response: Optional[TransferFunctionData] = None
    spine_response: Optional[TransferFunctionData] = None
    head_response: Optional[TransferFunctionData] = None
    
    # Zone data
    zones: Dict[str, ZoneData] = field(default_factory=dict)
    
    # DSP recommendations
    global_recommendations: List[str] = field(default_factory=list)
    per_channel_recommendations: Dict[int, List[str]] = field(default_factory=dict)
    suggested_eq_curve: List[Tuple[float, float]] = field(default_factory=list)  # (freq_hz, gain_db)
    
    # Raw data for advanced processing
    frequency_points_hz: List[float] = field(default_factory=list)
    mode_frequencies_hz: List[float] = field(default_factory=list)
    mode_shapes: Optional[List[List[List[float]]]] = None  # [mode][x][y] amplitude
    
    @classmethod
    def from_simulation(
        cls,
        genome: PlateGenome,
        fitness: FitnessResult,
        material: Material,
        person_weight_kg: float = 70.0,
        zone_weights: Optional[Dict[str, float]] = None,
    ) -> 'DSPExportResult':
        """
        Create DSP export from simulation results.
        
        Args:
            genome: Optimized plate genome
            fitness: Fitness evaluation result
            material: Material used
            person_weight_kg: Person weight for mass ratio
            zone_weights: spine/head weights used in optimization
        """
        result = cls()
        
        # Metadata
        result.export_timestamp = datetime.now().isoformat()
        result.optimization_id = f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Fitness scores
        result.total_fitness_score = fitness.total_fitness
        result.flatness_score = fitness.flatness_score
        result.spine_coupling_score = fitness.spine_coupling_score
        result.manufacturability_score = fitness.manufacturability_score
        
        # Material data
        result.material = cls._build_material_data(material, genome.thickness_base)
        
        # Geometry data
        result.geometry = cls._build_geometry_data(genome, material, person_weight_kg)
        
        # Exciter data
        result.exciters = cls._build_exciter_data(genome, fitness)
        
        # Modal resonances
        result.modal_resonances = cls._build_modal_data(fitness, material)
        
        # Transfer functions
        if fitness.frequency_response is not None:
            result.plate_response = cls._build_transfer_function(
                fitness.frequency_response[0],
                fitness.frequency_response[1],
            )
        
        if fitness.spine_response is not None:
            freqs = fitness.frequency_response[0] if fitness.frequency_response else np.linspace(20, 200, 50)
            result.spine_response = cls._build_transfer_function(freqs, fitness.spine_response)
        
        if fitness.head_response is not None:
            freqs = fitness.frequency_response[0] if fitness.frequency_response else np.linspace(20, 200, 50)
            result.head_response = cls._build_transfer_function(freqs, fitness.head_response)
        
        # Zone data
        zone_w = zone_weights or {'spine': 0.70, 'head': 0.30}
        result.zones = cls._build_zone_data(fitness, zone_w)
        
        # Mode frequencies
        result.mode_frequencies_hz = [float(f) for f in fitness.frequencies] if fitness.frequencies else []
        
        # Frequency points
        if fitness.frequency_response is not None:
            result.frequency_points_hz = [float(f) for f in fitness.frequency_response[0]]
        
        # Generate DSP recommendations
        result._generate_recommendations()
        
        return result
    
    @staticmethod
    def _build_material_data(material: Material, thickness: float) -> MaterialData:
        """Build material data structure."""
        # Speed of sound
        c_long = np.sqrt(material.E_longitudinal / material.density)
        c_trans = np.sqrt(material.E_transverse / material.density)
        
        # Acoustic impedance
        Z = material.density * c_long
        
        # Material type classification
        E_ratio = material.E_longitudinal / material.E_transverse
        if E_ratio > 5:
            mat_type = 'wood_orthotropic'
        elif E_ratio > 1.5:
            mat_type = 'wood_isotropic'
        elif material.density > 2000:
            mat_type = 'metal'
        else:
            mat_type = 'composite'
        
        # Resonance character based on damping
        if material.damping_ratio < 0.01:
            res_char = 'bright'
            sustain = 'long'
        elif material.damping_ratio < 0.02:
            res_char = 'neutral'
            sustain = 'medium'
        elif material.damping_ratio < 0.03:
            res_char = 'warm'
            sustain = 'medium'
        else:
            res_char = 'damped'
            sustain = 'short'
        
        # Frequency limits based on material
        # Low frequency rolloff ~ plate stiffness
        f_low = 10.0  # Most plates work down to 10 Hz
        # High frequency ~ speed of sound / thickness
        f_high = min(5000.0, c_long / (10 * thickness))
        
        return MaterialData(
            name=material.name,
            material_type=mat_type,
            density_kg_m3=material.density,
            E_longitudinal_GPa=material.E_longitudinal / 1e9,
            E_transverse_GPa=material.E_transverse / 1e9,
            poisson_ratio=material.poisson_ratio,
            damping_ratio=material.damping_ratio,
            fiber_direction_deg=0.0,  # Default along length
            speed_of_sound_longitudinal_m_s=c_long,
            speed_of_sound_transverse_m_s=c_trans,
            acoustic_impedance=Z,
            low_frequency_rolloff_hz=f_low,
            high_frequency_limit_hz=f_high,
            resonance_character=res_char,
            sustain_character=sustain,
        )
    
    @staticmethod
    def _build_geometry_data(
        genome: PlateGenome,
        material: Material,
        person_weight_kg: float
    ) -> PlateGeometryData:
        """Build geometry data structure."""
        area = genome.length * genome.width
        weight = area * genome.thickness_base * material.density
        
        # Cutout area fraction
        cutout_area = 0.0
        if genome.cutouts:
            for c in genome.cutouts:
                cutout_area += np.pi * c.width * c.height / 4 * area
        cutout_fraction = cutout_area / area if area > 0 else 0
        
        # Effective stiffness (simplified)
        E = material.E_longitudinal
        h = genome.thickness_base
        nu = material.poisson_ratio
        D = E * h**3 / (12 * (1 - nu**2))
        stiffness = D * (1 - 0.3 * cutout_fraction)  # Reduce for cutouts
        
        return PlateGeometryData(
            length_m=genome.length,
            width_m=genome.width,
            thickness_mm=genome.thickness_base * 1000,
            contour_type=genome.contour_type.value if isinstance(genome.contour_type, ContourType) else str(genome.contour_type),
            area_m2=area,
            weight_kg=weight,
            cutout_count=len(genome.cutouts) if genome.cutouts else 0,
            cutout_area_fraction=cutout_fraction,
            groove_count=len(genome.grooves) if genome.grooves else 0,
            stiffness_effective=stiffness,
            mass_ratio_to_body=weight / person_weight_kg,
        )
    
    @staticmethod
    def _build_exciter_data(genome: PlateGenome, fitness: FitnessResult) -> List[ExciterData]:
        """Build exciter data list."""
        exciters = []
        
        freq_response = fitness.frequency_response[1] if fitness.frequency_response else None
        
        for exc in genome.exciters:
            zone = 'head' if exc.y > 0.55 else 'feet'
            
            # Estimate coupling based on position
            # Center positions couple better
            x_dist = abs(exc.x - 0.5)
            y_dist = abs(exc.y - 0.5)
            coupling = max(0.3, 1.0 - x_dist - y_dist * 0.5)
            
            # Frequency response (placeholder if not available)
            if freq_response is not None:
                resp_list = [float(r) for r in freq_response]
            else:
                resp_list = [-6.0] * 50  # Default flat at -6dB
            
            # Recommended EQ based on zone
            if zone == 'head':
                # Head zone: boost mids for clarity
                eq = {'100': 0.0, '250': 2.0, '500': 1.0, '1000': 0.0}
                gain = 0.0
            else:
                # Feet zone: boost bass for tactile
                eq = {'40': 3.0, '80': 2.0, '160': 0.0, '320': -1.0}
                gain = 2.0
            
            exciters.append(ExciterData(
                channel=exc.channel,
                position_x=exc.x,
                position_y=exc.y,
                zone=zone,
                coupling_efficiency=coupling,
                frequency_response_db=resp_list,
                recommended_gain_db=gain,
                recommended_eq=eq,
            ))
        
        return exciters
    
    @staticmethod
    def _build_modal_data(fitness: FitnessResult, material: Material) -> List[ModalResonance]:
        """Build modal resonance data."""
        resonances = []
        
        for i, freq in enumerate(fitness.frequencies or []):
            # Estimate mode numbers (simplified)
            m = (i % 4) + 1
            n = (i // 4) + 1
            
            # Q factor from damping
            Q = 1 / (2 * material.damping_ratio)
            
            # Zone coupling estimate based on mode number
            # Lower modes couple better to large body zones
            if m == 1 and n == 1:
                zone_coupling = {'spine': 0.9, 'head': 0.7, 'feet': 0.8}
                rec = "Primary mode - critical for bass response. Consider slight boost."
            elif m <= 2 and n <= 2:
                zone_coupling = {'spine': 0.7, 'head': 0.5, 'feet': 0.6}
                rec = "Secondary mode - affects mid-bass. May need notch if ringing."
            else:
                zone_coupling = {'spine': 0.4, 'head': 0.3, 'feet': 0.3}
                rec = "Higher mode - contributes to texture. Usually leave flat."
            
            resonances.append(ModalResonance(
                mode_index=i,
                mode_mn=(m, n),
                frequency_hz=float(freq),
                q_factor=Q,
                amplitude_relative=1.0 / (i + 1),  # Decreasing amplitude
                damping_ratio=material.damping_ratio,
                zone_coupling=zone_coupling,
                dsp_recommendation=rec,
            ))
        
        return resonances
    
    @staticmethod
    def _build_transfer_function(
        frequencies: np.ndarray,
        response: np.ndarray
    ) -> TransferFunctionData:
        """Build transfer function data."""
        # Convert to dB if not already
        if np.max(np.abs(response)) < 100:  # Likely already dB
            mag_db = [float(r) for r in response]
        else:
            mag_db = [float(20 * np.log10(abs(r) + 1e-10)) for r in response]
        
        # Estimate phase and group delay (simplified)
        phase = [0.0] * len(frequencies)  # Would need complex response
        
        # Group delay estimate
        freq_list = list(frequencies)
        if len(freq_list) > 1:
            df = freq_list[1] - freq_list[0]
            # Simplified: assume linear phase
            group_delay = [1.0 / (f + 10) * 1000 for f in freq_list]  # ms
        else:
            group_delay = [0.0]
        
        return TransferFunctionData(
            frequencies_hz=[float(f) for f in frequencies],
            magnitude_db=mag_db,
            phase_deg=phase,
            group_delay_ms=group_delay,
        )
    
    @staticmethod
    def _build_zone_data(
        fitness: FitnessResult,
        zone_weights: Dict[str, float]
    ) -> Dict[str, ZoneData]:
        """Build zone data dictionary."""
        zones = {}
        
        # Spine zone
        spine_resp = fitness.spine_response if fitness.spine_response is not None else np.zeros(50)
        spine_dev = float(np.std(spine_resp)) if len(spine_resp) > 0 else 10.0
        
        zones['spine'] = ZoneData(
            name='spine',
            weight_in_optimization=zone_weights.get('spine', 0.7),
            target_frequencies_hz=[40.0, 80.0, 120.0, 160.0],  # VAT therapy frequencies
            achieved_response_db=[float(r) for r in spine_resp[:4]] if len(spine_resp) >= 4 else [-6.0] * 4,
            flatness_deviation_db=spine_dev,
            coupling_score=fitness.spine_coupling_score,
            recommended_eq_bands=[
                {'freq_hz': 40, 'gain_db': 2.0, 'q': 1.0},
                {'freq_hz': 80, 'gain_db': 1.0, 'q': 1.5},
                {'freq_hz': 120, 'gain_db': 0.0, 'q': 2.0},
            ],
        )
        
        # Head zone
        head_resp = fitness.head_response if fitness.head_response is not None else np.zeros(50)
        head_dev = float(np.std(head_resp)) if len(head_resp) > 0 else 10.0
        
        zones['head'] = ZoneData(
            name='head',
            weight_in_optimization=zone_weights.get('head', 0.3),
            target_frequencies_hz=[100.0, 200.0, 400.0, 800.0],  # Audio frequencies
            achieved_response_db=[float(r) for r in head_resp[:4]] if len(head_resp) >= 4 else [-6.0] * 4,
            flatness_deviation_db=head_dev,
            coupling_score=fitness.head_flatness_score,
            recommended_eq_bands=[
                {'freq_hz': 100, 'gain_db': 0.0, 'q': 2.0},
                {'freq_hz': 200, 'gain_db': 1.0, 'q': 2.0},
                {'freq_hz': 400, 'gain_db': 0.0, 'q': 2.0},
            ],
        )
        
        return zones
    
    def _generate_recommendations(self):
        """Generate DSP processing recommendations."""
        recommendations = []
        
        # Material-based recommendations
        if self.material:
            if self.material.resonance_character == 'bright':
                recommendations.append(
                    "Material is BRIGHT: Consider gentle high-shelf cut (-2dB above 1kHz) "
                    "to tame harshness. Material has excellent sustain."
                )
            elif self.material.resonance_character == 'damped':
                recommendations.append(
                    "Material is DAMPED: May need compression to extend sustain. "
                    "Consider transient shaping for attack definition."
                )
            
            if self.material.material_type == 'wood_orthotropic':
                recommendations.append(
                    f"ORTHOTROPIC WOOD detected: Fiber direction matters. "
                    f"E_long/E_trans ratio = {self.material.E_longitudinal_GPa/self.material.E_transverse_GPa:.1f}:1. "
                    f"Higher modes may have directional characteristics."
                )
        
        # Fitness-based recommendations
        if self.flatness_score < 0.7:
            recommendations.append(
                f"FLATNESS SCORE LOW ({self.flatness_score:.2f}): Significant EQ correction needed. "
                "Use parametric EQ to flatten response curve. See suggested_eq_curve."
            )
        
        if self.spine_coupling_score < 0.6:
            recommendations.append(
                f"SPINE COUPLING LOW ({self.spine_coupling_score:.2f}): Consider boosting "
                "40-80 Hz range by +3dB for better tactile response."
            )
        
        # Resonance-based recommendations
        if self.modal_resonances:
            # Find strongest resonance
            strongest = max(self.modal_resonances, key=lambda r: r.amplitude_relative)
            recommendations.append(
                f"PRIMARY RESONANCE at {strongest.frequency_hz:.1f} Hz (mode {strongest.mode_mn}). "
                f"Q={strongest.q_factor:.1f}. Consider notch filter if ringing is audible."
            )
        
        # Per-channel recommendations
        per_channel = {}
        for exc in self.exciters:
            ch_recs = []
            if exc.zone == 'head':
                ch_recs.append(f"CH{exc.channel} (HEAD): Prioritize clarity. HPF at 80Hz recommended.")
            else:
                ch_recs.append(f"CH{exc.channel} (FEET): Prioritize bass. LPF at 200Hz for sub focus.")
            
            if exc.coupling_efficiency < 0.6:
                ch_recs.append(f"  ⚠️ Low coupling ({exc.coupling_efficiency:.2f}). Consider +3dB boost.")
            
            per_channel[exc.channel] = ch_recs
        
        self.global_recommendations = recommendations
        self.per_channel_recommendations = per_channel
        
        # Generate suggested EQ curve
        self._generate_eq_curve()
    
    def _generate_eq_curve(self):
        """Generate suggested EQ correction curve."""
        eq_points = []
        
        # If we have plate response, invert it for correction
        if self.plate_response and len(self.plate_response.magnitude_db) > 0:
            freqs = self.plate_response.frequencies_hz
            mags = self.plate_response.magnitude_db
            
            # Target is 0 dB (flat)
            target = 0.0
            
            # Generate correction points (every 10th point for smooth curve)
            for i in range(0, len(freqs), max(1, len(freqs) // 10)):
                freq = freqs[i]
                correction = target - mags[i]
                # Limit correction range
                correction = max(-12.0, min(12.0, correction))
                eq_points.append((freq, correction))
        else:
            # Default gentle curve for typical plate
            eq_points = [
                (20.0, 3.0),    # Boost sub
                (40.0, 2.0),    # Boost bass
                (80.0, 0.0),    # Flat
                (160.0, -1.0),  # Slight cut
                (320.0, 0.0),   # Flat
                (640.0, 0.0),   # Flat
                (1280.0, -2.0), # Cut highs
            ]
        
        self.suggested_eq_curve = eq_points
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def convert_value(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_value(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            else:
                return obj
        
        return convert_value(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'DSPExportResult':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'DSPExportResult':
        """Create from dictionary."""
        result = cls()
        
        # Simple fields
        for key in ['export_version', 'export_timestamp', 'optimization_id',
                    'total_fitness_score', 'flatness_score', 'spine_coupling_score',
                    'manufacturability_score', 'global_recommendations',
                    'per_channel_recommendations', 'frequency_points_hz',
                    'mode_frequencies_hz']:
            if key in data:
                setattr(result, key, data[key])
        
        # Complex nested structures would need more handling
        # For now, return with basic fields populated
        return result


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def export_for_dsp(
    genome: PlateGenome,
    fitness: FitnessResult,
    material_name: str = "birch_plywood",
    person_weight_kg: float = 70.0,
    zone_weights: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> DSPExportResult:
    """
    Convenience function to export simulation results for DSP agent.
    
    Args:
        genome: Optimized plate genome
        fitness: Fitness evaluation result
        material_name: Material key from MATERIALS dict
        person_weight_kg: Person weight for calculations
        zone_weights: {'spine': 0.7, 'head': 0.3} weights
        save_path: Optional path to save JSON file
    
    Returns:
        DSPExportResult ready for DSP agent consumption
    
    Example:
        >>> result = export_for_dsp(best_genome, best_fitness)
        >>> print(result.to_json())
        >>> # Or pass to DSP agent
        >>> dsp_agent.process(result.to_dict())
    """
    material = MATERIALS.get(material_name, MATERIALS["birch_plywood"])
    
    result = DSPExportResult.from_simulation(
        genome=genome,
        fitness=fitness,
        material=material,
        person_weight_kg=person_weight_kg,
        zone_weights=zone_weights,
    )
    
    if save_path:
        result.save(save_path)
    
    return result
