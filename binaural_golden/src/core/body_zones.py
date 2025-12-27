"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BODY ZONES - Multimodal Analysis System                   ║
║                                                                              ║
║   "L'essere umano è la corda da riaccordare"                                 ║
║   "The human being is the string to retune"                                  ║
║                                                                              ║
║   Zone-based modular analysis for vibroacoustic plate design.               ║
║   NOT hardcoded frequencies - configurable targets per anatomical zone.     ║
║                                                                              ║
║   Physics references:                                                        ║
║   • Griffin (1990) - Handbook of Human Vibration                            ║
║   • Fairley & Griffin - Body resonances 4-80 Hz                              ║
║   • Skille/Bartel - VAT therapy 40 Hz                                       ║
║   • Schleske - Violin body acoustics                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import json

# Golden ratio φ
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618034


# ══════════════════════════════════════════════════════════════════════════════
# BODY ZONE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class ZoneType(Enum):
    """Anatomical zone types with their resonance characteristics."""
    HEAD = "head"
    NECK = "neck"
    CHEST_HEART = "chest_heart"
    SOLAR_PLEXUS = "solar_plexus"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    LEGS_UPPER = "legs_upper"
    LEGS_LOWER = "legs_lower"
    SPINE_CERVICAL = "spine_cervical"
    SPINE_THORACIC = "spine_thoracic"
    SPINE_LUMBAR = "spine_lumbar"
    FULL_BODY = "full_body"


@dataclass
class BodyResonance:
    """
    Body resonance characteristics from Griffin (1990) Handbook of Human Vibration.
    
    Supine human body has characteristic resonances:
    - Whole body (translational): 4-8 Hz
    - Spine (bending): 10-12 Hz
    - Head (pitch/roll): 20-30 Hz
    - Abdomen (visceral): 4-8 Hz
    - Chest wall: 50-60 Hz
    """
    frequency_primary: float  # Primary resonance Hz
    frequency_secondary: Optional[float] = None  # Secondary if exists
    damping_ratio: float = 0.15  # Typical body damping
    effective_mass_fraction: float = 1.0  # Fraction of body mass involved
    quality_factor: float = 3.0  # Q = 1/(2*zeta)


# Body resonance data from literature
BODY_RESONANCES: Dict[ZoneType, BodyResonance] = {
    ZoneType.FULL_BODY: BodyResonance(5.0, 8.0, 0.20, 1.0, 2.5),
    ZoneType.HEAD: BodyResonance(25.0, 30.0, 0.15, 0.08, 3.5),
    ZoneType.NECK: BodyResonance(15.0, 20.0, 0.12, 0.05, 4.0),
    ZoneType.CHEST_HEART: BodyResonance(55.0, 60.0, 0.10, 0.15, 5.0),
    ZoneType.SOLAR_PLEXUS: BodyResonance(40.0, None, 0.15, 0.10, 3.3),
    ZoneType.ABDOMEN: BodyResonance(6.0, 8.0, 0.25, 0.20, 2.0),
    ZoneType.PELVIS: BodyResonance(8.0, 12.0, 0.18, 0.15, 2.8),
    ZoneType.SPINE_CERVICAL: BodyResonance(12.0, 15.0, 0.12, 0.08, 4.2),
    ZoneType.SPINE_THORACIC: BodyResonance(10.0, 12.0, 0.12, 0.10, 4.2),
    ZoneType.SPINE_LUMBAR: BodyResonance(8.0, 10.0, 0.15, 0.12, 3.3),
    ZoneType.LEGS_UPPER: BodyResonance(15.0, None, 0.20, 0.20, 2.5),
    ZoneType.LEGS_LOWER: BodyResonance(20.0, None, 0.15, 0.10, 3.3),
}


@dataclass
class BodyZone:
    """
    Configurable body zone with target frequencies.
    
    Unlike hardcoded CHAKRA_TARGETS, this is fully configurable per zone.
    """
    zone_type: ZoneType
    name: str
    
    # Position on plate (normalized 0-1)
    position_start: float  # Along spine/length
    position_end: float
    lateral_extent: float  # Width fraction (0-0.5 each side)
    
    # Configurable frequency targets (NOT hardcoded!)
    target_frequencies: List[float] = field(default_factory=list)
    frequency_weights: List[float] = field(default_factory=list)  # Importance
    
    # Body resonance to couple with
    body_resonance: Optional[BodyResonance] = None
    
    # Optimization parameters
    optimization_weight: float = 1.0
    coupling_mode: str = "maximize"  # 'maximize', 'minimize', 'match'
    
    # Visual
    color: str = "#888888"
    
    def __post_init__(self):
        """Initialize body resonance from library if not provided."""
        if self.body_resonance is None:
            self.body_resonance = BODY_RESONANCES.get(self.zone_type)
        
        # Ensure weight list matches frequency list
        if len(self.frequency_weights) < len(self.target_frequencies):
            self.frequency_weights.extend(
                [1.0] * (len(self.target_frequencies) - len(self.frequency_weights))
            )
    
    def get_center_position(self) -> Tuple[float, float]:
        """Get center position (x, y) normalized."""
        return (
            (self.position_start + self.position_end) / 2,
            0.5  # Center of width
        )
    
    def get_area_fraction(self) -> float:
        """Get fraction of plate area this zone covers."""
        length_fraction = self.position_end - self.position_start
        width_fraction = 2 * self.lateral_extent
        return length_fraction * width_fraction
    
    def frequency_error(self, achieved_frequencies: List[float]) -> float:
        """
        Calculate weighted frequency error for this zone.
        
        Args:
            achieved_frequencies: Frequencies achieved at this zone from FEM
        
        Returns:
            Weighted sum of squared relative errors
        """
        if not self.target_frequencies or not achieved_frequencies:
            return 0.0
        
        error = 0.0
        for i, target in enumerate(self.target_frequencies):
            weight = self.frequency_weights[i] if i < len(self.frequency_weights) else 1.0
            
            # Find closest achieved frequency
            closest = min(achieved_frequencies, key=lambda f: abs(f - target))
            
            # Relative error
            rel_error = (closest - target) / target
            error += weight * rel_error ** 2
        
        return error * self.optimization_weight


# ══════════════════════════════════════════════════════════════════════════════
# ZONE CONFIGURATION PRESETS
# ══════════════════════════════════════════════════════════════════════════════

def create_chakra_zones() -> List[BodyZone]:
    """
    Create zone configuration based on 7 chakras.
    
    This is ONE possible configuration - user can create their own!
    """
    return [
        BodyZone(
            zone_type=ZoneType.PELVIS,
            name="Muladhara (Root)",
            position_start=0.00, position_end=0.08,
            lateral_extent=0.35,
            target_frequencies=[256.0],  # C4
            frequency_weights=[1.0],
            optimization_weight=1.0,
            color="#ff0000"
        ),
        BodyZone(
            zone_type=ZoneType.ABDOMEN,
            name="Svadhisthana (Sacral)",
            position_start=0.08, position_end=0.20,
            lateral_extent=0.30,
            target_frequencies=[288.0],  # D4
            frequency_weights=[1.0],
            optimization_weight=1.0,
            color="#ff8800"
        ),
        BodyZone(
            zone_type=ZoneType.SOLAR_PLEXUS,
            name="Manipura (Solar)",
            position_start=0.25, position_end=0.38,
            lateral_extent=0.25,
            target_frequencies=[320.0],  # E4
            frequency_weights=[1.0],
            optimization_weight=1.0,
            color="#ffff00"
        ),
        BodyZone(
            zone_type=ZoneType.CHEST_HEART,
            name="Anahata (Heart)",
            position_start=0.38, position_end=0.50,  # Golden ratio point!
            lateral_extent=0.30,
            target_frequencies=[341.3, 40.0],  # F4 + Bartel 40Hz
            frequency_weights=[1.5, 1.0],  # Heart is most important
            optimization_weight=2.0,  # Double weight for heart
            color="#00ff00"
        ),
        BodyZone(
            zone_type=ZoneType.NECK,
            name="Vishuddha (Throat)",
            position_start=0.65, position_end=0.78,
            lateral_extent=0.15,
            target_frequencies=[384.0],  # G4
            frequency_weights=[1.0],
            optimization_weight=1.0,
            color="#00bfff"
        ),
        BodyZone(
            zone_type=ZoneType.HEAD,
            name="Ajna (Third Eye)",
            position_start=0.85, position_end=0.92,
            lateral_extent=0.12,
            target_frequencies=[426.7],  # A4
            frequency_weights=[1.0],
            optimization_weight=1.0,
            color="#4400ff"
        ),
        BodyZone(
            zone_type=ZoneType.HEAD,
            name="Sahasrara (Crown)",
            position_start=0.92, position_end=1.00,
            lateral_extent=0.10,
            target_frequencies=[480.0],  # B4
            frequency_weights=[0.8],
            optimization_weight=0.8,
            color="#ff00ff"
        ),
    ]


def create_body_resonance_zones() -> List[BodyZone]:
    """
    Create zone configuration optimized for body resonance coupling.
    
    Based on Griffin (1990) human vibration handbook.
    Targets low frequency body resonances (4-80 Hz).
    """
    return [
        BodyZone(
            zone_type=ZoneType.FULL_BODY,
            name="Whole Body",
            position_start=0.0, position_end=1.0,
            lateral_extent=0.5,
            target_frequencies=[5.0, 8.0],  # Full body resonance
            frequency_weights=[1.0, 0.8],
            optimization_weight=0.5,
            coupling_mode="maximize",
            color="#cccccc"
        ),
        BodyZone(
            zone_type=ZoneType.SPINE_LUMBAR,
            name="Lumbar Spine",
            position_start=0.10, position_end=0.30,
            lateral_extent=0.20,
            target_frequencies=[8.0, 10.0],
            frequency_weights=[1.0, 0.8],
            optimization_weight=1.5,
            color="#ffa500"
        ),
        BodyZone(
            zone_type=ZoneType.SPINE_THORACIC,
            name="Thoracic Spine",
            position_start=0.30, position_end=0.60,
            lateral_extent=0.25,
            target_frequencies=[10.0, 12.0],
            frequency_weights=[1.0, 0.8],
            optimization_weight=1.5,
            color="#ff8c00"
        ),
        BodyZone(
            zone_type=ZoneType.CHEST_HEART,
            name="Heart/Chest",
            position_start=0.40, position_end=0.55,
            lateral_extent=0.30,
            target_frequencies=[40.0, 55.0],  # Bartel 40Hz + chest wall
            frequency_weights=[2.0, 1.0],  # 40Hz priority
            optimization_weight=2.0,
            coupling_mode="maximize",
            color="#ff0000"
        ),
        BodyZone(
            zone_type=ZoneType.HEAD,
            name="Head",
            position_start=0.85, position_end=1.00,
            lateral_extent=0.15,
            target_frequencies=[25.0, 30.0],
            frequency_weights=[1.0, 0.8],
            optimization_weight=1.0,
            color="#6666ff"
        ),
    ]


def create_vat_therapy_zones() -> List[BodyZone]:
    """
    Create zone configuration for Vibroacoustic Therapy (VAT).
    
    Based on Skille (Norway) and Bartel research.
    Focus on 40 Hz (anxiolytic) and 60-80 Hz (analgesic).
    """
    return [
        BodyZone(
            zone_type=ZoneType.FULL_BODY,
            name="40 Hz Global",
            position_start=0.0, position_end=1.0,
            lateral_extent=0.5,
            target_frequencies=[40.0],  # Key anxiolytic frequency
            frequency_weights=[2.0],
            optimization_weight=3.0,  # Highest priority
            coupling_mode="maximize",
            color="#00ff88"
        ),
        BodyZone(
            zone_type=ZoneType.CHEST_HEART,
            name="Chest 40 Hz",
            position_start=0.35, position_end=0.55,
            lateral_extent=0.35,
            target_frequencies=[40.0, 60.0],
            frequency_weights=[2.0, 1.0],
            optimization_weight=2.0,
            color="#ff4444"
        ),
        BodyZone(
            zone_type=ZoneType.SPINE_LUMBAR,
            name="Lower Back Pain",
            position_start=0.10, position_end=0.30,
            lateral_extent=0.25,
            target_frequencies=[60.0, 80.0],  # Analgesic range
            frequency_weights=[1.5, 1.0],
            optimization_weight=1.5,
            color="#ffaa00"
        ),
        BodyZone(
            zone_type=ZoneType.LEGS_UPPER,
            name="Legs Circulation",
            position_start=0.0, position_end=0.15,
            lateral_extent=0.40,
            target_frequencies=[52.0],  # Circulation
            frequency_weights=[1.0],
            optimization_weight=1.0,
            color="#4488ff"
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ZONE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAnalysisResult:
    """Results of modal analysis for a specific zone."""
    zone: BodyZone
    achieved_frequencies: List[float]
    mode_amplitudes: List[float]  # Vibration amplitude at zone
    frequency_error: float
    coupling_efficiency: float  # 0-1, how well it couples to body
    
    def to_dict(self) -> dict:
        return {
            "zone_name": self.zone.name,
            "zone_type": self.zone.zone_type.value,
            "target_frequencies": self.zone.target_frequencies,
            "achieved_frequencies": self.achieved_frequencies,
            "frequency_error": self.frequency_error,
            "coupling_efficiency": self.coupling_efficiency,
        }


class ZoneAnalyzer:
    """
    Analyzes plate modes at specific body zones.
    
    Takes FEM modal results and evaluates how well they match
    zone-specific frequency targets.
    """
    
    def __init__(self, zones: List[BodyZone]):
        """
        Args:
            zones: List of configured body zones
        """
        self.zones = zones
        self._validate_zones()
    
    def _validate_zones(self):
        """Ensure zones don't overlap excessively."""
        for i, z1 in enumerate(self.zones):
            for j, z2 in enumerate(self.zones):
                if i >= j:
                    continue
                
                # Check overlap
                if (z1.position_start < z2.position_end and 
                    z1.position_end > z2.position_start):
                    # Some overlap is OK, but warn if significant
                    overlap = (min(z1.position_end, z2.position_end) - 
                              max(z1.position_start, z2.position_start))
                    if overlap > 0.1:
                        print(f"⚠️ Zones '{z1.name}' and '{z2.name}' overlap by {overlap:.0%}")
    
    def analyze_mode_at_zone(
        self,
        zone: BodyZone,
        mode_shapes: np.ndarray,  # (n_modes, n_points_x, n_points_y)
        mode_frequencies: np.ndarray,  # (n_modes,)
        plate_length: float,
        plate_width: float
    ) -> ZoneAnalysisResult:
        """
        Analyze which modes contribute to vibration at a specific zone.
        
        Args:
            zone: Body zone to analyze
            mode_shapes: Modal displacement shapes from FEM
            mode_frequencies: Natural frequencies from FEM
            plate_length: Physical plate length [m]
            plate_width: Physical plate width [m]
        
        Returns:
            ZoneAnalysisResult with achieved frequencies and coupling
        """
        n_modes, nx, ny = mode_shapes.shape
        
        # Convert zone position to grid indices
        ix_start = int(zone.position_start * nx)
        ix_end = int(zone.position_end * nx)
        iy_center = ny // 2
        iy_extent = int(zone.lateral_extent * ny)
        iy_start = max(0, iy_center - iy_extent)
        iy_end = min(ny, iy_center + iy_extent)
        
        # Calculate mean amplitude of each mode in this zone
        mode_amplitudes = []
        for m in range(n_modes):
            zone_displacement = mode_shapes[m, ix_start:ix_end, iy_start:iy_end]
            # RMS amplitude (mode shapes are normalized, so this is relative)
            amplitude = np.sqrt(np.mean(zone_displacement ** 2))
            mode_amplitudes.append(amplitude)
        
        mode_amplitudes = np.array(mode_amplitudes)
        
        # Find modes with significant amplitude in this zone
        threshold = 0.3 * np.max(mode_amplitudes)
        active_modes = mode_amplitudes > threshold
        
        # Achieved frequencies are those with significant amplitude
        achieved_frequencies = mode_frequencies[active_modes].tolist()
        
        # Calculate frequency error
        freq_error = zone.frequency_error(achieved_frequencies)
        
        # Calculate coupling efficiency to body resonance
        coupling_eff = self._calculate_coupling_efficiency(
            zone, achieved_frequencies, mode_amplitudes[active_modes]
        )
        
        return ZoneAnalysisResult(
            zone=zone,
            achieved_frequencies=achieved_frequencies,
            mode_amplitudes=mode_amplitudes[active_modes].tolist(),
            frequency_error=freq_error,
            coupling_efficiency=coupling_eff
        )
    
    def _calculate_coupling_efficiency(
        self,
        zone: BodyZone,
        frequencies: List[float],
        amplitudes: np.ndarray
    ) -> float:
        """
        Calculate how well achieved frequencies couple to body resonance.
        
        Uses transfer function approach:
        H(f) = 1 / sqrt((1 - (f/f_n)²)² + (2*ζ*f/f_n)²)
        
        where f_n is body resonance frequency and ζ is damping ratio.
        """
        if not frequencies or zone.body_resonance is None:
            return 0.0
        
        br = zone.body_resonance
        f_n = br.frequency_primary
        zeta = br.damping_ratio
        
        total_coupling = 0.0
        total_weight = 0.0
        
        for f, amp in zip(frequencies, amplitudes):
            # Transfer function magnitude
            r = f / f_n
            H = 1.0 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
            
            # Weight by amplitude
            coupling = H * amp
            total_coupling += coupling
            total_weight += amp
        
        if total_weight > 0:
            return min(1.0, total_coupling / total_weight / br.quality_factor)
        return 0.0
    
    def analyze_all_zones(
        self,
        mode_shapes: np.ndarray,
        mode_frequencies: np.ndarray,
        plate_length: float,
        plate_width: float
    ) -> List[ZoneAnalysisResult]:
        """Analyze all configured zones."""
        results = []
        for zone in self.zones:
            result = self.analyze_mode_at_zone(
                zone, mode_shapes, mode_frequencies, plate_length, plate_width
            )
            results.append(result)
        return results
    
    def total_score(self, results: List[ZoneAnalysisResult]) -> float:
        """
        Calculate total optimization score from zone results.
        
        Lower is better (minimize total weighted error).
        """
        total_error = 0.0
        total_weight = 0.0
        
        for result in results:
            zone = result.zone
            weight = zone.optimization_weight
            
            # Frequency matching error
            freq_error = result.frequency_error
            
            # Coupling bonus (subtract because we want high coupling)
            coupling_bonus = result.coupling_efficiency * zone.optimization_weight
            
            total_error += weight * freq_error - coupling_bonus
            total_weight += weight
        
        return total_error / total_weight if total_weight > 0 else float('inf')


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def save_zone_config(zones: List[BodyZone], filepath: str):
    """Save zone configuration to JSON."""
    data = []
    for zone in zones:
        zone_dict = {
            "zone_type": zone.zone_type.value,
            "name": zone.name,
            "position_start": zone.position_start,
            "position_end": zone.position_end,
            "lateral_extent": zone.lateral_extent,
            "target_frequencies": zone.target_frequencies,
            "frequency_weights": zone.frequency_weights,
            "optimization_weight": zone.optimization_weight,
            "coupling_mode": zone.coupling_mode,
            "color": zone.color,
        }
        data.append(zone_dict)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_zone_config(filepath: str) -> List[BodyZone]:
    """Load zone configuration from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    zones = []
    for zone_dict in data:
        zone = BodyZone(
            zone_type=ZoneType(zone_dict["zone_type"]),
            name=zone_dict["name"],
            position_start=zone_dict["position_start"],
            position_end=zone_dict["position_end"],
            lateral_extent=zone_dict["lateral_extent"],
            target_frequencies=zone_dict.get("target_frequencies", []),
            frequency_weights=zone_dict.get("frequency_weights", []),
            optimization_weight=zone_dict.get("optimization_weight", 1.0),
            coupling_mode=zone_dict.get("coupling_mode", "maximize"),
            color=zone_dict.get("color", "#888888"),
        )
        zones.append(zone)
    
    return zones


# ══════════════════════════════════════════════════════════════════════════════
# QUICK API
# ══════════════════════════════════════════════════════════════════════════════

def create_custom_zones(
    n_zones: int,
    frequency_targets: List[List[float]],
    zone_names: Optional[List[str]] = None
) -> List[BodyZone]:
    """
    Quick creation of custom zones with even distribution.
    
    Args:
        n_zones: Number of zones to create
        frequency_targets: List of target frequencies per zone
        zone_names: Optional names for zones
    
    Returns:
        List of BodyZone objects
    """
    if zone_names is None:
        zone_names = [f"Zone_{i+1}" for i in range(n_zones)]
    
    zones = []
    segment_length = 1.0 / n_zones
    
    for i in range(n_zones):
        pos_start = i * segment_length
        pos_end = (i + 1) * segment_length
        
        targets = frequency_targets[i] if i < len(frequency_targets) else []
        
        zone = BodyZone(
            zone_type=ZoneType.FULL_BODY,  # Generic
            name=zone_names[i],
            position_start=pos_start,
            position_end=pos_end,
            lateral_extent=0.4,
            target_frequencies=targets,
            frequency_weights=[1.0] * len(targets),
            optimization_weight=1.0,
            color=f"#{hash(zone_names[i]) % 0xFFFFFF:06x}"
        )
        zones.append(zone)
    
    return zones


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Create chakra-based zones
    chakra_zones = create_chakra_zones()
    
    print("═" * 60)
    print(" CHAKRA ZONES CONFIGURATION")
    print("═" * 60)
    
    for zone in chakra_zones:
        print(f"\n{zone.name}:")
        print(f"  Position: {zone.position_start:.0%} - {zone.position_end:.0%}")
        print(f"  Targets: {zone.target_frequencies} Hz")
        print(f"  Weight: {zone.optimization_weight}")
        if zone.body_resonance:
            print(f"  Body resonance: {zone.body_resonance.frequency_primary} Hz")
    
    # Example: Create VAT therapy zones
    print("\n" + "═" * 60)
    print(" VAT THERAPY ZONES CONFIGURATION")
    print("═" * 60)
    
    vat_zones = create_vat_therapy_zones()
    for zone in vat_zones:
        print(f"\n{zone.name}:")
        print(f"  Targets: {zone.target_frequencies} Hz")
        print(f"  Mode: {zone.coupling_mode}")
    
    # Example: Save configuration
    save_zone_config(chakra_zones, "/tmp/chakra_zones.json")
    print("\n✓ Configuration saved to /tmp/chakra_zones.json")
