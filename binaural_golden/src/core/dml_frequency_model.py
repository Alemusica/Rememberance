"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  DML (Distributed Mode Loudspeaker) Frequency Response Model                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Based on research from:                                                      ║
║  - Harris (2010): DML fundamentals, panel as radiator                        ║
║  - Aures (2001): Golden ratio positioning 0.381L, 0.618L                     ║
║  - Bank (2010): Excitation point coupling theory                             ║
║  - Azizi (2015): GA optimization for flat panel speakers                     ║
║  - Wang (2020): Multiple exciter phase control                               ║
║                                                                               ║
║  KEY INSIGHT:                                                                 ║
║  Plate MODES are physical (20-200Hz for large plates), but the EXCITER       ║
║  can reproduce full 20-20kHz audio. The modes COLOR the sound - they         ║
║  determine HOW the plate vibrates at each frequency, not WHAT frequencies    ║
║  can be produced.                                                            ║
║                                                                               ║
║  For flat response: optimize EXCITER POSITION on mode shapes                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
GOLDEN_POSITIONS = [1/PHI, 1 - 1/PHI]  # 0.618, 0.382


class ExciterStrategy(Enum):
    """Exciter placement strategies from literature."""
    GOLDEN_RATIO = "golden_ratio"      # Aures (2001): 0.381L, 0.618L
    ANTINODE = "antinode"              # Bank (2010): Maximum coupling
    PHASE_OPTIMIZED = "phase_optimized"  # Wang (2020): Multi-exciter phase
    GENETIC = "genetic"                # Azizi (2015): GA optimization


@dataclass
class ModeShape:
    """
    Modal shape data for a plate mode.
    
    A mode shape describes HOW the plate deforms at its natural frequency.
    The shape determines coupling with exciters at different positions.
    """
    mode_number: Tuple[int, int]  # (m, n) - number of half-waves in x, y
    frequency_hz: float
    shape_matrix: np.ndarray  # 2D array of displacement at each point
    nodal_lines: List[float]  # Positions of zero displacement (normalized)
    antinode_positions: List[Tuple[float, float]]  # Maximum displacement positions
    
    @property
    def name(self) -> str:
        return f"Mode ({self.mode_number[0]},{self.mode_number[1]})"


@dataclass
class ExciterCoupling:
    """
    Coupling coefficient between exciter and mode.
    
    From Bank (2010): Coupling is proportional to mode shape amplitude
    at exciter position. Antinode = max coupling, node = zero coupling.
    """
    exciter_position: Tuple[float, float]  # (x, y) normalized [0, 1]
    mode: ModeShape
    coupling_coefficient: float  # 0 = no coupling, 1 = max coupling
    
    @classmethod
    def calculate(cls, exciter_pos: Tuple[float, float], mode: ModeShape) -> 'ExciterCoupling':
        """Calculate coupling coefficient for exciter-mode pair."""
        x, y = exciter_pos
        
        # Sample mode shape at exciter position
        nx, ny = mode.shape_matrix.shape
        ix = int(x * (nx - 1))
        iy = int(y * (ny - 1))
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        
        # Coupling is absolute value (both peaks and troughs couple)
        amplitude = abs(mode.shape_matrix[ix, iy])
        max_amplitude = np.max(np.abs(mode.shape_matrix))
        
        coupling = amplitude / max_amplitude if max_amplitude > 0 else 0.0
        
        return cls(
            exciter_position=exciter_pos,
            mode=mode,
            coupling_coefficient=coupling
        )


@dataclass
class DMLResponse:
    """
    Complete DML frequency response model.
    
    This models how a plate+exciter system responds across the full
    audio frequency range (20Hz - 20kHz).
    """
    plate_length: float  # meters
    plate_width: float
    plate_thickness: float
    
    modes: List[ModeShape] = field(default_factory=list)
    exciter_positions: List[Tuple[float, float]] = field(default_factory=list)
    
    # Frequency response (dB vs Hz)
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    response_db: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Quality metrics
    flatness_db: float = 0.0  # ±dB variation in target range
    energy_coverage: float = 0.0  # % of body zones with adequate energy
    low_freq_extension: float = 0.0  # -3dB point in Hz


class DMLFrequencyModel:
    """
    Model DML speaker frequency response based on plate modes and exciter position.
    
    KEY PHYSICS (from Harris 2010, Bank 2010):
    
    1. EXCITER drives the plate with audio signal (20Hz - 20kHz capable)
    2. PLATE has natural modes (resonances) at specific frequencies
    3. At mode frequency: plate vibrates strongly IF exciter is at antinode
    4. Between modes: plate still vibrates but with different pattern
    5. SOUND RADIATION depends on mode shape and frequency
    
    For BODY COUPLING (vibroacoustic therapy):
    - Low modes (20-200Hz) couple best to body (large wavelength)
    - Higher frequencies still couple but with less penetration
    - Exciter position affects WHICH modes are excited
    """
    
    def __init__(
        self,
        plate_length: float,
        plate_width: float,
        plate_thickness: float,
        material_density: float = 680.0,  # kg/m³ (birch plywood)
        youngs_modulus: float = 12e9,  # Pa
        poissons_ratio: float = 0.3
    ):
        self.L = plate_length
        self.W = plate_width
        self.t = plate_thickness
        self.rho = material_density
        self.E = youngs_modulus
        self.nu = poissons_ratio
        
        # Calculate bending stiffness
        self.D = (self.E * self.t**3) / (12 * (1 - self.nu**2))
        
        # Pre-calculate modes
        self.modes: List[ModeShape] = []
        self._calculate_modes()
    
    def _calculate_modes(self, max_mn: int = 10, max_freq: float = 500.0):
        """
        Calculate plate modal frequencies and shapes.
        
        Using Kirchhoff plate theory for simply supported rectangular plate:
        f_mn = (π/2) * sqrt(D/(ρt)) * ((m/L)² + (n/W)²)
        
        Mode shape: sin(mπx/L) * sin(nπy/W)
        """
        self.modes = []
        
        # Frequency coefficient
        freq_coeff = (np.pi / 2) * np.sqrt(self.D / (self.rho * self.t))
        
        for m in range(1, max_mn + 1):
            for n in range(1, max_mn + 1):
                # Modal frequency
                f_mn = freq_coeff * ((m / self.L)**2 + (n / self.W)**2)
                
                if f_mn > max_freq:
                    continue
                
                # Mode shape (sampled)
                nx, ny = 50, 50
                x = np.linspace(0, 1, nx)
                y = np.linspace(0, 1, ny)
                X, Y = np.meshgrid(x, y)
                
                shape = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
                
                # Nodal lines (where shape = 0)
                nodal_x = [i / m for i in range(1, m)]
                nodal_y = [j / n for j in range(1, n)]
                
                # Antinode positions (peaks)
                antinodes = []
                for i in range(m):
                    for j in range(n):
                        ax = (2*i + 1) / (2*m)
                        ay = (2*j + 1) / (2*n)
                        antinodes.append((ax, ay))
                
                self.modes.append(ModeShape(
                    mode_number=(m, n),
                    frequency_hz=f_mn,
                    shape_matrix=shape.T,
                    nodal_lines=nodal_x + nodal_y,
                    antinode_positions=antinodes
                ))
        
        # Sort by frequency
        self.modes.sort(key=lambda m: m.frequency_hz)
    
    def calculate_frequency_response(
        self,
        exciter_positions: List[Tuple[float, float]],
        freq_range: Tuple[float, float] = (20.0, 20000.0),
        n_points: int = 200
    ) -> DMLResponse:
        """
        Calculate frequency response for given exciter positions.
        
        The response combines:
        1. Modal peaks (at mode frequencies, scaled by coupling)
        2. Off-resonance behavior (interpolated)
        3. Multiple exciter summation (with phase)
        
        From Wang (2020): Multiple exciters can flatten response
        if positioned to complement each other's coupling.
        """
        frequencies = np.logspace(
            np.log10(freq_range[0]),
            np.log10(freq_range[1]),
            n_points
        )
        
        response = np.zeros(n_points)
        
        # Calculate coupling for each exciter-mode pair
        couplings: Dict[Tuple[float, float], List[ExciterCoupling]] = {}
        for pos in exciter_positions:
            couplings[pos] = [
                ExciterCoupling.calculate(pos, mode)
                for mode in self.modes
            ]
        
        # Modal Q factor (damping)
        Q = 20.0  # Typical for wood panel
        
        for i, f in enumerate(frequencies):
            total_response = 0.0
            
            for pos in exciter_positions:
                exciter_response = 0.0
                
                for coupling in couplings[pos]:
                    mode = coupling.mode
                    f_m = mode.frequency_hz
                    c = coupling.coupling_coefficient
                    
                    # Modal response (2nd order resonant system)
                    # H(f) = c / sqrt((1 - (f/f_m)²)² + (f/(Q*f_m))²)
                    ratio = f / f_m
                    denominator = np.sqrt((1 - ratio**2)**2 + (ratio / Q)**2)
                    
                    modal_contribution = c / max(denominator, 0.001)
                    exciter_response += modal_contribution
                
                # Off-resonance baseline (above modes, plate radiates)
                # High-frequency roll-off
                hf_rolloff = 1.0 / (1 + (f / 5000)**2)
                
                total_response += exciter_response * hf_rolloff
            
            # Average across exciters
            response[i] = total_response / len(exciter_positions)
        
        # Convert to dB (normalize to 0dB at 1kHz)
        ref_idx = np.argmin(np.abs(frequencies - 1000))
        response_db = 20 * np.log10(response / max(response[ref_idx], 1e-10))
        
        # Calculate metrics
        target_range = (frequencies > 50) & (frequencies < 15000)
        flatness = np.std(response_db[target_range])
        
        return DMLResponse(
            plate_length=self.L,
            plate_width=self.W,
            plate_thickness=self.t,
            modes=self.modes,
            exciter_positions=exciter_positions,
            frequencies=frequencies,
            response_db=response_db,
            flatness_db=float(flatness),
            energy_coverage=self._calculate_energy_coverage(response_db, frequencies),
            low_freq_extension=self._calculate_low_freq_extension(response_db, frequencies)
        )
    
    def _calculate_energy_coverage(
        self,
        response_db: np.ndarray,
        frequencies: np.ndarray
    ) -> float:
        """Calculate percentage of body-coupled frequency range with adequate energy."""
        # Body coupling range: 20-200Hz
        body_range = (frequencies >= 20) & (frequencies <= 200)
        body_response = response_db[body_range]
        
        # "Adequate" = within 6dB of mean
        mean_level = np.mean(body_response)
        adequate = np.abs(body_response - mean_level) < 6.0
        
        return np.sum(adequate) / len(body_response) * 100.0
    
    def _calculate_low_freq_extension(
        self,
        response_db: np.ndarray,
        frequencies: np.ndarray
    ) -> float:
        """Calculate -3dB low frequency extension point."""
        # Find where response drops 3dB from peak
        peak_level = np.max(response_db)
        below_3db = response_db < (peak_level - 3.0)
        
        # Find lowest frequency above -3dB point
        for i, freq in enumerate(frequencies):
            if not below_3db[i]:
                return freq
        
        return frequencies[0]
    
    def optimize_exciter_positions(
        self,
        n_exciters: int = 4,
        strategy: ExciterStrategy = ExciterStrategy.GOLDEN_RATIO,
        target_flatness_db: float = 3.0
    ) -> List[Tuple[float, float]]:
        """
        Optimize exciter positions for flat frequency response.
        
        Strategies from literature:
        
        1. GOLDEN_RATIO (Aures 2001):
           - Position at 0.381L, 0.618L from edges
           - Avoids common nodal positions
           - Simple, works well for 2 exciters
        
        2. ANTINODE (Bank 2010):
           - Position at antinodes of first N modes
           - Maximizes coupling to dominant modes
           - May not give flat response
        
        3. PHASE_OPTIMIZED (Wang 2020):
           - Optimize positions AND phases
           - Cancellation for flatness
           - Requires DSP control
        
        4. GENETIC (Azizi 2015):
           - GA optimization for broadband
           - Off-center, irregular positions
           - Best flatness but complex
        """
        positions = []
        
        if strategy == ExciterStrategy.GOLDEN_RATIO:
            # Aures (2001): Golden ratio positions avoid common nodes
            # For 4 exciters: 2x2 grid at φ positions
            x_positions = [1/PHI, 1 - 1/PHI]  # 0.382, 0.618
            y_positions = [1/PHI, 1 - 1/PHI]
            
            for x in x_positions[:min(2, n_exciters)]:
                for y in y_positions[:min(2, n_exciters // 2 + 1)]:
                    positions.append((x, y))
                    if len(positions) >= n_exciters:
                        break
                if len(positions) >= n_exciters:
                    break
        
        elif strategy == ExciterStrategy.ANTINODE:
            # Bank (2010): Position at antinodes of first modes
            for mode in self.modes[:n_exciters]:
                if mode.antinode_positions:
                    positions.append(mode.antinode_positions[0])
            
            # Fill remaining with golden ratio
            while len(positions) < n_exciters:
                x = 0.5 + 0.1 * len(positions)
                y = 1 / PHI
                positions.append((x % 1.0, y))
        
        elif strategy == ExciterStrategy.GENETIC:
            # Azizi (2015): GA optimization (simplified here)
            # Start with golden ratio, then optimize
            positions = self.optimize_exciter_positions(n_exciters, ExciterStrategy.GOLDEN_RATIO)
            
            # Simple hill-climbing optimization
            best_positions = positions.copy()
            best_flatness = float('inf')
            
            for _ in range(100):  # Iterations
                # Perturb positions
                trial = [(p[0] + np.random.normal(0, 0.05), 
                          p[1] + np.random.normal(0, 0.05)) for p in best_positions]
                trial = [(np.clip(x, 0.1, 0.9), np.clip(y, 0.1, 0.9)) for x, y in trial]
                
                response = self.calculate_frequency_response(trial)
                if response.flatness_db < best_flatness:
                    best_flatness = response.flatness_db
                    best_positions = trial
                    
                    if best_flatness < target_flatness_db:
                        break
            
            positions = best_positions
        
        elif strategy == ExciterStrategy.PHASE_OPTIMIZED:
            # Wang (2020): Positions for phase cancellation
            # Use opposing positions
            positions = [
                (0.25, 0.25),
                (0.75, 0.75),
                (0.25, 0.75),
                (0.75, 0.25)
            ][:n_exciters]
        
        return positions[:n_exciters]
    
    def get_body_coupling_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendations for body-coupled vibroacoustic response.
        
        For VAT (Vibroacoustic Therapy), we want:
        1. Strong low-frequency modes (20-80Hz) for deep tissue
        2. Mid-frequency coverage (80-200Hz) for muscle/bone
        3. Smooth response (no harsh peaks)
        
        Returns recommendations for:
        - Optimal exciter positions
        - Suggested EQ curve
        - Body zone targeting
        """
        recommendations = {
            "exciter_strategy": ExciterStrategy.GOLDEN_RATIO.value,
            "exciter_positions": [],
            "eq_bands": [],
            "zone_notes": [],
            "mode_notes": []
        }
        
        # Optimal positions for body coupling
        positions = self.optimize_exciter_positions(4, ExciterStrategy.GOLDEN_RATIO)
        recommendations["exciter_positions"] = positions
        
        # EQ recommendations based on mode spacing
        if self.modes:
            first_mode = self.modes[0].frequency_hz
            recommendations["eq_bands"].append({
                "freq": first_mode,
                "gain_db": -3.0,
                "note": f"Reduce peak at first mode ({first_mode:.1f}Hz)"
            })
            
            # Fill dips between modes
            for i in range(len(self.modes) - 1):
                f1 = self.modes[i].frequency_hz
                f2 = self.modes[i + 1].frequency_hz
                gap = f2 - f1
                if gap > 20:  # Significant gap
                    mid = (f1 + f2) / 2
                    recommendations["eq_bands"].append({
                        "freq": mid,
                        "gain_db": 3.0,
                        "note": f"Boost dip between modes at {mid:.1f}Hz"
                    })
        
        # Mode notes
        for mode in self.modes[:5]:
            m, n = mode.mode_number
            if m == 1 and n == 1:
                zone = "full body (deep tissue)"
            elif m == 1 or n == 1:
                zone = "longitudinal (spine)"
            else:
                zone = "local (joints/tissue)"
            
            recommendations["mode_notes"].append({
                "mode": f"({m},{n})",
                "frequency": mode.frequency_hz,
                "body_zone": zone
            })
        
        # Zone targeting notes
        recommendations["zone_notes"] = [
            {"zone": "Spine", "freq_range": "30-60Hz", "exciter": "Center-line positions"},
            {"zone": "Head/Neck", "freq_range": "60-100Hz", "exciter": "Upper φ position"},
            {"zone": "Pelvis/Lower", "freq_range": "20-40Hz", "exciter": "Lower φ position"},
            {"zone": "Full body", "freq_range": "40-80Hz", "exciter": "All exciters in phase"}
        ]
        
        return recommendations


def create_dml_model_for_genome(genome: Any) -> DMLFrequencyModel:
    """
    Create DML model from a PlateGenome.
    
    Args:
        genome: PlateGenome with plate definition
    
    Returns:
        DMLFrequencyModel configured for the genome
    """
    return DMLFrequencyModel(
        plate_length=genome.length,
        plate_width=genome.width,
        plate_thickness=genome.thickness_base,
        material_density=680.0,  # Birch plywood
        youngs_modulus=12e9
    )


def analyze_exciter_placement(genome: Any) -> Dict[str, Any]:
    """
    Analyze current exciter placement and suggest improvements.
    
    Returns analysis with:
    - Current coupling coefficients
    - Response flatness
    - Suggested position changes
    - Body coupling assessment
    """
    model = create_dml_model_for_genome(genome)
    
    # Get current exciter positions from genome
    current_positions = [(e.x, e.y) for e in genome.exciters]
    
    if not current_positions:
        current_positions = [(0.5, 0.5)]  # Default center
    
    # Calculate current response
    current_response = model.calculate_frequency_response(current_positions)
    
    # Calculate optimal positions
    optimal_positions = model.optimize_exciter_positions(
        len(current_positions),
        ExciterStrategy.GOLDEN_RATIO
    )
    optimal_response = model.calculate_frequency_response(optimal_positions)
    
    # Coupling analysis
    coupling_analysis = []
    for pos in current_positions:
        pos_couplings = []
        for mode in model.modes[:5]:
            c = ExciterCoupling.calculate(pos, mode)
            pos_couplings.append({
                "mode": mode.name,
                "coupling": c.coupling_coefficient
            })
        coupling_analysis.append({
            "position": pos,
            "couplings": pos_couplings
        })
    
    return {
        "current": {
            "positions": current_positions,
            "flatness_db": current_response.flatness_db,
            "energy_coverage": current_response.energy_coverage,
            "low_freq_extension": current_response.low_freq_extension
        },
        "optimal": {
            "positions": optimal_positions,
            "flatness_db": optimal_response.flatness_db,
            "energy_coverage": optimal_response.energy_coverage,
            "low_freq_extension": optimal_response.low_freq_extension
        },
        "improvement": {
            "flatness_improvement_db": current_response.flatness_db - optimal_response.flatness_db,
            "energy_improvement_pct": optimal_response.energy_coverage - current_response.energy_coverage
        },
        "coupling_analysis": coupling_analysis,
        "recommendations": model.get_body_coupling_recommendation()
    }
