"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              COUPLED SYSTEM - Plate + Body Vibroacoustic Model               ║
║                                                                              ║
║   2-DOF coupled system: Tavola (plate) + Corpo (human body)                  ║
║                                                                              ║
║   Physical Model:                                                             ║
║   ┌─────────┐                                                                ║
║   │  Body   │─────────── k_body, c_body (body stiffness, damping)           ║
║   │  (m_b)  │                                                                ║
║   └────┬────┘                                                                ║
║        │ k_contact, c_contact (body-plate interface)                         ║
║   ┌────┴────┐                                                                ║
║   │  Plate  │─────────── k_plate, c_plate (plate stiffness, damping)        ║
║   │  (m_p)  │                                                                ║
║   └────┬────┘                                                                ║
║        │ (exciter input)                                                     ║
║       ~~~                                                                    ║
║                                                                              ║
║   References:                                                                 ║
║   • Griffin (1990) - Handbook of Human Vibration                             ║
║   • Fairley & Griffin (1989) - Body vibration transmissibility               ║
║   • Skille (1989) - Vibroacoustic therapy                                    ║
║   • Schleske - Violin body acoustics                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
from enum import Enum
import scipy.linalg as la
from scipy.signal import freqresp

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlatePhysics:
    """
    Physical properties of the vibroacoustic plate.
    
    Based on lutherie principles (Schleske) and acoustic wood properties.
    """
    # Geometry
    length: float = 2.0  # [m] - body length supine
    width: float = 0.6   # [m] - shoulder width
    thickness: float = 0.02  # [m] - typical soundboard
    
    # Material (default: spruce, typical soundboard)
    density: float = 400.0  # [kg/m³]
    E_longitudinal: float = 12e9  # [Pa] Young's modulus along grain
    E_transverse: float = 0.8e9   # [Pa] Young's modulus across grain
    G_shear: float = 0.7e9        # [Pa] Shear modulus
    nu_poisson: float = 0.35      # Poisson's ratio
    
    # Damping
    loss_factor: float = 0.01  # tan(δ) for wood
    
    @property
    def area(self) -> float:
        """Plate surface area [m²]."""
        return self.length * self.width
    
    @property
    def mass(self) -> float:
        """Total plate mass [kg]."""
        return self.density * self.length * self.width * self.thickness
    
    @property
    def flexural_rigidity_long(self) -> float:
        """Flexural rigidity along grain D_L [N·m]."""
        h = self.thickness
        return self.E_longitudinal * h**3 / (12 * (1 - self.nu_poisson**2))
    
    @property
    def flexural_rigidity_trans(self) -> float:
        """Flexural rigidity across grain D_T [N·m]."""
        h = self.thickness
        return self.E_transverse * h**3 / (12 * (1 - self.nu_poisson**2))
    
    def fundamental_frequency(self) -> float:
        """
        Estimate fundamental frequency using Rayleigh-Ritz.
        
        f_11 ≈ (π/2) * sqrt(D / (ρ*h)) * (1/L² + 1/W²)
        """
        D_avg = np.sqrt(self.flexural_rigidity_long * self.flexural_rigidity_trans)
        rho_h = self.density * self.thickness
        
        f_11 = (np.pi / 2) * np.sqrt(D_avg / rho_h) * (
            1/self.length**2 + 1/self.width**2
        )
        return f_11


@dataclass
class HumanBody:
    """
    Simplified human body mechanical model.
    
    Based on Griffin (1990) and Fairley & Griffin (1989).
    
    The supine human body can be modeled as multiple masses
    connected by springs and dampers representing:
    - Skeletal stiffness
    - Muscle tone
    - Tissue damping
    """
    # Anthropometry
    height: float = 1.75  # [m]
    mass: float = 70.0    # [kg]
    
    # Mass distribution (fraction of total)
    head_mass_fraction: float = 0.08
    torso_mass_fraction: float = 0.50
    arms_mass_fraction: float = 0.10
    legs_mass_fraction: float = 0.32
    
    # Stiffness parameters [N/m]
    spine_stiffness: float = 50000.0  # Lumbar spine
    tissue_stiffness: float = 10000.0  # Soft tissue
    
    # Damping ratios
    spine_damping_ratio: float = 0.15
    tissue_damping_ratio: float = 0.30
    
    @property
    def effective_mass_supine(self) -> float:
        """
        Effective mass for vertical vibration when supine.
        
        Not all mass moves in phase - use ~60% based on Griffin.
        """
        return 0.60 * self.mass
    
    @property
    def effective_stiffness(self) -> float:
        """Combined body stiffness [N/m]."""
        # Series combination of spine and tissue
        return 1.0 / (1.0/self.spine_stiffness + 1.0/self.tissue_stiffness)
    
    @property
    def effective_damping(self) -> float:
        """Effective damping coefficient [N·s/m]."""
        k = self.effective_stiffness
        m = self.effective_mass_supine
        omega_n = np.sqrt(k / m)
        zeta = (self.spine_damping_ratio + self.tissue_damping_ratio) / 2
        return 2 * zeta * np.sqrt(k * m)
    
    @property
    def body_natural_frequency(self) -> float:
        """Natural frequency of whole body [Hz]."""
        k = self.effective_stiffness
        m = self.effective_mass_supine
        return np.sqrt(k / m) / (2 * np.pi)


# ══════════════════════════════════════════════════════════════════════════════
# 2-DOF COUPLED SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContactInterface:
    """
    Interface between plate and human body.
    
    Models the contact stiffness and damping when body rests on plate.
    """
    # Contact stiffness [N/m per m²]
    stiffness_per_area: float = 5000.0
    
    # Contact damping ratio
    damping_ratio: float = 0.20
    
    # Contact area fraction (how much of body touches plate)
    contact_area_fraction: float = 0.30
    
    def get_contact_stiffness(self, contact_area: float) -> float:
        """Total contact stiffness [N/m]."""
        return self.stiffness_per_area * contact_area
    
    def get_contact_damping(self, contact_area: float, mass: float) -> float:
        """Total contact damping [N·s/m]."""
        k = self.get_contact_stiffness(contact_area)
        omega_n = np.sqrt(k / mass)
        return 2 * self.damping_ratio * mass * omega_n


class CoupledSystem:
    """
    2-DOF coupled plate + body vibroacoustic system.
    
    State vector: [x_plate, x_body, v_plate, v_body]
    
    Equation of motion:
    M * ẍ + C * ẋ + K * x = F(t)
    
    where:
    M = [m_p,  0  ]    K = [k_p + k_c,  -k_c    ]
        [0,   m_b ]        [-k_c,        k_b + k_c]
    
    C = [c_p + c_c,  -c_c    ]
        [-c_c,        c_b + c_c]
    """
    
    def __init__(
        self,
        plate: PlatePhysics,
        body: HumanBody,
        contact: ContactInterface
    ):
        self.plate = plate
        self.body = body
        self.contact = contact
        
        # Build system matrices
        self._build_matrices()
    
    def _build_matrices(self):
        """Build mass, stiffness, and damping matrices."""
        # Masses
        m_p = self.plate.mass
        m_b = self.body.effective_mass_supine
        
        # Contact area
        contact_area = self.contact.contact_area_fraction * self.plate.area
        
        # Stiffnesses
        k_p = self._plate_effective_stiffness()
        k_b = self.body.effective_stiffness
        k_c = self.contact.get_contact_stiffness(contact_area)
        
        # Dampings
        c_p = self._plate_damping_coefficient()
        c_b = self.body.effective_damping
        c_c = self.contact.get_contact_damping(contact_area, m_b)
        
        # Mass matrix
        self.M = np.array([
            [m_p, 0],
            [0, m_b]
        ])
        
        # Stiffness matrix
        self.K = np.array([
            [k_p + k_c, -k_c],
            [-k_c, k_b + k_c]
        ])
        
        # Damping matrix
        self.C = np.array([
            [c_p + c_c, -c_c],
            [-c_c, c_b + c_c]
        ])
        
        # Store individual values
        self.m_p = m_p
        self.m_b = m_b
        self.k_p = k_p
        self.k_b = k_b
        self.k_c = k_c
        self.c_p = c_p
        self.c_b = c_b
        self.c_c = c_c
    
    def _plate_effective_stiffness(self) -> float:
        """
        Effective plate stiffness for first mode.
        
        k_eff = ω₁² * m_plate
        """
        omega1 = 2 * np.pi * self.plate.fundamental_frequency()
        return omega1**2 * self.plate.mass
    
    def _plate_damping_coefficient(self) -> float:
        """Plate damping coefficient from loss factor."""
        k = self._plate_effective_stiffness()
        m = self.plate.mass
        zeta = self.plate.loss_factor / 2  # η ≈ 2ζ for small damping
        return 2 * zeta * np.sqrt(k * m)
    
    def natural_frequencies(self) -> Tuple[float, float]:
        """
        Solve eigenvalue problem for natural frequencies.
        
        det(K - ω²M) = 0
        
        Returns:
            (f1, f2): Lower and upper natural frequencies [Hz]
        """
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = la.eig(self.K, self.M)
        
        # Convert to frequencies (take real part, should be positive)
        omega_sq = np.real(eigenvalues)
        omega_sq = np.sort(omega_sq[omega_sq > 0])
        
        frequencies = np.sqrt(omega_sq) / (2 * np.pi)
        
        if len(frequencies) >= 2:
            return (frequencies[0], frequencies[1])
        elif len(frequencies) == 1:
            return (frequencies[0], frequencies[0])
        else:
            return (0.0, 0.0)
    
    def mode_shapes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mode shapes (eigenvectors).
        
        Returns:
            (mode1, mode2): Mode shapes as 2-element arrays [plate, body]
        """
        eigenvalues, eigenvectors = la.eig(self.K, self.M)
        
        # Sort by frequency
        idx = np.argsort(np.real(eigenvalues))
        eigenvectors = eigenvectors[:, idx]
        
        # Normalize
        mode1 = np.real(eigenvectors[:, 0])
        mode2 = np.real(eigenvectors[:, 1])
        
        mode1 = mode1 / np.max(np.abs(mode1))
        mode2 = mode2 / np.max(np.abs(mode2))
        
        return (mode1, mode2)
    
    def transfer_function(
        self,
        frequencies: np.ndarray,
        input_dof: int = 0,  # 0 = plate, 1 = body
        output_dof: int = 1  # 0 = plate, 1 = body
    ) -> np.ndarray:
        """
        Calculate frequency response function H(f).
        
        H(ω) = (-ω²M + jωC + K)⁻¹
        
        Args:
            frequencies: Array of frequencies [Hz]
            input_dof: Input degree of freedom (0=plate, 1=body)
            output_dof: Output degree of freedom
        
        Returns:
            Complex transfer function values
        """
        omega = 2 * np.pi * frequencies
        H = np.zeros(len(frequencies), dtype=complex)
        
        for i, w in enumerate(omega):
            # Dynamic stiffness matrix
            D = -w**2 * self.M + 1j * w * self.C + self.K
            
            # Receptance matrix (inverse)
            try:
                D_inv = la.inv(D)
                H[i] = D_inv[output_dof, input_dof]
            except la.LinAlgError:
                H[i] = np.inf
        
        return H
    
    def transmissibility(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate body/plate displacement transmissibility.
        
        T(f) = |x_body / x_plate|
        
        This is key for VAT effectiveness - we want T ≈ 1 at therapy frequencies.
        """
        H_plate = self.transfer_function(frequencies, input_dof=0, output_dof=0)
        H_body = self.transfer_function(frequencies, input_dof=0, output_dof=1)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            T = np.abs(H_body / H_plate)
            T = np.nan_to_num(T, nan=0.0, posinf=1e6, neginf=0.0)
        
        return T
    
    def coupling_efficiency(self, target_frequency: float) -> float:
        """
        Calculate coupling efficiency at a specific frequency.
        
        Range 0-1, where 1 = perfect transfer of vibration to body.
        """
        freqs = np.array([target_frequency])
        T = self.transmissibility(freqs)[0]
        
        # Normalize: T=1 is optimal, both high and low are suboptimal
        efficiency = np.exp(-np.abs(np.log(T + 1e-10)))
        
        return min(1.0, efficiency)
    
    def optimize_plate_for_coupling(
        self,
        target_frequencies: List[float],
        frequency_weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Find optimal plate parameters for coupling at target frequencies.
        
        Returns recommended adjustments to plate properties.
        """
        if frequency_weights is None:
            frequency_weights = [1.0] * len(target_frequencies)
        
        # Current coupling at targets
        current_coupling = []
        for f in target_frequencies:
            current_coupling.append(self.coupling_efficiency(f))
        
        avg_coupling = np.average(current_coupling, weights=frequency_weights)
        
        # Natural frequencies
        f1, f2 = self.natural_frequencies()
        
        # Recommendations
        recommendations = {
            "current_coupling": avg_coupling,
            "natural_freq_1": f1,
            "natural_freq_2": f2,
        }
        
        # Check if targets are near resonances
        for i, f_target in enumerate(target_frequencies):
            if abs(f_target - f1) / f1 < 0.2:
                recommendations[f"target_{i}_near_mode_1"] = True
            if abs(f_target - f2) / f2 < 0.2:
                recommendations[f"target_{i}_near_mode_2"] = True
        
        # Suggest thickness adjustment
        # f ∝ h (thickness) for plates, so h_new/h_old = f_target/f_current
        if len(target_frequencies) > 0:
            f_target_main = target_frequencies[0]
            f_current = self.plate.fundamental_frequency()
            thickness_ratio = f_target_main / f_current
            recommendations["suggested_thickness_ratio"] = thickness_ratio
            recommendations["suggested_thickness_m"] = self.plate.thickness * thickness_ratio
        
        return recommendations


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-ZONE COUPLED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class ZoneCoupledSystem:
    """
    Coupled system with zone-specific body parameters.
    
    Different body zones have different mass, stiffness, and resonance.
    This creates a spatially-varying coupled model.
    """
    
    def __init__(
        self,
        plate: PlatePhysics,
        zones: List['BodyZone']  # from body_zones.py
    ):
        self.plate = plate
        self.zones = zones
        self.zone_systems: Dict[str, CoupledSystem] = {}
        
        self._build_zone_systems()
    
    def _build_zone_systems(self):
        """Create a coupled system for each zone."""
        for zone in self.zones:
            # Create body model for this zone
            zone_body = self._zone_to_body(zone)
            
            # Adjust contact based on zone area
            zone_contact = ContactInterface(
                contact_area_fraction=zone.get_area_fraction()
            )
            
            # Create coupled system
            system = CoupledSystem(self.plate, zone_body, zone_contact)
            self.zone_systems[zone.name] = system
    
    def _zone_to_body(self, zone: 'BodyZone') -> HumanBody:
        """Convert zone to equivalent body parameters."""
        # Default body
        body = HumanBody()
        
        # Adjust mass based on zone
        if zone.body_resonance:
            body.mass = body.mass * zone.body_resonance.effective_mass_fraction
        
        # Adjust stiffness to match zone resonance
        if zone.body_resonance:
            f_n = zone.body_resonance.frequency_primary
            m = body.effective_mass_supine
            # k = (2π f)² m
            body.spine_stiffness = (2 * np.pi * f_n)**2 * m
            body.tissue_stiffness = body.spine_stiffness * 0.5
            
            # Set damping ratio
            body.spine_damping_ratio = zone.body_resonance.damping_ratio
            body.tissue_damping_ratio = zone.body_resonance.damping_ratio * 1.5
        
        return body
    
    def zone_coupling_matrix(
        self,
        frequencies: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate coupling efficiency for each zone at all frequencies.
        
        Returns:
            Dict mapping zone name to array of coupling values
        """
        result = {}
        for zone in self.zones:
            system = self.zone_systems[zone.name]
            T = system.transmissibility(frequencies)
            result[zone.name] = T
        
        return result
    
    def total_coupling_score(
        self,
        target_frequencies: Optional[Dict[str, List[float]]] = None
    ) -> float:
        """
        Calculate total weighted coupling score across all zones.
        
        Args:
            target_frequencies: Optional dict mapping zone name to target freqs.
                              If None, uses zone's own targets.
        
        Returns:
            Score in [0, 1], higher is better
        """
        total_score = 0.0
        total_weight = 0.0
        
        for zone in self.zones:
            system = self.zone_systems[zone.name]
            
            # Get targets for this zone
            if target_frequencies and zone.name in target_frequencies:
                targets = target_frequencies[zone.name]
            else:
                targets = zone.target_frequencies
            
            # Calculate coupling at each target
            zone_score = 0.0
            for i, f in enumerate(targets):
                weight = (zone.frequency_weights[i] 
                         if i < len(zone.frequency_weights) else 1.0)
                coupling = system.coupling_efficiency(f)
                zone_score += weight * coupling
            
            # Weight by zone importance
            total_score += zone.optimization_weight * zone_score
            total_weight += zone.optimization_weight * len(targets)
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def sensitivity_analysis(
        self,
        parameter: str,
        delta: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate sensitivity of coupling score to plate parameter.
        
        ∂Score/∂parameter ≈ (Score(p+δ) - Score(p-δ)) / (2δ)
        
        Args:
            parameter: 'thickness', 'E_long', 'E_trans', 'density'
            delta: Relative perturbation
        
        Returns:
            Sensitivity dict with gradient and direction
        """
        original_value = getattr(self.plate, parameter)
        
        # Perturb +
        setattr(self.plate, parameter, original_value * (1 + delta))
        self._build_zone_systems()
        score_plus = self.total_coupling_score()
        
        # Perturb -
        setattr(self.plate, parameter, original_value * (1 - delta))
        self._build_zone_systems()
        score_minus = self.total_coupling_score()
        
        # Restore
        setattr(self.plate, parameter, original_value)
        self._build_zone_systems()
        
        # Gradient
        gradient = (score_plus - score_minus) / (2 * delta * original_value)
        
        return {
            "parameter": parameter,
            "gradient": gradient,
            "direction": "increase" if gradient > 0 else "decrease",
            "magnitude": abs(gradient),
            "score_at_current": self.total_coupling_score()
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK API
# ══════════════════════════════════════════════════════════════════════════════

def create_default_coupled_system() -> CoupledSystem:
    """Create a typical coupled system with default parameters."""
    plate = PlatePhysics()
    body = HumanBody()
    contact = ContactInterface()
    return CoupledSystem(plate, body, contact)


def analyze_coupling_at_frequency(frequency: float) -> Dict:
    """Quick analysis of coupling at a single frequency."""
    system = create_default_coupled_system()
    
    f1, f2 = system.natural_frequencies()
    coupling = system.coupling_efficiency(frequency)
    
    return {
        "target_frequency": frequency,
        "system_resonances": [f1, f2],
        "coupling_efficiency": coupling,
        "transmissibility": system.transmissibility(np.array([frequency]))[0]
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("═" * 60)
    print(" COUPLED SYSTEM ANALYSIS")
    print("═" * 60)
    
    # Create default system
    system = create_default_coupled_system()
    
    # Natural frequencies
    f1, f2 = system.natural_frequencies()
    print(f"\nNatural frequencies:")
    print(f"  Mode 1: {f1:.2f} Hz (in-phase)")
    print(f"  Mode 2: {f2:.2f} Hz (out-of-phase)")
    
    # Mode shapes
    m1, m2 = system.mode_shapes()
    print(f"\nMode shapes [plate, body]:")
    print(f"  Mode 1: {m1}")
    print(f"  Mode 2: {m2}")
    
    # Transmissibility plot
    frequencies = np.linspace(1, 100, 500)
    T = system.transmissibility(frequencies)
    
    print("\nCoupling efficiency at key frequencies:")
    for f in [5, 10, 20, 40, 60]:
        eff = system.coupling_efficiency(f)
        print(f"  {f} Hz: {eff:.2%}")
    
    # Optimization recommendations
    recommendations = system.optimize_plate_for_coupling([40.0, 60.0])
    print(f"\nRecommendations for 40/60 Hz targets:")
    print(f"  Current coupling: {recommendations['current_coupling']:.2%}")
    print(f"  Suggested thickness: {recommendations.get('suggested_thickness_m', 0)*1000:.1f} mm")
    
    # Plot
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Transmissibility
        ax1.semilogy(frequencies, T, 'b-', linewidth=2)
        ax1.axvline(f1, color='r', linestyle='--', label=f'Mode 1: {f1:.1f} Hz')
        ax1.axvline(f2, color='g', linestyle='--', label=f'Mode 2: {f2:.1f} Hz')
        ax1.axhline(1, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Transmissibility |x_body/x_plate|')
        ax1.set_title('Body-Plate Transmissibility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 100])
        
        # FRF
        H_plate = system.transfer_function(frequencies, 0, 0)
        H_body = system.transfer_function(frequencies, 0, 1)
        
        ax2.semilogy(frequencies, np.abs(H_plate), 'b-', label='Plate response', linewidth=2)
        ax2.semilogy(frequencies, np.abs(H_body), 'r-', label='Body response', linewidth=2)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Receptance [m/N]')
        ax2.set_title('Frequency Response Functions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        
        plt.tight_layout()
        plt.savefig('/tmp/coupled_system_analysis.png', dpi=150)
        print("\n✓ Plot saved to /tmp/coupled_system_analysis.png")
    except Exception as e:
        print(f"\n⚠️ Could not create plot: {e}")
