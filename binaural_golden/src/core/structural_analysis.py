"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              STRUCTURAL ANALYSIS - Deflection & Stress Calculation           ║
║                                                                              ║
║   Calcola:                                                                   ║
║   • Deflessione tavola sotto carico persona                                  ║
║   • Stress concentration ai cutouts                                          ║
║   • Distribuzione carico non uniforme (schiena/glutei/gambe)                 ║
║   • Verifica rigidezza minima                                                ║
║                                                                              ║
║   La persona è la "corda da accordare" - la tavola deve supportarla          ║
║   senza flettersi troppo (max ~10mm al centro)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum

# Try scikit-fem for accurate FEM
try:
    from skfem import *
    from skfem.helpers import ddot, sym_grad
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False

from scipy.interpolate import RectBivariateSpline


# ══════════════════════════════════════════════════════════════════════════════
# BODY LOAD DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BodyLoadDistribution:
    """
    Distribuzione del peso corporeo sulla tavola.
    
    La persona sdraiata NON ha peso uniforme:
    - Schiena (torace): ~40% del peso, zona centrale-alta
    - Glutei/bacino: ~35% del peso, zona centrale
    - Gambe: ~25% del peso, distribuito lungo
    
    Posizioni normalizzate (0 = piedi, 1 = testa)
    """
    # Zone corporee (start, end, weight_fraction)
    HEAD: Tuple[float, float, float] = (0.88, 1.00, 0.08)    # 8% peso
    SHOULDERS: Tuple[float, float, float] = (0.72, 0.88, 0.15)  # 15%
    THORAX: Tuple[float, float, float] = (0.55, 0.72, 0.25)   # 25% (schiena)
    PELVIS: Tuple[float, float, float] = (0.40, 0.55, 0.35)   # 35% (glutei)
    THIGHS: Tuple[float, float, float] = (0.20, 0.40, 0.12)   # 12%
    CALVES: Tuple[float, float, float] = (0.00, 0.20, 0.05)   # 5%
    
    def get_pressure_map(
        self,
        length: float,
        width: float,
        person_weight_kg: float,
        resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera mappa di pressione 2D sulla tavola.
        
        Args:
            length: Lunghezza tavola [m]
            width: Larghezza tavola [m]
            person_weight_kg: Peso persona [kg]
            resolution: Punti griglia
            
        Returns:
            (x_grid, y_grid, pressure_Pa) - Pressione in Pascal
        """
        x = np.linspace(0, length, resolution)
        y = np.linspace(0, width, resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Inizializza pressione
        pressure = np.zeros_like(X)
        
        # Larghezza corpo (centrato su y=width/2)
        body_half_width = width * 0.35  # ~70% della larghezza tavola
        y_center = width / 2
        
        # Per ogni zona corporea
        zones = [
            self.HEAD, self.SHOULDERS, self.THORAX,
            self.PELVIS, self.THIGHS, self.CALVES
        ]
        
        for x_start_norm, x_end_norm, weight_frac in zones:
            x_start = x_start_norm * length
            x_end = x_end_norm * length
            
            # Maschera spaziale (rettangolo smussato)
            x_mask = self._smooth_rect(X, x_start, x_end)
            y_mask = self._smooth_rect(Y, y_center - body_half_width, 
                                        y_center + body_half_width)
            
            # Area zona
            zone_length = x_end - x_start
            zone_area = zone_length * (2 * body_half_width)
            
            # Forza in questa zona [N]
            force_N = person_weight_kg * 9.81 * weight_frac
            
            # Pressione [Pa = N/m²]
            zone_pressure = force_N / max(zone_area, 0.01)
            
            pressure += zone_pressure * x_mask * y_mask
        
        return X, Y, pressure
    
    def _smooth_rect(self, arr: np.ndarray, low: float, high: float) -> np.ndarray:
        """Smooth rectangular mask with tanh edges."""
        sharpness = 20  # Edge sharpness
        mask = (np.tanh(sharpness * (arr - low)) + 1) / 2
        mask *= (np.tanh(sharpness * (high - arr)) + 1) / 2
        return mask


# ══════════════════════════════════════════════════════════════════════════════
# DEFLECTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeflectionResult:
    """Risultato analisi deflessione."""
    max_deflection_mm: float       # Deflessione massima [mm]
    deflection_field: np.ndarray   # Campo deflessione 2D
    x_grid: np.ndarray             # Griglia X
    y_grid: np.ndarray             # Griglia Y
    max_deflection_pos: Tuple[float, float]  # Posizione max deflessione
    is_acceptable: bool            # Deflessione < limite
    safety_factor: float           # Fattore di sicurezza


@dataclass
class StressResult:
    """Risultato analisi stress."""
    max_stress_MPa: float          # Stress massimo [MPa]
    stress_field: np.ndarray       # Campo stress Von Mises
    yield_stress_MPa: float        # Stress snervamento materiale
    safety_factor: float           # yield/max_stress
    is_safe: bool                  # safety_factor > 2.0
    stress_at_cutouts: List[float] # Stress ai bordi cutouts


class StructuralAnalyzer:
    """
    Analizzatore strutturale per tavola vibroacustica.
    
    Calcola deflessione e stress sotto carico persona.
    """
    
    # Limiti di progetto
    MAX_DEFLECTION_MM = 10.0  # Max deflessione accettabile
    MIN_SAFETY_FACTOR = 2.0   # Fattore sicurezza minimo
    
    # Stress snervamento materiali [MPa]
    YIELD_STRESS = {
        "birch_plywood": 40.0,
        "marine_plywood": 45.0,
        "spruce": 35.0,
        "oak": 50.0,
        "maple": 55.0,
        "mdf": 20.0,
    }
    
    def __init__(
        self,
        length: float,
        width: float,
        thickness: float,
        material: str = "birch_plywood",
        E_modulus: float = 13e9,  # Pa
        poisson: float = 0.33,
    ):
        """
        Args:
            length: Lunghezza tavola [m]
            width: Larghezza tavola [m]
            thickness: Spessore [m]
            material: Nome materiale
            E_modulus: Modulo Young [Pa]
            poisson: Coefficiente Poisson
        """
        self.length = length
        self.width = width
        self.thickness = thickness
        self.material = material
        self.E = E_modulus
        self.nu = poisson
        
        # Rigidezza flessionale D = E*h³/(12*(1-ν²))
        self.D = self.E * self.thickness**3 / (12 * (1 - self.nu**2))
        
        # Stress snervamento
        self.yield_stress = self.YIELD_STRESS.get(material, 40.0)
        
        # Body load distribution (MUST be in __init__)
        self.body_load = BodyLoadDistribution()
        
        # Grooves storage for local stiffness reduction
        self._grooves: List[Dict] = []
    
    def set_grooves(self, grooves: List) -> None:
        """
        Set grooves for local stiffness reduction calculation.
        
        LUTHERIE: Grooves (scanalature) reduce local bending stiffness,
        allowing fine tuning of mode frequencies without through-holes.
        
        Args:
            grooves: List of GrooveGene objects or dicts with groove params
        """
        self._grooves = []
        for g in grooves:
            if hasattr(g, 'x'):  # GrooveGene object
                self._grooves.append({
                    'x': g.x, 'y': g.y,
                    'length': g.length,
                    'angle': g.angle,
                    'depth': g.depth,
                    'width': g.width_mm / 1000,  # m
                })
            else:  # Dict
                self._grooves.append(g)
    
    def local_stiffness_factor(self, x_norm: float, y_norm: float) -> float:
        """
        Calculate local stiffness reduction factor due to grooves.
        
        LUTHERIE: Grooves reduce h locally → D_local = D * (1-depth)³
        
        Args:
            x_norm, y_norm: Normalized coordinates (0-1)
            
        Returns:
            Stiffness factor (0.3 - 1.0), where 1.0 = no reduction
        """
        if not self._grooves:
            return 1.0
        
        min_factor = 1.0
        
        for groove in self._grooves:
            gx, gy = groove['x'], groove['y']
            half_len = groove['length'] / 2
            half_width = groove['width'] / self.width / 2
            angle = groove['angle']
            depth = groove['depth']
            
            # Transform to groove-local coordinates
            dx = x_norm - gx
            dy = y_norm - gy
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx_rot = dx * cos_a + dy * sin_a
            dy_rot = -dx * sin_a + dy * cos_a
            
            # Check if point is within groove region
            if abs(dx_rot) < half_len and abs(dy_rot) < half_width:
                # Stiffness scales with h³
                h_remaining = 1.0 - depth
                local_factor = h_remaining ** 3
                min_factor = min(min_factor, local_factor)
        
        return max(0.3, min_factor)  # Never below 30% stiffness
    
    def effective_stiffness_map(self, resolution: int = 40) -> np.ndarray:
        """
        Generate 2D map of effective bending stiffness considering grooves.
        
        Returns:
            2D array of D_effective / D_nominal (values 0.3 - 1.0)
        """
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        
        stiffness_map = np.ones((resolution, resolution))
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                stiffness_map[i, j] = self.local_stiffness_factor(xi, yj)
        
        return stiffness_map
    
    def calculate_deflection(
        self,
        person_weight_kg: float,
        cutouts: Optional[List[Tuple[float, float, float]]] = None,
        resolution: int = 40,
        use_fem: bool = True
    ) -> DeflectionResult:
        """
        Calcola deflessione sotto carico persona.
        
        Args:
            person_weight_kg: Peso persona [kg]
            cutouts: Lista di (x, y, radius) per cutouts circolari
            resolution: Risoluzione griglia
            use_fem: Usa FEM se disponibile
            
        Returns:
            DeflectionResult
        """
        if use_fem and HAS_SKFEM:
            return self._deflection_fem(person_weight_kg, cutouts, resolution)
        else:
            return self._deflection_analytical(person_weight_kg, resolution)
    
    def _deflection_analytical(
        self,
        person_weight_kg: float,
        resolution: int
    ) -> DeflectionResult:
        """
        Calcolo analitico deflessione (Navier solution per piastra rettangolare).
        
        Piastra semplicemente appoggiata sui 4 lati.
        """
        a = self.length
        b = self.width
        D = self.D
        
        # Ottieni distribuzione carico
        X, Y, pressure = self.body_load.get_pressure_map(
            a, b, person_weight_kg, resolution
        )
        
        # Soluzione di Navier: serie doppia di Fourier
        w = np.zeros_like(X)
        
        # Numero termini serie
        n_terms = 15
        
        for m in range(1, n_terms + 1, 2):  # Solo dispari
            for n in range(1, n_terms + 1, 2):
                # Coefficiente Fourier del carico
                # Per carico uniforme: q_mn = 16*q0/(π²*m*n)
                # Approssimo con media della pressione pesata
                sin_mx = np.sin(m * np.pi * X / a)
                sin_ny = np.sin(n * np.pi * Y / b)
                
                q_mn = 4 / (a * b) * np.sum(pressure * sin_mx * sin_ny) * (a/resolution) * (b/resolution)
                
                # Denominatore
                denom = D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2
                
                # Contributo al campo deflessione
                w += (q_mn / denom) * sin_mx * sin_ny
        
        # Converti in mm
        w_mm = w * 1000
        
        # Trova massimo
        max_idx = np.unravel_index(np.argmax(np.abs(w_mm)), w_mm.shape)
        max_deflection = np.abs(w_mm[max_idx])
        max_pos = (X[max_idx], Y[max_idx])
        
        # Verifica accettabilità
        is_acceptable = max_deflection <= self.MAX_DEFLECTION_MM
        safety_factor = self.MAX_DEFLECTION_MM / max(max_deflection, 0.1)
        
        return DeflectionResult(
            max_deflection_mm=max_deflection,
            deflection_field=w_mm,
            x_grid=X,
            y_grid=Y,
            max_deflection_pos=max_pos,
            is_acceptable=is_acceptable,
            safety_factor=safety_factor
        )
    
    def _deflection_fem(
        self,
        person_weight_kg: float,
        cutouts: Optional[List[Tuple[float, float, float]]],
        resolution: int
    ) -> DeflectionResult:
        """
        Calcolo FEM con scikit-fem v11 API (più accurato, supporta cutouts).
        
        Usa approccio Reissner-Mindlin semplificato per piastra spessa.
        """
        from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, solve, condense
        from skfem.helpers import grad, dot
        
        # Crea mesh
        mesh = MeshTri.init_tensor(
            np.linspace(0, self.length, resolution),
            np.linspace(0, self.width, resolution)
        )
        
        # Rimuovi elementi nei cutouts (opzionale, se cutouts presenti)
        if cutouts:
            for cx, cy, r in cutouts:
                # Calcola centroidi elementi
                elem_centroids = mesh.p[:, mesh.t].mean(axis=2)
                dist = np.sqrt((elem_centroids[0] - cx)**2 + (elem_centroids[1] - cy)**2)
                keep = dist > r
                if np.any(~keep):
                    mesh = mesh.remove_elements(np.where(~keep)[0])
        
        # Basis con elementi P1 (lineare)
        basis = Basis(mesh, ElementTriP1())
        
        # Ottieni coordinate nodi
        X_mesh = mesh.p[0, :]
        Y_mesh = mesh.p[1, :]
        
        # Prepara mappa pressione
        X_grid, Y_grid, pressure = self.body_load.get_pressure_map(
            self.length, self.width, person_weight_kg, resolution
        )
        
        # Interpolatore per pressione
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (X_grid[:, 0], Y_grid[0, :]), pressure,
            bounds_error=False, fill_value=0
        )
        
        # Pressione su tutto il mesh (per quadrature)
        def pressure_at_points(x, y):
            pts = np.column_stack([x.flatten(), y.flatten()])
            return interp(pts).reshape(x.shape)
        
        # Rigidezza flessionale: per Kirchhoff plate, K = D * ∫∇²u·∇²v
        # Semplifico come Poisson-like: K = D * ∫∇u·∇v (approssimazione)
        @BilinearForm
        def stiffness(u, v, w):
            # Approssimazione: rigidezza come ∇u·∇v scalata
            return self.D * dot(grad(u), grad(v))
        
        K = stiffness.assemble(basis)
        
        # Carico distribuito
        @LinearForm
        def load(v, w):
            # w.x sono coordinate dei punti di quadratura [2, n_points]
            q = pressure_at_points(w.x[0], w.x[1])
            return q * v
        
        f = load.assemble(basis)
        
        # Boundary conditions (appoggio semplice: w=0 sui bordi)
        boundary_dofs = basis.get_dofs(mesh.boundary_facets()).flatten()
        
        # Risolvi sistema
        K_cond, f_cond, x_cond, I_cond = condense(K, f, D=boundary_dofs)
        w_interior = solve(K_cond, f_cond)
        
        # Ricostruisci soluzione completa
        w_fem = np.zeros(K.shape[0])
        w_fem[I_cond] = w_interior
        
        # Interpola su griglia regolare per output
        from scipy.interpolate import LinearNDInterpolator
        interp_w = LinearNDInterpolator(mesh.p.T, w_fem)
        w_grid = interp_w(X_grid, Y_grid)
        w_grid = np.nan_to_num(w_grid, nan=0.0)
        
        # Converti in mm (deflessione già in m dal sistema)
        # Nota: con approssimazione Poisson, scala per spessore
        scaling = (self.length / self.thickness) ** 2 / 1000  # fattore empirico
        w_mm = np.abs(w_grid) * scaling
        
        # Trova massimo
        max_idx = np.unravel_index(np.argmax(w_mm), w_mm.shape)
        max_deflection = w_mm[max_idx]
        max_pos = (X_grid[max_idx], Y_grid[max_idx])
        
        is_acceptable = max_deflection <= self.MAX_DEFLECTION_MM
        safety_factor = self.MAX_DEFLECTION_MM / max(max_deflection, 0.1)
        
        return DeflectionResult(
            max_deflection_mm=max_deflection,
            deflection_field=w_mm,
            x_grid=X_grid,
            y_grid=Y_grid,
            max_deflection_pos=max_pos,
            is_acceptable=is_acceptable,
            safety_factor=safety_factor
        )
    
    def calculate_stress(
        self,
        person_weight_kg: float,
        cutouts: Optional[List[Tuple[float, float, float]]] = None,
        resolution: int = 40
    ) -> StressResult:
        """
        Calcola stress (Von Mises) nella tavola.
        
        Per piastra: σ_max ≈ 6*M_max / h²
        dove M_max è il momento flettente massimo.
        """
        # Prima calcola deflessione
        defl = self.calculate_deflection(person_weight_kg, cutouts, resolution, use_fem=False)
        
        # Stress da curvatura: σ = E * z * κ
        # dove κ = ∂²w/∂x² + ∂²w/∂y² (curvatura)
        # z_max = h/2 (superficie)
        
        w = defl.deflection_field / 1000  # Torna in metri
        dx = self.length / resolution
        dy = self.width / resolution
        
        # Derivate seconde (curvatura)
        d2w_dx2 = np.gradient(np.gradient(w, dx, axis=0), dx, axis=0)
        d2w_dy2 = np.gradient(np.gradient(w, dy, axis=1), dy, axis=1)
        
        # Stress flessionale σ = E * (h/2) * |κ|
        curvature = np.abs(d2w_dx2) + np.abs(d2w_dy2)
        stress = self.E * (self.thickness / 2) * curvature
        
        # Converti in MPa
        stress_MPa = stress / 1e6
        
        # Stress ai cutouts (concentrazione)
        stress_at_cutouts = []
        if cutouts:
            for cx, cy, r in cutouts:
                # Trova punti vicini al bordo del cutout
                X, Y = defl.x_grid, defl.y_grid
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                near_edge = (dist > r * 0.9) & (dist < r * 1.3)
                if np.any(near_edge):
                    # Concentrazione stress (fattore ~2-3 tipico per fori)
                    edge_stress = np.max(stress_MPa[near_edge]) * 2.5
                    stress_at_cutouts.append(edge_stress)
        
        max_stress = np.max(stress_MPa)
        if stress_at_cutouts:
            max_stress = max(max_stress, max(stress_at_cutouts))
        
        safety_factor = self.yield_stress / max(max_stress, 0.1)
        is_safe = safety_factor >= self.MIN_SAFETY_FACTOR
        
        return StressResult(
            max_stress_MPa=max_stress,
            stress_field=stress_MPa,
            yield_stress_MPa=self.yield_stress,
            safety_factor=safety_factor,
            is_safe=is_safe,
            stress_at_cutouts=stress_at_cutouts
        )
    
    def verify_design(
        self,
        person_weight_kg: float,
        cutouts: Optional[List[Tuple[float, float, float]]] = None
    ) -> Dict[str, any]:
        """
        Verifica completa del design strutturale.
        
        Returns:
            Dict con risultati e raccomandazioni
        """
        defl = self.calculate_deflection(person_weight_kg, cutouts)
        stress = self.calculate_stress(person_weight_kg, cutouts)
        
        # Valutazione complessiva
        is_valid = defl.is_acceptable and stress.is_safe
        
        issues = []
        if not defl.is_acceptable:
            issues.append(f"Deflessione eccessiva: {defl.max_deflection_mm:.1f}mm > {self.MAX_DEFLECTION_MM}mm")
            issues.append(f"  → Aumentare spessore o usare materiale più rigido")
        
        if not stress.is_safe:
            issues.append(f"Stress eccessivo: {stress.max_stress_MPa:.1f}MPa (SF={stress.safety_factor:.2f})")
            issues.append(f"  → Ridurre cutouts o aumentare spessore")
        
        if stress.stress_at_cutouts:
            max_cutout_stress = max(stress.stress_at_cutouts)
            if max_cutout_stress > self.yield_stress * 0.6:
                issues.append(f"Stress alto ai cutouts: {max_cutout_stress:.1f}MPa")
                issues.append(f"  → Aumentare raggio cutouts o riposizionarli")
        
        return {
            "is_valid": is_valid,
            "deflection": defl,
            "stress": stress,
            "issues": issues,
            "thickness_m": self.thickness,
            "material": self.material,
            "person_weight_kg": person_weight_kg,
            "n_cutouts": len(cutouts) if cutouts else 0
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_plate_structural(
    length: float,
    width: float,
    thickness: float,
    person_weight_kg: float,
    material: str = "birch_plywood",
    cutouts: Optional[List[Tuple[float, float, float]]] = None
) -> Dict:
    """
    Quick structural check for plate design.
    
    Returns dict with is_valid, max_deflection_mm, max_stress_MPa, issues
    """
    analyzer = StructuralAnalyzer(
        length, width, thickness, material
    )
    return analyzer.verify_design(person_weight_kg, cutouts)


def minimum_thickness(
    length: float,
    width: float,
    person_weight_kg: float,
    material: str = "birch_plywood",
    max_deflection_mm: float = 10.0
) -> float:
    """
    Calcola spessore minimo per rispettare limite deflessione.
    
    Returns:
        Spessore minimo [m]
    """
    # Stima iniziale dalla formula analitica
    # w_max ≈ q*L⁴/(D) → D ≈ q*L⁴/w_max → h³ ≈ 12*(1-ν²)*q*L⁴/(E*w_max)
    
    E = 13e9  # Approssimazione
    nu = 0.33
    q = person_weight_kg * 9.81 / (length * width)  # Pressione media
    w_max = max_deflection_mm / 1000
    
    h_cubed = 12 * (1 - nu**2) * q * length**4 / (E * w_max * 100)  # Fattore empirico
    h_min = h_cubed ** (1/3)
    
    # Arrotonda a mm
    h_min = np.ceil(h_min * 1000) / 1000
    
    return max(h_min, 0.010)  # Minimo 10mm


# ══════════════════════════════════════════════════════════════════════════════
# PENINSULA DETECTION - Isolated Regions from Intersecting Cutouts
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PeninsulaResult:
    """
    Result of peninsula/isolated region detection.
    
    PARADIGM SHIFT (based on ABH research - Krylov 2014, Deng 2019):
    Isolated regions are NOT necessarily bad! They can:
    1. Focus vibrational energy (like Acoustic Black Holes)
    2. Create local resonators for specific frequency bands
    3. Improve low-frequency response when properly designed
    
    We evaluate BOTH the structural risk AND the acoustic benefit.
    """
    has_peninsulas: bool              # True if isolated regions exist
    n_regions: int                    # Number of connected regions (1 = OK)
    peninsula_areas: List[float]      # Area of each isolated region [m²]
    peninsula_positions: List[Tuple[float, float]]  # Centroid of each peninsula
    main_region_fraction: float       # Fraction of plate in main region
    
    # Legacy penalty (for backwards compatibility)
    structural_penalty: float         # Structural risk factor (0 = safe, 1 = risky)
    
    # NEW: ABH-inspired benefit analysis
    abh_benefit: float                # Energy focusing potential [0-1]
    resonator_potential: float        # Local resonance enhancement potential [0-1]
    taper_quality: float              # How well positioned for energy focusing [0-1]
    
    grid_visualization: np.ndarray    # 2D grid for debugging (labeled regions)


def detect_peninsulas(
    length: float,
    width: float,
    cutouts: List[Dict],
    resolution: int = 100
) -> PeninsulaResult:
    """
    Detect isolated regions (peninsulas) caused by intersecting cutouts.
    
    TOPOLOGY OPTIMIZATION INSIGHT:
    When cutouts intersect or come close, they can create isolated "islands"
    or thin "peninsulas" that are structurally weak and vibrate independently.
    These should be penalized in the fitness function.
    
    Uses connected component analysis on a rasterized plate representation.
    
    Args:
        length: Plate length [m]
        width: Plate width [m]  
        cutouts: List of cutout dicts with 'x', 'y', 'size', 'shape', 'rotation'
        resolution: Grid resolution for rasterization
        
    Returns:
        PeninsulaResult with detection results
    """
    from scipy import ndimage
    
    # Create rasterized plate (1 = solid, 0 = cutout)
    grid = np.ones((resolution, resolution), dtype=np.int32)
    
    # Pixel size
    dx = length / resolution
    dy = width / resolution
    
    # Create coordinate arrays
    x_coords = np.linspace(0, 1, resolution)  # Normalized
    y_coords = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Remove cutout regions from grid
    for cutout in cutouts:
        cx = cutout.get('x', 0.5)
        cy = cutout.get('y', 0.5)
        size = cutout.get('size', 0.05)
        shape = cutout.get('shape', 'ellipse')
        rotation = cutout.get('rotation', 0)
        aspect = cutout.get('aspect', 1.0)
        
        # Transform to cutout-local coordinates (with rotation)
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        dx_local = (X - cx) * cos_r + (Y - cy) * sin_r
        dy_local = -(X - cx) * sin_r + (Y - cy) * cos_r
        
        # Check if inside cutout based on shape
        if shape == 'ellipse':
            inside = (dx_local / size)**2 + (dy_local / (size * aspect))**2 < 1
        elif shape == 'rectangle':
            inside = (np.abs(dx_local) < size) & (np.abs(dy_local) < size * aspect)
        elif shape == 'circle':
            inside = dx_local**2 + dy_local**2 < size**2
        elif shape == 'stadium':
            # Rounded rectangle (pill shape)
            r = size * 0.5
            inside = ((np.abs(dx_local) < size - r) & (np.abs(dy_local) < size * aspect)) | \
                     ((dx_local - (size - r))**2 + dy_local**2 < r**2) | \
                     ((dx_local + (size - r))**2 + dy_local**2 < r**2)
        elif shape == 'triangle':
            # Equilateral triangle pointing up
            inside = (dy_local > -size * 0.5) & \
                     (dy_local < size * np.sqrt(3)/2 - np.abs(dx_local) * np.sqrt(3))
        elif shape in ['kidney', 's_curve', 'arc']:
            # Approximate as offset circles for these curved shapes
            inside = dx_local**2 + dy_local**2 < size**2
        else:
            # Default to ellipse
            inside = (dx_local / size)**2 + (dy_local / (size * aspect))**2 < 1
        
        # Mark as hole in grid
        grid[inside] = 0
    
    # Connected component analysis
    # Structure for 4-connectivity (only horizontal/vertical neighbors)
    # Use 8-connectivity for diagonal connections
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])
    
    labeled_grid, n_regions = ndimage.label(grid, structure=structure)
    
    # Find main region (largest)
    region_sizes = []
    for region_id in range(1, n_regions + 1):
        region_mask = labeled_grid == region_id
        region_sizes.append(np.sum(region_mask))
    
    if not region_sizes:
        # No solid regions at all (extreme case)
        return PeninsulaResult(
            has_peninsulas=False,
            n_regions=0,
            peninsula_areas=[],
            peninsula_positions=[],
            main_region_fraction=0.0,
            structural_penalty=1.0,
            grid_visualization=labeled_grid
        )
    
    main_region_idx = np.argmax(region_sizes) + 1
    main_region_size = region_sizes[main_region_idx - 1]
    total_solid = np.sum(grid)
    main_region_fraction = main_region_size / max(total_solid, 1)
    
    # Find peninsulas (regions that are not the main region)
    peninsula_areas = []
    peninsula_positions = []
    
    for region_id in range(1, n_regions + 1):
        if region_id == main_region_idx:
            continue
        
        region_mask = labeled_grid == region_id
        region_size = region_sizes[region_id - 1]
        
        # Area in m²
        area = region_size * dx * dy
        peninsula_areas.append(area)
        
        # Centroid
        coords = np.where(region_mask)
        centroid_x = np.mean(coords[0]) / resolution * length
        centroid_y = np.mean(coords[1]) / resolution * width
        peninsula_positions.append((centroid_x, centroid_y))
    
    # Calculate structural penalty
    # Small isolated regions are worse than one connected piece
    has_peninsulas = n_regions > 1
    
    if has_peninsulas:
        # ══════════════════════════════════════════════════════════════════
        # STRUCTURAL RISK (legacy penalty)
        # ══════════════════════════════════════════════════════════════════
        penalty = 0.0
        
        # Penalty for number of regions (exponential)
        penalty += 0.2 * (n_regions - 1)
        
        # Penalty for isolated area fraction
        isolated_fraction = 1.0 - main_region_fraction
        penalty += 0.5 * isolated_fraction
        
        # Cap penalty at 1.0
        penalty = min(1.0, penalty)
        
        # ══════════════════════════════════════════════════════════════════
        # ABH BENEFIT ANALYSIS (NEW! Based on Krylov 2014, Deng 2019)
        # 
        # Acoustic Black Holes show that isolated tapered regions can:
        # 1. Focus vibrational energy (not trap it)
        # 2. Create broadband absorption/radiation
        # 3. Enhance low-frequency response
        # ══════════════════════════════════════════════════════════════════
        
        abh_benefit = 0.0
        resonator_potential = 0.0
        taper_quality = 0.0
        
        for i, (area, pos) in enumerate(zip(peninsula_areas, peninsula_positions)):
            # Position score: edge peninsulas are better for ABH
            # (energy naturally flows to boundaries)
            x_norm = pos[0] / length
            y_norm = pos[1] / width
            edge_distance = min(x_norm, 1-x_norm, y_norm, 1-y_norm)
            position_score = 1.0 - edge_distance * 2  # Higher if closer to edge
            
            # Size score: medium-sized peninsulas are best
            # Too small = weak, too large = just another plate section
            plate_area = length * width
            area_ratio = area / plate_area
            # Optimal range: 5-20% of plate area
            if 0.05 <= area_ratio <= 0.20:
                size_score = 1.0
            elif area_ratio < 0.05:
                size_score = area_ratio / 0.05  # Linear ramp up
            else:
                size_score = max(0, 1.0 - (area_ratio - 0.20) / 0.30)
            
            # ABH benefit: combines position and size
            # Ref: Deng 2019 - ring-shaped ABH at edges most effective
            peninsula_abh = position_score * 0.6 + size_score * 0.4
            abh_benefit = max(abh_benefit, peninsula_abh)
            
            # Resonator potential: small isolated regions can resonate
            # Ref: Zhao 2014 - energy harvesting from ABH
            if area_ratio < 0.15:
                resonator_potential = max(resonator_potential, size_score * 0.8)
        
        # Taper quality: how well the connection to main plate
        # supports energy focusing (narrow connection = better taper)
        # This is approximated from the isolated fraction
        taper_quality = min(1.0, isolated_fraction * 5)  # 20% isolated = perfect
        
    else:
        penalty = 0.0
        abh_benefit = 0.0
        resonator_potential = 0.0
        taper_quality = 0.0
    
    return PeninsulaResult(
        has_peninsulas=has_peninsulas,
        n_regions=n_regions,
        peninsula_areas=peninsula_areas,
        peninsula_positions=peninsula_positions,
        main_region_fraction=main_region_fraction,
        structural_penalty=penalty,
        abh_benefit=abh_benefit,
        resonator_potential=resonator_potential,
        taper_quality=taper_quality,
        grid_visualization=labeled_grid
    )


def check_cutout_connectivity(cutouts: List[Dict], margin: float = 0.02) -> Dict:
    """
    Quick check if cutouts might create connectivity issues.
    
    Checks:
    1. Cutouts too close to each other (might merge)
    2. Cutouts too close to edge (might create thin strips)
    3. Cutouts overlapping
    
    Args:
        cutouts: List of cutout dicts
        margin: Minimum distance between cutouts [normalized]
        
    Returns:
        Dict with warnings and suggestions
    """
    warnings = []
    suggestions = []
    
    for i, c1 in enumerate(cutouts):
        x1, y1, s1 = c1.get('x', 0.5), c1.get('y', 0.5), c1.get('size', 0.05)
        
        # Check edge proximity
        if x1 - s1 < margin:
            warnings.append(f"Cutout {i} too close to left edge")
            suggestions.append(f"Move cutout {i} right or reduce size")
        if x1 + s1 > 1 - margin:
            warnings.append(f"Cutout {i} too close to right edge")
        if y1 - s1 < margin:
            warnings.append(f"Cutout {i} too close to bottom edge")
        if y1 + s1 > 1 - margin:
            warnings.append(f"Cutout {i} too close to top edge")
        
        # Check inter-cutout distance
        for j, c2 in enumerate(cutouts[i+1:], start=i+1):
            x2, y2, s2 = c2.get('x', 0.5), c2.get('y', 0.5), c2.get('size', 0.05)
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            min_dist = s1 + s2 + margin
            
            if dist < min_dist:
                if dist < s1 + s2:
                    warnings.append(f"Cutouts {i} and {j} OVERLAP!")
                else:
                    warnings.append(f"Cutouts {i} and {j} very close (may create thin strip)")
                suggestions.append(f"Increase distance between cutouts {i} and {j}")
    
    return {
        "has_issues": len(warnings) > 0,
        "warnings": warnings,
        "suggestions": suggestions,
        "n_cutouts": len(cutouts)
    }
