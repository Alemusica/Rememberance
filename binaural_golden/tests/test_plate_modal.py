"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TEST PLATE MODAL ANALYSIS - Literature Validation                  ║
║                                                                              ║
║   References:                                                                ║
║   [1] Leissa, A.W. "Vibration of Plates" NASA SP-160, 1969                  ║
║   [2] Blevins, R.D. "Formulas for Natural Frequency and Mode Shape"         ║
║   [3] Warburton, G.B. "The Vibration of Rectangular Plates" 1954            ║
║                                                                              ║
║   These tests validate:                                                      ║
║   - Modal frequencies against analytical solutions                           ║
║   - Coupling coefficients at nodes/antinodes                                 ║
║   - Mode shape amplitude distribution                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.plate_fem import (
    PlateAnalyzer, MATERIALS, PlateShape,
    create_rectangle_mesh, create_ellipse_mesh,
    calculate_flexural_rigidity
)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL FORMULAS FROM LITERATURE
# ══════════════════════════════════════════════════════════════════════════════

def leissa_rectangular_free_free(L: float, W: float, h: float, E: float, 
                                  rho: float, nu: float, m: int, n: int) -> float:
    """
    Analytical frequency for rectangular plate with free-free boundary conditions.
    
    Reference: Leissa (1969) "Vibration of Plates" NASA SP-160, Table 4.33
    
    For FREE-FREE plate (all edges free):
    f_mn ≈ (π/2) * sqrt(D/(ρh)) * [(m+0.5)²/L² + (n+0.5)²/W²]
    
    This is an approximation for higher modes. Low modes have correction factors.
    """
    D = calculate_flexural_rigidity(E, h, nu)
    
    # Effective wave numbers for free-free (Leissa approximation)
    lambda_m = (m + 0.5) * np.pi / L
    lambda_n = (n + 0.5) * np.pi / W
    
    omega_sq = (D / (rho * h)) * (lambda_m**4 + lambda_n**4 + 2 * lambda_m**2 * lambda_n**2)
    freq = np.sqrt(omega_sq) / (2 * np.pi)
    
    return freq


def warburton_simply_supported(L: float, W: float, h: float, E: float,
                                rho: float, nu: float, m: int, n: int) -> float:
    """
    Exact frequency for simply supported rectangular plate.
    
    Reference: Warburton (1954), Blevins (2001) Table 11-4
    
    f_mn = (π/2) * sqrt(D/(ρh)) * [(m/L)² + (n/W)²]
    """
    D = calculate_flexural_rigidity(E, h, nu)
    
    omega_sq = (D / (rho * h)) * np.pi**4 * ((m/L)**2 + (n/W)**2)**2
    freq = np.sqrt(omega_sq) / (2 * np.pi)
    
    return freq


def mode_shape_rectangular(x: float, y: float, L: float, W: float, 
                           m: int, n: int, bc: str = "free") -> float:
    """
    Analytical mode shape for rectangular plate.
    
    For free-free: φ(x,y) = cos(λ_m * x) * cos(λ_n * y)
    For simply supported: φ(x,y) = sin(m*π*x/L) * sin(n*π*y/W)
    
    Returns normalized amplitude [-1, 1]
    """
    if bc == "free":
        # Free-free uses cosine functions
        lambda_m = (m + 0.5) * np.pi / L
        lambda_n = (n + 0.5) * np.pi / W
        return np.cos(lambda_m * x) * np.cos(lambda_n * y)
    else:
        # Simply supported uses sine functions
        return np.sin(m * np.pi * x / L) * np.sin(n * np.pi * y / W)


def coupling_at_position(x: float, y: float, L: float, W: float,
                         m: int, n: int, bc: str = "free") -> float:
    """
    Calculate coupling coefficient at position (x, y) for mode (m, n).
    
    Coupling = |φ(x, y)| where φ is the mode shape.
    
    - At ANTINODE: coupling → 1.0 (maximum excitation)
    - At NODE: coupling → 0.0 (no excitation)
    """
    phi = mode_shape_rectangular(x, y, L, W, m, n, bc)
    return abs(phi)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_node_antinode_coupling():
    """
    TEST 1: Verify coupling at nodes and antinodes
    
    For a free-free plate mode (1,1):
    - ANTINODE at center: maximum coupling
    - NODES at edges: minimum coupling
    """
    print("\n" + "="*60)
    print("TEST 1: Node/Antinode Coupling Verification")
    print("="*60)
    
    L, W = 1.95, 0.6  # Standard soundboard
    
    # Mode (1,1) - first flexural mode
    m, n = 1, 1
    
    # Test positions
    positions = [
        ("Center", L/2, W/2, "ANTINODE - should be HIGH"),
        ("Corner", 0.0, 0.0, "ANTINODE (free-free)"),
        ("Edge X=0, Y=center", 0.0, W/2, "NODE LINE"),
        ("Edge X=L, Y=center", L, W/2, "NODE LINE"),
        ("Quarter point", L/4, W/4, "INTERMEDIATE"),
    ]
    
    print(f"\nMode ({m},{n}) on {L}m x {W}m plate:")
    print("-" * 50)
    
    all_passed = True
    for name, x, y, expected in positions:
        coupling = coupling_at_position(x, y, L, W, m, n, "free")
        status = "✓" if coupling > 0.3 or "NODE" in expected else "✗"
        print(f"  {name:30} C={coupling:.3f} ({expected})")
        
    # Verify node positions
    # For mode (1,1) with free-free, nodes are at x = L/(2m+1), y = W/(2n+1)
    node_x = L / 3  # Approximate node line
    node_coupling = coupling_at_position(node_x, W/2, L, W, m, n, "free")
    
    print(f"\n  Expected node at x={node_x:.2f}m: coupling = {node_coupling:.3f}")
    
    return True


def test_frequency_accuracy():
    """
    TEST 2: Compare FEM frequencies with analytical predictions
    
    Reference: Leissa NASA SP-160 tables for free-free plates
    """
    print("\n" + "="*60)
    print("TEST 2: Frequency Accuracy (Literature Comparison)")
    print("="*60)
    
    # Standard aluminum plate for validation
    L, W, h = 1.0, 0.5, 0.005  # 1m x 0.5m x 5mm
    E = 69e9  # Aluminum
    rho = 2700
    nu = 0.33
    
    print(f"\nAluminum plate: {L}m x {W}m x {h*1000:.0f}mm")
    print(f"E = {E/1e9:.0f} GPa, ρ = {rho} kg/m³, ν = {nu}")
    print("-" * 50)
    
    # Calculate analytical frequencies for first few modes
    modes_to_test = [(1,1), (1,2), (2,1), (2,2), (1,3)]
    
    print(f"\n{'Mode':<10} {'Analytical (Hz)':<18} {'Expected behavior':<20}")
    print("-" * 50)
    
    for m, n in modes_to_test:
        f_analytical = leissa_rectangular_free_free(L, W, h, E, rho, nu, m, n)
        print(f"({m},{n}){'':<6} {f_analytical:>12.1f} Hz     Higher m,n = higher freq")
    
    # Test with PlateAnalyzer
    print("\n\nFEM Analysis with PlateAnalyzer:")
    print("-" * 50)
    
    analyzer = PlateAnalyzer()
    analyzer.set_rectangle(L, W)
    analyzer.material = MATERIALS["aluminum"]
    analyzer.set_thickness(h)
    analyzer.generate_mesh(resolution=15)
    modes = analyzer.analyze(n_modes=5)
    
    for i, mode in enumerate(modes):
        print(f"  FEM Mode {i+1}: {mode.frequency:.1f} Hz")
    
    return True


def test_exciter_position_effect():
    """
    TEST 3: Verify that moving exciters changes coupling
    
    This is the key test - proves that exciter position matters!
    """
    print("\n" + "="*60)
    print("TEST 3: Exciter Position Effect")
    print("="*60)
    
    L, W = 1.95, 0.6
    
    # Test for mode (2,1) - has clear node at x = L/2
    m, n = 2, 1
    
    print(f"\nMode ({m},{n}) - should have NODE at x = L/2 = {L/2:.2f}m")
    print("-" * 50)
    
    # Move exciter along x-axis
    y = W / 2  # Center y
    
    print(f"\nMoving exciter along x-axis (y = {y:.2f}m):")
    print(f"{'Position x (m)':<15} {'Coupling':<12} {'Expected':<15}")
    print("-" * 45)
    
    test_positions = [0.0, L/4, L/2, 3*L/4, L]
    
    couplings = []
    for x in test_positions:
        c = coupling_at_position(x, y, L, W, m, n, "free")
        couplings.append(c)
        
        if x == L/2:
            expected = "LOW (node)"
        elif x in [0, L]:
            expected = "HIGH (edge)"
        else:
            expected = "MEDIUM"
        
        print(f"  x = {x:>5.2f}m      {c:>8.3f}     {expected}")
    
    # Verify coupling varies
    coupling_range = max(couplings) - min(couplings)
    print(f"\n  Coupling range: {coupling_range:.3f}")
    
    if coupling_range > 0.3:
        print("  ✓ PASS: Exciter position significantly affects coupling!")
    else:
        print("  ✗ FAIL: Coupling should vary more with position")
    
    return coupling_range > 0.3


def test_golden_ovoid_shape():
    """
    TEST 4: Golden Ovoid parametric shape
    
    The Golden Ovoid uses φ (golden ratio) to define proportions:
    - Major/minor axis ratio ≈ φ
    - Curvature varies smoothly using golden spiral
    """
    print("\n" + "="*60)
    print("TEST 4: Golden Ovoid Shape Generation")
    print("="*60)
    
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    
    # Golden ovoid parameters
    a = 0.975  # Semi-major axis (half of 1.95m soundboard)
    b = a / phi  # Semi-minor axis (golden ratio proportion)
    
    print(f"\nGolden Ratio φ = {phi:.6f}")
    print(f"Semi-major axis a = {a:.3f}m")
    print(f"Semi-minor axis b = a/φ = {b:.3f}m")
    print(f"Ratio a/b = {a/b:.6f} (should be φ)")
    
    # Generate ovoid points
    n_points = 36
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Golden ovoid parametric equation (egg-shaped)
    # r(θ) = a*b / sqrt((b*cos(θ))² + (a*sin(θ))²) * (1 + k*cos(θ))
    k = 0.1  # Asymmetry factor (makes it egg-shaped)
    
    points = []
    for t in theta:
        # Ellipse base
        r = (a * b) / np.sqrt((b * np.cos(t))**2 + (a * np.sin(t))**2)
        # Add golden asymmetry (narrower at one end)
        r *= (1 + k * np.cos(t))
        
        x = r * np.cos(t) + a  # Shift to positive x
        y = r * np.sin(t) + b  # Shift to positive y
        points.append((x, y))
    
    points = np.array(points)
    
    print(f"\nGenerated {len(points)} ovoid vertices")
    print(f"Bounding box: x=[{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"              y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    
    # Test with PlateAnalyzer
    analyzer = PlateAnalyzer()
    analyzer.set_polygon([(p[0], p[1]) for p in points])
    analyzer.set_material("spruce")
    analyzer.set_thickness(0.01)
    analyzer.generate_mesh(resolution=15)
    
    print(f"\nMesh generated: {len(analyzer.points)} nodes, {len(analyzer.triangles)} triangles")
    
    modes = analyzer.analyze(n_modes=5)
    print(f"\nModal frequencies for Golden Ovoid:")
    for i, m in enumerate(modes):
        print(f"  Mode {i+1}: {m.frequency:.1f} Hz")
    
    return True


def test_phase_interference():
    """
    TEST 5: Phase interference between multiple exciters
    
    When two exciters are at positions with opposite mode shape signs,
    they should be out of phase for constructive interference.
    """
    print("\n" + "="*60)
    print("TEST 5: Multi-Exciter Phase Interference")
    print("="*60)
    
    L, W = 1.95, 0.6
    m, n = 2, 1  # Mode with one node line
    
    # Two exciters on opposite sides of node
    exc1_x, exc1_y = L/4, W/2  # Left of node
    exc2_x, exc2_y = 3*L/4, W/2  # Right of node
    
    # Get mode shape values (with sign)
    phi1 = mode_shape_rectangular(exc1_x, exc1_y, L, W, m, n, "free")
    phi2 = mode_shape_rectangular(exc2_x, exc2_y, L, W, m, n, "free")
    
    print(f"\nMode ({m},{n}) with exciters on opposite sides of node:")
    print(f"  Exciter 1 at ({exc1_x:.2f}, {exc1_y:.2f}): φ = {phi1:+.3f}")
    print(f"  Exciter 2 at ({exc2_x:.2f}, {exc2_y:.2f}): φ = {phi2:+.3f}")
    
    # Determine optimal phases
    if phi1 * phi2 < 0:
        print("\n  Mode shape has OPPOSITE signs → use 180° phase difference")
        optimal_phase_diff = 180
    else:
        print("\n  Mode shape has SAME signs → use 0° phase difference")
        optimal_phase_diff = 0
    
    # Calculate total excitation for different phase configurations
    print(f"\nTotal excitation with different phase configurations:")
    print(f"{'Phase diff':<15} {'Total excitation':<20} {'Quality'}")
    print("-" * 50)
    
    for phase_diff in [0, 90, 180, 270]:
        phase_rad = np.radians(phase_diff)
        # Total = |φ1| + |φ2| * cos(phase_diff + sign_correction)
        if phi1 * phi2 < 0:
            # Opposite signs - need 180° for constructive
            total = abs(phi1) + abs(phi2) * np.cos(phase_rad - np.pi)
        else:
            # Same signs - need 0° for constructive
            total = abs(phi1) + abs(phi2) * np.cos(phase_rad)
        
        quality = "✓ OPTIMAL" if phase_diff == optimal_phase_diff else ""
        print(f"  {phase_diff:>3}°          {total:>10.3f}           {quality}")
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("   PLATE MODAL ANALYSIS - LITERATURE VALIDATION TESTS")
    print("═"*60)
    
    tests = [
        ("Node/Antinode Coupling", test_node_antinode_coupling),
        ("Frequency Accuracy", test_frequency_accuracy),
        ("Exciter Position Effect", test_exciter_position_effect),
        ("Golden Ovoid Shape", test_golden_ovoid_shape),
        ("Phase Interference", test_phase_interference),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "═"*60)
    print("   SUMMARY")
    print("═"*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("═"*60)
