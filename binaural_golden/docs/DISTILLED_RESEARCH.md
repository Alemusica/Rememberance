# Distilled Research Knowledge - Plate Optimization

## 1. Multi-Exciter Placement (Lu, Shen, Bai)

### Key Findings
- **Multiple exciters** (2-4) significantly improve frequency response flatness
- **Attached masses** can shape modal patterns without adding exciters
- **Genetic algorithms** outperform grid search for exciter placement

### Algorithm: Optimal Exciter Placement (Bai & Liu 2004)
```python
def optimize_exciter_placement(plate, n_exciters, target_zones):
    """
    Genetic algorithm for exciter placement.
    
    Chromosome: [x1, y1, x2, y2, ..., xn, yn]  # Exciter positions
    Fitness: -std(response_at_zones)  # Minimize variance
    
    Crossover: Blend (BLX-α with α=0.5)
    Mutation: Gaussian perturbation σ=0.05*plate_size
    Selection: Tournament k=3
    
    Constraint: Exciters must be > 5cm apart
    """
    # Key insight: Place exciters at modal nodal lines
    # to excite maximum number of modes uniformly
```

### Implementation Notes
- Lu 2012: "Exciters near edges excite more modes but less uniformly"
- Shen 2016: "Attached masses at 1/3 and 2/3 points improve flatness"
- Sum & Pan 2000: "Cross-coupling between modes affects zone response"

---

## 2. Acoustic Black Holes (Krylov, Deng, Zhao)

### Principle
ABH are regions where thickness tapers to near-zero, causing:
- Wave speed → 0 (waves can't escape)
- Energy concentration → High
- Dissipation → Very effective with damping layer

### Design Rules
```python
def abh_profile(x, h0, m=2, x_abh=0.1):
    """
    ABH thickness profile.
    
    h(x) = h0 * (x / x_abh)^m
    
    Parameters:
    - h0: Initial thickness
    - m: Power law exponent (m ≥ 2 for ABH effect)
    - x_abh: ABH length
    
    Krylov criterion: m ≥ 2 for wave trapping
    """
```

### For Plate Design
- **Corner peninsulas** can act as ABH (taper toward tip)
- **Edge peninsulas** focus energy along boundary
- **Damping layer** (e.g., viscoelastic tape) at ABH tip essential

### Net Benefit Calculation
```python
def peninsula_net_benefit(peninsula, plate):
    # From Deng 2019 + Feurtado 2017
    
    # Benefit factors
    edge_proximity = 1.0 if peninsula.touches_edge else 0.5
    taper_quality = assess_taper(peninsula)  # 0-1
    size_factor = peninsula.area / plate.area  # Small = good
    
    abh_benefit = edge_proximity * taper_quality * (1 - size_factor)
    
    # Risk factors
    structural_risk = 0.3 if peninsula.is_thin_neck else 0.0
    
    # Net score
    return abh_benefit * 0.7 - structural_risk * 0.3
```

---

## 3. Zone Frequency Response (Sum & Pan 2000)

### Modal Cross-Coupling
When exciter drives mode m, response at point (x,y) depends on:
```python
def zone_response(exciter_pos, zone_center, frequencies, modes):
    """
    Response at zone from exciter.
    
    H(ω) = Σ_n [φ_n(x_exc) * φ_n(x_zone)] / [ω_n² - ω² + j*2*ζ*ω_n*ω]
    
    Where:
    - φ_n = mode shape function
    - ω_n = natural frequency of mode n
    - ζ = damping ratio
    """
```

### Ear Zone Requirements (Binaural)
- **Flatness**: < 6dB variation across 50-8000 Hz
- **Uniformity**: L/R response within 2dB at all frequencies
- **Critical**: 1-4 kHz range (speech frequencies)

### Spine Zone Requirements (Vibroacoustic)
- **Energy**: Maximum in 20-300 Hz (body resonance)
- **Flatness**: < 10dB variation
- **Coupling**: Good at C7-L5 vertebrae positions

---

## 4. Topology Optimization (Christensen, Bezzola)

### SIMP Method (Solid Isotropic Material with Penalization)
```python
def simp_density(x, p=3):
    """
    Penalized density for topology optimization.
    
    E(x) = x^p * E0
    
    p=3 typical, forces x toward 0 or 1 (binary)
    """
```

### For Plate Cutouts
- Cutouts should align with modal nodal lines
- Avoid cutting through high-stress regions
- Maintain structural connectivity

### Loudspeaker Cabinet (Christensen 2008)
- Topology optimization reduces cabinet resonances
- Internal bracing patterns emerge naturally
- Can improve low-frequency response

---

## 5. Practical Implementation Guidelines

### For Golden Studio
1. **Start with rectangular plate** (easier modal analysis)
2. **Add 2-3 exciters** using GA placement
3. **Test peninsula shapes** - they may help, not hurt!
4. **Evaluate ear uniformity first** - hardest target to meet
5. **Add cutouts carefully** - check modal patterns

### Optimization Workflow
```python
def optimize_plate(person, constraints):
    # Phase 1: Coarse search (10 generations, 20 population)
    coarse_result = nsga2(
        population=20,
        generations=10,
        mutation_rate=0.2
    )
    
    # Phase 2: Fine-tune best candidates (20 generations)
    fine_result = nsga2(
        initial_population=coarse_result.top(5),
        generations=20,
        mutation_rate=0.05
    )
    
    # Phase 3: Local refinement
    return gradient_descent(fine_result.best)
```

### Known Issues to Solve
1. **ear_uniformity at 6%** - Need asymmetric exciter placement or contour
2. **Computation speed** - FEM is slow, use analytical mode first
3. **Convergence** - NSGA-II needs proper hyperparameters

---

## References (Full in vibroacoustic_references.bib)

- Lu et al. 2012 - Multi-exciter DML optimization
- Bai & Liu 2004 - GA for exciter placement
- Krylov 2014 - ABH theory
- Deng et al. 2019 - Ring-shaped ABH
- Sum & Pan 2000 - Modal cross-coupling
- Christensen 2008 - Cabinet topology optimization
