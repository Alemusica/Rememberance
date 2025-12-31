# Multi-Objective Evolution Stack - Design Document

## Current Problems

### 1. Simplistic Total Fitness (Scalar Combination)
```python
# Current approach - BAD for true multi-objective optimization
result.total_fitness = (
    weight_flatness * flatness_score +
    weight_spine * spine_coupling_score +
    weight_mass * low_mass_score +
    # ... more bonus/penalties
)
```

**Issues:**
- Single scalar loses trade-off information
- Fixed weights bias solutions toward one objective
- No Pareto front exploration
- Can't answer: "What's the best plate for 80% spine priority?"

### 2. Generic Cutout/Spiral Placement
- Cutouts placed "randomly" without physics guidance
- Spirals not symmetric when they should be
- No consideration of mode antinode positions
- ABH bonus is empirical, not physics-derived

### 3. Missing Labeled Objectives
- UI shows "combined 67%" but user doesn't know WHY
- No breakdown: "67% = 80% spine + 40% L/R balance + ..."
- Hard to debug evolution direction

---

## Solution: Multi-Dimensional Fitness Architecture

### A. Explicit Objective Vector (not scalar)

```python
@dataclass
class ObjectiveVector:
    """Explicit objectives for Pareto optimization."""
    
    # === ZONE FLATNESS (Hz response uniformity) ===
    spine_flatness: float  # 0-1, 1=perfectly flat 20-200Hz at spine
    ear_flatness: float    # 0-1, 1=perfectly flat at ears
    
    # === L/R BALANCE (Critical for binaural!) ===
    ear_lr_uniformity: float  # 0-1, 1=perfect L/R symmetry
    
    # === ENERGY DELIVERY ===
    spine_energy_efficiency: float  # How much exciter power reaches spine
    ear_energy_efficiency: float    # How much reaches ears
    
    # === STRUCTURAL ===
    mass_efficiency: float    # 1/mass normalized
    structural_safety: float  # deflection < 10mm → 1.0
    
    # === MANUFACTURABILITY ===
    cnc_complexity: float  # 1.0 = simple paths, 0.0 = very complex
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for NSGA-II."""
        return np.array([
            -self.spine_flatness,      # Minimize negative (maximize)
            -self.ear_flatness,
            -self.ear_lr_uniformity,
            -self.spine_energy_efficiency,
            -self.mass_efficiency,
            -self.structural_safety,
            -self.cnc_complexity,
        ])
```

### B. NSGA-II/III Pareto Front

```python
# In pymoo_optimizer.py - ENHANCED
class PlateOptimizationProblem(PymooProblem):
    def __init__(self, ...):
        super().__init__(
            n_var=n_decision_vars,
            n_obj=7,  # Multi-objective, not 1!
            n_ieq_constr=1,  # Structural constraint
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        F = []  # Objective values
        G = []  # Constraints
        
        for x in X:
            genome = self._decode_genome(x)
            obj_vec = self.evaluator.evaluate_multi(genome)
            F.append(obj_vec.to_array())
            G.append([genome.max_deflection - 10.0])  # <= 0
        
        out["F"] = np.array(F)
        out["G"] = np.array(G)
```

### C. Physics-Based Cutout Placement

Reference: Schleske 2002, Krylov 2014, Deng 2019

```python
class CutoutPlacementOptimizer:
    """
    Physics-based cutout placement for DML plates.
    
    Principles:
    1. Cutouts at MODE ANTINODES shift those frequencies maximally
    2. Cutouts at MODE NODES have minimal effect
    3. Edge cutouts → ABH energy focusing (Krylov 2014)
    4. SYMMETRIC cutouts for SYMMETRIC modes (reduce coupling)
    5. ASYMMETRIC cutouts for ASYMMETRIC modes (break degeneracy)
    """
    
    def suggest_cutout_positions(
        self,
        mode_shapes: np.ndarray,
        target_frequencies: List[float],
        current_frequencies: List[float],
    ) -> List[Tuple[float, float]]:
        """
        Suggest optimal cutout positions based on mode analysis.
        
        Args:
            mode_shapes: Modal analysis results
            target_frequencies: Desired frequency distribution
            current_frequencies: Current plate frequencies
        
        Returns:
            List of (x, y) normalized positions for cutouts
        """
        suggestions = []
        
        for i, (target_f, current_f) in enumerate(zip(target_frequencies, current_frequencies)):
            if abs(target_f - current_f) / current_f > 0.05:  # 5% deviation
                # Find antinode of this mode
                mode = mode_shapes[i]
                antinode_y, antinode_x = np.unravel_index(np.argmax(np.abs(mode)), mode.shape)
                antinode_x_norm = antinode_x / mode.shape[1]
                antinode_y_norm = antinode_y / mode.shape[0]
                
                # For SYMMETRIC modes (m, n both even), suggest symmetric cutouts
                if self._is_symmetric_mode(i):
                    # Suggest pair of cutouts symmetric about centerline
                    suggestions.append((antinode_x_norm, 0.5 - 0.15))
                    suggestions.append((antinode_x_norm, 0.5 + 0.15))
                else:
                    # For asymmetric modes, single cutout at antinode
                    suggestions.append((antinode_x_norm, antinode_y_norm))
        
        return suggestions
    
    def _is_symmetric_mode(self, mode_idx: int) -> bool:
        """Determine if mode has symmetric shape."""
        # Mode (m, n): symmetric if both m and n are odd
        # Mode indexing: sorted by frequency
        # Heuristic: lower modes tend to be symmetric
        return mode_idx < 3  # First 3 modes typically symmetric
```

### D. Spiral Placement Physics

```python
class SpiralCutoutStrategy:
    """
    Physics-based spiral cutout placement.
    
    PRINCIPLES (ABH + Lutherie):
    1. Spirals at CORNERS → Maximum ABH energy trapping (Deng 2019)
    2. Spirals should taper toward edge → ABH profile
    3. SYMMETRIC spirals at opposing corners for even modes
    4. Golden ratio spiral for optimal energy focusing
    
    CNC CONSIDERATIONS:
    - Spiral follows golden ratio: r(θ) = a * e^(b*θ) where b = ln(φ)/(π/2)
    - Minimum radius limited by CNC bit size (typ. 3mm)
    - Entry/exit paths for clean cuts
    """
    
    def generate_spiral_pair(
        self,
        plate_genome: PlateGenome,
        corner: str = "bottom_left",  # or "bottom_right", "top_left", "top_right"
        mirror: bool = True,  # Create symmetric pair
    ) -> List[CutoutGene]:
        """
        Generate physics-optimized spiral cutout(s).
        
        SYMMETRIC PLACEMENT (mirror=True):
        - bottom_left + top_right (diagonal)
        - Balances energy distribution for symmetric modes
        
        ASYMMETRIC PLACEMENT (mirror=False):
        - Single spiral for breaking mode degeneracy
        - Useful for shifting specific frequencies
        """
        spirals = []
        
        # Golden ratio spiral parameters
        phi = (1 + np.sqrt(5)) / 2  # 1.618...
        b = np.log(phi) / (np.pi / 2)  # Growth rate
        
        # Position based on corner
        corner_positions = {
            "bottom_left": (0.10, 0.10),
            "bottom_right": (0.10, 0.90),
            "top_left": (0.90, 0.10),
            "top_right": (0.90, 0.90),
        }
        
        x, y = corner_positions[corner]
        
        spirals.append(CutoutGene(
            x=x, y=y,
            width=0.08, height=0.08,
            shape="spiral",
            rotation=0.0,
        ))
        
        if mirror:
            # Diagonal mirror for symmetric effect
            mirror_corner = {
                "bottom_left": "top_right",
                "bottom_right": "top_left",
                "top_left": "bottom_right",
                "top_right": "bottom_left",
            }[corner]
            mx, my = corner_positions[mirror_corner]
            spirals.append(CutoutGene(
                x=mx, y=my,
                width=0.08, height=0.08,
                shape="spiral",
                rotation=np.pi,  # Rotated 180° for symmetry
            ))
        
        return spirals
```

---

## UI Enhancement: Labeled Fitness Display

### Current (Bad):
```
Combined: 67%
```

### Proposed (Good):
```
┌──────────────────────────────────────────┐
│ SPINE ZONE                               │
│   Flatness (20-200Hz): ████████░░ 82%    │
│   Energy efficiency:   ██████░░░░ 61%    │
├──────────────────────────────────────────┤
│ EAR/HEAD ZONE                            │
│   Flatness:            █████░░░░░ 54%    │
│   L/R Uniformity:      ██████████ 97%    │  ← Critical!
├──────────────────────────────────────────┤
│ STRUCTURAL                               │
│   Deflection: 6.2mm ✓ (< 10mm limit)     │
│   Mass: 18.4 kg (target: < 25kg)         │
├──────────────────────────────────────────┤
│ PARETO RANKING: Solution #3 of 12        │
│ Dominated by: None (Pareto optimal)      │
└──────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Multi-Objective Refactoring (This PR)
1. ✅ Add `ObjectiveVector` dataclass
2. ✅ Modify `FitnessEvaluator.evaluate()` to return multi-dimensional
3. ✅ Update `pymoo_optimizer.py` to use proper Pareto front
4. ✅ Add labeled breakdown to UI

### Phase 2: Physics-Based Cutouts (Next PR)
1. Add `CutoutPlacementOptimizer` class
2. Integrate modal analysis → cutout suggestion
3. Symmetric/asymmetric spiral logic
4. CNC path validation

### Phase 3: Interactive Pareto Exploration
1. UI slider: "Spine ↔ Head priority"
2. Real-time Pareto front visualization
3. User can pick any Pareto-optimal solution

---

## References

1. **Deb et al. 2002**: "A Fast and Elitist Multi-objective GA: NSGA-II"
2. **Krylov 2014**: "Acoustic Black Holes: Recent developments"
3. **Schleske 2002**: "On making violins - acoustics and perception"
4. **Bai & Liu 2004**: "Genetic algorithm for exciter placement in DML"
5. **Deng 2019**: "Ring-shaped ABH for broadband vibration isolation"
6. **Zhao 2025**: "ABH plates with cutouts for enhanced energy focusing"
