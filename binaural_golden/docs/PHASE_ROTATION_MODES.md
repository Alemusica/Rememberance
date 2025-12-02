# 🌳 Harmonic Tree - Phase Rotation Modes

## Overview

The Harmonic Tree generates a fundamental frequency plus harmonics arranged in a phyllotaxis pattern (like sunflower seeds). Each harmonic has a **phase offset** based on the Golden Angle (137.5°).

During **growth mode**, the phases can **evolve (rotate)** over time. This document describes the two available rotation modes.

---

## Mode 1: "Whole Tree Rotation" (Original Behavior)

**All phases rotate, including the fundamental.**

### Formula:
```
phase[i] = base_phase[i] + (elapsed_fraction × 2π × φ⁻ⁱ)
```

### Rotation speeds:
| Harmonic | Index | φ⁻ⁱ | Rotation at 100% |
|----------|-------|-----|------------------|
| **Fundamental** | 0 | 1.000 | **+360°** |
| H1 | 1 | 0.618 | +222.5° |
| H2 | 2 | 0.382 | +137.5° |
| H3 | 3 | 0.236 | +85.0° |
| H4 | 4 | 0.146 | +52.5° |
| H5 | 5 | 0.090 | +32.4° |

### Effect:
- The entire "tree" spirals as it grows
- Creates a swirling, cosmic effect
- Fundamental moves, so there's no fixed reference point
- All frequencies shift phase together (cohesive movement)

### Use case:
- Meditation: Creates a sense of flowing, cosmic rotation
- Trance: The constant phase movement induces altered states
- When you want everything to "breathe together"

---

## Mode 2: "Fixed Trunk, Rotating Branches" (New Behavior)

**Fundamental stays fixed at phase 0°, only harmonics rotate.**

### Formula:
```
phase[0] = 0  (always fixed)
phase[i] = base_phase[i] + (elapsed_fraction × 2π × φ⁻ⁱ)  for i > 0
```

### Rotation speeds:
| Harmonic | Index | φ⁻ⁱ | Rotation at 100% |
|----------|-------|-----|------------------|
| **Fundamental** | 0 | — | **0° (FIXED)** |
| H1 | 1 | 0.618 | +222.5° |
| H2 | 2 | 0.382 | +137.5° |
| H3 | 3 | 0.236 | +85.0° |
| H4 | 4 | 0.146 | +52.5° |
| H5 | 5 | 0.090 | +32.4° |

### Effect:
- The "trunk" (fundamental) is a stable reference
- Branches (harmonics) spiral around the fixed trunk
- Creates a sense of grounded growth
- Clearer harmonic relationships (reference point exists)

### Use case:
- Focus/Study: The stable fundamental provides grounding
- Healing: Fixed reference point for the brain to lock onto
- When you want harmonics to "dance around" a stable root

---

## UI Options

In the Harmonic Tree tab, under "Therapeutic Growth Mode":

1. **☑ Evolve phases during growth** - Master toggle for any phase rotation
2. **Rotation Mode** (when evolution is enabled):
   - **🌀 Whole Tree** - All phases rotate (original)
   - **🌲 Fixed Trunk** - Fundamental stays fixed (new)

---

## Mathematical Background

### Golden Angle (137.5°)
The golden angle is derived from the golden ratio:
```
φ = (1 + √5) / 2 ≈ 1.618034
Golden Angle = 360° × (1 - 1/φ) = 360° × φ⁻¹ ≈ 137.5077°
```

This is the angle that produces optimal packing in nature (sunflowers, pinecones, etc.).

### Phase Decay (φ⁻ⁱ)
Higher harmonics rotate more slowly because:
- Lower frequencies carry more energy
- Natural systems have more inertia at the base
- Creates a "trunk-to-branches" hierarchy

```
φ⁻⁰ = 1.000  (fundamental: fastest)
φ⁻¹ = 0.618
φ⁻² = 0.382
φ⁻³ = 0.236
φ⁻⁴ = 0.146
φ⁻⁵ = 0.090  (high harmonics: slowest)
```

---

## Code Location

The phase rotation logic is in `golden_studio.py`, method `_calculate_harmonics()`:

```python
# PHASES: Cumulative Golden Angle (phyllotaxis pattern)
base_phases = [(i * GOLDEN_ANGLE_RAD) % (2 * np.pi) for i in range(total)]

# Phase evolution during growth
if apply_growth and self.phase_evolution.get():
    phase_offset = elapsed_fraction * 2 * np.pi
    
    if self.fixed_trunk_mode.get():
        # Mode 2: Fixed trunk, rotating branches
        phases = [0.0]  # Fundamental fixed
        phases += [(base_phases[i] + phase_offset * PHI_CONJUGATE ** i) % (2 * np.pi) 
                   for i in range(1, total)]
    else:
        # Mode 1: Whole tree rotation (original)
        phases = [(base_phases[i] + phase_offset * PHI_CONJUGATE ** i) % (2 * np.pi) 
                  for i in range(total)]
else:
    phases = base_phases
```

---

## Commit History

- **Original behavior** preserved in commit: `🌳 Harmonic Tree: Phase evolution with rotating fundamental`
- **New mode added** in commit: (this commit)

Both behaviors are available via UI toggle.
