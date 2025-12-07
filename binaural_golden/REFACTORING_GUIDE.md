# 🏗️ Refactoring Architecture - Development Guide

## 📊 Current State

### Monolith to Modular Migration
The `golden_studio.py` file (4010 lines) is being refactored into a modular architecture.

### Progress
- [x] Create folder structure
- [x] Extract `AudioEngine` → `core/audio_engine.py`
- [x] Create `golden_math.py` with φ-based functions
- [x] Create `Program/Step` system for sequences
- [x] Move tests to `tests/` folder
- [x] Generate `chakra_sunrise.json` preset
- [x] Extract `VibroacousticTab` → `ui/vibroacoustic_tab.py` (~700 lines)
- [ ] Extract `BinauralTab` → `ui/tabs/binaural_tab.py`
- [ ] Extract `SpectralTab` → `ui/tabs/spectral_tab.py`
- [ ] Extract `MolecularTab` → `ui/tabs/molecular_tab.py`
- [ ] Extract `HarmonicTreeTab` → `ui/tabs/harmonic_tab.py`
- [ ] Create reusable widgets
- [ ] Integrate Program system with UI
- [ ] Delete legacy `gui_*.py` files
- [ ] Create new `main.py` entry point

---

## 📁 Target Structure

```
binaural_golden/
├── src/
│   ├── main.py                  # NEW entry point (clean)
│   │
│   ├── core/                    # ✅ DONE - Core audio & math
│   │   ├── __init__.py
│   │   ├── audio_engine.py      # AudioEngine class
│   │   └── golden_math.py       # φ functions (fade, ease, phases)
│   │
│   ├── programs/                # ✅ DONE - Sequence management
│   │   ├── __init__.py
│   │   ├── step.py              # Step, FrequencyConfig, PositionConfig
│   │   ├── program.py           # Program class + factories
│   │   └── presets/
│   │       └── chakra_sunrise.json
│   │
│   ├── ui/                      # 🔄 IN PROGRESS
│   │   ├── __init__.py
│   │   ├── main_window.py       # Main app window
│   │   ├── tabs/
│   │   │   ├── __init__.py
│   │   │   ├── binaural_tab.py
│   │   │   ├── spectral_tab.py
│   │   │   ├── molecular_tab.py
│   │   │   ├── harmonic_tab.py
│   │   │   └── vibroacoustic_tab.py
│   │   └── widgets/
│   │       ├── __init__.py
│   │       ├── frequency_slider.py
│   │       ├── phase_dial.py
│   │       ├── body_position.py
│   │       └── timeline_view.py
│   │
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   └── file_io.py
│   │
│   ├── golden_constants.py      # Keep (shared constants)
│   ├── soundboard_panning.py    # Keep (physics model)
│   ├── spectral_sound.py        # Keep (atomic sonification)
│   └── molecular_sound.py       # Keep (molecular sonification)
│
├── tests/                       # ✅ DONE - Test files
│   ├── test_chakra_sunrise.py
│   ├── test_click_debug.py
│   └── test_precision.py
│
└── docs/
    └── PHASE_ROTATION_MODES.md
```

---

## 🎯 Class Locations in Monolith

Reference for extraction (line numbers in `golden_studio.py`):

| Class | Lines | Target File |
|-------|-------|-------------|
| `AudioEngine` | 78-505 | ✅ `core/audio_engine.py` |
| `BinauralTab` | 506-1133 | `ui/tabs/binaural_tab.py` |
| `SpectralTab` | 1134-1429 | `ui/tabs/spectral_tab.py` |
| `MolecularTab` | 1430-1927 | `ui/tabs/molecular_tab.py` |
| `HarmonicTreeTab` | 1928-2896 | `ui/tabs/harmonic_tab.py` |
| `VibroacousticTab` | 2897-3884 | `ui/tabs/vibroacoustic_tab.py` |
| `GoldenSoundStudio` | 3885-4010 | `ui/main_window.py` |

---

## 📝 How to Extract a Tab

### Step 1: Create the file
```python
# ui/tabs/binaural_tab.py
"""
Binaural Beats Tab
"""
import tkinter as tk
from tkinter import ttk
# ... imports

class BinauralTab:
    # Copy from golden_studio.py lines 506-1133
    pass
```

### Step 2: Update imports
- Replace `from golden_constants import ...` with relative imports
- Import `AudioEngine` from `core.audio_engine`

### Step 3: Update `ui/tabs/__init__.py`
```python
from .binaural_tab import BinauralTab
```

### Step 4: Test independently
```python
python -c "from ui.tabs.binaural_tab import BinauralTab; print('OK')"
```

---

## 🔧 Key Components

### AudioEngine (core/audio_engine.py)
- Callback-based PyAudio streaming
- Two modes: `binaural` and `spectral`
- Real-time parameter updates (no glitches)
- Per-sample smooth interpolation

### Program/Step (programs/)
- `Step`: Single phase with frequencies, positions, fades
- `Program`: Sequence of steps with JSON serialization
- `create_chakra_sunrise()`: Factory for Chakra journey
- Body positions: HEAD, SOLAR_PLEXUS, SACRAL, FEET

### Golden Math (core/golden_math.py)
- `golden_fade(t)`: φ-based S-curve for smooth transitions
- `golden_ease(t)`: Easing functions
- `golden_phase_boundaries(n)`: Symmetric 1:φ:φ:φ:1 divisions
- `fibonacci_harmonics()`: Fibonacci frequency ratios

---

## ⚠️ Files to Delete (after migration)

Legacy GUI files (superseded by golden_studio.py):
- `gui_app.py` (928 lines)
- `gui_app_v2.py` (1234 lines)
- `gui_phase_angle.py` (1115 lines)
- `gui_professional.py` (1179 lines)
- `gui_sacred.py` (1096 lines)
- `gui_spectral.py` (692 lines)
- `gui_true_silence.py` (1125 lines)
- `gui_v3_smooth.py` (724 lines)

**Total: ~8000 lines of legacy code to remove**

---

## 🚀 Running the App

Currently still uses the monolith:
```bash
cd binaural_golden/src
python golden_studio.py
```

After refactoring:
```bash
cd binaural_golden/src
python main.py
```

---

## 📋 Next Steps (Priority Order)

1. **Extract VibroacousticTab** (most complex, includes Chakra Sunrise)
2. **Integrate Program system** with VibroacousticTab
3. **Extract remaining tabs** (simpler)
4. **Create main_window.py** as new entry point
5. **Delete legacy files**
6. **Add Program Editor UI** (create/edit programs visually)

---

## 🧪 Testing

Run tests from project root:
```bash
cd binaural_golden/tests
python test_chakra_sunrise.py
```

---

## 📚 Key Concepts

### Golden Ratio (φ)
- φ = 1.618033988749895
- φ⁻¹ = 0.618033988749895
- Used for: fade curves, phase timings, harmonic relationships

### Vibroacoustic Board
- 1950mm spruce board with longitudinal fibers
- Exciters at HEAD (0mm) and FEET (1950mm)
- Pan mapping: position_mm / 975 - 1 = pan (-1 to +1)

### Chakra Sunrise Journey
- 3 frequencies: Root, Perfect 4th (4/3), Octave (2x)
- 5 phases with golden proportions (1:φ:φ:φ:1)
- Convergence at SOLAR_PLEXUS (600mm)
