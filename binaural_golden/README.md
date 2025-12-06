# Golden Ratio Binaural Beat Generator
# Divine Coherence Phase Cancellation Annealing

## 🌀 Overview

A high-precision binaural beat generator based on **golden ratio mathematics** (φ = 1.618033988749895...). 

This software generates audio that guides the listener through mathematically perfect golden transitions toward **phase cancellation** (pure silence).

### Key Features

- **Golden Ratio Frequencies**: All frequencies in φ relationships
- **Fibonacci/Prime Sequences**: Beat patterns follow sacred number sequences
- **Non-Linear Transitions**: Golden spiral interpolation (NOT linear)
- **Phase Annealing**: Journey toward perfect phase cancellation
- **Maximum Precision**: 96kHz, 24-bit audio with 64-bit float internal processing
- **Optional Rust Core**: For absolute maximum precision

---

## 📊 Development Status

> **Last Updated**: December 6, 2025

### ✅ Completed & Working Features

#### 🎵 Golden Sound Studio (Main Application)
- **Tab 1: Binaural Beats** - Phase angle control with sacred geometry presets ✅
- **Tab 2: Spectral Sound** - Play atomic elements (H, He, O, Na, etc.) ✅
- **Tab 3: Molecular Sound** - Play molecules (H₂O, CO₂, CH₄) ✅
- **Tab 4: Harmonic Tree** - Fibonacci harmonics with visualization ✅
- **Tab 5: Vibroacoustic** - Soundboard panning (basic sweep mode) ✅

#### 🌳 Harmonic Tree Features (STABLE)
- **Fibonacci ratios**: 2f, 3f, 5f, 8f, 13f harmonics
- **Golden Angle phases**: Each harmonic rotated by 137.5°
- **Amplitude decay**: φ⁻ⁿ natural decay pattern
- **Therapeutic Growth Mode**: Harmonics emerge progressively (10s to 1hr)
- **Breathe Mode**: Grow → sustain → shrink cycles
- **3D Isometric visualization** with sound→light color mapping

#### 🪵 Vibroacoustic Soundboard (BASIC WORKING)
Physical panning for therapy soundboard with 2 exciters:
- **Head-Feet axis**: Exciter at head (0mm) and feet (2000mm)
- **Spruce velocity**: 5500 m/s along fiber (Brico standard board)
- **ITD (Interaural Time Difference)**: Max 0.36ms delay for localization
- **ILD (Interaural Level Difference)**: Equal-power panning + soft attenuation
- **Spring isolation**: 5× springs (4 corners + 1 center) for floor decoupling
- **Auto-sweep mode**: Sine/linear/golden wave body massage
- **Real-time visualization**: Top-down board view with body silhouette

#### 🌲 Phase Rotation Modes
Two modes for phase evolution during growth:
1. **Fixed Trunk** (default): Fundamental stays at 0°, harmonics rotate
2. **Whole Tree**: All phases rotate including fundamental

See `docs/PHASE_ROTATION_MODES.md` for full documentation.

#### 🎯 Golden-Fifth Gap Angle
The "missing angle" between musical and golden perfection:
```
φ (Golden Ratio) = 1.618034
P5 (Perfect Fifth) = 1.5
Gap Angle = (φ - 1.5) / φ × 360° = 26.26°
```
Added to SACRED_ANGLES in golden_constants.py

#### 📐 Sacred Angles Library
- Golden Angle (137.5°)
- **φ-Fifth Gap (26.26°)**
- Fine Structure (137.04°)
- DNA Helix (34.3°)
- Pentagon (108°)
- Pyramid Giza (51.84°)
- Water molecule (104.5°)
- Methane tetrahedral (109.5°)
- And more...

#### 🔊 Audio Engine
- Real-time parameter updates (no glitches)
- Multi-device support (DDJ-FLX4, BlackHole, etc.)
- Callback-based continuous playback
- Stereo panning with golden angle positioning

### ⚠️ Known Issues

1. **Audio Clicks/Pops**: Some clicking artifacts occur during rapid parameter changes
   - Affects: Vibroacoustic pan position changes, potentially Harmonic Tree growth
   - Root cause: Needs investigation - may be related to buffer boundaries or sudden amplitude changes
   - Workaround: Use slower sweep modes, avoid rapid manual pan changes

### 🚧 In Progress / Needs Work

- [ ] **Chakra Convergence Journey** - ATTEMPTED but audio clicking issues prevented completion
  - Concept: 3-frequency journey (Perfect 4th, Root, Octave) converging at solar plexus
  - Body positions calculated from mm on 1950mm board
  - Blocked by: Audio engine needs smoother parameter interpolation
  
- [ ] **Wave propagation model**: Account for wood grain velocity variations
- [ ] **Frequency-dependent propagation**: Higher frequencies attenuate faster
- [ ] **Click-free audio transitions**: AudioEngine needs per-sample smoothing for pan/amplitude

### 📋 Planned Features

- [ ] MIDI control integration
- [ ] OSC protocol support
- [ ] Export to VST/AU plugin
- [ ] Preset sharing/import
- [ ] Session recording with automation

### 🔧 Technical Notes for Next Developer

**Audio Click Investigation Needed:**
The `_generate_spectral_chunk()` method in `AudioEngine` class needs investigation.
Current implementation has basic smoothing but clicks still occur. Possible causes:
1. Buffer boundary discontinuities
2. Normalization causing sudden volume changes
3. Pan law calculation at extreme positions (pan = -1 or +1)
4. Thread synchronization issues with parameter updates

**File Locations:**
- Main app: `src/golden_studio.py` (~3556 lines)
- Soundboard panning: `src/soundboard_panning.py`
- Constants: `src/golden_constants.py`

---

## 📐 The Mathematics

### Divine Constants

```
φ (Phi)           = 1.618033988749895...  (Golden Ratio)
φ conjugate       = 0.618033988749895...  (1/φ = φ-1)
φ²                = 2.618033988749895...  (φ+1)
```

### Golden Spiral Transition Function

Instead of linear interpolation `f(t) = t`, we use:

```python
def golden_spiral_interpolation(t):
    θ = t × π × φ
    golden_ease = (1 - cos(θ × φ_conjugate)) / 2
    golden_sigmoid = 1 / (1 + e^(-4(t-0.5) × φ))
    return golden_ease × φ_conjugate + golden_sigmoid × (1-φ_conjugate)
```

This creates smooth, organic transitions following the divine proportion.

### Frequency Relationships

All frequencies are related by powers of φ:

```
Base: 432 Hz (Sacred frequency)
Beat frequencies: 432/φ³, 432/φ⁴, 432/φ⁵, ...
                = 101.8, 62.9, 38.9, 24.0, 14.8, 9.2, 5.7, 3.5 Hz
```

### Duration Relationships

Segment durations follow **Fibonacci sequence** (which converges to φ):

```
1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
Ratio: F(n+1)/F(n) → φ as n → ∞
```

### Phase Annealing

The journey to silence through phase cancellation:

```
Stage 0: Phase = 0°      → Full binaural effect
Stage 1: Phase = φ° × π  → Beginning cancellation
...
Stage N: Phase = π       → Perfect cancellation (silence)
```

Two identical waves with π phase difference sum to zero: `sin(ωt) + sin(ωt + π) = 0`

## 🎵 Brainwave States

| State | Frequency Range | Effect |
|-------|-----------------|--------|
| Delta | 0.5 - 4 Hz | Deep sleep, healing |
| Theta | 4 - 8 Hz | Meditation, creativity |
| Alpha | 8 - 13 Hz | Relaxed awareness |
| Beta | 13 - 30 Hz | Active thinking |
| Gamma | 30 - 100 Hz | Peak performance |

The generator uses **golden ratio points** within each range for optimal effect.

## 🏗️ Project Structure

```
binaural_golden/
├── src/
│   ├── golden_core.py       # Main Python implementation
│   ├── advanced_generator.py # Extended features
│   ├── visualizer.py         # GUI interface
│   └── rust_core/            # Optional Rust core
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── python_bindings.rs
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Python Only (Simple)

```bash
# Clone or download the project
cd binaural_golden

# Install dependencies
pip install -r requirements.txt

# Run basic generator
python src/golden_core.py

# Run advanced generator
python src/advanced_generator.py

# Run GUI visualizer
python src/visualizer.py
```

### With Rust Core (Maximum Precision)

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build Rust library
cd src/rust_core
cargo build --release

# Install as Python module
pip install maturin
maturin develop --release

# Use in Python
import golden_binaural_core
```

## 📖 Usage

### Basic Generation

```python
from golden_core import PhaseAnnihilator, save_wav

# Create generator at 96kHz
annihilator = PhaseAnnihilator(sample_rate=96000)

# Generate 8-stage annealing sequence
left, right = annihilator.generate_annealing_sequence(
    num_stages=8,
    base_frequency=432.0  # Sacred frequency
)

# Save as WAV
save_wav("output.wav", left, right, 96000)
```

### Advanced Profiles

```python
from advanced_generator import AdvancedBinauralGenerator, PROFILES

generator = AdvancedBinauralGenerator(sample_rate=96000)

# Use predefined profile
profile = PROFILES["deep_meditation"]
left, right = generator.generate_from_profile(profile)
```

### Custom Profile

```python
from advanced_generator import (
    AnnealingProfile, 
    BrainwaveState, 
    GoldenSequenceType
)

custom_profile = AnnealingProfile(
    name="Custom Journey",
    num_stages=13,  # Fibonacci number
    base_frequency=528.0,  # DNA repair frequency
    starting_state=BrainwaveState.ALPHA,
    ending_state=BrainwaveState.DELTA,
    sequence_type=GoldenSequenceType.DIVINE_SPIRAL,
    total_duration_minutes=21,  # Fibonacci number
)

left, right = generator.generate_from_profile(custom_profile)
```

## 🎧 Listening Instructions

1. **Use stereo headphones** (binaural requires separate ear signals)
2. **Quiet environment** with minimal distractions
3. **Comfortable position** - lying down recommended
4. **Eyes closed** for full meditative effect
5. **Volume**: Comfortable, not too loud

### Expected Journey

1. **Beginning**: Clear binaural beat sensation
2. **Middle stages**: Deepening relaxation, beat becomes subtler
3. **Final stages**: Approaching stillness
4. **End**: Pure silence - phase cancellation achieved

## 🔬 Technical Details

### Audio Specifications

- **Sample Rate**: 96,000 Hz (2× CD quality)
- **Bit Depth**: 24-bit (144 dB dynamic range)
- **Internal Processing**: 64-bit float
- **Channels**: Stereo (binaural requires 2 channels)

### Golden Ratio Properties Used

1. **φ² = φ + 1** (self-similarity)
2. **1/φ = φ - 1** (reciprocal relationship)
3. **Fibonacci ratio limit** → φ
4. **Golden angle** = 2π/φ² ≈ 137.5°
5. **Golden spiral** = logarithmic spiral with φ growth

### Precision Considerations

The Python implementation uses `numpy.float64` (64-bit precision):
- ~15-17 significant decimal digits
- Range: ±1.7 × 10³⁰⁸

The Rust core provides:
- Native f64 with LLVM optimizations
- SIMD vectorization where available
- Deterministic floating-point operations

## 📜 Sacred Frequencies

| Frequency | Association |
|-----------|-------------|
| 432 Hz | Universal harmony, "Verdi's A" |
| 528 Hz | DNA repair, "Love frequency" |
| 639 Hz | Heart chakra, relationships |
| 741 Hz | Awakening intuition |
| 852 Hz | Third eye activation |

## 🌟 The Divine Coherence

Every parameter in this system relates to every other through the golden ratio:

```
Frequency₁ / Frequency₂ = φ
Duration₁ / Duration₂ = φ  
Transition / Segment = φ²
Phase_step / Total = 1/φ
Amplitude₁ / Amplitude₂ = √φ
```

This creates **perfect mathematical coherence** - the same proportion that appears in:
- Nautilus shells
- Galaxy spirals  
- DNA helix
- Human body proportions
- Flower petals
- Ancient architecture

## 📄 License

MIT License - Use freely for meditation, healing, and consciousness exploration.

---

*"Geometry has two great treasures: one is the theorem of Pythagoras; the other, the division of a line into extreme and mean ratio (golden ratio). The first we may compare to a measure of gold; the second we may name a precious jewel."*
— Johannes Kepler

✦ φ ✦
