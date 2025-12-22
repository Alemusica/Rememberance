# Changelog - Rememberance

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2024-01-20 - Raspberry Pi Modular Architecture

### Added

#### Core Audio Engine
- **Backend abstraction layer** (`core/audio_backend.py`)
  - PyAudioBackend for desktop/testing
  - ALSABackend for Raspberry Pi direct hardware access
  - DummyBackend for silent testing
  - Auto-detection of platform (Pi vs desktop)
  - Unified callback interface across backends

#### EMDR Module (`modules/emdr/`)
- **Standalone bilateral audio generator**
  - 4 bilateral modes: standard, golden_phase, breathing, theta_pulse
  - Real-time parameter updates (speed, frequency, mode, amplitude)
  - Golden ratio phase relationships (137.5° hemispheric offset)
  - Smooth fade-in/fade-out with φ-based curves

- **Journey system** for trauma annealing
  - 4 preset programs: Gentle Release, Deep Processing, Hemispheric Sync, Golden Spiral
  - Fibonacci-timed phase progressions
  - Automatic phase transitions
  - Progress tracking and status reporting

- **Sacred frequency presets**
  - Solfeggio scale (174-963 Hz)
  - 432 Hz Verdi tuning
  - Brainwave entrainment (gamma, alpha, theta)

#### Vibroacoustic Module (`modules/vibroacoustic/`)
- **Physical panning engine**
  - ITD (Inter-Transducer Delay) simulation
  - ILD (Inter-Transducer Level Difference) modeling
  - Propagation delay calculation (5500 m/s spruce velocity)
  - 2-channel and 4-channel (quad) support

- **Golden frequency distribution**
  - φ-based harmonic spacing
  - Golden angle position distribution
  - Multi-harmonic field generation

- **Dynamic vibroacoustic programs**
  - Smooth panning along body axis
  - Golden ratio modulation timing
  - Real-time frame generation

#### Web Interface (`interfaces/web/`)
- **Flask REST API**
  - EMDR control endpoints (start/stop/params)
  - Vibroacoustic control endpoints
  - Status monitoring with real-time updates
  - Device selection and configuration
  - Preset/journey listing

- **Mobile-first HTML interface**
  - Touch-optimized responsive design
  - Real-time parameter sliders
  - Journey selection and progress tracking
  - Mode switching (EMDR/Vibroacoustic)
  - Dark theme with gradient background

#### Deployment Tools
- **Automated Pi installer** (`firmware/pi_setup/install.sh`)
  - System package updates
  - Python virtual environment setup
  - ALSA configuration for JAB4
  - Avahi mDNS for .local hostname
  - Firewall configuration
  - Systemd service installation

- **Systemd service** (`services/systemd/rememberance.service`)
  - Autostart on boot
  - Automatic restart on failure
  - Real-time audio priority (Nice=-10)
  - Security hardening (NoNewPrivileges, PrivateTmp)
  - Journal logging

#### Hardware Documentation
- **Complete JAB4 guide** (`hardware/JAB4_DOCUMENTATION.md`)
  - Bill of Materials with pricing
  - Wiring diagrams (ASCII art)
  - I2S connection to Raspberry Pi
  - DSP configuration (SigmaStudio)
  - Calibration procedures
  - Troubleshooting guide
  - Safety warnings

#### Documentation
- **Modular architecture README** (`README_PI_MODULAR.md`)
  - Quick start guides (Pi + desktop)
  - API reference with examples
  - Development documentation
  - Module usage examples
  - Troubleshooting section
  - Roadmap for future versions

- **Python requirements** (`requirements_pi.txt`)
  - Minimal dependencies for Pi deployment
  - Version pinning for stability

### Changed
- **Refactored monolithic architecture** into modular components
- **Separated UI from business logic** for headless operation
- **Improved audio callback system** with thread-safe parameter updates
- **Enhanced EMDR implementation** with journey system
- **Optimized for Raspberry Pi** with ALSA backend and systemd integration

### Fixed
- Thread safety issues in audio parameter updates
- Phase accumulation overflow in long-running sessions
- Audio glitches during parameter changes
- Memory leaks in callback loops

### Breaking Changes
- Desktop Tkinter app (`golden_studio.py`) still works but deprecated
- New modular imports required:
  ```python
  from core.audio_backend import create_audio_backend
  from modules.emdr import EMDRGenerator
  from modules.vibroacoustic import VibroacousticPanner
  ```

---

## [1.0.0] - 2024-01-15 - Desktop EMDR Implementation

### Added
- **EMDR bilateral audio tab** in desktop app
  - Manual control (speed, frequency, mode, volume)
  - Bilateral visualization with L/R indicators
  - Sacred frequency presets
  - 4 bilateral modes

- **Tkinter UI integration**
  - Tab 6: EMDR in main notebook
  - Real-time parameter updates
  - Golden theme styling

### Changed
- Updated `golden_studio.py` to include EMDR tab
- Added `ui/emdr_tab.py` with 800+ lines of implementation

---

## [0.9.0] - 2024-01-10 - Vibroacoustic Therapy Foundation

### Added
- **Vibroacoustic soundboard module** (`soundboard_panning.py`)
  - Physical modeling for spruce board
  - ITD/ILD panning calculations
  - Exciter position configuration

- **Hardware planning**
  - BOM for vibroacoustic setup
  - Exciter specifications (Visaton EX 60S)
  - Amplifier selection (Behringer EPQ304)

- **Core audio engine** (`core/audio_engine.py`)
  - Callback-based real-time generation
  - Binaural and spectral modes
  - Phase tracking and waveform generation

### Changed
- Refactored audio generation to use vectorized NumPy
- Improved phase continuity across callbacks

---

## [0.8.0] - 2024-01-05 - Golden Ratio Mathematics

### Added
- **Golden constants module** (`golden_constants.py`)
  - φ (PHI) and φ⁻¹ calculations
  - Golden angle (137.5°)
  - Solfeggio frequency definitions
  - Golden fade curves

- **Sacred geometry utilities**
  - Golden spiral generation
  - Fibonacci sequence helpers
  - Harmonic tree calculations

---

## [0.7.0] - 2023-12-20 - Initial Desktop Application

### Added
- **Desktop Tkinter application** (`golden_studio.py`)
  - Multi-tab interface
  - Real-time audio generation
  - Device selection
  - Parameter controls

- **Binaural beat generation**
  - Frequency and phase controls
  - Waveform selection
  - Volume management

---

_For older versions, see git history._

---

## Versioning Policy

- **Major version** (x.0.0): Breaking changes, architecture overhauls
- **Minor version** (0.x.0): New features, modules, significant improvements
- **Patch version** (0.0.x): Bug fixes, documentation updates, minor tweaks

---

_"Integrate hemispheres, anneal traumas, restore order"_
