# Rememberance - Raspberry Pi Modular Architecture

## Overview

Modular architecture for deploying **Rememberance** vibroacoustic therapy system on Raspberry Pi with JAB4 audio processor.

This branch transforms the monolithic desktop Tkinter app into:
- ✅ **Backend-agnostic audio engine** (PyAudio/ALSA/PipeWire)
- ✅ **Standalone therapy modules** (EMDR, vibroacoustic)
- ✅ **Mobile web interface** (Flask REST API + responsive HTML)
- ✅ **Headless operation** (systemd autostart, no display needed)
- ✅ **Production-ready deployment** (automated installation scripts)

---

## Architecture

```
binaural_golden/
├── core/
│   ├── audio_backend.py     # Backend abstraction (PyAudio/ALSA/Dummy)
│   ├── audio_engine.py      # Original real-time engine (legacy)
│   └── golden_math.py       # φ constants and sacred geometry
│
├── modules/
│   ├── emdr/
│   │   └── __init__.py      # Bilateral audio generator + journeys
│   └── vibroacoustic/
│       └── __init__.py      # Physical panning + golden harmonics
│
├── interfaces/
│   └── web/
│       ├── app.py           # Flask REST API server
│       └── templates/
│           └── index.html   # Mobile web interface
│
├── firmware/
│   └── pi_setup/
│       └── install.sh       # Automated Pi installation
│
├── services/
│   └── systemd/
│       └── rememberance.service  # Autostart service
│
└── hardware/
    ├── JAB4_DOCUMENTATION.md     # Complete hardware guide
    └── schematics/                # (future: wiring diagrams)
```

---

## Quick Start

### Option 1: Raspberry Pi Deployment

```bash
# 1. Clone repository on Pi
git clone https://github.com/yourusername/Rememberance.git
cd Rememberance/binaural_golden

# 2. Run automated installer
sudo chmod +x firmware/pi_setup/install.sh
sudo firmware/pi_setup/install.sh

# 3. Access web interface
# Open browser: http://rememberance.local:5000
```

### Option 2: Desktop Testing (macOS/Linux)

```bash
# 1. Install dependencies
pip install numpy flask flask-cors pyalsaaudio pyaudio

# 2. Set backend to PyAudio
export AUDIO_BACKEND=pyaudio

# 3. Start server
python interfaces/web/app.py

# 4. Open browser
open http://localhost:5000
```

---

## Features

### EMDR Bilateral Stimulation
- **Modes**: Standard, Golden Phase (φ), Breathing Sync, Theta Pulse
- **Frequencies**: Solfeggio scale (396-963Hz), 432Hz, brainwave entrainment
- **Speed**: 0.5-2.0 Hz (clinically validated range)
- **Journeys**: Pre-programmed trauma annealing sequences
  - 🌅 Gentle Release (5 min)
  - 🔥 Deep Processing (10 min)
  - 🧠 Hemispheric Sync (8 min)
  - 🌀 Golden Spiral (12 min)

### Vibroacoustic Therapy
- **Physical Modeling**: ITD/ILD panning on spruce soundboard
- **Propagation Delays**: Realistic 5500 m/s velocity simulation
- **Golden Harmonics**: φ-based frequency distribution
- **Dynamic Panning**: Smooth motion along body axis

### Web Interface
- **Mobile-First Design**: Touch-optimized responsive UI
- **Real-Time Control**: Update parameters during session
- **Journey Tracking**: Progress bars and phase indicators
- **Device Selection**: Choose audio output devices

---

## Hardware Setup

### Minimal Configuration (€125)
- **JAB4 WONDOM AA-JA33285**: 4-channel DSP amplifier (€44)
- **4× Dayton DAEX25**: 25mm exciters (€48)
- **3× Alpha RV24AF-10**: Control potentiometers (€8)
- **Power Supply**: 12V 5A adapter (€10)
- **Enclosure**: 150×100×40mm project box (€8)
- **Misc**: Wires, connectors (€8)

### Advanced Configuration (+€65)
- Add **Raspberry Pi 5 (4GB)** for headless operation
- Optional: **HiFiBerry DAC+ ADC Pro** for audiophile quality

See [hardware/JAB4_DOCUMENTATION.md](hardware/JAB4_DOCUMENTATION.md) for complete wiring diagrams, BOM, and calibration procedures.

---

## API Reference

### Status Endpoint
```http
GET /api/status
```

Response:
```json
{
  "mode": "emdr",
  "backend": "ALSABackend",
  "playing": true,
  "emdr": {
    "speed": 1.0,
    "freq": 432,
    "mode": "golden_phase",
    "amplitude": 0.5
  }
}
```

### Start EMDR Session
```http
POST /api/emdr/start
Content-Type: application/json

{
  "speed": 1.0,
  "freq": 432,
  "mode": "golden_phase",
  "amplitude": 0.5
}
```

Or start a journey:
```json
{
  "journey": "gentle_release"
}
```

### Stop EMDR
```http
POST /api/emdr/stop
```

### Update Parameters (Real-Time)
```http
POST /api/emdr/params
Content-Type: application/json

{
  "speed": 1.5,
  "freq": 528
}
```

### Get Presets
```http
GET /api/emdr/presets
```

Response includes all frequency presets and journey programs.

### Start Vibroacoustic
```http
POST /api/vibro/start
Content-Type: application/json

{
  "freq": 432,
  "modulation": 0.1
}
```

### List Audio Devices
```http
GET /api/devices
```

---

## Development

### Module Structure

#### `core/audio_backend.py`
Backend abstraction layer:
- **PyAudioBackend**: Desktop/testing
- **ALSABackend**: Raspberry Pi direct hardware
- **DummyBackend**: Silent testing
- **Auto-detection**: Chooses appropriate backend

Factory function:
```python
from core.audio_backend import create_audio_backend

backend = create_audio_backend(
    backend_type="alsa",  # or None for auto-detect
    sample_rate=48000,
    channels=2,
    buffer_size=2048
)
```

#### `modules/emdr/__init__.py`
Standalone EMDR generator:
```python
from modules.emdr import EMDRGenerator, BilateralMode

gen = EMDRGenerator(sample_rate=48000)
gen.set_parameters(speed=1.0, freq=432, mode=BilateralMode.GOLDEN_PHASE)
gen.start()

# Audio callback
def callback(num_frames):
    return gen.generate_frame(num_frames)

backend.start(callback)
```

Journey programs:
```python
from modules.emdr import EMDRJourney, ANNEALING_PROGRAMS

journey = EMDRJourney(gen, ANNEALING_PROGRAMS["gentle_release"])
journey.start()

# Update periodically
while journey.running:
    journey.update()
    progress = journey.get_progress()
    print(f"Phase: {progress['phase_name']}")
```

#### `modules/vibroacoustic/__init__.py`
Physical panning engine:
```python
from modules.vibroacoustic import VibroacousticPanner

panner = VibroacousticPanner(
    sample_rate=48000,
    num_channels=2,
    board_length_mm=1950,
    velocity_ms=5500
)

# Pan mono audio to stereo with ITD/ILD
head_audio, feet_audio = panner.pan_2ch(mono_audio, position=0.5)
```

Golden harmonics:
```python
from modules.vibroacoustic import GoldenFrequencyPanner

golden = GoldenFrequencyPanner(panner, base_freq=432)
field = golden.generate_harmonic_field(
    num_harmonics=8,
    duration_sec=10
)
```

---

## Testing

### Test Audio Backend
```bash
cd core
python audio_backend.py
```

Plays 2-second 440Hz sine wave through detected backend.

### Test EMDR Module
```bash
cd modules/emdr
python __init__.py
```

Generates bilateral audio and simulates journey progress.

### Test Vibroacoustic Module
```bash
cd modules/vibroacoustic
python __init__.py
```

Tests static panning, golden harmonics, and dynamic programs.

### Test Web Server (Desktop)
```bash
export AUDIO_BACKEND=dummy  # Silent mode
python interfaces/web/app.py
```

Open http://localhost:5000

---

## Deployment

### Systemd Service Management

```bash
# Start service
sudo systemctl start rememberance

# Stop service
sudo systemctl stop rememberance

# Restart
sudo systemctl restart rememberance

# View status
sudo systemctl status rememberance

# View logs (real-time)
sudo journalctl -u rememberance -f

# Enable autostart on boot
sudo systemctl enable rememberance

# Disable autostart
sudo systemctl disable rememberance
```

### Network Access

The service binds to `0.0.0.0:5000`, accessible from any device on the local network:

```
http://raspberry-pi-ip:5000
http://rememberance.local:5000  (if Avahi/mDNS enabled)
```

### Security Considerations

1. **Firewall**: Port 5000 is opened by installer
2. **HTTPS**: Add reverse proxy (nginx) for SSL if exposed to internet
3. **Authentication**: Consider adding Flask-Login for multi-user
4. **Local Network Only**: Do NOT expose to public internet without proper security

---

## Troubleshooting

### No Audio Output
1. Check backend: `cat /proc/asound/cards`
2. Test ALSA: `speaker-test -c 2 -t sine`
3. Verify device: `aplay -l`
4. Check logs: `journalctl -u rememberance -f`

### Web Interface Not Accessible
1. Check service: `systemctl status rememberance`
2. Test locally: `curl http://localhost:5000/api/status`
3. Check firewall: `sudo ufw status`
4. Verify IP: `hostname -I`

### High CPU Usage
1. Increase buffer size in backend (2048 → 4096)
2. Reduce update frequency in web UI
3. Check for background processes

### Audio Glitches
1. Set nice priority: `Nice=-10` in systemd service
2. Use ALSA hardware device directly (not PulseAudio)
3. Disable WiFi power management: `iwconfig wlan0 power off`

---

## Roadmap

### v2.0 (Current Branch)
- ✅ Modular architecture
- ✅ Backend abstraction
- ✅ Web interface
- ✅ Automated deployment

### v2.1 (Planned)
- [ ] WebSocket for real-time updates
- [ ] Session recording/playback
- [ ] User profiles and favorites
- [ ] Advanced visualizations (waveform, spectrum)

### v2.2 (Future)
- [ ] Multi-zone support (multiple soundboards)
- [ ] MIDI controller integration
- [ ] Cloud sync for therapy sessions
- [ ] Machine learning for personalized programs

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

See [LICENSE](../LICENSE) file.

---

## Acknowledgments

- **EMDR Protocol**: Francine Shapiro's research on bilateral stimulation
- **Golden Ratio**: Ancient sacred geometry principles
- **Solfeggio Frequencies**: Healing frequencies from Gregorian chants
- **JAB4 Platform**: WONDOM audio engineering
- **Raspberry Pi Foundation**: Accessible computing for all

---

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Email: support@rememberance.therapy
- Community forum: (coming soon)

---

_"Integrate hemispheres, anneal traumas, restore order"_

🌊 Rememberance - Vibroacoustic Therapy for the Digital Age
