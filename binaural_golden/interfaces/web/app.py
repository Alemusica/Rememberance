"""
Rememberance Web Interface
===========================

Flask REST API for Raspberry Pi headless operation.
Mobile-first HTML interface for controlling EMDR and vibroacoustic therapy.

Endpoints:
- GET  /              - Mobile web interface
- GET  /api/status    - System status
- POST /api/emdr/start - Start EMDR session
- POST /api/emdr/stop  - Stop EMDR
- POST /api/emdr/params - Update EMDR parameters
- POST /api/vibro/start - Start vibroacoustic
- POST /api/vibro/stop  - Stop vibroacoustic
- GET  /api/devices    - List audio devices
- POST /api/device     - Set audio device
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os

# Add binaural_golden directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # interfaces/web/
binaural_golden_dir = os.path.dirname(os.path.dirname(current_dir))  # binaural_golden/
sys.path.insert(0, binaural_golden_dir)

from core.audio_backend import create_audio_backend, AudioBackend
from modules.emdr import (
    EMDRGenerator, EMDRJourney, BilateralMode, 
    ANNEALING_PROGRAMS, EMDR_FREQUENCIES
)
from modules.vibroacoustic import (
    VibroacousticPanner, VibroacousticProgram
)
from modules.harmonic_tree import (
    HarmonicTreeGenerator, HarmonicMode, AmplitudeDecay
)
from modules.spectral import (
    SpectralSoundGenerator, PhaseMode, ELEMENTS
)


app = Flask(__name__)
CORS(app)  # Enable CORS for mobile browsers

# Global state
audio_backend: AudioBackend = None
emdr_generator: EMDRGenerator = None
emdr_journey: EMDRJourney = None
vibro_program: VibroacousticProgram = None
vibro_panner: VibroacousticPanner = None
harmonic_tree: HarmonicTreeGenerator = None
spectral_sound: SpectralSoundGenerator = None

current_mode = "idle"  # "idle", "emdr", "vibroacoustic", "harmonic", "spectral"


def init_audio():
    """Initialize audio backend"""
    global audio_backend, emdr_generator, vibro_panner, harmonic_tree, spectral_sound
    
    # Create backend (auto-detect Pi vs desktop)
    audio_backend = create_audio_backend(
        sample_rate=48000,
        channels=2,
        buffer_size=2048
    )
    
    # Create generators
    emdr_generator = EMDRGenerator(sample_rate=48000)
    vibro_panner = VibroacousticPanner(sample_rate=48000, num_channels=2)
    harmonic_tree = HarmonicTreeGenerator(sample_rate=48000)
    spectral_sound = SpectralSoundGenerator(sample_rate=48000)
    
    print("✓ Audio system initialized")


def emdr_callback(num_frames: int):
    """Audio callback for EMDR mode"""
    return emdr_generator.generate_frame(num_frames)


def vibro_callback(num_frames: int):
    """Audio callback for vibroacoustic mode"""
    return vibro_program.generate_frame(num_frames)


def harmonic_callback(num_frames: int):
    """Audio callback for harmonic tree mode"""
    return harmonic_tree.generate_frame(num_frames)


def spectral_callback(num_frames: int):
    """Audio callback for spectral sound mode"""
    return spectral_sound.generate_frame(num_frames)
    return emdr_generator.generate_frame(num_frames)


def vibro_callback(num_frames: int):
    """Audio callback for vibroacoustic mode"""
    return vibro_program.generate_frame(num_frames)


# ══════════════════════════════════════════════════════════════════════════════
# WEB ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Mobile web interface"""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Get system status"""
    status = {
        "mode": current_mode,
        "backend": type(audio_backend).__name__ if audio_backend else "none",
        "playing": audio_backend.playing if audio_backend else False,
    }
    
    # Add mode-specific info
    if current_mode == "emdr":
        if emdr_journey and emdr_journey.running:
            progress = emdr_journey.get_progress()
            status["journey"] = progress
        else:
            status["emdr"] = {
                "speed": emdr_generator.bilateral_speed,
                "freq": emdr_generator.carrier_freq,
                "mode": emdr_generator.mode.value,
                "amplitude": emdr_generator.amplitude,
            }
    
    elif current_mode == "vibroacoustic":
        status["vibro"] = {
            "freq": vibro_program.frequency,
            "modulation": vibro_program.modulation_freq,
        }
    
    return jsonify(status)


@app.route('/api/devices')
def api_devices():
    """List audio output devices"""
    if not audio_backend:
        return jsonify({"error": "Audio not initialized"}), 500
    
    devices = audio_backend.list_devices()
    return jsonify({"devices": devices})


@app.route('/api/device', methods=['POST'])
def api_set_device():
    """Set audio output device"""
    if not audio_backend:
        return jsonify({"error": "Audio not initialized"}), 500
    
    data = request.json
    device_index = data.get('device_index', 0)
    
    try:
        audio_backend.set_device(device_index)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# EMDR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/emdr/start', methods=['POST'])
def api_emdr_start():
    """Start EMDR session"""
    global current_mode, emdr_journey
    
    if current_mode != "idle":
        return jsonify({"error": f"Already in {current_mode} mode"}), 400
    
    data = request.json
    journey_name = data.get('journey', None)
    
    try:
        if journey_name and journey_name in ANNEALING_PROGRAMS:
            # Start journey program
            program = ANNEALING_PROGRAMS[journey_name]
            emdr_journey = EMDRJourney(emdr_generator, program)
            emdr_journey.start()
            audio_backend.start(emdr_callback)
            current_mode = "emdr"
            return jsonify({
                "success": True, 
                "mode": "journey",
                "journey": program["name"]
            })
        else:
            # Manual EMDR
            speed = data.get('speed', 1.0)
            freq = data.get('freq', 432.0)
            mode = data.get('mode', 'standard')
            amplitude = data.get('amplitude', 0.5)
            
            # Set parameters
            emdr_generator.set_parameters(
                speed=speed,
                freq=freq,
                mode=BilateralMode[mode.upper()] if isinstance(mode, str) else mode,
                amplitude=amplitude
            )
            
            emdr_generator.start()
            audio_backend.start(emdr_callback)
            current_mode = "emdr"
            
            return jsonify({
                "success": True,
                "mode": "manual",
                "params": {
                    "speed": speed,
                    "freq": freq,
                    "mode": mode,
                    "amplitude": amplitude
                }
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/emdr/stop', methods=['POST'])
def api_emdr_stop():
    """Stop EMDR session"""
    global current_mode, emdr_journey
    
    if current_mode != "emdr":
        return jsonify({"error": "Not in EMDR mode"}), 400
    
    try:
        if emdr_journey:
            emdr_journey.stop()
            emdr_journey = None
        else:
            emdr_generator.stop()
        
        audio_backend.stop()
        current_mode = "idle"
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/emdr/params', methods=['POST'])
def api_emdr_params():
    """Update EMDR parameters in real-time"""
    if current_mode != "emdr":
        return jsonify({"error": "Not in EMDR mode"}), 400
    
    data = request.json
    
    try:
        # Update parameters
        if 'speed' in data:
            emdr_generator.set_parameters(speed=data['speed'])
        if 'freq' in data:
            emdr_generator.set_parameters(freq=data['freq'])
        if 'mode' in data:
            mode = BilateralMode[data['mode'].upper()]
            emdr_generator.set_parameters(mode=mode)
        if 'amplitude' in data:
            emdr_generator.set_parameters(amplitude=data['amplitude'])
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/emdr/presets')
def api_emdr_presets():
    """Get EMDR frequency presets and journey programs"""
    return jsonify({
        "frequencies": EMDR_FREQUENCIES,
        "journeys": {
            name: {
                "name": prog["name"],
                "duration_min": prog["duration_min"],
                "description": prog["description"]
            }
            for name, prog in ANNEALING_PROGRAMS.items()
        }
    })


# ══════════════════════════════════════════════════════════════════════════════
# VIBROACOUSTIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/vibro/start', methods=['POST'])
def api_vibro_start():
    """Start vibroacoustic program"""
    global current_mode, vibro_program
    
    if current_mode != "idle":
        return jsonify({"error": f"Already in {current_mode} mode"}), 400
    
    data = request.json
    freq = data.get('freq', 432.0)
    modulation = data.get('modulation', 0.1)
    
    try:
        vibro_program = VibroacousticProgram(
            panner=vibro_panner,
            frequency=freq,
            modulation_freq=modulation
        )
        
        audio_backend.start(vibro_callback)
        current_mode = "vibroacoustic"
        
        return jsonify({
            "success": True,
            "freq": freq,
            "modulation": modulation
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vibro/stop', methods=['POST'])
def api_vibro_stop():
    """Stop vibroacoustic program"""
    global current_mode, vibro_program
    
    if current_mode != "vibroacoustic":
        return jsonify({"error": "Not in vibroacoustic mode"}), 400
    
    try:
        audio_backend.stop()
        vibro_program = None
        current_mode = "idle"
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# HARMONIC TREE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/harmonic/start', methods=['POST'])
def api_harmonic_start():
    """Start harmonic tree"""
    global current_mode
    
    if current_mode != "idle":
        return jsonify({"error": f"Already in {current_mode} mode"}), 400
    
    data = request.json
    fundamental = data.get('fundamental', 432.0)
    num_harmonics = data.get('num_harmonics', 5)
    mode = data.get('mode', 'fibonacci')
    amplitude = data.get('amplitude', 0.5)
    growth = data.get('growth_enabled', True)
    growth_duration = data.get('growth_duration', 60)
    
    try:
        harmonic_tree.set_parameters(
            fundamental=fundamental,
            num_harmonics=num_harmonics,
            harmonic_mode=HarmonicMode[mode.upper()],
            amplitude=amplitude
        )
        harmonic_tree.set_growth(growth, growth_duration)
        harmonic_tree.start()
        audio_backend.start(harmonic_callback)
        current_mode = "harmonic"
        
        return jsonify({
            "success": True,
            "fundamental": fundamental,
            "num_harmonics": num_harmonics,
            "mode": mode,
            "growth_enabled": growth
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/harmonic/stop', methods=['POST'])
def api_harmonic_stop():
    """Stop harmonic tree"""
    global current_mode
    
    if current_mode != "harmonic":
        return jsonify({"error": "Not in harmonic mode"}), 400
    
    try:
        harmonic_tree.stop()
        audio_backend.stop()
        current_mode = "idle"
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL SOUND ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/spectral/start', methods=['POST'])
def api_spectral_start():
    """Start spectral sound"""
    global current_mode
    
    if current_mode != "idle":
        return jsonify({"error": f"Already in {current_mode} mode"}), 400
    
    data = request.json
    element = data.get('element', 'hydrogen')
    phase_mode = data.get('phase_mode', 'golden')
    amplitude = data.get('amplitude', 0.5)
    
    try:
        spectral_sound.set_parameters(
            element=element,
            phase_mode=PhaseMode[phase_mode.upper()],
            amplitude=amplitude
        )
        spectral_sound.start()
        audio_backend.start(spectral_callback)
        current_mode = "spectral"
        
        return jsonify({
            "success": True,
            "element": element,
            "phase_mode": phase_mode
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/spectral/stop', methods=['POST'])
def api_spectral_stop():
    """Stop spectral sound"""
    global current_mode
    
    if current_mode != "spectral":
        return jsonify({"error": "Not in spectral mode"}), 400
    
    try:
        spectral_sound.stop()
        audio_backend.stop()
        current_mode = "idle"
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/spectral/elements')
def api_spectral_elements():
    """Get available elements"""
    return jsonify({
        "elements": list(ELEMENTS.keys())
    })


@app.route('/api/test')
def api_test():
    """Test endpoint to verify API is working"""
    return jsonify({
        "status": "ok",
        "modules": {
            "emdr": emdr is not None,
            "vibroacoustic": vibro_panner is not None,
            "harmonic_tree": harmonic_tree is not None,
            "spectral_sound": spectral_sound is not None
        },
        "backend": audio_backend.__class__.__name__,
        "current_mode": current_mode
    })


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Start Flask server"""
    print("╔════════════════════════════════════════════════╗")
    print("║     REMEMBERANCE - Vibroacoustic Therapy      ║")
    print("║           Raspberry Pi Web Interface           ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    # Initialize audio
    init_audio()
    
    # Get IP address for mobile access
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Use port from environment or default to 5001
    port = int(os.environ.get('FLASK_PORT', 5001))
    
    print(f"\n🌐 Web interface starting...")
    print(f"   Local:   http://localhost:{port}")
    print(f"   Network: http://{local_ip}:{port}")
    print(f"\n📱 Open this URL on your phone/tablet\n")
    
    # Start Flask (use 0.0.0.0 to allow network access)
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
