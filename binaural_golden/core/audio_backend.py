"""
Backend-agnostic Audio Engine
==============================

Supports multiple audio backends for Raspberry Pi deployment:
- PyAudio (desktop/testing)
- ALSA (Pi direct hardware access)
- PipeWire (modern Pi audio)
- Dummy (testing without hardware)

Choose backend automatically or explicitly via environment:
    AUDIO_BACKEND=alsa python app.py
    AUDIO_BACKEND=pyaudio python app.py
"""

import numpy as np
import threading
import os
from enum import Enum
from typing import Optional, Callable, List
from abc import ABC, abstractmethod


class AudioBackendType(Enum):
    """Supported audio backends"""
    PYAUDIO = "pyaudio"
    ALSA = "alsa"
    PIPEWIRE = "pipewire"
    DUMMY = "dummy"


class AudioBackend(ABC):
    """Abstract base for audio backends"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, buffer_size: int = 2048):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.playing = False
    
    @abstractmethod
    def start(self, callback: Callable[[int], np.ndarray]) -> bool:
        """
        Start audio playback with callback
        
        Args:
            callback: Function that generates audio frames (num_frames) -> np.ndarray[float32]
                     Returns shape: (num_frames, channels)
        
        Returns:
            True if started successfully
        """
        pass
    
    @abstractmethod
    def stop(self):
        """Stop audio playback"""
        pass
    
    @abstractmethod
    def list_devices(self) -> List[dict]:
        """Return list of available output devices"""
        pass
    
    @abstractmethod
    def set_device(self, device_index: int):
        """Set output device"""
        pass


class PyAudioBackend(AudioBackend):
    """PyAudio backend for desktop/testing"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, buffer_size: int = 2048):
        super().__init__(sample_rate, channels, buffer_size)
        
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            self.stream = None
            self.callback_fn = None
            print("✓ PyAudio backend initialized")
        except ImportError:
            raise RuntimeError("PyAudio not available. Install with: pip install pyaudio")
    
    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback wrapper"""
        if self.callback_fn is None:
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), 
                    self.pyaudio.paContinue)
        
        try:
            # Generate audio from callback
            audio_data = self.callback_fn(frame_count)  # Shape: (frame_count, channels)
            
            # Clip to [-1, 1] and convert to bytes
            audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
            return (audio_data.tobytes(), self.pyaudio.paContinue)
        except Exception as e:
            print(f"❌ Audio callback error: {e}")
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), 
                    self.pyaudio.paContinue)
    
    def start(self, callback: Callable[[int], np.ndarray]) -> bool:
        """Start PyAudio stream"""
        if self.playing:
            return True
        
        try:
            self.callback_fn = callback
            self.stream = self.pyaudio.open(
                format=self.pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._pyaudio_callback
            )
            self.playing = True
            print("🔊 PyAudio stream started")
            return True
        except Exception as e:
            print(f"❌ Failed to start PyAudio: {e}")
            return False
    
    def stop(self):
        """Stop PyAudio stream"""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                self.playing = False
                print("🔇 PyAudio stream stopped")
            except Exception as e:
                print(f"⚠️ Error stopping PyAudio: {e}")
    
    def list_devices(self) -> List[dict]:
        """List PyAudio output devices"""
        devices = []
        try:
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxOutputChannels']
                    })
        except Exception as e:
            print(f"⚠️ Error listing devices: {e}")
        return devices
    
    def set_device(self, device_index: int):
        """Set PyAudio output device (requires restart)"""
        # PyAudio requires stream restart to change device
        was_playing = self.playing
        if was_playing:
            self.stop()
        
        # Store device index for next stream open
        self.device_index = device_index
        
        if was_playing:
            self.start(self.callback_fn)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
        if hasattr(self, 'pyaudio'):
            self.pyaudio.terminate()


class ALSABackend(AudioBackend):
    """ALSA backend for Raspberry Pi direct hardware access"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, buffer_size: int = 2048):
        super().__init__(sample_rate, channels, buffer_size)
        
        try:
            import alsaaudio
            self.alsaaudio = alsaaudio
            self.device = None
            self.thread = None
            self.callback_fn = None
            self._stop_flag = threading.Event()
            print("✓ ALSA backend initialized")
        except ImportError:
            raise RuntimeError("ALSA not available. Install with: pip install pyalsaaudio")
    
    def _alsa_thread(self):
        """ALSA playback thread"""
        while not self._stop_flag.is_set():
            try:
                # Generate audio
                audio_data = self.callback_fn(self.buffer_size)  # (frames, channels)
                
                # Clip and convert to int16
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                
                # Write to ALSA
                self.device.write(audio_bytes)
                
            except Exception as e:
                print(f"❌ ALSA thread error: {e}")
                break
    
    def start(self, callback: Callable[[int], np.ndarray]) -> bool:
        """Start ALSA playback"""
        if self.playing:
            return True
        
        try:
            # Open ALSA device
            self.device = self.alsaaudio.PCM(
                type=self.alsaaudio.PCM_PLAYBACK,
                mode=self.alsaaudio.PCM_NORMAL,
                device='default'
            )
            
            # Configure ALSA
            self.device.setchannels(self.channels)
            self.device.setrate(self.sample_rate)
            self.device.setformat(self.alsaaudio.PCM_FORMAT_S16_LE)
            self.device.setperiodsize(self.buffer_size)
            
            # Start playback thread
            self.callback_fn = callback
            self._stop_flag.clear()
            self.thread = threading.Thread(target=self._alsa_thread, daemon=True)
            self.thread.start()
            
            self.playing = True
            print("🔊 ALSA playback started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start ALSA: {e}")
            return False
    
    def stop(self):
        """Stop ALSA playback"""
        if not self.playing:
            return
        
        try:
            self._stop_flag.set()
            if self.thread:
                self.thread.join(timeout=1.0)
            if self.device:
                self.device.close()
            self.playing = False
            print("🔇 ALSA playback stopped")
        except Exception as e:
            print(f"⚠️ Error stopping ALSA: {e}")
    
    def list_devices(self) -> List[dict]:
        """List ALSA output devices"""
        devices = []
        try:
            cards = self.alsaaudio.cards()
            for i, card in enumerate(cards):
                devices.append({
                    'index': i,
                    'name': card,
                    'channels': 2  # Assume stereo
                })
        except Exception as e:
            print(f"⚠️ Error listing ALSA devices: {e}")
        return devices
    
    def set_device(self, device_index: int):
        """Set ALSA output device"""
        # ALSA requires restart to change device
        was_playing = self.playing
        if was_playing:
            self.stop()
        
        # Get device name
        cards = self.alsaaudio.cards()
        if 0 <= device_index < len(cards):
            self.device_name = f"hw:{device_index}"
        
        if was_playing:
            self.start(self.callback_fn)


class DummyBackend(AudioBackend):
    """Dummy backend for testing without hardware"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, buffer_size: int = 2048):
        super().__init__(sample_rate, channels, buffer_size)
        self.thread = None
        self.callback_fn = None
        self._stop_flag = threading.Event()
        print("✓ Dummy backend initialized (silent)")
    
    def _dummy_thread(self):
        """Simulate audio callback timing"""
        import time
        interval = self.buffer_size / self.sample_rate
        
        while not self._stop_flag.is_set():
            if self.callback_fn:
                try:
                    # Call callback but discard audio
                    _ = self.callback_fn(self.buffer_size)
                except Exception as e:
                    print(f"❌ Dummy callback error: {e}")
            
            time.sleep(interval)
    
    def start(self, callback: Callable[[int], np.ndarray]) -> bool:
        """Start dummy playback"""
        if self.playing:
            return True
        
        self.callback_fn = callback
        self._stop_flag.clear()
        self.thread = threading.Thread(target=self._dummy_thread, daemon=True)
        self.thread.start()
        self.playing = True
        print("🔊 Dummy playback started (silent)")
        return True
    
    def stop(self):
        """Stop dummy playback"""
        if not self.playing:
            return
        
        self._stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.playing = False
        print("🔇 Dummy playback stopped")
    
    def list_devices(self) -> List[dict]:
        """Return dummy device"""
        return [{'index': 0, 'name': 'Dummy Output', 'channels': 2}]
    
    def set_device(self, device_index: int):
        """Dummy device selection"""
        pass


def create_audio_backend(backend_type: Optional[str] = None, **kwargs) -> AudioBackend:
    """
    Factory function to create audio backend
    
    Args:
        backend_type: "pyaudio", "alsa", "pipewire", "dummy", or None for auto-detect
        **kwargs: Backend-specific parameters (sample_rate, channels, buffer_size)
    
    Returns:
        AudioBackend instance
    
    Auto-detection priority:
        1. AUDIO_BACKEND environment variable
        2. Pi detection -> ALSA
        3. Desktop -> PyAudio
        4. Fallback -> Dummy
    """
    
    # Get backend from environment or argument
    if backend_type is None:
        backend_type = os.environ.get('AUDIO_BACKEND', None)
    
    # Auto-detect if not specified
    if backend_type is None:
        # Check if running on Raspberry Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    backend_type = 'alsa'
                    print("🍓 Detected Raspberry Pi -> Using ALSA")
        except:
            pass
        
        # Default to PyAudio on desktop
        if backend_type is None:
            try:
                import pyaudio
                backend_type = 'pyaudio'
                print("🖥️ Using PyAudio backend")
            except ImportError:
                backend_type = 'dummy'
                print("⚠️ No audio backend available, using Dummy")
    
    # Create backend
    backend_type = backend_type.lower()
    
    if backend_type == 'pyaudio':
        return PyAudioBackend(**kwargs)
    elif backend_type == 'alsa':
        return ALSABackend(**kwargs)
    elif backend_type == 'dummy':
        return DummyBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


# Example usage:
if __name__ == "__main__":
    # Test backend creation
    backend = create_audio_backend()
    print(f"✓ Created backend: {type(backend).__name__}")
    print(f"  Devices: {backend.list_devices()}")
    
    # Test audio generation
    def test_callback(num_frames):
        """Generate 440Hz sine wave"""
        t = np.arange(num_frames) / 44100
        sine = np.sin(2 * np.pi * 440 * t) * 0.3
        return np.column_stack([sine, sine])  # Stereo
    
    print("\n🧪 Testing backend (2 seconds)...")
    backend.start(test_callback)
    
    import time
    time.sleep(2)
    
    backend.stop()
    print("✓ Test complete")
