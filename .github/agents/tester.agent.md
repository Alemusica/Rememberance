---
description: Genera test pytest per audio DSP e GUI
name: Test Generator
tools: ['codebase', 'search', 'editFiles', 'usages', 'runInTerminal']
model: Claude Sonnet 4
handoffs:
  - label: ▶️ Esegui Test
    agent: agent
    prompt: Esegui i test generati con pytest.
    send: false
  - label: 🔍 Review Test
    agent: Code Reviewer
    prompt: Fai review dei test generati per completezza.
    send: false
---

# 🧪 Test Generation Mode - Rememberance

Genera test completi per moduli audio DSP e GUI con pytest.

## Struttura Test

```
tests/
├── conftest.py              # Fixtures globali audio
├── test_core/
│   ├── test_audio_engine.py
│   ├── test_golden_math.py
│   ├── test_plate_physics.py
│   └── test_sacred_geometry.py
├── test_modules/
│   ├── test_emdr.py
│   ├── test_vibroacoustic.py
│   └── test_spectral.py
└── test_integration/
    └── test_full_pipeline.py
```

## Fixtures Audio

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_rate():
    """Standard audio sample rate."""
    return 44100

@pytest.fixture
def block_size():
    """Standard audio block size."""
    return 256

@pytest.fixture
def phi():
    """Golden ratio constant."""
    return 1.618033988749895

@pytest.fixture
def test_duration():
    """Standard test duration in seconds."""
    return 1.0

@pytest.fixture
def silence(block_size):
    """Silent audio block."""
    return np.zeros(block_size, dtype=np.float32)

@pytest.fixture
def impulse(block_size):
    """Impulse signal for testing."""
    signal = np.zeros(block_size, dtype=np.float32)
    signal[0] = 1.0
    return signal

@pytest.fixture
def sine_440(sample_rate, test_duration):
    """440Hz sine wave for testing."""
    t = np.linspace(0, test_duration, int(sample_rate * test_duration), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)
```

## Template Test DSP

```python
import pytest
import numpy as np

class TestGoldenMath:
    """Test suite for golden ratio calculations."""
    
    def test_phi_precision(self, phi):
        """PHI should have correct precision."""
        from src.core.golden_math import PHI
        assert abs(PHI - phi) < 1e-15
    
    def test_phi_squared(self, phi):
        """PHI² = PHI + 1."""
        from src.core.golden_math import PHI, PHI_SQUARED
        assert abs(PHI_SQUARED - (PHI + 1)) < 1e-14
    
    def test_phi_inverse(self, phi):
        """1/PHI = PHI - 1."""
        from src.core.golden_math import PHI, PHI_INVERSE
        assert abs(PHI_INVERSE - (PHI - 1)) < 1e-14


class TestAudioEngine:
    """Test suite for audio engine."""
    
    def test_output_shape(self, sample_rate, block_size):
        """Output should match expected shape."""
        from src.core.audio_engine import AudioEngine
        engine = AudioEngine(sample_rate=sample_rate)
        output = engine.generate_block(block_size)
        assert output.shape == (block_size,)
    
    def test_output_dtype(self, block_size):
        """Output should be float32."""
        from src.core.audio_engine import AudioEngine
        engine = AudioEngine()
        output = engine.generate_block(block_size)
        assert output.dtype == np.float32
    
    def test_amplitude_range(self, block_size):
        """Output amplitude should be normalized."""
        from src.core.audio_engine import AudioEngine
        engine = AudioEngine()
        output = engine.generate_block(block_size)
        assert np.all(np.abs(output) <= 1.0)
    
    def test_no_nan(self, block_size):
        """Output should contain no NaN values."""
        from src.core.audio_engine import AudioEngine
        engine = AudioEngine()
        output = engine.generate_block(block_size)
        assert not np.any(np.isnan(output))


class TestBinauralBeat:
    """Test binaural beat generation."""
    
    def test_stereo_output(self, sample_rate, test_duration):
        """Should produce stereo output."""
        from src.core.audio_engine import generate_binaural
        left, right = generate_binaural(440, 10, test_duration, sample_rate)
        assert left.shape == right.shape
    
    def test_frequency_difference(self, sample_rate):
        """Beat frequency should be difference of L/R."""
        # FFT analysis to verify frequency content
        pass
    
    def test_phase_coherence(self):
        """Channels should maintain phase relationship."""
        pass
```

## Test Parametrizzati per Frequenze

```python
import pytest

SOLFEGGIO_FREQS = [396, 417, 528, 639, 741, 852]
CHAKRA_FREQS = [256, 288, 320, 341.3, 384, 426.7, 480]

@pytest.mark.parametrize("freq", SOLFEGGIO_FREQS)
def test_solfeggio_generation(freq, sample_rate):
    """Test each Solfeggio frequency generates correctly."""
    from src.core.audio_engine import generate_sine
    signal = generate_sine(freq, 1.0, sample_rate)
    
    # Verify fundamental frequency via FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    peak_freq = abs(freqs[np.argmax(np.abs(fft))])
    
    assert abs(peak_freq - freq) < 1.0  # Within 1Hz tolerance

@pytest.mark.parametrize("freq", CHAKRA_FREQS)
def test_chakra_frequencies(freq):
    """Test chakra frequency generation."""
    # Similar FFT verification
    pass
```

## Run Commands

```bash
# Esegui tutti i test
cd binaural_golden && python -m pytest tests/ -v

# Con coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Solo test veloci (no integration)
python -m pytest tests/ -v -m "not slow"

# Test specifico modulo
python -m pytest tests/test_core/test_golden_math.py -v
```

## Coverage Goals

- **Core audio**: > 90%
- **Golden math**: 100% (critici)
- **GUI**: > 70% (harder to test)
- **Integration**: Critical paths
