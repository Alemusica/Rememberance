---
agent: Test Generator
tools: ['codebase', 'editFiles', 'runInTerminal', 'usages']
description: Genera test pytest per modulo DSP
---

# 🧪 Generate DSP Tests

Genera test completi pytest per il modulo audio specificato.

## Target Module
${input:module:Path del modulo (es. src/core/audio_engine.py)}

## Test Coverage Requirements
- Test per ogni funzione pubblica
- Test edge cases (empty arrays, NaN, inf)
- Test precisione numerica (tolleranza 1e-6)
- Test frequenze standard (440Hz, Solfeggio, Chakra)

## Template Test

```python
import pytest
import numpy as np

class Test{ModuleName}:
    
    @pytest.fixture
    def sample_rate(self):
        return 44100
    
    def test_{function_name}_basic(self, sample_rate):
        """Test basic functionality."""
        result = function_under_test(...)
        assert result is not None
        
    def test_{function_name}_dtype(self):
        """Output should be float32."""
        result = function_under_test(...)
        assert result.dtype == np.float32
        
    def test_{function_name}_no_nan(self):
        """Output should not contain NaN."""
        result = function_under_test(...)
        assert not np.any(np.isnan(result))
```

## Run After Generation
```bash
cd binaural_golden && python -m pytest tests/ -v
```
