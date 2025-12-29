---
agent: GUI Designer
tools: ['codebase', 'editFiles', 'problems']
description: Crea nuovo tab GUI per Golden Studio
---

# 🎨 Create New GUI Tab

Crea un nuovo tab per Golden Studio seguendo il design system esistente.

## Parametri
- **Nome Tab**: ${input:name:Nome del tab (es. EMDR Control)}
- **Modulo Audio**: ${input:module:Modulo audio da controllare (es. modules/emdr)}
- **Descrizione**: ${input:description:Funzionalità principale del tab}

## File da Creare
`src/ui/tabs/{name_snake}_tab.py`

## Template Structure

```python
"""
{Name} Tab - {Description}
"""
import tkinter as tk
from tkinter import ttk
import numpy as np

from ui.theme import COLORS, FONTS, PlateLabStyle
from core.audio_engine import AudioEngine

class {Name}Tab(ttk.Frame):
    """Tab for {description}."""
    
    def __init__(self, parent, audio_engine: AudioEngine):
        super().__init__(parent)
        self.audio = audio_engine
        
        self._create_widgets()
        self._layout_widgets()
        self._bind_events()
        
    def _create_widgets(self):
        """Create all widgets."""
        # Control panel
        self.control_frame = ttk.LabelFrame(self, text="Controls")
        
        # Visualization
        self.canvas = tk.Canvas(
            self, width=400, height=300,
            bg=COLORS['bg_dark'], highlightthickness=0
        )
        
        # Transport (Play/Stop)
        self.play_btn = ttk.Button(
            self.control_frame, text="▶ Play",
            command=self._on_play
        )
        self.stop_btn = ttk.Button(
            self.control_frame, text="⏹ Stop",
            command=self._on_stop
        )
        
    def _layout_widgets(self):
        """Layout all widgets."""
        self.control_frame.pack(side='left', fill='y', padx=5, pady=5)
        self.canvas.pack(side='right', fill='both', expand=True)
        
        self.play_btn.pack(pady=5)
        self.stop_btn.pack(pady=5)
        
    def _bind_events(self):
        """Bind event handlers."""
        pass
        
    def _on_play(self):
        """Handle play button."""
        self.audio.start()
        
    def _on_stop(self):
        """Handle stop button."""
        self.audio.stop()
```

## Integration
Aggiungi il tab in `src/golden_studio.py`:

```python
from ui.tabs.{name_snake}_tab import {Name}Tab

# Nel metodo _create_tabs():
self.{name_snake}_tab = {Name}Tab(self.notebook, self.audio_engine)
self.notebook.add(self.{name_snake}_tab, text="{Name}")
```
