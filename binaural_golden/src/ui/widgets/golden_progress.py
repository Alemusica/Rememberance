"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         GOLDEN PROGRESS BAR                                 ║
║                                                                              ║
║   Modern progress bar with golden gradient and glow effects                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from typing import Optional, Callable
from dataclasses import dataclass
import math
import time

from ..theme import PlateLabStyle

STYLE = PlateLabStyle()


@dataclass
class ProgressConfig:
    """Configuration for GoldenProgressBar."""
    width: int = 200
    height: int = 20
    show_label: bool = True
    glow: bool = True
    animated: bool = True
    corner_radius: int = 10
    gradient_steps: int = 20


class GoldenProgressBar(tk.Canvas):
    """
    Progress bar dorata con gradiente oro e effetto glow animato.
    
    Features:
    - Gradiente oro (#ffd700 → #cc9900)
    - Glow effect animato
    - Label integrato personalizzabile
    - Supporto indeterminate mode
    - Animazioni smooth
    
    Example:
        progress = GoldenProgressBar(
            parent, 
            config=ProgressConfig(width=300, glow=True)
        )
        progress.set_value(75, "Processing...")
    """
    
    def __init__(self, parent, *, config: Optional[ProgressConfig] = None, 
                 on_complete: Optional[Callable] = None):
        self.config = config or ProgressConfig()
        
        super().__init__(
            parent,
            width=self.config.width,
            height=self.config.height,
            bg=STYLE.BG_DARK,
            highlightthickness=0
        )
        
        self._value = 0.0
        self._target_value = 0.0
        self._label_text = ""
        self._indeterminate = False
        self._animation_id = None
        self._glow_phase = 0.0
        self._on_complete = on_complete
        
        # Create gradient colors
        self._gradient_colors = self._create_gradient()
        
        # Initial draw
        self._draw()
    
    def _create_gradient(self) -> list[str]:
        """Create gradient color list from gold to dark gold."""
        colors = []
        steps = self.config.gradient_steps
        
        # Parse hex colors
        light_rgb = self._hex_to_rgb(STYLE.GOLD)
        dark_rgb = self._hex_to_rgb(STYLE.GOLD_DARK)
        
        for i in range(steps):
            ratio = i / (steps - 1)
            r = int(light_rgb[0] + (dark_rgb[0] - light_rgb[0]) * ratio)
            g = int(light_rgb[1] + (dark_rgb[1] - light_rgb[1]) * ratio)
            b = int(light_rgb[2] + (dark_rgb[2] - light_rgb[2]) * ratio)
            colors.append(f"#{r:02x}{g:02x}{b:02x}")
        
        return colors
    
    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _draw(self):
        """Draw the progress bar."""
        self.delete("all")
        
        # Background
        self.create_rectangle(
            2, 2, self.config.width - 2, self.config.height - 2,
            fill=STYLE.BG_MEDIUM, outline=STYLE.BG_LIGHT, width=1,
            tags="background"
        )
        
        # Progress fill
        if self._value > 0:
            self._draw_progress()
        
        # Glow effect
        if self.config.glow:
            self._draw_glow()
        
        # Label
        if self.config.show_label and self._label_text:
            self._draw_label()
    
    def _draw_progress(self):
        """Draw the progress fill with gradient."""
        if self._indeterminate:
            self._draw_indeterminate()
            return
        
        progress_width = (self.config.width - 4) * (self._value / 100.0)
        
        if progress_width <= 0:
            return
        
        # Draw gradient segments
        segment_width = progress_width / len(self._gradient_colors)
        
        for i, color in enumerate(self._gradient_colors):
            x1 = 2 + i * segment_width
            x2 = min(2 + (i + 1) * segment_width, 2 + progress_width)
            
            if x2 <= x1:
                break
            
            self.create_rectangle(
                x1, 3, x2, self.config.height - 3,
                fill=color, outline="",
                tags="progress"
            )
    
    def _draw_indeterminate(self):
        """Draw indeterminate progress animation."""
        # Pulsing effect
        pulse = (math.sin(self._glow_phase * 2) + 1) / 2
        alpha = int(100 + 155 * pulse)
        
        # Moving gradient
        bar_width = self.config.width * 0.3
        x_pos = (self.config.width - bar_width) * pulse
        
        self.create_rectangle(
            x_pos, 3, x_pos + bar_width, self.config.height - 3,
            fill=STYLE.GOLD, outline="",
            tags="progress"
        )
    
    def _draw_glow(self):
        """Draw animated glow effect around progress."""
        if self._value <= 0 and not self._indeterminate:
            return
        
        # Pulsing glow
        intensity = (math.sin(self._glow_phase) + 1) / 4 + 0.3
        
        # Outer glow
        self.create_rectangle(
            0, 0, self.config.width, self.config.height,
            outline=STYLE.GOLD_LIGHT, width=2,
            tags="glow"
        )
    
    def _draw_label(self):
        """Draw centered label text."""
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        
        # Shadow
        self.create_text(
            center_x + 1, center_y + 1,
            text=self._label_text,
            fill=STYLE.BG_DARK,
            font=STYLE.font_small,
            tags="label"
        )
        
        # Main text
        self.create_text(
            center_x, center_y,
            text=self._label_text,
            fill=STYLE.TEXT_PRIMARY,
            font=STYLE.font_small,
            tags="label"
        )
    
    def _animate_step(self):
        """Single animation step."""
        # Smooth value transition
        if abs(self._value - self._target_value) > 0.1:
            diff = self._target_value - self._value
            self._value += diff * 0.15  # Smooth easing
        else:
            self._value = self._target_value
        
        # Update glow phase
        self._glow_phase += 0.2
        
        # Redraw
        self._draw()
        
        # Continue animation
        should_continue = (
            abs(self._value - self._target_value) > 0.1 or
            self.config.glow or
            self._indeterminate
        )
        
        if should_continue and self.config.animated:
            self._animation_id = self.after(50, self._animate_step)
        else:
            self._animation_id = None
            if self._value >= 100 and self._on_complete:
                self._on_complete()
    
    def set_value(self, value: float, label: str = ""):
        """Set progress value (0-100) with optional label."""
        self._target_value = max(0, min(100, value))
        self._label_text = label
        self._indeterminate = False
        
        if self.config.animated and not self._animation_id:
            self._animate_step()
        elif not self.config.animated:
            self._value = self._target_value
            self._draw()
    
    def set_indeterminate(self, active: bool = True, label: str = "Processing..."):
        """Enable/disable indeterminate mode."""
        self._indeterminate = active
        self._label_text = label if active else ""
        
        if active and self.config.animated and not self._animation_id:
            self._animate_step()
        elif not active:
            self._value = 0
            self._target_value = 0
            self._draw()
    
    def reset(self):
        """Reset progress to 0."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        
        self._value = 0
        self._target_value = 0
        self._label_text = ""
        self._indeterminate = False
        self._draw()
    
    def destroy(self):
        """Clean up animation before destroying."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
        super().destroy()
