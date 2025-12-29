"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           MODERN BUTTON                                     ║
║                                                                              ║
║   Golden-styled button with hover effects, icons, and animations          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from typing import Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path

from ..theme import PlateLabStyle

STYLE = PlateLabStyle()


class ButtonVariant(Enum):
    """Button style variants."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
    GHOST = "ghost"


class ButtonSize(Enum):
    """Button size variants."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class ButtonConfig:
    """Configuration for GoldenButton."""
    text: str = "Button"
    variant: ButtonVariant = ButtonVariant.PRIMARY
    size: ButtonSize = ButtonSize.MEDIUM
    width: Optional[int] = None
    height: Optional[int] = None
    icon: Optional[str] = None  # Unicode icon or path
    icon_position: str = "left"  # "left", "right", "top", "bottom"
    disabled: bool = False
    animated: bool = True
    show_glow: bool = True
    border_radius: int = 6
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get button dimensions based on size."""
        if self.width is not None and self.height is not None:
            return self.width, self.height
        
        size_map = {
            ButtonSize.SMALL: (80, 28),
            ButtonSize.MEDIUM: (120, 36),
            ButtonSize.LARGE: (160, 44),
            ButtonSize.XLARGE: (200, 52)
        }
        
        return size_map.get(self.size, size_map[ButtonSize.MEDIUM])
    
    def get_font(self) -> Tuple[str, int, str]:
        """Get font specification based on size."""
        font_sizes = {
            ButtonSize.SMALL: STYLE.FONT_SIZE_SM,
            ButtonSize.MEDIUM: STYLE.FONT_SIZE_MD,
            ButtonSize.LARGE: STYLE.FONT_SIZE_LG,
            ButtonSize.XLARGE: STYLE.FONT_SIZE_XL
        }
        
        size = font_sizes.get(self.size, STYLE.FONT_SIZE_MD)
        return (STYLE.FONT_MAIN, size, "bold")
    
    def get_padding(self) -> int:
        """Get internal padding based on size."""
        padding_map = {
            ButtonSize.SMALL: STYLE.PAD_SM,
            ButtonSize.MEDIUM: STYLE.PAD_MD,
            ButtonSize.LARGE: STYLE.PAD_LG,
            ButtonSize.XLARGE: STYLE.PAD_XL
        }
        
        return padding_map.get(self.size, STYLE.PAD_MD)


class GoldenButton(tk.Canvas):
    """
    Button moderno con styling dorato.
    
    Features:
    - Hover effect con glow animato
    - Click animation con scaling
    - Supporto icone (unicode o custom)
    - Varianti colore multiple
    - Stati disabled con feedback visivo
    - Animazioni smooth
    
    Example:
        button = GoldenButton(
            parent,
            config=ButtonConfig(
                text="Start Evolution",
                variant=ButtonVariant.PRIMARY,
                size=ButtonSize.LARGE,
                icon="▶",  # Play icon
                show_glow=True
            ),
            command=lambda: print("Evolution started!")
        )
    """
    
    def __init__(self, parent, *, config: ButtonConfig,
                 command: Optional[Callable[[], None]] = None):
        
        self.config = config
        self._command = command
        
        # Get dimensions
        width, height = self.config.get_dimensions()
        
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=STYLE.BG_DARK,
            highlightthickness=0,
            relief="flat",
            bd=0
        )
        
        # State
        self._is_hovered = False
        self._is_pressed = False
        self._animation_id = None
        self._glow_phase = 0.0
        self._scale_factor = 1.0
        self._target_scale = 1.0
        
        # Colors based on variant
        self._colors = self._get_variant_colors()
        
        # Bind events
        self._bind_events()
        
        # Initial draw
        self._draw()
        
        # Start glow animation if enabled
        if self.config.show_glow and not self.config.disabled:
            self._start_glow_animation()
    
    def _get_variant_colors(self) -> dict:
        """Get colors for button variant."""
        color_maps = {
            ButtonVariant.PRIMARY: {
                "bg": STYLE.GOLD,
                "bg_hover": STYLE.GOLD_LIGHT,
                "bg_pressed": STYLE.GOLD_DARK,
                "text": STYLE.TEXT_DARK,
                "border": STYLE.GOLD_LIGHT,
                "glow": STYLE.GOLD_LIGHT
            },
            ButtonVariant.SECONDARY: {
                "bg": STYLE.BG_MEDIUM,
                "bg_hover": STYLE.BG_LIGHT,
                "bg_pressed": STYLE.BG_HIGHLIGHT,
                "text": STYLE.TEXT_PRIMARY,
                "border": STYLE.GOLD,
                "glow": STYLE.GOLD
            },
            ButtonVariant.SUCCESS: {
                "bg": STYLE.SUCCESS,
                "bg_hover": "#5cbf60",
                "bg_pressed": "#3e8e41",
                "text": STYLE.TEXT_DARK,
                "border": "#5cbf60",
                "glow": "#5cbf60"
            },
            ButtonVariant.WARNING: {
                "bg": STYLE.WARNING,
                "bg_hover": "#ffb74d",
                "bg_pressed": "#e65100",
                "text": STYLE.TEXT_DARK,
                "border": "#ffb74d",
                "glow": "#ffb74d"
            },
            ButtonVariant.DANGER: {
                "bg": STYLE.ERROR,
                "bg_hover": "#f66",
                "bg_pressed": "#d32f2f",
                "text": STYLE.TEXT_PRIMARY,
                "border": "#f66",
                "glow": "#f66"
            },
            ButtonVariant.GHOST: {
                "bg": "transparent",
                "bg_hover": STYLE.GOLD + "20",
                "bg_pressed": STYLE.GOLD + "40",
                "text": STYLE.GOLD,
                "border": STYLE.GOLD,
                "glow": STYLE.GOLD
            }
        }
        
        colors = color_maps.get(self.config.variant, color_maps[ButtonVariant.PRIMARY])
        
        # Apply disabled state
        if self.config.disabled:
            colors = {
                "bg": STYLE.BG_MEDIUM,
                "bg_hover": STYLE.BG_MEDIUM,
                "bg_pressed": STYLE.BG_MEDIUM,
                "text": STYLE.TEXT_MUTED,
                "border": STYLE.BG_LIGHT,
                "glow": STYLE.BG_LIGHT
            }
        
        return colors
    
    def _bind_events(self):
        """Bind mouse events."""
        if not self.config.disabled:
            self.bind("<Button-1>", self._on_click)
            self.bind("<ButtonRelease-1>", self._on_release)
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
    
    def _draw(self):
        """Draw the button."""
        self.delete("all")
        
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        
        # Apply scaling animation
        if self._scale_factor != 1.0:
            scale_width = int(width * self._scale_factor)
            scale_height = int(height * self._scale_factor)
            offset_x = (width - scale_width) // 2
            offset_y = (height - scale_height) // 2
        else:
            scale_width, scale_height = width, height
            offset_x, offset_y = 0, 0
        
        # Determine current colors
        if self._is_pressed:
            bg_color = self._colors["bg_pressed"]
        elif self._is_hovered:
            bg_color = self._colors["bg_hover"]
        else:
            bg_color = self._colors["bg"]
        
        # Draw glow effect first
        if self.config.show_glow and self._is_hovered and not self.config.disabled:
            self._draw_glow(offset_x, offset_y, scale_width, scale_height)
        
        # Draw button background
        if bg_color != "transparent":
            if self.config.border_radius > 0:
                self._draw_rounded_rect(
                    offset_x, offset_y,
                    offset_x + scale_width, offset_y + scale_height,
                    self.config.border_radius,
                    fill=bg_color,
                    outline=self._colors["border"],
                    width=2 if self._is_hovered else 1,
                    tags="background"
                )
            else:
                self.create_rectangle(
                    offset_x, offset_y,
                    offset_x + scale_width, offset_y + scale_height,
                    fill=bg_color,
                    outline=self._colors["border"],
                    width=2 if self._is_hovered else 1,
                    tags="background"
                )
        else:
            # Ghost variant - just border
            if self.config.border_radius > 0:
                self._draw_rounded_rect(
                    offset_x, offset_y,
                    offset_x + scale_width, offset_y + scale_height,
                    self.config.border_radius,
                    fill="",
                    outline=self._colors["border"],
                    width=2,
                    tags="background"
                )
            else:
                self.create_rectangle(
                    offset_x, offset_y,
                    offset_x + scale_width, offset_y + scale_height,
                    fill="",
                    outline=self._colors["border"],
                    width=2,
                    tags="background"
                )
        
        # Draw content (icon + text)
        self._draw_content(offset_x, offset_y, scale_width, scale_height)
    
    def _draw_glow(self, x: int, y: int, width: int, height: int):
        """Draw glow effect around button."""
        # Animated glow intensity
        intensity = (math.sin(self._glow_phase) + 1) / 2 * 0.5 + 0.3
        glow_size = int(6 * intensity)
        
        # Multiple glow layers for smooth effect
        for i in range(3):
            layer_alpha = int(255 * intensity / (i + 1) / 3)
            
            if self.config.border_radius > 0:
                self._draw_rounded_rect(
                    x - glow_size + i, y - glow_size + i,
                    x + width + glow_size - i, y + height + glow_size - i,
                    self.config.border_radius + glow_size,
                    fill="",
                    outline=self._colors["glow"],
                    width=2,
                    tags="glow"
                )
            else:
                self.create_rectangle(
                    x - glow_size + i, y - glow_size + i,
                    x + width + glow_size - i, y + height + glow_size - i,
                    fill="",
                    outline=self._colors["glow"],
                    width=2,
                    tags="glow"
                )
    
    def _draw_rounded_rect(self, x1: int, y1: int, x2: int, y2: int, radius: int,
                          **kwargs):
        """Draw rounded rectangle (simplified version)."""
        # Simplified rounded rectangle using lines and arcs
        # For a full implementation, you'd want to use PIL or custom path drawing
        
        # For now, just draw a regular rectangle
        # In a production version, implement proper rounded corners
        self.create_rectangle(x1, y1, x2, y2, **kwargs)
    
    def _draw_content(self, x: int, y: int, width: int, height: int):
        """Draw button content (icon + text)."""
        padding = self.config.get_padding()
        content_width = width - 2 * padding
        content_height = height - 2 * padding
        
        # Calculate icon and text sizes
        icon_size = 0
        text_width = 0
        
        if self.config.icon:
            # Estimate icon size (would be more precise with actual font metrics)
            icon_size = int(self.config.get_font()[1] * 1.2)
        
        if self.config.text:
            # Estimate text width (rough approximation)
            text_width = len(self.config.text) * self.config.get_font()[1] * 0.6
        
        # Layout based on icon position
        if self.config.icon and self.config.text:
            if self.config.icon_position in ["left", "right"]:
                # Horizontal layout
                total_width = icon_size + text_width + padding // 2
                start_x = x + (width - total_width) // 2
                center_y = y + height // 2
                
                if self.config.icon_position == "left":
                    icon_x = start_x
                    text_x = icon_x + icon_size + padding // 2
                else:  # right
                    text_x = start_x
                    icon_x = text_x + text_width + padding // 2
                
                # Draw icon
                self.create_text(
                    icon_x + icon_size // 2, center_y,
                    text=self.config.icon,
                    fill=self._colors["text"],
                    font=self.config.get_font(),
                    anchor="center",
                    tags="icon"
                )
                
                # Draw text
                self.create_text(
                    text_x, center_y,
                    text=self.config.text,
                    fill=self._colors["text"],
                    font=self.config.get_font(),
                    anchor="w" if self.config.icon_position == "left" else "e",
                    tags="text"
                )
                
            else:
                # Vertical layout (top/bottom)
                center_x = x + width // 2
                total_height = icon_size + self.config.get_font()[1] + padding // 2
                start_y = y + (height - total_height) // 2
                
                if self.config.icon_position == "top":
                    icon_y = start_y + icon_size // 2
                    text_y = icon_y + icon_size // 2 + padding // 2
                else:  # bottom
                    text_y = start_y
                    icon_y = text_y + self.config.get_font()[1] + padding // 2
                
                # Draw icon
                self.create_text(
                    center_x, icon_y,
                    text=self.config.icon,
                    fill=self._colors["text"],
                    font=self.config.get_font(),
                    anchor="center",
                    tags="icon"
                )
                
                # Draw text
                self.create_text(
                    center_x, text_y,
                    text=self.config.text,
                    fill=self._colors["text"],
                    font=self.config.get_font(),
                    anchor="center",
                    tags="text"
                )
        
        elif self.config.icon:
            # Icon only
            center_x = x + width // 2
            center_y = y + height // 2
            
            self.create_text(
                center_x, center_y,
                text=self.config.icon,
                fill=self._colors["text"],
                font=self.config.get_font(),
                anchor="center",
                tags="icon"
            )
        
        elif self.config.text:
            # Text only
            center_x = x + width // 2
            center_y = y + height // 2
            
            self.create_text(
                center_x, center_y,
                text=self.config.text,
                fill=self._colors["text"],
                font=self.config.get_font(),
                anchor="center",
                tags="text"
            )
    
    def _start_glow_animation(self):
        """Start glow animation loop."""
        if self.config.animated and not self.config.disabled:
            self._animate_glow()
    
    def _animate_glow(self):
        """Single glow animation step."""
        self._glow_phase += 0.15
        
        # Update scale animation
        if abs(self._scale_factor - self._target_scale) > 0.01:
            diff = self._target_scale - self._scale_factor
            self._scale_factor += diff * 0.2
        else:
            self._scale_factor = self._target_scale
        
        # Redraw if anything changed
        if self.config.show_glow and self._is_hovered:
            self._draw()
        elif abs(self._scale_factor - self._target_scale) > 0.01:
            self._draw()
        
        # Continue animation
        should_continue = (
            (self.config.show_glow and self._is_hovered) or
            abs(self._scale_factor - self._target_scale) > 0.01
        )
        
        if should_continue and self.config.animated:
            self._animation_id = self.after(50, self._animate_glow)
        else:
            self._animation_id = None
    
    def _on_click(self, event):
        """Handle mouse click."""
        if self.config.disabled:
            return
        
        self._is_pressed = True
        self._target_scale = 0.95  # Slight shrink on click
        
        if not self._animation_id:
            self._animate_glow()
        
        self._draw()
    
    def _on_release(self, event):
        """Handle mouse release."""
        if self.config.disabled:
            return
        
        self._is_pressed = False
        self._target_scale = 1.0
        
        # Execute command if click was released over button
        if (0 <= event.x <= self.winfo_width() and 
            0 <= event.y <= self.winfo_height()):
            if self._command:
                self._command()
        
        if not self._animation_id:
            self._animate_glow()
        
        self._draw()
    
    def _on_enter(self, event):
        """Handle mouse enter."""
        if self.config.disabled:
            return
        
        self._is_hovered = True
        
        if self.config.show_glow and not self._animation_id:
            self._start_glow_animation()
        
        self._draw()
    
    def _on_leave(self, event):
        """Handle mouse leave."""
        if self.config.disabled:
            return
        
        self._is_hovered = False
        self._is_pressed = False
        self._target_scale = 1.0
        
        self._draw()
    
    # Public API methods
    
    def set_text(self, text: str):
        """Update button text."""
        self.config.text = text
        self._draw()
    
    def set_icon(self, icon: Optional[str]):
        """Update button icon."""
        self.config.icon = icon
        self._draw()
    
    def set_enabled(self, enabled: bool):
        """Enable/disable button."""
        self.config.disabled = not enabled
        self._colors = self._get_variant_colors()
        
        # Rebind events
        self.unbind("<Button-1>")
        self.unbind("<ButtonRelease-1>")
        self.unbind("<Enter>")
        self.unbind("<Leave>")
        
        if not self.config.disabled:
            self._bind_events()
        
        self._draw()
    
    def set_variant(self, variant: ButtonVariant):
        """Change button variant."""
        self.config.variant = variant
        self._colors = self._get_variant_colors()
        self._draw()
    
    def set_command(self, command: Optional[Callable[[], None]]):
        """Update button command."""
        self._command = command
    
    def simulate_click(self):
        """Programmatically trigger button click."""
        if not self.config.disabled and self._command:
            # Brief animation
            self._is_pressed = True
            self._target_scale = 0.95
            self._draw()
            
            # Execute command after short delay
            self.after(100, lambda: [
                setattr(self, '_is_pressed', False),
                setattr(self, '_target_scale', 1.0),
                self._draw(),
                self._command() if self._command else None
            ])
    
    def destroy(self):
        """Clean up animations before destroying."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
        super().destroy()
