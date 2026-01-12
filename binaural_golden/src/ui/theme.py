"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          UI THEME & STYLES                                   ║
║                                                                              ║
║   Centralized styling for the Plate Lab interface with improved contrast    ║
║   and readability. Golden ratio-based color palette.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PlateLabStyle:
    """
    Complete styling configuration for Plate Lab UI.
    
    Design principles:
    - High contrast for readability (WCAG AA minimum)
    - Golden ratio proportions where applicable
    - Dark theme with golden accents
    - Clear visual hierarchy
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    # COLOR PALETTE
    # ══════════════════════════════════════════════════════════════════════════
    
    # Base colors (dark theme)
    BG_DARK: str = "#1a1a2e"          # Main background
    BG_MEDIUM: str = "#252544"        # Secondary background
    BG_LIGHT: str = "#2d2d52"         # Tertiary / cards
    BG_HIGHLIGHT: str = "#3a3a66"     # Hover states
    
    # Text colors (high contrast)
    TEXT_PRIMARY: str = "#f5f5f5"     # Main text - high contrast
    TEXT_SECONDARY: str = "#b8b8d0"   # Secondary text
    TEXT_MUTED: str = "#808096"       # Muted/disabled text
    TEXT_DARK: str = "#1a1a2e"        # Dark text on light backgrounds
    
    # Accent colors
    GOLD: str = "#ffd700"             # Primary accent (golden)
    GOLD_LIGHT: str = "#ffe066"       # Light gold
    GOLD_DARK: str = "#cc9900"        # Dark gold
    
    # Status colors
    SUCCESS: str = "#4caf50"          # Green - success/positive
    WARNING: str = "#ff9800"          # Orange - warning
    ERROR: str = "#f44336"            # Red - error
    INFO: str = "#2196f3"             # Blue - info
    
    # Chakra colors
    CHAKRA_ROOT: str = "#ff0000"
    CHAKRA_SACRAL: str = "#ff8800"
    CHAKRA_SOLAR: str = "#ffff00"
    CHAKRA_HEART: str = "#00ff00"
    CHAKRA_THROAT: str = "#00bfff"
    CHAKRA_THIRD_EYE: str = "#4400ff"
    CHAKRA_CROWN: str = "#ff00ff"
    
    # Element colors
    PLATE_FILL: str = "#8b5a2b"       # Wood-like brown
    PLATE_STROKE: str = "#ffd700"     # Golden outline
    EXCITER_FILL: str = "#4a90d9"     # Blue for exciters
    BODY_OVERLAY: str = "#ffffff40"   # Semi-transparent white
    
    # ══════════════════════════════════════════════════════════════════════════
    # TYPOGRAPHY
    # ══════════════════════════════════════════════════════════════════════════
    
    # Font families (with fallbacks)
    FONT_MAIN: str = "Segoe UI"
    FONT_MONO: str = "Consolas"
    FONT_HEADING: str = "Segoe UI Semibold"
    
    # Font sizes (in points) - golden ratio scaled
    FONT_SIZE_XS: int = 9
    FONT_SIZE_SM: int = 10
    FONT_SIZE_MD: int = 12           # Base size
    FONT_SIZE_LG: int = 14
    FONT_SIZE_XL: int = 16
    FONT_SIZE_XXL: int = 20
    FONT_SIZE_TITLE: int = 24
    
    # Font specifications for Tkinter
    @property
    def font_normal(self) -> Tuple[str, int]:
        return (self.FONT_MAIN, self.FONT_SIZE_MD)
    
    @property
    def font_bold(self) -> Tuple[str, int, str]:
        return (self.FONT_MAIN, self.FONT_SIZE_MD, "bold")
    
    @property
    def font_heading(self) -> Tuple[str, int, str]:
        return (self.FONT_HEADING, self.FONT_SIZE_LG, "bold")
    
    @property
    def font_title(self) -> Tuple[str, int, str]:
        return (self.FONT_HEADING, self.FONT_SIZE_XL, "bold")
    
    @property
    def font_small(self) -> Tuple[str, int]:
        return (self.FONT_MAIN, self.FONT_SIZE_SM)
    
    @property
    def font_mono(self) -> Tuple[str, int]:
        return (self.FONT_MONO, self.FONT_SIZE_SM)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SPACING & LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    
    # Padding values
    PAD_XS: int = 2
    PAD_SM: int = 5
    PAD_MD: int = 10
    PAD_LG: int = 15
    PAD_XL: int = 20
    
    # Border radius (for rounded elements if supported)
    BORDER_RADIUS_SM: int = 3
    BORDER_RADIUS_MD: int = 5
    BORDER_RADIUS_LG: int = 10
    
    # Border widths
    BORDER_THIN: int = 1
    BORDER_MEDIUM: int = 2
    BORDER_THICK: int = 3
    
    # Sidebar width
    SIDEBAR_WIDTH: int = 320
    SIDEBAR_MIN_WIDTH: int = 280
    
    # Canvas default size
    CANVAS_MIN_WIDTH: int = 600
    CANVAS_MIN_HEIGHT: int = 400
    
    # ══════════════════════════════════════════════════════════════════════════
    # WIDGET STYLING
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_button_style(self, variant: str = "default") -> Dict:
        """Get button styling based on variant."""
        base = {
            "font": self.font_normal,
            "padx": self.PAD_MD,
            "pady": self.PAD_SM,
            "relief": "flat",
            "borderwidth": 0,
            "cursor": "hand2",
        }
        
        variants = {
            "default": {
                **base,
                "bg": self.BG_LIGHT,
                "fg": self.TEXT_PRIMARY,
                "activebackground": self.BG_HIGHLIGHT,
                "activeforeground": self.TEXT_PRIMARY,
            },
            "primary": {
                **base,
                "bg": self.GOLD_DARK,
                "fg": self.TEXT_DARK,
                "activebackground": self.GOLD,
                "activeforeground": self.TEXT_DARK,
            },
            "success": {
                **base,
                "bg": self.SUCCESS,
                "fg": self.TEXT_PRIMARY,
                "activebackground": "#66bb6a",
            },
            "warning": {
                **base,
                "bg": self.WARNING,
                "fg": self.TEXT_DARK,
            },
            "danger": {
                **base,
                "bg": self.ERROR,
                "fg": self.TEXT_PRIMARY,
            },
        }
        
        return variants.get(variant, variants["default"])
    
    def get_entry_style(self) -> Dict:
        """Get entry/input field styling."""
        return {
            "font": self.font_normal,
            "bg": self.BG_MEDIUM,
            "fg": self.TEXT_PRIMARY,
            "insertbackground": self.GOLD,  # Cursor color
            "relief": "flat",
            "highlightthickness": 1,
            "highlightcolor": self.GOLD,
            "highlightbackground": self.BG_LIGHT,
        }
    
    def get_label_style(self, variant: str = "default") -> Dict:
        """Get label styling."""
        base = {
            "font": self.font_normal,
            "bg": self.BG_DARK,
            "fg": self.TEXT_PRIMARY,
        }
        
        variants = {
            "default": base,
            "heading": {
                **base,
                "font": self.font_heading,
                "fg": self.GOLD,
            },
            "muted": {
                **base,
                "fg": self.TEXT_MUTED,
                "font": self.font_small,
            },
            "value": {
                **base,
                "font": self.font_mono,
                "fg": self.GOLD_LIGHT,
            },
        }
        
        return variants.get(variant, base)
    
    def get_frame_style(self, variant: str = "default") -> Dict:
        """Get frame styling."""
        variants = {
            "default": {
                "bg": self.BG_DARK,
            },
            "card": {
                "bg": self.BG_LIGHT,
                "highlightthickness": 1,
                "highlightbackground": self.BG_HIGHLIGHT,
            },
            "sidebar": {
                "bg": self.BG_MEDIUM,
            },
        }
        return variants.get(variant, variants["default"])
    
    def get_scale_style(self) -> Dict:
        """Get scale/slider styling."""
        return {
            "bg": self.BG_DARK,
            "fg": self.TEXT_PRIMARY,
            "troughcolor": self.BG_LIGHT,
            "highlightthickness": 0,
            "font": self.font_small,
            "activebackground": self.GOLD,
        }
    
    def get_listbox_style(self) -> Dict:
        """Get listbox styling."""
        return {
            "font": self.font_normal,
            "bg": self.BG_MEDIUM,
            "fg": self.TEXT_PRIMARY,
            "selectbackground": self.GOLD_DARK,
            "selectforeground": self.TEXT_DARK,
            "highlightthickness": 0,
            "relief": "flat",
        }
    
    def get_canvas_style(self) -> Dict:
        """Get canvas styling."""
        return {
            "bg": self.BG_DARK,
            "highlightthickness": 0,
        }
    
    def get_scrollbar_style(self) -> Dict:
        """Get scrollbar styling."""
        return {
            "bg": self.BG_MEDIUM,
            "troughcolor": self.BG_DARK,
            "activebackground": self.GOLD_DARK,
            "highlightthickness": 0,
        }
    
    # ══════════════════════════════════════════════════════════════════════════
    # CHAKRA HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_chakra_colors(self) -> Dict[str, str]:
        """Get all chakra colors."""
        return {
            "Muladhara": self.CHAKRA_ROOT,
            "Svadhisthana": self.CHAKRA_SACRAL,
            "Manipura": self.CHAKRA_SOLAR,
            "Anahata": self.CHAKRA_HEART,
            "Vishuddha": self.CHAKRA_THROAT,
            "Ajna": self.CHAKRA_THIRD_EYE,
            "Sahasrara": self.CHAKRA_CROWN,
        }
    
    def chakra_color_at_position(self, position: float) -> str:
        """
        Interpolate chakra color at normalized position (0-1).
        0 = root, 1 = crown.
        """
        import colorsys
        
        # Chakra hues (in degrees)
        # Red -> Orange -> Yellow -> Green -> Blue -> Indigo -> Violet
        chakra_hues = [0, 30, 60, 120, 195, 260, 300]  # Degrees
        chakra_positions = [0, 0.15, 0.35, 0.38, 0.65, 0.85, 1.0]
        
        # Find surrounding chakras
        for i in range(len(chakra_positions) - 1):
            if chakra_positions[i] <= position <= chakra_positions[i + 1]:
                # Interpolate hue
                t = (position - chakra_positions[i]) / (chakra_positions[i + 1] - chakra_positions[i])
                hue = chakra_hues[i] + t * (chakra_hues[i + 1] - chakra_hues[i])
                hue = hue / 360  # Normalize to 0-1
                
                # Convert to RGB
                r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
                return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        
        return self.CHAKRA_CROWN


# Default style instance
STYLE = PlateLabStyle()


# ══════════════════════════════════════════════════════════════════════════════
# TTK THEME CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def configure_ttk_style(style_obj):
    """
    Configure ttk style with Plate Lab theme.
    
    Args:
        style_obj: ttk.Style() instance
    """
    s = STYLE
    
    # General settings
    style_obj.configure(".",
        background=s.BG_DARK,
        foreground=s.TEXT_PRIMARY,
        fieldbackground=s.BG_MEDIUM,
        font=s.font_normal,
    )
    
    # Frame
    style_obj.configure("TFrame",
        background=s.BG_DARK,
    )
    style_obj.configure("Card.TFrame",
        background=s.BG_LIGHT,
    )
    style_obj.configure("Sidebar.TFrame",
        background=s.BG_MEDIUM,
    )
    
    # Label
    style_obj.configure("TLabel",
        background=s.BG_DARK,
        foreground=s.TEXT_PRIMARY,
        font=s.font_normal,
    )
    style_obj.configure("Heading.TLabel",
        foreground=s.GOLD,
        font=s.font_heading,
    )
    style_obj.configure("Title.TLabel",
        foreground=s.GOLD,
        font=s.font_title,
    )
    style_obj.configure("Muted.TLabel",
        foreground=s.TEXT_MUTED,
        font=s.font_small,
    )
    style_obj.configure("Value.TLabel",
        foreground=s.GOLD_LIGHT,
        font=s.font_mono,
    )
    
    # Button
    style_obj.configure("TButton",
        background=s.BG_LIGHT,
        foreground=s.TEXT_PRIMARY,
        font=s.font_normal,
        padding=(s.PAD_MD, s.PAD_SM),
    )
    style_obj.map("TButton",
        background=[("active", s.BG_HIGHLIGHT), ("pressed", s.GOLD_DARK)],
        foreground=[("active", s.TEXT_PRIMARY), ("pressed", s.TEXT_DARK)],
    )
    
    style_obj.configure("Primary.TButton",
        background=s.GOLD_DARK,
        foreground=s.TEXT_DARK,
    )
    style_obj.map("Primary.TButton",
        background=[("active", s.GOLD), ("pressed", s.GOLD_LIGHT)],
    )
    
    style_obj.configure("Success.TButton",
        background=s.SUCCESS,
        foreground=s.TEXT_PRIMARY,
    )
    
    style_obj.configure("Danger.TButton",
        background=s.ERROR,
        foreground=s.TEXT_PRIMARY,
    )
    
    # Entry
    style_obj.configure("TEntry",
        fieldbackground=s.BG_MEDIUM,
        foreground=s.TEXT_PRIMARY,
        insertcolor=s.GOLD,
    )
    
    # Spinbox
    style_obj.configure("TSpinbox",
        fieldbackground=s.BG_MEDIUM,
        foreground=s.TEXT_PRIMARY,
        arrowcolor=s.TEXT_PRIMARY,
    )
    
    # Combobox
    style_obj.configure("TCombobox",
        fieldbackground=s.BG_MEDIUM,
        foreground=s.TEXT_PRIMARY,
        arrowcolor=s.TEXT_PRIMARY,
        selectbackground=s.GOLD_DARK,
        selectforeground=s.TEXT_DARK,
    )
    
    # Scale
    style_obj.configure("TScale",
        background=s.BG_DARK,
        troughcolor=s.BG_LIGHT,
    )
    style_obj.configure("Horizontal.TScale",
        background=s.BG_DARK,
        troughcolor=s.BG_LIGHT,
    )
    
    # Progressbar
    style_obj.configure("TProgressbar",
        background=s.GOLD,
        troughcolor=s.BG_LIGHT,
    )
    
    # Notebook (tabs)
    style_obj.configure("TNotebook",
        background=s.BG_DARK,
    )
    style_obj.configure("TNotebook.Tab",
        background=s.BG_MEDIUM,
        foreground=s.TEXT_PRIMARY,
        padding=(s.PAD_MD, s.PAD_SM),
    )
    style_obj.map("TNotebook.Tab",
        background=[("selected", s.BG_LIGHT)],
        foreground=[("selected", s.GOLD)],
    )
    
    # Scrollbar
    style_obj.configure("TScrollbar",
        background=s.BG_MEDIUM,
        troughcolor=s.BG_DARK,
        arrowcolor=s.TEXT_PRIMARY,
    )
    style_obj.map("TScrollbar",
        background=[("active", s.GOLD_DARK)],
    )
    
    # Separator
    style_obj.configure("TSeparator",
        background=s.BG_HIGHLIGHT,
    )
    
    # Checkbutton
    style_obj.configure("TCheckbutton",
        background=s.BG_DARK,
        foreground=s.TEXT_PRIMARY,
    )
    style_obj.map("TCheckbutton",
        background=[("active", s.BG_DARK)],
        foreground=[("active", s.GOLD)],
    )
    
    # Radiobutton
    style_obj.configure("TRadiobutton",
        background=s.BG_DARK,
        foreground=s.TEXT_PRIMARY,
    )
    style_obj.map("TRadiobutton",
        background=[("active", s.BG_DARK)],
        foreground=[("active", s.GOLD)],
    )
    
    # Labelframe
    style_obj.configure("TLabelframe",
        background=s.BG_DARK,
    )
    style_obj.configure("TLabelframe.Label",
        background=s.BG_DARK,
        foreground=s.GOLD,
        font=s.font_heading,
    )
    
    # Treeview
    style_obj.configure("Treeview",
        background=s.BG_MEDIUM,
        foreground=s.TEXT_PRIMARY,
        fieldbackground=s.BG_MEDIUM,
    )
    style_obj.map("Treeview",
        background=[("selected", s.GOLD_DARK)],
        foreground=[("selected", s.TEXT_DARK)],
    )
    style_obj.configure("Treeview.Heading",
        background=s.BG_LIGHT,
        foreground=s.GOLD,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_dark_title_bar(window):
    """
    Apply dark title bar on Windows 10/11.
    
    Args:
        window: Tkinter Tk or Toplevel window
    """
    try:
        import ctypes
        window.update()
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(True)),
            ctypes.sizeof(ctypes.c_int)
        )
    except:
        pass  # Not on Windows or not supported


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def blend_colors(color1: str, color2: str, t: float = 0.5) -> str:
    """Blend two colors. t=0 gives color1, t=1 gives color2."""
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    r = int(r1 + t * (r2 - r1))
    g = int(g1 + t * (g2 - g1))
    b = int(b1 + t * (b2 - b1))
    
    return rgb_to_hex(r, g, b)


def darken_color(color: str, factor: float = 0.2) -> str:
    """Darken a color by a factor."""
    return blend_colors(color, "#000000", factor)


def lighten_color(color: str, factor: float = 0.2) -> str:
    """Lighten a color by a factor."""
    return blend_colors(color, "#ffffff", factor)


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PLATE LAB THEME")
    print("=" * 60)
    
    s = STYLE
    
    print(f"\n🎨 COLOR PALETTE:")
    print(f"   Background: {s.BG_DARK}")
    print(f"   Text: {s.TEXT_PRIMARY}")
    print(f"   Gold: {s.GOLD}")
    
    print(f"\n📝 TYPOGRAPHY:")
    print(f"   Normal: {s.font_normal}")
    print(f"   Heading: {s.font_heading}")
    
    print(f"\n🌈 CHAKRA COLORS:")
    for name, color in s.get_chakra_colors().items():
        print(f"   {name}: {color}")
