"""
Golden Swiss Theme - Swiss Typography Design with φ Proportions

Design principles:
- Swiss/International Typographic Style (clean, grid-based, minimal)
- All proportions based on Golden Ratio (φ = 1.618033988749895)
- Rounded corners using φ-derived radii
- Neutral color palette with golden accents
- Strong hierarchy through type scale

φ-based measurements:
- Base unit: 8px (universal grid)
- Spacing: 8, 13, 21, 34, 55, 89... (Fibonacci)
- Corner radii: 3, 5, 8, 13, 21... (Fibonacci)
- Font scale: 11, 13, 16, 21, 26, 34, 42, 55... (≈ ×φ)
"""

import tkinter as tk
from tkinter import ttk
import math

# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498949  # 1/φ = φ-1

# Fibonacci sequence for spacing and sizing
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Base unit (8px grid - common in Swiss design)
BASE = 8


# ══════════════════════════════════════════════════════════════════════════════
# φ-BASED SPACING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class Spacing:
    """Fibonacci-based spacing for consistent rhythm"""
    XXXS = 1    # 1px - hairline
    XXS = 2     # 2px
    XS = 3      # 3px
    SM = 5      # 5px
    MD = 8      # 8px - base unit
    LG = 13     # 13px
    XL = 21     # 21px
    XXL = 34    # 34px
    XXXL = 55   # 55px
    HUGE = 89   # 89px


class Radius:
    """φ-derived corner radii"""
    NONE = 0
    XS = 3      # Small elements
    SM = 5      # Buttons, inputs
    MD = 8      # Cards, panels
    LG = 13     # Large containers
    XL = 21     # Hero sections
    FULL = 999  # Pill shape


class FontSize:
    """Type scale based on φ multiplier (≈ 1.618)"""
    CAPTION = 11    # 11px - captions, labels
    BODY_SM = 13    # 13px - small body
    BODY = 16       # 16px - base body (16 ≈ 13×φ×0.77)
    BODY_LG = 18    # 18px - emphasized body
    H6 = 21         # 21px - smallest heading
    H5 = 26         # 26px ≈ 21×φ×0.77
    H4 = 34         # 34px - section heading
    H3 = 42         # 42px ≈ 34×φ×0.77
    H2 = 55         # 55px - page heading
    H1 = 68         # 68px ≈ 55×φ×0.77
    DISPLAY = 89    # 89px - hero display


# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Swiss Neutral with Golden Accent
# ══════════════════════════════════════════════════════════════════════════════

class Colors:
    """
    Swiss-inspired neutral palette with golden accents.
    Dark theme optimized for audio applications.
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # Background hierarchy (darkest to lightest)
    # ─────────────────────────────────────────────────────────────────────
    BG_DEEPEST = "#0a0a0c"      # Absolute background
    BG_BASE = "#101014"         # Main background
    BG_ELEVATED = "#18181c"     # Cards, panels
    BG_SURFACE = "#1e1e24"      # Interactive surfaces
    BG_HOVER = "#26262e"        # Hover state
    BG_ACTIVE = "#2e2e38"       # Active/pressed state
    
    # ─────────────────────────────────────────────────────────────────────
    # Text hierarchy
    # ─────────────────────────────────────────────────────────────────────
    TEXT_PRIMARY = "#f5f5f7"    # Primary text (high contrast)
    TEXT_SECONDARY = "#a1a1a6"  # Secondary text
    TEXT_TERTIARY = "#6e6e73"   # Disabled, hints
    TEXT_INVERSE = "#0a0a0c"    # Text on light backgrounds
    
    # ─────────────────────────────────────────────────────────────────────
    # Borders
    # ─────────────────────────────────────────────────────────────────────
    BORDER_SUBTLE = "#2a2a32"   # Subtle separation
    BORDER_DEFAULT = "#38383f"  # Default borders
    BORDER_STRONG = "#48484f"   # Emphasized borders
    
    # ─────────────────────────────────────────────────────────────────────
    # Golden Accent (derived from golden ratio hue)
    # Hue 43° (gold) with φ-based saturation/lightness
    # ─────────────────────────────────────────────────────────────────────
    GOLD_DARK = "#8b6914"       # Dark gold
    GOLD = "#d4a520"            # Primary gold
    GOLD_LIGHT = "#f0c850"      # Light gold
    GOLD_SUBTLE = "#2d2510"     # Subtle gold bg
    
    # ─────────────────────────────────────────────────────────────────────
    # Semantic colors
    # ─────────────────────────────────────────────────────────────────────
    SUCCESS = "#34c759"         # Green - success
    WARNING = "#ff9f0a"         # Orange - warning
    ERROR = "#ff453a"           # Red - error
    INFO = "#64d2ff"            # Blue - info
    
    # ─────────────────────────────────────────────────────────────────────
    # Chakra colors (for body visualization)
    # ─────────────────────────────────────────────────────────────────────
    CHAKRA_ROOT = "#ff4444"     # Red - Muladhara
    CHAKRA_SACRAL = "#ff8c00"   # Orange - Svadhisthana
    CHAKRA_SOLAR = "#ffd700"    # Yellow - Manipura
    CHAKRA_HEART = "#00ff7f"    # Green - Anahata
    CHAKRA_THROAT = "#00bfff"   # Blue - Vishuddha
    CHAKRA_THIRD_EYE = "#8a2be2" # Indigo - Ajna
    CHAKRA_CROWN = "#da70d6"    # Violet - Sahasrara


# ══════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY - Swiss Style (Helvetica Neue / SF Pro / System)
# ══════════════════════════════════════════════════════════════════════════════

class Typography:
    """
    Swiss typography settings.
    Uses system fonts for best rendering.
    """
    
    # Font families (fallback chain)
    FAMILY_SANS = ("SF Pro Display", "Helvetica Neue", "Helvetica", "Arial", "sans-serif")
    FAMILY_MONO = ("SF Mono", "Monaco", "Menlo", "Consolas", "monospace")
    
    # Font weights
    WEIGHT_LIGHT = "light"
    WEIGHT_REGULAR = "normal"
    WEIGHT_MEDIUM = "medium"
    WEIGHT_SEMIBOLD = "semibold"
    WEIGHT_BOLD = "bold"
    
    # Letter spacing (em units, Swiss style uses tight tracking)
    TRACKING_TIGHT = -0.02
    TRACKING_NORMAL = 0
    TRACKING_WIDE = 0.05
    
    # Line height (φ-based)
    LINE_HEIGHT_TIGHT = 1.2      # Headings
    LINE_HEIGHT_NORMAL = 1.5     # Body (≈ φ × 0.93)
    LINE_HEIGHT_RELAXED = 1.618  # Exactly φ for meditative text


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN GRID SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def golden_grid(columns: int = 8) -> list:
    """
    Create a golden ratio grid.
    Each column width relates to the next by φ.
    """
    widths = []
    current = 1.0
    for _ in range(columns):
        widths.append(current)
        current *= PHI_CONJUGATE
    # Normalize to sum = 1
    total = sum(widths)
    return [w / total for w in widths]


def phi_dimensions(base_width: int) -> tuple:
    """Calculate φ-proportioned dimensions from width"""
    height = int(base_width / PHI)
    return base_width, height


def phi_padding(base: int = Spacing.MD) -> tuple:
    """
    Calculate asymmetric padding using φ.
    Returns (vertical, horizontal) where horizontal = vertical × φ
    """
    vertical = base
    horizontal = int(base * PHI)
    return vertical, horizontal


# ══════════════════════════════════════════════════════════════════════════════
# TTK STYLE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def configure_golden_theme(root: tk.Tk) -> ttk.Style:
    """
    Configure ttk styles with Golden Swiss design.
    Call this once at app startup.
    """
    style = ttk.Style(root)
    
    # Use clam as base (most customizable)
    style.theme_use('clam')
    
    # ─────────────────────────────────────────────────────────────────────
    # FRAME STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TFrame',
        background=Colors.BG_BASE
    )
    
    style.configure('Card.TFrame',
        background=Colors.BG_ELEVATED,
        relief='flat'
    )
    
    style.configure('Surface.TFrame',
        background=Colors.BG_SURFACE
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # LABEL STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TLabel',
        background=Colors.BG_BASE,
        foreground=Colors.TEXT_PRIMARY,
        font=(Typography.FAMILY_SANS[0], FontSize.BODY)
    )
    
    style.configure('Heading.TLabel',
        background=Colors.BG_BASE,
        foreground=Colors.TEXT_PRIMARY,
        font=(Typography.FAMILY_SANS[0], FontSize.H4, 'bold')
    )
    
    style.configure('Caption.TLabel',
        background=Colors.BG_BASE,
        foreground=Colors.TEXT_SECONDARY,
        font=(Typography.FAMILY_SANS[0], FontSize.CAPTION)
    )
    
    style.configure('Gold.TLabel',
        background=Colors.BG_BASE,
        foreground=Colors.GOLD,
        font=(Typography.FAMILY_SANS[0], FontSize.BODY)
    )
    
    style.configure('Mono.TLabel',
        background=Colors.BG_BASE,
        foreground=Colors.TEXT_SECONDARY,
        font=(Typography.FAMILY_MONO[0], FontSize.BODY_SM)
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # BUTTON STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TButton',
        background=Colors.BG_SURFACE,
        foreground=Colors.TEXT_PRIMARY,
        borderwidth=0,
        focuscolor=Colors.GOLD,
        font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM, 'medium'),
        padding=(Spacing.LG, Spacing.MD)  # φ ratio: 13/8 ≈ 1.625 ≈ φ
    )
    
    style.map('TButton',
        background=[
            ('active', Colors.BG_ACTIVE),
            ('pressed', Colors.BG_HOVER),
            ('disabled', Colors.BG_ELEVATED)
        ],
        foreground=[
            ('disabled', Colors.TEXT_TERTIARY)
        ]
    )
    
    # Primary button (gold accent)
    style.configure('Primary.TButton',
        background=Colors.GOLD,
        foreground=Colors.TEXT_INVERSE,
        font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM, 'bold')
    )
    
    style.map('Primary.TButton',
        background=[
            ('active', Colors.GOLD_LIGHT),
            ('pressed', Colors.GOLD_DARK),
            ('disabled', Colors.BG_SURFACE)
        ]
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # ENTRY STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TEntry',
        fieldbackground=Colors.BG_SURFACE,
        foreground=Colors.TEXT_PRIMARY,
        insertcolor=Colors.GOLD,
        borderwidth=1,
        relief='flat',
        padding=(Spacing.MD, Spacing.SM),
        font=(Typography.FAMILY_MONO[0], FontSize.BODY)
    )
    
    style.map('TEntry',
        fieldbackground=[
            ('focus', Colors.BG_HOVER),
            ('disabled', Colors.BG_ELEVATED)
        ],
        bordercolor=[
            ('focus', Colors.GOLD),
            ('!focus', Colors.BORDER_DEFAULT)
        ]
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # SCALE (SLIDER) STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TScale',
        background=Colors.BG_BASE,
        troughcolor=Colors.BG_SURFACE,
        sliderrelief='flat',
        sliderlength=Spacing.XL  # 21px - Fibonacci
    )
    
    style.configure('Horizontal.TScale',
        sliderlength=Spacing.XL
    )
    
    # Gold accent scale
    style.configure('Gold.Horizontal.TScale',
        background=Colors.GOLD,
        troughcolor=Colors.GOLD_SUBTLE
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # PROGRESSBAR STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TProgressbar',
        background=Colors.GOLD,
        troughcolor=Colors.BG_SURFACE,
        borderwidth=0,
        thickness=Spacing.SM  # 5px - subtle
    )
    
    style.configure('Horizontal.TProgressbar',
        thickness=Spacing.SM
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # NOTEBOOK (TABS) STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TNotebook',
        background=Colors.BG_BASE,
        borderwidth=0,
        tabmargins=(Spacing.XS, Spacing.XS, Spacing.XS, 0)
    )
    
    style.configure('TNotebook.Tab',
        background=Colors.BG_ELEVATED,
        foreground=Colors.TEXT_SECONDARY,
        padding=(Spacing.XL, Spacing.MD),  # 21×8 ≈ φ ratio
        font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM)
    )
    
    style.map('TNotebook.Tab',
        background=[
            ('selected', Colors.BG_BASE),
            ('active', Colors.BG_SURFACE)
        ],
        foreground=[
            ('selected', Colors.GOLD),
            ('active', Colors.TEXT_PRIMARY)
        ],
        expand=[
            ('selected', (0, 0, 0, 2))
        ]
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # LABELFRAME STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TLabelframe',
        background=Colors.BG_ELEVATED,
        borderwidth=1,
        relief='flat'
    )
    
    style.configure('TLabelframe.Label',
        background=Colors.BG_ELEVATED,
        foreground=Colors.GOLD,
        font=(Typography.FAMILY_SANS[0], FontSize.CAPTION, 'bold')
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # CHECKBUTTON & RADIOBUTTON STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TCheckbutton',
        background=Colors.BG_BASE,
        foreground=Colors.TEXT_PRIMARY,
        focuscolor=Colors.GOLD,
        font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM)
    )
    
    style.map('TCheckbutton',
        background=[
            ('active', Colors.BG_HOVER)
        ],
        indicatorcolor=[
            ('selected', Colors.GOLD),
            ('!selected', Colors.BG_SURFACE)
        ]
    )
    
    style.configure('TRadiobutton',
        background=Colors.BG_BASE,
        foreground=Colors.TEXT_PRIMARY,
        focuscolor=Colors.GOLD,
        font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM)
    )
    
    style.map('TRadiobutton',
        background=[
            ('active', Colors.BG_HOVER)
        ],
        indicatorcolor=[
            ('selected', Colors.GOLD),
            ('!selected', Colors.BG_SURFACE)
        ]
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # SEPARATOR STYLES
    # ─────────────────────────────────────────────────────────────────────
    style.configure('TSeparator',
        background=Colors.BORDER_SUBTLE
    )
    
    return style


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_rounded_rectangle(canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int,
                             radius: int = Radius.MD, **kwargs) -> int:
    """
    Draw a rounded rectangle on a canvas.
    Uses φ-based radius by default.
    """
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
        x1 + radius, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def golden_canvas(parent, base_width: int = 300, bg: str = Colors.BG_ELEVATED) -> tk.Canvas:
    """Create a canvas with φ proportions"""
    width = base_width
    height = int(base_width / PHI)
    return tk.Canvas(parent, width=width, height=height, bg=bg, 
                     highlightthickness=0, bd=0)


class GoldenCard(ttk.Frame):
    """
    A card container with φ-rounded corners and golden proportions.
    Uses canvas for true rounded corners.
    """
    
    def __init__(self, parent, title: str = None, **kwargs):
        super().__init__(parent, style='Card.TFrame', **kwargs)
        
        self.corner_radius = Radius.MD
        
        # Optional title
        if title:
            title_label = ttk.Label(self, text=title, style='Gold.TLabel',
                                   font=(Typography.FAMILY_SANS[0], FontSize.CAPTION, 'bold'))
            title_label.pack(anchor='w', padx=Spacing.LG, pady=(Spacing.MD, Spacing.SM))
        
        # Content frame
        self.content = ttk.Frame(self, style='Card.TFrame')
        self.content.pack(fill='both', expand=True, padx=Spacing.LG, pady=Spacing.MD)


class GoldenButton(tk.Canvas):
    """
    A button with true φ-rounded corners.
    """
    
    def __init__(self, parent, text: str, command=None, 
                 primary: bool = False, width: int = None, **kwargs):
        
        # Calculate dimensions using φ
        self.text = text
        self.command = command
        self.primary = primary
        
        # Measure text
        temp_label = ttk.Label(parent, text=text, 
                              font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM, 'medium'))
        temp_label.update_idletasks()
        text_width = temp_label.winfo_reqwidth()
        temp_label.destroy()
        
        # φ-based padding: horizontal = vertical × φ
        pad_v = Spacing.MD  # 8px
        pad_h = int(pad_v * PHI)  # 13px
        
        btn_width = width or (text_width + 2 * pad_h)
        btn_height = FontSize.BODY_SM + 2 * pad_v
        
        super().__init__(parent, width=btn_width, height=btn_height,
                        bg=Colors.BG_BASE, highlightthickness=0, **kwargs)
        
        self.btn_width = btn_width
        self.btn_height = btn_height
        self.radius = Radius.SM
        
        # Colors based on style
        if primary:
            self.bg_normal = Colors.GOLD
            self.bg_hover = Colors.GOLD_LIGHT
            self.bg_active = Colors.GOLD_DARK
            self.fg = Colors.TEXT_INVERSE
        else:
            self.bg_normal = Colors.BG_SURFACE
            self.bg_hover = Colors.BG_HOVER
            self.bg_active = Colors.BG_ACTIVE
            self.fg = Colors.TEXT_PRIMARY
        
        self._draw(self.bg_normal)
        
        # Bindings
        self.bind('<Enter>', lambda e: self._draw(self.bg_hover))
        self.bind('<Leave>', lambda e: self._draw(self.bg_normal))
        self.bind('<Button-1>', lambda e: self._on_click())
        self.bind('<ButtonRelease-1>', lambda e: self._draw(self.bg_hover))
    
    def _draw(self, bg_color: str):
        """Redraw button with specified background"""
        self.delete('all')
        
        # Rounded rectangle background
        create_rounded_rectangle(self, 0, 0, self.btn_width, self.btn_height,
                                radius=self.radius, fill=bg_color, outline='')
        
        # Text centered
        self.create_text(self.btn_width // 2, self.btn_height // 2,
                        text=self.text, fill=self.fg,
                        font=(Typography.FAMILY_SANS[0], FontSize.BODY_SM, 'medium'))
    
    def _on_click(self):
        """Handle click"""
        self._draw(self.bg_active)
        if self.command:
            self.command()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_golden_geometry(window: tk.Tk, base_width: int = 987):
    """
    Apply φ-proportioned window geometry.
    Default width 987 (Fibonacci) → height 610 (≈987/φ)
    """
    width = base_width
    height = int(base_width / PHI)
    
    # Center on screen
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.configure(bg=Colors.BG_BASE)


def setup_golden_app(root: tk.Tk, title: str = "Golden Studio"):
    """
    Complete golden theme setup for app.
    Call this at app startup.
    """
    root.title(title)
    root.configure(bg=Colors.BG_BASE)
    
    # Apply φ geometry
    apply_golden_geometry(root)
    
    # Configure ttk styles
    style = configure_golden_theme(root)
    
    return style


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    style = setup_golden_app(root, "Golden Swiss Theme Demo")
    
    # Demo content
    main = ttk.Frame(root, padding=Spacing.XL)
    main.pack(fill='both', expand=True)
    
    # Heading
    ttk.Label(main, text="φ Golden Swiss Theme", 
             style='Heading.TLabel').pack(anchor='w', pady=(0, Spacing.LG))
    
    # Caption
    ttk.Label(main, text="Swiss typography · Golden proportions · Fibonacci spacing",
             style='Caption.TLabel').pack(anchor='w', pady=(0, Spacing.XXL))
    
    # Card demo
    card = GoldenCard(main, title="FREQUENCY CONTROL")
    card.pack(fill='x', pady=Spacing.MD)
    
    ttk.Label(card.content, text="Base Frequency", 
             style='Caption.TLabel').pack(anchor='w')
    
    freq_var = tk.DoubleVar(value=432.0)
    ttk.Scale(card.content, from_=20, to=500, variable=freq_var,
             orient='horizontal').pack(fill='x', pady=Spacing.SM)
    
    ttk.Label(card.content, text="432.0 Hz", 
             style='Mono.TLabel').pack(anchor='e')
    
    # Buttons
    btn_frame = ttk.Frame(main)
    btn_frame.pack(fill='x', pady=Spacing.XL)
    
    GoldenButton(btn_frame, "Secondary", command=lambda: print("Secondary")).pack(side='left', padx=Spacing.SM)
    GoldenButton(btn_frame, "▶ Play", command=lambda: print("Play"), primary=True).pack(side='left', padx=Spacing.SM)
    
    # Progress
    ttk.Progressbar(main, value=61.8, mode='determinate').pack(fill='x', pady=Spacing.LG)
    
    root.mainloop()
