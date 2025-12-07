"""
UI module - User interface components for Golden Studio

Design System: Golden Swiss Typography
- Swiss/International Typographic Style (clean, grid-based, minimal)
- All proportions based on Golden Ratio (φ = 1.618033988749895)
- Rounded corners using φ-derived radii (Fibonacci: 3, 5, 8, 13, 21...)
- Font scale: 11, 13, 16, 21, 26, 34, 42, 55... (×φ progression)
- Spacing: Fibonacci sequence (1, 2, 3, 5, 8, 13, 21, 34, 55...)
"""

from .golden_theme import (
    # Constants
    PHI, PHI_CONJUGATE, FIB,
    # Design tokens
    Spacing, Radius, FontSize, Colors, Typography,
    # Theme setup
    configure_golden_theme, setup_golden_app, apply_golden_geometry,
    # Utilities
    golden_grid, phi_dimensions, phi_padding,
    create_rounded_rectangle, golden_canvas,
    # Custom widgets
    GoldenCard, GoldenButton
)

from .vibroacoustic_tab import VibroacousticTab

__all__ = [
    # Theme
    'PHI', 'PHI_CONJUGATE', 'FIB',
    'Spacing', 'Radius', 'FontSize', 'Colors', 'Typography',
    'configure_golden_theme', 'setup_golden_app', 'apply_golden_geometry',
    'golden_grid', 'phi_dimensions', 'phi_padding',
    'create_rounded_rectangle', 'golden_canvas',
    'GoldenCard', 'GoldenButton',
    # Tabs
    'VibroacousticTab'
]
