"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CONTROL PANEL - Configuration Widget                      ║
║                                                                              ║
║   Sidebar with plate configuration controls:                                 ║
║   • Dimensions (length, width, thickness)                                    ║
║   • Material selection                                                        ║
║   • Shape selection                                                          ║
║   • Analysis controls                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from core.materials import MATERIALS

# Import state type
from ..viewmodel import PlateState


class ControlPanel(ttk.Frame):
    """
    Sidebar control panel for plate configuration.
    """
    
    def __init__(
        self,
        parent,
        on_dimension_change: Callable[[float, float, float], None] = None,
        on_material_change: Callable[[str], None] = None,
        on_analyze: Callable[[], None] = None,
    ):
        super().__init__(parent)
        
        # Callbacks
        self._on_dimension_change = on_dimension_change
        self._on_material_change = on_material_change
        self._on_analyze = on_analyze
        
        # Variables
        self._length_var = tk.StringVar(value="1950")
        self._width_var = tk.StringVar(value="600")
        self._thickness_var = tk.StringVar(value="18")
        self._material_var = tk.StringVar(value="birch_plywood")
        
        # Build UI
        self._create_widgets()
    
    def _create_widgets(self):
        """Create control widgets."""
        
        # === Dimensions Section ===
        dim_frame = ttk.LabelFrame(self, text="📐 Dimensions", padding=10)
        dim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Length
        ttk.Label(dim_frame, text="Length (mm):").grid(
            row=0, column=0, sticky="w", pady=2
        )
        length_spin = ttk.Spinbox(
            dim_frame,
            from_=500, to=3000, increment=50,
            textvariable=self._length_var,
            width=10
        )
        length_spin.grid(row=0, column=1, pady=2)
        length_spin.bind("<Return>", self._on_dim_changed)
        length_spin.bind("<FocusOut>", self._on_dim_changed)
        
        # Width
        ttk.Label(dim_frame, text="Width (mm):").grid(
            row=1, column=0, sticky="w", pady=2
        )
        width_spin = ttk.Spinbox(
            dim_frame,
            from_=300, to=1500, increment=50,
            textvariable=self._width_var,
            width=10
        )
        width_spin.grid(row=1, column=1, pady=2)
        width_spin.bind("<Return>", self._on_dim_changed)
        width_spin.bind("<FocusOut>", self._on_dim_changed)
        
        # Thickness
        ttk.Label(dim_frame, text="Thickness (mm):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        thick_spin = ttk.Spinbox(
            dim_frame,
            from_=6, to=50, increment=1,
            textvariable=self._thickness_var,
            width=10
        )
        thick_spin.grid(row=2, column=1, pady=2)
        thick_spin.bind("<Return>", self._on_dim_changed)
        thick_spin.bind("<FocusOut>", self._on_dim_changed)
        
        # Golden ratio button
        ttk.Button(
            dim_frame,
            text="φ Golden Ratio",
            command=self._apply_golden_ratio
        ).grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        
        # === Material Section ===
        mat_frame = ttk.LabelFrame(self, text="🪵 Material", padding=10)
        mat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Material combobox
        materials = list(MATERIALS.keys())
        mat_combo = ttk.Combobox(
            mat_frame,
            textvariable=self._material_var,
            values=materials,
            state="readonly",
            width=18
        )
        mat_combo.pack(fill=tk.X, pady=2)
        mat_combo.bind("<<ComboboxSelected>>", self._on_material_selected)
        
        # Material info
        self._mat_info_label = ttk.Label(
            mat_frame,
            text="",
            font=("SF Pro", 9),
            wraplength=200
        )
        self._mat_info_label.pack(fill=tk.X, pady=5)
        
        # === Shape Section ===
        shape_frame = ttk.LabelFrame(self, text="🔷 Shape", padding=10)
        shape_frame.pack(fill=tk.X, padx=5, pady=5)
        
        shapes = [
            ("Rectangle", "rectangle"),
            ("Ellipse", "ellipse"),
            ("Golden Ovoid", "golden_ovoid"),
        ]
        
        self._shape_var = tk.StringVar(value="rectangle")
        
        for text, value in shapes:
            ttk.Radiobutton(
                shape_frame,
                text=text,
                variable=self._shape_var,
                value=value
            ).pack(anchor="w")
        
        # === Analysis Button ===
        ttk.Button(
            self,
            text="⚡ Analyze Modes",
            command=self._on_analyze_clicked
        ).pack(fill=tk.X, padx=5, pady=10)
        
        # === Info Section ===
        self._info_frame = ttk.LabelFrame(self, text="ℹ️ Info", padding=10)
        self._info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self._info_label = ttk.Label(
            self._info_frame,
            text="Ready",
            font=("SF Pro", 9),
            wraplength=200
        )
        self._info_label.pack(fill=tk.X)
        
        # Update material info
        self._update_material_info()
    
    def update_state(self, state: PlateState):
        """Update controls from state."""
        # Update dimensions (convert to mm)
        self._length_var.set(str(int(state.length * 1000)))
        self._width_var.set(str(int(state.width * 1000)))
        self._thickness_var.set(str(int(state.thickness * 1000)))
        
        # Update material
        self._material_var.set(state.material_key)
        self._update_material_info()
        
        # Update info
        if state.fem_result:
            n_modes = len(state.fem_result.modes)
            freqs = state.fem_result.frequencies
            info = f"Found {n_modes} modes\n"
            info += f"Range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz"
            self._info_label.config(text=info)
        elif state.is_computing:
            self._info_label.config(text="Computing...")
        elif state.error_message:
            self._info_label.config(text=f"Error: {state.error_message}")
    
    def _on_dim_changed(self, event=None):
        """Dimension value changed."""
        try:
            length = float(self._length_var.get()) / 1000  # mm to m
            width = float(self._width_var.get()) / 1000
            thickness = float(self._thickness_var.get()) / 1000
            
            if self._on_dimension_change:
                self._on_dimension_change(length, width, thickness)
        except ValueError:
            pass
    
    def _on_material_selected(self, event=None):
        """Material selected."""
        material = self._material_var.get()
        self._update_material_info()
        
        if self._on_material_change:
            self._on_material_change(material)
    
    def _on_analyze_clicked(self):
        """Analyze button clicked."""
        if self._on_analyze:
            self._on_analyze()
    
    def _apply_golden_ratio(self):
        """Apply golden ratio to dimensions."""
        PHI = 1.618033988749895
        
        try:
            length = float(self._length_var.get())
            width = length / PHI
            self._width_var.set(str(int(width)))
            self._on_dim_changed()
        except ValueError:
            pass
    
    def _update_material_info(self):
        """Update material info label."""
        mat_key = self._material_var.get()
        if mat_key in MATERIALS:
            mat = MATERIALS[mat_key]
            info = f"ρ={mat.density:.0f} kg/m³\n"
            info += f"E={mat.E_mean/1e9:.1f} GPa\n"
            info += f"ζ={mat.damping_ratio:.3f}"
            self._mat_info_label.config(text=info)
