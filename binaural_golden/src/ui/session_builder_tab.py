"""
Session Builder Tab - Visual Session Designer
==============================================

Visual interface for building sound therapy sessions:
- Pie chart view: Steps as "slices of cake"
- Linear flowchart: Single-branch timeline
- Load/save presets
- Drag & drop step reordering

Design: Clean Swiss style with φ proportions
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import json
from typing import List, Optional, Callable
from dataclasses import dataclass

from .golden_theme import (
    Colors, Spacing, Radius, FontSize, Typography,
    PHI, create_rounded_rectangle, golden_canvas
)

# Try to import Program system
try:
    from programs import Program, Step, FrequencyConfig, PositionConfig, FadeCurve, BODY_POSITIONS
    from programs import orchestra_tuning, chakra_journey, binaural_sweep, harmonic_meditation
    PROGRAMS_AVAILABLE = True
except ImportError:
    PROGRAMS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# STEP COLORS - Distinct colors for different step types
# ══════════════════════════════════════════════════════════════════════════════

STEP_COLORS = [
    "#e74c3c",  # Red
    "#e67e22",  # Orange
    "#f1c40f",  # Yellow
    "#2ecc71",  # Green
    "#1abc9c",  # Teal
    "#3498db",  # Blue
    "#9b59b6",  # Purple
    "#e91e63",  # Pink
    "#00bcd4",  # Cyan
    "#8bc34a",  # Light Green
    "#ff9800",  # Amber
    "#673ab7",  # Deep Purple
]


def get_step_color(index: int) -> str:
    """Get color for step by index"""
    return STEP_COLORS[index % len(STEP_COLORS)]


# ══════════════════════════════════════════════════════════════════════════════
# PIE CHART WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class PieChartWidget(tk.Canvas):
    """
    Pie chart showing session steps as slices.
    Each slice represents a step's duration proportion.
    """
    
    def __init__(self, parent, size: int = 300, **kwargs):
        kwargs.setdefault('bg', Colors.BG_BASE)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, width=size, height=size, **kwargs)
        
        self.size = size
        self.steps: List[dict] = []
        self.selected_index: int = -1
        self.on_select: Optional[Callable[[int], None]] = None
        self.on_hover: Optional[Callable[[int], None]] = None
        
        # Interaction
        self.bind('<Button-1>', self._on_click)
        self.bind('<Motion>', self._on_motion)
        self.bind('<Leave>', self._on_leave)
        
        self._hover_index = -1
        
    def set_steps(self, steps: List[dict]):
        """
        Set steps to display.
        Each step dict should have: name, duration_sec
        """
        self.steps = steps
        self.redraw()
        
    def set_selected(self, index: int):
        """Set selected slice"""
        self.selected_index = index
        self.redraw()
        
    def redraw(self):
        """Redraw the pie chart"""
        self.delete('all')
        
        if not self.steps:
            # Empty state
            self._draw_empty()
            return
            
        cx = self.size / 2
        cy = self.size / 2
        radius = self.size / 2 - Spacing.XL
        
        # Calculate total duration
        total_duration = sum(s.get('duration_sec', 1) for s in self.steps)
        if total_duration == 0:
            total_duration = 1
            
        # Draw slices
        start_angle = 90  # Start from top
        
        for i, step in enumerate(self.steps):
            duration = step.get('duration_sec', 1)
            extent = (duration / total_duration) * 360
            
            color = get_step_color(i)
            
            # Highlight selected/hovered
            r = radius
            if i == self.selected_index:
                r = radius + 8  # Pop out selected
            elif i == self._hover_index:
                r = radius + 4
                
            # Draw slice
            x0 = cx - r
            y0 = cy - r
            x1 = cx + r
            y1 = cy + r
            
            # Determine outline
            outline = Colors.GOLD if i == self.selected_index else Colors.BG_ELEVATED
            outline_width = 3 if i == self.selected_index else 1
            
            self.create_arc(
                x0, y0, x1, y1,
                start=start_angle,
                extent=-extent,  # Clockwise
                fill=color,
                outline=outline,
                width=outline_width,
                tags=f'slice_{i}'
            )
            
            # Draw label at slice center
            if extent > 15:  # Only show label if slice is big enough
                label_angle = math.radians(start_angle - extent / 2)
                label_r = r * 0.65
                lx = cx + label_r * math.cos(label_angle)
                ly = cy - label_r * math.sin(label_angle)
                
                # Truncate name
                name = step.get('name', f'Step {i+1}')
                if len(name) > 10:
                    name = name[:8] + '...'
                    
                self.create_text(
                    lx, ly,
                    text=name,
                    fill=Colors.TEXT_PRIMARY,
                    font=(Typography.FONT_FAMILY, FontSize.CAPTION, 'bold'),
                    anchor='center',
                    tags=f'label_{i}'
                )
            
            start_angle -= extent
            
        # Draw center circle (donut style)
        inner_r = radius * 0.35
        self.create_oval(
            cx - inner_r, cy - inner_r,
            cx + inner_r, cy + inner_r,
            fill=Colors.BG_ELEVATED,
            outline=Colors.BORDER_DEFAULT,
            width=2
        )
        
        # Center text - total duration
        total_min = total_duration / 60
        if total_min >= 1:
            time_text = f"{total_min:.1f} min"
        else:
            time_text = f"{total_duration:.0f} sec"
            
        self.create_text(
            cx, cy - 8,
            text=time_text,
            fill=Colors.GOLD,
            font=(Typography.FONT_FAMILY, FontSize.BODY_LG, 'bold')
        )
        self.create_text(
            cx, cy + 12,
            text=f"{len(self.steps)} steps",
            fill=Colors.TEXT_SECONDARY,
            font=(Typography.FONT_FAMILY, FontSize.CAPTION)
        )
        
    def _draw_empty(self):
        """Draw empty state"""
        cx = self.size / 2
        cy = self.size / 2
        radius = self.size / 2 - Spacing.XL
        
        # Empty circle
        self.create_oval(
            cx - radius, cy - radius,
            cx + radius, cy + radius,
            fill=Colors.BG_ELEVATED,
            outline=Colors.BORDER_SUBTLE,
            width=2,
            dash=(8, 4)
        )
        
        self.create_text(
            cx, cy - 10,
            text="No Session",
            fill=Colors.TEXT_TERTIARY,
            font=(Typography.FONT_FAMILY, FontSize.BODY_LG)
        )
        self.create_text(
            cx, cy + 15,
            text="Load or create a program",
            fill=Colors.TEXT_TERTIARY,
            font=(Typography.FONT_FAMILY, FontSize.CAPTION)
        )
        
    def _get_slice_at(self, x: int, y: int) -> int:
        """Get slice index at coordinates"""
        if not self.steps:
            return -1
            
        cx = self.size / 2
        cy = self.size / 2
        
        # Calculate angle from center
        dx = x - cx
        dy = cy - y  # Invert Y for standard math
        
        if dx == 0 and dy == 0:
            return -1
            
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
            
        # Adjust for start angle (90°)
        angle = (90 - angle) % 360
        
        # Find slice
        total_duration = sum(s.get('duration_sec', 1) for s in self.steps)
        if total_duration == 0:
            return -1
            
        cumulative = 0
        for i, step in enumerate(self.steps):
            duration = step.get('duration_sec', 1)
            extent = (duration / total_duration) * 360
            
            if cumulative <= angle < cumulative + extent:
                # Check if within radius
                r = math.sqrt(dx**2 + dy**2)
                radius = self.size / 2 - Spacing.XL
                inner_r = radius * 0.35
                
                if inner_r < r < radius + 10:
                    return i
                    
            cumulative += extent
            
        return -1
        
    def _on_click(self, event):
        """Handle click"""
        index = self._get_slice_at(event.x, event.y)
        if index >= 0:
            self.selected_index = index
            self.redraw()
            if self.on_select:
                self.on_select(index)
                
    def _on_motion(self, event):
        """Handle mouse motion"""
        index = self._get_slice_at(event.x, event.y)
        if index != self._hover_index:
            self._hover_index = index
            self.redraw()
            if self.on_hover:
                self.on_hover(index)
                
    def _on_leave(self, event):
        """Handle mouse leave"""
        self._hover_index = -1
        self.redraw()


# ══════════════════════════════════════════════════════════════════════════════
# FLOW CHART (LINEAR TIMELINE) WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class FlowChartWidget(tk.Canvas):
    """
    Linear flowchart showing steps as connected nodes.
    Single branch - simple timeline view.
    """
    
    NODE_WIDTH = 120
    NODE_HEIGHT = 60
    NODE_SPACING = 30
    CONNECTOR_LENGTH = 40
    
    def __init__(self, parent, height: int = 100, **kwargs):
        kwargs.setdefault('bg', Colors.BG_BASE)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, height=height, **kwargs)
        
        self.steps: List[dict] = []
        self.selected_index: int = -1
        self.on_select: Optional[Callable[[int], None]] = None
        
        # Scrolling
        self.bind('<Button-1>', self._on_click)
        self.bind('<Configure>', lambda e: self.redraw())
        
        # Enable horizontal scrolling
        self.bind('<MouseWheel>', self._on_mousewheel)
        self.bind('<Shift-MouseWheel>', self._on_mousewheel)
        
    def set_steps(self, steps: List[dict]):
        """Set steps to display"""
        self.steps = steps
        self._update_scroll_region()
        self.redraw()
        
    def set_selected(self, index: int):
        """Set selected node"""
        self.selected_index = index
        self.redraw()
        
        # Scroll to selected
        if index >= 0 and self.steps:
            self._scroll_to_node(index)
            
    def _update_scroll_region(self):
        """Update canvas scroll region"""
        if not self.steps:
            self.configure(scrollregion=(0, 0, self.winfo_width(), 100))
            return
            
        total_width = (
            Spacing.XL +  # Left padding
            len(self.steps) * (self.NODE_WIDTH + self.CONNECTOR_LENGTH) +
            Spacing.XL  # Right padding
        )
        self.configure(scrollregion=(0, 0, total_width, 100))
        
    def _scroll_to_node(self, index: int):
        """Scroll to make node visible"""
        x = Spacing.XL + index * (self.NODE_WIDTH + self.CONNECTOR_LENGTH)
        visible_width = self.winfo_width()
        
        # Check if node is outside visible area
        scroll_x = self.canvasx(0)
        if x < scroll_x or x + self.NODE_WIDTH > scroll_x + visible_width:
            # Scroll to center the node
            target_x = x - visible_width / 2 + self.NODE_WIDTH / 2
            self.xview_moveto(max(0, target_x / self.winfo_reqwidth()))
            
    def redraw(self):
        """Redraw the flowchart"""
        self.delete('all')
        
        if not self.steps:
            self._draw_empty()
            return
            
        height = self.winfo_height()
        cy = height / 2
        
        # Starting position
        x = Spacing.XL
        
        for i, step in enumerate(self.steps):
            color = get_step_color(i)
            is_selected = i == self.selected_index
            
            # Draw connector (except for first)
            if i > 0:
                self.create_line(
                    x - self.CONNECTOR_LENGTH, cy,
                    x, cy,
                    fill=Colors.GOLD if is_selected else Colors.BORDER_DEFAULT,
                    width=3,
                    arrow=tk.LAST,
                    arrowshape=(8, 10, 4)
                )
                
            # Draw node background
            node_y = cy - self.NODE_HEIGHT / 2
            
            # Shadow for selected
            if is_selected:
                create_rounded_rectangle(
                    self,
                    x + 2, node_y + 2,
                    x + self.NODE_WIDTH + 2, node_y + self.NODE_HEIGHT + 2,
                    radius=Radius.MD,
                    fill=Colors.BG_DEEPEST,
                    outline=''
                )
                
            # Node
            outline_color = Colors.GOLD if is_selected else Colors.BORDER_DEFAULT
            outline_width = 3 if is_selected else 1
            
            create_rounded_rectangle(
                self,
                x, node_y,
                x + self.NODE_WIDTH, node_y + self.NODE_HEIGHT,
                radius=Radius.MD,
                fill=color,
                outline=outline_color,
                width=outline_width,
                tags=f'node_{i}'
            )
            
            # Step number badge
            badge_size = 20
            self.create_oval(
                x + 4, node_y + 4,
                x + 4 + badge_size, node_y + 4 + badge_size,
                fill=Colors.BG_BASE,
                outline=Colors.TEXT_PRIMARY,
                width=1
            )
            self.create_text(
                x + 4 + badge_size / 2, node_y + 4 + badge_size / 2,
                text=str(i + 1),
                fill=Colors.TEXT_PRIMARY,
                font=(Typography.FONT_FAMILY, FontSize.CAPTION, 'bold')
            )
            
            # Step name
            name = step.get('name', f'Step {i+1}')
            if len(name) > 14:
                name = name[:12] + '...'
                
            self.create_text(
                x + self.NODE_WIDTH / 2, cy - 5,
                text=name,
                fill=Colors.TEXT_INVERSE,
                font=(Typography.FONT_FAMILY, FontSize.CAPTION, 'bold'),
                width=self.NODE_WIDTH - 10,
                anchor='center'
            )
            
            # Duration
            duration = step.get('duration_sec', 0)
            if duration >= 60:
                dur_text = f"{duration/60:.1f}m"
            else:
                dur_text = f"{duration:.0f}s"
                
            self.create_text(
                x + self.NODE_WIDTH / 2, cy + 15,
                text=dur_text,
                fill=Colors.TEXT_INVERSE,
                font=(Typography.FONT_FAMILY, FontSize.CAPTION),
                anchor='center'
            )
            
            x += self.NODE_WIDTH + self.CONNECTOR_LENGTH
            
        # End marker
        self.create_oval(
            x - self.CONNECTOR_LENGTH + 5, cy - 8,
            x - self.CONNECTOR_LENGTH + 21, cy + 8,
            fill=Colors.GOLD,
            outline=Colors.GOLD_DARK
        )
        self.create_text(
            x - self.CONNECTOR_LENGTH + 13, cy,
            text="▶",
            fill=Colors.BG_BASE,
            font=(Typography.FONT_FAMILY, FontSize.CAPTION)
        )
        
    def _draw_empty(self):
        """Draw empty state"""
        height = self.winfo_height()
        width = self.winfo_width()
        
        self.create_text(
            width / 2, height / 2,
            text="No steps - Add or load a program",
            fill=Colors.TEXT_TERTIARY,
            font=(Typography.FONT_FAMILY, FontSize.BODY)
        )
        
    def _on_click(self, event):
        """Handle click"""
        x = self.canvasx(event.x)
        
        # Find clicked node
        node_x = Spacing.XL
        for i in range(len(self.steps)):
            if node_x <= x <= node_x + self.NODE_WIDTH:
                self.selected_index = i
                self.redraw()
                if self.on_select:
                    self.on_select(i)
                return
            node_x += self.NODE_WIDTH + self.CONNECTOR_LENGTH
            
    def _on_mousewheel(self, event):
        """Handle mousewheel for scrolling"""
        self.xview_scroll(int(-1 * (event.delta / 120)), "units")


# ══════════════════════════════════════════════════════════════════════════════
# STEP EDITOR PANEL
# ══════════════════════════════════════════════════════════════════════════════

class StepEditorPanel(ttk.Frame):
    """Panel for editing a single step"""
    
    def __init__(self, parent, on_change: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_change = on_change
        self.current_step: Optional[dict] = None
        self._build_ui()
        
    def _build_ui(self):
        """Build the editor UI"""
        # Title
        self.title_label = ttk.Label(
            self,
            text="Step Editor",
            font=(Typography.FONT_FAMILY, FontSize.H6, 'bold')
        )
        self.title_label.pack(pady=(0, Spacing.MD), anchor='w')
        
        # Form frame
        form = ttk.Frame(self)
        form.pack(fill='x')
        
        # Name
        ttk.Label(form, text="Name:").grid(row=0, column=0, sticky='w', pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(form, textvariable=self.name_var, width=25)
        self.name_entry.grid(row=0, column=1, sticky='ew', padx=(8, 0), pady=2)
        self.name_var.trace_add('write', self._on_edit)
        
        # Duration
        ttk.Label(form, text="Duration (s):").grid(row=1, column=0, sticky='w', pady=2)
        self.duration_var = tk.DoubleVar(value=10.0)
        self.duration_spin = ttk.Spinbox(
            form, 
            textvariable=self.duration_var,
            from_=1, to=600, increment=5,
            width=10
        )
        self.duration_spin.grid(row=1, column=1, sticky='w', padx=(8, 0), pady=2)
        self.duration_var.trace_add('write', self._on_edit)
        
        # Frequency (primary)
        ttk.Label(form, text="Frequency (Hz):").grid(row=2, column=0, sticky='w', pady=2)
        self.freq_var = tk.DoubleVar(value=432.0)
        self.freq_spin = ttk.Spinbox(
            form,
            textvariable=self.freq_var,
            from_=20, to=2000, increment=1,
            width=10
        )
        self.freq_spin.grid(row=2, column=1, sticky='w', padx=(8, 0), pady=2)
        self.freq_var.trace_add('write', self._on_edit)
        
        # Fade In
        ttk.Label(form, text="Fade In (s):").grid(row=3, column=0, sticky='w', pady=2)
        self.fade_in_var = tk.DoubleVar(value=2.0)
        self.fade_in_spin = ttk.Spinbox(
            form,
            textvariable=self.fade_in_var,
            from_=0, to=60, increment=0.5,
            width=10
        )
        self.fade_in_spin.grid(row=3, column=1, sticky='w', padx=(8, 0), pady=2)
        self.fade_in_var.trace_add('write', self._on_edit)
        
        # Fade Out
        ttk.Label(form, text="Fade Out (s):").grid(row=4, column=0, sticky='w', pady=2)
        self.fade_out_var = tk.DoubleVar(value=2.0)
        self.fade_out_spin = ttk.Spinbox(
            form,
            textvariable=self.fade_out_var,
            from_=0, to=60, increment=0.5,
            width=10
        )
        self.fade_out_spin.grid(row=4, column=1, sticky='w', padx=(8, 0), pady=2)
        self.fade_out_var.trace_add('write', self._on_edit)
        
        # Position
        ttk.Label(form, text="Body Position:").grid(row=5, column=0, sticky='w', pady=2)
        self.position_var = tk.StringVar(value='SOLAR_PLEXUS')
        positions = ['HEAD', 'THROAT', 'HEART', 'SOLAR_PLEXUS', 'SACRAL', 'ROOT', 'KNEES', 'FEET']
        self.position_combo = ttk.Combobox(
            form,
            textvariable=self.position_var,
            values=positions,
            state='readonly',
            width=15
        )
        self.position_combo.grid(row=5, column=1, sticky='w', padx=(8, 0), pady=2)
        self.position_var.trace_add('write', self._on_edit)
        
        form.columnconfigure(1, weight=1)
        
        # Description
        desc_frame = ttk.LabelFrame(self, text="Description", padding=Spacing.SM)
        desc_frame.pack(fill='x', pady=(Spacing.MD, 0))
        
        self.desc_text = tk.Text(desc_frame, height=3, wrap='word')
        self.desc_text.pack(fill='x')
        self.desc_text.bind('<KeyRelease>', self._on_edit)
        
        # Initial state
        self.set_step(None)
        
    def set_step(self, step: Optional[dict]):
        """Set step to edit"""
        self.current_step = step
        
        if step is None:
            self.name_entry.configure(state='disabled')
            self.duration_spin.configure(state='disabled')
            self.freq_spin.configure(state='disabled')
            self.fade_in_spin.configure(state='disabled')
            self.fade_out_spin.configure(state='disabled')
            self.position_combo.configure(state='disabled')
            self.desc_text.configure(state='disabled')
            self.title_label.configure(text="Step Editor")
            return
            
        # Enable all
        self.name_entry.configure(state='normal')
        self.duration_spin.configure(state='normal')
        self.freq_spin.configure(state='normal')
        self.fade_in_spin.configure(state='normal')
        self.fade_out_spin.configure(state='normal')
        self.position_combo.configure(state='readonly')
        self.desc_text.configure(state='normal')
        
        # Populate
        self.name_var.set(step.get('name', ''))
        self.duration_var.set(step.get('duration_sec', 10))
        self.fade_in_var.set(step.get('fade_in_sec', 0))
        self.fade_out_var.set(step.get('fade_out_sec', 0))
        
        # Get primary frequency
        freqs = step.get('frequencies', [])
        if freqs:
            self.freq_var.set(freqs[0].get('frequency_hz', 432))
        else:
            self.freq_var.set(432)
            
        # Get position
        positions = step.get('positions', [])
        if positions:
            self.position_var.set(positions[0].get('label', 'SOLAR_PLEXUS'))
        else:
            self.position_var.set('SOLAR_PLEXUS')
            
        # Description
        self.desc_text.delete('1.0', 'end')
        self.desc_text.insert('1.0', step.get('description', ''))
        
        self.title_label.configure(text=f"Editing: {step.get('name', 'Step')}")
        
    def get_step_data(self) -> dict:
        """Get current edited step data"""
        if self.current_step is None:
            return {}
            
        return {
            'name': self.name_var.get(),
            'duration_sec': self.duration_var.get(),
            'frequencies': [{
                'frequency_hz': self.freq_var.get(),
                'amplitude': 1.0,
                'phase_offset_rad': 0.0,
                'label': f"{self.freq_var.get():.0f} Hz"
            }],
            'positions': [{
                'position_mm': {'HEAD': 0, 'THROAT': 200, 'HEART': 450, 
                               'SOLAR_PLEXUS': 600, 'SACRAL': 800, 
                               'ROOT': 1000, 'KNEES': 1400, 'FEET': 1750
                              }.get(self.position_var.get(), 600),
                'label': self.position_var.get()
            }],
            'fade_in_sec': self.fade_in_var.get(),
            'fade_out_sec': self.fade_out_var.get(),
            'fade_curve': 'golden',
            'description': self.desc_text.get('1.0', 'end-1c')
        }
        
    def _on_edit(self, *args):
        """Called when any field changes"""
        if self.on_change and self.current_step is not None:
            self.on_change(self.get_step_data())


# ══════════════════════════════════════════════════════════════════════════════
# QUICK GENERATOR PANEL
# ══════════════════════════════════════════════════════════════════════════════

class QuickGeneratorPanel(ttk.LabelFrame):
    """Panel for quick program generation"""
    
    def __init__(self, parent, on_generate: Callable[[dict], None], **kwargs):
        super().__init__(parent, text="Quick Generator", padding=Spacing.MD, **kwargs)
        self.on_generate = on_generate
        self._build_ui()
        
    def _build_ui(self):
        """Build generator UI"""
        # Program type
        type_frame = ttk.Frame(self)
        type_frame.pack(fill='x', pady=(0, Spacing.SM))
        
        ttk.Label(type_frame, text="Type:").pack(side='left')
        self.type_var = tk.StringVar(value='orchestra')
        types = ['orchestra', 'chakra', 'binaural', 'harmonic']
        self.type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.type_var,
            values=types,
            state='readonly',
            width=12
        )
        self.type_combo.pack(side='left', padx=(8, 0))
        
        # Base frequency
        freq_frame = ttk.Frame(self)
        freq_frame.pack(fill='x', pady=Spacing.XS)
        
        ttk.Label(freq_frame, text="Base Hz:").pack(side='left')
        self.base_freq_var = tk.DoubleVar(value=432.0)
        freq_spin = ttk.Spinbox(
            freq_frame,
            textvariable=self.base_freq_var,
            from_=100, to=1000, increment=1,
            width=8
        )
        freq_spin.pack(side='left', padx=(8, 0))
        
        # Duration
        dur_frame = ttk.Frame(self)
        dur_frame.pack(fill='x', pady=Spacing.XS)
        
        ttk.Label(dur_frame, text="Duration:").pack(side='left')
        self.duration_var = tk.IntVar(value=5)
        dur_spin = ttk.Spinbox(
            dur_frame,
            textvariable=self.duration_var,
            from_=1, to=60, increment=1,
            width=5
        )
        dur_spin.pack(side='left', padx=(8, 0))
        ttk.Label(dur_frame, text="min").pack(side='left', padx=(4, 0))
        
        # Generate button
        gen_btn = ttk.Button(
            self,
            text="⚡ Generate",
            command=self._generate,
            style='Accent.TButton'
        )
        gen_btn.pack(fill='x', pady=(Spacing.MD, 0))
        
    def _generate(self):
        """Generate program"""
        if not PROGRAMS_AVAILABLE:
            messagebox.showerror("Error", "Program generators not available")
            return
            
        program_type = self.type_var.get()
        base_freq = self.base_freq_var.get()
        duration_min = self.duration_var.get()
        
        try:
            if program_type == 'orchestra':
                prog = orchestra_tuning(base_freq, duration_min=duration_min)
            elif program_type == 'chakra':
                prog = chakra_journey(duration_min=duration_min)
            elif program_type == 'binaural':
                prog = binaural_sweep(carrier=base_freq, duration_min=duration_min)
            elif program_type == 'harmonic':
                prog = harmonic_meditation(base_freq, duration_min=duration_min)
            else:
                return
                
            # Convert to dict
            self.on_generate(prog.to_dict())
            
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SESSION BUILDER TAB
# ══════════════════════════════════════════════════════════════════════════════

class SessionBuilderTab(ttk.Frame):
    """
    Main Session Builder tab combining all components.
    
    Layout:
    ┌─────────────────────────────────────────────────────┐
    │  Toolbar: Load | Save | Add | Delete | Generate    │
    ├────────────────────────┬────────────────────────────┤
    │                        │                            │
    │     Pie Chart          │      Step Editor           │
    │    (Session View)      │                            │
    │                        ├────────────────────────────┤
    │                        │   Quick Generator          │
    ├────────────────────────┴────────────────────────────┤
    │               Flow Chart (Timeline)                 │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.program_data: dict = {
            'name': 'New Session',
            'description': '',
            'steps': []
        }
        self.current_file: Optional[str] = None
        
        self._build_ui()
        self._update_views()
        
    def _build_ui(self):
        """Build the main UI"""
        # ─────────────────────────────────────────────────────────────────
        # Toolbar
        # ─────────────────────────────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', padx=Spacing.MD, pady=Spacing.MD)
        
        ttk.Button(toolbar, text="📂 Load", command=self._load_program).pack(side='left', padx=2)
        ttk.Button(toolbar, text="💾 Save", command=self._save_program).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', padx=8, fill='y')
        ttk.Button(toolbar, text="➕ Add Step", command=self._add_step).pack(side='left', padx=2)
        ttk.Button(toolbar, text="🗑️ Delete", command=self._delete_step).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', padx=8, fill='y')
        ttk.Button(toolbar, text="⬆️ Move Up", command=self._move_up).pack(side='left', padx=2)
        ttk.Button(toolbar, text="⬇️ Move Down", command=self._move_down).pack(side='left', padx=2)
        
        # Program name on right
        ttk.Label(toolbar, text="Session:").pack(side='right')
        self.name_var = tk.StringVar(value='New Session')
        name_entry = ttk.Entry(toolbar, textvariable=self.name_var, width=20)
        name_entry.pack(side='right', padx=(0, 8))
        self.name_var.trace_add('write', lambda *a: self._update_program_name())
        
        # ─────────────────────────────────────────────────────────────────
        # Main content area
        # ─────────────────────────────────────────────────────────────────
        content = ttk.Frame(self)
        content.pack(fill='both', expand=True, padx=Spacing.MD)
        
        # Left: Pie chart
        left_frame = ttk.Frame(content)
        left_frame.pack(side='left', fill='both', padx=(0, Spacing.MD))
        
        pie_label = ttk.Label(
            left_frame,
            text="Session Overview",
            font=(Typography.FONT_FAMILY, FontSize.H6, 'bold')
        )
        pie_label.pack(pady=(0, Spacing.SM))
        
        self.pie_chart = PieChartWidget(left_frame, size=280)
        self.pie_chart.pack()
        self.pie_chart.on_select = self._on_step_selected
        
        # Right: Editor + Generator
        right_frame = ttk.Frame(content)
        right_frame.pack(side='left', fill='both', expand=True)
        
        # Step editor
        self.step_editor = StepEditorPanel(right_frame, on_change=self._on_step_edited)
        self.step_editor.pack(fill='x', pady=(0, Spacing.MD))
        
        # Quick generator
        self.generator = QuickGeneratorPanel(right_frame, on_generate=self._on_program_generated)
        self.generator.pack(fill='x')
        
        # ─────────────────────────────────────────────────────────────────
        # Bottom: Flow chart
        # ─────────────────────────────────────────────────────────────────
        flow_frame = ttk.LabelFrame(self, text="Timeline", padding=Spacing.SM)
        flow_frame.pack(fill='x', padx=Spacing.MD, pady=Spacing.MD)
        
        # Scrollable flow chart
        self.flow_chart = FlowChartWidget(flow_frame, height=90)
        self.flow_chart.pack(fill='x')
        self.flow_chart.on_select = self._on_step_selected
        
        # Scrollbar
        h_scroll = ttk.Scrollbar(flow_frame, orient='horizontal', command=self.flow_chart.xview)
        h_scroll.pack(fill='x')
        self.flow_chart.configure(xscrollcommand=h_scroll.set)
        
    def _update_views(self):
        """Update all views with current data"""
        steps = self.program_data.get('steps', [])
        self.pie_chart.set_steps(steps)
        self.flow_chart.set_steps(steps)
        self.name_var.set(self.program_data.get('name', 'New Session'))
        
    def _on_step_selected(self, index: int):
        """Handle step selection"""
        self.pie_chart.set_selected(index)
        self.flow_chart.set_selected(index)
        
        steps = self.program_data.get('steps', [])
        if 0 <= index < len(steps):
            self.step_editor.set_step(steps[index])
        else:
            self.step_editor.set_step(None)
            
    def _on_step_edited(self, step_data: dict):
        """Handle step edit"""
        index = self.pie_chart.selected_index
        steps = self.program_data.get('steps', [])
        
        if 0 <= index < len(steps):
            steps[index].update(step_data)
            self._update_views()
            self.pie_chart.set_selected(index)
            self.flow_chart.set_selected(index)
            
    def _on_program_generated(self, program_data: dict):
        """Handle generated program"""
        self.program_data = program_data
        self.current_file = None
        self._update_views()
        self._on_step_selected(0)  # Select first step
        
    def _update_program_name(self):
        """Update program name"""
        self.program_data['name'] = self.name_var.get()
        
    def _load_program(self):
        """Load program from file"""
        filetypes = [('JSON files', '*.json'), ('All files', '*.*')]
        filename = filedialog.askopenfilename(
            title='Load Session',
            filetypes=filetypes,
            initialdir='.'
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.program_data = data
                self.current_file = filename
                self._update_views()
                if data.get('steps'):
                    self._on_step_selected(0)
            except Exception as e:
                messagebox.showerror("Load Error", str(e))
                
    def _save_program(self):
        """Save program to file"""
        filetypes = [('JSON files', '*.json')]
        filename = filedialog.asksaveasfilename(
            title='Save Session',
            filetypes=filetypes,
            defaultextension='.json',
            initialfile=self.program_data.get('name', 'session') + '.json'
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.program_data, f, indent=2)
                self.current_file = filename
                messagebox.showinfo("Saved", f"Session saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
                
    def _add_step(self):
        """Add a new step"""
        new_step = {
            'name': f'Step {len(self.program_data.get("steps", [])) + 1}',
            'duration_sec': 30.0,
            'frequencies': [{'frequency_hz': 432.0, 'amplitude': 1.0, 'phase_offset_rad': 0.0, 'label': '432 Hz'}],
            'positions': [{'position_mm': 600.0, 'label': 'SOLAR_PLEXUS'}],
            'fade_in_sec': 3.0,
            'fade_out_sec': 3.0,
            'fade_curve': 'golden',
            'description': ''
        }
        
        if 'steps' not in self.program_data:
            self.program_data['steps'] = []
            
        self.program_data['steps'].append(new_step)
        self._update_views()
        self._on_step_selected(len(self.program_data['steps']) - 1)
        
    def _delete_step(self):
        """Delete selected step"""
        index = self.pie_chart.selected_index
        steps = self.program_data.get('steps', [])
        
        if 0 <= index < len(steps):
            del steps[index]
            self._update_views()
            
            # Select another step
            if steps:
                self._on_step_selected(min(index, len(steps) - 1))
            else:
                self._on_step_selected(-1)
                
    def _move_up(self):
        """Move selected step up"""
        index = self.pie_chart.selected_index
        steps = self.program_data.get('steps', [])
        
        if index > 0:
            steps[index], steps[index - 1] = steps[index - 1], steps[index]
            self._update_views()
            self._on_step_selected(index - 1)
            
    def _move_down(self):
        """Move selected step down"""
        index = self.pie_chart.selected_index
        steps = self.program_data.get('steps', [])
        
        if 0 <= index < len(steps) - 1:
            steps[index], steps[index + 1] = steps[index + 1], steps[index]
            self._update_views()
            self._on_step_selected(index + 1)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Session Builder Test")
    root.geometry("900x700")
    root.configure(bg=Colors.BG_BASE)
    
    # Apply theme
    from golden_theme import configure_golden_theme
    configure_golden_theme(root)
    
    tab = SessionBuilderTab(root)
    tab.pack(fill='both', expand=True)
    
    root.mainloop()
