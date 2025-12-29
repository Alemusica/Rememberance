"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLATE LAB WIDGETS - Reusable UI Components               ║
║                                                                              ║
║   Contains scrollable sidebar, parameter panels, and canvas components      ║
║   for the Plate Lab vibroacoustic plate designer.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Tuple, Dict, Any

# Import theme
try:
    from ui.theme import STYLE, PlateLabStyle, configure_ttk_style
except ImportError:
    from theme import STYLE, PlateLabStyle, configure_ttk_style


# ══════════════════════════════════════════════════════════════════════════════
# SCROLLABLE SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

class ScrollableSidebar(tk.Frame):
    """
    A scrollable sidebar panel with vertical scrolling support.
    
    Solves the issue of content overflowing in the sidebar by adding
    a scrollbar that appears when content exceeds the visible area.
    """
    
    def __init__(self, parent, width: int = 320, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.style = STYLE
        self.configure(bg=self.style.BG_MEDIUM)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(
            self,
            width=width - 20,  # Leave room for scrollbar
            bg=self.style.BG_MEDIUM,
            highlightthickness=0,
        )
        
        # Scrollbar
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )
        
        # Inner frame that holds all content
        self.inner_frame = tk.Frame(
            self.canvas,
            bg=self.style.BG_MEDIUM,
        )
        
        # Window on canvas
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.inner_frame,
            anchor="nw",
            width=width - 20
        )
        
        # Configure scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack elements
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind events
        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Enable mouse wheel scrolling
        self.bind_mousewheel()
    
    def _on_frame_configure(self, event):
        """Update scroll region when inner frame changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Adjust inner frame width when canvas resizes."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def bind_mousewheel(self):
        """Bind mouse wheel scrolling."""
        def _on_mousewheel(event):
            # Cross-platform wheel handling
            if event.num == 5 or event.delta < 0:
                self.canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                self.canvas.yview_scroll(-1, "units")
        
        # Linux/macOS
        self.canvas.bind("<Button-4>", _on_mousewheel)
        self.canvas.bind("<Button-5>", _on_mousewheel)
        # Windows
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Also bind to inner frame
        self.inner_frame.bind("<Button-4>", _on_mousewheel)
        self.inner_frame.bind("<Button-5>", _on_mousewheel)
        self.inner_frame.bind("<MouseWheel>", _on_mousewheel)
    
    def scroll_to_top(self):
        """Scroll to the top of the sidebar."""
        self.canvas.yview_moveto(0)
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the sidebar."""
        self.canvas.yview_moveto(1)
    
    @property
    def content(self) -> tk.Frame:
        """Return the inner frame where content should be added."""
        return self.inner_frame


# ══════════════════════════════════════════════════════════════════════════════
# COLLAPSIBLE SECTION
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleSection(tk.Frame):
    """
    A collapsible section with a header that can be clicked to show/hide content.
    """
    
    def __init__(self, parent, title: str, expanded: bool = True, 
                 icon: str = "▼", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.style = STYLE
        self.configure(bg=self.style.BG_MEDIUM)
        
        self.expanded = expanded
        self.icon_expanded = icon
        self.icon_collapsed = "▶"
        
        # Header
        self.header = tk.Frame(self, bg=self.style.BG_LIGHT)
        self.header.pack(fill=tk.X, pady=(0, 1))
        
        # Toggle button
        self.toggle_btn = tk.Button(
            self.header,
            text=f"{self.icon_expanded if expanded else self.icon_collapsed} {title}",
            bg=self.style.BG_LIGHT,
            fg=self.style.GOLD,
            font=self.style.font_heading,
            relief="flat",
            anchor="w",
            cursor="hand2",
            command=self.toggle
        )
        self.toggle_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Content frame
        self.content_frame = tk.Frame(self, bg=self.style.BG_MEDIUM)
        if expanded:
            self.content_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.title = title
    
    def toggle(self):
        """Toggle the section visibility."""
        self.expanded = not self.expanded
        
        if self.expanded:
            self.content_frame.pack(fill=tk.X, padx=5, pady=5)
            self.toggle_btn.config(text=f"{self.icon_expanded} {self.title}")
        else:
            self.content_frame.pack_forget()
            self.toggle_btn.config(text=f"{self.icon_collapsed} {self.title}")
    
    def expand(self):
        """Expand the section."""
        if not self.expanded:
            self.toggle()
    
    def collapse(self):
        """Collapse the section."""
        if self.expanded:
            self.toggle()
    
    @property
    def content(self) -> tk.Frame:
        """Return the content frame."""
        return self.content_frame


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER ENTRY
# ══════════════════════════════════════════════════════════════════════════════

class ParameterEntry(tk.Frame):
    """
    A labeled parameter entry with optional units and validation.
    """
    
    def __init__(self, parent, label: str, default: str = "", 
                 units: str = "", width: int = 8,
                 validate_numeric: bool = True,
                 on_change: Optional[Callable] = None,
                 **kwargs):
        super().__init__(parent, **kwargs)
        
        self.style = STYLE
        self.configure(bg=self.style.BG_MEDIUM)
        self.on_change = on_change
        
        # Label
        self.label = tk.Label(
            self,
            text=f"{label}:",
            bg=self.style.BG_MEDIUM,
            fg=self.style.TEXT_PRIMARY,
            font=self.style.font_normal
        )
        self.label.pack(side=tk.LEFT)
        
        # Entry
        self.var = tk.StringVar(value=default)
        self.entry = tk.Entry(
            self,
            textvariable=self.var,
            width=width,
            **self.style.get_entry_style()
        )
        self.entry.pack(side=tk.LEFT, padx=5)
        
        # Units label
        if units:
            tk.Label(
                self,
                text=units,
                bg=self.style.BG_MEDIUM,
                fg=self.style.TEXT_MUTED,
                font=self.style.font_small
            ).pack(side=tk.LEFT)
        
        # Validation
        if validate_numeric:
            self.entry.bind("<FocusOut>", self._validate)
        
        if on_change:
            self.var.trace_add("write", lambda *args: on_change(self.get()))
    
    def _validate(self, event=None):
        """Validate numeric input."""
        try:
            float(self.var.get())
            self.entry.config(fg=self.style.TEXT_PRIMARY)
        except ValueError:
            self.entry.config(fg=self.style.ERROR)
    
    def get(self) -> str:
        """Get the current value."""
        return self.var.get()
    
    def set(self, value: str):
        """Set the value."""
        self.var.set(value)
    
    def get_float(self, default: float = 0.0) -> float:
        """Get value as float."""
        try:
            return float(self.var.get())
        except ValueError:
            return default


# ══════════════════════════════════════════════════════════════════════════════
# BUTTON GRID
# ══════════════════════════════════════════════════════════════════════════════

class ButtonGrid(tk.Frame):
    """
    A grid of buttons with uniform sizing.
    """
    
    def __init__(self, parent, buttons: List[Tuple[str, Callable]], 
                 columns: int = 3, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.style = STYLE
        self.configure(bg=self.style.BG_MEDIUM)
        
        for i, (text, command) in enumerate(buttons):
            row = i // columns
            col = i % columns
            
            btn = tk.Button(
                self,
                text=text,
                command=command,
                **self.style.get_button_style()
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        
        # Make columns expand equally
        for c in range(columns):
            self.columnconfigure(c, weight=1)


# ══════════════════════════════════════════════════════════════════════════════
# MODE LISTBOX
# ══════════════════════════════════════════════════════════════════════════════

class ModeListBox(tk.Frame):
    """
    A listbox for displaying and selecting vibration modes with scrollbar.
    """
    
    def __init__(self, parent, height: int = 8, 
                 on_select: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.style = STYLE
        self.configure(bg=self.style.BG_MEDIUM)
        self.on_select = on_select
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        self.listbox = tk.Listbox(
            self,
            height=height,
            yscrollcommand=scrollbar.set,
            **self.style.get_listbox_style()
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.listbox.yview)
        
        # Bind selection
        if on_select:
            self.listbox.bind("<<ListboxSelect>>", 
                              lambda e: on_select(self.get_selection()))
    
    def clear(self):
        """Clear all items."""
        self.listbox.delete(0, tk.END)
    
    def add_mode(self, index: int, frequency: float, mode_name: str = ""):
        """Add a mode to the list."""
        if mode_name:
            text = f"Modo {index+1}: {frequency:.1f} Hz ({mode_name})"
        else:
            text = f"Modo {index+1}: {frequency:.1f} Hz"
        self.listbox.insert(tk.END, text)
    
    def select(self, index: int):
        """Select a mode by index."""
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
    
    def get_selection(self) -> Optional[int]:
        """Get the selected index."""
        sel = self.listbox.curselection()
        return sel[0] if sel else None


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP CANVAS
# ══════════════════════════════════════════════════════════════════════════════

class HeatmapCanvas(tk.Canvas):
    """
    A canvas for displaying mode shape heatmaps with draggable exciters.
    """
    
    def __init__(self, parent, width: int = 600, height: int = 400,
                 on_exciter_move: Optional[Callable] = None,
                 on_click: Optional[Callable] = None,
                 **kwargs):
        self.style = STYLE
        
        super().__init__(
            parent,
            width=width,
            height=height,
            **self.style.get_canvas_style(),
            highlightthickness=2,
            highlightbackground=self.style.GOLD,
        )
        
        self.on_exciter_move = on_exciter_move
        self.on_click_callback = on_click
        self.margin = 20
        
        # Dragging state
        self.dragging_item = None
        self.drag_data = {}
        
        # Bind events
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_resize)
    
    def _on_click(self, event):
        """Handle click."""
        # Check if clicked on an exciter
        item = self.find_closest(event.x, event.y)
        if item:
            tags = self.gettags(item[0])
            if "exciter" in tags:
                self.dragging_item = item[0]
                self.drag_data = {
                    'x': event.x,
                    'y': event.y,
                    'item': item[0]
                }
                return
        
        if self.on_click_callback:
            # Convert to normalized coordinates
            norm_x = (event.x - self.margin) / (self.winfo_width() - 2 * self.margin)
            norm_y = 1 - (event.y - self.margin) / (self.winfo_height() - 2 * self.margin)
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))
            self.on_click_callback(norm_x, norm_y)
    
    def _on_drag(self, event):
        """Handle drag."""
        if self.dragging_item and self.on_exciter_move:
            norm_x = (event.x - self.margin) / (self.winfo_width() - 2 * self.margin)
            norm_y = 1 - (event.y - self.margin) / (self.winfo_height() - 2 * self.margin)
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))
            
            # Get exciter index from tag
            tags = self.gettags(self.dragging_item)
            for tag in tags:
                if tag.startswith("exc_"):
                    idx = int(tag.split("_")[1])
                    self.on_exciter_move(idx, norm_x, norm_y)
                    break
    
    def _on_release(self, event):
        """Handle release."""
        self.dragging_item = None
        self.drag_data = {}
    
    def _on_resize(self, event):
        """Handle resize."""
        # Will be handled by parent to redraw
        pass
    
    def draw_plate_outline(self, shape: str = "rectangle"):
        """Draw the plate outline."""
        m = self.margin
        w, h = self.winfo_width(), self.winfo_height()
        
        if shape == "ellipse":
            self.create_oval(
                m, m, w - m, h - m,
                outline=self.style.GOLD, width=2
            )
        else:
            self.create_rectangle(
                m, m, w - m, h - m,
                outline=self.style.GOLD, width=2
            )
    
    def draw_exciter(self, index: int, x: float, y: float, 
                     color: str = "#ff6b6b", coupling: float = 1.0):
        """
        Draw an exciter at normalized position (x, y).
        """
        m = self.margin
        w, h = self.winfo_width(), self.winfo_height()
        
        cx = m + x * (w - 2 * m)
        cy = m + (1 - y) * (h - 2 * m)
        r = 15
        
        # Main circle
        self.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="white", width=2,
            tags=("exciter", f"exc_{index}")
        )
        
        # Label
        self.create_text(
            cx, cy,
            text=str(index + 1),
            fill="white",
            font=("Arial", 10, "bold"),
            tags=("exciter", f"exc_{index}")
        )
        
        # Coupling arc
        coupling_color = self._coupling_to_color(coupling)
        self.create_arc(
            cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5,
            start=0, extent=360 * coupling,
            outline=coupling_color, width=3, style=tk.ARC
        )
    
    def _coupling_to_color(self, coupling: float) -> str:
        """Convert coupling value to color."""
        r = int(255 * (1 - coupling))
        g = int(255 * coupling)
        return f"#{r:02x}{g:02x}44"
    
    def draw_heatmap_pixel(self, x: int, y: int, size: int, value: float):
        """Draw a single heatmap pixel."""
        if value >= 0:
            r = int(255 * value)
            g = int(50 * (1 - value))
            b = int(50 * (1 - value))
        else:
            r = int(50 * (1 + value))
            g = int(50 * (1 + value))
            b = int(255 * (-value))
        
        color = f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
        self.create_rectangle(x, y, x + size, y + size, fill=color, outline="")


# ══════════════════════════════════════════════════════════════════════════════
# INFO PANEL
# ══════════════════════════════════════════════════════════════════════════════

class InfoPanel(tk.Frame):
    """
    A panel for displaying key-value information.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.style = STYLE
        self.configure(bg=self.style.BG_MEDIUM)
        
        self.labels: Dict[str, tk.Label] = {}
    
    def add_info(self, key: str, label: str, initial_value: str = "--"):
        """Add an info row."""
        frame = tk.Frame(self, bg=self.style.BG_MEDIUM)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(
            frame,
            text=f"{label}:",
            bg=self.style.BG_MEDIUM,
            fg=self.style.TEXT_SECONDARY,
            font=self.style.font_small
        ).pack(side=tk.LEFT)
        
        value_label = tk.Label(
            frame,
            text=initial_value,
            bg=self.style.BG_MEDIUM,
            fg=self.style.GOLD_LIGHT,
            font=self.style.font_mono
        )
        value_label.pack(side=tk.RIGHT)
        
        self.labels[key] = value_label
    
    def update_info(self, key: str, value: str):
        """Update an info value."""
        if key in self.labels:
            self.labels[key].config(text=value)


# ══════════════════════════════════════════════════════════════════════════════
# CHAKRA INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

class ChakraIndicator(tk.Canvas):
    """
    A vertical indicator showing the 7 chakras with frequency labels.
    """
    
    CHAKRAS = [
        ("Muladhara", 256.0, "#ff0000"),
        ("Svadhisthana", 288.0, "#ff8800"),
        ("Manipura", 320.0, "#ffff00"),
        ("Anahata", 341.3, "#00ff00"),
        ("Vishuddha", 384.0, "#00bfff"),
        ("Ajna", 426.7, "#4400ff"),
        ("Sahasrara", 480.0, "#ff00ff"),
    ]
    
    def __init__(self, parent, width: int = 30, height: int = 200, **kwargs):
        super().__init__(parent, width=width, height=height, 
                         bg=STYLE.BG_DARK, highlightthickness=0, **kwargs)
        
        self.active_chakra = -1
        self.draw()
    
    def draw(self):
        """Draw the chakra indicator."""
        self.delete("all")
        
        w = self.winfo_width() or 30
        h = self.winfo_height() or 200
        
        margin = 10
        usable_h = h - 2 * margin
        
        for i, (name, freq, color) in enumerate(self.CHAKRAS):
            y = margin + (1 - i / 6) * usable_h
            r = 8 if i == self.active_chakra else 5
            
            # Draw chakra circle
            self.create_oval(
                w/2 - r, y - r, w/2 + r, y + r,
                fill=color if i == self.active_chakra else "",
                outline=color, width=2
            )
            
            # Draw frequency label on hover
            if i == self.active_chakra:
                self.create_text(
                    w + 5, y,
                    text=f"{freq:.0f}Hz",
                    fill=color,
                    anchor="w",
                    font=("Arial", 8)
                )
    
    def set_active(self, freq: float):
        """Set active chakra based on frequency."""
        self.active_chakra = -1
        
        for i, (_, chakra_freq, _) in enumerate(self.CHAKRAS):
            if abs(freq - chakra_freq) < 20:
                self.active_chakra = i
                break
        
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Widget Test")
    root.geometry("800x600")
    root.configure(bg=STYLE.BG_DARK)
    
    # Configure ttk style
    style = ttk.Style()
    configure_ttk_style(style)
    
    # Test scrollable sidebar
    sidebar = ScrollableSidebar(root, width=300)
    sidebar.pack(side=tk.LEFT, fill=tk.Y)
    
    # Add some collapsible sections
    for i in range(5):
        section = CollapsibleSection(sidebar.content, f"Section {i+1}")
        section.pack(fill=tk.X, pady=2)
        
        for j in range(3):
            entry = ParameterEntry(section.content, f"Param {j+1}", 
                                   default="1.0", units="m")
            entry.pack(fill=tk.X)
    
    # Test canvas
    canvas = HeatmapCanvas(root)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    root.mainloop()
