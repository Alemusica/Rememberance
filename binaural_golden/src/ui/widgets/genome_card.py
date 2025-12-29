"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            GENOME CARD                                      ║
║                                                                              ║
║   Modern card for displaying plate genome information and fitness           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math

from ..theme import PlateLabStyle

STYLE = PlateLabStyle()


class ContourType(Enum):
    """Plate contour types."""
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    ROUNDED_RECT = "rounded_rect"
    CUSTOM = "custom"


@dataclass
class PlateGenome:
    """Simplified plate genome data structure."""
    length: float
    width: float
    thickness: float
    contour_type: ContourType = ContourType.ELLIPSE
    fitness_scores: Optional[Dict[str, float]] = None
    generation: int = 0
    id: str = ""
    
    def __post_init__(self):
        if self.fitness_scores is None:
            self.fitness_scores = {
                "Flatness": 0.0,
                "Spine": 0.0, 
                "Mass": 0.0,
                "Coupling": 0.0,
                "Total": 0.0
            }


@dataclass
class CardConfig:
    """Configuration for GenomeCard."""
    width: int = 280
    height: int = 200
    show_thumbnail: bool = True
    show_fitness_badge: bool = True
    show_dimensions: bool = True
    interactive: bool = True
    thumbnail_size: int = 80


class GenomeCard(tk.Frame):
    """
    Card moderna che mostra informazioni genome tavola.
    
    Features:
    - Miniatura forma tavola
    - Dimensioni (L × W × T) formattate
    - Fitness score badge colorato
    - Contour type icon
    - Hover effects
    - Click callbacks
    
    Example:
        genome = PlateGenome(
            length=2.0, width=1.0, thickness=0.02,
            fitness_scores={"Total": 0.85}
        )
        card = GenomeCard(
            parent, 
            genome=genome,
            on_click=lambda g: print(f"Clicked: {g.id}")
        )
    """
    
    def __init__(self, parent, *, genome: PlateGenome, 
                 config: Optional[CardConfig] = None,
                 on_click: Optional[Callable[[PlateGenome], None]] = None,
                 on_double_click: Optional[Callable[[PlateGenome], None]] = None):
        
        self.genome = genome
        self.config = config or CardConfig()
        self._on_click = on_click
        self._on_double_click = on_double_click
        self._is_selected = False
        self._is_hovered = False
        
        super().__init__(
            parent,
            width=self.config.width,
            height=self.config.height,
            bg=STYLE.BG_MEDIUM,
            relief="flat",
            borderwidth=2
        )
        
        # Prevent frame from shrinking
        self.pack_propagate(False)
        self.grid_propagate(False)
        
        # Create UI elements
        self._create_widgets()
        
        # Bind events
        self._bind_events()
    
    def _create_widgets(self):
        """Create card UI elements."""
        # Main container with padding
        container = tk.Frame(self, bg=STYLE.BG_MEDIUM)
        container.pack(fill="both", expand=True, padx=STYLE.PAD_MD, pady=STYLE.PAD_MD)
        
        # Header with generation and ID
        header_frame = tk.Frame(container, bg=STYLE.BG_MEDIUM)
        header_frame.pack(fill="x", pady=(0, STYLE.PAD_SM))
        
        # Generation badge
        gen_label = tk.Label(
            header_frame,
            text=f"Gen {self.genome.generation}",
            bg=STYLE.BG_HIGHLIGHT,
            fg=STYLE.TEXT_SECONDARY,
            font=STYLE.font_small,
            padx=STYLE.PAD_SM,
            pady=2
        )
        gen_label.pack(side="left")
        
        # Fitness badge
        if self.config.show_fitness_badge:
            self._create_fitness_badge(header_frame)
        
        # Content area
        content_frame = tk.Frame(container, bg=STYLE.BG_MEDIUM)
        content_frame.pack(fill="both", expand=True)
        
        # Thumbnail
        if self.config.show_thumbnail:
            self._create_thumbnail(content_frame)
        
        # Info panel
        info_frame = tk.Frame(content_frame, bg=STYLE.BG_MEDIUM)
        if self.config.show_thumbnail:
            info_frame.pack(side="right", fill="both", expand=True, padx=(STYLE.PAD_SM, 0))
        else:
            info_frame.pack(fill="both", expand=True)
        
        # Dimensions
        if self.config.show_dimensions:
            self._create_dimensions_info(info_frame)
        
        # Contour type
        self._create_contour_info(info_frame)
        
        # Fitness details (if space permits)
        if self.config.height > 150:
            self._create_fitness_details(info_frame)
    
    def _create_fitness_badge(self, parent):
        """Create fitness score badge."""
        total_fitness = self.genome.fitness_scores.get("Total", 0.0)
        
        # Color based on fitness level
        if total_fitness >= 0.8:
            badge_color = STYLE.SUCCESS
        elif total_fitness >= 0.6:
            badge_color = STYLE.GOLD
        elif total_fitness >= 0.4:
            badge_color = STYLE.WARNING
        else:
            badge_color = STYLE.ERROR
        
        badge = tk.Label(
            parent,
            text=f"{total_fitness:.2f}",
            bg=badge_color,
            fg=STYLE.TEXT_DARK,
            font=STYLE.font_bold,
            padx=STYLE.PAD_SM,
            pady=2
        )
        badge.pack(side="right")
    
    def _create_thumbnail(self, parent):
        """Create plate shape thumbnail."""
        thumbnail_frame = tk.Frame(parent, bg=STYLE.BG_MEDIUM)
        thumbnail_frame.pack(side="left", padx=(0, STYLE.PAD_SM))
        
        # Canvas for drawing plate shape
        self.thumbnail_canvas = tk.Canvas(
            thumbnail_frame,
            width=self.config.thumbnail_size,
            height=self.config.thumbnail_size,
            bg=STYLE.BG_DARK,
            highlightthickness=1,
            highlightbackground=STYLE.BG_LIGHT
        )
        self.thumbnail_canvas.pack()
        
        # Draw plate shape
        self._draw_plate_thumbnail()
    
    def _draw_plate_thumbnail(self):
        """Draw simplified plate shape in thumbnail."""
        canvas = self.thumbnail_canvas
        size = self.config.thumbnail_size
        margin = 10
        
        # Calculate scaling
        max_dim = max(self.genome.length, self.genome.width)
        scale = (size - 2 * margin) / max_dim
        
        width_px = self.genome.width * scale
        height_px = self.genome.length * scale
        
        # Center the shape
        cx = size // 2
        cy = size // 2
        x1 = cx - width_px // 2
        y1 = cy - height_px // 2
        x2 = cx + width_px // 2
        y2 = cy + height_px // 2
        
        # Draw based on contour type
        if self.genome.contour_type == ContourType.ELLIPSE:
            canvas.create_oval(
                x1, y1, x2, y2,
                fill=STYLE.PLATE_FILL,
                outline=STYLE.PLATE_STROKE,
                width=2,
                tags="plate"
            )
        else:  # Rectangle or rounded rectangle
            canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=STYLE.PLATE_FILL,
                outline=STYLE.PLATE_STROKE,
                width=2,
                tags="plate"
            )
        
        # Add thickness indicator (3D effect)
        offset = min(4, int(self.genome.thickness * 100))
        if offset > 1:
            canvas.create_line(
                x2, y1, x2 + offset, y1 - offset,
                fill=STYLE.PLATE_STROKE, width=2
            )
            canvas.create_line(
                x2, y2, x2 + offset, y2 - offset,
                fill=STYLE.PLATE_STROKE, width=2
            )
            canvas.create_line(
                x2 + offset, y1 - offset, x2 + offset, y2 - offset,
                fill=STYLE.PLATE_STROKE, width=2
            )
    
    def _create_dimensions_info(self, parent):
        """Create dimensions display."""
        dim_frame = tk.Frame(parent, bg=STYLE.BG_MEDIUM)
        dim_frame.pack(fill="x", pady=(0, STYLE.PAD_SM))
        
        tk.Label(
            dim_frame,
            text="Dimensions",
            bg=STYLE.BG_MEDIUM,
            fg=STYLE.TEXT_SECONDARY,
            font=STYLE.font_small
        ).pack(anchor="w")
        
        # L × W × T format
        dimensions_text = f"{self.genome.length:.2f} × {self.genome.width:.2f} × {self.genome.thickness:.3f} m"
        
        tk.Label(
            dim_frame,
            text=dimensions_text,
            bg=STYLE.BG_MEDIUM,
            fg=STYLE.TEXT_PRIMARY,
            font=STYLE.font_mono
        ).pack(anchor="w")
    
    def _create_contour_info(self, parent):
        """Create contour type info."""
        contour_frame = tk.Frame(parent, bg=STYLE.BG_MEDIUM)
        contour_frame.pack(fill="x", pady=(0, STYLE.PAD_SM))
        
        tk.Label(
            contour_frame,
            text="Shape",
            bg=STYLE.BG_MEDIUM,
            fg=STYLE.TEXT_SECONDARY,
            font=STYLE.font_small
        ).pack(anchor="w")
        
        # Contour type with icon
        contour_text = self.genome.contour_type.value.replace("_", " ").title()
        
        tk.Label(
            contour_frame,
            text=f"◯ {contour_text}" if self.genome.contour_type == ContourType.ELLIPSE else f"■ {contour_text}",
            bg=STYLE.BG_MEDIUM,
            fg=STYLE.TEXT_PRIMARY,
            font=STYLE.font_normal
        ).pack(anchor="w")
    
    def _create_fitness_details(self, parent):
        """Create detailed fitness scores."""
        fitness_frame = tk.Frame(parent, bg=STYLE.BG_MEDIUM)
        fitness_frame.pack(fill="both", expand=True, pady=(STYLE.PAD_SM, 0))
        
        tk.Label(
            fitness_frame,
            text="Fitness",
            bg=STYLE.BG_MEDIUM,
            fg=STYLE.TEXT_SECONDARY,
            font=STYLE.font_small
        ).pack(anchor="w")
        
        # Show top 3 fitness components
        fitness_items = [(k, v) for k, v in self.genome.fitness_scores.items() if k != "Total"]
        fitness_items.sort(key=lambda x: x[1], reverse=True)
        
        for label, score in fitness_items[:3]:
            score_frame = tk.Frame(fitness_frame, bg=STYLE.BG_MEDIUM)
            score_frame.pack(fill="x")
            
            tk.Label(
                score_frame,
                text=label,
                bg=STYLE.BG_MEDIUM,
                fg=STYLE.TEXT_SECONDARY,
                font=STYLE.font_small
            ).pack(side="left")
            
            tk.Label(
                score_frame,
                text=f"{score:.2f}",
                bg=STYLE.BG_MEDIUM,
                fg=STYLE.TEXT_PRIMARY,
                font=STYLE.font_mono
            ).pack(side="right")
    
    def _bind_events(self):
        """Bind mouse events to card and children."""
        def bind_recursive(widget):
            widget.bind("<Button-1>", self._on_click_event)
            widget.bind("<Double-Button-1>", self._on_double_click_event)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
            
            # Bind to all children recursively
            for child in widget.winfo_children():
                bind_recursive(child)
        
        bind_recursive(self)
    
    def _on_click_event(self, event):
        """Handle single click."""
        if self._on_click:
            self._on_click(self.genome)
    
    def _on_double_click_event(self, event):
        """Handle double click."""
        if self._on_double_click:
            self._on_double_click(self.genome)
    
    def _on_enter(self, event):
        """Handle mouse enter (hover)."""
        if not self.config.interactive:
            return
        
        self._is_hovered = True
        self._update_appearance()
    
    def _on_leave(self, event):
        """Handle mouse leave."""
        if not self.config.interactive:
            return
        
        self._is_hovered = False
        self._update_appearance()
    
    def _update_appearance(self):
        """Update card appearance based on state."""
        if self._is_selected:
            bg_color = STYLE.BG_HIGHLIGHT
            border_color = STYLE.GOLD
            border_width = 3
        elif self._is_hovered:
            bg_color = STYLE.BG_LIGHT
            border_color = STYLE.GOLD_LIGHT
            border_width = 2
        else:
            bg_color = STYLE.BG_MEDIUM
            border_color = STYLE.BG_LIGHT
            border_width = 1
        
        self.configure(
            bg=bg_color,
            highlightbackground=border_color,
            highlightthickness=border_width
        )
        
        # Update all child widgets
        def update_recursive(widget):
            try:
                if isinstance(widget, (tk.Frame, tk.Label)):
                    widget.configure(bg=bg_color)
            except tk.TclError:
                pass
            
            for child in widget.winfo_children():
                update_recursive(child)
        
        update_recursive(self)
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self._is_selected = selected
        self._update_appearance()
    
    def update_genome(self, genome: PlateGenome):
        """Update card with new genome data."""
        self.genome = genome
        
        # Clear and recreate widgets
        for child in self.winfo_children():
            child.destroy()
        
        self._create_widgets()
        self._bind_events()
        self._update_appearance()
