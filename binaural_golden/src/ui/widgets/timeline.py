"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EVOLUTION TIMELINE                                  ║
║                                                                              ║
║   Timeline widget for tracking evolution generations and fitness           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from typing import List, Optional, Callable, Tuple, Dict
from dataclasses import dataclass
import math

from ..theme import PlateLabStyle

STYLE = PlateLabStyle()


@dataclass
class GenerationData:
    """Data for a single generation."""
    generation: int
    best_fitness: float
    avg_fitness: float
    population_size: int
    timestamp: Optional[float] = None
    notes: str = ""


@dataclass
class TimelineConfig:
    """Configuration for EvolutionTimeline."""
    width: int = 800
    height: int = 120
    show_avg_fitness: bool = True
    show_population: bool = False
    interactive: bool = True
    auto_scale: bool = True
    margin: int = 40
    point_radius: int = 4
    line_width: int = 2


class EvolutionTimeline(tk.Canvas):
    """
    Timeline orizzontale per evoluzione genetica.
    
    Features:
    - Generazioni come punti sulla timeline
    - Best fitness come linea continua
    - Average fitness opzionale
    - Current position marker
    - Click per jump to generation
    - Zoom e pan orizzontale
    - Tooltip con dettagli generazione
    
    Example:
        timeline = EvolutionTimeline(
            parent,
            config=TimelineConfig(width=1000, show_avg_fitness=True)
        )
        
        # Add generation data
        timeline.add_generation(GenerationData(
            generation=0, best_fitness=0.45, avg_fitness=0.32
        ))
        timeline.add_generation(GenerationData(
            generation=1, best_fitness=0.52, avg_fitness=0.38
        ))
    """
    
    def __init__(self, parent, *, config: Optional[TimelineConfig] = None,
                 on_generation_click: Optional[Callable[[int], None]] = None,
                 on_range_select: Optional[Callable[[int, int], None]] = None):
        
        self.config = config or TimelineConfig()
        
        super().__init__(
            parent,
            width=self.config.width,
            height=self.config.height,
            bg=STYLE.BG_DARK,
            highlightthickness=0
        )
        
        # Data
        self._generations: List[GenerationData] = []
        self._current_generation = 0
        
        # View state
        self._view_start = 0
        self._view_end = 10  # Initial view window
        self._fitness_min = 0.0
        self._fitness_max = 1.0
        
        # Interaction state
        self._dragging = False
        self._drag_start_x = 0
        self._hover_generation = None
        self._selecting_range = False
        self._selection_start = None
        
        # Callbacks
        self._on_generation_click = on_generation_click
        self._on_range_select = on_range_select
        
        # Bind events
        self._bind_events()
        
        # Initial draw
        self._draw()
    
    def _bind_events(self):
        """Bind mouse events."""
        if self.config.interactive:
            self.bind("<Button-1>", self._on_click)
            self.bind("<B1-Motion>", self._on_drag)
            self.bind("<ButtonRelease-1>", self._on_release)
            self.bind("<Motion>", self._on_mouse_move)
            self.bind("<MouseWheel>", self._on_mouse_wheel)
            self.bind("<Button-4>", self._on_mouse_wheel)  # Linux
            self.bind("<Button-5>", self._on_mouse_wheel)  # Linux
            self.bind("<Leave>", self._on_mouse_leave)
            
            # Keyboard shortcuts
            self.focus_set()
            self.bind("<Key>", self._on_key_press)
    
    def _generation_to_x(self, generation: int) -> int:
        """Convert generation number to x coordinate."""
        if not self._generations:
            return self.config.margin
        
        view_range = max(1, self._view_end - self._view_start)
        plot_width = self.config.width - 2 * self.config.margin
        
        relative_pos = (generation - self._view_start) / view_range
        return int(self.config.margin + relative_pos * plot_width)
    
    def _x_to_generation(self, x: int) -> float:
        """Convert x coordinate to generation number."""
        plot_width = self.config.width - 2 * self.config.margin
        relative_pos = (x - self.config.margin) / plot_width
        
        view_range = self._view_end - self._view_start
        return self._view_start + relative_pos * view_range
    
    def _fitness_to_y(self, fitness: float) -> int:
        """Convert fitness value to y coordinate."""
        plot_height = self.config.height - 2 * self.config.margin
        
        if self._fitness_max <= self._fitness_min:
            return self.config.height - self.config.margin
        
        normalized = (fitness - self._fitness_min) / (self._fitness_max - self._fitness_min)
        normalized = max(0, min(1, normalized))
        
        # Flip Y coordinate (higher fitness = lower Y)
        return int(self.config.height - self.config.margin - normalized * plot_height)
    
    def _update_view_range(self):
        """Update view range based on data and auto-scale."""
        if not self._generations:
            return
        
        if self.config.auto_scale:
            # Update fitness range
            fitness_values = []
            for gen in self._generations:
                fitness_values.extend([gen.best_fitness, gen.avg_fitness])
            
            if fitness_values:
                self._fitness_min = min(fitness_values) * 0.95
                self._fitness_max = max(fitness_values) * 1.05
            
            # Update generation range
            max_gen = max(g.generation for g in self._generations)
            if self._view_end < max_gen:
                self._view_end = max_gen + 1
    
    def _draw(self):
        """Draw the complete timeline."""
        self.delete("all")
        
        if not self._generations:
            self._draw_empty_state()
            return
        
        self._update_view_range()
        
        # Draw axes
        self._draw_axes()
        
        # Draw fitness lines
        self._draw_fitness_lines()
        
        # Draw generation points
        self._draw_generation_points()
        
        # Draw current generation marker
        self._draw_current_marker()
        
        # Draw selection if active
        if self._selection_start is not None:
            self._draw_selection()
    
    def _draw_empty_state(self):
        """Draw empty state message."""
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        
        self.create_text(
            center_x, center_y,
            text="No generation data available",
            fill=STYLE.TEXT_MUTED,
            font=STYLE.font_normal,
            anchor="center",
            tags="empty"
        )
    
    def _draw_axes(self):
        """Draw coordinate axes and labels."""
        margin = self.config.margin
        width = self.config.width
        height = self.config.height
        
        # X-axis (bottom)
        self.create_line(
            margin, height - margin,
            width - margin, height - margin,
            fill=STYLE.BG_LIGHT, width=2,
            tags="axis"
        )
        
        # Y-axis (left)
        self.create_line(
            margin, margin,
            margin, height - margin,
            fill=STYLE.BG_LIGHT, width=2,
            tags="axis"
        )
        
        # X-axis labels (generations)
        gen_step = max(1, int((self._view_end - self._view_start) / 10))
        for gen in range(int(self._view_start), int(self._view_end) + 1, gen_step):
            x = self._generation_to_x(gen)
            
            # Tick mark
            self.create_line(
                x, height - margin - 5,
                x, height - margin + 5,
                fill=STYLE.TEXT_SECONDARY, width=1,
                tags="axis"
            )
            
            # Label
            self.create_text(
                x, height - margin + 15,
                text=str(gen),
                fill=STYLE.TEXT_SECONDARY,
                font=STYLE.font_small,
                anchor="center",
                tags="axis"
            )
        
        # Y-axis labels (fitness)
        fitness_step = (self._fitness_max - self._fitness_min) / 5
        if fitness_step > 0:
            for i in range(6):
                fitness = self._fitness_min + i * fitness_step
                y = self._fitness_to_y(fitness)
                
                # Tick mark
                self.create_line(
                    margin - 5, y,
                    margin + 5, y,
                    fill=STYLE.TEXT_SECONDARY, width=1,
                    tags="axis"
                )
                
                # Label
                self.create_text(
                    margin - 10, y,
                    text=f"{fitness:.2f}",
                    fill=STYLE.TEXT_SECONDARY,
                    font=STYLE.font_small,
                    anchor="e",
                    tags="axis"
                )
        
        # Axis labels
        self.create_text(
            width // 2, height - 5,
            text="Generation",
            fill=STYLE.TEXT_PRIMARY,
            font=STYLE.font_small,
            anchor="center",
            tags="axis"
        )
        
        self.create_text(
            5, height // 2,
            text="Fitness",
            fill=STYLE.TEXT_PRIMARY,
            font=STYLE.font_small,
            anchor="center",
            angle=90,
            tags="axis"
        )
    
    def _draw_fitness_lines(self):
        """Draw fitness evolution lines."""
        if len(self._generations) < 2:
            return
        
        # Best fitness line
        best_points = []
        avg_points = []
        
        for gen_data in self._generations:
            if self._view_start <= gen_data.generation <= self._view_end:
                x = self._generation_to_x(gen_data.generation)
                
                # Best fitness
                y_best = self._fitness_to_y(gen_data.best_fitness)
                best_points.extend([x, y_best])
                
                # Average fitness
                if self.config.show_avg_fitness:
                    y_avg = self._fitness_to_y(gen_data.avg_fitness)
                    avg_points.extend([x, y_avg])
        
        # Draw best fitness line
        if len(best_points) >= 4:
            self.create_line(
                best_points,
                fill=STYLE.GOLD,
                width=self.config.line_width,
                smooth=True,
                tags="best_fitness"
            )
        
        # Draw average fitness line
        if self.config.show_avg_fitness and len(avg_points) >= 4:
            self.create_line(
                avg_points,
                fill=STYLE.INFO + "80",  # Semi-transparent
                width=self.config.line_width,
                smooth=True,
                dash=(5, 3),
                tags="avg_fitness"
            )
    
    def _draw_generation_points(self):
        """Draw individual generation points."""
        radius = self.config.point_radius
        
        for gen_data in self._generations:
            if self._view_start <= gen_data.generation <= self._view_end:
                x = self._generation_to_x(gen_data.generation)
                y_best = self._fitness_to_y(gen_data.best_fitness)
                
                # Point color based on fitness improvement
                if gen_data.generation > 0:
                    prev_gen = next(
                        (g for g in self._generations if g.generation == gen_data.generation - 1),
                        None
                    )
                    if prev_gen and gen_data.best_fitness > prev_gen.best_fitness:
                        point_color = STYLE.SUCCESS  # Improved
                    else:
                        point_color = STYLE.GOLD  # No improvement
                else:
                    point_color = STYLE.GOLD
                
                # Highlight if hovering
                if gen_data.generation == self._hover_generation:
                    point_color = STYLE.GOLD_LIGHT
                    radius += 2
                
                # Draw point
                self.create_oval(
                    x - radius, y_best - radius,
                    x + radius, y_best + radius,
                    fill=point_color,
                    outline=STYLE.TEXT_PRIMARY,
                    width=1,
                    tags=f"point_{gen_data.generation}"
                )
                
                # Population size indicator (if enabled)
                if self.config.show_population:
                    pop_height = min(20, gen_data.population_size // 5)
                    
                    self.create_rectangle(
                        x - 3, self.config.height - self.config.margin,
                        x + 3, self.config.height - self.config.margin - pop_height,
                        fill=STYLE.BG_LIGHT,
                        outline="",
                        tags=f"population_{gen_data.generation}"
                    )
    
    def _draw_current_marker(self):
        """Draw current generation marker."""
        x = self._generation_to_x(self._current_generation)
        
        # Vertical line
        self.create_line(
            x, self.config.margin,
            x, self.config.height - self.config.margin,
            fill=STYLE.ERROR,
            width=3,
            tags="current_marker"
        )
        
        # Triangle marker at top
        triangle_size = 8
        self.create_polygon(
            x, self.config.margin - 5,
            x - triangle_size, self.config.margin - triangle_size - 5,
            x + triangle_size, self.config.margin - triangle_size - 5,
            fill=STYLE.ERROR,
            outline=STYLE.TEXT_PRIMARY,
            tags="current_marker"
        )
    
    def _draw_selection(self):
        """Draw selection range."""
        if self._selection_start is None:
            return
        
        start_x = self._generation_to_x(self._selection_start)
        end_x = self._generation_to_x(self._current_generation)
        
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        
        # Selection rectangle
        self.create_rectangle(
            start_x, self.config.margin,
            end_x, self.config.height - self.config.margin,
            fill=STYLE.GOLD + "20",  # Very transparent
            outline=STYLE.GOLD,
            width=2, dash=(3, 3),
            tags="selection"
        )
    
    def _on_click(self, event):
        """Handle mouse click."""
        # Convert click to generation
        clicked_gen = self._x_to_generation(event.x)
        clicked_gen = max(0, round(clicked_gen))
        
        # Check if shift is held for range selection
        if event.state & 0x1:  # Shift key
            if self._selection_start is None:
                self._selection_start = self._current_generation
            
            if self._on_range_select:
                start = min(self._selection_start, clicked_gen)
                end = max(self._selection_start, clicked_gen)
                self._on_range_select(start, end)
        else:
            # Single generation click
            self._selection_start = None
            
            if self._on_generation_click:
                self._on_generation_click(clicked_gen)
        
        self.set_current_generation(clicked_gen)
    
    def _on_drag(self, event):
        """Handle mouse drag (pan)."""
        if not self._dragging:
            self._dragging = True
            self._drag_start_x = event.x
            return
        
        # Calculate pan delta
        dx = event.x - self._drag_start_x
        
        # Convert to generation units
        plot_width = self.config.width - 2 * self.config.margin
        view_range = self._view_end - self._view_start
        gen_delta = (dx / plot_width) * view_range
        
        # Update view
        self._view_start -= gen_delta
        self._view_end -= gen_delta
        
        # Constrain to valid range
        if self._view_start < 0:
            self._view_end += -self._view_start
            self._view_start = 0
        
        self._drag_start_x = event.x
        self._draw()
    
    def _on_release(self, event):
        """Handle mouse release."""
        self._dragging = False
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for hover effects."""
        # Find nearest generation point
        hover_gen = None
        min_dist = 20  # Maximum hover distance
        
        for gen_data in self._generations:
            if self._view_start <= gen_data.generation <= self._view_end:
                x = self._generation_to_x(gen_data.generation)
                y = self._fitness_to_y(gen_data.best_fitness)
                
                dist = math.sqrt((event.x - x)**2 + (event.y - y)**2)
                if dist < min_dist:
                    hover_gen = gen_data.generation
                    min_dist = dist
        
        if hover_gen != self._hover_generation:
            self._hover_generation = hover_gen
            self._draw()  # Redraw to show hover effect
            
            # Show tooltip
            if hover_gen is not None:
                self._show_tooltip(event.x, event.y, hover_gen)
            else:
                self.delete("tooltip")
    
    def _show_tooltip(self, x: int, y: int, generation: int):
        """Show tooltip for generation."""
        gen_data = next(
            (g for g in self._generations if g.generation == generation),
            None
        )
        
        if not gen_data:
            return
        
        # Tooltip text
        lines = [
            f"Generation {gen_data.generation}",
            f"Best: {gen_data.best_fitness:.3f}",
            f"Avg: {gen_data.avg_fitness:.3f}",
            f"Pop: {gen_data.population_size}"
        ]
        
        if gen_data.notes:
            lines.append(f"Note: {gen_data.notes}")
        
        tooltip_text = "\n".join(lines)
        
        # Tooltip background
        bbox = self.bbox(self.create_text(0, 0, text=tooltip_text))
        if bbox:
            width = bbox[2] - bbox[0] + 10
            height = bbox[3] - bbox[1] + 10
        else:
            width, height = 100, 60
        
        # Position tooltip
        tooltip_x = x + 10
        tooltip_y = y - height - 10
        
        # Keep tooltip in bounds
        if tooltip_x + width > self.config.width:
            tooltip_x = x - width - 10
        if tooltip_y < 0:
            tooltip_y = y + 20
        
        self.delete("tooltip")
        
        # Background
        self.create_rectangle(
            tooltip_x, tooltip_y,
            tooltip_x + width, tooltip_y + height,
            fill=STYLE.BG_MEDIUM,
            outline=STYLE.GOLD,
            width=1,
            tags="tooltip"
        )
        
        # Text
        self.create_text(
            tooltip_x + 5, tooltip_y + 5,
            text=tooltip_text,
            fill=STYLE.TEXT_PRIMARY,
            font=STYLE.font_small,
            anchor="nw",
            tags="tooltip"
        )
    
    def _on_mouse_leave(self, event):
        """Handle mouse leaving widget."""
        self._hover_generation = None
        self.delete("tooltip")
        self._draw()
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Zoom in
            zoom_factor = 0.8
        else:  # Zoom out
            zoom_factor = 1.25
        
        # Zoom around mouse position
        mouse_gen = self._x_to_generation(event.x)
        view_range = self._view_end - self._view_start
        new_range = view_range * zoom_factor
        
        # Calculate new view bounds
        left_ratio = (mouse_gen - self._view_start) / view_range
        
        self._view_start = mouse_gen - new_range * left_ratio
        self._view_end = self._view_start + new_range
        
        # Constrain
        if self._view_start < 0:
            self._view_end += -self._view_start
            self._view_start = 0
        
        self._draw()
    
    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.char == 'r':  # Reset view
            self.reset_view()
        elif event.char == 'f':  # Fit to data
            self.fit_to_data()
        elif event.keysym == 'Left':
            self.navigate_generation(-1)
        elif event.keysym == 'Right':
            self.navigate_generation(1)
    
    # Public API methods
    
    def add_generation(self, generation_data: GenerationData):
        """Add new generation data."""
        # Remove existing data for same generation
        self._generations = [g for g in self._generations if g.generation != generation_data.generation]
        
        # Add new data
        self._generations.append(generation_data)
        
        # Sort by generation
        self._generations.sort(key=lambda g: g.generation)
        
        # Update view if auto-scaling
        if self.config.auto_scale:
            max_gen = max(g.generation for g in self._generations)
            if max_gen >= self._view_end:
                self._view_end = max_gen + 1
        
        self._draw()
    
    def set_current_generation(self, generation: int):
        """Set current generation marker."""
        self._current_generation = generation
        self._draw()
    
    def set_view_range(self, start_gen: int, end_gen: int):
        """Set visible generation range."""
        self._view_start = start_gen
        self._view_end = end_gen
        self._draw()
    
    def reset_view(self):
        """Reset view to show all data."""
        if self._generations:
            self._view_start = min(g.generation for g in self._generations)
            self._view_end = max(g.generation for g in self._generations) + 1
        else:
            self._view_start = 0
            self._view_end = 10
        
        self._draw()
    
    def fit_to_data(self):
        """Fit view to show all generation data optimally."""
        self.reset_view()
    
    def navigate_generation(self, delta: int):
        """Navigate to relative generation."""
        new_gen = self._current_generation + delta
        
        if self._generations:
            valid_gens = [g.generation for g in self._generations]
            
            # Find nearest valid generation
            if new_gen in valid_gens:
                target_gen = new_gen
            else:
                # Find closest
                target_gen = min(valid_gens, key=lambda g: abs(g - new_gen))
        else:
            target_gen = max(0, new_gen)
        
        # Update view if needed
        if target_gen < self._view_start or target_gen >= self._view_end:
            view_range = self._view_end - self._view_start
            self._view_start = target_gen - view_range // 4
            self._view_end = self._view_start + view_range
        
        if self._on_generation_click:
            self._on_generation_click(target_gen)
        
        self.set_current_generation(target_gen)
    
    def get_generation_data(self, generation: int) -> Optional[GenerationData]:
        """Get data for specific generation."""
        return next(
            (g for g in self._generations if g.generation == generation),
            None
        )
    
    def clear_data(self):
        """Clear all generation data."""
        self._generations.clear()
        self._current_generation = 0
        self._view_start = 0
        self._view_end = 10
        self._draw()
    
    def export_data(self) -> List[GenerationData]:
        """Export all generation data."""
        return self._generations.copy()
