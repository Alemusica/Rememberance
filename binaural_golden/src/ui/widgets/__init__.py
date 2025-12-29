"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         UI WIDGETS MODULE                                   ║
║                                                                              ║
║   Modern and reusable UI components for Golden Studio                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Progress and feedback widgets
from .golden_progress import GoldenProgressBar, ProgressConfig

# Data visualization widgets
from .fitness_radar import FitnessRadar, RadarConfig
from .timeline import EvolutionTimeline, TimelineConfig, GenerationData

# Content display widgets
from .genome_card import GenomeCard, CardConfig, PlateGenome, ContourType
from .plate_canvas import (
    PlateCanvas, CanvasConfig, LayerType, 
    Person, Exciter
)

# Interactive widgets
from .modern_button import (
    GoldenButton, ButtonConfig, 
    ButtonVariant, ButtonSize
)

__all__ = [
    # Progress
    "GoldenProgressBar",
    "ProgressConfig",
    
    # Visualization
    "FitnessRadar", 
    "RadarConfig",
    "EvolutionTimeline",
    "TimelineConfig",
    "GenerationData",
    
    # Content
    "GenomeCard",
    "CardConfig", 
    "PlateGenome",
    "ContourType",
    "PlateCanvas",
    "CanvasConfig",
    "LayerType",
    "Person",
    "Exciter",
    
    # Interactive
    "GoldenButton",
    "ButtonConfig",
    "ButtonVariant", 
    "ButtonSize",
]


# Convenience functions for quick widget creation

def create_progress_bar(parent, **kwargs):
    """Create a golden progress bar with default config."""
    config = ProgressConfig(**kwargs)
    return GoldenProgressBar(parent, config=config)


def create_fitness_radar(parent, **kwargs):
    """Create a fitness radar chart with default config."""
    config = RadarConfig(**kwargs)
    return FitnessRadar(parent, config=config)


def create_evolution_timeline(parent, **kwargs):
    """Create an evolution timeline with default config."""
    config = TimelineConfig(**kwargs)
    return EvolutionTimeline(parent, config=config)


def create_genome_card(parent, genome, **kwargs):
    """Create a genome card with default config."""
    config = CardConfig(**kwargs)
    return GenomeCard(parent, genome=genome, config=config)


def create_plate_canvas(parent, **kwargs):
    """Create a plate canvas with default config."""
    config = CanvasConfig(**kwargs)
    return PlateCanvas(parent, config=config)


def create_golden_button(parent, text="Button", **kwargs):
    """Create a golden button with default config."""
    config = ButtonConfig(text=text, **kwargs)
    return GoldenButton(parent, config=config)


# Quick button variants
def create_primary_button(parent, text, command=None, **kwargs):
    """Create a primary (gold) button."""
    config = ButtonConfig(
        text=text, 
        variant=ButtonVariant.PRIMARY, 
        **kwargs
    )
    return GoldenButton(parent, config=config, command=command)


def create_secondary_button(parent, text, command=None, **kwargs):
    """Create a secondary button."""
    config = ButtonConfig(
        text=text, 
        variant=ButtonVariant.SECONDARY, 
        **kwargs
    )
    return GoldenButton(parent, config=config, command=command)


def create_success_button(parent, text, command=None, **kwargs):
    """Create a success (green) button."""
    config = ButtonConfig(
        text=text, 
        variant=ButtonVariant.SUCCESS, 
        **kwargs
    )
    return GoldenButton(parent, config=config, command=command)


def create_danger_button(parent, text, command=None, **kwargs):
    """Create a danger (red) button."""
    config = ButtonConfig(
        text=text, 
        variant=ButtonVariant.DANGER, 
        **kwargs
    )
    return GoldenButton(parent, config=config, command=command)


# Sample data creators for testing
def create_sample_plate_genome():
    """Create sample plate genome for testing."""
    return PlateGenome(
        length=2.0,
        width=1.2, 
        thickness=0.02,
        contour_type=ContourType.ELLIPSE,
        fitness_scores={
            "Flatness": 0.75,
            "Spine": 0.68,
            "Mass": 0.82,
            "Coupling": 0.71,
            "Total": 0.74
        },
        generation=5,
        id="sample_001"
    )


def create_sample_generation_data():
    """Create sample generation data for timeline testing."""
    return [
        GenerationData(0, 0.45, 0.32, 50),
        GenerationData(1, 0.52, 0.38, 50),
        GenerationData(2, 0.48, 0.41, 50),
        GenerationData(3, 0.61, 0.47, 50),
        GenerationData(4, 0.59, 0.49, 50),
        GenerationData(5, 0.74, 0.56, 50),
        GenerationData(6, 0.71, 0.58, 50),
        GenerationData(7, 0.78, 0.61, 50),
        GenerationData(8, 0.76, 0.63, 50),
        GenerationData(9, 0.82, 0.68, 50),
    ]


def create_sample_person():
    """Create sample person for plate canvas testing."""
    return Person(
        height=1.75,
        width=0.45,
        position=(0.0, 0.1),
        visible=True
    )


# Widget demonstration function
def demo_widgets(parent):
    """Create a demonstration of all widgets."""
    import tkinter as tk
    from tkinter import ttk
    
    # Create demo window
    demo_frame = ttk.Frame(parent)
    
    # Progress bar demo
    progress = create_progress_bar(demo_frame, width=300)
    progress.pack(pady=10)
    progress.set_value(65, "Processing...")
    
    # Button demos
    button_frame = tk.Frame(demo_frame, bg="black")
    button_frame.pack(pady=10)
    
    primary_btn = create_primary_button(
        button_frame, "Primary", 
        command=lambda: print("Primary clicked")
    )
    primary_btn.pack(side="left", padx=5)
    
    secondary_btn = create_secondary_button(
        button_frame, "Secondary",
        command=lambda: print("Secondary clicked")
    )
    secondary_btn.pack(side="left", padx=5)
    
    success_btn = create_success_button(
        button_frame, "Success",
        command=lambda: print("Success clicked") 
    )
    success_btn.pack(side="left", padx=5)
    
    # Fitness radar demo
    radar = create_fitness_radar(demo_frame, size=150)
    radar.pack(pady=10)
    radar.update_scores({
        "Flatness": 0.75,
        "Spine": 0.68, 
        "Mass": 0.82,
        "Coupling": 0.71,
        "Total": 0.74
    })
    
    # Genome card demo
    sample_genome = create_sample_plate_genome()
    card = create_genome_card(demo_frame, sample_genome)
    card.pack(pady=10)
    
    # Timeline demo
    timeline = create_evolution_timeline(demo_frame, width=400, height=100)
    timeline.pack(pady=10)
    
    for gen_data in create_sample_generation_data():
        timeline.add_generation(gen_data)
    
    return demo_frame