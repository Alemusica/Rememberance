"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              EVOLUTION LOGGER - Detailed Physics-Driven Logging              ║
║                                                                              ║
║   Structured logging for understanding the physics-driven evolution flow:    ║
║   1. Zone Priority (spine 70%, head 30%) → fitness weights                   ║
║   2. Physics-Based Shape Modification → cutout placement                      ║
║   3. Modal Analysis → frequency calculation, mode shapes                      ║
║   4. Target Comparison → flatness score, coupling score                       ║
║   5. Iteration → GA improvement                                               ║
║                                                                              ║
║   LOG LEVELS:                                                                 ║
║   - DEBUG: All internal calculations                                          ║
║   - INFO: Key decisions and results                                           ║
║   - WARNING: Issues (low scores, missing data)                                ║
║                                                                              ║
║   References:                                                                 ║
║   - Schleske 2002: Modal analysis for lutherie tuning                         ║
║   - Krylov ABH: Peninsula energy focusing                                     ║
║   - Bai & Liu 2004: GA for exciter placement                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import json

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM LOG HANDLER - Captures logs for UI display
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionLogHandler(logging.Handler):
    """Custom handler that stores logs for UI display."""
    
    def __init__(self, max_entries: int = 500):
        super().__init__()
        self.max_entries = max_entries
        self.logs: List[Dict] = []
        self._callbacks: List[Callable[[Dict], None]] = []
    
    def emit(self, record: logging.LogRecord):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "category": getattr(record, "category", "general"),
            "message": record.getMessage(),
            "data": getattr(record, "data", None),
        }
        
        self.logs.append(log_entry)
        
        # Trim if too many
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]
        
        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(log_entry)
            except Exception:
                pass
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """Add callback to be notified on new log entries."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_recent(self, n: int = 50, category: Optional[str] = None) -> List[Dict]:
        """Get recent log entries, optionally filtered by category."""
        if category:
            filtered = [l for l in self.logs if l["category"] == category]
            return filtered[-n:]
        return self.logs[-n:]
    
    def clear(self):
        """Clear all logs."""
        self.logs = []


# Global handler instance
_evolution_handler = EvolutionLogHandler()


def get_evolution_handler() -> EvolutionLogHandler:
    """Get the global evolution log handler."""
    return _evolution_handler


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_evolution_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup evolution logging with custom handler.
    
    Returns the configured logger.
    """
    logger = logging.getLogger("golden_studio.evolution")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add custom handler
    logger.addHandler(_evolution_handler)
    
    # Console handler for debug
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)
    
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED LOGGING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def log_zone_priority(
    spine_weight: float,
    head_weight: float,
    logger: Optional[logging.Logger] = None
):
    """
    Log zone priority configuration.
    
    Physics: Zone weights determine WHERE optimization focuses energy.
    - spine=0.7 → 70% of fitness depends on spine response flatness
    - head=0.3 → 30% of fitness depends on ear response
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"🎯 ZONE PRIORITY: spine={spine_weight:.0%}, head={head_weight:.0%}",
        None, None
    )
    record.category = "zone_priority"
    record.data = {
        "spine_weight": spine_weight,
        "head_weight": head_weight,
        "physics": "Zone weights from slider control WHERE vibration energy is optimized"
    }
    logger.handle(record)


def log_modal_analysis(
    plate_length: float,
    plate_width: float,
    thickness: float,
    n_modes: int,
    frequencies: List[float],
    resolution: tuple,
    logger: Optional[logging.Logger] = None
):
    """
    Log modal analysis results.
    
    Physics: Modal analysis reveals natural vibration patterns.
    - Resolution MUST be > distance between cutouts to see effects
    - Low modes (< 200 Hz) dominate therapeutic response
    - Mode shapes show antinodes (high amplitude) and nodes (zero amplitude)
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    # Calculate grid spacing
    dx = plate_length / resolution[0] * 1000  # mm
    dy = plate_width / resolution[1] * 1000    # mm
    
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"📊 MODAL ANALYSIS: {n_modes} modes computed, f1={frequencies[0]:.1f}Hz, "
        f"grid={resolution[0]}x{resolution[1]} (Δx={dx:.1f}mm, Δy={dy:.1f}mm)",
        None, None
    )
    record.category = "modal_analysis"
    record.data = {
        "plate_dimensions": f"{plate_length*100:.1f}cm × {plate_width*100:.1f}cm × {thickness*1000:.1f}mm",
        "n_modes": n_modes,
        "frequencies_hz": frequencies[:5],  # First 5
        "resolution": resolution,
        "grid_spacing_mm": (dx, dy),
        "physics": "Resolution must be finer than cutout distance to capture effects (Schleske 2002)"
    }
    logger.handle(record)
    
    # Warn if resolution is too coarse
    typical_cutout_dist = 50  # mm typical minimum cutout distance
    if min(dx, dy) > typical_cutout_dist:
        warn_record = logger.makeRecord(
            logger.name, logging.WARNING, "", 0,
            f"⚠️ Modal grid spacing ({min(dx,dy):.1f}mm) > typical cutout distance ({typical_cutout_dist}mm). "
            f"Consider increasing resolution for accurate cutout effect prediction.",
            None, None
        )
        warn_record.category = "modal_analysis"
        logger.handle(warn_record)


def log_cutout_placement(
    cutout_index: int,
    x: float,
    y: float,
    shape: str,
    purpose: str,
    target_mode: Optional[str],
    expected_freq_shift: float,
    confidence: float,
    physics_guided: bool,
    logger: Optional[logging.Logger] = None
):
    """
    Log cutout placement decision.
    
    Physics: Cutouts modify plate stiffness and mass at specific locations.
    - Cutout at ANTINODE → max frequency shift (Rayleigh-Ritz)
    - Cutout at NODE → minimal effect on that mode
    - F-holes in violins are positioned to tune Helmholtz + corpus modes
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    method = "PHYSICS-GUIDED" if physics_guided else "RANDOM"
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"✂️ CUTOUT #{cutout_index} [{method}]: pos=({x:.2f}, {y:.2f}), shape={shape}, "
        f"purpose={purpose}, Δf≈{expected_freq_shift:+.1f}Hz, confidence={confidence:.0%}",
        None, None
    )
    record.category = "cutout_placement"
    record.data = {
        "cutout_index": cutout_index,
        "position": (x, y),
        "shape": shape,
        "purpose": purpose,
        "target_mode": target_mode,
        "expected_freq_shift_hz": expected_freq_shift,
        "confidence": confidence,
        "physics_guided": physics_guided,
        "physics": "Rayleigh-Ritz: Δω²/ω² ≈ -(1/M)ΔM + (1/K)ΔK"
    }
    logger.handle(record)


def log_fitness_evaluation(
    genome_id: int,
    flatness_score: float,
    spine_flatness: float,
    head_flatness: float,
    ear_uniformity: float,
    spine_coupling: float,
    structural_score: float,
    total_fitness: float,
    zone_weights: tuple,
    n_cutouts: int,
    logger: Optional[logging.Logger] = None
):
    """
    Log fitness evaluation results.
    
    Physics: Fitness combines multiple objectives:
    - flatness_score = zone_weights.spine * spine_flatness + zone_weights.head * head_flatness
    - ear_uniformity = L/R balance (critical for binaural)
    - structural_score = deflection safety (< 10mm under person weight)
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    record = logger.makeRecord(
        logger.name, logging.DEBUG, "", 0,
        f"📈 FITNESS #{genome_id}: total={total_fitness:.3f} | "
        f"spine_flat={spine_flatness:.2f}, head_flat={head_flatness:.2f}, "
        f"ear_LR={ear_uniformity:.2f}, coupling={spine_coupling:.2f}, "
        f"struct={structural_score:.2f} | cutouts={n_cutouts}",
        None, None
    )
    record.category = "fitness"
    record.data = {
        "genome_id": genome_id,
        "total_fitness": total_fitness,
        "flatness_score": flatness_score,
        "spine_flatness": spine_flatness,
        "head_flatness": head_flatness,
        "ear_uniformity": ear_uniformity,
        "spine_coupling": spine_coupling,
        "structural_score": structural_score,
        "zone_weights": zone_weights,
        "n_cutouts": n_cutouts,
        "physics": f"Combined objective: {zone_weights[0]:.0%} spine + {zone_weights[1]:.0%} head"
    }
    logger.handle(record)


def log_generation_summary(
    generation: int,
    best_fitness: float,
    avg_fitness: float,
    diversity: float,
    mutation_sigma: float,
    best_n_cutouts: int,
    best_n_grooves: int,
    best_contour: str,
    improvement: float,
    logger: Optional[logging.Logger] = None
):
    """
    Log generation summary.
    
    Physics: GA evolves population toward better physics:
    - Diversity ensures exploration of parameter space
    - Adaptive mutation refines solutions over time
    - Elitism preserves best physics configurations
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    trend = "↑" if improvement > 0.001 else ("→" if improvement > -0.001 else "↓")
    
    # Build features string
    features = []
    if best_n_cutouts > 0:
        features.append(f"{best_n_cutouts} cutouts")
    if best_n_grooves > 0:
        features.append(f"{best_n_grooves} grooves")
    features_str = ", ".join(features) if features else "no features"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"🧬 GEN {generation:3d} {trend}: best={best_fitness:.4f} (Δ={improvement:+.4f}), "
        f"avg={avg_fitness:.3f}, div={diversity:.2f}, σ={mutation_sigma:.3f} | "
        f"best: {best_contour} with {features_str}",
        None, None
    )
    record.category = "generation"
    record.data = {
        "generation": generation,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "diversity": diversity,
        "mutation_sigma": mutation_sigma,
        "improvement": improvement,
        "best_n_cutouts": best_n_cutouts,
        "best_n_grooves": best_n_grooves,
        "best_contour": best_contour,
        "trend": trend,
        "physics": "GA explores plate parameter space guided by physics-based fitness"
    }
    logger.handle(record)


def log_comparison_with_target(
    spine_target_db: float,
    spine_achieved_db: float,
    head_target_db: float,
    head_achieved_db: float,
    ear_target_uniformity: float,
    ear_achieved_uniformity: float,
    logger: Optional[logging.Logger] = None
):
    """
    Log comparison between achieved and target response.
    
    Physics targets from research:
    - Spine: < 10dB variation in 20-300Hz (Sum & Pan 2000)
    - Head: < 6dB variation in 50-8000Hz (Lu 2012)
    - Ear uniformity: > 90% L/R balance (binaural requirement)
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    spine_ok = "✓" if spine_achieved_db <= spine_target_db else "✗"
    head_ok = "✓" if head_achieved_db <= head_target_db else "✗"
    ear_ok = "✓" if ear_achieved_uniformity >= ear_target_uniformity else "✗"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"🎯 TARGET COMPARISON:\n"
        f"   Spine: {spine_achieved_db:.1f}dB vs ≤{spine_target_db:.1f}dB target {spine_ok}\n"
        f"   Head:  {head_achieved_db:.1f}dB vs ≤{head_target_db:.1f}dB target {head_ok}\n"
        f"   Ear L/R: {ear_achieved_uniformity:.0%} vs ≥{ear_target_uniformity:.0%} target {ear_ok}",
        None, None
    )
    record.category = "target_comparison"
    record.data = {
        "spine_target_db": spine_target_db,
        "spine_achieved_db": spine_achieved_db,
        "spine_met": spine_achieved_db <= spine_target_db,
        "head_target_db": head_target_db,
        "head_achieved_db": head_achieved_db,
        "head_met": head_achieved_db <= head_target_db,
        "ear_target": ear_target_uniformity,
        "ear_achieved": ear_achieved_uniformity,
        "ear_met": ear_achieved_uniformity >= ear_target_uniformity,
        "physics": "Research-based targets from Lu 2012, Sum & Pan 2000"
    }
    logger.handle(record)


def log_physics_decision(
    decision: str,
    reason: str,
    parameters: Dict,
    reference: str,
    logger: Optional[logging.Logger] = None
):
    """
    Log a physics-based decision with research reference.
    """
    if logger is None:
        logger = logging.getLogger("golden_studio.evolution")
    
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"🔬 PHYSICS DECISION: {decision}\n   Reason: {reason}\n   Ref: {reference}",
        None, None
    )
    record.category = "physics_decision"
    record.data = {
        "decision": decision,
        "reason": reason,
        "parameters": parameters,
        "reference": reference
    }
    logger.handle(record)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_evolution_summary(
    handler: Optional[EvolutionLogHandler] = None
) -> str:
    """
    Generate a human-readable summary of the evolution process.
    """
    if handler is None:
        handler = _evolution_handler
    
    logs = handler.logs
    
    # Count by category
    categories = {}
    for log in logs:
        cat = log["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    # Extract key metrics
    gen_logs = [l for l in logs if l["category"] == "generation"]
    fitness_values = [l["data"]["best_fitness"] for l in gen_logs if l.get("data")]
    
    summary = [
        "═" * 60,
        "EVOLUTION SUMMARY",
        "═" * 60,
        f"Total log entries: {len(logs)}",
        f"Categories: {categories}",
    ]
    
    if fitness_values:
        summary.extend([
            f"Generations: {len(fitness_values)}",
            f"Initial fitness: {fitness_values[0]:.4f}",
            f"Final fitness: {fitness_values[-1]:.4f}",
            f"Improvement: {fitness_values[-1] - fitness_values[0]:.4f}",
        ])
    
    # Physics decisions
    physics_logs = [l for l in logs if l["category"] == "physics_decision"]
    if physics_logs:
        summary.append("\nPhysics-guided decisions:")
        for log in physics_logs[-5:]:  # Last 5
            summary.append(f"  • {log['message']}")
    
    summary.append("═" * 60)
    
    return "\n".join(summary)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT FOR UI
# ══════════════════════════════════════════════════════════════════════════════

def get_formatted_logs(
    n: int = 20,
    category: Optional[str] = None,
    level: Optional[str] = None,
    handler: Optional[EvolutionLogHandler] = None
) -> List[str]:
    """
    Get formatted log strings for UI display.
    
    Returns list of formatted strings with emoji prefixes.
    """
    if handler is None:
        handler = _evolution_handler
    
    logs = handler.get_recent(n, category)
    
    # Filter by level if specified
    if level:
        logs = [l for l in logs if l["level"] == level]
    
    formatted = []
    for log in logs:
        timestamp = log["timestamp"].split("T")[1][:8]  # HH:MM:SS
        level_emoji = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
        }.get(log["level"], "•")
        
        formatted.append(f"[{timestamp}] {level_emoji} {log['message']}")
    
    return formatted
