"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  VIRTUAL CNC - Centralized Plate Modification Interface                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Digital CNC at the disposal of the genetic algorithm.                        ║
║                                                                               ║
║  Operations:                                                                  ║
║  - ADD_CUTOUT: Through holes (ABH energy focusing)                           ║
║  - ADD_GROOVE: Partial depth channels (wave guiding)                         ║
║  - ADD_MASS: Attached masses (modal tuning - Shen 2016)                      ║
║  - THICKNESS_TAPER: Variable thickness (ABH effect - Krylov 2014)            ║
║  - CONTOUR_MODIFY: Edge shaping (peninsula ABH)                              ║
║                                                                               ║
║  The GA can issue operations like a real CNC program.                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import Enum, auto
import numpy as np
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class CNCOperation(Enum):
    """Available CNC operations for plate modification."""
    ADD_CUTOUT = auto()       # Through hole
    REMOVE_CUTOUT = auto()    # Remove existing cutout
    MODIFY_CUTOUT = auto()    # Change cutout params
    
    ADD_GROOVE = auto()       # Partial depth channel
    REMOVE_GROOVE = auto()
    MODIFY_GROOVE = auto()
    
    ADD_MASS = auto()         # Attached tuning mass
    REMOVE_MASS = auto()
    MODIFY_MASS = auto()
    
    SET_THICKNESS = auto()    # Uniform thickness
    SET_TAPER = auto()        # Variable thickness profile
    
    SET_CONTOUR = auto()      # Change plate outline
    ADD_PENINSULA = auto()    # Add ABH-focusing region
    
    ADD_EXCITER = auto()      # Exciter position
    REMOVE_EXCITER = auto()
    MOVE_EXCITER = auto()


@dataclass
class CNCInstruction:
    """
    Single CNC instruction for plate modification.
    
    Like G-code but for virtual plate shaping.
    """
    operation: CNCOperation
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = execute first
    
    # Metadata for GA tracking
    generation: int = 0
    fitness_delta: float = 0.0  # How much this improved fitness
    
    def __repr__(self):
        return f"CNC({self.operation.name}, {self.params})"


# ═══════════════════════════════════════════════════════════════════════════════
# VIRTUAL CNC MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class VirtualCNC:
    """
    Virtual CNC machine for plate modification.
    
    The genetic algorithm uses this as its "tool" to shape plates.
    Centralizes all geometry modifications in one place.
    
    Usage:
        cnc = VirtualCNC(genome)
        cnc.add_cutout(x=0.3, y=0.5, shape='ellipse', width=0.1, height=0.15)
        cnc.add_groove(x=0.2, y=0.5, length=0.3, depth=0.5, angle=0)
        cnc.add_mass(x=0.33, y=0.5, mass_kg=0.050, material='brass')
        modified_genome = cnc.execute()
    """
    
    def __init__(self, genome: Any):
        """
        Initialize with a PlateGenome to modify.
        
        Args:
            genome: PlateGenome instance
        """
        self.original_genome = genome
        self.working_genome = deepcopy(genome)
        self.instructions: List[CNCInstruction] = []
        self.history: List[CNCInstruction] = []
        
        # Constraints from genome
        self.max_cutouts = getattr(genome, 'max_cutouts', 6)
        self.max_grooves = getattr(genome, 'max_grooves', 8)
        self.max_masses = getattr(genome, 'max_attached_masses', 4)
        self.max_exciters = getattr(genome, 'max_exciters', 4)
        
        # Import genome classes lazily
        self._CutoutGene = None
        self._GrooveGene = None
        self._AttachedMassGene = None
        self._ExciterPosition = None
    
    def _ensure_imports(self):
        """Lazy import of genome classes."""
        if self._CutoutGene is None:
            try:
                from .plate_genome import (
                    CutoutGene, GrooveGene, AttachedMassGene, 
                    ExciterPosition, OPTIMAL_MASS_POSITIONS
                )
            except ImportError:
                from src.core.plate_genome import (
                    CutoutGene, GrooveGene, AttachedMassGene,
                    ExciterPosition, OPTIMAL_MASS_POSITIONS
                )
            self._CutoutGene = CutoutGene
            self._GrooveGene = GrooveGene
            self._AttachedMassGene = AttachedMassGene
            self._ExciterPosition = ExciterPosition
            self._OPTIMAL_MASS_POSITIONS = OPTIMAL_MASS_POSITIONS
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CUTOUT OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_cutout(
        self,
        x: float,
        y: float,
        shape: str = 'ellipse',
        width: float = 0.08,
        height: float = 0.12,
        rotation: float = 0.0,
        taper: float = 0.0,
        purpose: str = 'abh'
    ) -> 'VirtualCNC':
        """
        Add a cutout (through hole) to the plate.
        
        Args:
            x, y: Normalized position [0, 1]
            shape: 'circle', 'ellipse', 'rectangle', 'f_hole'
            width, height: Normalized dimensions
            rotation: Radians
            taper: Edge taper for ABH effect
            purpose: 'abh', 'weight_reduction', 'acoustic', 'structural'
        
        Returns:
            self for chaining
        """
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.ADD_CUTOUT,
            params={
                'x': np.clip(x, 0.1, 0.9),
                'y': np.clip(y, 0.1, 0.9),
                'shape': shape,
                'width': np.clip(width, 0.02, 0.3),
                'height': np.clip(height, 0.02, 0.4),
                'rotation': rotation % (2 * np.pi),
                'taper': np.clip(taper, 0, 1),
                'purpose': purpose
            },
            priority=10
        ))
        return self
    
    def remove_cutout(self, index: int) -> 'VirtualCNC':
        """Remove cutout at index."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.REMOVE_CUTOUT,
            params={'index': index},
            priority=5
        ))
        return self
    
    def modify_cutout(self, index: int, **kwargs) -> 'VirtualCNC':
        """Modify existing cutout parameters."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.MODIFY_CUTOUT,
            params={'index': index, **kwargs},
            priority=8
        ))
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GROOVE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_groove(
        self,
        x: float,
        y: float,
        length: float = 0.2,
        depth: float = 0.5,
        width: float = 0.01,
        angle: float = 0.0,
        profile: str = 'v'
    ) -> 'VirtualCNC':
        """
        Add a groove (partial depth channel) to the plate.
        
        Args:
            x, y: Center position (normalized)
            length: Groove length (normalized to plate diagonal)
            depth: Depth as fraction of thickness [0, 0.8]
            width: Groove width (normalized)
            angle: Orientation in degrees
            profile: 'v', 'u', 'square' - cross-section
        
        Returns:
            self for chaining
        """
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.ADD_GROOVE,
            params={
                'x': np.clip(x, 0.05, 0.95),
                'y': np.clip(y, 0.05, 0.95),
                'length': np.clip(length, 0.05, 0.5),
                'depth': np.clip(depth, 0.1, 0.8),
                'width': np.clip(width, 0.005, 0.05),
                'angle': angle % 360,
                'profile': profile
            },
            priority=9
        ))
        return self
    
    def add_crossover_groove(
        self,
        x: float,
        y: float,
        frequency: float = 300.0
    ) -> 'VirtualCNC':
        """
        Add groove optimized for crossover frequency.
        
        Groove dimensions calculated from target frequency.
        """
        # Wavelength at crossover
        c_plate = 3000  # Approximate wave speed in plywood m/s
        wavelength = c_plate / frequency
        
        # Groove length = λ/4 for quarter-wave effect
        length = wavelength / 4 / max(self.working_genome.length, self.working_genome.width)
        depth = 0.5  # Half thickness
        
        return self.add_groove(x, y, length=length, depth=depth, angle=0)
    
    def remove_groove(self, index: int) -> 'VirtualCNC':
        """Remove groove at index."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.REMOVE_GROOVE,
            params={'index': index},
            priority=5
        ))
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ATTACHED MASS OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_mass(
        self,
        x: float,
        y: float,
        mass_kg: float = 0.050,
        material: str = 'brass',
        mount_type: str = 'bolt'
    ) -> 'VirtualCNC':
        """
        Add attached mass for modal tuning (Shen 2016).
        
        Args:
            x, y: Position (normalized)
            mass_kg: Mass in kg
            material: 'brass', 'steel', 'lead', 'aluminum'
            mount_type: 'bolt', 'pocket', 'adhesive'
        
        Returns:
            self for chaining
        """
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.ADD_MASS,
            params={
                'x': np.clip(x, 0.1, 0.9),
                'y': np.clip(y, 0.1, 0.9),
                'mass_kg': np.clip(mass_kg, 0.010, 0.500),
                'material': material,
                'mount_type': mount_type
            },
            priority=7
        ))
        return self
    
    def add_optimal_masses(self, count: int = 2) -> 'VirtualCNC':
        """
        Add masses at optimal positions from literature (Shen 2016).
        
        Positions: (1/3, 1/2), (2/3, 1/2), (1/2, 1/3), (1/2, 2/3)
        """
        self._ensure_imports()
        positions = self._OPTIMAL_MASS_POSITIONS[:count]
        
        for x, y in positions:
            self.add_mass(x, y, mass_kg=0.050, material='brass')
        
        return self
    
    def remove_mass(self, index: int) -> 'VirtualCNC':
        """Remove attached mass at index."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.REMOVE_MASS,
            params={'index': index},
            priority=5
        ))
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXCITER OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_exciter(
        self,
        x: float,
        y: float,
        diameter: float = 0.025,
        power_w: float = 15.0
    ) -> 'VirtualCNC':
        """
        Add exciter position.
        
        Args:
            x, y: Position (normalized)
            diameter: Exciter diameter in meters
            power_w: Power rating in watts
        
        Returns:
            self for chaining
        """
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.ADD_EXCITER,
            params={
                'x': np.clip(x, 0.1, 0.9),
                'y': np.clip(y, 0.1, 0.9),
                'diameter': diameter,
                'power_w': power_w
            },
            priority=10
        ))
        return self
    
    def move_exciter(self, index: int, new_x: float, new_y: float) -> 'VirtualCNC':
        """Move existing exciter to new position."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.MOVE_EXCITER,
            params={
                'index': index,
                'x': np.clip(new_x, 0.1, 0.9),
                'y': np.clip(new_y, 0.1, 0.9)
            },
            priority=8
        ))
        return self
    
    def remove_exciter(self, index: int) -> 'VirtualCNC':
        """Remove exciter at index."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.REMOVE_EXCITER,
            params={'index': index},
            priority=5
        ))
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # THICKNESS OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def set_thickness(self, thickness_mm: float) -> 'VirtualCNC':
        """Set uniform plate thickness."""
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.SET_THICKNESS,
            params={'thickness': thickness_mm / 1000.0},  # Convert to meters
            priority=15
        ))
        return self
    
    def set_taper(self, variation: float, direction: str = 'radial') -> 'VirtualCNC':
        """
        Set thickness taper for ABH effect (Krylov 2014).
        
        Args:
            variation: Thickness variation [0, 0.5] (0=uniform, 0.5=50% taper)
            direction: 'radial', 'longitudinal', 'corner'
        
        Returns:
            self for chaining
        """
        self.instructions.append(CNCInstruction(
            operation=CNCOperation.SET_TAPER,
            params={
                'variation': np.clip(variation, 0, 0.5),
                'direction': direction
            },
            priority=14
        ))
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def execute(self) -> Any:
        """
        Execute all queued instructions and return modified genome.
        
        Instructions are sorted by priority (highest first).
        
        Returns:
            Modified PlateGenome
        """
        self._ensure_imports()
        
        # Sort by priority (descending)
        sorted_instructions = sorted(
            self.instructions, 
            key=lambda i: i.priority, 
            reverse=True
        )
        
        for instr in sorted_instructions:
            self._execute_instruction(instr)
            self.history.append(instr)
        
        # Clear pending instructions
        self.instructions = []
        
        return self.working_genome
    
    def _execute_instruction(self, instr: CNCInstruction):
        """Execute a single instruction."""
        op = instr.operation
        params = instr.params
        genome = self.working_genome
        
        # ─────────────────────────────────────────────────────────────────────
        # CUTOUT OPERATIONS
        # ─────────────────────────────────────────────────────────────────────
        if op == CNCOperation.ADD_CUTOUT:
            if len(genome.cutouts) < self.max_cutouts:
                cutout = self._CutoutGene(
                    x=params['x'],
                    y=params['y'],
                    shape=params['shape'],
                    width=params['width'],
                    height=params['height'],
                    rotation=params['rotation']
                    # Note: taper and purpose stored in params but not in CutoutGene
                )
                genome.cutouts.append(cutout)
        
        elif op == CNCOperation.REMOVE_CUTOUT:
            idx = params['index']
            if 0 <= idx < len(genome.cutouts):
                genome.cutouts.pop(idx)
        
        elif op == CNCOperation.MODIFY_CUTOUT:
            idx = params.pop('index')
            if 0 <= idx < len(genome.cutouts):
                cutout = genome.cutouts[idx]
                for key, value in params.items():
                    if hasattr(cutout, key):
                        setattr(cutout, key, value)
        
        # ─────────────────────────────────────────────────────────────────────
        # GROOVE OPERATIONS
        # ─────────────────────────────────────────────────────────────────────
        elif op == CNCOperation.ADD_GROOVE:
            if len(genome.grooves) < self.max_grooves:
                groove = self._GrooveGene(
                    x=params['x'],
                    y=params['y'],
                    length=params['length'],
                    depth=params['depth'],
                    width_mm=params.get('width', 0.01) * 1000,  # Convert normalized to mm
                    angle=np.radians(params.get('angle', 0))  # Convert degrees to radians
                )
                genome.grooves.append(groove)
        
        elif op == CNCOperation.REMOVE_GROOVE:
            idx = params['index']
            if 0 <= idx < len(genome.grooves):
                genome.grooves.pop(idx)
        
        # ─────────────────────────────────────────────────────────────────────
        # MASS OPERATIONS
        # ─────────────────────────────────────────────────────────────────────
        elif op == CNCOperation.ADD_MASS:
            if len(genome.attached_masses) < self.max_masses:
                mass = self._AttachedMassGene(
                    x=params['x'],
                    y=params['y'],
                    mass_kg=params['mass_kg'],
                    material=params['material'],
                    mount_type=params['mount_type']
                )
                genome.attached_masses.append(mass)
        
        elif op == CNCOperation.REMOVE_MASS:
            idx = params['index']
            if 0 <= idx < len(genome.attached_masses):
                genome.attached_masses.pop(idx)
        
        # ─────────────────────────────────────────────────────────────────────
        # EXCITER OPERATIONS
        # ─────────────────────────────────────────────────────────────────────
        elif op == CNCOperation.ADD_EXCITER:
            if len(genome.exciters) < self.max_exciters:
                exciter = self._ExciterPosition(
                    x=params['x'],
                    y=params['y'],
                    diameter=params.get('diameter', 0.025)
                )
                genome.exciters.append(exciter)
        
        elif op == CNCOperation.MOVE_EXCITER:
            idx = params['index']
            if 0 <= idx < len(genome.exciters):
                genome.exciters[idx].x = params['x']
                genome.exciters[idx].y = params['y']
        
        elif op == CNCOperation.REMOVE_EXCITER:
            idx = params['index']
            if 0 <= idx < len(genome.exciters):
                genome.exciters.pop(idx)
        
        # ─────────────────────────────────────────────────────────────────────
        # THICKNESS OPERATIONS
        # ─────────────────────────────────────────────────────────────────────
        elif op == CNCOperation.SET_THICKNESS:
            genome.thickness_base = params['thickness']
        
        elif op == CNCOperation.SET_TAPER:
            genome.thickness_variation = params['variation']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GA INTEGRATION HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_operation_count(self) -> Dict[str, int]:
        """Get count of each operation type in history."""
        counts = {}
        for instr in self.history:
            name = instr.operation.name
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def clear_history(self):
        """Clear operation history."""
        self.history = []
    
    def rollback(self) -> Any:
        """Rollback to original genome."""
        self.working_genome = deepcopy(self.original_genome)
        self.instructions = []
        return self.working_genome
    
    def clone(self) -> 'VirtualCNC':
        """Create a copy of this CNC with current state."""
        new_cnc = VirtualCNC(self.working_genome)
        new_cnc.history = list(self.history)
        return new_cnc
    
    def to_gcode_comments(self) -> List[str]:
        """Convert operation history to G-code comments for documentation."""
        lines = [
            "; ═══════════════════════════════════════════════════════════════",
            "; Virtual CNC Operation History",
            "; ═══════════════════════════════════════════════════════════════",
        ]
        
        for i, instr in enumerate(self.history):
            lines.append(f"; OP{i+1}: {instr.operation.name}")
            for k, v in instr.params.items():
                lines.append(f";   {k}: {v}")
        
        return lines


# ═══════════════════════════════════════════════════════════════════════════════
# GA MUTATION OPERATORS USING CNC
# ═══════════════════════════════════════════════════════════════════════════════

def mutate_with_cnc(
    genome: Any,
    mutation_rate: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> Any:
    """
    Apply random CNC mutations to a genome.
    
    This is the interface for GA to use CNC for mutations.
    
    Args:
        genome: PlateGenome to mutate
        mutation_rate: Probability of each operation type
        rng: Random generator
    
    Returns:
        Mutated genome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    cnc = VirtualCNC(genome)
    
    # Possible mutations
    if rng.random() < mutation_rate:
        # Add/modify cutout
        if len(genome.cutouts) < cnc.max_cutouts and rng.random() < 0.5:
            cnc.add_cutout(
                x=rng.uniform(0.15, 0.85),
                y=rng.uniform(0.15, 0.85),
                shape=rng.choice(['circle', 'ellipse', 'f_hole']),
                width=rng.uniform(0.05, 0.15),
                height=rng.uniform(0.08, 0.25)
            )
        elif len(genome.cutouts) > 0 and rng.random() < 0.3:
            cnc.remove_cutout(rng.integers(0, len(genome.cutouts)))
    
    if rng.random() < mutation_rate:
        # Add/modify groove
        if len(genome.grooves) < cnc.max_grooves and rng.random() < 0.5:
            cnc.add_groove(
                x=rng.uniform(0.1, 0.9),
                y=rng.uniform(0.1, 0.9),
                length=rng.uniform(0.1, 0.3),
                depth=rng.uniform(0.3, 0.6),
                angle=rng.uniform(0, 180)
            )
        elif len(genome.grooves) > 0 and rng.random() < 0.3:
            cnc.remove_groove(rng.integers(0, len(genome.grooves)))
    
    if rng.random() < mutation_rate * 0.5:
        # Add/modify mass (less frequent)
        if len(genome.attached_masses) < cnc.max_masses and rng.random() < 0.4:
            cnc.add_mass(
                x=rng.uniform(0.2, 0.8),
                y=rng.uniform(0.2, 0.8),
                mass_kg=rng.uniform(0.020, 0.100),
                material=rng.choice(['brass', 'steel', 'lead'])
            )
    
    if rng.random() < mutation_rate * 0.3:
        # Modify thickness
        if rng.random() < 0.5:
            cnc.set_taper(
                variation=rng.uniform(0, 0.3),
                direction=rng.choice(['radial', 'longitudinal', 'corner'])
            )
    
    if rng.random() < mutation_rate:
        # Move exciter
        if len(genome.exciters) > 0:
            idx = rng.integers(0, len(genome.exciters))
            cnc.move_exciter(
                idx,
                genome.exciters[idx].x + rng.uniform(-0.1, 0.1),
                genome.exciters[idx].y + rng.uniform(-0.1, 0.1)
            )
    
    return cnc.execute()


def crossover_with_cnc(
    parent1: Any,
    parent2: Any,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Any, Any]:
    """
    Crossover two genomes using CNC operations.
    
    Mixes features from both parents:
    - Cutouts from one, grooves from other
    - Masses from both (deduplicated)
    - Exciters averaged
    
    Args:
        parent1, parent2: Parent genomes
        rng: Random generator
    
    Returns:
        Two child genomes
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Child 1: Base from parent1, features from parent2
    cnc1 = VirtualCNC(parent1)
    
    # Add some cutouts from parent2
    for cutout in parent2.cutouts:
        if rng.random() < 0.5:
            cnc1.add_cutout(
                x=cutout.x, y=cutout.y,
                shape=cutout.shape,
                width=cutout.width,
                height=cutout.height,
                rotation=cutout.rotation
            )
    
    # Add grooves from parent2
    for groove in parent2.grooves:
        if rng.random() < 0.5:
            cnc1.add_groove(
                x=groove.x, y=groove.y,
                length=groove.length,
                depth=groove.depth,
                angle=getattr(groove, 'angle', 0)
            )
    
    child1 = cnc1.execute()
    
    # Child 2: Base from parent2, features from parent1
    cnc2 = VirtualCNC(parent2)
    
    for cutout in parent1.cutouts:
        if rng.random() < 0.5:
            cnc2.add_cutout(
                x=cutout.x, y=cutout.y,
                shape=cutout.shape,
                width=cutout.width,
                height=cutout.height
            )
    
    for groove in parent1.grooves:
        if rng.random() < 0.5:
            cnc2.add_groove(
                x=groove.x, y=groove.y,
                length=groove.length,
                depth=groove.depth
            )
    
    child2 = cnc2.execute()
    
    return child1, child2


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_abh_plate(
    length: float = 0.5,
    width: float = 0.4,
    n_cutouts: int = 2,
    n_exciters: int = 2
) -> Any:
    """
    Create a plate optimized for ABH effect.
    
    Uses Virtual CNC to add:
    - Corner-tapered thickness
    - Strategic elliptical cutouts
    - Optimal exciter positions
    
    Args:
        length, width: Plate dimensions in meters
        n_cutouts: Number of ABH cutouts
        n_exciters: Number of exciters
    
    Returns:
        PlateGenome configured for ABH
    """
    try:
        from .plate_genome import PlateGenome, ContourType
    except ImportError:
        from src.core.plate_genome import PlateGenome, ContourType
    
    genome = PlateGenome(
        length=length,
        width=width,
        contour_type=ContourType.RECTANGLE
    )
    
    cnc = VirtualCNC(genome)
    
    # ABH taper
    cnc.set_taper(variation=0.2, direction='corner')
    
    # ABH cutouts at acoustic nodes
    if n_cutouts >= 1:
        cnc.add_cutout(x=0.25, y=0.5, shape='ellipse', width=0.08, height=0.15, purpose='abh')
    if n_cutouts >= 2:
        cnc.add_cutout(x=0.75, y=0.5, shape='ellipse', width=0.08, height=0.15, purpose='abh')
    
    # Exciters at anti-nodes
    if n_exciters >= 1:
        cnc.add_exciter(x=0.35, y=0.5)
    if n_exciters >= 2:
        cnc.add_exciter(x=0.65, y=0.5)
    
    return cnc.execute()


def print_cnc_summary(genome: Any) -> str:
    """Print summary of CNC features on a genome."""
    lines = [
        "╔════════════════════════════════════════╗",
        "║       Virtual CNC Plate Summary        ║",
        "╠════════════════════════════════════════╣",
        f"║  Dimensions: {genome.length*1000:.0f}mm x {genome.width*1000:.0f}mm",
        f"║  Thickness:  {genome.thickness_base*1000:.1f}mm (var: {genome.thickness_variation*100:.0f}%)",
        f"║  Cutouts:    {len(genome.cutouts)}",
        f"║  Grooves:    {len(genome.grooves)}",
        f"║  Masses:     {len(genome.attached_masses)}",
        f"║  Exciters:   {len(genome.exciters)}",
        "╚════════════════════════════════════════╝",
    ]
    
    return "\n".join(lines)
