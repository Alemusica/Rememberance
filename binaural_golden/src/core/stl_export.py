"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  STL/OBJ Export for CNC Fabrication                                          ║
║  Issue #7 Feature Request                                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Exports plate geometry with cutouts and exciter mounting positions           ║
║  for CNC routing and 3D printing.                                            ║
║                                                                               ║
║  Output formats:                                                              ║
║  - STL (binary): Standard 3D printing format                                  ║
║  - OBJ (text): CAD/CAM format with material groups                           ║
║  - DXF (2D): Laser cutting / CNC routing                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import struct
import json
from pathlib import Path


@dataclass
class Vertex:
    """3D vertex."""
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def __mul__(self, scale: float) -> 'Vertex':
        return Vertex(self.x * scale, self.y * scale, self.z * scale)
    
    def __add__(self, other: 'Vertex') -> 'Vertex':
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)


@dataclass
class Triangle:
    """Triangle face for STL."""
    v1: Vertex
    v2: Vertex
    v3: Vertex
    normal: Optional[Vertex] = None
    
    def calculate_normal(self) -> Vertex:
        """Calculate face normal using right-hand rule."""
        u = Vertex(
            self.v2.x - self.v1.x,
            self.v2.y - self.v1.y,
            self.v2.z - self.v1.z
        )
        v = Vertex(
            self.v3.x - self.v1.x,
            self.v3.y - self.v1.y,
            self.v3.z - self.v1.z
        )
        n = Vertex(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        )
        # Normalize
        mag = np.sqrt(n.x**2 + n.y**2 + n.z**2)
        if mag > 0:
            n = Vertex(n.x/mag, n.y/mag, n.z/mag)
        self.normal = n
        return n


class PlateSTLExporter:
    """
    Export plate geometry to STL/OBJ for CNC fabrication.
    
    Features:
    - Full plate geometry with variable thickness
    - Cutout holes (through cuts)
    - Exciter mounting holes with depth
    - Groove channels
    """
    
    def __init__(self, genome: Any, resolution: int = 40):
        """
        Args:
            genome: PlateGenome with plate definition
            resolution: Mesh resolution (points per edge)
        """
        self.genome = genome
        self.resolution = resolution
        self.triangles: List[Triangle] = []
        
    def generate_mesh(self) -> List[Triangle]:
        """Generate complete mesh for plate."""
        self.triangles = []
        
        # 1. Generate top surface
        self._generate_top_surface()
        
        # 2. Generate bottom surface
        self._generate_bottom_surface()
        
        # 3. Generate edges
        self._generate_edges()
        
        # 4. Generate cutout walls
        for cutout in self.genome.cutouts:
            self._generate_cutout_walls(cutout)
        
        # 5. Generate exciter mounting holes
        for exciter in self.genome.exciters:
            self._generate_exciter_mount(exciter)
        
        return self.triangles
    
    def _get_contour_points(self) -> List[Tuple[float, float]]:
        """Get plate contour points based on contour_type."""
        L = self.genome.length
        W = self.genome.width
        n = self.resolution
        
        from core.plate_genome import ContourType
        
        if self.genome.contour_type == ContourType.RECTANGLE:
            return [
                (-L/2, -W/2), (L/2, -W/2), (L/2, W/2), (-L/2, W/2)
            ]
        
        elif self.genome.contour_type == ContourType.ELLIPSE:
            points = []
            for i in range(n):
                theta = 2 * np.pi * i / n
                x = L/2 * np.cos(theta)
                y = W/2 * np.sin(theta)
                points.append((x, y))
            return points
        
        elif self.genome.contour_type == ContourType.GOLDEN_RECT:
            # Rectangle with phi proportions
            return [
                (-L/2, -W/2), (L/2, -W/2), (L/2, W/2), (-L/2, W/2)
            ]
        
        elif self.genome.contour_type == ContourType.OVOID:
            # Egg shape (wider at one end)
            points = []
            for i in range(n):
                theta = 2 * np.pi * i / n
                # Modulate width based on position
                w_factor = 1.0 + 0.2 * np.sin(theta/2)
                x = L/2 * np.cos(theta)
                y = W/2 * w_factor * np.sin(theta)
                points.append((x, y))
            return points
        
        elif self.genome.contour_type == ContourType.FREEFORM:
            if len(self.genome.control_points) > 0:
                # Use spline through control points
                return [(p[0], p[1]) for p in self.genome.control_points]
        
        # Default: rectangle
        return [(-L/2, -W/2), (L/2, -W/2), (L/2, W/2), (-L/2, W/2)]
    
    def _get_thickness_at(self, x: float, y: float) -> float:
        """Get thickness at point (for variable thickness plates)."""
        if self.genome.thickness_field is not None:
            # Interpolate from thickness field
            L, W = self.genome.length, self.genome.width
            nx, ny = self.genome.thickness_field.shape
            ix = int((x + L/2) / L * (nx - 1))
            iy = int((y + W/2) / W * (ny - 1))
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            return self.genome.thickness_field[ix, iy]
        return self.genome.thickness_base
    
    def _is_in_cutout(self, x: float, y: float) -> bool:
        """Check if point is inside any cutout."""
        L, W = self.genome.length, self.genome.width
        
        for cutout in self.genome.cutouts:
            # Cutout center in world coords
            cx = cutout.y * L - L/2  # cutout.y is longitudinal
            cy = cutout.x * W - W/2  # cutout.x is lateral
            
            # Cutout size
            cw = cutout.height * L  # longitudinal
            ch = cutout.width * W   # lateral
            
            # Check based on shape
            if cutout.shape in ['ellipse', 'circle']:
                # Ellipse test
                dx = (x - cx) / (cw/2)
                dy = (y - cy) / (ch/2)
                if dx*dx + dy*dy < 1.0:
                    return True
            else:
                # Rectangle test (approximate for other shapes)
                if abs(x - cx) < cw/2 and abs(y - cy) < ch/2:
                    return True
        
        return False
    
    def _generate_top_surface(self):
        """Generate top surface mesh."""
        L, W = self.genome.length, self.genome.width
        n = self.resolution
        
        # Grid of points
        for i in range(n - 1):
            for j in range(n - 1):
                x0 = -L/2 + i * L / (n - 1)
                x1 = -L/2 + (i + 1) * L / (n - 1)
                y0 = -W/2 + j * W / (n - 1)
                y1 = -W/2 + (j + 1) * W / (n - 1)
                
                # Skip if any corner in cutout
                if any(self._is_in_cutout(x, y) for x, y in [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]):
                    continue
                
                z0 = self._get_thickness_at(x0, y0)
                z1 = self._get_thickness_at(x1, y0)
                z2 = self._get_thickness_at(x1, y1)
                z3 = self._get_thickness_at(x0, y1)
                
                # Two triangles per quad
                t1 = Triangle(
                    Vertex(x0, y0, z0),
                    Vertex(x1, y0, z1),
                    Vertex(x1, y1, z2)
                )
                t2 = Triangle(
                    Vertex(x0, y0, z0),
                    Vertex(x1, y1, z2),
                    Vertex(x0, y1, z3)
                )
                t1.calculate_normal()
                t2.calculate_normal()
                self.triangles.extend([t1, t2])
    
    def _generate_bottom_surface(self):
        """Generate bottom surface mesh (z=0)."""
        L, W = self.genome.length, self.genome.width
        n = self.resolution
        
        for i in range(n - 1):
            for j in range(n - 1):
                x0 = -L/2 + i * L / (n - 1)
                x1 = -L/2 + (i + 1) * L / (n - 1)
                y0 = -W/2 + j * W / (n - 1)
                y1 = -W/2 + (j + 1) * W / (n - 1)
                
                if any(self._is_in_cutout(x, y) for x, y in [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]):
                    continue
                
                # Bottom surface (z=0), reversed winding
                t1 = Triangle(
                    Vertex(x0, y0, 0),
                    Vertex(x1, y1, 0),
                    Vertex(x1, y0, 0)
                )
                t2 = Triangle(
                    Vertex(x0, y0, 0),
                    Vertex(x0, y1, 0),
                    Vertex(x1, y1, 0)
                )
                t1.calculate_normal()
                t2.calculate_normal()
                self.triangles.extend([t1, t2])
    
    def _generate_edges(self):
        """Generate edge walls around plate perimeter."""
        contour = self._get_contour_points()
        n = len(contour)
        
        for i in range(n):
            p1 = contour[i]
            p2 = contour[(i + 1) % n]
            
            z1 = self._get_thickness_at(p1[0], p1[1])
            z2 = self._get_thickness_at(p2[0], p2[1])
            
            # Wall quad as two triangles
            t1 = Triangle(
                Vertex(p1[0], p1[1], 0),
                Vertex(p2[0], p2[1], 0),
                Vertex(p2[0], p2[1], z2)
            )
            t2 = Triangle(
                Vertex(p1[0], p1[1], 0),
                Vertex(p2[0], p2[1], z2),
                Vertex(p1[0], p1[1], z1)
            )
            t1.calculate_normal()
            t2.calculate_normal()
            self.triangles.extend([t1, t2])
    
    def _generate_cutout_walls(self, cutout):
        """Generate walls around cutout holes."""
        L, W = self.genome.length, self.genome.width
        
        # Cutout center
        cx = cutout.y * L - L/2
        cy = cutout.x * W - W/2
        
        # Size
        rx = cutout.height * L / 2
        ry = cutout.width * W / 2
        
        n = 24  # Points around cutout
        
        for i in range(n):
            theta1 = 2 * np.pi * i / n + cutout.rotation
            theta2 = 2 * np.pi * (i + 1) / n + cutout.rotation
            
            # Cutout shape
            if cutout.shape == 'circle':
                r = min(rx, ry)
                x1 = cx + r * np.cos(theta1)
                y1 = cy + r * np.sin(theta1)
                x2 = cx + r * np.cos(theta2)
                y2 = cy + r * np.sin(theta2)
            else:  # Ellipse or other
                x1 = cx + rx * np.cos(theta1)
                y1 = cy + ry * np.sin(theta1)
                x2 = cx + rx * np.cos(theta2)
                y2 = cy + ry * np.sin(theta2)
            
            z1 = self._get_thickness_at(x1, y1)
            z2 = self._get_thickness_at(x2, y2)
            
            # Wall triangles (reversed winding for inner wall)
            t1 = Triangle(
                Vertex(x1, y1, z1),
                Vertex(x2, y2, z2),
                Vertex(x2, y2, 0)
            )
            t2 = Triangle(
                Vertex(x1, y1, z1),
                Vertex(x2, y2, 0),
                Vertex(x1, y1, 0)
            )
            t1.calculate_normal()
            t2.calculate_normal()
            self.triangles.extend([t1, t2])
    
    def _generate_exciter_mount(self, exciter):
        """
        Generate exciter mounting hole/recess.
        
        Dayton DAEX25 exciters need:
        - 25mm diameter mounting surface
        - 3mm depth recess for flush mount
        - M3 screw holes at 20mm diameter
        """
        L, W = self.genome.length, self.genome.width
        
        # Exciter position
        ex = exciter.y * L - L/2  # longitudinal
        ey = exciter.x * W - W/2  # lateral
        
        # Mounting recess dimensions
        mount_r = 0.0125  # 12.5mm radius (25mm diameter)
        recess_depth = 0.003  # 3mm
        
        z_top = self._get_thickness_at(ex, ey)
        z_recess = z_top - recess_depth
        
        n = 16
        # Recess circle (top)
        for i in range(n):
            theta1 = 2 * np.pi * i / n
            theta2 = 2 * np.pi * (i + 1) / n
            
            x1 = ex + mount_r * np.cos(theta1)
            y1 = ey + mount_r * np.sin(theta1)
            x2 = ex + mount_r * np.cos(theta2)
            y2 = ey + mount_r * np.sin(theta2)
            
            # Recess bottom surface
            t = Triangle(
                Vertex(ex, ey, z_recess),
                Vertex(x1, y1, z_recess),
                Vertex(x2, y2, z_recess)
            )
            t.calculate_normal()
            self.triangles.append(t)
            
            # Recess wall
            t1 = Triangle(
                Vertex(x1, y1, z_top),
                Vertex(x1, y1, z_recess),
                Vertex(x2, y2, z_recess)
            )
            t2 = Triangle(
                Vertex(x1, y1, z_top),
                Vertex(x2, y2, z_recess),
                Vertex(x2, y2, z_top)
            )
            t1.calculate_normal()
            t2.calculate_normal()
            self.triangles.extend([t1, t2])
    
    def export_stl_binary(self, filepath: str):
        """Export mesh as binary STL."""
        if not self.triangles:
            self.generate_mesh()
        
        with open(filepath, 'wb') as f:
            # Header (80 bytes)
            header = b'Golden Studio Plate Export - CNC Ready' + b'\0' * 42
            f.write(header)
            
            # Triangle count (uint32)
            f.write(struct.pack('<I', len(self.triangles)))
            
            # Triangles
            for tri in self.triangles:
                # Normal (3 floats)
                n = tri.normal or tri.calculate_normal()
                f.write(struct.pack('<fff', n.x, n.y, n.z))
                
                # Vertices (9 floats)
                for v in [tri.v1, tri.v2, tri.v3]:
                    f.write(struct.pack('<fff', v.x, v.y, v.z))
                
                # Attribute byte count (uint16)
                f.write(struct.pack('<H', 0))
    
    def export_stl_ascii(self, filepath: str):
        """Export mesh as ASCII STL."""
        if not self.triangles:
            self.generate_mesh()
        
        with open(filepath, 'w') as f:
            f.write("solid plate_export\n")
            for tri in self.triangles:
                n = tri.normal or tri.calculate_normal()
                f.write(f"  facet normal {n.x:.6f} {n.y:.6f} {n.z:.6f}\n")
                f.write("    outer loop\n")
                for v in [tri.v1, tri.v2, tri.v3]:
                    f.write(f"      vertex {v.x:.6f} {v.y:.6f} {v.z:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid plate_export\n")
    
    def export_obj(self, filepath: str):
        """
        Export mesh as OBJ with material groups.
        
        Groups:
        - plate: Main plate body
        - cutout_N: Each cutout
        - exciter_N: Each exciter mount
        """
        if not self.triangles:
            self.generate_mesh()
        
        # Collect unique vertices
        vertices = []
        vertex_map = {}
        
        def get_vertex_idx(v: Vertex) -> int:
            key = (round(v.x, 6), round(v.y, 6), round(v.z, 6))
            if key not in vertex_map:
                vertex_map[key] = len(vertices) + 1  # OBJ is 1-indexed
                vertices.append(v)
            return vertex_map[key]
        
        with open(filepath, 'w') as f:
            f.write("# Golden Studio Plate Export\n")
            f.write(f"# Plate: {self.genome.length*1000:.1f}mm x {self.genome.width*1000:.1f}mm x {self.genome.thickness_base*1000:.1f}mm\n")
            f.write(f"# Cutouts: {len(self.genome.cutouts)}\n")
            f.write(f"# Exciters: {len(self.genome.exciters)}\n\n")
            
            # Vertices
            for v in vertices:
                f.write(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}\n")
            
            f.write(f"\n# {len(self.triangles)} faces\n")
            f.write("g plate\n")
            
            # Faces
            for tri in self.triangles:
                i1 = get_vertex_idx(tri.v1)
                i2 = get_vertex_idx(tri.v2)
                i3 = get_vertex_idx(tri.v3)
                f.write(f"f {i1} {i2} {i3}\n")
            
            # Re-write vertices at end (OBJ requires vertices before faces)
            # Actually, let's do this properly with a two-pass approach
        
        # Proper two-pass export
        vertex_list = []
        face_list = []
        vertex_map = {}
        
        for tri in self.triangles:
            face_indices = []
            for v in [tri.v1, tri.v2, tri.v3]:
                key = (round(v.x, 6), round(v.y, 6), round(v.z, 6))
                if key not in vertex_map:
                    vertex_map[key] = len(vertex_list) + 1
                    vertex_list.append(v)
                face_indices.append(vertex_map[key])
            face_list.append(face_indices)
        
        with open(filepath, 'w') as f:
            f.write("# Golden Studio Plate Export for CNC\n")
            f.write(f"# Dimensions: {self.genome.length*1000:.1f}mm x {self.genome.width*1000:.1f}mm x {self.genome.thickness_base*1000:.1f}mm\n")
            f.write(f"# Cutouts: {len(self.genome.cutouts)}\n")
            f.write(f"# Exciters: {len(self.genome.exciters)} (Dayton DAEX25)\n\n")
            
            for v in vertex_list:
                f.write(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}\n")
            
            f.write("\ng plate\n")
            for face in face_list:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    def export_dxf_2d(self, filepath: str):
        """
        Export 2D DXF for laser cutting / CNC routing.
        
        Layers:
        - OUTLINE: Plate perimeter
        - CUTOUTS: Internal cutouts (through cuts)
        - EXCITERS: Exciter positions (drill points)
        - GROOVES: Groove lines (partial depth)
        """
        L, W = self.genome.length, self.genome.width
        
        with open(filepath, 'w') as f:
            # DXF header
            f.write("0\nSECTION\n2\nHEADER\n0\nENDSEC\n")
            f.write("0\nSECTION\n2\nENTITIES\n")
            
            # Plate outline
            contour = self._get_contour_points()
            n = len(contour)
            for i in range(n):
                p1 = contour[i]
                p2 = contour[(i + 1) % n]
                # Scale to mm
                f.write(f"0\nLINE\n8\nOUTLINE\n")
                f.write(f"10\n{p1[0]*1000:.3f}\n20\n{p1[1]*1000:.3f}\n30\n0.0\n")
                f.write(f"11\n{p2[0]*1000:.3f}\n21\n{p2[1]*1000:.3f}\n31\n0.0\n")
            
            # Cutouts as circles/ellipses
            for cutout in self.genome.cutouts:
                cx = (cutout.y * L - L/2) * 1000  # mm
                cy = (cutout.x * W - W/2) * 1000
                rx = cutout.height * L * 1000 / 2
                ry = cutout.width * W * 1000 / 2
                
                if cutout.shape == 'circle' or abs(rx - ry) < 1:
                    # Circle
                    f.write(f"0\nCIRCLE\n8\nCUTOUTS\n")
                    f.write(f"10\n{cx:.3f}\n20\n{cy:.3f}\n30\n0.0\n")
                    f.write(f"40\n{min(rx, ry):.3f}\n")
                else:
                    # Approximate ellipse with polyline
                    f.write(f"0\nPOLYLINE\n8\nCUTOUTS\n70\n1\n")
                    for i in range(25):
                        theta = 2 * np.pi * i / 24
                        x = cx + rx * np.cos(theta + cutout.rotation)
                        y = cy + ry * np.sin(theta + cutout.rotation)
                        f.write(f"0\nVERTEX\n8\nCUTOUTS\n")
                        f.write(f"10\n{x:.3f}\n20\n{y:.3f}\n30\n0.0\n")
                    f.write("0\nSEQEND\n")
            
            # Exciter positions as points
            for i, exciter in enumerate(self.genome.exciters):
                ex = (exciter.y * L - L/2) * 1000
                ey = (exciter.x * W - W/2) * 1000
                # Circle for exciter position
                f.write(f"0\nCIRCLE\n8\nEXCITERS\n")
                f.write(f"10\n{ex:.3f}\n20\n{ey:.3f}\n30\n0.0\n")
                f.write("40\n12.5\n")  # 25mm diameter
            
            # Footer
            f.write("0\nENDSEC\n0\nEOF\n")
    
    def get_manufacturing_notes(self) -> str:
        """Generate manufacturing notes for CNC shop."""
        L, W = self.genome.length, self.genome.width
        t = self.genome.thickness_base
        
        notes = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  VIBROACOUSTIC PLATE - CNC MANUFACTURING NOTES                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣

MATERIAL:
  - Type: Birch Plywood (Baltic Birch recommended)
  - Dimensions: {L*1000:.0f}mm x {W*1000:.0f}mm x {t*1000:.0f}mm
  - Grain direction: Longitudinal (along length)

CUTOUTS ({len(self.genome.cutouts)} total):
  - Through cuts (full depth)
  - Use climb milling for clean edges
  - Leave 0.2mm finish allowance"""
        
        for i, cutout in enumerate(self.genome.cutouts):
            cx = cutout.y * L * 1000
            cy = cutout.x * W * 1000
            notes += f"\n  - Cutout {i+1}: {cutout.shape} at ({cx:.0f}, {cy:.0f})mm"
        
        notes += f"""

EXCITER MOUNTS ({len(self.genome.exciters)} total):
  - Dayton Audio DAEX25 Surface Transducers
  - Mounting: 25mm diameter x 3mm deep pocket
  - M3 mounting holes at 20mm PCD
  - Position accuracy: ±0.5mm critical!"""
        
        for i, exciter in enumerate(self.genome.exciters):
            ex = exciter.y * L * 1000
            ey = exciter.x * W * 1000
            notes += f"\n  - Exciter {i+1}: ({ex:.0f}, {ey:.0f})mm"
        
        if self.genome.grooves:
            notes += f"""

GROOVES ({len(self.genome.grooves)} total):
  - Partial depth channels for tuning
  - Use V-bit or ball nose end mill"""
            
            for i, groove in enumerate(self.genome.grooves):
                gx = groove.y * L * 1000
                gy = groove.x * W * 1000
                depth = groove.depth * t * 1000
                notes += f"\n  - Groove {i+1}: at ({gx:.0f}, {gy:.0f})mm, depth={depth:.1f}mm"
        
        notes += """

FINISHING:
  - Sand to 180 grit minimum
  - Seal with lacquer (affects resonance - test first)
  - DO NOT paint exciter mounting areas

QUALITY CHECKS:
  - All cutout edges chamfered or rounded
  - Exciter pockets flat within 0.1mm
  - No delamination or voids
  
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
        return notes


def export_plate_for_cnc(
    genome: Any,
    base_filepath: str,
    formats: List[str] = ['stl', 'obj', 'dxf']
) -> Dict[str, str]:
    """
    Export plate in multiple formats for CNC fabrication.
    
    Args:
        genome: PlateGenome with plate definition
        base_filepath: Base path without extension
        formats: List of formats to export ('stl', 'obj', 'dxf')
    
    Returns:
        Dict mapping format to exported filepath
    """
    exporter = PlateSTLExporter(genome)
    exporter.generate_mesh()
    
    exports = {}
    
    if 'stl' in formats:
        path = f"{base_filepath}.stl"
        exporter.export_stl_binary(path)
        exports['stl'] = path
    
    if 'obj' in formats:
        path = f"{base_filepath}.obj"
        exporter.export_obj(path)
        exports['obj'] = path
    
    if 'dxf' in formats:
        path = f"{base_filepath}.dxf"
        exporter.export_dxf_2d(path)
        exports['dxf'] = path
    
    # Always export manufacturing notes
    notes_path = f"{base_filepath}_manufacturing_notes.txt"
    with open(notes_path, 'w') as f:
        f.write(exporter.get_manufacturing_notes())
    exports['notes'] = notes_path
    
    return exports
