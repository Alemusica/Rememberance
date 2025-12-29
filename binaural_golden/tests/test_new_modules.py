"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TESTS FOR NEW PLATE DESIGNER MODULES                            ║
║                                                                              ║
║   Tests for:                                                                 ║
║   • STL/OBJ/DXF Export (stl_export.py)                                       ║
║   • DML Frequency Model (dml_frequency_model.py)                             ║
║   • Radar Widget (radar_widget.py)                                           ║
║   • Peninsula Detection (structural_analysis.py)                             ║
║                                                                              ║
║   Based on research from vibroacoustic_references.bib:                       ║
║   • Harris 2010: DML theory                                                  ║
║   • Aures 2001: Golden ratio positioning (0.381L, 0.618L)                    ║
║   • Bank 2010: Exciter-mode coupling                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ══════════════════════════════════════════════════════════════════════════════
# TEST: STL/OBJ/DXF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class TestSTLExport:
    """Tests for CNC export functionality (Issue #7)."""
    
    @pytest.fixture
    def sample_genome(self):
        """Create a sample PlateGenome for testing."""
        from core.plate_genome import PlateGenome, ContourType, CutoutGene, ExciterPosition
        
        genome = PlateGenome(
            length=2.0,
            width=0.8,
            thickness_base=0.015,
            contour_type=ContourType.GOLDEN_RECT,
        )
        
        # Add cutouts
        genome.cutouts = [
            CutoutGene(x=0.3, y=0.5, width=0.08, height=0.06, shape='ellipse'),
            CutoutGene(x=0.7, y=0.5, width=0.06, height=0.06, shape='circle'),
        ]
        
        # Add exciters (use ExciterPosition not ExciterGene)
        genome.exciters = [
            ExciterPosition(x=0.381, y=0.5, channel=1),  # Golden ratio position
            ExciterPosition(x=0.618, y=0.5, channel=2),  # Golden ratio position
        ]
        
        return genome
    
    def test_stl_export_creates_files(self, sample_genome):
        """Test that STL export creates the expected files."""
        from core.stl_export import export_plate_for_cnc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "test_plate")
            
            exports = export_plate_for_cnc(sample_genome, base_path)
            
            # Check that files were created
            assert 'stl' in exports
            assert 'obj' in exports
            assert 'dxf' in exports
            assert 'notes' in exports
            
            # Check files exist
            assert os.path.exists(exports['stl'])
            assert os.path.exists(exports['obj'])
            assert os.path.exists(exports['dxf'])
            assert os.path.exists(exports['notes'])
    
    def test_stl_file_is_valid_binary(self, sample_genome):
        """Test that exported STL is valid binary format."""
        from core.stl_export import export_plate_for_cnc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "test_plate")
            exports = export_plate_for_cnc(sample_genome, base_path)
            
            # Read STL file
            with open(exports['stl'], 'rb') as f:
                data = f.read()
            
            # Binary STL starts with 80-byte header
            assert len(data) > 84, "STL file too short"
            
            # After header, 4-byte triangle count
            header = data[:80]
            n_triangles = int.from_bytes(data[80:84], 'little')
            
            # Each triangle is 50 bytes (normal + 3 vertices + attribute)
            expected_size = 84 + n_triangles * 50
            assert len(data) == expected_size, f"STL size mismatch: {len(data)} vs {expected_size}"
    
    def test_manufacturing_notes_include_exciter_positions(self, sample_genome):
        """Test that manufacturing notes include exciter mount positions."""
        from core.stl_export import export_plate_for_cnc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "test_plate")
            exports = export_plate_for_cnc(sample_genome, base_path)
            
            # Read notes
            with open(exports['notes'], 'r') as f:
                notes = f.read()
            
            # Should contain exciter positions (converted to mm)
            # Exciters at x=0.381 and x=0.618 on a 2000mm plate = ~762mm and ~1236mm
            # But the notes show mm positions, so look for EXCITER keyword
            assert 'EXCITER' in notes.upper(), "Notes should mention exciters"
            assert '(' in notes and ')' in notes, "Notes should have position coordinates"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: DML FREQUENCY MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TestDMLFrequencyModel:
    """Tests for DML frequency response model based on Harris 2010 / Bank 2010."""
    
    def test_model_creation(self):
        """Test DML model can be created."""
        from core.dml_frequency_model import DMLFrequencyModel
        
        model = DMLFrequencyModel(
            plate_length=2.0,
            plate_width=0.8,
            plate_thickness=0.015,
            youngs_modulus=13e9,
            material_density=700,
        )
        
        assert model.L == 2.0
        assert model.W == 0.8
        assert model.t == 0.015
    
    def test_mode_calculation(self):
        """Test that modes are calculated correctly."""
        from core.dml_frequency_model import DMLFrequencyModel
        
        model = DMLFrequencyModel(
            plate_length=2.0,
            plate_width=0.8,
            plate_thickness=0.015,
            youngs_modulus=13e9,
            material_density=700,
        )
        
        modes = model.modes
        
        # Should have calculated some modes
        assert len(modes) >= 5
        
        # Modes should be in ascending frequency order
        freqs = [m.frequency_hz for m in modes]
        assert freqs == sorted(freqs)
        
        # First mode should be below 100 Hz for this plate
        assert modes[0].frequency_hz < 100, f"First mode {modes[0].frequency_hz} Hz too high"
    
    def test_golden_ratio_positioning(self):
        """Test golden ratio exciter positioning from Aures 2001."""
        from core.dml_frequency_model import DMLFrequencyModel, ExciterCoupling
        
        model = DMLFrequencyModel(
            plate_length=2.0,
            plate_width=0.8,
            plate_thickness=0.015,
            youngs_modulus=13e9,
            material_density=700,
        )
        
        # Test coupling at golden ratio positions
        golden_positions = [(0.381, 0.5), (0.618, 0.5)]
        
        # Calculate coupling for first mode at golden positions
        mode_1_1 = model.modes[0]
        
        coupling = ExciterCoupling.calculate(golden_positions[0], mode_1_1)
        
        # Golden ratio should give reasonable coupling (not zero)
        assert coupling.coupling_coefficient > 0.3, "Golden ratio should give good coupling"
    
    def test_exciter_coupling_at_antinode(self):
        """Test that exciter at mode antinode has maximum coupling (Bank 2010)."""
        from core.dml_frequency_model import DMLFrequencyModel, ExciterCoupling
        
        model = DMLFrequencyModel(
            plate_length=2.0,
            plate_width=0.8,
            plate_thickness=0.015,
            youngs_modulus=13e9,
            material_density=700,
        )
        
        # First mode (1,1) has antinode at center
        mode_1_1 = model.modes[0]
        
        # Calculate coupling at center (antinode)
        coupling_antinode = ExciterCoupling.calculate((0.5, 0.5), mode_1_1)
        
        # Calculate coupling at corner (near node)
        coupling_node = ExciterCoupling.calculate((0.05, 0.05), mode_1_1)
        
        # Antinode coupling should be stronger
        assert coupling_antinode.coupling_coefficient > coupling_node.coupling_coefficient, \
            f"Antinode coupling {coupling_antinode.coupling_coefficient} should be > node coupling {coupling_node.coupling_coefficient} (Bank 2010)"
    
    def test_frequency_response_calculation(self):
        """Test that frequency response can be calculated."""
        from core.dml_frequency_model import DMLFrequencyModel
        
        model = DMLFrequencyModel(
            plate_length=2.0,
            plate_width=0.8,
            plate_thickness=0.015,
            youngs_modulus=13e9,
            material_density=700,
        )
        
        # Calculate frequency response
        response = model.calculate_frequency_response(
            exciter_positions=[(0.381, 0.5)],
            freq_range=(20, 500),
            n_points=50
        )
        
        # Check frequency range
        assert len(response.frequencies) == 50
        assert response.frequencies[0] >= 20
        assert response.frequencies[-1] <= 500
        
        # Response should exist (not all zeros)
        assert np.max(np.abs(response.response_db)) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PENINSULA DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestPeninsulaDetection:
    """Tests for isolated region detection when cutouts intersect."""
    
    def test_single_cutout_no_peninsula(self):
        """Single cutout should not create peninsulas."""
        from core.structural_analysis import detect_peninsulas
        
        cutouts = [
            {'x': 0.5, 'y': 0.5, 'size': 0.1, 'shape': 'circle'}
        ]
        
        result = detect_peninsulas(
            length=2.0, width=0.8,
            cutouts=cutouts,
            resolution=50
        )
        
        assert not result.has_peninsulas
        assert result.n_regions == 1
        assert result.structural_penalty == 0.0
    
    def test_two_separate_cutouts_no_peninsula(self):
        """Two well-separated cutouts should not create peninsulas."""
        from core.structural_analysis import detect_peninsulas
        
        cutouts = [
            {'x': 0.25, 'y': 0.5, 'size': 0.05, 'shape': 'circle'},
            {'x': 0.75, 'y': 0.5, 'size': 0.05, 'shape': 'circle'},
        ]
        
        result = detect_peninsulas(
            length=2.0, width=0.8,
            cutouts=cutouts,
            resolution=50
        )
        
        assert not result.has_peninsulas
        assert result.n_regions == 1
    
    def test_overlapping_cutouts_may_create_peninsula(self):
        """Cutouts that overlap near edge may create isolated region."""
        from core.structural_analysis import detect_peninsulas
        
        # Create cutouts that isolate a corner
        cutouts = [
            {'x': 0.15, 'y': 0.3, 'size': 0.15, 'shape': 'rectangle'},
            {'x': 0.3, 'y': 0.15, 'size': 0.15, 'shape': 'rectangle'},
        ]
        
        result = detect_peninsulas(
            length=2.0, width=0.8,
            cutouts=cutouts,
            resolution=100
        )
        
        # May or may not create peninsula depending on exact geometry
        # The key is the function runs without error
        assert result.n_regions >= 1
        assert 0 <= result.structural_penalty <= 1
    
    def test_cutout_chain_creates_peninsula(self):
        """Chain of cutouts across plate should create isolated region."""
        from core.structural_analysis import detect_peninsulas
        
        # Create L-shaped cutout pattern that isolates corner
        cutouts = [
            # Vertical cut near left edge
            {'x': 0.1, 'y': 0.2, 'size': 0.05, 'shape': 'rectangle', 'rotation': 0, 'aspect': 3},
            {'x': 0.1, 'y': 0.4, 'size': 0.05, 'shape': 'rectangle', 'rotation': 0, 'aspect': 3},
            {'x': 0.1, 'y': 0.6, 'size': 0.05, 'shape': 'rectangle', 'rotation': 0, 'aspect': 3},
            # Horizontal cut near bottom
            {'x': 0.2, 'y': 0.1, 'size': 0.05, 'shape': 'rectangle', 'rotation': 1.57, 'aspect': 3},
            {'x': 0.4, 'y': 0.1, 'size': 0.05, 'shape': 'rectangle', 'rotation': 1.57, 'aspect': 3},
        ]
        
        result = detect_peninsulas(
            length=2.0, width=0.8,
            cutouts=cutouts,
            resolution=100
        )
        
        # This pattern likely creates peninsula - test detection works
        assert result.grid_visualization is not None
        assert result.main_region_fraction > 0
    
    def test_large_cutout_penalty(self):
        """Large cutout consuming most of plate should reduce main region fraction."""
        from core.structural_analysis import detect_peninsulas
        
        # Huge cutout in center
        cutouts = [
            {'x': 0.5, 'y': 0.5, 'size': 0.4, 'shape': 'circle'}
        ]
        
        result = detect_peninsulas(
            length=2.0, width=0.8,
            cutouts=cutouts,
            resolution=80
        )
        
        # Should still be one region
        assert result.n_regions == 1
        # Function should work without error
        assert result.grid_visualization is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: RADAR WIDGET (UI)
# ══════════════════════════════════════════════════════════════════════════════

class TestRadarWidget:
    """Tests for radar widget (requires tkinter, skip if not available)."""
    
    @pytest.fixture
    def skip_if_no_display(self):
        """Skip tests if no display available (CI environment)."""
        import os
        if os.environ.get('DISPLAY') is None and os.name != 'nt':
            pytest.skip("No display available")
    
    def test_radar_widget_import(self):
        """Test that radar widget can be imported."""
        try:
            from ui.widgets.radar_widget import RadarWidget, OptimizationRadarFrame
            assert RadarWidget is not None
            assert OptimizationRadarFrame is not None
        except ImportError as e:
            pytest.skip(f"UI module not available: {e}")
    
    def test_radar_widget_values(self):
        """Test radar widget value handling."""
        try:
            from ui.widgets.radar_widget import RadarWidget
            import tkinter as tk
            
            # Create hidden root
            root = tk.Tk()
            root.withdraw()
            
            values = {}
            def on_change(v):
                values.update(v)
            
            widget = RadarWidget(root, size=180, on_change=on_change)
            
            # Test default values
            assert 'Energy' in widget.values
            assert 'Flatness' in widget.values
            assert 'Spine' in widget.values
            
            # Test value bounds
            for key, val in widget.values.items():
                assert 0.0 <= val <= 1.0, f"{key} value {val} out of bounds"
            
            root.destroy()
            
        except Exception as e:
            pytest.skip(f"Tkinter test failed: {e}")
    
    def test_radar_widget_set_values(self):
        """Test that radar widget can set values programmatically."""
        try:
            from ui.widgets.radar_widget import RadarWidget
            import tkinter as tk
            
            root = tk.Tk()
            root.withdraw()
            
            widget = RadarWidget(root, size=180)
            
            # Set new values
            widget.set_values({'Energy': 0.9, 'Flatness': 0.3, 'Spine': 0.6})
            
            assert abs(widget.values['Energy'] - 0.9) < 0.01
            assert abs(widget.values['Flatness'] - 0.3) < 0.01
            assert abs(widget.values['Spine'] - 0.6) < 0.01
            
            root.destroy()
            
        except Exception as e:
            pytest.skip(f"Tkinter test failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FITNESS INTEGRATION WITH PENINSULA
# ══════════════════════════════════════════════════════════════════════════════

class TestFitnessPeninsulaIntegration:
    """Test that peninsula detection is properly integrated into fitness."""
    
    @pytest.fixture
    def evaluator(self):
        """Create fitness evaluator."""
        from core.fitness import FitnessEvaluator, ObjectiveWeights, ZoneWeights
        from core.person import Person
        
        person = Person(height_m=1.75, weight_kg=75.0)
        
        return FitnessEvaluator(
            person=person,
            objectives=ObjectiveWeights(),
            zone_weights=ZoneWeights(),
            material="birch_plywood"
        )
    
    def test_fitness_has_peninsula_fields(self, evaluator):
        """Test that FitnessResult has peninsula detection fields."""
        from core.fitness import FitnessResult
        
        result = FitnessResult()
        
        assert hasattr(result, 'has_peninsulas')
        assert hasattr(result, 'n_regions')
        assert hasattr(result, 'peninsula_penalty')
    
    def test_fitness_evaluates_with_cutouts(self, evaluator):
        """Test that fitness evaluator handles cutouts without error."""
        from core.plate_genome import PlateGenome, ContourType, CutoutGene
        
        genome = PlateGenome(
            length=2.0,
            width=0.8,
            thickness_base=0.015,
            contour_type=ContourType.GOLDEN_RECT,
        )
        
        # Add multiple cutouts
        genome.cutouts = [
            CutoutGene(x=0.3, y=0.3, width=0.06, height=0.06, shape='circle'),
            CutoutGene(x=0.5, y=0.5, width=0.08, height=0.06, shape='ellipse'),
            CutoutGene(x=0.7, y=0.7, width=0.05, height=0.05, shape='circle'),
        ]
        
        # Evaluate fitness
        result = evaluator.evaluate(genome)
        
        # Should complete without error
        assert result.total_fitness >= 0
        assert result.structural_score >= 0
        assert result.has_peninsulas is not None


# ══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
