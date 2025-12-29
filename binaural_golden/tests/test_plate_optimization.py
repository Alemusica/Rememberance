#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
TEST COMPLETO - Sistema Ottimizzazione Tavola Vibroacustica
═══════════════════════════════════════════════════════════════════════════════

Test di integrazione per verificare tutti i moduli:
- body_zones.py: Modello zone corporee
- coupled_system.py: Sistema accoppiato tavola-corpo
- iterative_optimizer.py: SIMP/RAMP topology optimization
- jax_plate_fem.py: FEM differenziabile (se JAX disponibile)
- plate_optimizer.py: API high-level

Esegui con: pytest tests/test_plate_optimization.py -v
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBodyZones:
    """Test del modello zone corporee."""
    
    def test_chakra_zones_creation(self):
        """Verifica creazione zone chakra."""
        from core.body_zones import create_chakra_zones
        
        zones = create_chakra_zones()
        
        assert len(zones) == 7, "Devono esserci 7 chakra"
        
        # Verifica range frequenze
        for zone in zones:
            assert zone.f_min < zone.f_max
            assert zone.f_center == (zone.f_min + zone.f_max) / 2
            assert 0 < zone.position <= 1.0
            assert 0 < zone.weight <= 2.0
    
    def test_vat_zones_creation(self):
        """Verifica creazione zone VAT terapeutico."""
        from core.body_zones import create_vat_therapy_zones
        
        zones = create_vat_therapy_zones()
        
        assert len(zones) == 5
        
        # Verifica frequenze VAT standard
        freq_names = [z.name for z in zones]
        assert "Rilassamento Profondo" in freq_names
    
    def test_body_resonance_zones(self):
        """Verifica zone risonanza fisiologica."""
        from core.body_zones import create_body_resonance_zones
        
        zones = create_body_resonance_zones()
        
        assert len(zones) == 8
        
        # Verifica che le frequenze siano ordinate
        frequencies = [z.f_center for z in zones]
        for i in range(1, len(frequencies)):
            assert frequencies[i] > frequencies[i-1], \
                "Zone devono essere ordinate per frequenza"
    
    def test_body_zone_model(self):
        """Test completo BodyZoneModel."""
        from core.body_zones import BodyZoneModel
        
        model = BodyZoneModel(preset="chakra")
        
        # Ottieni frequenze target
        targets = model.get_target_frequencies()
        assert len(targets) == 7
        
        # Verifica coverage
        coverage = model.get_frequency_coverage()
        assert 'total_zones' in coverage
        assert 'min_freq' in coverage
        assert 'max_freq' in coverage
        
        # Test zona più vicina
        closest = model.find_closest_zone(40.0)
        assert closest is not None
        assert closest.name == "Root (Muladhara)"  # 25-50 Hz
    
    def test_model_preset_switch(self):
        """Verifica cambio preset."""
        from core.body_zones import BodyZoneModel
        
        model = BodyZoneModel(preset="chakra")
        assert len(model.zones) == 7
        
        model.set_preset("vat")
        assert len(model.zones) == 5
        
        model.set_preset("body_resonance")
        assert len(model.zones) == 8


class TestCoupledSystem:
    """Test sistema accoppiato tavola-corpo."""
    
    def test_basic_coupled_system(self):
        """Test sistema base 2-DOF."""
        from core.coupled_system import CoupledSystem
        
        system = CoupledSystem(
            m_plate=50.0,
            m_body=70.0,
            k_plate=1e5,
            k_coupling=5e4,
            c_coupling=500.0
        )
        
        # Calcola funzione di trasferimento
        freqs = np.linspace(10, 200, 100)
        H = system.transfer_function(freqs)
        
        assert len(H) == len(freqs)
        assert np.all(np.isfinite(H))
        
        # Trova risonanze
        resonances = system.find_resonances()
        assert len(resonances) >= 2  # Almeno 2 modi per 2-DOF
    
    def test_transmissibility(self):
        """Test trasmissibilità."""
        from core.coupled_system import CoupledSystem
        
        system = CoupledSystem()
        
        freqs = np.linspace(10, 200, 100)
        T = system.transmissibility(freqs)
        
        assert len(T) == len(freqs)
        assert np.all(T >= 0)  # Trasmissibilità non negativa
    
    def test_zone_coupled_system(self):
        """Test sistema con zone."""
        from core.coupled_system import ZoneCoupledSystem
        from core.body_zones import create_chakra_zones
        
        zones = create_chakra_zones()
        system = ZoneCoupledSystem(zones)
        
        # Ottimizza coupling per target 40 Hz
        optimal_k = system.optimize_coupling_for_target(40.0)
        
        assert optimal_k > 0
        assert 1e3 < optimal_k < 1e6  # Range ragionevole
        
        # Calcola score zone
        scores = system.compute_zone_scores()
        
        assert len(scores) == len(zones)
        for score in scores:
            assert 'zone' in score
            assert 'effectiveness' in score
            assert 0 <= score['effectiveness'] <= 1


class TestIterativeOptimizer:
    """Test ottimizzatore SIMP/RAMP."""
    
    def test_simp_interpolation(self):
        """Test interpolazione SIMP."""
        from core.iterative_optimizer import simp
        
        # Test bounds
        assert simp(0.0) == 1e-9  # epsilon
        assert simp(1.0) == 1.0  # Max
        
        # Test monotonia
        densities = np.linspace(0, 1, 100)
        stiffness = simp(densities)
        
        for i in range(1, len(stiffness)):
            assert stiffness[i] >= stiffness[i-1]
    
    def test_ramp_interpolation(self):
        """Test interpolazione RAMP."""
        from core.iterative_optimizer import ramp
        
        assert ramp(0.0) == 1e-9
        assert ramp(1.0) == 1.0
        
        # RAMP < SIMP per densità intermedie
        from core.iterative_optimizer import simp
        
        for rho in [0.3, 0.5, 0.7]:
            assert ramp(rho) <= simp(rho)
    
    def test_density_filter(self):
        """Test filtro densità."""
        from core.iterative_optimizer import density_filter
        
        # Crea pattern con rumore
        density = np.random.rand(40, 12)
        
        # Applica filtro
        filtered = density_filter(density, R=0.05)
        
        # Filtro deve smussare
        assert np.std(filtered) < np.std(density)
        
        # Deve preservare bounds
        assert np.all(filtered >= 0)
        assert np.all(filtered <= 1)
    
    def test_simple_fem_solver(self):
        """Test solver FEM semplificato."""
        from core.iterative_optimizer import simple_plate_fem
        
        density = np.ones((40, 12))
        freqs, sensitivity = simple_plate_fem(density, n_modes=5)
        
        assert len(freqs) == 5
        assert freqs[0] > 0  # Prima frequenza positiva
        
        # Frequenze ordinate
        for i in range(1, len(freqs)):
            assert freqs[i] >= freqs[i-1]
        
        # Sensitivity shape
        assert sensitivity.shape == (5, 40, 12)
    
    def test_zone_optimizer(self):
        """Test ottimizzatore con zone."""
        from core.iterative_optimizer import ZoneIterativeOptimizer
        from core.body_zones import create_chakra_zones
        
        zones = create_chakra_zones()
        optimizer = ZoneIterativeOptimizer(
            zones=zones,
            plate_length=2.0,
            plate_width=0.6,
            n_modes=5
        )
        
        # Verifica setup
        assert optimizer.target_frequencies is not None
        assert len(optimizer.weights) == 5
        
        # Test singola iterazione
        density = np.ones((40, 12)) * 0.5
        new_density = optimizer.oc_update(density)
        
        assert new_density.shape == density.shape
        assert np.all(new_density >= 0)
        assert np.all(new_density <= 1)


class TestCreateJaxFemSolver:
    """Test factory per JAX FEM solver."""
    
    def test_solver_creation(self):
        """Test creazione solver."""
        from core.iterative_optimizer import create_jax_fem_solver
        
        # Deve ritornare una funzione callable
        solver = create_jax_fem_solver(length=2.0, width=0.6)
        
        assert callable(solver)
        
        # Test chiamata
        density = np.ones((40, 12)) * 0.5
        freqs, sens = solver(density)
        
        assert len(freqs) > 0
        assert sens.shape[1:] == density.shape


class TestPlateOptimizer:
    """Test API high-level."""
    
    def test_zone_optimize_plate_basic(self):
        """Test ottimizzazione base."""
        from core.plate_optimizer import zone_optimize_plate
        
        result = zone_optimize_plate(
            target_frequencies=[40.0, 80.0, 120.0],
            plate_dims=(2.0, 0.6, 0.015),
            n_iterations=5,  # Poche iterazioni per test
            verbose=False
        )
        
        assert result is not None
        assert 'optimal_density' in result
        assert 'final_frequencies' in result
        assert 'convergence' in result
        
        # Verifica shape
        density = result['optimal_density']
        assert density.shape[0] > density.shape[1]  # length > width
    
    def test_zone_optimize_with_preset(self):
        """Test con preset zone."""
        from core.plate_optimizer import zone_optimize_plate
        
        result = zone_optimize_plate(
            preset="chakra",
            plate_dims=(2.0, 0.6, 0.015),
            n_iterations=3,
            verbose=False
        )
        
        # Con 7 chakra, target dovrebbe includere 7 frequenze
        assert len(result['target_frequencies']) == 7


class TestIntegration:
    """Test di integrazione end-to-end."""
    
    def test_full_optimization_pipeline(self):
        """Test pipeline completo."""
        from core.body_zones import BodyZoneModel
        from core.coupled_system import ZoneCoupledSystem
        from core.iterative_optimizer import ZoneIterativeOptimizer
        
        # 1. Crea modello zone
        model = BodyZoneModel(preset="vat")
        zones = model.zones
        
        # 2. Crea sistema accoppiato
        coupled = ZoneCoupledSystem(zones)
        
        # 3. Ottimizza coupling
        optimal_k = coupled.optimize_coupling_for_target(40.0)
        
        # 4. Crea ottimizzatore
        optimizer = ZoneIterativeOptimizer(
            zones=zones,
            plate_length=2.0,
            plate_width=0.6,
            n_modes=5
        )
        
        # 5. Esegui poche iterazioni
        density = optimizer.initialize_density()
        
        for _ in range(3):
            density = optimizer.oc_update(density)
        
        # Verifica convergenza
        final_freqs, _ = optimizer.fem_solver(density)
        targets = optimizer.target_frequencies
        
        # Almeno alcune frequenze dovrebbero essere vicine ai target
        errors = []
        for tf in targets:
            min_err = min(abs(f - tf) / tf for f in final_freqs)
            errors.append(min_err)
        
        avg_error = np.mean(errors)
        assert avg_error < 0.5  # Errore medio < 50%
    
    def test_export_format(self):
        """Test formato export risultati."""
        from core.plate_optimizer import zone_optimize_plate
        
        result = zone_optimize_plate(
            target_frequencies=[40.0, 80.0],
            n_iterations=2,
            verbose=False
        )
        
        # Verifica formato risultato
        assert isinstance(result, dict)
        
        required_keys = [
            'optimal_density',
            'final_frequencies', 
            'target_frequencies',
            'convergence',
            'history'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    
    print("=" * 70)
    print("TEST SISTEMA OTTIMIZZAZIONE TAVOLA VIBROACUSTICA")
    print("=" * 70)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
