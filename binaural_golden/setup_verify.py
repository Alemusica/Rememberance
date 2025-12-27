#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SETUP E VERIFICA DIPENDENZE - Golden Binaural Platform
═══════════════════════════════════════════════════════════════════════════════

Esegui: python setup_verify.py

Questo script:
1. Verifica dipendenze installate
2. Mostra dipendenze mancanti
3. Genera comando pip install
"""

import sys
import subprocess
from typing import List, Tuple, Dict


def check_package(name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Verifica se un package è installato.
    
    Returns:
        (installed, version)
    """
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def main():
    """Main verification."""
    print("═" * 70)
    print("🔍 VERIFICA DIPENDENZE - Golden Binaural Platform")
    print("═" * 70)
    print()
    
    # Definizione dipendenze
    packages = [
        # (pip_name, import_name, descrizione, obbligatorio)
        ("numpy", "numpy", "Array numerici", True),
        ("scipy", "scipy", "Funzioni scientifiche", True),
        ("matplotlib", "matplotlib", "Visualizzazione", True),
        ("scikit-fem", "skfem", "FEM modal analysis", True),
        ("meshio", "meshio", "I/O mesh", False),
        ("jax", "jax", "Differentiable computing", True),
        ("jaxlib", "jaxlib", "JAX backend", True),
        ("pytest", "pytest", "Testing", False),
        ("sounddevice", "sounddevice", "Audio I/O", False),
        ("PyQt5", "PyQt5", "GUI Framework", False),
    ]
    
    installed = []
    missing = []
    
    print("Package              Status      Versione      Descrizione")
    print("-" * 70)
    
    for pip_name, import_name, desc, required in packages:
        is_installed, version = check_package(pip_name, import_name)
        
        if is_installed:
            status = "✅ OK"
            installed.append(pip_name)
            version_str = version[:12] if version else "-"
        else:
            if required:
                status = "❌ MANCA"
                missing.append(pip_name)
            else:
                status = "⚠️  N/A"
            version_str = "-"
        
        print(f"{pip_name:<20} {status:<12} {version_str:<14} {desc}")
    
    print()
    print("═" * 70)
    
    # Summary
    print(f"\n📊 RIEPILOGO:")
    print(f"   Installati: {len(installed)}/{len(packages)}")
    print(f"   Mancanti (obbligatori): {len(missing)}")
    
    if missing:
        print(f"\n⚠️  DIPENDENZE MANCANTI:")
        for pkg in missing:
            print(f"   • {pkg}")
        
        # Generate install command
        install_cmd = f"pip install {' '.join(missing)}"
        
        print(f"\n📦 COMANDO PER INSTALLARE:")
        print(f"   {install_cmd}")
        
        print("\n" + "═" * 70)
        print("💡 Oppure installa tutto con:")
        print("   pip install -r requirements.txt")
        print("═" * 70)
        
        return 1
    else:
        print("\n✅ TUTTE LE DIPENDENZE OBBLIGATORIE SONO INSTALLATE!")
        
        # Test import dei nostri moduli
        print("\n🔬 VERIFICA MODULI INTERNI:")
        
        sys.path.insert(0, 'src')
        
        internal_modules = [
            ("core.body_zones", "Zone corporee"),
            ("core.coupled_system", "Sistema accoppiato"),
            ("core.iterative_optimizer", "Ottimizzatore SIMP/RAMP"),
            ("core.plate_fem", "FEM tavola"),
            ("core.plate_optimizer", "API ottimizzazione"),
        ]
        
        for module_name, desc in internal_modules:
            try:
                __import__(module_name)
                print(f"   ✅ {module_name:<30} {desc}")
            except ImportError as e:
                print(f"   ❌ {module_name:<30} Errore: {e}")
        
        # Test JAX FEM
        print("\n🧮 TEST JAX FEM:")
        try:
            from core.jax_plate_fem import JAXPlateFEM
            print("   ✅ JAXPlateFEM importato")
            
            # Quick test
            import numpy as np
            fem = JAXPlateFEM(length=2.0, width=0.6, nx=20, ny=6)
            density = np.ones((20, 6)) * 0.5
            freqs = fem.compute_frequencies(density, n_modes=3)
            print(f"   ✅ Test FEM: Prime 3 frequenze = {freqs[:3]}")
            
        except ImportError as e:
            print(f"   ⚠️  JAX non disponibile: {e}")
            print("      Il sistema userà il solver semplificato")
        except Exception as e:
            print(f"   ❌ Errore test JAX FEM: {e}")
        
        print("\n" + "═" * 70)
        print("🎉 SISTEMA PRONTO PER L'USO!")
        print("═" * 70)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
