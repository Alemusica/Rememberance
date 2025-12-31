#!/usr/bin/env python3
"""Test crossover properly handles max_grooves=0."""

import sys
sys.path.insert(0, 'src')

from core.plate_genome import PlateGenome, ContourType, GrooveGene

def test_crossover_respects_zero_grooves():
    """When either parent has max_grooves=0, child should have no grooves."""
    g1 = PlateGenome(
        length=1.8, width=0.7, thickness_base=0.015,
        contour_type=ContourType.RECTANGLE,
        max_grooves=0, max_cutouts=4,
    )

    g2 = PlateGenome(
        length=1.8, width=0.7, thickness_base=0.015,
        contour_type=ContourType.RECTANGLE,
        max_grooves=8, max_cutouts=4,
    )
    g2.grooves = [
        GrooveGene(x=0.3, y=0.5, length=0.1, angle=0.0, depth=0.3, width_mm=5.0),
        GrooveGene(x=0.7, y=0.5, length=0.1, angle=0.0, depth=0.3, width_mm=5.0),
    ]

    print(f"g1: max_grooves={g1.max_grooves}, grooves={len(g1.grooves)}")
    print(f"g2: max_grooves={g2.max_grooves}, grooves={len(g2.grooves)}")

    child1 = g1.crossover(g2)
    print(f"g1.crossover(g2): max_grooves={child1.max_grooves}, grooves={len(child1.grooves)}")
    assert child1.max_grooves == 0, f"Expected max_grooves=0, got {child1.max_grooves}"
    assert len(child1.grooves) == 0, f"Expected 0 grooves, got {len(child1.grooves)}"

    child2 = g2.crossover(g1)
    print(f"g2.crossover(g1): max_grooves={child2.max_grooves}, grooves={len(child2.grooves)}")
    assert child2.max_grooves == 0, f"Expected max_grooves=0, got {child2.max_grooves}"
    assert len(child2.grooves) == 0, f"Expected 0 grooves, got {len(child2.grooves)}"

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_crossover_respects_zero_grooves()
