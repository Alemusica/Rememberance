"""
Test structural spring support enforcement.

ISSUE: In the screenshot, springs were positioned in the CENTER of the plate
instead of at the CORNERS. This is structurally invalid - the plate cannot
stand if springs are clustered in the middle!

SOLUTION: enforce_structural_support() forces springs to valid positions
BEFORE any acoustic optimization.
"""

from src.core.plate_genome import PlateGenome, SpringSupportGene, ContourType
import numpy as np


def test_spring_coverage_detection():
    """Test that we can detect poor spring coverage."""
    # Springs in center (BAD - like the screenshot)
    bad_springs = [
        SpringSupportGene(x=0.35, y=0.40, stiffness_n_m=8000),
        SpringSupportGene(x=0.65, y=0.40, stiffness_n_m=8000),
        SpringSupportGene(x=0.35, y=0.60, stiffness_n_m=8000),
        SpringSupportGene(x=0.65, y=0.60, stiffness_n_m=8000),
        SpringSupportGene(x=0.50, y=0.50, stiffness_n_m=8000),
    ]
    
    genome = PlateGenome(
        length=1.9, width=0.68,
        spring_supports=bad_springs,
    )
    
    coverage = genome._get_spring_coverage()
    
    print(f"Bad springs coverage: {coverage['score']*100:.0f}%")
    print(f"Missing corners: {coverage['missing']}")
    
    # Should detect that corners are NOT covered
    assert coverage['score'] < 0.5, "Should detect poor coverage"
    assert len(coverage['missing']) >= 3, "Should be missing at least 3 corners"


def test_structural_support_enforcement():
    """Test that enforce_structural_support() fixes bad spring positions."""
    # Springs in center (BAD)
    bad_springs = [
        SpringSupportGene(x=0.35, y=0.40, stiffness_n_m=8000),
        SpringSupportGene(x=0.65, y=0.40, stiffness_n_m=8000),
        SpringSupportGene(x=0.35, y=0.60, stiffness_n_m=8000),
        SpringSupportGene(x=0.65, y=0.60, stiffness_n_m=8000),
        SpringSupportGene(x=0.50, y=0.50, stiffness_n_m=8000),
    ]
    
    genome = PlateGenome(
        length=1.9, width=0.68,
        spring_supports=bad_springs,
    )
    
    # Before enforcement
    coverage_before = genome._get_spring_coverage()
    penalty_before = genome.structural_support_penalty()
    
    print(f"\nBEFORE enforcement:")
    print(f"  Coverage: {coverage_before['score']*100:.0f}%")
    print(f"  Penalty: {penalty_before*100:.0f}%")
    
    # Apply enforcement
    fixed = genome.enforce_structural_support()
    
    # After enforcement
    coverage_after = fixed._get_spring_coverage()
    penalty_after = fixed.structural_support_penalty()
    
    print(f"\nAFTER enforcement:")
    print(f"  Coverage: {coverage_after['score']*100:.0f}%")
    print(f"  Penalty: {penalty_after*100:.0f}%")
    
    for i, s in enumerate(fixed.spring_supports):
        is_corner, region = s.is_in_corner_region()
        print(f"    Spring {i+1}: x={s.x:.2f}, y={s.y:.2f} → {region}")
    
    # Should improve coverage significantly
    assert coverage_after['score'] > coverage_before['score'], "Should improve coverage"
    assert coverage_after['score'] >= 0.75, "Should cover at least 3/4 corners"
    assert penalty_after < penalty_before, "Should reduce penalty"


def test_mutation_preserves_structural_support():
    """Test that mutation doesn't break structural support."""
    # Good springs (at corners)
    good_springs = [
        SpringSupportGene(x=0.10, y=0.10, stiffness_n_m=8000),  # feet_left
        SpringSupportGene(x=0.90, y=0.10, stiffness_n_m=8000),  # feet_right
        SpringSupportGene(x=0.10, y=0.90, stiffness_n_m=8000),  # head_left
        SpringSupportGene(x=0.90, y=0.90, stiffness_n_m=8000),  # head_right
        SpringSupportGene(x=0.50, y=0.50, stiffness_n_m=8000),  # center
    ]
    
    genome = PlateGenome(
        length=1.9, width=0.68,
        spring_supports=good_springs,
        enforce_symmetry=True,
    )
    
    print("\nMutation test (5 iterations):")
    
    current = genome
    for i in range(5):
        mutated = current.mutate()
        coverage = mutated._get_spring_coverage()
        penalty = mutated.structural_support_penalty()
        
        print(f"  Gen {i+1}: coverage={coverage['score']*100:.0f}%, penalty={penalty*100:.0f}%")
        
        # Should maintain structural support through mutations
        assert coverage['score'] >= 0.5, f"Gen {i+1}: Coverage dropped too low!"
        
        current = mutated
    
    print("  ✅ Structural support maintained through mutations")


def test_penalty_calculation():
    """Test that penalty correctly identifies clustered springs."""
    # Clustered springs (all near center)
    clustered = PlateGenome(
        length=1.9, width=0.68,
        spring_supports=[
            SpringSupportGene(x=0.45, y=0.45, stiffness_n_m=8000),
            SpringSupportGene(x=0.55, y=0.45, stiffness_n_m=8000),
            SpringSupportGene(x=0.45, y=0.55, stiffness_n_m=8000),
            SpringSupportGene(x=0.55, y=0.55, stiffness_n_m=8000),
        ],
    )
    
    # Spread springs (at corners)
    spread = PlateGenome(
        length=1.9, width=0.68,
        spring_supports=[
            SpringSupportGene(x=0.10, y=0.10, stiffness_n_m=8000),
            SpringSupportGene(x=0.90, y=0.10, stiffness_n_m=8000),
            SpringSupportGene(x=0.10, y=0.90, stiffness_n_m=8000),
            SpringSupportGene(x=0.90, y=0.90, stiffness_n_m=8000),
        ],
    )
    
    penalty_clustered = clustered.structural_support_penalty()
    penalty_spread = spread.structural_support_penalty()
    
    print(f"\nPenalty comparison:")
    print(f"  Clustered springs: {penalty_clustered*100:.0f}%")
    print(f"  Spread springs: {penalty_spread*100:.0f}%")
    
    assert penalty_clustered > penalty_spread, "Clustered should have higher penalty"
    assert penalty_spread < 0.3, "Spread springs should have low penalty"


if __name__ == "__main__":
    print("=" * 60)
    print("STRUCTURAL SPRING SUPPORT TESTS")
    print("=" * 60)
    
    test_spring_coverage_detection()
    print("\n✅ Coverage detection: PASSED")
    
    test_structural_support_enforcement()
    print("\n✅ Structural enforcement: PASSED")
    
    test_mutation_preserves_structural_support()
    print("\n✅ Mutation preservation: PASSED")
    
    test_penalty_calculation()
    print("\n✅ Penalty calculation: PASSED")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
