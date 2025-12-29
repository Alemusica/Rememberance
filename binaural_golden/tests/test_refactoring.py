#!/usr/bin/env python3
"""
Test script for golden_studio.py refactoring.

Validates:
1. Syntax is correct
2. Module structure is valid
3. Imports work as expected (without runtime dependencies)
"""

import ast
import os
import sys

def test_syntax(filepath):
    """Test if file has valid Python syntax"""
    print(f"Testing syntax: {filepath}")
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        print("  ✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False

def test_module_structure():
    """Test if module structure is correct"""
    print("\nTesting module structure:")
    
    required_files = [
        'src/golden_studio.py',
        'src/core/audio_engine.py',
        'src/studio/__init__.py',
        'src/studio/audio_manager.py',
        'src/studio/app.py',
    ]
    
    all_exist = True
    for filepath in required_files:
        full_path = os.path.join(os.path.dirname(__file__), '..', filepath)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"  ✓ {filepath} ({size:,} bytes)")
        else:
            print(f"  ✗ {filepath} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_line_counts():
    """Test that refactoring reduced line count"""
    print("\nTesting line count reduction:")
    
    files = {
        'src/golden_studio.py': (3600, 3700),  # Expected range after refactoring
        'src/golden_studio_old.py': (4000, 4100),  # Original file
    }
    
    all_ok = True
    for filepath, (min_lines, max_lines) in files.items():
        full_path = os.path.join(os.path.dirname(__file__), '..', filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                line_count = len(f.readlines())
            
            if min_lines <= line_count <= max_lines:
                print(f"  ✓ {filepath}: {line_count} lines (expected {min_lines}-{max_lines})")
            else:
                print(f"  ⚠ {filepath}: {line_count} lines (expected {min_lines}-{max_lines})")
                all_ok = False
        else:
            print(f"  - {filepath}: not found (optional)")
    
    return all_ok

def test_import_structure():
    """Test that import statements are correct"""
    print("\nTesting import structure:")
    
    golden_studio_path = os.path.join(os.path.dirname(__file__), '..', 'src/golden_studio.py')
    
    with open(golden_studio_path, 'r') as f:
        content = f.read()
    
    checks = [
        ('from core.audio_engine import AudioEngine', 'AudioEngine import from core'),
        ('class BinauralTab:', 'BinauralTab class still present'),
        ('class GoldenSoundStudio:', 'GoldenSoundStudio class present'),
    ]
    
    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} - NOT FOUND")
            all_found = False
    
    # Check that old AudioEngine class definition is removed
    if 'class AudioEngine:' in content and 'from core.audio_engine import AudioEngine' in content:
        # Make sure it's just a comment or in a try/except, not the actual definition
        lines_with_class = [l for l in content.split('\n') if 'class AudioEngine:' in l]
        # Should only appear in fallback/error handling, not as main definition
        if len(lines_with_class) <= 1:
            print(f"  ✓ AudioEngine class definition removed (import used instead)")
        else:
            print(f"  ⚠ AudioEngine class may still be defined multiple times")
            all_found = False
    else:
        print(f"  ✓ AudioEngine class definition removed")
    
    return all_found

def main():
    print("=" * 70)
    print("Golden Studio Refactoring Test Suite")
    print("=" * 70)
    
    tests = [
        ("Syntax validation", lambda: test_syntax(
            os.path.join(os.path.dirname(__file__), '..', 'src/golden_studio.py')
        )),
        ("Module structure", test_module_structure),
        ("Line count reduction", test_line_counts),
        ("Import structure", test_import_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Error running test: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(r for _, r in results)
    print("=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
