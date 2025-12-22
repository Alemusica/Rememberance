#!/usr/bin/env python3
"""
Test script for Rememberance Web API
Verifies all endpoints are working correctly
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:5001"

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        
    def test(self, name: str, method: str, endpoint: str, data: Dict[str, Any] = None, expected_status: int = 200):
        """Test an API endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == expected_status:
                print(f"✅ {name}")
                self.passed += 1
                return response.json() if response.content else None
            else:
                print(f"❌ {name} - Status {response.status_code}, expected {expected_status}")
                print(f"   Response: {response.text[:200]}")
                self.failed += 1
                return None
                
        except Exception as e:
            print(f"❌ {name} - Exception: {e}")
            self.failed += 1
            return None
    
    def test_mode_cycle(self, mode_name: str, start_endpoint: str, stop_endpoint: str, start_params: Dict[str, Any]):
        """Test a complete mode cycle: start -> status -> stop"""
        print(f"\n🧪 Testing {mode_name} mode...")
        
        # Start mode
        result = self.test(
            f"Start {mode_name}",
            "POST",
            start_endpoint,
            start_params
        )
        
        if result:
            time.sleep(0.5)
            
            # Check status
            status = self.test(f"{mode_name} status check", "GET", "/api/status")
            if status and status.get("mode") == mode_name.lower():
                print(f"   ✓ Mode confirmed: {status['mode']}")
            else:
                print(f"   ⚠️ Mode mismatch: expected {mode_name.lower()}, got {status.get('mode') if status else 'None'}")
            
            time.sleep(0.5)
            
            # Stop mode
            self.test(f"Stop {mode_name}", "POST", stop_endpoint)
            
            time.sleep(0.5)
            
            # Verify idle
            status = self.test(f"{mode_name} idle check", "GET", "/api/status")
            if status and status.get("mode") == "idle":
                print(f"   ✓ Returned to idle")
            else:
                print(f"   ⚠️ Not idle after stop: {status.get('mode') if status else 'None'}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║     REMEMBERANCE WEB API TEST SUITE                       ║")
        print("╚════════════════════════════════════════════════════════════╝\n")
        
        # Basic connectivity
        print("📡 Testing basic connectivity...")
        status = self.test("API health check", "GET", "/api/test")
        if status:
            print(f"   Backend: {status.get('backend')}")
            print(f"   Modules loaded: {sum(status.get('modules', {}).values())}/4")
            for module, loaded in status.get('modules', {}).items():
                print(f"     {'✅' if loaded else '❌'} {module}")
        
        # Test each mode
        self.test_mode_cycle(
            "EMDR",
            "/api/emdr/start",
            "/api/emdr/stop",
            {"speed": 1.0, "freq": 432.0, "mode": "golden_phase", "amplitude": 0.5}
        )
        
        self.test_mode_cycle(
            "Vibroacoustic",
            "/api/vibro/start",
            "/api/vibro/stop",
            {"freq": 432.0, "modulation": 0.1}
        )
        
        self.test_mode_cycle(
            "Harmonic",
            "/api/harmonic/start",
            "/api/harmonic/stop",
            {"fundamental": 432.0, "num_harmonics": 5, "mode": "fibonacci", "amplitude": 0.5, "growth_enabled": False}
        )
        
        self.test_mode_cycle(
            "Spectral",
            "/api/spectral/start",
            "/api/spectral/stop",
            {"element": "hydrogen", "phase_mode": "golden", "amplitude": 0.5}
        )
        
        # Test element list
        print(f"\n🧪 Testing utility endpoints...")
        elements = self.test("Get spectral elements", "GET", "/api/spectral/elements")
        if elements:
            print(f"   Available elements: {', '.join(elements.get('elements', []))}")
        
        # Summary
        print("\n" + "="*60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Success rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
        print("="*60)
        
        return self.failed == 0


if __name__ == "__main__":
    import sys
    
    tester = APITester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
