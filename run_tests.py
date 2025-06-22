#!/usr/bin/env python3
"""Test runner script for the anime MCP server."""
import subprocess
import sys
import os


def run_tests():
    """Run the test suite with appropriate options."""
    print("🧪 Running Anime MCP Server Test Suite")
    print("=" * 50)
    
    # Ensure we're in the project root
    if not os.path.exists("src"):
        print("❌ Error: Run this script from the project root directory")
        sys.exit(1)
    
    # Install test dependencies if not already installed
    try:
        import pytest
        import pytest_asyncio
        import pytest_cov
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Test commands
    commands = [
        # Unit tests - fast, no external dependencies
        {
            "name": "Unit Tests",
            "cmd": ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short", "-m", "unit"],
            "description": "Fast unit tests for individual components"
        },
        
        # Integration tests - require mocked external services
        {
            "name": "Integration Tests", 
            "cmd": ["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short", "-m", "integration"],
            "description": "Integration tests with mocked services"
        },
        
        # E2E tests - comprehensive but slower
        {
            "name": "End-to-End Tests",
            "cmd": ["python", "-m", "pytest", "tests/e2e/", "-v", "--tb=short", "-m", "e2e", "--maxfail=3"],
            "description": "Complete workflow tests"
        },
        
        # All tests with coverage
        {
            "name": "Full Test Suite with Coverage",
            "cmd": ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing", "--cov-report=html"],
            "description": "Complete test suite with coverage report"
        }
    ]
    
    # Run tests based on command line argument
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "unit":
            selected_commands = [commands[0]]
        elif test_type == "integration":
            selected_commands = [commands[1]]
        elif test_type == "e2e":
            selected_commands = [commands[2]]
        elif test_type == "coverage":
            selected_commands = [commands[3]]
        elif test_type == "fast":
            selected_commands = commands[:2]  # Unit + Integration
        else:
            print(f"❌ Unknown test type: {test_type}")
            print("Available options: unit, integration, e2e, coverage, fast")
            sys.exit(1)
    else:
        # Default: run unit and integration tests
        selected_commands = commands[:2]
    
    # Execute selected test commands
    success_count = 0
    total_count = len(selected_commands)
    
    for i, test_config in enumerate(selected_commands, 1):
        print(f"\n[{i}/{total_count}] {test_config['name']}")
        print(f"📋 {test_config['description']}")
        print("-" * 40)
        
        try:
            result = subprocess.run(test_config["cmd"], check=True)
            print(f"✅ {test_config['name']} passed")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {test_config['name']} failed with exit code {e.returncode}")
            if "coverage" not in test_config["name"].lower():
                # Don't exit on coverage failures, just report them
                continue
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {success_count}/{total_count} test suites passed")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()