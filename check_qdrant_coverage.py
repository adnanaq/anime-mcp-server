#!/usr/bin/env python3
"""
Script to analyze QdrantClient test coverage by examining source code
and identifying which methods and code paths need additional testing.
"""

import ast
import re
from pathlib import Path

def analyze_qdrant_client():
    """Analyze QdrantClient source code to identify methods and coverage needs."""
    
    qdrant_path = Path("src/vector/qdrant_client.py")
    test_path = Path("tests/unit/vector/test_qdrant_client.py")
    
    if not qdrant_path.exists():
        print(f"❌ QdrantClient not found at {qdrant_path}")
        return
        
    if not test_path.exists():
        print(f"❌ Test file not found at {test_path}")
        return
    
    # Read source code
    with open(qdrant_path, 'r') as f:
        source_code = f.read()
    
    # Read test file
    with open(test_path, 'r') as f:
        test_code = f.read()
    
    # Parse the source to find methods
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"❌ Syntax error in source: {e}")
        return
    
    # Find all class methods
    methods = []
    class_found = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QdrantClient":
            class_found = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
    
    if not class_found:
        print("❌ QdrantClient class not found")
        return
    
    print("📊 QdrantClient Methods Analysis:")
    print("=" * 50)
    
    tested_methods = []
    untested_methods = []
    
    for method in methods:
        if method.startswith('_'):
            # Check for private method testing patterns
            test_patterns = [
                f"test_{method}",
                f"test_{method[1:]}",  # Remove leading underscore
                method in test_code,
                f".{method}(" in test_code
            ]
        else:
            # Check for public method testing patterns
            test_patterns = [
                f"test_{method}",
                method in test_code,
                f".{method}(" in test_code
            ]
        
        if any(test_patterns):
            tested_methods.append(method)
        else:
            untested_methods.append(method)
    
    print(f"✅ Tested Methods ({len(tested_methods)}):")
    for method in sorted(tested_methods):
        print(f"   - {method}")
    
    print(f"\n❌ Methods Needing Tests ({len(untested_methods)}):")
    for method in sorted(untested_methods):
        print(f"   - {method}")
    
    # Analyze specific code patterns that need testing
    print(f"\n🔍 Code Patterns Analysis:")
    print("=" * 50)
    
    # Look for exception handling
    exception_count = source_code.count("except Exception")
    exception_as_count = source_code.count("except Exception as")
    print(f"Exception handlers: {exception_count + exception_as_count}")
    
    # Look for async methods
    async_count = source_code.count("async def")
    print(f"Async methods: {async_count}")
    
    # Look for conditional branches
    if_count = source_code.count("if ")
    elif_count = source_code.count("elif ")
    print(f"Conditional branches: {if_count + elif_count}")
    
    # Look for logging statements
    logger_count = source_code.count("logger.")
    print(f"Logger statements: {logger_count}")
    
    # Calculate approximate coverage based on methods
    total_methods = len(methods)
    tested_method_count = len(tested_methods)
    
    if total_methods > 0:
        method_coverage = (tested_method_count / total_methods) * 100
        print(f"\n📈 Estimated Method Coverage: {method_coverage:.1f}% ({tested_method_count}/{total_methods})")
    
    # Look for specific patterns that need testing
    print(f"\n🎯 Specific Coverage Needs:")
    print("=" * 50)
    
    patterns_to_check = [
        ("Multi-vector support", "_supports_multi_vector"),
        ("Vision processor", "vision_processor"),
        ("Image embeddings", "_create_image_embedding"),
        ("Collection management", "_ensure_collection_exists"),
        ("Error handling", "except Exception"),
        ("Async operations", "async def"),
        ("Filter building", "_build_filter"),
        ("Health checks", "health_check"),
        ("Statistics", "get_stats"),
    ]
    
    for description, pattern in patterns_to_check:
        if pattern in source_code:
            test_coverage = pattern in test_code or f"test_{pattern.replace('_', '')}" in test_code
            status = "✅" if test_coverage else "❌"
            print(f"   {status} {description}")
    
    print(f"\n📝 Recommendations for 100% Coverage:")
    print("=" * 50)
    print("1. Add tests for all untested methods listed above")
    print("2. Test all exception handling paths")
    print("3. Test async method error scenarios")
    print("4. Test multi-vector vs single-vector paths")
    print("5. Test vision processor initialization failures")
    print("6. Test collection creation edge cases")
    print("7. Test embedding generation with various inputs")
    print("8. Test filter building with edge cases")

if __name__ == "__main__":
    analyze_qdrant_client()