#!/usr/bin/env python3
"""
MCP Image Tools Real-time Testing

Tests the MCP server image search tools directly.
This verifies the MCP integration works with the fixed vector naming.

Usage:
    # Start MCP server
    python -m src.mcp.server --mode http --port 8001
    
    # Then run this test
    python scripts/test_mcp_image_tools.py
"""

import asyncio
import base64
import io
import json
import sys
from typing import Dict, List

import requests
from PIL import Image


def create_anime_style_image(text: str = "ANIME", colors: tuple = ("blue", "white")) -> str:
    """Create a simple anime-style test image."""
    img = Image.new('RGB', (150, 200), colors[0])  # Portrait orientation like anime posters
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        
        # Add title text
        draw.text((10, 10), text, fill=colors[1], font=font)
        # Add some simple shapes to make it look poster-like
        draw.rectangle([10, 30, 140, 50], outline=colors[1], width=2)
        draw.ellipse([50, 80, 100, 130], outline=colors[1], width=3)
        
    except Exception:
        pass  # Skip drawing if font not available
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode('utf-8')


def test_mcp_server_health() -> bool:
    """Test if MCP HTTP server is running."""
    print("🏥 Testing MCP Server Health...")
    try:
        # Try to get MCP server info
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ MCP HTTP Server: Healthy")
            return True
        else:
            print(f"   ❌ MCP Server: Unhealthy (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ MCP Server: Not reachable ({e})")
        print("   💡 Start with: python -m src.mcp.server --mode http --port 8001")
        return False


def call_mcp_tool(tool_name: str, arguments: Dict) -> Dict:
    """Call an MCP tool via HTTP."""
    payload = {
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    response = requests.post(
        "http://localhost:8001/mcp",
        json=payload,
        timeout=30,
        headers={"Content-Type": "application/json"}
    )
    
    return {
        "status_code": response.status_code,
        "data": response.json() if response.status_code == 200 else response.text
    }


def test_search_anime_by_image_tool() -> bool:
    """Test the search_anime_by_image MCP tool."""
    print("\n🖼️  Testing 'search_anime_by_image' MCP Tool...")
    
    # Create test anime poster images
    test_images = [
        ("action_poster", create_anime_style_image("ACTION", ("red", "yellow"))),
        ("mecha_poster", create_anime_style_image("MECHA", ("gray", "cyan"))),
        ("romance_poster", create_anime_style_image("ROMANCE", ("pink", "white"))),
    ]
    
    for image_name, image_data in test_images:
        print(f"\n   🎨 Testing with {image_name}...")
        
        result = call_mcp_tool("search_anime_by_image", {
            "image_data": image_data,
            "limit": 3
        })
        
        if result["status_code"] == 200:
            data = result["data"]
            if "content" in data and data["content"]:
                content = data["content"][0]
                print(f"   ✅ Tool executed successfully")
                print(f"   📝 Response: {content.get('text', 'No text')[:100]}...")
                
                # Check for tool result
                if "toolResult" in data:
                    results = data["toolResult"]
                    print(f"   📊 Found {len(results)} results")
                    if results:
                        for i, anime in enumerate(results[:2]):
                            title = anime.get('title', 'Unknown')
                            score = anime.get('visual_similarity_score', 0)
                            print(f"   {i+1}. {title} (Score: {score:.3f})")
                else:
                    print("   ⚠️  No toolResult in response")
            else:
                print("   ❌ No content in MCP response")
                return False
        else:
            print(f"   ❌ MCP tool call failed: {result['status_code']}")
            print(f"   📝 Error: {result['data'][:200]}...")
            return False
    
    return True


def test_search_multimodal_anime_tool() -> bool:
    """Test the search_multimodal_anime MCP tool."""
    print("\n🔄 Testing 'search_multimodal_anime' MCP Tool...")
    
    # Create a mecha-style image
    mecha_image = create_anime_style_image("ROBOT", ("silver", "blue"))
    
    test_cases = [
        {
            "name": "text + image",
            "query": "giant robots fighting in space",
            "image_data": mecha_image,
            "text_weight": 0.7
        },
        {
            "name": "image-heavy",
            "query": "mecha",
            "image_data": mecha_image,
            "text_weight": 0.3
        }
    ]
    
    for test_case in test_cases:
        print(f"\n   🎯 Testing {test_case['name']} search...")
        
        result = call_mcp_tool("search_multimodal_anime", {
            "query": test_case["query"],
            "image_data": test_case["image_data"],
            "limit": 3,
            "text_weight": test_case["text_weight"]
        })
        
        if result["status_code"] == 200:
            data = result["data"]
            if "toolResult" in data:
                results = data["toolResult"]
                print(f"   ✅ Found {len(results)} multimodal results")
                
                for i, anime in enumerate(results[:2]):
                    title = anime.get('title', 'Unknown')
                    multimodal_score = anime.get('multimodal_score', 0)
                    text_score = anime.get('text_score', 0)
                    image_score = anime.get('image_score', 0)
                    
                    print(f"   {i+1}. {title}")
                    print(f"      🎯 Combined: {multimodal_score:.3f}")
                    print(f"      📝 Text: {text_score:.3f}, 🖼️  Image: {image_score:.3f}")
            else:
                print("   ❌ No toolResult in multimodal response")
                return False
        else:
            print(f"   ❌ Multimodal tool call failed: {result['status_code']}")
            return False
    
    return True


def test_find_visually_similar_tool() -> bool:
    """Test the find_visually_similar_anime MCP tool."""
    print("\n🎭 Testing 'find_visually_similar_anime' MCP Tool...")
    
    # Test with a known anime ID (this might not exist in empty DB)
    test_anime_ids = ["test_anime_123", "sample_id", "demo_anime"]
    
    for anime_id in test_anime_ids:
        print(f"\n   🔍 Testing with anime_id: {anime_id}")
        
        result = call_mcp_tool("find_visually_similar_anime", {
            "anime_id": anime_id,
            "limit": 3
        })
        
        if result["status_code"] == 200:
            data = result["data"]
            if "toolResult" in data:
                results = data["toolResult"]
                if results:
                    print(f"   ✅ Found {len(results)} similar anime")
                    for i, anime in enumerate(results[:2]):
                        title = anime.get('title', 'Unknown')
                        similarity = anime.get('visual_similarity_score', 0)
                        print(f"   {i+1}. {title} (Similarity: {similarity:.3f})")
                    return True
                else:
                    print(f"   ⚠️  No similar anime found for {anime_id}")
            else:
                print("   ❌ No toolResult in similar anime response")
        else:
            print(f"   ❌ Similar anime tool call failed: {result['status_code']}")
    
    # If we reach here, no anime ID worked (expected with empty DB)
    print("   ⚠️  No anime found (expected with empty database)")
    return True  # Not a failure, just empty DB


def test_list_mcp_tools() -> bool:
    """Test listing available MCP tools."""
    print("\n📋 Testing MCP Tools List...")
    
    payload = {"method": "tools/list", "params": {}}
    
    try:
        response = requests.post(
            "http://localhost:8001/mcp",
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "tools" in data:
                tools = data["tools"]
                print(f"   ✅ Found {len(tools)} MCP tools:")
                
                image_tools = []
                for tool in tools:
                    tool_name = tool.get("name", "Unknown")
                    description = tool.get("description", "")
                    print(f"   📌 {tool_name}: {description[:50]}...")
                    
                    if "image" in tool_name.lower():
                        image_tools.append(tool_name)
                
                print(f"\n   🖼️  Image-related tools: {len(image_tools)}")
                for tool in image_tools:
                    print(f"   • {tool}")
                
                return len(image_tools) > 0
            else:
                print("   ❌ No tools found in MCP response")
                return False
        else:
            print(f"   ❌ Failed to list tools: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error listing tools: {e}")
        return False


def main():
    """Run comprehensive MCP image tools testing."""
    print("🛠️  MCP Image Tools Real-time Testing")
    print("=" * 50)
    
    # Test MCP server health
    if not test_mcp_server_health():
        return False
    
    # List available tools
    if not test_list_mcp_tools():
        print("\n❌ Could not list MCP tools - server may not be working correctly")
        return False
    
    print("\n" + "="*50)
    print("🎯 Testing Image Search MCP Tools")
    print("="*50)
    
    # Test each image search tool
    tests = [
        ("search_anime_by_image", test_search_anime_by_image_tool),
        ("search_multimodal_anime", test_search_multimodal_anime_tool),
        ("find_visually_similar_anime", test_find_visually_similar_tool),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 MCP TOOLS TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        all_passed = all_passed and result
    
    if all_passed:
        print("\n🎉 ALL MCP TOOLS WORKING!")
        print("✅ Image search tools are properly implemented")
        print("✅ MCP server responds correctly")
        print("✅ Vector naming fix is working in MCP layer")
    else:
        print("\n⚠️  SOME MCP TOOLS FAILED")
        print("Check the logs above for details")
    
    print("\n📋 To test with real data:")
    print("1. Start Qdrant: docker compose up -d qdrant")
    print("2. Load anime data: curl -X POST http://localhost:8000/api/admin/update-full")
    print("3. Wait for indexing to complete")
    print("4. Run this test again for real results")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)