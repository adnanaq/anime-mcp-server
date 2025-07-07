#!/usr/bin/env python3
"""
Test different MCP protocol interactions to understand what works
"""

import asyncio
import json
import subprocess
import sys
import time
import os

async def test_mcp_protocol():
    """Test various MCP protocol interactions."""
    print("🔍 Testing MCP Protocol Interactions")
    
    # Start server
    print("🚀 Starting server...")
    process = subprocess.Popen(
        [sys.executable, "-m", "src.anime_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        cwd=os.getcwd(),
        env=dict(os.environ, PYTHONPATH=os.getcwd())
    )
    
    try:
        # Wait for startup
        await asyncio.sleep(3)
        
        if process.poll() is not None:
            stderr = process.stderr.read()
            print(f"❌ Server died: {stderr}")
            return
            
        print("✅ Server running")
        
        # Test requests in order
        test_requests = [
            {
                "name": "1. Initialize",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"}
                    }
                }
            },
            {
                "name": "2. List Tools (no params)",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list"
                }
            },
            {
                "name": "3. List Tools (empty params)",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/list",
                    "params": {}
                }
            },
            {
                "name": "4. List Tools (cursor param)",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/list",
                    "params": {"cursor": None}
                }
            },
            {
                "name": "5. Call search_anime directly",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "search_anime",
                        "arguments": {"query": "naruto"}
                    }
                }
            },
            {
                "name": "6. Call get_anime_stats",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {
                        "name": "get_anime_stats",
                        "arguments": {}
                    }
                }
            }
        ]
        
        for req_info in test_requests:
            print(f"\n📡 {req_info['name']}")
            request_json = json.dumps(req_info["request"]) + "\n"
            
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Wait for response
            response_received = False
            for _ in range(10):  # 10 second timeout
                if process.poll() is not None:
                    print("❌ Server died")
                    break
                    
                line = process.stdout.readline()
                if line.strip():
                    print(f"📨 Response: {line.strip()}")
                    try:
                        response = json.loads(line.strip())
                        
                        # Analyze response
                        if "result" in response:
                            print("✅ Success")
                            if "tools" in response["result"]:
                                tools = response["result"]["tools"]
                                print(f"📋 Found {len(tools)} tools:")
                                for tool in tools[:3]:  # Show first 3
                                    print(f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:50]}...")
                        elif "error" in response:
                            error = response["error"]
                            print(f"❌ Error: {error.get('message', str(error))}")
                        else:
                            print("📋 Other response")
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON error: {e}")
                    
                    response_received = True
                    break
                    
                await asyncio.sleep(1)
                
            if not response_received:
                print("❌ No response received")
                
            # Small delay between requests
            await asyncio.sleep(0.5)
            
    finally:
        print("\n🧹 Cleaning up...")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    asyncio.run(test_mcp_protocol())