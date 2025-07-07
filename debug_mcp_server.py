#!/usr/bin/env python3
"""
Debug MCP Server - Check what's actually happening
"""

import asyncio
import json
import subprocess
import sys
import time
import os

async def debug_mcp_server():
    """Debug the MCP server to see what's happening."""
    print("🔍 Debugging MCP Server Communication")
    
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
        
        # Check if running
        if process.poll() is not None:
            stderr = process.stderr.read()
            stdout = process.stdout.read()
            print(f"❌ Server died: {stderr}")
            print(f"Stdout: {stdout}")
            return
            
        print("✅ Server running")
        
        # Try different MCP requests
        requests = [
            {
                "name": "Initialize",
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
                "name": "List Tools",
                "request": {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list"
                }
            }
        ]
        
        for req_info in requests:
            print(f"\n📡 Sending: {req_info['name']}")
            request_json = json.dumps(req_info["request"]) + "\n"
            print(f"Request: {request_json.strip()}")
            
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Wait for response
            for _ in range(10):  # 10 second timeout
                if process.poll() is not None:
                    print("❌ Server died")
                    break
                    
                line = process.stdout.readline()
                if line.strip():
                    print(f"📨 Response: {line.strip()}")
                    try:
                        response = json.loads(line.strip())
                        print(f"📋 Parsed: {json.dumps(response, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"⚠️  Invalid JSON response")
                    break
                    
                await asyncio.sleep(1)
            else:
                print("❌ No response received")
                
        # Check for any stderr output
        print("\n📝 Checking stderr...")
        stderr_line = process.stderr.readline()
        if stderr_line:
            print(f"Stderr: {stderr_line}")
            
    finally:
        print("🧹 Cleaning up...")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    asyncio.run(debug_mcp_server())