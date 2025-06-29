#!/usr/bin/env python3
"""Monitor indexing progress in real-time."""

import asyncio
import time
from src.vector.qdrant_client import QdrantClient
from src.config import get_settings

async def monitor_progress():
    """Monitor indexing progress."""
    settings = get_settings()
    client = QdrantClient(settings=settings)
    
    target_total = 38894
    start_time = time.time()
    
    print("📊 Monitoring indexing progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            try:
                stats = await client.get_stats()
                current_count = stats.get('total_documents', 0)
                
                elapsed = time.time() - start_time
                percentage = (current_count / target_total) * 100
                
                if current_count > 0:
                    rate = current_count / elapsed * 60  # per minute
                    eta_minutes = (target_total - current_count) / (current_count / elapsed) / 60
                    
                    print(f"\r🔄 Progress: {current_count:,}/{target_total:,} ({percentage:.1f}%) | "
                          f"Rate: {rate:.0f}/min | ETA: {eta_minutes:.1f}min", end="", flush=True)
                else:
                    print(f"\r⏳ Waiting for indexing to start... ({elapsed:.0f}s)", end="", flush=True)
                
                if current_count >= target_total:
                    print(f"\n🎉 Indexing complete! {current_count:,} documents indexed in {elapsed/60:.1f} minutes")
                    break
                    
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"\r❌ Error checking progress: {e}", end="", flush=True)
                await asyncio.sleep(10)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")

if __name__ == "__main__":
    asyncio.run(monitor_progress())