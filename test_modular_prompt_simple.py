#!/usr/bin/env python3
"""
Simple test for NEW modular prompt system
"""

import asyncio
import json
import logging
import time
from src.services.iterative_ai_enrichment import IterativeAIEnrichmentAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_new_system():
    """Test the NEW modular prompt system"""
    
    # Load test data
    with open('base_anime_sample.json', 'r') as f:
        anime_data = json.load(f)
    
    # Find Dandadan
    dandadan = None
    for anime in anime_data:
        if anime.get('title') == 'Dandadan':
            dandadan = anime
            break
    
    print(f"Testing NEW system with: {dandadan['title']}")
    
    # Initialize agent
    agent = IterativeAIEnrichmentAgent(ai_provider="openai")
    
    # Call NEW method directly
    start_time = time.time()
    result = await agent._ai_enrich_data_v2(dandadan)
    end_time = time.time()
    
    print(f"✅ NEW system completed in {end_time - start_time:.2f}s")
    
    # Save result
    timestamp = int(time.time())
    output_file = f"new_system_result_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"📄 Result saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(test_new_system())