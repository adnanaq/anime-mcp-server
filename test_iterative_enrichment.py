#!/usr/bin/env python3
"""
Test the iterative AI enrichment with real offline database data
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.iterative_ai_enrichment import IterativeAIEnrichmentAgent, enrich_anime_for_vector_indexing


async def test_with_offline_data():
    """Test with actual offline database entry"""
    
    # Load our base anime sample
    print("\n📂 Loading base anime sample...")
    with open('base_anime_sample.json', 'r', encoding='utf-8') as f:
        anime_list = json.load(f)
    
    # Get the first anime entry as test data
    first_anime = anime_list[0]
    print(f"🎯 Testing with: {first_anime['title']}")
    print(f"📋 Original data fields: {list(first_anime.keys())}")
    
    # Test our enrichment function
    print("\n🚀 Running full enrichment...")
    
    # Create agent to see AI enrichment response
    agent = IterativeAIEnrichmentAgent()
    
    if agent.ai_client:
        print("\n--- FULL AI ENRICHMENT RESPONSE ---")
        ai_enriched = await agent._ai_enrich_data(first_anime)
        print(json.dumps(ai_enriched, indent=2, ensure_ascii=False))
        print("--- END FULL RESPONSE ---\n")
    
    enriched_result = await enrich_anime_for_vector_indexing(first_anime)
    
    # Save result to JSON file
    output_file = "iterative_enrichment_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Result saved to: {output_file}")
    print(f"📊 Original fields: {len(first_anime)}")
    print(f"📊 Enriched fields: {len(enriched_result)}")
    
    return enriched_result


if __name__ == "__main__":
    result = asyncio.run(test_with_offline_data())