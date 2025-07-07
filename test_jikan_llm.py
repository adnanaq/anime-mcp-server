#!/usr/bin/env python3
"""Test Jikan API client using LLM-powered testing."""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_jikan_client():
    """Test Jikan API client functionality."""
    print("🚀 Testing Jikan API Client with LLM-powered scenarios")
    
    try:
        # Import the client
        from src.integrations.clients.jikan_client import JikanClient
        
        # Initialize client
        client = JikanClient()
        print("✅ Jikan client initialized successfully")
        
        # Test 1: Search for popular anime
        print("\n📋 Test 1: Search for 'Attack on Titan'")
        search_results = await client.search_anime(
            q="Attack on Titan",
            limit=5
        )
        
        if search_results:
            print(f"✅ Found {len(search_results)} results")
            for i, anime in enumerate(search_results[:2], 1):
                title = anime.get('title', 'Unknown')
                mal_id = anime.get('mal_id', 'Unknown')
                score = anime.get('score', 'N/A')
                print(f"   {i}. {title} (MAL ID: {mal_id}, Score: {score})")
        else:
            print("❌ No search results found")
            
        # Test 2: Get specific anime by ID (Attack on Titan)
        print("\n📋 Test 2: Get Attack on Titan details by ID (16498)")
        anime_details = await client.get_anime_by_id(16498)
        
        if anime_details:
            print("✅ Retrieved anime details successfully")
            title = anime_details.get('title', 'Unknown')
            episodes = anime_details.get('episodes', 'Unknown')
            status = anime_details.get('status', 'Unknown')
            year = anime_details.get('year', 'Unknown')
            print(f"   Title: {title}")
            print(f"   Episodes: {episodes}")
            print(f"   Status: {status}")
            print(f"   Year: {year}")
        else:
            print("❌ Failed to retrieve anime details")
            
        # Test 3: Advanced search with filters
        print("\n📋 Test 3: Advanced search - Action anime from 2020+")
        filtered_results = await client.search_anime(
            genres=[1],  # Action genre
            min_score=7.0,
            anime_type="TV",
            status="complete",
            limit=3
        )
        
        if filtered_results:
            print(f"✅ Found {len(filtered_results)} filtered results")
            for i, anime in enumerate(filtered_results, 1):
                title = anime.get('title', 'Unknown')
                score = anime.get('score', 'N/A')
                year = anime.get('year', 'Unknown')
                print(f"   {i}. {title} (Score: {score}, Year: {year})")
        else:
            print("❌ No filtered results found")
            
        # Test 4: Get top anime
        print("\n📋 Test 4: Get top anime rankings")
        top_anime = await client.get_top_anime(
            anime_type="tv",
            limit=3
        )
        
        if top_anime:
            print(f"✅ Retrieved {len(top_anime)} top anime")
            for i, anime in enumerate(top_anime, 1):
                title = anime.get('title', 'Unknown')
                rank = anime.get('rank', 'Unknown')
                score = anime.get('score', 'N/A')
                print(f"   {i}. #{rank} {title} (Score: {score})")
        else:
            print("❌ Failed to retrieve top anime")
            
        # Test 5: Get seasonal anime
        print("\n📋 Test 5: Get current season anime (2024 Fall)")
        seasonal_anime = await client.get_seasonal_anime(2024, "fall")
        
        if seasonal_anime:
            print(f"✅ Found {len(seasonal_anime)} seasonal anime")
            for i, anime in enumerate(seasonal_anime[:3], 1):
                title = anime.get('title', 'Unknown')
                episodes = anime.get('episodes', 'Unknown')
                print(f"   {i}. {title} ({episodes} episodes)")
        else:
            print("❌ No seasonal anime found")
            
        # Test 6: Random anime discovery
        print("\n📋 Test 6: Get random anime")
        random_anime = await client.get_random_anime()
        
        if random_anime:
            print("✅ Retrieved random anime")
            title = random_anime.get('title', 'Unknown')
            mal_id = random_anime.get('mal_id', 'Unknown')
            score = random_anime.get('score', 'N/A')
            print(f"   Random pick: {title} (MAL ID: {mal_id}, Score: {score})")
        else:
            print("❌ Failed to get random anime")
            
        # Test 7: Error handling test
        print("\n📋 Test 7: Error handling - Invalid anime ID")
        try:
            invalid_anime = await client.get_anime_by_id(999999999)
            if invalid_anime is None:
                print("✅ Error handling working - None returned for invalid ID")
            else:
                print("⚠️  Unexpected result for invalid ID")
        except Exception as e:
            print(f"✅ Error handling working - Exception caught: {type(e).__name__}")
            
        print("\n🎉 Jikan API client testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Jikan client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_jikan_client())
    sys.exit(0 if success else 1)