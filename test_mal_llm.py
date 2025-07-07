#!/usr/bin/env python3
"""Test MAL API client using LLM-powered testing."""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_mal_client():
    """Test MAL API client functionality."""
    print("🚀 Testing MAL API Client with LLM-powered scenarios")
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        mal_client_id = os.getenv('MAL_CLIENT_ID')
        
        if not mal_client_id:
            print("❌ MAL_CLIENT_ID not found in environment variables")
            print("   Set MAL_CLIENT_ID in .env file to test MAL API")
            return False
        
        # Import the client
        from src.integrations.clients.mal_client import MALClient
        
        # Initialize client with client ID only (no OAuth2 for basic testing)
        client = MALClient(client_id=mal_client_id)
        print(f"✅ MAL client initialized successfully with client ID: {mal_client_id[:8]}...")
        
        # Test 1: Search for popular anime
        print("\n📋 Test 1: Search for 'One Piece'")
        try:
            search_results = await client.search_anime(
                q="One Piece",
                limit=5,
                fields="id,title,mean,num_episodes,status,start_date"
            )
            
            if search_results:
                print(f"✅ Found {len(search_results)} results")
                for i, anime in enumerate(search_results[:2], 1):
                    # MAL API returns nested structure with "node" containing data
                    node = anime.get('node', anime)
                    title = node.get('title', 'Unknown')
                    anime_id = node.get('id', 'Unknown')
                    mean = node.get('mean', 'N/A')
                    episodes = node.get('num_episodes', 'Unknown')
                    print(f"   {i}. {title} (MAL ID: {anime_id}, Score: {mean}, Episodes: {episodes})")
            else:
                print("❌ No search results found")
        except Exception as e:
            print(f"❌ Search test failed: {e}")
            
        # Test 2: Get specific anime by ID (One Piece - MAL ID: 21)
        print("\n📋 Test 2: Get One Piece details by ID (21)")
        try:
            anime_details = await client.get_anime_by_id(
                21,
                fields="id,title,mean,num_episodes,status,start_date,synopsis,genres"
            )
            
            if anime_details:
                print("✅ Retrieved anime details successfully")
                title = anime_details.get('title', 'Unknown')
                episodes = anime_details.get('num_episodes', 'Unknown')
                status = anime_details.get('status', 'Unknown')
                mean = anime_details.get('mean', 'Unknown')
                start_date = anime_details.get('start_date', 'Unknown')
                print(f"   Title: {title}")
                print(f"   Episodes: {episodes}")
                print(f"   Status: {status}")
                print(f"   Score: {mean}")
                print(f"   Start Date: {start_date}")
                
                # Show genres if available
                genres = anime_details.get('genres', [])
                if genres:
                    genre_names = [g.get('name', '') for g in genres[:3]]
                    print(f"   Genres: {', '.join(genre_names)}")
            else:
                print("❌ Failed to retrieve anime details")
        except Exception as e:
            print(f"❌ Get anime by ID test failed: {e}")
            
        # Test 3: Get anime ranking
        print("\n📋 Test 3: Get top anime rankings (TV series)")
        try:
            top_anime = await client.get_anime_ranking(
                ranking_type="tv",
                limit=3,
                fields="title,mean,num_episodes"
            )
            
            if top_anime:
                print(f"✅ Retrieved {len(top_anime)} top anime")
                for i, anime_entry in enumerate(top_anime, 1):
                    # MAL ranking API returns {"node": anime_data, "ranking": {...}}
                    node = anime_entry.get('node', {})
                    ranking = anime_entry.get('ranking', {})
                    title = node.get('title', 'Unknown')
                    mean = node.get('mean', 'N/A')
                    rank = ranking.get('rank', 'Unknown')
                    print(f"   {i}. #{rank} {title} (Score: {mean})")
            else:
                print("❌ Failed to retrieve top anime")
        except Exception as e:
            print(f"❌ Ranking test failed: {e}")
            
        # Test 4: Get seasonal anime  
        print("\n📋 Test 4: Get seasonal anime (2024 Fall)")
        try:
            seasonal_anime = await client.get_seasonal_anime(
                2024, 
                "fall",
                limit=3,
                fields="title,mean,num_episodes,status"
            )
            
            if seasonal_anime:
                print(f"✅ Found {len(seasonal_anime)} seasonal anime")
                for i, anime in enumerate(seasonal_anime[:3], 1):
                    # Seasonal API returns nested structure
                    node = anime.get('node', anime)
                    title = node.get('title', 'Unknown')
                    episodes = node.get('num_episodes', 'Unknown')
                    status = node.get('status', 'Unknown')
                    print(f"   {i}. {title} ({episodes} episodes, {status})")
            else:
                print("❌ No seasonal anime found")
        except Exception as e:
            print(f"❌ Seasonal anime test failed: {e}")
            
        # Test 5: Advanced search with fields
        print("\n📋 Test 5: Advanced search - 'Naruto' with detailed fields")
        try:
            detailed_results = await client.search_anime(
                q="Naruto",
                limit=2,
                fields="id,title,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,studios"
            )
            
            if detailed_results:
                print(f"✅ Found {len(detailed_results)} detailed results")
                for i, anime in enumerate(detailed_results, 1):
                    node = anime.get('node', anime)
                    title = node.get('title', 'Unknown')
                    anime_id = node.get('id', 'Unknown')
                    media_type = node.get('media_type', 'Unknown')
                    status = node.get('status', 'Unknown')
                    print(f"   {i}. {title} (ID: {anime_id}, Type: {media_type}, Status: {status})")
            else:
                print("❌ No detailed search results found")
        except Exception as e:
            print(f"❌ Detailed search test failed: {e}")
            
        # Test 6: Error handling test
        print("\n📋 Test 6: Error handling - Invalid anime ID")
        try:
            invalid_anime = await client.get_anime_by_id(999999999)
            if invalid_anime is None:
                print("✅ Error handling working - None returned for invalid ID")
            else:
                print("⚠️  Unexpected result for invalid ID")
        except Exception as e:
            print(f"✅ Error handling working - Exception caught: {type(e).__name__}")
            
        # Test 7: Empty query test
        print("\n📋 Test 7: Error handling - Empty search query")
        try:
            empty_results = await client.search_anime(q="")
            print("⚠️  Empty query should have raised ValueError")
        except ValueError as e:
            print("✅ Error handling working - ValueError for empty query")
        except Exception as e:
            print(f"⚠️  Unexpected exception for empty query: {type(e).__name__}")
            
        print("\n🎉 MAL API client testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ MAL client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mal_client())
    sys.exit(0 if success else 1)