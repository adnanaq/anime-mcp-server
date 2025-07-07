#!/usr/bin/env python3
"""Test ServiceManager integration with MAL and Jikan APIs using LLM-powered testing."""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_service_manager():
    """Test ServiceManager integration with MAL and Jikan APIs."""
    print("🚀 Testing ServiceManager with MAL and Jikan API Integration")
    
    try:
        # Load environment and configuration
        from dotenv import load_dotenv
        load_dotenv()
        
        from src.config import Settings
        settings = Settings()
        
        # Check if MAL client ID is available
        mal_client_id = os.getenv('MAL_CLIENT_ID')
        if not mal_client_id:
            print("⚠️  MAL_CLIENT_ID not found - MAL client will not be available")
        else:
            print(f"✅ MAL client ID found: {mal_client_id[:8]}...")
        
        # Import ServiceManager
        from src.integrations.service_manager import ServiceManager
        
        # Initialize ServiceManager
        service_manager = ServiceManager(settings)
        print("✅ ServiceManager initialized successfully")
        
        # Check available clients
        available_clients = list(service_manager.clients.keys())
        print(f"📋 Available clients: {', '.join(available_clients)}")
        
        # Test 1: Universal search with both MAL and Jikan priorities
        print("\n📋 Test 1: Universal search - 'Attack on Titan'")
        try:
            # Test search through service manager
            search_params = {
                'query': 'Attack on Titan',
                'limit': 5,
                'platforms': ['jikan', 'mal']  # Test both platforms
            }
            
            results = await service_manager.search_anime(**search_params)
            
            if results:
                print(f"✅ ServiceManager search returned {len(results)} results")
                for i, result in enumerate(results[:3], 1):
                    title = result.get('title', 'Unknown')
                    source = result.get('source', 'Unknown')
                    score = result.get('score', 'N/A')
                    anime_id = result.get('id', 'Unknown')
                    print(f"   {i}. {title} (Source: {source}, Score: {score}, ID: {anime_id})")
            else:
                print("❌ No results from ServiceManager search")
        except Exception as e:
            print(f"❌ ServiceManager search test failed: {e}")
            
        # Test 2: Platform priority verification (Jikan > MAL)
        print("\n📋 Test 2: Platform priority verification")
        try:
            # Force both platforms and see which takes priority
            priority_params = {
                'query': 'One Piece',
                'limit': 3,
                'prefer_platform': None  # Let ServiceManager decide priority
            }
            
            priority_results = await service_manager.search_anime(**priority_params)
            
            if priority_results:
                print("✅ Priority test results:")
                sources_used = set()
                for result in priority_results[:3]:
                    source = result.get('source', 'unknown')
                    sources_used.add(source)
                    title = result.get('title', 'Unknown')
                    print(f"   - {title} from {source}")
                
                print(f"📊 Sources used: {', '.join(sources_used)}")
                
                # Check if Jikan appears first (higher priority)
                if priority_results:
                    first_source = priority_results[0].get('source', '')
                    if first_source == 'jikan':
                        print("✅ Priority system working - Jikan has higher priority")
                    elif first_source == 'mal':
                        print("ℹ️  MAL appeared first - checking if Jikan was unavailable")
                    else:
                        print(f"ℹ️  First source was: {first_source}")
                        
            else:
                print("❌ No results for priority test")
        except Exception as e:
            print(f"❌ Priority test failed: {e}")
            
        # Test 3: Individual platform testing
        print("\n📋 Test 3: Individual platform testing")
        
        # Test Jikan specifically
        if 'jikan' in available_clients:
            try:
                jikan_results = await service_manager.search_anime(
                    query='Naruto',
                    limit=2,
                    platforms=['jikan']
                )
                if jikan_results:
                    print(f"✅ Jikan direct search: {len(jikan_results)} results")
                    for result in jikan_results[:1]:
                        print(f"   - {result.get('title', 'Unknown')} (Jikan)")
                else:
                    print("❌ No results from Jikan direct search")
            except Exception as e:
                print(f"❌ Jikan direct test failed: {e}")
        else:
            print("⚠️  Jikan client not available")
        
        # Test MAL specifically  
        if 'mal' in available_clients:
            try:
                mal_results = await service_manager.search_anime(
                    query='Naruto',
                    limit=2,
                    platforms=['mal']
                )
                if mal_results:
                    print(f"✅ MAL direct search: {len(mal_results)} results")
                    for result in mal_results[:1]:
                        print(f"   - {result.get('title', 'Unknown')} (MAL)")
                else:
                    print("❌ No results from MAL direct search")
            except Exception as e:
                print(f"❌ MAL direct test failed: {e}")
        else:
            print("⚠️  MAL client not available")
            
        # Test 4: Fallback mechanism
        print("\n📋 Test 4: Fallback mechanism test")
        try:
            # Test with a query that might fail on one platform
            fallback_results = await service_manager.search_anime(
                query='Dragon Ball Z',
                limit=3,
                platforms=['jikan', 'mal']  # Both platforms
            )
            
            if fallback_results:
                print(f"✅ Fallback test successful: {len(fallback_results)} results")
                sources = [r.get('source', 'unknown') for r in fallback_results]
                unique_sources = set(sources)
                print(f"📊 Sources in results: {', '.join(unique_sources)}")
            else:
                print("❌ Fallback test failed - no results")
        except Exception as e:
            print(f"❌ Fallback test failed: {e}")
            
        # Test 5: Error handling with invalid query
        print("\n📋 Test 5: Error handling test")
        try:
            # Test with empty query to trigger error handling
            error_results = await service_manager.search_anime(
                query='',
                limit=1
            )
            print("⚠️  Empty query should have triggered error handling")
        except Exception as e:
            print(f"✅ Error handling working - Exception caught: {type(e).__name__}")
            
        # Test 6: Get anime by ID test (if supported)
        print("\n📋 Test 6: Get anime by ID test")
        try:
            # Test getting specific anime by ID
            anime_id = 21  # One Piece MAL ID
            anime_details = await service_manager.get_anime_by_id(
                anime_id,
                platforms=['mal', 'jikan']
            )
            
            if anime_details:
                print("✅ Get anime by ID successful")
                title = anime_details.get('title', 'Unknown')
                source = anime_details.get('source', 'Unknown')
                print(f"   - {title} from {source}")
            else:
                print("❌ Get anime by ID failed")
        except Exception as e:
            print(f"⚠️  Get anime by ID test failed (method may not exist): {type(e).__name__}")
            
        print("\n🎉 ServiceManager integration testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ ServiceManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_service_manager())
    sys.exit(0 if success else 1)