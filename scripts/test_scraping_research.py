#!/usr/bin/env python3
"""Research script to test scraping approaches for anime data sources.

This script tests different approaches for scraping anime data from various sources
to determine what libraries and techniques we'll need.
"""

import asyncio
import time
from typing import Dict, List, Optional
import aiohttp
import requests
from urllib.parse import urljoin, urlparse

# Test sites to understand structure and accessibility
TEST_SITES = {
    "anime-planet": {
        "base_url": "https://www.anime-planet.com",
        "sample_anime": "/anime/cowboy-bebop",
        "search_url": "/anime/all?name=",
        "description": "26k+ anime entries, no official API"
    },
    "anisearch": {
        "base_url": "https://www.anisearch.com",
        "sample_anime": "/anime/1,cowboy-bebop",
        "search_url": "/anime/index/page?char=all&text=",
        "description": "20k+ anime entries, German/English"
    },
    "livechart": {
        "base_url": "https://www.livechart.me",
        "sample_anime": "/anime/10793",
        "search_url": "/search?q=",
        "description": "Seasonal schedules, airing times"
    },
    "animecountdown": {
        "base_url": "https://animecountdown.com",
        "sample_anime": "/86",
        "search_url": "/search?q=",
        "description": "Air dates and countdowns"
    }
}

class ScrapingResearch:
    """Research scraping approaches for anime data sources."""
    
    def __init__(self):
        self.session = None
        self.results = {}
        
    async def test_basic_http_access(self, site_name: str, site_info: dict) -> Dict:
        """Test basic HTTP access to a site."""
        print(f"\n🔍 Testing {site_name}...")
        result = {
            "site": site_name,
            "base_url": site_info["base_url"],
            "accessible": False,
            "status_code": None,
            "headers": {},
            "requires_js": False,
            "cloudflare": False,
            "response_time": None
        }
        
        try:
            start_time = time.time()
            
            # Test with common user agent
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with self.session.get(
                site_info["base_url"] + site_info["sample_anime"],
                headers=headers,
                allow_redirects=True,
                timeout=10
            ) as response:
                result["status_code"] = response.status
                result["response_time"] = time.time() - start_time
                result["headers"] = dict(response.headers)
                
                # Check for common indicators
                content = await response.text()
                result["accessible"] = response.status == 200
                
                # Check for JavaScript requirements
                if "noscript" in content.lower() or "enable javascript" in content.lower():
                    result["requires_js"] = True
                
                # Check for Cloudflare
                if "cloudflare" in content.lower() or "cf-ray" in str(response.headers).lower():
                    result["cloudflare"] = True
                
                # Sample content check
                result["content_length"] = len(content)
                result["has_anime_data"] = any(keyword in content.lower() for keyword in ["episode", "genre", "studio", "synopsis"])
                
                print(f"  ✅ Status: {response.status}")
                print(f"  ⏱️  Response time: {result['response_time']:.2f}s")
                print(f"  📄 Content length: {result['content_length']} bytes")
                print(f"  🔧 Requires JS: {result['requires_js']}")
                print(f"  ☁️  Cloudflare: {result['cloudflare']}")
                
        except asyncio.TimeoutError:
            result["error"] = "Timeout"
            print(f"  ❌ Timeout after 10 seconds")
        except Exception as e:
            result["error"] = str(e)
            print(f"  ❌ Error: {e}")
            
        return result
    
    async def test_search_functionality(self, site_name: str, site_info: dict) -> Dict:
        """Test search functionality on a site."""
        print(f"\n🔎 Testing search for {site_name}...")
        
        search_query = "cowboy bebop"
        search_url = site_info["base_url"] + site_info["search_url"] + search_query.replace(" ", "%20")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with self.session.get(search_url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Look for search result indicators
                    has_results = any(indicator in content.lower() for indicator in [
                        "search results", "results", search_query.lower(),
                        "cowboy", "bebop"
                    ])
                    
                    print(f"  ✅ Search accessible: {search_url}")
                    print(f"  📊 Has results: {has_results}")
                    
                    return {
                        "search_works": True,
                        "url": search_url,
                        "has_results": has_results
                    }
                else:
                    print(f"  ❌ Search returned status: {response.status}")
                    return {"search_works": False, "status": response.status}
                    
        except Exception as e:
            print(f"  ❌ Search error: {e}")
            return {"search_works": False, "error": str(e)}
    
    async def analyze_content_structure(self, site_name: str, site_info: dict) -> Dict:
        """Analyze HTML structure to understand data extraction needs."""
        print(f"\n📐 Analyzing content structure for {site_name}...")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with self.session.get(
                site_info["base_url"] + site_info["sample_anime"],
                headers=headers,
                timeout=10
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Analyze common patterns
                    analysis = {
                        "has_json_ld": '"@type"' in content and '"@context"' in content,
                        "has_microdata": 'itemscope' in content or 'itemprop' in content,
                        "has_og_tags": 'property="og:' in content,
                        "uses_react": 'react' in content.lower() or '__NEXT_DATA__' in content,
                        "uses_vue": 'vue' in content.lower() or 'v-if' in content,
                        "api_endpoints": self._find_api_endpoints(content),
                        "data_attributes": self._find_data_attributes(content)
                    }
                    
                    print(f"  📊 JSON-LD: {analysis['has_json_ld']}")
                    print(f"  🏷️  Microdata: {analysis['has_microdata']}")
                    print(f"  🔖 OpenGraph: {analysis['has_og_tags']}")
                    print(f"  ⚛️  React/Next: {analysis['uses_react']}")
                    print(f"  🟢 Vue: {analysis['uses_vue']}")
                    if analysis['api_endpoints']:
                        print(f"  🔌 Found API endpoints: {len(analysis['api_endpoints'])}")
                    
                    return analysis
                    
        except Exception as e:
            print(f"  ❌ Analysis error: {e}")
            return {"error": str(e)}
    
    def _find_api_endpoints(self, content: str) -> List[str]:
        """Find potential API endpoints in page content."""
        import re
        
        # Common API patterns
        patterns = [
            r'["\']/(api/[^"\']+)["\']',
            r'fetch\(["\']([^"\']+)["\']',
            r'axios\.[get|post|put|delete]\(["\']([^"\']+)["\']',
            r'["\']https?://[^/]+/(api/[^"\']+)["\']'
        ]
        
        endpoints = set()
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            endpoints.update(matches)
            
        return list(endpoints)[:10]  # Limit to 10 for display
    
    def _find_data_attributes(self, content: str) -> Dict:
        """Find data attributes that might contain structured data."""
        import re
        
        data_attrs = {}
        
        # Look for data attributes with JSON
        pattern = r'data-([a-z\-]+)=["\']({[^"\']+})["\']'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        for attr_name, json_str in matches[:5]:  # Limit to 5
            try:
                import json
                data = json.loads(json_str.replace('&quot;', '"'))
                data_attrs[f"data-{attr_name}"] = "JSON object found"
            except:
                pass
                
        return data_attrs
    
    async def run_research(self):
        """Run all research tests."""
        print("🚀 Starting anime scraping research...")
        print("=" * 60)
        
        self.session = aiohttp.ClientSession()
        
        try:
            # Test each site
            for site_name, site_info in TEST_SITES.items():
                site_results = {}
                
                # Basic HTTP access
                access_result = await self.test_basic_http_access(site_name, site_info)
                site_results["access"] = access_result
                
                # Only test further if site is accessible
                if access_result.get("accessible"):
                    # Test search
                    search_result = await self.test_search_functionality(site_name, site_info)
                    site_results["search"] = search_result
                    
                    # Analyze structure
                    structure_result = await self.analyze_content_structure(site_name, site_info)
                    site_results["structure"] = structure_result
                
                self.results[site_name] = site_results
                print("-" * 60)
                
        finally:
            await self.session.close()
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print research summary."""
        print("\n📊 RESEARCH SUMMARY")
        print("=" * 60)
        
        for site_name, results in self.results.items():
            print(f"\n{site_name.upper()}:")
            
            access = results.get("access", {})
            if access.get("accessible"):
                print(f"  ✅ Accessible via simple HTTP")
                print(f"  ⏱️  Response time: {access.get('response_time', 'N/A'):.2f}s")
                
                if access.get("requires_js"):
                    print(f"  ⚠️  Requires JavaScript rendering")
                if access.get("cloudflare"):
                    print(f"  ⚠️  Protected by Cloudflare")
                    
                if "search" in results:
                    search = results["search"]
                    if search.get("search_works"):
                        print(f"  ✅ Search functionality works")
                    else:
                        print(f"  ❌ Search not working")
                        
                if "structure" in results:
                    structure = results["structure"]
                    if structure.get("has_json_ld"):
                        print(f"  ✅ Has structured data (JSON-LD)")
                    if structure.get("api_endpoints"):
                        print(f"  🔌 Found {len(structure['api_endpoints'])} potential API endpoints")
            else:
                print(f"  ❌ Not accessible: {access.get('error', 'Unknown error')}")
        
        print("\n🔧 RECOMMENDED TOOLS:")
        
        # Determine what tools we need
        needs_js = any(r.get("access", {}).get("requires_js") for r in self.results.values())
        has_cloudflare = any(r.get("access", {}).get("cloudflare") for r in self.results.values())
        
        if needs_js or has_cloudflare:
            print("  - Playwright or Selenium for JavaScript rendering")
            print("  - cloudscraper for Cloudflare bypass")
        else:
            print("  - aiohttp + BeautifulSoup4 for simple HTML parsing")
            print("  - httpx as alternative async HTTP client")
        
        print("  - lxml for fast XML/HTML parsing")
        print("  - selectolax for very fast HTML parsing (if performance critical)")
        print("  - Scrapy framework (if building comprehensive scrapers)")


if __name__ == "__main__":
    research = ScrapingResearch()
    asyncio.run(research.run_research())