#!/usr/bin/env python3
"""Enhanced scraping test with better selectors"""

import json
import cloudscraper
from bs4 import BeautifulSoup
import re

class EnhancedScrapingTester:
    def __init__(self):
        # Create scraper with browser settings
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'linux',
                'desktop': True
            }
        )
        self.results = {}
    
    def test_anime_planet_detailed(self):
        """Test Anime-Planet with better selectors"""
        print("\n🔍 Testing Anime-Planet (detailed)...")
        url = "https://www.anime-planet.com/anime/death-note"
        
        try:
            response = self.scraper.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Print page structure to understand it better
                print("  Analyzing page structure...")
                
                # Try multiple selectors
                title = soup.find('h1', {'itemprop': 'name'}) or soup.find('h1')
                
                # Synopsis might be in different places
                synopsis = (
                    soup.find('p', {'itemprop': 'description'}) or
                    soup.find('div', class_='synopsis') or
                    soup.find('section', class_='entryDetails')
                )
                
                # Look for meta tags as fallback
                if not synopsis:
                    meta_desc = soup.find('meta', {'name': 'description'})
                    if meta_desc:
                        synopsis = {'text': meta_desc.get('content', '')}
                
                # Extract more details
                details = {
                    'title': title.text.strip() if title else None,
                    'synopsis': synopsis.text.strip() if hasattr(synopsis, 'text') else str(synopsis),
                    'status_code': response.status_code,
                    'page_title': soup.find('title').text if soup.find('title') else None,
                    'has_cloudflare': 'cf-browser-verification' in response.text
                }
                
                print(f"  ✅ Extracted: {details['title']}")
                print(f"  Synopsis found: {'Yes' if details['synopsis'] else 'No'}")
                return details
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {'error': str(e)}
    
    def test_livechart_detailed(self):
        """Test LiveChart with JavaScript content"""
        print("\n🔍 Testing LiveChart (detailed)...")
        url = "https://www.livechart.me/anime/3437"
        
        try:
            # Add more headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = self.scraper.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Check for JSON-LD data
                json_ld = soup.find('script', {'type': 'application/ld+json'})
                if json_ld:
                    try:
                        data = json.loads(json_ld.string)
                        print("  Found JSON-LD data!")
                        return {
                            'success': True,
                            'json_ld_data': data,
                            'title': data.get('name'),
                            'description': data.get('description')
                        }
                    except:
                        pass
                
                # Try regular extraction
                title = soup.find('h1') or soup.find('div', class_='anime-title')
                
                # Check if we hit a loading page
                if 'loading' in response.text.lower() or 'javascript' in response.text.lower():
                    return {
                        'status': 'requires_javascript',
                        'note': 'This site loads content dynamically with JavaScript'
                    }
                
                return {
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'content_length': len(response.text),
                    'has_js_requirement': 'noscript' in response.text
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {'error': str(e)}
    
    def test_anisearch_detailed(self):
        """Test AniSearch with German content"""
        print("\n🔍 Testing AniSearch (German site)...")
        url = "https://www.anisearch.com/anime/3633,death-note"
        
        try:
            # German site might need different headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9,de;q=0.8'
            }
            
            response = self.scraper.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # German site structure
                title = soup.find('h1') or soup.find('span', {'itemprop': 'name'})
                
                # Look for description in various places
                desc_selectors = [
                    ('div', {'class': 'details-text'}),
                    ('div', {'class': 'description'}),
                    ('div', {'itemprop': 'description'}),
                    ('p', {'class': 'abstract'})
                ]
                
                synopsis = None
                for tag, attrs in desc_selectors:
                    synopsis = soup.find(tag, attrs)
                    if synopsis:
                        break
                
                # Check meta tags
                meta_desc = soup.find('meta', {'property': 'og:description'})
                
                return {
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'synopsis': synopsis.text.strip() if synopsis else None,
                    'meta_description': meta_desc.get('content') if meta_desc else None,
                    'page_lang': soup.get('lang', 'unknown'),
                    'is_german': 'de' in str(soup.get('lang', ''))
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {'error': str(e)}
    
    def analyze_cloudflare_protection(self):
        """Check which sites have Cloudflare protection"""
        print("\n🔍 Analyzing Cloudflare protection...")
        
        test_sites = {
            'anime_planet': 'https://www.anime-planet.com',
            'livechart': 'https://www.livechart.me',
            'anisearch': 'https://www.anisearch.com',
            'simkl': 'https://simkl.com',
            'ann': 'https://www.animenewsnetwork.com'
        }
        
        cf_results = {}
        for site, url in test_sites.items():
            try:
                response = self.scraper.get(url, timeout=10)
                cf_indicators = [
                    'cf-ray' in response.headers,
                    'cloudflare' in response.headers.get('server', '').lower(),
                    'cf-browser-verification' in response.text,
                    '__cf_bm' in response.cookies
                ]
                
                cf_results[site] = {
                    'has_cloudflare': any(cf_indicators),
                    'status_code': response.status_code,
                    'indicators': [i for i, check in enumerate(cf_indicators) if check]
                }
                
            except Exception as e:
                cf_results[site] = {'error': str(e)}
        
        return cf_results
    
    def run_enhanced_tests(self):
        """Run all enhanced tests"""
        print("🚀 Running enhanced scraping tests...")
        print("=" * 60)
        
        # Test individual sites
        self.results['anime_planet'] = self.test_anime_planet_detailed()
        self.results['livechart'] = self.test_livechart_detailed()
        self.results['anisearch'] = self.test_anisearch_detailed()
        
        # Analyze Cloudflare
        self.results['cloudflare_analysis'] = self.analyze_cloudflare_protection()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 ENHANCED RESULTS:")
        print("=" * 60)
        
        # Print readable summary
        for site, result in self.results.items():
            if site == 'cloudflare_analysis':
                print(f"\n🛡️ Cloudflare Protection:")
                for s, info in result.items():
                    if 'has_cloudflare' in info:
                        print(f"  {s}: {'Protected' if info['has_cloudflare'] else 'Not Protected'}")
            else:
                print(f"\n{site}:")
                if 'title' in result:
                    print(f"  Title: {result.get('title', 'Not found')}")
                if 'synopsis' in result:
                    print(f"  Synopsis: {'Found' if result.get('synopsis') else 'Not found'}")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                if 'status' in result:
                    print(f"  Status: {result['status']}")
        
        # Save detailed results
        with open('scraping_test_detailed.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n💾 Detailed results saved to scraping_test_detailed.json")

if __name__ == "__main__":
    tester = EnhancedScrapingTester()
    tester.run_enhanced_tests()