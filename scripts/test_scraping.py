#!/usr/bin/env python3
"""Test scraping capabilities for non-API anime sources"""

import asyncio
import json
from typing import Dict, Optional
import cloudscraper
import requests
from bs4 import BeautifulSoup
import httpx

# Test anime: Death Note (popular, should be on all sites)
TEST_ANIME = "Death Note"

class ScrapingTester:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.results = {}
    
    def test_anime_planet(self) -> Dict:
        """Test Anime-Planet scraping"""
        print("\n🔍 Testing Anime-Planet...")
        url = "https://www.anime-planet.com/anime/death-note"
        
        try:
            # Try with cloudscraper
            response = self.scraper.get(url)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data
                title = soup.find('h1', {'itemprop': 'name'})
                synopsis = soup.find('div', class_='synopsisMobile')
                if not synopsis:
                    synopsis = soup.find('div', class_='synopsis')
                
                rating = soup.find('div', class_='avgRating')
                episodes = soup.find('span', class_='type')
                
                result = {
                    'success': True,
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'synopsis': synopsis.text.strip()[:200] + '...' if synopsis else None,
                    'rating': rating.text.strip() if rating else None,
                    'episodes': episodes.text.strip() if episodes else None,
                }
                print(f"  ✅ Success! Found: {result['title']}")
                return result
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': 'Non-200 status code'
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_anisearch(self) -> Dict:
        """Test AniSearch scraping (German site)"""
        print("\n🔍 Testing AniSearch...")
        url = "https://www.anisearch.com/anime/3633,death-note"
        
        try:
            response = self.scraper.get(url)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data (German site)
                title = soup.find('h1', class_='title')
                synopsis = soup.find('div', class_='description')
                rating = soup.find('div', class_='rating-box')
                
                result = {
                    'success': True,
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'synopsis': synopsis.text.strip()[:200] + '...' if synopsis else None,
                    'rating': rating.text.strip() if rating else None,
                    'note': 'German language site'
                }
                print(f"  ✅ Success! Found: {result['title']}")
                return result
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': 'Non-200 status code'
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_livechart(self) -> Dict:
        """Test LiveChart scraping"""
        print("\n🔍 Testing LiveChart...")
        url = "https://www.livechart.me/anime/3437"  # Death Note ID
        
        try:
            # LiveChart might need different headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = self.scraper.get(url, headers=headers)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data
                title = soup.find('h1', class_='text-xl') or soup.find('h1')
                synopsis = soup.find('div', class_='text-sm')
                
                result = {
                    'success': True,
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'synopsis': synopsis.text.strip()[:200] + '...' if synopsis else None,
                    'note': 'Airing schedule focused'
                }
                print(f"  ✅ Success! Found: {result['title']}")
                return result
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': 'Non-200 status code'
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_anime_news_network(self) -> Dict:
        """Test Anime News Network encyclopedia"""
        print("\n🔍 Testing Anime News Network...")
        url = "https://www.animenewsnetwork.com/encyclopedia/anime.php?id=6592"  # Death Note
        
        try:
            response = self.scraper.get(url)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data
                title = soup.find('h1', id='page_header')
                # ANN has a unique structure
                info_div = soup.find('div', id='infotype-11')  # Plot Summary
                
                result = {
                    'success': True,
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'synopsis': info_div.text.strip()[:200] + '...' if info_div else None,
                    'note': 'Encyclopedia format'
                }
                print(f"  ✅ Success! Found: {result['title']}")
                return result
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': 'Non-200 status code'
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_simkl(self) -> Dict:
        """Test SIMKL scraping"""
        print("\n🔍 Testing SIMKL...")
        url = "https://simkl.com/anime/46/death-note"
        
        try:
            response = self.scraper.get(url)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data
                title = soup.find('h1', class_='SimklTVAbout__title')
                synopsis = soup.find('div', class_='SimklTVAbout__synopsis')
                
                result = {
                    'success': True,
                    'status_code': response.status_code,
                    'title': title.text.strip() if title else None,
                    'synopsis': synopsis.text.strip()[:200] + '...' if synopsis else None,
                    'note': 'Watch tracking focused'
                }
                print(f"  ✅ Success! Found: {result['title']}")
                return result
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': 'Non-200 status code'
                }
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all scraping tests"""
        print("🚀 Starting scraping tests for non-API anime sources...")
        print("=" * 60)
        
        self.results['anime_planet'] = self.test_anime_planet()
        self.results['anisearch'] = self.test_anisearch()
        self.results['livechart'] = self.test_livechart()
        self.results['anime_news_network'] = self.test_anime_news_network()
        self.results['simkl'] = self.test_simkl()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY:")
        print("=" * 60)
        
        successful = 0
        failed = 0
        
        for site, result in self.results.items():
            if result['success']:
                successful += 1
                print(f"✅ {site}: SUCCESS")
            else:
                failed += 1
                print(f"❌ {site}: FAILED - {result.get('error', 'Unknown error')}")
        
        print(f"\nTotal: {successful} successful, {failed} failed")
        
        # Save results
        with open('scraping_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n💾 Results saved to scraping_test_results.json")

if __name__ == "__main__":
    tester = ScrapingTester()
    tester.run_all_tests()