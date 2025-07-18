#!/usr/bin/env python3
"""
AniDB Helper for AI Enrichment Integration

Helper function to fetch AniDB data using XML API for AI enrichment pipeline.
"""

import argparse
import asyncio
import gzip
import io
import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
import aiohttp
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AniDBEnrichmentHelper:
    """Helper for AniDB XML API data fetching in AI enrichment pipeline."""
    
    def __init__(self, client_name: str = None, client_version: str = None):
        """Initialize AniDB enrichment helper."""
        self.base_url = "http://api.anidb.net:9001/httpapi"
        # Use environment variables if available, fallback to defaults
        self.client_name = client_name or os.getenv("ANIDB_CLIENT", "animeenrichment")
        self.client_version = client_version or os.getenv("ANIDB_CLIENTVER", "1.0")
        self.session = None
        # AniDB has very strict rate limits (1 request per 2 seconds)
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests
        
    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed AniDB rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        
    async def _make_request(self, params: Dict[str, Any]) -> Optional[str]:
        """Make XML request to AniDB API."""
        await self._wait_for_rate_limit()
        
        # Add required client parameters
        params.update({
            "client": self.client_name,
            "clientver": self.client_version,
            "protover": os.getenv("ANIDB_PROTOVER", "1"),
        })
        
        try:
            if not self.session:
                # Set headers to handle compression and identify client properly
                headers = {
                    'Accept-Encoding': 'gzip, deflate',
                    'User-Agent': f'{self.client_name}/{self.client_version}',
                    'Accept': 'application/xml, text/xml',
                    'Connection': 'close'  # Force close connection after each request
                }
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60),  # Longer timeout for AniDB
                    headers=headers,
                    connector=aiohttp.TCPConnector(limit=1)  # Single connection
                )
            
            logger.info(f"AniDB request: {self.base_url} with params: {params}")
            
            async with self.session.get(self.base_url, params=params) as response:
                logger.info(f"AniDB response status: {response.status}")
                logger.info(f"AniDB response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    # Handle compressed response
                    content = await response.read()
                    logger.info(f"Response content length: {len(content)} bytes")
                    
                    # Check if content is gzipped by magic bytes
                    if content.startswith(b'\x1f\x8b'):
                        logger.info("Content is gzip compressed, decompressing...")
                        try:
                            content = gzip.decompress(content)
                            logger.info(f"Decompressed content length: {len(content)} bytes")
                        except Exception as e:
                            logger.error(f"Failed to decompress gzipped content: {e}")
                            return None
                    
                    # Try to decode as text
                    try:
                        text_content = content.decode('utf-8')
                        logger.info(f"Decoded text preview: {text_content[:200]}")
                        return text_content
                    except UnicodeDecodeError as e:
                        logger.error(f"Failed to decode as UTF-8: {e}")
                        # Try other encodings
                        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                text_content = content.decode(encoding)
                                logger.info(f"Successfully decoded with {encoding}")
                                return text_content
                            except:
                                continue
                        logger.error("Failed to decode with any encoding")
                        return None
                        
                elif response.status == 503:
                    logger.warning("AniDB service unavailable (503)")
                    return None
                elif response.status == 555:
                    logger.warning("AniDB banned/blocked (555)")
                    return None
                else:
                    logger.warning(f"AniDB API error: HTTP {response.status}")
                    error_content = await response.text()
                    logger.info(f"Error response: {error_content[:200]}")
                    return None
        except Exception as e:
            logger.error(f"AniDB API request failed: {e}")
            return None
    
    def _parse_anime_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse anime XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return {}

        # Extract basic anime information
        anime_data = {
            "anidb_id": root.get("id"),
            "type": root.find("type").text if root.find("type") is not None else None,
            "episodecount": root.find("episodecount").text if root.find("episodecount") is not None else None,
            "startdate": root.find("startdate").text if root.find("startdate") is not None else None,
            "enddate": root.find("enddate").text if root.find("enddate") is not None else None,
            "description": root.find("description").text if root.find("description") is not None else None,
            "url": root.find("url").text if root.find("url") is not None else None,
            "picture": root.find("picture").text if root.find("picture") is not None else None,
        }

        # Extract titles
        titles_element = root.find("titles")
        titles = {}
        if titles_element is not None:
            for title in titles_element.findall("title"):
                title_type = title.get("type", "unknown")
                lang = title.get("xml:lang", "unknown")
                
                if title_type == "main":
                    titles["main"] = title.text
                elif title_type == "official":
                    if lang == "en":
                        titles["english"] = title.text
                    elif lang == "ja":
                        titles["japanese"] = title.text
                elif title_type == "synonym":
                    if "synonyms" not in titles:
                        titles["synonyms"] = []
                    titles["synonyms"].append(title.text)
        anime_data["titles"] = titles

        # Extract tags
        tags_element = root.find("tags")
        tags = []
        if tags_element is not None:
            for tag in tags_element.findall("tag"):
                name_element = tag.find("name")
                description_element = tag.find("description")
                if name_element is not None:
                    tag_data = {
                        "id": tag.get("id"),
                        "name": name_element.text,
                        "count": int(tag.get("count", 0)),
                        "weight": int(tag.get("weight", 0)),
                    }
                    if description_element is not None:
                        tag_data["description"] = description_element.text
                    tags.append(tag_data)
        anime_data["tags"] = tags

        # Extract ratings
        ratings_element = root.find("ratings")
        ratings = {}
        if ratings_element is not None:
            permanent = ratings_element.find("permanent")
            temporary = ratings_element.find("temporary")
            review = ratings_element.find("review")
            
            if permanent is not None:
                ratings["permanent"] = {
                    "value": float(permanent.text) if permanent.text else None,
                    "count": int(permanent.get("count", 0)),
                }
            if temporary is not None:
                ratings["temporary"] = {
                    "value": float(temporary.text) if temporary.text else None,
                    "count": int(temporary.get("count", 0)),
                }
            if review is not None:
                ratings["review"] = {
                    "value": float(review.text) if review.text else None,
                    "count": int(review.get("count", 0)),
                }
        anime_data["ratings"] = ratings

        # Extract categories/genres
        categories_element = root.find("categories")
        categories = []
        if categories_element is not None:
            for category in categories_element.findall("category"):
                name_element = category.find("name")
                if name_element is not None:
                    categories.append({
                        "id": category.get("id"),
                        "name": name_element.text,
                        "weight": int(category.get("weight", 0)),
                        "hentai": category.get("hentai") == "true",
                    })
        anime_data["categories"] = categories

        # Extract creator information
        creators_element = root.find("creators")
        creators = []
        if creators_element is not None:
            for creator in creators_element.findall("name"):
                creators.append({
                    "id": creator.get("id"),
                    "name": creator.text,
                    "type": creator.get("type"),
                })
        anime_data["creators"] = creators

        # Extract characters
        characters_element = root.find("characters")
        characters = []
        if characters_element is not None:
            for character in characters_element.findall("character"):
                char_data = {
                    "id": character.get("id"),
                    "type": character.get("type"),
                    "update": character.get("update"),
                }
                
                # Get character details
                name_element = character.find("name")
                if name_element is not None:
                    char_data["name"] = name_element.text
                
                gender_element = character.find("gender")
                if gender_element is not None:
                    char_data["gender"] = gender_element.text
                
                char_type_element = character.find("charactertype")
                if char_type_element is not None:
                    char_data["character_type"] = char_type_element.text
                    char_data["character_type_id"] = char_type_element.get("id")
                
                description_element = character.find("description")
                if description_element is not None:
                    char_data["description"] = description_element.text
                
                # Character rating
                rating_element = character.find("rating")
                if rating_element is not None:
                    char_data["rating"] = float(rating_element.text) if rating_element.text else None
                    char_data["rating_votes"] = int(rating_element.get("votes", 0))
                
                # Character picture
                picture_element = character.find("picture")
                if picture_element is not None:
                    char_data["picture"] = picture_element.text
                
                # Voice actor (seiyuu)
                seiyuu_element = character.find("seiyuu")
                if seiyuu_element is not None:
                    char_data["seiyuu"] = {
                        "name": seiyuu_element.text,
                        "id": seiyuu_element.get("id"),
                        "picture": seiyuu_element.get("picture")
                    }
                
                characters.append(char_data)
        
        anime_data["characters"] = characters

        return anime_data
    
    def _parse_episode_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse episode XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return {}

        episode_data = {
            "anidb_id": root.get("id"),
            "anime_id": root.get("aid"),
            "episode_number": root.find("epno").text if root.find("epno") is not None else None,
            "length": int(root.find("length").text) if root.find("length") is not None and root.find("length").text else None,
            "airdate": root.find("airdate").text if root.find("airdate") is not None else None,
            "rating": float(root.find("rating").text) if root.find("rating") is not None and root.find("rating").text else None,
            "votes": int(root.find("votes").text) if root.find("votes") is not None and root.find("votes").text else None,
            "summary": root.find("summary").text if root.find("summary") is not None else None,
        }

        # Extract episode titles
        titles = {}
        for title in root.findall("title"):
            lang = title.get("xml:lang") or title.get("{http://www.w3.org/XML/1998/namespace}lang", "unknown")
            if lang == "en":
                titles["english"] = title.text
            elif lang == "ja":
                titles["japanese"] = title.text
            elif lang == "x-jat":
                titles["romaji"] = title.text
            else:
                if "other" not in titles:
                    titles["other"] = []
                titles["other"].append({"lang": lang, "title": title.text})
        episode_data["titles"] = titles

        return episode_data
    
    async def get_anime_by_id(self, anidb_id: int) -> Optional[Dict[str, Any]]:
        """Get anime information by AniDB ID."""
        try:
            params = {"request": "anime", "aid": anidb_id}
            xml_response = await self._make_request(params)
            
            if not xml_response:
                logger.warning(f"No response for AniDB ID: {anidb_id}")
                return None
            
            if "<error" in xml_response:
                logger.warning(f"Error response for AniDB ID {anidb_id}: {xml_response[:200]}")
                return None
            
            # Log first 200 chars of response for debugging
            logger.info(f"AniDB response preview: {xml_response[:200]}")
                
            return self._parse_anime_xml(xml_response)
        except Exception as e:
            logger.error(f"Failed to fetch anime by AniDB ID {anidb_id}: {e}")
            return None
    
    async def get_episode_by_id(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get episode information by AniDB episode ID."""
        try:
            params = {"request": "episode", "eid": episode_id}
            xml_response = await self._make_request(params)
            
            if not xml_response or "<error" in xml_response:
                logger.warning(f"No data or error for AniDB episode ID: {episode_id}")
                return None
                
            return self._parse_episode_xml(xml_response)
        except Exception as e:
            logger.error(f"Failed to fetch episode by AniDB ID {episode_id}: {e}")
            return None
    
    async def search_anime_by_name(self, anime_name: str) -> Optional[List[Dict[str, Any]]]:
        """Search anime by name using AniDB API."""
        try:
            params = {"request": "anime", "aname": anime_name}
            xml_response = await self._make_request(params)
            
            if not xml_response or "<error" in xml_response:
                logger.warning(f"No search results for: {anime_name}")
                return None
                
            # AniDB search returns single anime, not a list
            anime_data = self._parse_anime_xml(xml_response)
            return [anime_data] if anime_data else None
        except Exception as e:
            logger.error(f"Failed to search anime by name '{anime_name}': {e}")
            return None
    
    async def fetch_all_data(self, anidb_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive AniDB data for an anime by AniDB ID.
        
        Args:
            anidb_id: The AniDB anime ID
            
        Returns:
            Dict containing comprehensive AniDB data or None if not found
        """
        try:
            anime_data = await self.get_anime_by_id(anidb_id)
            if not anime_data:
                logger.warning(f"No AniDB data found for ID: {anidb_id}")
                return None
                
            logger.info(f"Successfully fetched AniDB data for ID: {anidb_id}")
            return anime_data
            
        except Exception as e:
            logger.error(f"Error in fetch_all_data for AniDB ID {anidb_id}: {e}")
            return None
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()


async def main():
    """Main function for testing AniDB data fetching."""
    parser = argparse.ArgumentParser(description="Test AniDB data fetching")
    parser.add_argument("--anidb-id", type=int, help="AniDB ID to fetch")
    parser.add_argument("--search-name", type=str, help="Search anime by name")
    parser.add_argument("--output", type=str, default="test_anidb_output.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    helper = AniDBEnrichmentHelper()
    
    try:
        if args.anidb_id:
            # Fetch data by ID
            anime_data = await helper.fetch_all_data(args.anidb_id)
        elif args.search_name:
            # Search by name
            search_results = await helper.search_anime_by_name(args.search_name)
            anime_data = search_results[0] if search_results else None
        else:
            logger.error("Must provide either --anidb-id or --search-name")
            return
        
        if anime_data:
            # Save to file
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
        else:
            logger.error(f"No data found")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    finally:
        await helper.close()


if __name__ == "__main__":
    asyncio.run(main())