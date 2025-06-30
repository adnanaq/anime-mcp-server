"""AniDB XML client implementation."""

import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from .base_client import BaseClient


class AniDBClient(BaseClient):
    """AniDB XML API client."""
    
    def __init__(self, client_name: str, client_version: str, username: str = None, password: str = None, **kwargs):
        """Initialize AniDB client."""
        super().__init__(**kwargs)
        self.base_url = "http://api.anidb.net:9001/httpapi"
        self.client_name = client_name
        self.client_version = client_version
        self.username = username
        self.password = password
        self.session_key = None
    
    async def _make_request(self, params: Dict[str, Any], timeout: int = 30) -> str:
        """Make XML request to AniDB API."""
        # Add required client parameters
        params.update({
            "client": self.client_name,
            "clientver": self.client_version,
            "protover": "1"
        })
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise Exception("Circuit breaker is open")
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"AniDB API HTTP {response.status} error")
                
                xml_content = await response.text()
                return xml_content
    
    async def authenticate(self) -> str:
        """Authenticate with AniDB and get session key."""
        if not self.username or not self.password:
            raise Exception("Authentication credentials required")
        
        params = {
            "request": "auth",
            "user": self.username,
            "pass": self.password
        }
        
        xml_response = await self._make_request(params)
        
        # Parse XML to extract session key
        root = ET.fromstring(xml_response)
        session_element = root.find("session")
        if session_element is not None:
            session_key = session_element.text
            self.session_key = session_key
            return session_key
        
        raise Exception("Authentication failed")
    
    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime information by AniDB ID."""
        # Check cache first
        cache_key = f"anidb_anime_{anime_id}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                pass
        
        try:
            params = {
                "request": "anime",
                "aid": anime_id
            }
            
            xml_response = await self._make_request(params)
            
            # Check for error response
            if "<error" in xml_response:
                return None
            
            # Parse XML response
            result = self._parse_anime_xml(xml_response)
            
            # Cache the result
            if self.cache_manager:
                try:
                    await self.cache_manager.set(cache_key, result)
                except:
                    pass
            
            return result
        
        except Exception as e:
            # Re-raise circuit breaker and XML parsing exceptions
            if "circuit breaker" in str(e).lower() or "xml parsing" in str(e).lower():
                raise
            return None
    
    def _parse_anime_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse anime XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise Exception(f"XML parsing error: {str(e)}")
        
        # Extract basic anime information
        anime_data = {
            "id": root.get("id"),
            "type": root.find("type").text if root.find("type") is not None else None,
            "episodecount": root.find("episodecount").text if root.find("episodecount") is not None else None,
            "startdate": root.find("startdate").text if root.find("startdate") is not None else None,
            "enddate": root.find("enddate").text if root.find("enddate") is not None else None,
            "description": root.find("description").text if root.find("description") is not None else None,
        }
        
        # Extract titles
        titles_element = root.find("titles")
        titles = {}
        if titles_element is not None:
            for title in titles_element.findall("title"):
                title_type = title.get("type", "unknown")
                if title_type == "main":
                    titles["main"] = title.text
                elif title_type == "official":
                    lang = title.get("xml:lang", "unknown")
                    if lang == "en":
                        titles["english"] = title.text
                    elif lang == "ja":
                        titles["japanese"] = title.text
        anime_data["titles"] = titles
        
        # Extract tags
        tags_element = root.find("tags")
        tags = []
        if tags_element is not None:
            for tag in tags_element.findall("tag"):
                name_element = tag.find("name")
                if name_element is not None:
                    tags.append({
                        "id": tag.get("id"),
                        "name": name_element.text,
                        "count": tag.get("count"),
                        "weight": tag.get("weight")
                    })
        anime_data["tags"] = tags
        
        # Extract ratings
        ratings_element = root.find("ratings")
        ratings = {}
        if ratings_element is not None:
            permanent = ratings_element.find("permanent")
            if permanent is not None:
                ratings["permanent"] = {
                    "value": permanent.text,
                    "count": permanent.get("count")
                }
        anime_data["ratings"] = ratings
        
        return anime_data
    
    async def get_episode_by_id(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get episode information by AniDB episode ID."""
        try:
            params = {
                "request": "episode",
                "eid": episode_id
            }
            
            xml_response = await self._make_request(params)
            
            # Check for error response
            if "<error" in xml_response:
                return None
            
            # Parse XML response
            result = self._parse_episode_xml(xml_response)
            return result
        
        except Exception:
            return None
    
    async def get_anime_characters(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime characters by AniDB anime ID."""
        try:
            params = {
                "request": "anime",
                "aid": anime_id,
                "amask": "c000000000000000"  # Request character data
            }
            
            xml_response = await self._make_request(params)
            
            # Check for error response
            if "<error" in xml_response:
                return []
            
            # Parse XML response
            characters = self._parse_characters_xml(xml_response)
            return characters
        
        except Exception:
            return []
    
    async def search_anime_by_name(self, anime_name: str) -> Optional[Dict[str, Any]]:
        """Search anime by name using AniDB API."""
        try:
            params = {
                "request": "anime",
                "aname": anime_name
            }
            
            xml_response = await self._make_request(params)
            
            # Check for error response
            if "<error" in xml_response:
                return None
            
            # Parse XML response
            result = self._parse_anime_xml(xml_response)
            return result
        
        except Exception:
            return None
    
    async def logout(self) -> bool:
        """Logout from AniDB and clear session."""
        if not self.session_key:
            return True
        
        try:
            params = {
                "request": "logout",
                "s": self.session_key
            }
            
            await self._make_request(params)
            self.session_key = None
            return True
        
        except Exception:
            return False
    
    def _parse_episode_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse episode XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise Exception(f"XML parsing error: {str(e)}")
        
        episode_data = {
            "id": root.get("id"),
            "aid": root.get("aid"),
            "epno": root.find("epno").text if root.find("epno") is not None else None,
            "length": root.find("length").text if root.find("length") is not None else None,
            "airdate": root.find("airdate").text if root.find("airdate") is not None else None,
            "rating": root.find("rating").text if root.find("rating") is not None else None,
            "summary": root.find("summary").text if root.find("summary") is not None else None,
        }
        
        # Extract titles (multiple title elements directly under root)
        titles = {}
        for title in root.findall("title"):
            # XML namespace handling - need to check both ways
            lang = title.get("xml:lang") or title.get("{http://www.w3.org/XML/1998/namespace}lang", "unknown")
            if lang == "en":
                titles["en"] = title.text
            elif lang == "ja":
                titles["ja"] = title.text
            else:
                titles[lang] = title.text
        episode_data["titles"] = titles
        
        return episode_data
    
    def _parse_characters_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse characters XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise Exception(f"XML parsing error: {str(e)}")
        characters = []
        
        # Look for characters in the response
        for character in root.findall(".//character"):
            character_data = {
                "id": character.get("id"),
                "type": character.get("type"),
                "name": character.find("name").text if character.find("name") is not None else None,
                "gender": character.find("gender").text if character.find("gender") is not None else None,
                "description": character.find("description").text if character.find("description") is not None else None,
                "picture": character.find("picture").text if character.find("picture") is not None else None,
            }
            
            # Extract seiyuu information
            seiyuu = character.find("seiyuu")
            if seiyuu is not None:
                character_data["seiyuu"] = {
                    "id": seiyuu.get("id"),
                    "name": seiyuu.text,
                    "picture": seiyuu.get("picture")
                }
            
            characters.append(character_data)
        
        return characters