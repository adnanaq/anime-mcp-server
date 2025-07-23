"""Simkl.com scraping client implementation."""

import re
from typing import Any, Dict, List, Optional

from ..base_scraper import BaseScraper


class SimklScraper(BaseScraper):
    """Simkl.com scraping client for anime database."""

    def __init__(self, **kwargs):
        """Initialize Simkl scraper."""
        super().__init__(service_name="simkl", **kwargs)
        self.base_url = "https://simkl.com"

    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime information by Simkl ID."""
        # Check cache first
        cache_key = f"simkl_anime_{anime_id}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                pass

        try:
            url = f"{self.base_url}/anime/{anime_id}"
            response = await self._make_request(url, timeout=10)

            # Parse HTML
            soup = self._parse_html(response["content"])

            # Check for MODB's dead entry pattern first
            if self._is_dead_entry(response["content"]):
                return None

            # Extract base data
            result = self._extract_base_data(soup, url)
            result["domain"] = "simkl"
            result["simkl_id"] = anime_id
            result["id"] = anime_id  # Map to universal id field

            # Extract anime-specific data
            anime_data = self._extract_anime_data(soup)
            result.update(anime_data)

            # Cache the result
            if self.cache_manager and result.get("title"):
                try:
                    await self.cache_manager.set(cache_key, result)
                except:
                    pass

            return result if result.get("title") else None

        except Exception as e:
            # Re-raise circuit breaker exceptions
            if "circuit breaker" in str(e).lower():
                raise
            return None

    def _is_dead_entry(self, html_content: str) -> bool:
        """Check if this is a dead entry using MODB's detection pattern."""
        # MODB's exact dead entry pattern
        dead_entry_indicator = '<meta property="og:title" content="Simkl - Watch and Track Movies, Anime, TV Shows" />'
        return dead_entry_indicator in html_content

    def _extract_anime_data(self, soup) -> Dict[str, Any]:
        """Extract anime-specific data from Simkl page."""
        data = {}

        # Extract title from multiple sources
        title_data = self._extract_title_data(soup)
        data.update(title_data)

        # Extract description/synopsis
        description = self._extract_description(soup)
        if description:
            data["description"] = description

        # Extract metadata
        metadata = self._extract_metadata(soup)
        data.update(metadata)

        # Extract genres
        genres = self._extract_genres(soup)
        if genres:
            data["genres"] = genres

        # Extract rating/score
        rating = self._extract_rating(soup)
        if rating:
            data["rating"] = rating

        return data

    def _extract_title_data(self, soup) -> Dict[str, Any]:
        """Extract title information from Simkl page."""
        data = {}

        # Strategy 1: Try OpenGraph title (most reliable)
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
            # Clean Simkl-specific parts
            title = re.sub(r' \| Simkl.*$', '', title)
            if title and not title.lower().startswith('simkl'):
                data["title"] = title

        # Strategy 2: Try h1 heading
        if not data.get("title"):
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text().strip()
                if title and not title.lower().startswith('simkl'):
                    data["title"] = title

        # Strategy 3: Try page title
        if not data.get("title"):
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text().strip()
                # Clean Simkl-specific parts
                title = re.sub(r' \| Simkl.*$', '', title_text)
                if title and title != title_text and not title.lower().startswith('simkl'):
                    data["title"] = title

        # Strategy 4: Try JSON-LD structured data
        if not data.get("title"):
            json_ld_script = soup.find('script', type='application/ld+json')
            if json_ld_script:
                try:
                    import json
                    json_data = json.loads(json_ld_script.string)
                    if isinstance(json_data, dict) and 'name' in json_data:
                        name = json_data['name'].strip()
                        if name and not name.lower().startswith('simkl'):
                            data["title"] = name
                except:
                    pass

        return data

    def _extract_description(self, soup) -> Optional[str]:
        """Extract description/synopsis from Simkl page."""
        # Try OpenGraph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            desc = og_desc['content'].strip()
            if len(desc) > 50:  # Ensure it's substantial
                return desc

        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
            if len(desc) > 50:
                return desc

        # Try specific content selectors
        desc_selectors = [
            '.plot', '.description', '.synopsis', '.overview',
            '[data-description]', '.show-description'
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                desc = self._clean_text(element.get_text())
                if len(desc) > 50:
                    return desc

        return None

    def _extract_metadata(self, soup) -> Dict[str, Any]:
        """Extract metadata from Simkl page."""
        data = {}

        # Extract year/date information
        date_elements = soup.find_all(['time', 'span'], class_=re.compile(r'date|year'))
        for element in date_elements:
            date_text = element.get_text().strip()
            if re.match(r'\d{4}', date_text):
                data['year'] = date_text
                break

        # Extract type information (TV, Movie, OVA, etc.)
        type_elements = soup.find_all(['span', 'div'], class_=re.compile(r'type|format'))
        for element in type_elements:
            type_text = element.get_text().strip()
            if type_text and len(type_text) < 20:  # Reasonable type length
                data['type'] = type_text
                break

        # Extract episode count
        episode_elements = soup.find_all(['span', 'div'], string=re.compile(r'\d+\s+episodes?', re.I))
        for element in episode_elements:
            episode_match = re.search(r'(\d+)\s+episodes?', element.get_text(), re.I)
            if episode_match:
                data['episodes'] = int(episode_match.group(1))
                break

        return data

    def _extract_genres(self, soup) -> List[str]:
        """Extract genres from Simkl page."""
        genres = []

        # Look for genre containers
        genre_selectors = [
            '.genres a', '.genre-tag', '.tags a',
            'a[href*="/genre/"]', '.category-tag'
        ]

        for selector in genre_selectors:
            genre_elements = soup.select(selector)
            for element in genre_elements:
                genre_text = self._clean_text(element.get_text())
                if genre_text and genre_text not in genres and len(genre_text) < 30:
                    genres.append(genre_text)

        return genres[:10]  # Limit to 10 genres

    def _extract_rating(self, soup) -> Optional[str]:
        """Extract rating/score from Simkl page."""
        # Try various rating selectors
        rating_selectors = [
            '.rating-value', '.score', '.imdb-rating',
            '[data-rating]', '.user-rating'
        ]

        for selector in rating_selectors:
            rating_elem = soup.select_one(selector)
            if rating_elem:
                rating_text = self._clean_text(rating_elem.get_text())
                # Extract numeric rating
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    return rating_match.group(1)

        return None

    async def search_anime(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search anime on Simkl - may be limited due to JavaScript requirements."""
        # Simkl search might require JavaScript, return empty for now
        # This could be enhanced later with more sophisticated techniques
        return []