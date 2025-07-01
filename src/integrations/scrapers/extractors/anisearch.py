"""AniSearch.de scraping client implementation."""

import re
from typing import Any, Dict, List, Optional

from ..base_scraper import BaseScraper


class AniSearchScraper(BaseScraper):
    """AniSearch.de scraping client for German anime database."""

    def __init__(self, **kwargs):
        """Initialize AniSearch scraper."""
        super().__init__(**kwargs)
        self.base_url = "https://www.anisearch.de"

    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime information by AniSearch ID."""
        # Check cache first
        cache_key = f"anisearch_anime_{anime_id}"
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

            # Extract base data
            result = self._extract_base_data(soup, url)
            result["domain"] = "anisearch"
            result["anisearch_id"] = anime_id

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

    async def get_anime_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get anime information by AniSearch slug."""
        # Check cache first
        cache_key = f"anisearch_slug_{slug}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                pass

        try:
            url = f"{self.base_url}/anime/{slug}"
            response = await self._make_request(url, timeout=10)

            # Parse and extract data (same as by ID)
            soup = self._parse_html(response["content"])
            result = self._extract_base_data(soup, url)
            result["domain"] = "anisearch"
            result["slug"] = slug

            # Extract anime data
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
            if "circuit breaker" in str(e).lower():
                raise
            return None

    async def search_anime(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search anime on AniSearch.de - DISABLED (confirmed non-functional).

        INVESTIGATION RESULTS:
        - AniSearch.de requires JavaScript to load search results dynamically
        - Static HTML contains only search filters and interface
        - Page shows: "Du benötigst JavaScript, um aniSearch in vollem Funktionsumfang nutzen zu können!"
        - Tested URL: /anime/index?q=query returns 0 results consistently
        - Individual anime pages work perfectly with rich JSON-LD data

        CONCLUSION: Search functionality cannot work with static HTML scraping.
        Individual anime page parsing remains fully functional.
        """
        return []

    def _extract_anime_data(self, soup) -> Dict[str, Any]:
        """Extract anime-specific data from AniSearch page."""
        data = {}

        # Title - multiple possible selectors
        title_selectors = [
            ("h1", {"class": "anime-title"}),
            ("h1", {}),
            (".anime-title", {}),
            ("h2", {"class": "title"}),
        ]

        title = None
        for tag, attrs in title_selectors:
            if "." in tag:
                title_elem = soup.select_one(tag)
            else:
                title_elem = soup.find(tag, attrs)

            if title_elem:
                title = self._clean_text(title_elem.text)
                break

        data["title"] = title

        # Description from various locations
        description_selectors = [
            (".anime-description p", {}),
            (".description", {}),
            (".synopsis", {}),
            ("div", {"class": "anime-description"}),
        ]

        description = None
        for selector, attrs in description_selectors:
            if "." in selector or " " in selector:
                desc_elem = soup.select_one(selector)
            else:
                desc_elem = soup.find(selector, attrs)

            if desc_elem:
                description = self._clean_text(desc_elem.text)
                break

        data["description"] = description

        # Extract metadata from info items
        info_data = self._extract_info_items(soup)
        data.update(info_data)

        # Extract genres
        genres = self._extract_genres(soup)
        data["genres"] = genres

        # Extract rating
        rating = self._extract_rating(soup)
        if rating:
            data["rating"] = rating

        return data

    def _extract_info_items(self, soup) -> Dict[str, Any]:
        """Extract information from AniSearch info items."""
        info_data = {}

        # Look for info items in various locations
        info_containers = [
            soup.find("div", class_="anime-info"),
            soup.find("div", class_="info-box"),
            soup.find(".anime-details"),
            soup.find(".details"),
        ]

        for container in info_containers:
            if not container:
                continue

            # Find info items
            info_items = container.find_all("div", class_="info-item")

            for item in info_items:
                label_elem = item.find("span", class_="label") or item.find(".label")
                value_elem = item.find("span", class_="value") or item.find(".value")

                if label_elem and value_elem:
                    label = self._clean_text(label_elem.text).lower().replace(":", "")
                    value = self._clean_text(value_elem.text)

                    # Map German labels to English
                    if label in ["typ", "type"]:
                        info_data["type"] = value
                    elif label in ["episoden", "episodes"]:
                        info_data["episodes"] = value
                    elif label in ["status"]:
                        info_data["status"] = value
                    elif label in ["jahr", "year"]:
                        info_data["year"] = value
                    elif label in ["studio", "studios"]:
                        info_data["studio"] = value
                    elif label in ["genre", "genres"]:
                        info_data["genre"] = value

            if info_data:  # If we found data, break
                break

        return info_data

    def _extract_genres(self, soup) -> List[str]:
        """Extract genres/tags from the page."""
        genres = []

        # Look for genre containers
        genre_containers = [
            soup.find("div", class_="anime-genres"),
            soup.find("div", class_="genres"),
            soup.find(".genre-tags"),
            soup.find_all("a", class_="genre-tag"),
            soup.find_all("a", href=re.compile(r"/genre/")),
        ]

        for container in genre_containers:
            if not container:
                continue

            if isinstance(container, list):
                # List of genre links
                for link in container:
                    genre_text = self._clean_text(link.text)
                    if genre_text and genre_text not in genres:
                        genres.append(genre_text)
            else:
                # Container with genre links
                genre_links = container.find_all("a")
                for link in genre_links:
                    genre_text = self._clean_text(link.text)
                    if genre_text and genre_text not in genres:
                        genres.append(genre_text)

        return genres[:10]  # Limit to 10 genres

    def _extract_rating(self, soup) -> Optional[str]:
        """Extract rating/score if available."""
        rating_selectors = [
            (".rating-value", {}),
            (".score", {}),
            (".rating", {}),
            (".bewertung", {}),
            ("span", {"class": "rating-value"}),
        ]

        for selector, attrs in rating_selectors:
            if "." in selector:
                rating_elem = soup.select_one(selector)
            else:
                rating_elem = soup.find(selector, attrs)

            if rating_elem:
                rating_text = self._clean_text(rating_elem.text)
                # Extract numeric rating (handle German format "9.0 von 10")
                rating_match = re.search(r"(\d+\.?\d*)", rating_text)
                if rating_match:
                    return rating_match.group(1)

        return None
