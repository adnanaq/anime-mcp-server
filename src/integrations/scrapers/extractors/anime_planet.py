"""Anime-Planet scraping client implementation."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

from ..base_scraper import BaseScraper


class AnimePlanetScraper(BaseScraper):
    """Anime-Planet scraping client."""

    def __init__(self, **kwargs):
        """Initialize Anime-Planet scraper."""
        super().__init__(**kwargs)
        self.base_url = "https://www.anime-planet.com"

    async def get_anime_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get anime information by Anime-Planet slug."""
        # Check cache first
        cache_key = f"animeplanet_anime_{slug}"
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

            # Check if Cloudflare blocked us
            if (
                response.get("is_cloudflare_protected")
                and "checking your browser" in response["content"].lower()
            ):
                return None

            # Parse HTML
            soup = self._parse_html(response["content"])

            # Extract base data
            result = self._extract_base_data(soup, url)
            result["domain"] = "anime-planet"
            result["slug"] = slug

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

    async def search_anime(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search anime on Anime-Planet."""
        try:
            # Anime-Planet search URL
            search_url = f"{self.base_url}/anime/all?name={quote(query)}"
            response = await self._make_request(search_url, timeout=10)

            # Parse search results
            soup = self._parse_html(response["content"])
            results = self._parse_search_results(soup, limit)

            return results

        except Exception:
            return []

    def _extract_anime_data(self, soup) -> Dict[str, Any]:
        """Extract anime-specific data from Anime-Planet page."""
        data = {}

        # Title - multiple possible selectors
        title_selectors = [
            ("h1", {"itemprop": "name"}),
            ("h1", {}),
            (".title", {}),
            ("h2", {"class": "title"}),
        ]

        title = None
        for tag, attrs in title_selectors:
            title_elem = soup.find(tag, attrs)
            if title_elem:
                title = self._clean_text(title_elem.text)
                break

        data["title"] = title

        # Synopsis/Description - multiple possible locations
        synopsis_selectors = [
            ("p", {"itemprop": "description"}),
            (".entryDetails p", {}),
            (".synopsis", {}),
            (".description", {}),
            (".entry-description", {}),
        ]

        synopsis = None
        for selector, attrs in synopsis_selectors:
            if "." in selector:
                # CSS selector
                synopsis_elem = soup.select_one(selector)
            else:
                synopsis_elem = soup.find(selector, attrs)

            if synopsis_elem:
                synopsis = self._clean_text(synopsis_elem.text)
                break

        data["synopsis"] = synopsis

        # Extract metadata from info table
        info_data = self._extract_info_table(soup)
        data.update(info_data)

        # Extract tags/genres
        tags = self._extract_tags(soup)
        data["tags"] = tags

        # Extract rating if available
        rating = self._extract_rating(soup)
        if rating:
            data["rating"] = rating

        return data

    def _extract_info_table(self, soup) -> Dict[str, Any]:
        """Extract information from the anime info table."""
        info_data = {}

        # Look for info table in various locations
        info_containers = [
            soup.find("table"),
            soup.find(".entryBox table"),
            soup.find(".info-table"),
            soup.find(".anime-info"),
        ]

        for container in info_containers:
            if not container:
                continue

            # Extract key-value pairs from table rows
            rows = container.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = self._clean_text(cells[0].text).lower().replace(":", "")
                    value = self._clean_text(cells[1].text)

                    # Map common fields
                    if key in ["type"]:
                        info_data["type"] = value
                    elif key in ["episodes", "episode count"]:
                        info_data["episodes"] = value
                    elif key in ["status"]:
                        info_data["status"] = value
                    elif key in ["aired", "air date"]:
                        info_data["aired"] = value
                    elif key in ["studio", "studios"]:
                        info_data["studio"] = value
                    elif key in ["year"]:
                        info_data["year"] = value

            if info_data:  # If we found data, break
                break

        return info_data

    def _extract_tags(self, soup) -> List[str]:
        """Extract tags/genres from the page."""
        tags = []

        # Look for tags in various locations
        tag_containers = [
            soup.find(".tags"),
            soup.find(".genres"),
            soup.find(".tags-list"),
            soup.find_all("a", href=re.compile(r"/tags/")),
            soup.find_all("a", href=re.compile(r"/genre/")),
        ]

        for container in tag_containers:
            if not container:
                continue

            if isinstance(container, list):
                # List of tag links
                for link in container:
                    tag_text = self._clean_text(link.text)
                    if tag_text and tag_text not in tags:
                        tags.append(tag_text)
            else:
                # Container with tag links
                tag_links = container.find_all("a")
                for link in tag_links:
                    tag_text = self._clean_text(link.text)
                    if tag_text and tag_text not in tags:
                        tags.append(tag_text)

        return tags[:10]  # Limit to 10 tags

    def _extract_rating(self, soup) -> Optional[str]:
        """Extract rating/score if available."""
        rating_selectors = [
            (".rating", {}),
            (".score", {}),
            (".avg-rating", {}),
            ("span", {"class": "rating"}),
        ]

        for selector, attrs in rating_selectors:
            if "." in selector:
                rating_elem = soup.select_one(selector)
            else:
                rating_elem = soup.find(selector, attrs)

            if rating_elem:
                rating_text = self._clean_text(rating_elem.text)
                # Extract numeric rating
                rating_match = re.search(r"(\d+\.?\d*)", rating_text)
                if rating_match:
                    return rating_match.group(1)

        return None

    def _parse_search_results(self, soup, limit: int) -> List[Dict[str, Any]]:
        """Parse search results from Anime-Planet search page."""
        results = []

        # Look for search result containers - Anime-Planet uses <li class="card"> in <ul class="cardDeck">
        result_containers = [
            soup.find_all("li", class_="card"),  # Actual structure used by Anime-Planet
            soup.find_all("div", class_="search-result"),
            soup.find_all("div", class_="card"),
            soup.find_all("div", class_="anime-card"),
            soup.find_all("li", class_=re.compile(r".*result.*")),
        ]

        for containers in result_containers:
            if not containers:
                continue

            for container in containers[:limit]:
                result = self._parse_single_search_result(container)
                if result:
                    results.append(result)

            if results:  # If we found results, break
                break

        return results[:limit]

    def _parse_single_search_result(self, container) -> Optional[Dict[str, Any]]:
        """Parse a single search result item."""
        try:
            # Extract title and link - Anime-Planet structure: <a><h3 class="cardName">Title</h3></a>
            title_link = container.find("a")
            if not title_link:
                return None

            href = title_link.get("href", "")

            # Extract title from h3.cardName inside the link
            title_elem = title_link.find("h3", class_="cardName")
            if title_elem:
                title = self._clean_text(title_elem.text)
            else:
                # Fallback to link text
                title = self._clean_text(title_link.text)

            # Extract slug from href
            slug_match = re.search(r"/anime/([^/?]+)", href)
            slug = slug_match.group(1) if slug_match else None

            if not title or not slug:
                return None

            result = {
                "title": title,
                "slug": slug,
                "url": urljoin(self.base_url, href),
                "domain": "anime-planet",
            }

            # Extract additional info if available
            type_elem = container.find("div", class_="type") or container.find(
                "div", class_="anime-type"
            )
            if type_elem:
                result["type"] = self._clean_text(type_elem.text)

            synopsis_elem = container.find("p", class_="synopsis") or container.find(
                "p"
            )
            if synopsis_elem:
                synopsis = self._clean_text(synopsis_elem.text)
                if synopsis and len(synopsis) > 20:  # Only if substantial
                    result["synopsis"] = (
                        synopsis[:200] + "..." if len(synopsis) > 200 else synopsis
                    )

            return result

        except Exception:
            return None
