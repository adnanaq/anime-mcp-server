"""AniSearch.de scraping client implementation."""

import re
from typing import Any, Dict, List, Optional

from ..base_scraper import BaseScraper


class AniSearchScraper(BaseScraper):
    """AniSearch.com scraping client for international anime database."""

    def __init__(self, **kwargs):
        """Initialize AniSearch scraper."""
        super().__init__(service_name="anisearch", **kwargs)
        self.base_url = "https://www.anisearch.com"

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
            result["id"] = slug  # Map to universal id field

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

        # Extract description/synopsis from content areas
        description = self._extract_actual_synopsis(soup)
        data["description"] = description

        # Extract metadata from header elements
        header_data = self._extract_header_data(soup)
        data.update(header_data)

        # Extract title variants and primary title
        title_data = self._extract_title_variants(soup)
        data.update(title_data)

        # Extract enhanced data from JSON-LD and OpenGraph
        enhanced_data = self._extract_enhanced_anisearch_data(soup)
        data.update(enhanced_data)

        # Extract additional missing properties
        additional_data = self._extract_additional_properties(soup)
        data.update(additional_data)

        # Extract genres
        genres = self._extract_genres(soup)
        data["genres"] = genres

        # Extract rating
        rating = self._extract_rating(soup)
        if rating:
            data["rating"] = rating

        return data

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

    def _extract_header_data(self, soup) -> Dict[str, Any]:
        """Extract data from AniSearch header-value pairs."""
        data = {}

        # Find all header elements
        headers = soup.find_all("span", class_="header")

        for header in headers:
            header_text = self._clean_text(header.text).lower().replace(":", "")

            # Get the parent element to find associated data
            parent = header.parent
            if not parent:
                continue

            # Extract different types of data based on header
            if header_text in ["studio", "studios"]:
                studios = self._extract_studios_from_parent(parent)
                if studios:
                    data["studios"] = studios

            elif header_text in ["adaptiert von", "adapted from", "source"]:
                source = self._extract_source_from_parent(parent)
                if source:
                    data["source"] = source

            elif header_text in ["staff"]:
                staff = self._extract_staff_from_parent(parent)
                if staff:
                    data["staff"] = staff

            elif header_text in ["webseite", "website"]:
                external_links = self._extract_external_links_from_parent(parent)
                if external_links:
                    data["external_links"] = external_links

            elif header_text in ["status"]:
                status = self._extract_status_from_parent(parent)
                if status:
                    data["status"] = status

        # Enhanced staff extraction fallback only for .creators selector
        if not data.get("staff"):
            creators_elements = soup.select(".creators")
            if creators_elements:
                enhanced_staff = self._extract_staff_from_creators_selector(soup)
                if enhanced_staff:
                    data["staff"] = enhanced_staff

        return data

    def _extract_studios_from_parent(self, parent) -> List[str]:
        """Extract studio information from parent element."""
        studios = []

        # Look for company links (href might be relative)
        for link in parent.find_all("a"):
            href = link.get("href", "")
            if "company/" in href:  # Remove leading slash requirement
                studio_name = self._clean_text(link.text)
                if studio_name and studio_name not in studios:
                    studios.append(studio_name)

        return studios

    def _extract_source_from_parent(self, parent) -> Optional[str]:
        """Extract source material from parent element."""
        # Get text content from the same parent, after the header
        text_content = parent.get_text()
        header_elem = parent.find("span", class_="header")
        if header_elem:
            header_text = header_elem.text
            remaining_text = text_content.replace(header_text, "").strip()
            if remaining_text:
                return remaining_text

        # Get next sibling text
        next_elem = parent.find_next_sibling()
        if next_elem and hasattr(next_elem, "text"):
            source_text = self._clean_text(next_elem.text)
            if source_text and ":" not in source_text:  # Avoid getting next header
                return source_text

        return None

    def _extract_staff_from_parent(self, parent) -> List[Dict[str, str]]:
        """Extract staff information from parent element."""
        staff = []

        # Look for person links (href might be relative)
        for link in parent.find_all("a"):
            href = link.get("href", "")
            if "person/" in href:  # Remove leading slash requirement
                staff_name = self._clean_text(link.text)
                if staff_name:
                    # Make href absolute if relative, using the current base_url
                    full_href = (
                        href if href.startswith("http") else f"{self.base_url}/{href}"
                    )
                    staff.append(
                        {
                            "name": staff_name,
                            "url": full_href,
                            "role": "Staff",  # AniSearch doesn't specify roles clearly
                        }
                    )

        return staff

    def _extract_staff_from_creators_selector(self, soup) -> List[Dict[str, str]]:
        """Extract staff from .creators selector as per analysis document."""
        staff = []

        # Try .creators selector (from analysis document)
        creators_elements = soup.select(".creators")
        for creator_elem in creators_elements:
            # Find all person links within creators
            for link in creator_elem.find_all("a"):
                href = link.get("href", "")
                if "person/" in href:
                    staff_name = self._clean_text(link.text)
                    if staff_name:
                        full_href = (
                            href
                            if href.startswith("http")
                            else f"{self.base_url}/{href}"
                        )
                        staff.append(
                            {"name": staff_name, "url": full_href, "role": "Creator"}
                        )

        return staff

    def _extract_external_links_from_parent(self, parent) -> List[Dict[str, str]]:
        """Extract external links from parent element."""
        links = []

        for link in parent.find_all("a"):
            href = link.get("href", "")
            text = self._clean_text(link.text)
            if href.startswith("http") and text:
                links.append({"name": text, "url": href})

        return links

    def _extract_status_from_parent(self, parent) -> Optional[str]:
        """Extract status from parent element."""
        # Get text content from the same parent, after the header
        text_content = parent.get_text()
        header_elem = parent.find("span", class_="header")
        if header_elem:
            header_text = header_elem.text
            remaining_text = text_content.replace(header_text, "").strip().lower()

            # Map German status to universal status
            if "abgeschlossen" in remaining_text or "beendet" in remaining_text:
                return "COMPLETED"
            elif "laufend" in remaining_text or "ongoing" in remaining_text:
                return "ONGOING"
            elif "angekündigt" in remaining_text or "upcoming" in remaining_text:
                return "UPCOMING"
            elif remaining_text:
                return remaining_text.upper()

        return None

    def _extract_enhanced_anisearch_data(self, soup) -> Dict[str, Any]:
        """Extract enhanced data using JSON-LD and OpenGraph metadata."""
        data = {}

        # Extract from JSON-LD data (already in base_data)
        json_ld = soup.find("script", type="application/ld+json")
        if json_ld:
            try:
                import json

                json_data = json.loads(json_ld.string)

                # Extract additional properties
                if "startDate" in json_data:
                    start_date = json_data.get("startDate")
                    data["start_date"] = start_date

                if "endDate" in json_data:
                    end_date = json_data.get("endDate")
                    data["end_date"] = end_date

                # Derive status from dates
                if "startDate" in json_data and "endDate" in json_data:
                    start_date = json_data.get("startDate")
                    end_date = json_data.get("endDate")

                    if not data.get("status"):
                        data["status"] = self._derive_status_from_dates(
                            start_date, end_date
                        )

                    # Extract year
                    if start_date:
                        data["year"] = start_date[:4]

                # Extract episode count
                if "numberOfEpisodes" in json_data:
                    data["episodes"] = json_data["numberOfEpisodes"]

                # Extract rating data
                if "aggregateRating" in json_data:
                    rating_data = json_data["aggregateRating"]
                    data.update(
                        {
                            "score": rating_data.get("ratingValue"),
                            "score_count": rating_data.get("ratingCount"),
                            "rating_scale_min": rating_data.get("worstRating"),
                            "rating_scale_max": rating_data.get("bestRating"),
                        }
                    )

                # Extract canonical ID
                if "@id" in json_data:
                    data["canonical_id"] = json_data["@id"]

            except (json.JSONDecodeError, AttributeError):
                pass

        # Extract from OpenGraph data
        og_data = {}
        for meta in soup.find_all("meta", property=True):
            prop = meta.get("property", "")
            content = meta.get("content", "")
            if prop.startswith("og:"):
                og_data[prop] = content

        # Map OpenGraph to universal properties
        if "og:title" in og_data:
            # Check if different from main title for alternative titles
            og_title = og_data["og:title"]
            main_title = data.get("title", "")
            if og_title and og_title != main_title:
                data["title_alternative"] = og_title

        if "og:type" in og_data:
            # Map og:type to universal format
            og_type = og_data["og:type"]
            if og_type == "video.tv_show":
                data["format"] = "TV"
            elif og_type == "video.movie":
                data["format"] = "MOVIE"
            else:
                data["format"] = og_type

        return data

    def _derive_status_from_dates(self, start_date: str, end_date: str) -> str:
        """Derive anime status from start and end dates."""
        from datetime import datetime

        try:
            now = datetime.now()
            start = datetime.fromisoformat(start_date) if start_date else None
            end = datetime.fromisoformat(end_date) if end_date else None

            if not start:
                return "UNKNOWN"

            if start > now:
                return "UPCOMING"
            elif end and end < now:
                return "COMPLETED"
            elif not end:
                return "ONGOING"
            else:
                return "ONGOING"

        except (ValueError, TypeError):
            return "UNKNOWN"

    def _extract_title_variants(self, soup) -> Dict[str, Any]:
        """Extract primary title, native title and synonyms from AniSearch page."""
        data = {}

        # Primary title from h1 or OpenGraph
        primary_title = None
        h1_elem = soup.find("h1")
        if h1_elem:
            primary_title = self._clean_text(h1_elem.text)

        # Fallback to OpenGraph title
        if not primary_title:
            og_title = soup.find("meta", property="og:title")
            if og_title:
                primary_title = og_title.get("content", "")

        data["title"] = primary_title

        # Find title elements that are likely anime titles (not character names)
        title_elements = soup.find_all(["div", "span"], class_="title")

        anime_titles = []
        native_title = None
        english_title = None

        for elem in title_elements:
            title_text = self._clean_text(elem.text)
            if not title_text or title_text in anime_titles:
                continue

            # Skip character names and other non-anime titles
            parent = elem.parent
            parent_classes = parent.get("class", []) if parent else []

            # Skip if this is likely a character name (very short single names)
            if (
                len(title_text.split()) == 1
                and len(title_text) < 10
                and title_text.isupper()
                and "details" in parent_classes
            ):
                continue

            anime_titles.append(title_text)

            # Check if this contains Japanese characters (native title)
            has_japanese = any(ord(char) > 127 for char in title_text)
            if has_japanese and not native_title:
                native_title = title_text
            elif not has_japanese and not english_title:
                english_title = title_text

        # Set native title
        if native_title:
            data["title_native"] = native_title

        # Set English title if different from main title
        import re

        main_title_clean = (
            re.sub(r"\s*\(\d{4}\)\s*$", "", primary_title or "")
            if primary_title
            else ""
        )
        if (
            english_title
            and english_title != main_title_clean
            and english_title != primary_title
        ):
            data["title_english"] = english_title

        # Create synonyms list (exclude main, native, and English titles)
        exclude_titles = {main_title_clean, native_title, english_title, primary_title}

        synonyms = []
        for title in anime_titles:
            # Clean up title and check if it's a valid synonym
            if title not in exclude_titles and title:
                # Skip obvious non-synonym titles
                skip_patterns = [" - op:", "cm gekijou", "opening", "ending"]
                if not any(skip in title.lower() for skip in skip_patterns):
                    synonyms.append(title)

        if synonyms:
            data["synonyms"] = synonyms

        return data

    def _extract_additional_properties(self, soup) -> Dict[str, Any]:
        """Extract additional properties from analysis document."""
        data = {}

        # Extract episode duration and calculate total runtime
        duration_info = self._extract_duration(soup)
        if duration_info:
            data["episode_duration"] = duration_info

            # Calculate total runtime if we have episodes and duration
            episodes = data.get("episodes") or self._get_episode_count_from_json(soup)
            if episodes and duration_info:
                try:
                    # Parse duration (e.g., "24 min" -> 24)
                    duration_minutes = int(
                        "".join(filter(str.isdigit, str(duration_info)))
                    )
                    episode_count = int(str(episodes))
                    total_minutes = duration_minutes * episode_count
                    data["total_runtime"] = (
                        f"{total_minutes} minutes ({total_minutes // 60}h {total_minutes % 60}m)"
                    )
                except (ValueError, TypeError):
                    pass

        # Extract streaming platforms
        streaming_data = self._extract_streaming_platforms(soup)
        if streaming_data:
            data["streaming_links"] = streaming_data

        # Extract broadcast schedule
        broadcast_info = self._extract_broadcast_schedule(soup)
        if broadcast_info:
            data["broadcast_schedule"] = broadcast_info

        return data

    def _extract_duration(self, soup) -> Optional[str]:
        """Extract episode duration from page text."""
        import re

        # Look for duration text patterns in page text
        page_text = soup.get_text()
        match = re.search(r"(\d+)\s*min", page_text, re.IGNORECASE)
        return f"{match.group(1)} min" if match else None

    def _get_episode_count_from_json(self, soup) -> Optional[str]:
        """Get episode count from JSON-LD if not already extracted."""
        json_ld = soup.find("script", type="application/ld+json")
        if json_ld:
            try:
                import json

                data = json.loads(json_ld.string)
                return data.get("numberOfEpisodes")
            except:
                pass
        return None

    def _extract_streaming_platforms(self, soup) -> List[Dict[str, str]]:
        """Extract streaming platform links."""
        streaming_links = []

        # Look for streaming-related links
        streaming_keywords = [
            "crunchyroll",
            "netflix",
            "amazon",
            "funimation",
            "hulu",
            "stream",
        ]

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.text.strip()

            # Check if this is a streaming platform link
            if any(
                keyword in href.lower() or keyword in text.lower()
                for keyword in streaming_keywords
            ):
                if href.startswith("http") and text:
                    streaming_links.append(
                        {
                            "name": text,
                            "url": href,
                            "platform": self._identify_streaming_platform(href, text),
                        }
                    )

        return streaming_links

    def _identify_streaming_platform(self, url: str, text: str) -> str:
        """Identify the streaming platform from URL or text."""
        url_lower = url.lower()
        text_lower = text.lower()

        platforms = {
            "crunchyroll": "Crunchyroll",
            "netflix": "Netflix",
            "amazon": "Amazon Prime",
            "funimation": "Funimation",
            "hulu": "Hulu",
            "disney": "Disney+",
        }

        for key, name in platforms.items():
            if key in url_lower or key in text_lower:
                return name

        return "Other"

    def _extract_broadcast_schedule(self, soup) -> Optional[str]:
        """Extract broadcast schedule information."""
        # Look for broadcast/schedule related text
        schedule_keywords = ["broadcast", "airs", "schedule", "time", "jst", "every"]

        for text in soup.find_all(string=True):
            text_content = text.strip()
            if any(keyword in text_content.lower() for keyword in schedule_keywords):
                # Look for time patterns (e.g., "Sunday 23:15", "Every Tuesday")
                import re

                time_patterns = [
                    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+\d{1,2}:\d{2}",
                    r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                    r"\d{1,2}:\d{2}\s*(jst|pst|est)",
                ]

                for pattern in time_patterns:
                    match = re.search(pattern, text_content, re.IGNORECASE)
                    if match:
                        return match.group(0)

        return None

    def _extract_actual_synopsis(self, soup) -> Optional[str]:
        """Extract actual anime synopsis from textblock details-text with lang=en."""
        # Look for the exact element: div with lang="en" and class="textblock details-text"
        synopsis_element = soup.find(
            "div", {"lang": "en", "class": "textblock details-text"}
        )

        if synopsis_element:
            text = self._clean_text(synopsis_element.text)

            if len(text) > 50:  # Make sure it's substantial
                # Clean up the text by removing technical annotations
                end_markers = ["annotation:", "source:", "note:", "episodes were later"]
                clean_text = text

                for marker in end_markers:
                    if marker.lower() in text.lower():
                        clean_text = text[: text.lower().find(marker.lower())].strip()
                        break

                return clean_text

        return None
