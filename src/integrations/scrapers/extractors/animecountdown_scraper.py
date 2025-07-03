"""AnimeCountdown.net scraping client implementation."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

from ..base_scraper import BaseScraper


class AnimeCountdownScraper(BaseScraper):
    """AnimeCountdown.net scraping client for anime release countdowns."""

    def __init__(self, **kwargs):
        """Initialize AnimeCountdown scraper."""
        super().__init__(service_name="animecountdown", **kwargs)
        self.base_url = "https://animecountdown.com"

    async def get_anime_countdown_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get anime countdown information by slug."""
        # Check cache first
        cache_key = f"animecountdown_slug_{slug}"
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

            # Parse HTML
            soup = self._parse_html(response["content"])

            # Extract base data
            result = self._extract_base_data(soup, url)
            result["domain"] = "animecountdown"
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

    async def get_anime_countdown_by_id(
        self, anime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get anime countdown information by ID."""
        # Check cache first
        cache_key = f"animecountdown_id_{anime_id}"
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

            # Parse and extract data (same as by slug)
            soup = self._parse_html(response["content"])
            result = self._extract_base_data(soup, url)
            result["domain"] = "animecountdown"
            result["animecountdown_id"] = anime_id

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

    async def search_anime_countdowns(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search anime countdowns on AnimeCountdown.net."""
        try:
            # AnimeCountdown search URL
            search_url = f"{self.base_url}/search?q={quote(query)}"
            response = await self._make_request(search_url, timeout=10)

            # Parse search results
            soup = self._parse_html(response["content"])
            results = self._parse_search_results(soup, limit)

            return results

        except Exception:
            return []

    async def get_upcoming_anime(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get upcoming anime countdowns."""
        try:
            # AnimeCountdown upcoming URL
            upcoming_url = f"{self.base_url}/upcoming"
            response = await self._make_request(upcoming_url, timeout=10)

            # Parse upcoming results
            soup = self._parse_html(response["content"])
            results = self._parse_upcoming_results(soup, limit)

            return results

        except Exception:
            return []

    async def get_episode_countdowns(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get episode countdown information for specific anime."""
        try:
            # Same as get_anime_countdown_by_slug but focused on episode data
            url = f"{self.base_url}/anime/{slug}"
            response = await self._make_request(url, timeout=10)

            soup = self._parse_html(response["content"])
            episodes = self._extract_episode_countdowns(soup)

            if episodes:
                return {
                    "anime_slug": slug,
                    "episodes": episodes,
                    "source": "animecountdown",
                }

            return None

        except Exception:
            return None

    # === Generic methods for service compatibility ===

    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Generic search anime method - delegates to search_anime_countdowns."""
        return await self.search_anime_countdowns(query)

    async def get_anime_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Generic get anime by slug - delegates to get_anime_countdown_by_slug."""
        return await self.get_anime_countdown_by_slug(slug)

    async def get_anime_countdown(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get countdown info - alias for get_anime_countdown_by_slug."""
        return await self.get_anime_countdown_by_slug(slug)

    async def get_currently_airing(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get currently airing anime - filter upcoming for currently airing."""
        try:
            upcoming = await self.get_upcoming_anime(limit * 2)  # Get more to filter
            if not upcoming:
                return []

            # Filter for currently airing (simple heuristic - items with countdown info)
            currently_airing = []
            for anime in upcoming:
                countdown_info = anime.get("countdown_info", {})
                if countdown_info.get("is_airing") or countdown_info.get(
                    "next_episode"
                ):
                    currently_airing.append(anime)
                if len(currently_airing) >= limit:
                    break

            return currently_airing[:limit]
        except Exception:
            return []

    async def get_popular_anime(
        self, time_period: str = "all_time", limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Get popular anime - use search for popular terms as fallback."""
        try:
            # Since AnimeCountdown doesn't have explicit popular lists,
            # search for popular anime terms and return results
            popular_terms = [
                "demon slayer",
                "attack on titan",
                "naruto",
                "one piece",
                "dragon ball",
            ]
            all_results = []

            for term in popular_terms:
                if len(all_results) >= limit:
                    break
                results = await self.search_anime_countdowns(term, limit=5)
                all_results.extend(results)

            # Remove duplicates and limit
            seen_titles = set()
            unique_results = []
            for anime in all_results:
                title = anime.get("title", "").lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(anime)
                if len(unique_results) >= limit:
                    break

            return unique_results[:limit]
        except Exception:
            return []

    def _extract_anime_data(self, soup) -> Dict[str, Any]:
        """Extract anime-specific data from AnimeCountdown page."""
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

        # Extract countdown timer data
        countdown_data = self._extract_countdown_timer(soup)
        if countdown_data:
            data["countdown"] = countdown_data

        # Extract release information
        release_data = self._extract_release_info(soup)
        data.update(release_data)

        # Extract anime details
        detail_data = self._extract_anime_details(soup)
        data.update(detail_data)

        return data

    def _extract_countdown_timer(self, soup) -> Optional[Dict[str, Any]]:
        """Extract countdown timer data from the page."""
        countdown_data = {}

        # Look for countdown containers using both find and select methods
        countdown_containers = [
            soup.find("div", class_="countdown-timer"),
            soup.find("div", class_="countdown-display"),
            soup.select_one(".timer"),
            soup.select_one(".countdown"),
        ]

        for container in countdown_containers:
            if not container:
                continue

            # Extract individual time components
            days_elem = container.find("span", class_="days")
            if days_elem:
                countdown_data["days"] = self._clean_text(days_elem.text)

            hours_elem = container.find("span", class_="hours")
            if hours_elem:
                countdown_data["hours"] = self._clean_text(hours_elem.text)

            minutes_elem = container.find("span", class_="minutes")
            if minutes_elem:
                countdown_data["minutes"] = self._clean_text(minutes_elem.text)

            # If no individual components, try to parse text patterns
            if not countdown_data:
                text = self._clean_text(container.text)

                # Pattern: "5d 12h 30m remaining"
                time_match = re.search(r"(\d+)d.*?(\d+)h.*?(\d+)m", text)
                if time_match:
                    countdown_data["days"] = time_match.group(1)
                    countdown_data["hours"] = time_match.group(2)
                    countdown_data["minutes"] = time_match.group(3)
                else:
                    # Pattern: "72 hours left"
                    hours_match = re.search(
                        r"(\d+)\s*hours?\s*(?:left|remaining)", text
                    )
                    if hours_match:
                        countdown_data["hours"] = hours_match.group(1)

                    # Pattern: "5 days remaining"
                    days_match = re.search(r"(\d+)\s*days?\s*(?:left|remaining)", text)
                    if days_match:
                        countdown_data["days"] = days_match.group(1)

            if countdown_data:  # If we found data, break
                break

        # Extract release date from data attributes
        timer_elem = soup.find(attrs={"data-release-date": True})
        if timer_elem:
            countdown_data["release_timestamp"] = timer_elem.get("data-release-date")

        return countdown_data if countdown_data else None

    def _extract_release_info(self, soup) -> Dict[str, Any]:
        """Extract release information from the page."""
        release_data = {}

        # Look for release info container
        release_containers = [
            soup.find("div", class_="release-info"),
            soup.find("div", class_="release-details"),
            soup.find(".release-data"),
        ]

        for container in release_containers:
            if not container:
                continue

            # Release date
            release_date_elem = container.find("div", class_="release-date")
            if release_date_elem:
                release_data["release_date"] = self._clean_text(release_date_elem.text)

            # Release time
            release_time_elem = container.find("div", class_="release-time")
            if release_time_elem:
                release_data["release_time"] = self._clean_text(release_time_elem.text)

            # Anime type
            type_elem = container.find("div", class_="anime-type")
            if type_elem:
                release_data["type"] = self._clean_text(type_elem.text)

            # Episode info
            episode_elem = container.find("div", class_="episode-info")
            if episode_elem:
                release_data["episode_info"] = self._clean_text(episode_elem.text)

            if release_data:  # If we found data, break
                break

        return release_data

    def _extract_anime_details(self, soup) -> Dict[str, Any]:
        """Extract anime details from detail items."""
        detail_data = {}

        # Look for detail containers
        detail_containers = [
            soup.find("div", class_="anime-details"),
            soup.find("div", class_="details"),
            soup.find(".anime-info"),
        ]

        for container in detail_containers:
            if not container:
                continue

            # Find detail items
            detail_items = container.find_all("div", class_="detail-item")

            for item in detail_items:
                label_elem = item.find("span", class_="label")
                value_elem = item.find("span", class_="value")

                if label_elem and value_elem:
                    label = self._clean_text(label_elem.text).lower().replace(":", "")
                    value = self._clean_text(value_elem.text)

                    # Map labels to fields
                    if label in ["studio", "studios"]:
                        detail_data["studio"] = value
                    elif label in ["genre", "genres"]:
                        detail_data["genres"] = [g.strip() for g in value.split(",")]
                    elif label in ["status"]:
                        detail_data["status"] = value
                    elif label in ["type"]:
                        detail_data["type"] = value
                    elif label in ["episodes"]:
                        detail_data["episodes"] = value

            if detail_data:  # If we found data, break
                break

        return detail_data

    def _extract_episode_countdowns(self, soup) -> List[Dict[str, Any]]:
        """Extract episode countdown data from the page."""
        episodes = []

        # Look for episode countdown containers
        episode_containers = [
            soup.find("div", class_="related-episodes"),
            soup.find("div", class_="episodes-list"),
            soup.find(".episode-countdowns"),
        ]

        for container in episode_containers:
            if not container:
                continue

            # Find episode elements
            episode_elements = container.find_all("div", class_="episode-countdown")

            for episode_elem in episode_elements:
                episode_data = {}

                # Episode number
                episode_num = episode_elem.get("data-episode")
                if episode_num:
                    episode_data["episode_number"] = int(episode_num)

                # Air date
                air_date = episode_elem.get("data-air-date")
                if air_date:
                    episode_data["air_date_raw"] = air_date

                # Episode title
                title_elem = episode_elem.find("span", class_="episode-title")
                if title_elem:
                    episode_data["title"] = self._clean_text(title_elem.text)

                # Air date (human readable)
                air_date_elem = episode_elem.find("span", class_="air-date")
                if air_date_elem:
                    episode_data["air_date"] = self._clean_text(air_date_elem.text)

                if episode_data:
                    episodes.append(episode_data)

            if episodes:  # If we found episodes, break
                break

        return episodes

    def _parse_search_results(self, soup, limit: int) -> List[Dict[str, Any]]:
        """Parse search results from AnimeCountdown search page."""
        results = []

        # Look for search result containers
        result_containers = [
            soup.find_all("div", class_="countdown-item"),
            soup.find_all("div", class_="result-item"),
            soup.find_all("div", class_="search-result"),
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
            # Extract title and link
            title_link = container.find("a") or (
                container.find("h3").find("a") if container.find("h3") else None
            )
            if not title_link:
                return None

            title = self._clean_text(title_link.text)
            href = title_link.get("href", "")

            # Extract slug from href
            slug_match = re.search(r"/anime/([^/?]+)", href)
            slug = slug_match.group(1) if slug_match else None

            if not title or not slug:
                return None

            result = {
                "title": title,
                "slug": slug,
                "url": urljoin(self.base_url, href),
                "domain": "animecountdown",
            }

            # Extract countdown preview
            countdown_elem = container.find(
                "div", class_="countdown-preview"
            ) or container.find("span", class_="countdown")
            if countdown_elem:
                result["countdown_preview"] = self._clean_text(countdown_elem.text)

            # Extract type
            type_elem = container.find("div", class_="anime-type")
            if type_elem:
                result["type"] = self._clean_text(type_elem.text)

            # Extract release date
            date_elem = container.find("div", class_="release-date")
            if date_elem:
                result["release_date"] = self._clean_text(date_elem.text)

            return result

        except Exception:
            return None

    def _parse_upcoming_results(self, soup, limit: int) -> List[Dict[str, Any]]:
        """Parse upcoming anime results."""
        results = []

        # Look for upcoming list containers
        upcoming_containers = [
            soup.find("div", class_="upcoming-list"),
            soup.find("div", class_="countdown-list"),
            soup.find("div", class_="anime-list"),
        ]

        for container in upcoming_containers:
            if not container:
                continue

            # Find countdown items
            countdown_items = container.find_all("div", class_="countdown-item")

            for item in countdown_items[:limit]:
                result = self._parse_single_search_result(item)
                if result:
                    # Add release date from data attribute if available
                    release_attr = item.get("data-release")
                    if release_attr:
                        result["release_date_raw"] = release_attr

                    results.append(result)

            if results:
                break

        return results[:limit]
