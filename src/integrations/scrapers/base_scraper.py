"""Base scraping client using cloudscraper and BeautifulSoup."""

import asyncio
import json
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import cloudscraper
from bs4 import BeautifulSoup

from ..clients.base_client import BaseClient


class BaseScraper(BaseClient):
    """Base class for web scraping clients using cloudscraper."""

    def __init__(self, service_name: str, **kwargs):
        """Initialize base scraper."""
        super().__init__(service_name=service_name, **kwargs)
        self.scraper = None

    def _create_scraper(self) -> cloudscraper.CloudScraper:
        """Create cloudscraper instance with optimal settings."""
        # Use working approach from existing scripts
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "linux", "desktop": True}
        )
        return scraper

    async def _make_request(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request using cloudscraper.

        Args:
            url: URL to scrape
            **kwargs: Additional arguments for the request

        Returns:
            Dict containing response data and metadata
        """
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise Exception("Circuit breaker is open")

        # Rate limiting is now handled by BaseClient's make_request method

        # Create scraper if not exists
        if not self.scraper:
            self.scraper = self._create_scraper()

        try:
            # Use working approach: run in thread pool without complex headers
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.scraper.get(url, timeout=kwargs.get("timeout", 10))
            )

            # Check response status
            if response.status_code >= 400:
                raise Exception(f"HTTP {response.status_code} error for {url}")

            # Simple text extraction - let cloudscraper handle encoding
            content = response.text

            return {
                "url": url,
                "status_code": response.status_code,
                "content": content,
                "headers": dict(response.headers),
                "cookies": dict(response.cookies),
                "encoding": response.encoding,
                "is_cloudflare_protected": self._detect_cloudflare(response),
            }

        except Exception as e:
            # Log error but don't re-raise to maintain graceful degradation
            error_msg = f"Scraping error for {url}: {str(e)}"
            if self.error_handler:
                await self.error_handler.handle_error(error_msg)
            raise Exception(error_msg)

    def _detect_cloudflare(self, response) -> bool:
        """Detect if response is from Cloudflare protection."""
        cf_indicators = [
            "cf-ray" in response.headers,
            "cloudflare" in response.headers.get("server", "").lower(),
            "cf-browser-verification" in response.text,
            "__cf_bm" in response.cookies,
            "checking your browser" in response.text.lower(),
        ]
        return any(cf_indicators)

    def _parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content with BeautifulSoup."""
        return BeautifulSoup(html, "html.parser")

    def _extract_json_ld(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract JSON-LD structured data from HTML."""
        scripts = soup.find_all("script", type="application/ld+json")

        for script in scripts:
            if not script.string:
                continue

            try:
                data = json.loads(script.string)

                # Look for relevant schema types
                if isinstance(data, dict):
                    schema_type = data.get("@type", "")
                    if schema_type in [
                        "TVSeries",
                        "Movie",
                        "CreativeWork",
                        "VideoObject",
                        "WebPage",
                    ]:
                        return data
                elif isinstance(data, list):
                    # Sometimes there are multiple schemas
                    for item in data:
                        if isinstance(item, dict):
                            schema_type = item.get("@type", "")
                            if schema_type in [
                                "TVSeries",
                                "Movie",
                                "CreativeWork",
                                "VideoObject",
                            ]:
                                return item
            except json.JSONDecodeError:
                continue

        return None

    def _extract_opengraph(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract OpenGraph metadata."""
        og_data = {}

        # Find all OpenGraph meta tags
        og_tags = soup.find_all("meta", property=re.compile(r"^og:"))
        for tag in og_tags:
            property_name = tag.get("property", "").replace("og:", "")
            content = tag.get("content", "")
            if property_name and content:
                og_data[property_name] = content

        return og_data

    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract standard meta tags."""
        meta_data = {}

        # Common meta tags
        meta_names = ["description", "keywords", "author", "title"]
        for name in meta_names:
            tag = soup.find("meta", {"name": name})
            if tag and tag.get("content"):
                meta_data[name] = tag.get("content")

        return meta_data

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove HTML entities and extra whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = text.strip()

        # Remove common markup remnants
        text = re.sub(r"<[^>]+>", "", text)  # Remove any remaining HTML tags
        text = re.sub(r"\[.*?\]", "", text)  # Remove [tags]

        return text

    def _extract_base_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract basic data common to all anime sites."""
        base_data = {
            "url": url,
            "domain": urlparse(url).netloc,
            "page_title": None,
            "meta_description": None,
            "json_ld": None,
            "opengraph": None,
        }

        # Page title
        title_tag = soup.find("title")
        if title_tag:
            base_data["page_title"] = self._clean_text(title_tag.text)

        # Meta description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            base_data["meta_description"] = meta_desc.get("content", "")

        # Structured data
        json_ld = self._extract_json_ld(soup)
        if json_ld:
            base_data["json_ld"] = json_ld

        # OpenGraph
        og_data = self._extract_opengraph(soup)
        if og_data:
            base_data["opengraph"] = og_data

        return base_data

    async def close(self):
        """Close the scraper session."""
        if self.scraper:
            # Cloudscraper doesn't need explicit closing like aiohttp
            self.scraper = None
