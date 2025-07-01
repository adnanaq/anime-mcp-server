"""Tests for AniDB XML client."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.integrations.cache_manager import CollaborativeCacheSystem
from src.integrations.clients.anidb_client import AniDBClient
from src.integrations.error_handling import CircuitBreaker, ErrorContext


class TestAniDBClient:
    """Test the AniDB XML client."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for AniDB client."""
        cache_manager = Mock(spec=CollaborativeCacheSystem)
        cache_manager.get = AsyncMock(return_value=None)

        circuit_breaker = Mock(spec=CircuitBreaker)
        circuit_breaker.is_open = Mock(return_value=False)

        rate_limiter = Mock()
        rate_limiter.acquire = AsyncMock()

        return {
            "circuit_breaker": circuit_breaker,
            "rate_limiter": rate_limiter,  # 1 req/sec
            "cache_manager": cache_manager,
            "error_handler": Mock(spec=ErrorContext),
        }

    @pytest.fixture
    def anidb_client(self, mock_dependencies):
        """Create AniDB client with mocked dependencies."""
        return AniDBClient(
            client_name="animemcp", client_version="1.0.0", **mock_dependencies
        )

    @pytest.fixture
    def anidb_client_with_auth(self, mock_dependencies):
        """Create AniDB client with authentication credentials."""
        return AniDBClient(
            client_name="animemcp",
            client_version="1.0.0",
            username="testuser",
            password="testpass",
            **mock_dependencies,
        )

    @pytest.fixture
    def sample_anime_xml_response(self):
        """Sample AniDB XML anime response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<anime id="1" restricted="false">
    <type>TV Series</type>
    <episodecount>26</episodecount>
    <startdate>1998-10-20</startdate>
    <enddate>1999-04-24</enddate>
    <titles>
        <title xml:lang="x-jat" type="main">Cowboy Bebop</title>
        <title xml:lang="en" type="official">Cowboy Bebop</title>
        <title xml:lang="ja" type="official">カウボーイビバップ</title>
        <title xml:lang="x-jat" type="synonym">CB</title>
    </titles>
    <creators>
        <name id="718" type="Direction">Watanabe Shinichiro</name>
        <name id="2186" type="Original Work">Hajime Yatate</name>
        <name id="5734" type="Series Composition">Nobumoto Keiko</name>
    </creators>
    <description>In the year 2071, humanity has colonized several of the planets and moons of the solar system leaving the now uninhabitable surface of planet Earth behind. The Inter Solar System Police attempts to keep peace in the galaxy, aided in part by outlaw bounty hunters, referred to as "Cowboys." The ragtag team aboard the spaceship "Bebop" are two such individuals.</description>
    <ratings>
        <permanent count="8875">8.81</permanent>
        <temporary count="8968">8.79</temporary>
        <review count="12">8.45</review>
    </ratings>
    <picture>233.jpg</picture>
    <resources>
        <resource type="1">
            <externalentity>
                <identifier>12345</identifier>
                <url>http://myanimelist.net/anime/12345</url>
            </externalentity>
        </resource>
        <resource type="2">
            <externalentity>
                <identifier>1</identifier>
                <url>http://anilist.co/anime/1</url>
            </externalentity>
        </resource>
    </resources>
    <tags>
        <tag id="2604" count="8" weight="600">
            <name>action</name>
            <description>Action anime usually involve a fairly straightforward story of good guys versus bad guys, where most disputes are resolved by using physical force.</description>
        </tag>
        <tag id="2607" count="7" weight="500">
            <name>space</name>
            <description>Space anime are set in outer space or involve space travel as a major plot element.</description>
        </tag>
        <tag id="2611" count="6" weight="400">
            <name>bounty hunter</name>
            <description>A bounty hunter is someone who captures fugitives for a monetary reward.</description>
        </tag>
    </tags>
    <episodes>
        <episode id="1" update="2008-01-01">
            <epno type="1">1</epno>
            <length>24</length>
            <airdate>1998-10-23</airdate>
            <rating votes="145">8.34</rating>
            <title xml:lang="en">Asteroid Blues</title>
            <summary>Spike and Jet pursue a dealer named Asimov Solensan who has stolen a case of blood-eye...</summary>
        </episode>
    </episodes>
</anime>"""

    @pytest.fixture
    def sample_auth_response(self):
        """Sample AniDB authentication response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<response>
    <session>S12345ABCDEF</session>
    <salt>randomsalt123</salt>
</response>"""

    @pytest.fixture
    def sample_episode_xml_response(self):
        """Sample AniDB episode XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<episode id="1" aid="1" update="2008-01-01">
    <epno type="1">1</epno>
    <length>24</length>
    <airdate>1998-10-23</airdate>
    <rating votes="145">8.34</rating>
    <title xml:lang="en">Asteroid Blues</title>
    <title xml:lang="ja">アステロイド・ブルース</title>
    <summary>Spike and Jet pursue a dealer named Asimov Solensan who has stolen a case of blood-eye, a dangerous stimulant. During a shootout, Asimov's girlfriend is fatally wounded and Asimov is killed by Spike.</summary>
    <resources>
        <resource type="1">
            <externalentity>
                <identifier>12345</identifier>
                <url>http://myanimelist.net/anime/12345/episode/1</url>
            </externalentity>
        </resource>
    </resources>
</episode>"""

    @pytest.fixture
    def sample_character_xml_response(self):
        """Sample AniDB character XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<characters>
    <character id="123" type="main character" update="2010-05-15">
        <name>Spike Spiegel</name>
        <gender>male</gender>
        <description>Spike Spiegel is a tall and lean 27-year-old bounty hunter born on Mars...</description>
        <picture>123.jpg</picture>
        <seiyuu id="456" picture="456.jpg">Yamadera Koichi</seiyuu>
    </character>
    <character id="124" type="main character" update="2010-05-15">
        <name>Jet Black</name>
        <gender>male</gender>
        <description>Jet Black is a 36-year-old former ISSP officer...</description>
        <picture>124.jpg</picture>
        <seiyuu id="457" picture="457.jpg">Ishizuka Unshou</seiyuu>
    </character>
</characters>"""

    def test_client_initialization_without_auth(self, mock_dependencies):
        """Test AniDB client initialization without authentication."""
        client = AniDBClient(
            client_name="animemcp", client_version="1.0.0", **mock_dependencies
        )

        assert client.circuit_breaker == mock_dependencies["circuit_breaker"]
        assert client.rate_limiter == mock_dependencies["rate_limiter"]
        assert client.cache_manager == mock_dependencies["cache_manager"]
        assert client.error_handler == mock_dependencies["error_handler"]
        assert client.base_url == "http://api.anidb.net:9001/httpapi"
        assert client.client_name == "animemcp"
        assert client.client_version == "1.0.0"
        assert client.username is None
        assert client.password is None
        assert client.session_key is None

    def test_client_initialization_with_auth(self, mock_dependencies):
        """Test AniDB client initialization with authentication."""
        username = "testuser"
        password = "testpass"

        client = AniDBClient(
            client_name="animemcp",
            client_version="1.0.0",
            username=username,
            password=password,
            **mock_dependencies,
        )

        assert client.username == username
        assert client.password == password
        assert client.session_key is None

    @pytest.mark.asyncio
    async def test_authenticate_success(
        self, anidb_client_with_auth, sample_auth_response
    ):
        """Test successful authentication."""
        with patch.object(
            anidb_client_with_auth, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_auth_response

            session_key = await anidb_client_with_auth.authenticate()

            assert session_key == "S12345ABCDEF"
            assert anidb_client_with_auth.session_key == "S12345ABCDEF"

            # Verify auth request was made
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[0][0]  # First positional argument
            assert params["request"] == "auth"
            assert params["user"] == "testuser"

    @pytest.mark.asyncio
    async def test_authenticate_without_credentials_fails(self, anidb_client):
        """Test authentication fails without credentials."""
        with pytest.raises(Exception) as exc_info:
            await anidb_client.authenticate()

        assert "credentials required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_anime_by_id_success(
        self, anidb_client, sample_anime_xml_response
    ):
        """Test successful anime retrieval by ID."""
        anime_id = 1

        with patch.object(
            anidb_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_anime_xml_response

            result = await anidb_client.get_anime_by_id(anime_id)

            assert result is not None
            assert result["id"] == "1"
            assert result["titles"]["main"] == "Cowboy Bebop"
            assert result["episodecount"] == "26"
            assert result["type"] == "TV Series"
            assert len(result["tags"]) == 3

            # Verify API was called with correct parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[0][0]  # First positional argument
            assert params["request"] == "anime"
            assert params["aid"] == anime_id

    @pytest.mark.asyncio
    async def test_get_anime_by_id_not_found(self, anidb_client):
        """Test anime retrieval with non-existent ID."""
        anime_id = 999999

        error_response = """<?xml version="1.0" encoding="UTF-8"?>
<error code="330">No such anime</error>"""

        with patch.object(
            anidb_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = error_response

            result = await anidb_client.get_anime_by_id(anime_id)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_episode_by_id_success(
        self, anidb_client, sample_episode_xml_response
    ):
        """Test successful episode retrieval by ID."""
        episode_id = 1

        with patch.object(
            anidb_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_episode_xml_response

            result = await anidb_client.get_episode_by_id(episode_id)

            assert result is not None
            assert result["id"] == "1"
            assert result["titles"]["en"] == "Asteroid Blues"
            assert result["epno"] == "1"
            assert result["length"] == "24"
            assert result["airdate"] == "1998-10-23"

            # Verify episode request was made
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[0][0]  # First positional argument
            assert params["request"] == "episode"
            assert params["eid"] == episode_id

    @pytest.mark.asyncio
    async def test_get_anime_characters_success(
        self, anidb_client, sample_character_xml_response
    ):
        """Test successful anime characters retrieval."""
        anime_id = 1

        with patch.object(
            anidb_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_character_xml_response

            characters = await anidb_client.get_anime_characters(anime_id)

            assert len(characters) == 2
            assert characters[0]["name"] == "Spike Spiegel"
            assert characters[0]["type"] == "main character"
            assert characters[0]["gender"] == "male"
            assert characters[1]["name"] == "Jet Black"
            assert characters[0]["seiyuu"]["name"] == "Yamadera Koichi"

            # Verify character request was made
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_by_name(self, anidb_client, sample_anime_xml_response):
        """Test anime search by name."""
        anime_name = "Cowboy Bebop"

        with patch.object(
            anidb_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_anime_xml_response

            result = await anidb_client.search_anime_by_name(anime_name)

            assert result is not None
            assert result["titles"]["main"] == "Cowboy Bebop"

            # Verify search request was made
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[0][0]  # First positional argument
            assert params["request"] == "anime"
            assert params["aname"] == anime_name

    @pytest.mark.asyncio
    async def test_xml_request_success(self, anidb_client):
        """Test successful XML API request."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = Mock()
            mock_response.text = AsyncMock(
                return_value='<?xml version="1.0"?><response>success</response>'
            )
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "text/xml"}
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            params = {"request": "anime", "aid": 1}
            result = await anidb_client._make_request(params)

            # Verify XML response was returned
            assert '<?xml version="1.0"?>' in result
            assert "<response>success</response>" in result

            # Verify client parameters were included
            call_args = mock_get.call_args
            request_params = call_args[1]["params"]
            assert request_params["client"] == "animemcp"
            assert request_params["clientver"] == "1.0.0"
            assert request_params["protover"] == "1"

    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self, anidb_client):
        """Test that rate limiting is enforced (1 req/sec)."""
        # AniDB has strict 1 req/sec rate limiting
        assert anidb_client.rate_limiter is not None

        # Mock the actual HTTP session to avoid real network calls
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = Mock()
            mock_response.text = AsyncMock(
                return_value='<?xml version="1.0"?><anime id="1"></anime>'
            )
            mock_response.status = 200
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Make multiple requests
            await anidb_client.get_anime_by_id(1)
            await anidb_client.get_anime_by_id(2)

            # Verify rate limiter was called for each request
            assert anidb_client.rate_limiter.acquire.call_count == 2

    @pytest.mark.asyncio
    async def test_error_handling_xml_parsing(self, anidb_client):
        """Test XML parsing error handling."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = Mock()
            mock_response.text = AsyncMock(return_value="invalid xml content")
            mock_response.status = 200
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(Exception) as exc_info:
                await anidb_client.get_anime_by_id(1)

            assert "xml parsing" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_anidb_error_codes(self, anidb_client):
        """Test AniDB-specific error code handling."""
        error_cases = [
            ("500", "Login failed"),
            ("501", "Login first"),
            ("505", "Illegal username or password"),
            ("330", "No such anime"),
            ("340", "No such episode"),
            ("350", "No such file"),
            ("555", "Banned"),
        ]

        for error_code, error_msg in error_cases:
            error_response = (
                f'<?xml version="1.0"?><error code="{error_code}">{error_msg}</error>'
            )

            with patch.object(
                anidb_client, "_make_request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = error_response

                result = await anidb_client.get_anime_by_id(1)

                assert result is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, anidb_client):
        """Test circuit breaker integration."""
        # Mock circuit breaker to simulate open state
        anidb_client.circuit_breaker.is_open = Mock(return_value=True)

        with pytest.raises(Exception) as exc_info:
            await anidb_client.get_anime_by_id(1)

        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cache_integration(self, anidb_client, sample_anime_xml_response):
        """Test cache integration."""
        anime_id = 1
        cache_key = f"anidb_anime_{anime_id}"

        # Mock parsed response for cache
        parsed_response = {
            "id": "1",
            "titles": {"main": "Cowboy Bebop"},
            "type": "TV Series",
            "episodecount": "26",
        }

        # Mock cache hit
        anidb_client.cache_manager.get = AsyncMock(return_value=parsed_response)

        result = await anidb_client.get_anime_by_id(anime_id)

        assert result["id"] == "1"
        assert result["titles"]["main"] == "Cowboy Bebop"
        anidb_client.cache_manager.get.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_session_management(self, anidb_client_with_auth):
        """Test session management and logout."""
        # Mock authentication
        with patch.object(
            anidb_client_with_auth, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            # First call: authentication
            auth_response = (
                '<?xml version="1.0"?><response><session>S12345</session></response>'
            )
            # Second call: logout
            logout_response = '<?xml version="1.0"?><response>Logged out</response>'

            mock_request.side_effect = [auth_response, logout_response]

            # Authenticate
            session_key = await anidb_client_with_auth.authenticate()
            assert session_key == "S12345"

            # Logout
            result = await anidb_client_with_auth.logout()
            assert result is True
            assert anidb_client_with_auth.session_key is None

            # Verify both calls were made
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, anidb_client):
        """Test connection timeout handling."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = asyncio.TimeoutError("Connection timeout")

            with pytest.raises(Exception) as exc_info:
                await anidb_client._make_request(
                    {"request": "anime", "aid": 1}, timeout=5
                )

            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_xml_namespace_handling(self, anidb_client):
        """Test XML namespace and encoding handling."""
        xml_with_namespace = """<?xml version="1.0" encoding="UTF-8"?>
<anime xmlns="http://api.anidb.net" id="1">
    <title xml:lang="en">Test Anime</title>
    <description>Test description with üñíçødé characters</description>
</anime>"""

        with patch.object(
            anidb_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = xml_with_namespace

            result = await anidb_client.get_anime_by_id(1)

            # Should handle XML namespaces and unicode properly
            assert result is not None

    @pytest.mark.asyncio
    async def test_client_registration_required(self, anidb_client):
        """Test that client registration is enforced."""
        # AniDB requires client registration
        assert anidb_client.client_name is not None
        assert anidb_client.client_version is not None

        # Client parameters should be included in all requests
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = Mock()
            mock_response.text = AsyncMock(
                return_value='<?xml version="1.0"?><response></response>'
            )
            mock_response.status = 200
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            await anidb_client._make_request({"request": "anime", "aid": 1})

            # Verify client parameters were included
            call_args = mock_get.call_args
            params = call_args[1]["params"]
            assert params["client"] == "animemcp"
            assert params["clientver"] == "1.0.0"
            assert params["protover"] == "1"
