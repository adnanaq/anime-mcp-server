"""Comprehensive tests for SmartScheduler with 100% coverage."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from src.services.smart_scheduler import SmartScheduler


@pytest.fixture
def scheduler():
    """Create SmartScheduler instance."""
    return SmartScheduler()


@pytest.fixture
def sample_releases():
    """Sample GitHub releases data."""
    now = datetime.utcnow()
    return [
        {
            "tag_name": "v1.2.3",
            "published_at": (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "body": "Weekly update with new anime entries",
        },
        {
            "tag_name": "v1.2.2",
            "published_at": (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "body": "Previous weekly update",
        },
        {
            "tag_name": "v1.2.1",
            "published_at": (now - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "body": "Older update",
        },
    ]


@pytest.fixture
def sample_commits():
    """Sample GitHub commits data."""
    now = datetime.utcnow()
    return [
        {
            "sha": "abc123",
            "commit": {
                "author": {
                    "date": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                },
                "message": "Update anime database",
            },
        },
        {
            "sha": "def456",
            "commit": {
                "author": {
                    "date": (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
                },
                "message": "Fix data formatting",
            },
        },
    ]


class TestSmartSchedulerInitialization:
    """Test SmartScheduler initialization."""

    def test_init(self):
        """Test SmartScheduler initialization."""
        scheduler = SmartScheduler()

        assert (
            scheduler.github_api_url
            == "https://api.github.com/repos/manami-project/anime-offline-database"
        )
        assert (
            scheduler.releases_url
            == "https://api.github.com/repos/manami-project/anime-offline-database/releases"
        )
        assert (
            scheduler.commits_url
            == "https://api.github.com/repos/manami-project/anime-offline-database/commits"
        )


class TestRecentReleaseChecking:
    """Test recent release checking functionality."""

    @pytest.mark.asyncio
    async def test_check_recent_releases_with_recent(self, scheduler, sample_releases):
        """Test checking recent releases when recent releases exist."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_releases)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_recent_releases(hours_back=24)

            assert result["has_recent_releases"] is True
            assert len(result["recent_releases"]) >= 1
            assert "tag_name" in result["recent_releases"][0]
            assert "hours_ago" in result["recent_releases"][0]

    @pytest.mark.asyncio
    async def test_check_recent_releases_no_recent(self, scheduler):
        """Test checking recent releases when no recent releases exist."""
        # Create old releases (older than 24 hours)
        now = datetime.utcnow()
        old_releases = [
            {
                "tag_name": "v1.0.0",
                "published_at": (now - timedelta(days=2)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "body": "Old release",
            }
        ]

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=old_releases)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_recent_releases(hours_back=24)

            assert result["has_recent_releases"] is False
            assert len(result["recent_releases"]) == 0

    @pytest.mark.asyncio
    async def test_check_recent_releases_http_error(self, scheduler):
        """Test checking recent releases with HTTP error."""
        mock_response = Mock()
        mock_response.status = 404

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_recent_releases()

            assert result["has_recent_releases"] is False
            assert result["recent_releases"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_check_recent_releases_network_error(self, scheduler):
        """Test checking recent releases with network error."""
        with patch(
            "aiohttp.ClientSession.get",
            side_effect=aiohttp.ClientError("Network error"),
        ):
            result = await scheduler.check_recent_releases()

            assert result["has_recent_releases"] is False
            assert result["recent_releases"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_check_recent_releases_json_error(self, scheduler):
        """Test checking recent releases with JSON parsing error."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_recent_releases()

            assert result["has_recent_releases"] is False
            assert "error" in result


class TestCommitActivityChecking:
    """Test commit activity checking functionality."""

    @pytest.mark.asyncio
    async def test_check_commit_activity_success(self, scheduler, sample_commits):
        """Test checking commit activity successfully."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_commits)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_commit_activity(hours_back=48)

            assert "recent_commits" in result
            assert "commit_frequency" in result
            assert len(result["recent_commits"]) >= 1

    @pytest.mark.asyncio
    async def test_check_commit_activity_no_recent_commits(self, scheduler):
        """Test checking commit activity when no recent commits."""
        # Create old commits
        now = datetime.utcnow()
        old_commits = [
            {
                "sha": "old123",
                "commit": {
                    "author": {
                        "date": (now - timedelta(days=10)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    },
                    "message": "Old commit",
                },
            }
        ]

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=old_commits)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_commit_activity(hours_back=24)

            assert len(result["recent_commits"]) == 0
            assert result["commit_frequency"] == 0

    @pytest.mark.asyncio
    async def test_check_commit_activity_error(self, scheduler):
        """Test checking commit activity with error."""
        with patch("aiohttp.ClientSession.get", side_effect=Exception("API error")):
            result = await scheduler.check_commit_activity()

            assert "error" in result
            assert result["recent_commits"] == []


class TestOptimalScheduleAnalysis:
    """Test optimal schedule analysis functionality."""

    @pytest.mark.asyncio
    async def test_get_optimal_update_schedule_high_activity(self, scheduler):
        """Test optimal schedule calculation with high activity."""
        recent_releases = {
            "has_recent_releases": True,
            "recent_releases": [{"hours_ago": 2}],
        }
        commit_activity = {
            "recent_commits": [{"sha": "abc"}, {"sha": "def"}],
            "commit_frequency": 2.5,
        }

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler, "check_commit_activity", return_value=commit_activity
            ),
        ):
            result = await scheduler.get_optimal_update_schedule()

            assert "recommended_frequency" in result
            assert "optimal_time" in result
            assert "reasoning" in result
            assert result["recommended_frequency"] in ["daily", "bi-daily", "weekly"]

    @pytest.mark.asyncio
    async def test_get_optimal_update_schedule_low_activity(self, scheduler):
        """Test optimal schedule calculation with low activity."""
        recent_releases = {"has_recent_releases": False, "recent_releases": []}
        commit_activity = {"recent_commits": [], "commit_frequency": 0.1}

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler, "check_commit_activity", return_value=commit_activity
            ),
        ):
            result = await scheduler.get_optimal_update_schedule()

            assert result["recommended_frequency"] == "weekly"
            assert "low activity" in result["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_get_optimal_update_schedule_medium_activity(self, scheduler):
        """Test optimal schedule calculation with medium activity."""
        recent_releases = {"has_recent_releases": False, "recent_releases": []}
        commit_activity = {"recent_commits": [{"sha": "abc"}], "commit_frequency": 1.5}

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler, "check_commit_activity", return_value=commit_activity
            ),
        ):
            result = await scheduler.get_optimal_update_schedule()

            assert result["recommended_frequency"] in ["bi-weekly", "weekly"]

    @pytest.mark.asyncio
    async def test_get_optimal_update_schedule_error_handling(self, scheduler):
        """Test optimal schedule calculation with errors."""
        with (
            patch.object(
                scheduler, "check_recent_releases", side_effect=Exception("Error")
            ),
            patch.object(
                scheduler,
                "check_commit_activity",
                return_value={"recent_commits": [], "commit_frequency": 0},
            ),
        ):
            result = await scheduler.get_optimal_update_schedule()

            # Should still return a reasonable default
            assert "recommended_frequency" in result
            assert result["recommended_frequency"] == "weekly"


class TestUpdateSafetyChecking:
    """Test update safety checking functionality."""

    @pytest.mark.asyncio
    async def test_is_safe_to_update_safe(self, scheduler):
        """Test safety check when it's safe to update."""
        recent_releases = {"has_recent_releases": False, "recent_releases": []}

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler,
                "check_commit_activity",
                return_value={"has_recent_commits": False, "recent_commits": []},
            ),
        ):
            result = await scheduler.is_safe_to_update(min_hours_after_release=2)

            assert result["safe_to_update"] is True
            assert result["risk_level"] == "low"
            assert len(result["reasons"]) == 1
            assert "No recent releases detected" in result["reasons"][0]

    @pytest.mark.asyncio
    async def test_is_safe_to_update_recent_release(self, scheduler):
        """Test safety check when there's a recent release."""
        recent_releases = {
            "has_recent_releases": True,
            "recent_releases": [{"tag_name": "v1.2.3", "hours_ago": 1}],
            "latest_release_hours_ago": 1,
        }

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler,
                "check_commit_activity",
                return_value={"has_recent_commits": False, "recent_commits": []},
            ),
        ):
            result = await scheduler.is_safe_to_update(min_hours_after_release=2)

            assert result["safe_to_update"] is False
            assert result["risk_level"] == "medium"
            assert len(result["reasons"]) > 0
            assert "recent release detected" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_is_safe_to_update_old_enough_release(self, scheduler):
        """Test safety check when recent release is old enough."""
        recent_releases = {
            "has_recent_releases": True,
            "recent_releases": [{"tag_name": "v1.2.3", "hours_ago": 5}],
            "latest_release_hours_ago": 5,
        }

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler,
                "check_commit_activity",
                return_value={"has_recent_commits": False, "recent_commits": []},
            ),
        ):
            result = await scheduler.is_safe_to_update(min_hours_after_release=2)

            assert result["safe_to_update"] is True
            assert result["risk_level"] == "low"

    @pytest.mark.asyncio
    async def test_is_safe_to_update_multiple_recent_releases(self, scheduler):
        """Test safety check with multiple recent releases."""
        recent_releases = {
            "has_recent_releases": True,
            "recent_releases": [
                {"tag_name": "v1.2.3", "hours_ago": 1},
                {"tag_name": "v1.2.2", "hours_ago": 0.5},
            ],
            "latest_release_hours_ago": 0.5,
        }

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler,
                "check_commit_activity",
                return_value={"has_recent_commits": False, "recent_commits": []},
            ),
        ):
            result = await scheduler.is_safe_to_update(min_hours_after_release=2)

            assert result["safe_to_update"] is False
            assert result["risk_level"] == "high"
            assert "multiple recent releases" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_is_safe_to_update_error_handling(self, scheduler):
        """Test safety check error handling."""
        with patch.object(
            scheduler, "check_recent_releases", side_effect=Exception("API error")
        ):
            result = await scheduler.is_safe_to_update()

            # Should default to safe when unable to check
            assert result["safe_to_update"] is True
            assert "warning" in result


class TestReleasePatternAnalysis:
    """Test release pattern analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_release_patterns_regular(self, scheduler):
        """Test analyzing regular release patterns."""
        # Create releases with regular weekly pattern
        now = datetime.utcnow()
        regular_releases = []
        for i in range(4):
            regular_releases.append(
                {
                    "tag_name": f"v1.{i}.0",
                    "published_at": (now - timedelta(weeks=i)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "body": f"Weekly update {i}",
                }
            )

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=regular_releases)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.analyze_release_patterns()

            assert "pattern_detected" in result
            assert "average_interval_days" in result
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_analyze_release_patterns_irregular(self, scheduler):
        """Test analyzing irregular release patterns."""
        # Create releases with irregular pattern
        now = datetime.utcnow()
        irregular_releases = [
            {
                "tag_name": "v1.0.0",
                "published_at": (now - timedelta(days=1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "body": "Release 1",
            },
            {
                "tag_name": "v0.9.0",
                "published_at": (now - timedelta(days=15)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "body": "Release 2",
            },
            {
                "tag_name": "v0.8.0",
                "published_at": (now - timedelta(days=45)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "body": "Release 3",
            },
        ]

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=irregular_releases)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.analyze_release_patterns()

            assert result["pattern_detected"] in ["irregular", "semi-regular"]
            assert result["confidence"] in ["low", "medium"]

    @pytest.mark.asyncio
    async def test_analyze_release_patterns_insufficient_data(self, scheduler):
        """Test analyzing release patterns with insufficient data."""
        few_releases = [
            {
                "tag_name": "v1.0.0",
                "published_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "body": "Only release",
            }
        ]

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=few_releases)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.analyze_release_patterns()

            assert result["pattern_detected"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_analyze_release_patterns_error(self, scheduler):
        """Test analyzing release patterns with error."""
        with patch("aiohttp.ClientSession.get", side_effect=Exception("API error")):
            result = await scheduler.analyze_release_patterns()

            assert "error" in result


class TestTimeCalculations:
    """Test time calculation utilities."""

    def test_calculate_hours_since_release(self, scheduler):
        """Test calculating hours since release."""
        now = datetime.utcnow()
        release_time = now - timedelta(hours=5, minutes=30)

        release_data = {"published_at": release_time.strftime("%Y-%m-%dT%H:%M:%SZ")}

        hours = scheduler._calculate_hours_since_release(release_data)

        # Should be approximately 5.5 hours
        assert 5.0 <= hours <= 6.0

    def test_calculate_hours_since_release_invalid_format(self, scheduler):
        """Test calculating hours with invalid date format."""
        release_data = {"published_at": "invalid-date-format"}

        hours = scheduler._calculate_hours_since_release(release_data)

        # Should return a large number for invalid dates
        assert hours > 24

    def test_parse_iso_timestamp(self, scheduler):
        """Test parsing ISO timestamp."""
        timestamp_str = "2024-01-15T10:30:00Z"

        result = scheduler._parse_iso_timestamp(timestamp_str)

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_iso_timestamp_with_microseconds(self, scheduler):
        """Test parsing ISO timestamp with microseconds."""
        timestamp_str = "2024-01-15T10:30:00.123456Z"

        result = scheduler._parse_iso_timestamp(timestamp_str)

        assert result.year == 2024
        assert result.microsecond == 123456

    def test_parse_iso_timestamp_invalid(self, scheduler):
        """Test parsing invalid ISO timestamp."""
        invalid_timestamp = "not-a-timestamp"

        result = scheduler._parse_iso_timestamp(invalid_timestamp)

        # Should return a very old date for invalid timestamps
        assert result.year < 2020


class TestSmartSchedulerMissingCoverage:
    """Test missing coverage lines to reach 100%."""

    @pytest.mark.asyncio
    async def test_check_recent_releases_github_api_error(self, scheduler):
        """Test GitHub API error handling - covers lines 113-114."""
        mock_response = Mock()
        mock_response.status = 403
        mock_response.json = AsyncMock(side_effect=Exception("Rate limit exceeded"))

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_recent_releases()

            # Should handle GitHub API error gracefully
            assert result["has_recent_releases"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_check_commit_activity_complex_recent_analysis(self, scheduler):
        """Test complex recent commit analysis - covers lines 160-169."""
        now = datetime.utcnow()

        # Create commits with complex timing patterns to trigger analysis logic
        complex_commits = [
            {
                "sha": "recent1",
                "commit": {
                    "author": {
                        "date": (now - timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    },
                    "message": "Recent commit 1",
                },
            },
            {
                "sha": "recent2",
                "commit": {
                    "author": {
                        "date": (now - timedelta(hours=3)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    },
                    "message": "Recent commit 2",
                },
            },
            {
                "sha": "recent3",
                "commit": {
                    "author": {
                        "date": (now - timedelta(hours=12)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    },
                    "message": "Recent commit 3",
                },
            },
            {
                "sha": "older",
                "commit": {
                    "author": {
                        "date": (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    },
                    "message": "Older commit",
                },
            },
        ]

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=complex_commits)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scheduler.check_commit_activity(hours_back=24)

            # Should analyze recent commits and calculate frequency
            assert "recent_commits" in result
            assert "commit_frequency" in result
            assert len(result["recent_commits"]) == 3  # Only commits within 24 hours
            assert result["commit_frequency"] > 0

    @pytest.mark.asyncio
    async def test_is_safe_to_update_conditional_paths(self, scheduler):
        """Test conditional update safety paths - covers lines 207-208."""
        # Test with recent commits but old releases
        with (
            patch.object(
                scheduler,
                "check_recent_releases",
                return_value={"has_recent_releases": False, "recent_releases": []},
            ),
            patch.object(
                scheduler,
                "check_commit_activity",
                return_value={
                    "has_recent_commits": True,
                    "recent_commits": [{"sha": "abc"}],
                },
            ),
        ):
            result = await scheduler.is_safe_to_update()

            # Should evaluate commit activity when no recent releases
            assert "safe_to_update" in result
            assert "risk_level" in result

    @pytest.mark.asyncio
    async def test_get_optimal_update_schedule_comprehensive_logic(self, scheduler):
        """Test comprehensive optimal schedule logic - covers lines 217-285."""
        # Test with specific conditions that trigger different scheduling paths
        recent_releases = {
            "has_recent_releases": True,
            "recent_releases": [{"hours_ago": 12, "tag_name": "v1.2.3"}],
        }

        commit_activity = {
            "recent_commits": [
                {"sha": "abc", "hours_ago": 6},
                {"sha": "def", "hours_ago": 18},
            ],
            "commit_frequency": 3.2,  # High frequency
        }

        with (
            patch.object(
                scheduler, "check_recent_releases", return_value=recent_releases
            ),
            patch.object(
                scheduler, "check_commit_activity", return_value=commit_activity
            ),
            patch.object(
                scheduler,
                "analyze_release_patterns",
                return_value={
                    "pattern_detected": "regular",
                    "average_interval_days": 7,
                    "confidence": "high",
                },
            ),
        ):
            result = await scheduler.get_optimal_update_schedule()

            # Should analyze all factors and provide comprehensive recommendation
            assert "recommended_frequency" in result
            assert "optimal_time" in result
            assert "reasoning" in result
            assert "cron_schedule" in result
            assert "analysis" in result

            # Should have detailed analysis
            analysis = result["analysis"]
            assert "recent_activity" in analysis
            assert "release_patterns" in analysis
            assert "recommended_schedule" in analysis
