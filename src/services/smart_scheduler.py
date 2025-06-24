# src/services/smart_scheduler.py - Smart Update Scheduling
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import aiohttp

logger = logging.getLogger(__name__)


class SmartScheduler:
    """Smart scheduler that adapts to anime-offline-database release patterns"""

    def __init__(self):
        self.github_api_url = (
            "https://api.github.com/repos/manami-project/anime-offline-database"
        )
        self.releases_url = f"{self.github_api_url}/releases"
        self.commits_url = f"{self.github_api_url}/commits"

    async def check_recent_releases(self, hours_back: int = 24) -> Dict[str, Any]:
        """Check for releases in the last N hours"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.releases_url}?per_page=10") as response:
                    if response.status == 200:
                        releases = await response.json()

                        recent_releases = []
                        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

                        for release in releases:
                            published_at = datetime.fromisoformat(
                                release["published_at"].replace("Z", "+00:00")
                            ).replace(tzinfo=None)

                            if published_at > cutoff_time:
                                recent_releases.append(
                                    {
                                        "tag_name": release["tag_name"],
                                        "published_at": published_at.isoformat(),
                                        "hours_ago": (
                                            datetime.utcnow() - published_at
                                        ).total_seconds()
                                        / 3600,
                                    }
                                )

                        return {
                            "has_recent_releases": len(recent_releases) > 0,
                            "releases": recent_releases,
                            "latest_release_hours_ago": (
                                recent_releases[0]["hours_ago"]
                                if recent_releases
                                else None
                            ),
                        }
                    else:
                        logger.error(f"GitHub API error: {response.status}")
                        return {"has_recent_releases": False, "releases": []}

        except Exception as e:
            logger.error(f"Error checking releases: {e}")
            return {"has_recent_releases": False, "releases": []}

    async def check_commit_activity(self, hours_back: int = 6) -> Dict[str, Any]:
        """Check for recent commit activity that might indicate ongoing updates"""
        try:
            since = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat() + "Z"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.commits_url}?since={since}&per_page=20"
                ) as response:
                    if response.status == 200:
                        commits = await response.json()

                        recent_commits = []
                        for commit in commits:
                            commit_date = datetime.fromisoformat(
                                commit["commit"]["committer"]["date"].replace(
                                    "Z", "+00:00"
                                )
                            ).replace(tzinfo=None)

                            hours_ago = (
                                datetime.utcnow() - commit_date
                            ).total_seconds() / 3600

                            recent_commits.append(
                                {
                                    "sha": commit["sha"][:8],
                                    "message": commit["commit"]["message"],
                                    "date": commit_date.isoformat(),
                                    "hours_ago": hours_ago,
                                }
                            )

                        return {
                            "has_recent_commits": len(recent_commits) > 0,
                            "commits": recent_commits,
                            "latest_commit_hours_ago": (
                                recent_commits[0]["hours_ago"]
                                if recent_commits
                                else None
                            ),
                        }
                    else:
                        logger.error(f"GitHub API error: {response.status}")
                        return {"has_recent_commits": False, "commits": []}

        except Exception as e:
            logger.error(f"Error checking commits: {e}")
            return {"has_recent_commits": False, "commits": []}

    async def is_safe_to_update(
        self, min_hours_after_release: int = 2
    ) -> Dict[str, Any]:
        """Determine if it's safe to update based on recent activity"""
        logger.info("🔍 Checking if it's safe to update...")

        # Check recent releases
        release_info = await self.check_recent_releases(hours_back=48)

        # Check recent commits (might indicate ongoing work)
        commit_info = await self.check_commit_activity(hours_back=6)

        # Decision logic
        safe_to_update = True
        reasons = []

        if release_info["has_recent_releases"]:
            latest_release_hours = release_info["latest_release_hours_ago"]
            if latest_release_hours < min_hours_after_release:
                safe_to_update = False
                reasons.append(
                    f"Recent release only {latest_release_hours:.1f} hours ago (waiting {min_hours_after_release}h)"
                )
            else:
                reasons.append(
                    f"Release {latest_release_hours:.1f} hours ago - safe to update"
                )
        else:
            reasons.append("No recent releases detected")

        if commit_info["has_recent_commits"]:
            latest_commit_hours = commit_info["latest_commit_hours_ago"]
            if (
                latest_commit_hours < 1
            ):  # Very recent commits might indicate ongoing work
                safe_to_update = False
                reasons.append(
                    f"Very recent commit {latest_commit_hours:.1f} hours ago - might be ongoing work"
                )
            else:
                reasons.append(
                    f"Recent commit {latest_commit_hours:.1f} hours ago - seems stable"
                )

        result = {
            "safe_to_update": safe_to_update,
            "reasons": reasons,
            "release_info": release_info,
            "commit_info": commit_info,
            "recommendation": self._get_recommendation(
                safe_to_update, release_info, commit_info
            ),
        }

        logger.info(
            f"🎯 Update safety check: {'✅ SAFE' if safe_to_update else '⚠️ WAIT'}"
        )
        for reason in reasons:
            logger.info(f"   • {reason}")

        return result

    def _get_recommendation(self, safe: bool, releases: Dict, commits: Dict) -> str:
        """Generate human-readable recommendation"""
        if not safe:
            if releases["has_recent_releases"]:
                hours = releases["latest_release_hours_ago"]
                return f"Wait {2 - hours:.1f} more hours after the recent release"
            if commits["has_recent_commits"]:
                return "Wait for commit activity to settle (check again in 1-2 hours)"

        if releases["has_recent_releases"]:
            return "Safe to update - recent release has had time to stabilize"

        return "Safe to update - no recent activity detected"

    async def get_optimal_update_schedule(self) -> Dict[str, Any]:
        """Analyze release patterns and suggest optimal cron schedule"""
        try:
            # Get last 10 releases to analyze patterns
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.releases_url}?per_page=10") as response:
                    if response.status == 200:
                        releases = await response.json()

                        # Analyze release day patterns
                        day_counts = {
                            0: 0,
                            1: 0,
                            2: 0,
                            3: 0,
                            4: 0,
                            5: 0,
                            6: 0,
                        }  # Mon=0, Sun=6
                        hour_counts = {}

                        for release in releases:
                            published_at = datetime.fromisoformat(
                                release["published_at"].replace("Z", "+00:00")
                            )

                            day_of_week = published_at.weekday()
                            hour = published_at.hour

                            day_counts[day_of_week] += 1
                            hour_counts[hour] = hour_counts.get(hour, 0) + 1

                        # Find most common day and hour
                        most_common_day = max(day_counts, key=day_counts.get)
                        most_common_hour = (
                            max(hour_counts, key=hour_counts.get) if hour_counts else 14
                        )

                        # Suggest cron schedule: 4 hours after most common time, next day
                        suggested_hour = (most_common_hour + 4) % 24
                        suggested_day = (
                            most_common_day + 1
                        ) % 7  # Day after most common

                        day_names = [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday",
                        ]

                        return {
                            "analysis": {
                                "most_common_release_day": day_names[most_common_day],
                                "most_common_release_hour": most_common_hour,
                                "day_distribution": {
                                    day_names[i]: count
                                    for i, count in day_counts.items()
                                },
                                "hour_distribution": hour_counts,
                            },
                            "recommendation": {
                                "cron_schedule": f"0 {suggested_hour} * * {suggested_day}",
                                "description": f"Every {day_names[suggested_day]} at {suggested_hour:02d}:00 (4 hours after typical {day_names[most_common_day]} {most_common_hour:02d}:00 releases)",
                                "rationale": f"Based on analysis, releases typically happen on {day_names[most_common_day]} around {most_common_hour:02d}:00. Waiting until {day_names[suggested_day]} {suggested_hour:02d}:00 provides 4+ hour buffer.",
                            },
                        }
                    else:
                        return {"error": f"GitHub API error: {response.status}"}

        except Exception as e:
            logger.error(f"Error analyzing schedule: {e}")
            return {"error": str(e)}
