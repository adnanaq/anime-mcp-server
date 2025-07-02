"""Universal schema mappers for anime data sources."""

from .anilist_mapper import AniListMapper
from .mal_mapper import MALMapper
from .jikan_mapper import JikanMapper
from .kitsu_mapper import KitsuMapper
from .anidb_mapper import AniDBMapper
from .animeplanet_mapper import AnimePlanetMapper
from .animeschedule_mapper import AnimeScheduleMapper
from .anisearch_mapper import AniSearchMapper

__all__ = [
    "AniListMapper",
    "MALMapper", 
    "JikanMapper",
    "KitsuMapper",
    "AniDBMapper",
    "AnimePlanetMapper",
    "AnimeScheduleMapper",
    "AniSearchMapper",
]