"""
Source mapping configuration for AI enrichment agent.
Based on the _sources schema from enhanced_anime_schema_example.json
"""

from typing import Dict, List, Union

# Single-source fields with primary and fallback sources
SINGLE_SOURCE_MAPPING: Dict[str, Dict[str, str]] = {
    "genres": {"primary": "anilist", "fallback": "jikan"},
    "demographics": {"primary": "jikan", "fallback": "anilist"},
    "themes": {"primary": "anilist", "fallback": "jikan"},
    "source_material": {"primary": "jikan", "fallback": "anilist"},
    "rating": {"primary": "kitsu", "fallback": "jikan"},
    "content_warnings": {"primary": "jikan", "fallback": "anilist"},
    "aired_dates": {"primary": "jikan", "fallback": "anilist"},
    "broadcast": {"primary": "animeschedule", "fallback": "jikan"},
    "staff": {"primary": "jikan", "fallback": "anilist"},
    "opening_themes": {"primary": "jikan", "fallback": "anilist"},
    "ending_themes": {"primary": "jikan", "fallback": "anilist"},
    "episode_details": {"primary": "jikan", "fallback": "anilist"},
    "relations": {"primary": "jikan", "fallback": "anilist"},
    "awards": {"primary": "jikan", "fallback": "anilist"},
    "external_links": {"primary": "animeschedule", "fallback": "jikan"},
    "licensors": {"primary": "animeschedule", "fallback": "jikan"},
    "streaming_licenses": {"primary": "animeschedule", "fallback": "jikan"},
}

# Multi-source fields (fetch from all applicable sources)
MULTI_SOURCE_MAPPING: Dict[str, List[str]] = {
    "statistics": ["jikan", "anilist", "kitsu", "animeschedule"],
    "images": ["jikan", "anilist", "kitsu", "animeschedule"],
    "characters": ["jikan", "anilist"],  # Special character merging
    "streaming_info": ["animeschedule", "kitsu"],
    "popularity_trends": ["jikan", "anilist", "kitsu"],
}

# Fields that are already handled or don't need API calls
EXISTING_FIELDS = {
    # From anime-offline-database (keep unchanged)
    "sources": "anime-offline-database",
    "title": "anime-offline-database", 
    "type": "anime-offline-database",
    "episodes": "anime-offline-database",
    "status": "anime-offline-database",
    "animeSeason": "anime-offline-database",
    "picture": "anime-offline-database",
    "thumbnail": "anime-offline-database",
    "duration": "anime-offline-database",
    "score": "anime-offline-database",
    "synonyms": "anime-offline-database",
    "tags": "anime-offline-database",
    "studios": "anime-offline-database",
    "producers": "anime-offline-database",
    
    # Current enrichment fields (already enhanced)
    "synopsis": "multi-source",
    "trailers": "jikan",
    "enrichment_metadata": "system-generated",
    "enhanced_metadata": "system-generated",
}

def get_sources_for_field(field_name: str) -> Union[Dict[str, str], List[str], str, None]:
    """
    Get the source configuration for a specific field.
    
    Args:
        field_name: Name of the field to get sources for
        
    Returns:
        - Dict with primary/fallback for single-source fields
        - List of sources for multi-source fields  
        - String for existing fields
        - None if field not found
    """
    if field_name in SINGLE_SOURCE_MAPPING:
        return SINGLE_SOURCE_MAPPING[field_name]
    elif field_name in MULTI_SOURCE_MAPPING:
        return MULTI_SOURCE_MAPPING[field_name]
    elif field_name in EXISTING_FIELDS:
        return EXISTING_FIELDS[field_name]
    else:
        return None

def get_all_required_sources() -> List[str]:
    """Get list of all API sources needed for enrichment."""
    sources = set()
    
    # Add sources from single-source mapping
    for config in SINGLE_SOURCE_MAPPING.values():
        sources.add(config["primary"])
        sources.add(config["fallback"])
    
    # Add sources from multi-source mapping
    for source_list in MULTI_SOURCE_MAPPING.values():
        sources.update(source_list)
    
    # Remove non-API sources
    sources.discard("anime-offline-database")
    sources.discard("multi-source") 
    sources.discard("system-generated")
    
    return sorted(list(sources))

def should_fetch_from_source(field_name: str, source: str) -> bool:
    """
    Check if a specific source should be called for a field.
    
    Args:
        field_name: Name of the field
        source: API source name (jikan, anilist, kitsu, animeschedule)
        
    Returns:
        True if this source should be called for this field
    """
    field_sources = get_sources_for_field(field_name)
    
    if isinstance(field_sources, dict):  # Single-source field
        return source in [field_sources["primary"], field_sources["fallback"]]
    elif isinstance(field_sources, list):  # Multi-source field
        return source in field_sources
    else:
        return False

def get_fields_for_source(source: str) -> Dict[str, str]:
    """
    Get all fields that should be fetched from a specific source.
    
    Args:
        source: API source name
        
    Returns:
        Dict mapping field names to their fetch priority (primary/fallback/multi)
    """
    fields = {}
    
    # Check single-source fields
    for field, config in SINGLE_SOURCE_MAPPING.items():
        if config["primary"] == source:
            fields[field] = "primary"
        elif config["fallback"] == source:
            fields[field] = "fallback"
    
    # Check multi-source fields
    for field, sources in MULTI_SOURCE_MAPPING.items():
        if source in sources:
            fields[field] = "multi"
    
    return fields

# AI will handle intelligent mapping of platform-specific fields to standardized schema
# No hardcoded mappings needed - let AI do the smart conversion