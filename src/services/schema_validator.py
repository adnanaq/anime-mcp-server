"""
Schema validation and compliance checking for AI enrichment results.
Ensures output matches enhanced_anime_schema_example.json structure exactly.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from pydantic import ValidationError

try:
    from ..models.anime import AnimeEntry, CharacterEntry, StatisticsEntry
except ImportError:
    from src.models.anime import AnimeEntry, CharacterEntry, StatisticsEntry

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates AI enrichment results against expected schema"""
    
    def __init__(self):
        """Initialize schema validator"""
        self.required_fields = self._get_required_fields()
        self.optional_fields = self._get_optional_fields()
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields that must be present"""
        return [
            # Core anime-offline-database fields (must preserve)
            "sources", "title", "type", "episodes", "status",
            
            # Enhanced fields (can be empty but must exist)
            "genres", "demographics", "themes", "statistics",
            "images", "characters", "streaming_info", "staff",
            "opening_themes", "ending_themes", "external_links",
            "relations", "awards", "enhanced_metadata"
        ]
    
    def _get_optional_fields(self) -> List[str]:
        """Get list of optional fields that may or may not be present"""
        return [
            "synopsis", "animeSeason", "picture", "thumbnail", 
            "duration", "score", "synonyms", "tags", "studios", 
            "producers", "source_material", "rating", "content_warnings",
            "aired_dates", "broadcast", "licensors", "streaming_licenses",
            "episode_details", "popularity_trends", "trailers",
            "enrichment_metadata"
        ]
    
    def validate_enriched_anime(self, enriched_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate enriched anime data against schema requirements.
        
        Args:
            enriched_data: The AI-enriched anime data
            
        Returns:
            Tuple of (is_valid, errors, validation_report)
        """
        errors = []
        validation_report = {
            "schema_compliance": False,
            "required_fields_present": 0,
            "total_required_fields": len(self.required_fields),
            "missing_fields": [],
            "invalid_fields": [],
            "statistics_validation": {},
            "characters_validation": {},
            "data_quality_score": 0.0
        }
        
        try:
            # 1. Check required fields presence
            missing_fields = []
            for field in self.required_fields:
                if field not in enriched_data:
                    missing_fields.append(field)
                    errors.append(f"Missing required field: {field}")
                else:
                    validation_report["required_fields_present"] += 1
            
            validation_report["missing_fields"] = missing_fields
            
            # 2. Validate Pydantic model compliance
            try:
                anime_entry = AnimeEntry(**enriched_data)
                validation_report["pydantic_validation"] = True
            except ValidationError as e:
                validation_report["pydantic_validation"] = False
                for error in e.errors():
                    field_path = " -> ".join(str(x) for x in error["loc"])
                    errors.append(f"Pydantic validation error in {field_path}: {error['msg']}")
                    validation_report["invalid_fields"].append(field_path)
            
            # 3. Validate statistics schema compliance
            stats_validation = self._validate_statistics(enriched_data.get("statistics", {}))
            validation_report["statistics_validation"] = stats_validation
            if not stats_validation["valid"]:
                errors.extend(stats_validation["errors"])
            
            # 4. Validate characters schema compliance  
            chars_validation = self._validate_characters(enriched_data.get("characters", []))
            validation_report["characters_validation"] = chars_validation
            if not chars_validation["valid"]:
                errors.extend(chars_validation["errors"])
            
            # 5. Calculate data quality score
            validation_report["data_quality_score"] = self._calculate_quality_score(enriched_data, validation_report)
            
            # Overall validation result
            is_valid = (
                len(missing_fields) == 0 and
                validation_report["pydantic_validation"] and
                stats_validation["valid"] and
                chars_validation["valid"]
            )
            validation_report["schema_compliance"] = is_valid
            
            return is_valid, errors, validation_report
            
        except Exception as e:
            errors.append(f"Validation process failed: {str(e)}")
            return False, errors, validation_report
    
    def _validate_statistics(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistics schema compliance"""
        validation = {
            "valid": True,
            "errors": [],
            "platforms_validated": 0,
            "standardized_fields": []
        }
        
        expected_fields = ["score", "scored_by", "rank", "popularity_rank", "members", "favorites"]
        
        for platform, stats in statistics.items():
            if not isinstance(stats, dict):
                validation["errors"].append(f"Statistics for {platform} must be a dictionary")
                validation["valid"] = False
                continue
            
            # Validate against StatisticsEntry model
            try:
                StatisticsEntry(**stats)
                validation["platforms_validated"] += 1
                
                # Check for standardized fields
                for field in expected_fields:
                    if field in stats and field not in validation["standardized_fields"]:
                        validation["standardized_fields"].append(field)
                        
            except ValidationError as e:
                validation["errors"].append(f"Statistics validation error for {platform}: {e}")
                validation["valid"] = False
        
        return validation
    
    def _validate_characters(self, characters: List[Any]) -> Dict[str, Any]:
        """Validate characters schema compliance"""
        validation = {
            "valid": True,
            "errors": [],
            "characters_validated": 0,
            "multi_source_characters": 0,
            "characters_with_images": 0
        }
        
        for i, char in enumerate(characters):
            if not isinstance(char, dict):
                validation["errors"].append(f"Character {i} must be a dictionary")
                validation["valid"] = False
                continue
            
            # Validate against CharacterEntry model
            try:
                CharacterEntry(**char)
                validation["characters_validated"] += 1
                
                # Check for multi-source data
                char_ids = char.get("character_ids", {})
                if len(char_ids) > 1:
                    validation["multi_source_characters"] += 1
                
                # Check for images
                images = char.get("images", {})
                if images:
                    validation["characters_with_images"] += 1
                    
            except ValidationError as e:
                validation["errors"].append(f"Character {i} validation error: {e}")
                validation["valid"] = False
        
        return validation
    
    def _calculate_quality_score(self, enriched_data: Dict[str, Any], validation_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)"""
        score = 0.0
        total_possible = 10.0
        
        # Schema compliance (3 points)
        if validation_report["schema_compliance"]:
            score += 3.0
        else:
            # Partial credit for required fields
            field_ratio = validation_report["required_fields_present"] / validation_report["total_required_fields"]
            score += 3.0 * field_ratio
        
        # Statistics quality (2 points)
        stats_validation = validation_report["statistics_validation"]
        if stats_validation.get("platforms_validated", 0) > 0:
            score += 2.0 * min(stats_validation["platforms_validated"] / 4, 1.0)  # Max 4 platforms
        
        # Characters quality (2 points)
        chars_validation = validation_report["characters_validation"]
        if chars_validation.get("characters_validated", 0) > 0:
            score += 1.0  # Base point for having characters
            if chars_validation.get("multi_source_characters", 0) > 0:
                score += 1.0  # Bonus for multi-source characters
        
        # Data richness (2 points)
        rich_fields = ["synopsis", "trailers", "opening_themes", "ending_themes", "awards"]
        present_rich_fields = sum(1 for field in rich_fields if enriched_data.get(field))
        score += 2.0 * (present_rich_fields / len(rich_fields))
        
        # Multi-platform integration (1 point)
        multi_platform_fields = ["statistics", "images", "characters"]
        multi_platform_score = 0
        for field in multi_platform_fields:
            field_data = enriched_data.get(field, {})
            if isinstance(field_data, dict) and len(field_data) > 1:
                multi_platform_score += 1
            elif isinstance(field_data, list) and field_data:
                # Check if characters have multi-source data
                if field == "characters":
                    multi_source_chars = sum(
                        1 for char in field_data 
                        if isinstance(char, dict) and len(char.get("character_ids", {})) > 1
                    )
                    if multi_source_chars > 0:
                        multi_platform_score += 1
        
        score += 1.0 * (multi_platform_score / len(multi_platform_fields))
        
        return min(score / total_possible, 1.0)
    
    def generate_validation_report(self, enriched_data: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        is_valid, errors, validation_report = self.validate_enriched_anime(enriched_data)
        
        report = f"""
🔍 ANIME ENRICHMENT VALIDATION REPORT
{'=' * 50}

📊 OVERALL RESULT: {'✅ VALID' if is_valid else '❌ INVALID'}
🎯 Data Quality Score: {validation_report['data_quality_score']:.2%}

📋 SCHEMA COMPLIANCE:
• Required fields: {validation_report['required_fields_present']}/{validation_report['total_required_fields']}
• Pydantic validation: {'✅' if validation_report.get('pydantic_validation') else '❌'}
• Missing fields: {validation_report['missing_fields'] or 'None'}

📈 STATISTICS VALIDATION:
• Platforms validated: {validation_report['statistics_validation'].get('platforms_validated', 0)}
• Standardized fields: {validation_report['statistics_validation'].get('standardized_fields', [])}

👥 CHARACTERS VALIDATION:  
• Characters validated: {validation_report['characters_validation'].get('characters_validated', 0)}
• Multi-source characters: {validation_report['characters_validation'].get('multi_source_characters', 0)}
• Characters with images: {validation_report['characters_validation'].get('characters_with_images', 0)}

❌ ERRORS ({len(errors)}):
{chr(10).join(f"• {error}" for error in errors) if errors else "• None"}

{'🎉 Schema validation passed!' if is_valid else '⚠️  Please fix errors above.'}
"""
        return report


# Convenience function
def validate_ai_enrichment_result(enriched_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Convenience function to validate AI enrichment result.
    
    Args:
        enriched_data: The AI-enriched anime data
        
    Returns:
        Tuple of (is_valid, validation_report_text)
    """
    validator = SchemaValidator()
    is_valid, errors, validation_report = validator.validate_enriched_anime(enriched_data)
    report_text = validator.generate_validation_report(enriched_data)
    
    return is_valid, report_text