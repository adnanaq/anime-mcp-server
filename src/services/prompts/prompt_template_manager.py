#!/usr/bin/env python3
"""
Prompt Template Manager for Modular AI Enrichment

Manages file-based prompt templates for the 4-stage enrichment pipeline.
Each stage has specialized prompts for optimal performance and token efficiency.
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages modular prompt templates for AI enrichment stages"""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize prompt template manager
        
        Args:
            prompts_dir: Path to prompts directory (auto-detected if None)
        """
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            # Auto-detect prompts directory
            current_file = Path(__file__)
            self.prompts_dir = current_file.parent
        
        self.base_dir = self.prompts_dir / "base"
        self.stages_dir = self.prompts_dir / "stages" 
        self.schemas_dir = self.prompts_dir / "schemas"
        
        logger.info(f"Initialized PromptTemplateManager with directory: {self.prompts_dir}")
        
        # Cache for loaded templates
        self._template_cache = {}
        
        # Stage configuration
        self.stage_config = {
            1: {
                "file": "01_metadata_extraction.txt",
                "description": "Extract metadata fields (synopsis, genres, basic info)",
                "token_reduction": "90%"
            },
            2: {
                "file": "02_episode_processing.txt", 
                "description": "Process episode details and timing",
                "token_reduction": "90%"
            },
            3: {
                "file": "03_relationship_analysis.txt",
                "description": "Analyze relatedAnime URLs with intelligent title extraction",
                "token_reduction": "95%"
            },
            4: {
                "file": "04_statistics_media.txt",
                "description": "Extract statistics and media fields",
                "token_reduction": "90%"
            },
            5: {
                "file": "05_character_processing.txt",
                "description": "Process character data and voice actors",
                "token_reduction": "85%"
            }
        }
    
    def load_template(self, template_name: str) -> Optional[str]:
        """Load a template from file with caching
        
        Args:
            template_name: Name of template file (e.g., "01_metadata_extraction.txt")
            
        Returns:
            Template content or None if not found
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Try different directories
        template_paths = [
            self.stages_dir / template_name,
            self.base_dir / template_name,
            self.schemas_dir / template_name
        ]
        
        for template_path in template_paths:
            if template_path.exists():
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self._template_cache[template_name] = content
                    logger.debug(f"✅ Loaded template: {template_name}")
                    return content
                except Exception as e:
                    logger.error(f"❌ Error loading template {template_name}: {e}")
                    return None
        
        logger.error(f"❌ Template not found: {template_name}")
        return None
    
    def build_stage_prompt(self, stage_num: int, **kwargs) -> str:
        """Build a complete prompt for a specific stage
        
        Args:
            stage_num: Stage number (1-5)
            **kwargs: Template variables to substitute
            
        Returns:
            Complete formatted prompt
        """
        if stage_num not in self.stage_config:
            raise ValueError(f"Invalid stage number: {stage_num}")
        
        stage_info = self.stage_config[stage_num]
        template_content = self.load_template(stage_info["file"])
        
        if not template_content:
            raise ValueError(f"Template not found for stage {stage_num}")
        
        # Substitute template variables
        try:
            formatted_prompt = template_content.format(**kwargs)
            logger.debug(f"✅ Built prompt for stage {stage_num}")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"❌ Missing template variable for stage {stage_num}: {e}")
            raise ValueError(f"Missing template variable: {e}")
    
    def get_stage_info(self, stage_num: int) -> Dict[str, Any]:
        """Get information about a specific stage
        
        Args:
            stage_num: Stage number (1-5)
            
        Returns:
            Stage configuration dictionary
        """
        return self.stage_config.get(stage_num, {})
    
    def validate_templates(self) -> Dict[str, bool]:
        """Validate that all required templates exist
        
        Returns:
            Dictionary mapping template names to validation status
        """
        validation_results = {}
        
        for stage_num, config in self.stage_config.items():
            template_name = config["file"]
            template_content = self.load_template(template_name)
            validation_results[template_name] = template_content is not None
            
            if template_content is None:
                logger.error(f"❌ Template validation failed: {template_name} - Template not found: {self.stages_dir / template_name}")
            else:
                logger.debug(f"✅ Template validation passed: {template_name}")
        
        return validation_results
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def get_all_stages(self) -> Dict[int, Dict[str, Any]]:
        """Get configuration for all stages"""
        return self.stage_config.copy()