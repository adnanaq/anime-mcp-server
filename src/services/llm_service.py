"""
LLM service for query analysis and content generation.

Provides centralized LLM integration for the anime MCP server,
handling query analysis, content generation, and structured outputs.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """
    Centralized LLM service for query analysis and content generation.
    
    Supports multiple LLM providers (OpenAI, Anthropic) with fallback strategies
    and structured output generation.
    """
    
    def __init__(self):
        """Initialize LLM service with available providers."""
        self.primary_llm = None
        self.fallback_llm = None
        
        # Initialize OpenAI if available
        if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            try:
                self.primary_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI LLM initialized as primary")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}")
        
        # Initialize Anthropic if available
        if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
            try:
                anthropic_llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0.1,
                    api_key=settings.anthropic_api_key
                )
                
                if not self.primary_llm:
                    self.primary_llm = anthropic_llm
                    logger.info("Anthropic LLM initialized as primary")
                else:
                    self.fallback_llm = anthropic_llm
                    logger.info("Anthropic LLM initialized as fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic LLM: {e}")
        
        if not self.primary_llm:
            logger.warning("No LLM providers available - some features may be limited")
    
    async def analyze_query(
        self,
        query: str,
        context: Optional[str] = None,
        output_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user query to extract intent and parameters.
        
        Args:
            query: User query string
            context: Optional context for analysis
            output_model: Optional Pydantic model for structured output
            
        Returns:
            Dictionary with analysis results
        """
        if not self.primary_llm:
            logger.warning("No LLM available for query analysis")
            return {
                "intent_type": "search",
                "confidence": 0.5,
                "parameters": {"query": query},
                "reasoning": "No LLM available, using fallback analysis"
            }
        
        try:
            # Build analysis prompt
            system_prompt = self._build_analysis_prompt(context)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this query: {query}")
            ]
            
            # Get LLM response
            if output_model:
                # Use structured output if model provided
                structured_llm = self.primary_llm.with_structured_output(output_model)
                response = await structured_llm.ainvoke(messages)
                return response.dict() if hasattr(response, 'dict') else response
            else:
                # Use regular text response
                response = await self.primary_llm.ainvoke(messages)
                return self._parse_analysis_response(response.content)
                
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            
            # Try fallback LLM
            if self.fallback_llm:
                try:
                    messages = [
                        SystemMessage(content=self._build_analysis_prompt(context)),
                        HumanMessage(content=f"Analyze this query: {query}")
                    ]
                    response = await self.fallback_llm.ainvoke(messages)
                    return self._parse_analysis_response(response.content)
                except Exception as fallback_error:
                    logger.error(f"Fallback analysis failed: {fallback_error}")
            
            # Return basic fallback analysis
            return {
                "intent_type": "search",
                "confidence": 0.3,
                "parameters": {"query": query},
                "reasoning": f"LLM analysis failed: {str(e)}"
            }
    
    def _build_analysis_prompt(self, context: Optional[str] = None) -> str:
        """Build system prompt for query analysis."""
        base_prompt = """
        You are an expert anime query analyzer. Analyze user queries to extract:
        
        1. Intent type (search, discover, recommend, schedule, compare, etc.)
        2. Confidence level (0-1)
        3. Extracted parameters (titles, genres, years, etc.)
        4. Reasoning for your analysis
        
        Focus on understanding what the user wants to find or discover about anime.
        """
        
        if context:
            base_prompt += f"\n\nContext: {context}"
        
        return base_prompt
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        # Simple parsing - in production, this would be more sophisticated
        lines = response.strip().split('\n')
        
        result = {
            "intent_type": "search",
            "confidence": 0.7,
            "parameters": {},
            "reasoning": response
        }
        
        # Extract intent type
        for line in lines:
            if "intent:" in line.lower():
                intent = line.split(":", 1)[1].strip().lower()
                if intent in ["search", "discover", "recommend", "schedule", "compare"]:
                    result["intent_type"] = intent
                    break
        
        # Extract confidence
        for line in lines:
            if "confidence:" in line.lower():
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    result["confidence"] = max(0, min(1, confidence))
                except:
                    pass
                break
        
        return result
    
    async def generate_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 500
    ) -> str:
        """
        Generate content using LLM.
        
        Args:
            prompt: Content generation prompt
            context: Optional context
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated content string
        """
        if not self.primary_llm:
            return f"Content generation not available (no LLM configured)"
        
        try:
            messages = [
                SystemMessage(content=context or "You are a helpful anime assistant."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.primary_llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            
            # Try fallback
            if self.fallback_llm:
                try:
                    messages = [
                        SystemMessage(content=context or "You are a helpful anime assistant."),
                        HumanMessage(content=prompt)
                    ]
                    response = await self.fallback_llm.ainvoke(messages)
                    return response.content
                except Exception as fallback_error:
                    logger.error(f"Fallback content generation failed: {fallback_error}")
            
            return f"Content generation failed: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.primary_llm is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "primary_llm": "available" if self.primary_llm else "unavailable",
            "fallback_llm": "available" if self.fallback_llm else "unavailable",
            "service_available": self.is_available(),
            "timestamp": datetime.now().isoformat()
        }


# Global instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service