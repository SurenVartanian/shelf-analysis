"""
Vision models for shelf analysis.

Models specific to vision analysis and LLM interactions.
"""

from typing import List, Optional
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field


class VisionModel(str, Enum):
    """Supported vision models via LiteLLM"""
    # Google Cloud Vertex AI Models
    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    GEMINI_FLASH_LITE = "gemini-flash-lite"
    
    # Anthropic Models (via Google Cloud)
    CLAUDE_4_SONNET = "claude-4-sonnet"
    
    # OpenAI Models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"


@dataclass
class VisionAnalysisConfig:
    """Configuration for vision analysis"""
    model: VisionModel
    max_tokens: int = 16000  # Reduced for GPT-4O compatibility
    temperature: float = 0.1
    timeout: int = 60
    retry_attempts: int = 3


class Product(BaseModel):
    """Individual product on shelf"""
    name: str = Field(description="Short product name (if readable)")
    full_name: Optional[str] = Field(description="Full product name with brand (if readable)")
    count: int = Field(description="Approximate quantity based on visible packages")


class ShelfItem(BaseModel):
    """Individual shelf item"""
    shelf_position: int = Field(description="Shelf position (top-down: 1 = top)")
    shelf_visibility: int = Field(description="Shelf visibility score (0-100)")
    products: List[Product] = Field(description="List of products on the shelf")


class ScoreItem(BaseModel):
    """Individual score item"""
    value: int = Field(description="Score value (0-100)")
    comment: str = Field(description="Short explanation of the score")


class ScoresDict(BaseModel):
    """Scores dictionary"""
    banner_visibility: ScoreItem = Field(description="Banner visibility score")
    product_filling: ScoreItem = Field(description="Product filling score")
    promo_match: ScoreItem = Field(description="Promotional material matching score")
    product_neatness: ScoreItem = Field(description="Product neatness score")
    display_cleanliness: ScoreItem = Field(description="Display cleanliness score")
    shelf_arrangement: ScoreItem = Field(description="Shelf arrangement quality score")
    overall_score: ScoreItem = Field(description="Overall score")


class EmptySpace(BaseModel):
    """Empty space detection"""
    shelf_level: int = Field(description="Shelf level where empty space is located")
    area_percentage: float = Field(description="Percentage of shelf area that is empty (0-100)")
    confidence: float = Field(description="Confidence level of empty space detection (0-100)")


class Banner(BaseModel):
    """Promotional banner detection"""
    text: str = Field(description="Banner text content (in Spanish)")
    position: str = Field(description="Banner position on display")
    visibility: int = Field(description="Banner visibility score (0-100)")
    confidence: float = Field(description="Confidence level of banner detection (0-100)")


class VisionAnalysisResult(BaseModel):
    """Complete vision analysis result - LLM should return exactly this format"""
    model_used: str = Field(description="Model used for analysis")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    is_display_detected: bool = Field(description="Whether a store display is detected")
    banners: List[Banner] = Field(description="List of promotional banners detected")
    shelves: List[ShelfItem] = Field(description="List of shelves with products")
    scores: ScoresDict = Field(description="Analysis scores with value and comment")
    total_score: int = Field(description="Overall score (0-100)")
    general_comment: str = Field(description="General analysis comment")
    raw_response: Optional[str] = None 