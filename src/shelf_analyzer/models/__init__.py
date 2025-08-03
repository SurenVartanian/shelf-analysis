"""
Models package for shelf analysis.

This package contains all Pydantic models used throughout the application.
"""

from .base_models import (
    ObjectType,
    BoundingBox,
    DetectedObject,
    ShelfRegion,
    LLMAnalysis,
    AnalysisResult
)

from .vision_models import (
    VisionModel,
    VisionAnalysisConfig,
    Product,
    ShelfItem,
    ScoreItem,
    ScoresDict,
    EmptySpace,
    Banner,
    VisionAnalysisResult
)

__all__ = [
    # Base models
    "ObjectType",
    "BoundingBox", 
    "DetectedObject",
    "ShelfRegion",
    "LLMAnalysis",
    "AnalysisResult",
    
    # Vision models
    "VisionModel",
    "VisionAnalysisConfig",
    "Product",
    "ShelfItem",
    "ScoreItem",
    "ScoresDict",
    "EmptySpace",
    "Banner",
    "VisionAnalysisResult"
] 