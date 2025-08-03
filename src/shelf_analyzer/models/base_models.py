"""
Base models for shelf analysis.

General models used across different services.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class ObjectType(str, Enum):
    """Detected object types for retail shelf analysis"""
    BOTTLE = "bottle"          # Beverages, water, soda bottles
    CAN = "can"                # Beer, soda, canned goods
    BOX = "box"                # Cereal, crackers, larger packages
    PACKAGE = "package"        # Chips, candy, small snacks
    FRUIT = "fruit"            # Individual fruits (apples, bananas)
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class DetectedObject(BaseModel):
    """Single detected object"""
    object_type: ObjectType
    bounding_box: BoundingBox
    confidence: float
    label: Optional[str] = None


class ShelfRegion(BaseModel):
    """Empty shelf region"""
    bounding_box: BoundingBox
    shelf_level: Optional[int] = None
    area_percentage: float
    marked_color: str = "red"


class LLMAnalysis(BaseModel):
    """LLM-powered analysis results"""
    summary: str
    product_count: int
    empty_count: int
    recommendations: List[str]
    confidence: float


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    objects: List[DetectedObject]
    empty_spaces: List[ShelfRegion]
    llm_analysis: Optional[LLMAnalysis] = None
    image_info: Dict[str, Any]
    processing_time_ms: Optional[float] = None 