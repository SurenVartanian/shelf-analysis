"""
Base Detection Strategy Interface

Defines the contract for all shelf detection strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import logging

from ...models.base_models import ShelfRegion

logger = logging.getLogger(__name__)


class BaseDetectionStrategy(ABC):
    """Abstract base class for shelf detection strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def detect_shelf_regions(
        self, 
        image: np.ndarray, 
        detected_objects: List = None
    ) -> Tuple[List[ShelfRegion], str]:
        """
        Detect shelf regions in the image
        
        Args:
            image: OpenCV image array
            detected_objects: Optional list of detected objects from YOLO
            
        Returns:
            Tuple of (list of shelf regions, detection method name)
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Get the name of this detection strategy"""
        return self.name
    
    def log_detection_result(self, regions: List[ShelfRegion], method: str):
        """Log detection results"""
        self.logger.info(f"Strategy '{self.name}' detected {len(regions)} regions using {method}")
        for i, region in enumerate(regions):
            self.logger.debug(f"  Region {i+1}: {region.x1},{region.y1} to {region.x2},{region.y2}") 