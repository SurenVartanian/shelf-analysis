"""
Image Cropper Utility

Handles image cropping operations for shelf regions.
"""

import cv2
import numpy as np
from typing import Optional
import logging

from ...models.base_models import ShelfRegion

logger = logging.getLogger(__name__)


class ImageCropper:
    """Utility class for cropping images based on shelf regions"""
    
    @staticmethod
    def crop_shelf_region(image: np.ndarray, shelf_region: ShelfRegion) -> Optional[np.ndarray]:
        """
        Crop image to a specific shelf region
        
        Args:
            image: OpenCV image array
            shelf_region: Shelf region to crop to
            
        Returns:
            Cropped image array or None if cropping fails
        """
        try:
            height, width = image.shape[:2]
            x1 = max(0, int(shelf_region.x1))
            y1 = max(0, int(shelf_region.y1))
            x2 = min(width, int(shelf_region.x2))
            y2 = min(height, int(shelf_region.y2))
            
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid crop region: {x1},{y1} to {x2},{y2}")
                return None
            
            cropped = image[y1:y2, x1:x2]
            
            logger.info(f"Cropped image from {width}x{height} to {cropped.shape[1]}x{cropped.shape[0]}")
            return cropped
            
        except Exception as e:
            logger.error(f"Failed to crop image: {e}")
            return None
    
    @staticmethod
    def crop_to_bytes(cropped_image: np.ndarray, image_type: str = "image/jpeg") -> bytes:
        """
        Convert cropped OpenCV image to bytes
        
        Args:
            cropped_image: OpenCV image array
            image_type: MIME type for encoding
            
        Returns:
            Image bytes
        """
        try:
            if "jpeg" in image_type.lower() or "jpg" in image_type.lower():
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                ext = '.jpg'
            elif "png" in image_type.lower():
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                ext = '.png'
            elif "webp" in image_type.lower():
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, 95]
                ext = '.webp'
            else:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                ext = '.jpg'
            
            success, encoded_image = cv2.imencode(ext, cropped_image, encode_params)
            
            if not success:
                raise ValueError("Failed to encode image")
            
            return encoded_image.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            raise
    
    @staticmethod
    def create_shelf_region_from_object(obj) -> ShelfRegion:
        """
        Create a ShelfRegion from a detected object
        
        Args:
            obj: Detected object with bounding box
            
        Returns:
            ShelfRegion object
        """
        return ShelfRegion(
            x1=obj.bounding_box.x1,
            y1=obj.bounding_box.y1,
            x2=obj.bounding_box.x2,
            y2=obj.bounding_box.y2,
            confidence=obj.confidence,
            detection_method="object_detection"
        ) 