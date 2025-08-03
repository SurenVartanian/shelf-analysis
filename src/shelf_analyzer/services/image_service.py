import cv2
import numpy as np
from PIL import Image
import io
from typing import Union
import logging

logger = logging.getLogger(__name__)


class ImageService:
    """Image processing service"""
    
    def __init__(self):
        self._ready = True
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._ready
    
    async def process_image(self, image_data: bytes) -> np.ndarray:
        """
        Process uploaded image data into OpenCV format
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            OpenCV image array (BGR format)
        """
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
            
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Processed image: {opencv_image.shape}")
            return opencv_image
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def resize_image(
        self, 
        image: np.ndarray, 
        max_width: int = 1024, 
        max_height: int = 768
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: OpenCV image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
            
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized
        
        return image
    
    def encode_image_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """
        Encode OpenCV image to base64 string
        
        Args:
            image: OpenCV image
            format: Image format (JPEG, PNG)
            
        Returns:
            Base64 encoded image string
        """
        import base64
        
        # Encode image
        _, buffer = cv2.imencode(f'.{format.lower()}', image)
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/{format.lower()};base64,{image_base64}"
