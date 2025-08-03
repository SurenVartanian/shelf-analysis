"""
Vision Analysis Helper Service

Extracts shared logic for vision analysis endpoints to eliminate code duplication.
"""

import logging
import numpy as np
import cv2
from fastapi import HTTPException, UploadFile
from typing import Tuple, Union, List
from fastapi.responses import StreamingResponse
import json

from .litellm_vision_service import LiteLLMVisionService
from .yolo_service import YOLOService
from .shelf_cropper_service import ShelfCropperService
from .image_processing.image_cropper import ImageCropper

logger = logging.getLogger(__name__)


class VisionAnalysisHelper:
    """Helper service for vision analysis operations."""
    
    def __init__(self, yolo_service: YOLOService, shelf_cropper_service: ShelfCropperService):
        self.yolo_service = yolo_service
        self.shelf_cropper_service = shelf_cropper_service
        self.image_cropper = ImageCropper()
    
    async def prepare_vision_analysis(
        self, 
        image: UploadFile, 
        model: str, 
        vision_services: dict[str, LiteLLMVisionService]
    ) -> Tuple[LiteLLMVisionService, bytes, str, dict]:
        """
        Prepare vision analysis by validating inputs and processing image.
        
        Args:
            image: Uploaded image file
            model: Model name to use
            vision_services: Dictionary of available vision services
            
        Returns:
            Tuple of (vision_service, image_data, content_type, yolo_context)
            
        Raises:
            HTTPException: If validation fails
        """
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if model not in vision_services:
            available_models = list(vision_services.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model}' not available. Available: {available_models}"
            )
        
        image_data = await image.read()
        
        vision_service = vision_services[model]
        
        if not vision_service.is_ready():
            raise HTTPException(status_code=503, detail=f"Vision service {model} not ready")
        
        logger.info("Detecting main frame for cropping...")
        
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detected_objects = await self.yolo_service.detect_objects(cv_image)
        shelf_regions, detection_method = await self.shelf_cropper_service._identify_shelf_regions(cv_image, detected_objects)
        
        yolo_context = await self._prepare_yolo_context(detected_objects, shelf_regions, detection_method, cv_image)
        
        if shelf_regions:
            main_frame = shelf_regions[0]  # Use the first (and only) detected region
            cropped_cv_image = self.image_cropper.crop_shelf_region(cv_image, main_frame)
            
            if cropped_cv_image is not None:
                cropped_image_data = self.image_cropper.crop_to_bytes(cropped_cv_image, image.content_type)
                logger.info(f"Using cropped main frame ({detection_method}) for LLM analysis")
                image_data = cropped_image_data
            else:
                logger.warning("Cropping failed, using original image")
        else:
            logger.warning("No main frame detected, using original image")
        
        logger.info(f"UI requested model: '{model}', using vision service: {type(vision_service).__name__}")
        logger.info(f"Vision service config model: {vision_service.config.model}")
        
        return vision_service, image_data, image.content_type, yolo_context
    
    async def _prepare_yolo_context(self, detected_objects: List, shelf_regions: List, detection_method: str, cv_image: np.ndarray) -> dict:
        """Prepare YOLO detection context for LLM prompt"""
        object_counts = {}
        for obj in detected_objects:
            obj_type = obj.object_type.value
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        object_type_summary = []
        for obj_type, count in object_counts.items():
            object_type_summary.append(f"{count} {obj_type}s")
        
        # Detect empty spaces
        empty_spaces = []
        if detected_objects and cv_image is not None:
            try:
                empty_spaces = await self.yolo_service.detect_empty_spaces(cv_image, detected_objects)
            except Exception as e:
                logger.warning(f"Empty space detection failed: {e}")
        
        return {
            "total_objects": len(detected_objects),
            "object_types": object_type_summary,
            "object_counts": object_counts,
            "shelf_regions_detected": len(shelf_regions),
            "detection_method": detection_method,
            "empty_spaces_detected": len(empty_spaces),
            "cropping_applied": len(shelf_regions) > 0
        }
    
    async def analyze_with_vision_service(
        self, 
        vision_service: LiteLLMVisionService, 
        image_data: bytes, 
        content_type: str, 
        yolo_context: dict,
        stream: bool = False
    ) -> Union[dict, StreamingResponse]:
        """
        Perform vision analysis with the prepared service and data.
        
        Args:
            vision_service: Prepared vision service
            image_data: Processed image data
            content_type: Image content type
            yolo_context: YOLO detection context
            stream: Whether to return streaming response
            
        Returns:
            Analysis result or streaming response
        """
        if stream:
            async def generate_stream():
                try:
                    async for chunk in vision_service.analyze_shelf_stream(image_data, content_type, yolo_context):
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield "data: {\"status\": \"complete\"}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            result = await vision_service.analyze_shelf(image_data, content_type, yolo_context)
            logger.info(f"Vision analysis completed")
            return result 