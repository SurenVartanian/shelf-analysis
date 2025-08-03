import base64
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
import time
import os

import litellm

from ..models.vision_models import (
    VisionModel,
    VisionAnalysisConfig,
    Product,
    ShelfItem,
    ScoreItem,
    ScoresDict,
    EmptySpace,
    VisionAnalysisResult
)

logger = logging.getLogger(__name__)


class LiteLLMVisionService:
    """Vision analysis service using LiteLLM for multi-model support"""
    
    def __init__(self, config: VisionAnalysisConfig):
        self.config = config
        self._initialized = False
        
    async def initialize(self):
        """Initialize the LiteLLM service"""
        try:
            import logging
            litellm_logger = logging.getLogger("litellm")
            litellm_logger.setLevel(logging.DEBUG)
            
            litellm.set_verbose = True
            
            logger.info("LiteLLM debug logging enabled")
            
            if self.config.model.value in ["gemini-flash", "gemini-pro", "gemini-flash-lite"]:
                if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set for Vertex AI")
                if not os.getenv("GOOGLE_CLOUD_PROJECT"):
                    logger.warning("GOOGLE_CLOUD_PROJECT not set for Vertex AI")
                    
            elif self.config.model.value in ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"]:
                if not os.getenv("OPENAI_API_KEY"):
                    logger.warning("OPENAI_API_KEY not set")
                    
            self._initialized = True
            logger.info(f"LiteLLM Vision Service initialized with model: {self.config.model.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LiteLLM Vision Service: {e}")
            self._initialized = False
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._initialized
    
    def _get_litellm_model_name(self) -> str:
        """Get the actual LiteLLM model name based on our model enum"""
        vertex_project = os.getenv("VERTEX_PROJECT", "392356656271")
        vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")
        
        model_mapping = {
            # Google Cloud Vertex AI Models - use the correct format
            VisionModel.GEMINI_FLASH.value: f"gemini-2.5-flash",
            VisionModel.GEMINI_PRO.value: f"gemini-2.5-pro", 
            VisionModel.GEMINI_FLASH_LITE.value: f"gemini-2.5-flash-lite",
            
            # OpenAI Models
            VisionModel.GPT_4O.value: "gpt-4o",
            VisionModel.GPT_4O_MINI.value: "gpt-4o-mini",
            VisionModel.GPT_4_VISION_PREVIEW.value: "gpt-4-vision-preview"
        }
        
        base_model = model_mapping.get(self.config.model.value, self.config.model.value)
        
        # For Google models, use the vertex_ai provider with project/location
        if self.config.model.value in ["gemini-flash", "gemini-pro", "gemini-flash-lite"]:
            final_model = f"vertex_ai/{base_model}"
        else:
            final_model = base_model
            
        logger.info(f"Model mapping: {self.config.model.value} -> {final_model}")
        return final_model
    
    async def analyze_shelf(self, image_data: bytes, image_type: str = "image/jpeg", yolo_context: dict = None) -> VisionAnalysisResult:
        """
        Analyze shelf image using LiteLLM vision model
        
        Args:
            image_data: Image data in bytes
            image_type: Image MIME type
            yolo_context: YOLO detection context (optional)
            
        Returns:
            VisionAnalysisResult with detailed analysis
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized")
        
        start_time = time.time()
        
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            messages = self._create_vision_messages(image_base64, image_type, yolo_context)
            
            result = await self._create_litellm_completion(image_data, image_type)
            
            response_content = result.choices[0].message.content
            logger.info(f"LiteLLM response received: {type(result)}")
            logger.info(f"Raw LLM response content: {response_content}")
            
            analysis_data = json.loads(response_content)
            
            processing_time = (time.time() - start_time) * 1000
            analysis_data['processing_time_ms'] = processing_time
            analysis_data['model_used'] = self.config.model.value
            
            result = VisionAnalysisResult(**analysis_data)
            logger.info("Successfully parsed LLM response with structured schema")
            return result
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise
    
    async def analyze_shelf_stream(self, image_data: bytes, image_type: str = "image/jpeg", yolo_context: dict = None) -> AsyncGenerator[str, None]:
        """
        Analyze shelf image using LiteLLM vision model (streaming)
        
        Args:
            image_data: Image data in bytes
            image_type: Image MIME type
            yolo_context: YOLO detection context (optional)
            
        Yields:
            Streaming chunks of analysis
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized")
        
        try:
            response = await self._create_litellm_completion(image_data, image_type, stream=True)
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming vision analysis failed: {e}")
            raise
    
    async def _create_litellm_completion(self, image_data: bytes, image_type: str, stream: bool = False):
        """Create LiteLLM completion with unified parameters"""
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        messages = self._create_vision_messages(image_base64, image_type)
        
        model_name = self._get_litellm_model_name()
        logger.info(f"Using LiteLLM model: {model_name} for config model: {self.config.model.value}, stream={stream}")
        
        request_params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
        }
        
        if stream:
            request_params["stream"] = True
        
        supported_models = [
            "gemini-flash", "gemini-pro", "gemini-flash-lite",  # Vertex AI
            "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"    # OpenAI
        ]
        
        if self.config.model.value in supported_models:
            request_params["response_format"] = VisionAnalysisResult
            logger.info(f"Added structured output format for {self.config.model.value}")
        
        logger.info(f"Sending to LiteLLM - Model: {model_name}, request_params {request_params}")
        
        try:
            logger.info("About to call litellm.acompletion...")
            result = await litellm.acompletion(**request_params)
            logger.info(f"LiteLLM response received: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            raise
    
    def _create_vision_messages(self, image_base64: str, image_type: str, yolo_context: dict = None) -> List[Dict[str, Any]]:
        """Create messages for vision model analysis"""
        system_prompt = """
You are a specialist in visual analysis of store displays in Mexico.
You can only see a photo. Do not assume anything—analyze only what is clearly visible.

Important:
- If the image does NOT show a store display or product stand:
  - Set "is_display_detected": false.
  - All scores (including display_cleanliness) must be 0.
  - Do not include any banners or products.
  - In comments, mention that the image doesn't show a store display.

- If a display is shown, set "is_display_detected": true and perform the analysis.

- A promotional banner is only considered such if it's a clearly marked ad (poster, label, or sticker
  with logos, offers, or promo text). Don't include generic decor or text.

- If no banners are detected:
  - Leave the banners list empty.
  - Set banner_visibility to 0 and comment that no banners were found.

Your task:

1. Detect promotional banners (only clearly visible and properly marked; text should be in Spanish).

2. Identify shelves. For each shelf, return:
   - shelf_position (top-down: 1 = top)
   - shelf_visibility (0–100)
   - products: list of products on the shelf

3. For each product:
   - name: short name (if readable or recognizable)
   - full_name: full name (if readable)
   - count: approximate quantity based on visible packages
   Do not invent product names or include unidentifiable items.

4. Rate the following (0–100) with a short (1–2 sentence) comment:
   - Banner visibility
   - Product filling
   - Promo match
   - Product neatness
   - Display cleanliness
   - Shelf arrangement quality

5. Total score = average of all above, with a general visual impression comment.
"""
        
        # Create user prompt with YOLO context if available
        if yolo_context:
            yolo_info = f"""
YOLO Detection Results:
- Detected {yolo_context['total_objects']} objects
- Object types: {', '.join(yolo_context['object_types'])}
- Shelf regions detected: {yolo_context['shelf_regions_detected']}
- Empty spaces detected: {yolo_context['empty_spaces_detected']}
- Detection method: {yolo_context['detection_method']}
- Cropping applied: {'Yes' if yolo_context['cropping_applied'] else 'No'}

Use YOLO's detection as a guide, but rely on your visual analysis for final results.
If you see products YOLO missed, include them. If YOLO detected something you can't see, exclude it.
"""
        else:
            yolo_info = ""
        
        user_prompt = f"""
Analyze this display photo and identify products and scores.
{yolo_info}
‼️ If you cannot clearly see a product, **do not include it** in the result.
Do not invent brands or positions that aren't visible.

Focus on:
1. Product identification and counting on each shelf
2. Shelf visibility and organization
3. Overall display quality assessment
4. Banner detection and promotional materials

Provide your analysis in the specified JSON format.
"""
        
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_type};base64,{image_base64}"}
                    }
                ]
            }
        ]

