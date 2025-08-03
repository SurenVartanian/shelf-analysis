from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
import logging
import json
import numpy as np
import cv2

from .models import ShelfRegion
from .models.vision_models import (
    VisionAnalysisConfig, 
    VisionModel, 
    VisionAnalysisResult
)
from .services.yolo_service import YOLOService
from .services.image_service import ImageService
from .services.litellm_vision_service import LiteLLMVisionService
from .services.shelf_cropper_service import ShelfCropperService
from .services.vision_analysis_helper import VisionAnalysisHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shelf Analysis API",
    description="Analyze shelf images for product detection and empty space identification",
    version="0.1.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
yolo_service = YOLOService()  # Standard YOLO model
custom_yolo_service = YOLOService(use_custom_model=True)  # Custom trained model
image_service = ImageService()
shelf_cropper_service = ShelfCropperService(yolo_service)
custom_shelf_cropper_service = ShelfCropperService(custom_yolo_service)
vision_analysis_helper = VisionAnalysisHelper(yolo_service, shelf_cropper_service)
custom_vision_analysis_helper = VisionAnalysisHelper(custom_yolo_service, custom_shelf_cropper_service)

# Initialize LiteLLM Vision services for different models
vision_services = {
    "gpt-4o": LiteLLMVisionService(VisionAnalysisConfig(model=VisionModel.GPT_4O)),
    "gpt-4o-mini": LiteLLMVisionService(VisionAnalysisConfig(model=VisionModel.GPT_4O_MINI)),
    "gpt-4-vision-preview": LiteLLMVisionService(VisionAnalysisConfig(model=VisionModel.GPT_4_VISION_PREVIEW)),
    "gemini-flash": LiteLLMVisionService(VisionAnalysisConfig(model=VisionModel.GEMINI_FLASH)),
    "gemini-pro": LiteLLMVisionService(VisionAnalysisConfig(model=VisionModel.GEMINI_PRO)),
    "gemini-flash-lite": LiteLLMVisionService(VisionAnalysisConfig(model=VisionModel.GEMINI_FLASH_LITE))
}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Shelf Analysis API...")
    try:
        logger.info("Initializing standard YOLO service...")
        await yolo_service.initialize()
        logger.info("Standard YOLO service initialized successfully")
        
        logger.info("Initializing custom YOLO service...")
        await custom_yolo_service.initialize()
        logger.info("Custom YOLO service initialized successfully")
        
        logger.info("Initializing Vision services...")
        for model_name, service in vision_services.items():
            await service.initialize()
            logger.info(f"Vision service {model_name} initialized")
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api")
async def api_info():
    """API info endpoint"""
    return {"message": "Shelf Analysis API is running", "version": "0.1.0"}


@app.post("/analyze-vision")
async def analyze_shelf_vision(
    image: UploadFile = File(...),
    model: str = Form("gpt-4o")
):
    """
    Analyze shelf using LiteLLM vision models
    
    Args:
        image: Uploaded image file
        model: Vision model to use (gpt-4o, gemini-flash, claude-sonnet)
    
    Returns:
        VisionAnalysisResult with detailed shelf analysis
    """
    logger.info(f"=== VISION ANALYSIS REQUEST ===")
    logger.info(f"Received model parameter: '{model}'")
    logger.info(f"Available models in vision_services: {list(vision_services.keys())}")
    
    try:
        logger.info(f"Image filename: {image.filename}")
        logger.info(f"Image content type: {image.content_type}")
        logger.info(f"Model parameter received: '{model}'")
        
        vision_service, image_data, content_type, yolo_context = await vision_analysis_helper.prepare_vision_analysis(
            image, model, vision_services
        )
        return await vision_analysis_helper.analyze_with_vision_service(
            vision_service, image_data, content_type, yolo_context, stream=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-vision-stream")
async def analyze_shelf_vision_stream(
    image: UploadFile = File(...),
    model: str = Form("gpt-4o")
):
    """
    Stream shelf analysis using LiteLLM vision models
    
    Args:
        image: Uploaded image file
        model: Vision model to use
    
    Returns:
        StreamingResponse with real-time analysis
    """
    try:
        vision_service, image_data, content_type, yolo_context = await vision_analysis_helper.prepare_vision_analysis(
            image, model, vision_services
        )
        return await vision_analysis_helper.analyze_with_vision_service(
            vision_service, image_data, content_type, yolo_context, stream=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@app.post("/analyze-vision-custom")
async def analyze_shelf_vision_custom(
    image: UploadFile = File(...),
    model: str = Form("gpt-4o")
):
    """
    Analyze shelf image using custom trained YOLO model for object detection
    """
    try:
        vision_service, image_data, content_type, yolo_context = await custom_vision_analysis_helper.prepare_vision_analysis(
            image, model, vision_services
        )
        
        result = await custom_vision_analysis_helper.analyze_with_vision_service(
            vision_service, image_data, content_type, yolo_context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Custom vision analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-vision-custom-stream")
async def analyze_shelf_vision_custom_stream(
    image: UploadFile = File(...),
    model: str = Form("gpt-4o")
):
    """
    Analyze shelf image using custom trained YOLO model with streaming response
    """
    try:
        # Prepare vision analysis using custom YOLO model
        vision_service, image_data, content_type, yolo_context = await custom_vision_analysis_helper.prepare_vision_analysis(
            image, model, vision_services
        )
        
        return await custom_vision_analysis_helper.analyze_with_vision_service(
            vision_service, image_data, content_type, yolo_context, stream=True
        )
        
    except Exception as e:
        logger.error(f"Custom vision analysis streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@app.post("/analyze-vision-direct")
async def analyze_shelf_vision_direct(
    image: UploadFile = File(...),
    model: str = Form("gpt-4o"),
    stream: bool = Form(False)
):
    """
    Analyze shelf image directly with LLM (no YOLO cropping)
    This endpoint bypasses YOLO preprocessing for comparison
    """
    try:
        # Read image data
        image_data = await image.read()
        image_type = image.content_type or "image/jpeg"
        
        if model not in vision_services:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
        
        vision_service = vision_services[model]
        
        if not vision_service.is_ready():
            raise HTTPException(status_code=503, detail="Vision service not ready")
        
        logger.info(f"Starting direct vision analysis with model: {model}")
        
        if stream:
            # Streaming response
            async def generate_stream():
                try:
                    async for chunk in vision_service.analyze_shelf_stream(image_data, image_type):
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Non-streaming response
            result = await vision_service.analyze_shelf(image_data, image_type)
            return JSONResponse(content=result.model_dump())
            
    except Exception as e:
        logger.error(f"Direct vision analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "yolo": yolo_service.is_ready(),
            "image": image_service.is_ready()
        }
    }


@app.post("/debug-shelf-detection")
async def debug_shelf_detection(
    image: UploadFile = File(...)
):
    """
    Debug endpoint to see what shelf detection finds
    
    Returns:
        Detailed information about shelf detection process
    """
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_data = await image.read()
        
        # Initialize shelf cropper
        shelf_cropper = ShelfCropperService(yolo_service)
        
        # Get detailed debug info
        debug_info = await shelf_cropper.debug_shelf_detection(image_data, image.content_type)
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Shelf detection debug failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


@app.post("/visualize-cropping")
async def visualize_cropping(image: UploadFile = File(...)):
    """Enhanced visualization showing ALL YOLO detections and shelf regions"""
    try:
        # Read image data
        image_data = await image.read()
        
        # Convert to OpenCV format for visualization
        nparr = np.frombuffer(image_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Detect objects and shelf regions
        detected_objects = await yolo_service.detect_objects(original_image)
        shelf_regions, detection_method = await shelf_cropper_service._identify_shelf_regions(original_image, detected_objects)
        
        # Create a copy for visualization
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]
        
        # Color scheme for different object types
        color_map = {
            'bottle': (255, 0, 0),      # Blue
            'can': (0, 255, 0),         # Green  
            'package': (0, 0, 255),     # Red
            'fruit': (255, 255, 0),     # Cyan
            'box': (255, 0, 255),       # Magenta
            'person': (0, 255, 255),    # Yellow
            'chair': (128, 0, 128),     # Purple
            'tv': (255, 165, 0),        # Orange
            'laptop': (0, 128, 128),    # Teal
            'cell phone': (128, 128, 0), # Olive
        }
        
        # Draw ALL YOLO detections first (in background)
        object_counts = {}
        for obj in detected_objects:
            x1, y1, x2, y2 = (
                int(obj.bounding_box.x1),
                int(obj.bounding_box.y1),
                int(obj.bounding_box.x2),
                int(obj.bounding_box.y2)
            )
            
            # Get color for object type
            obj_type = obj.object_type.value
            color = color_map.get(obj_type, (128, 128, 128))  # Gray for unknown types
            
            # Count objects by type
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            # Draw rectangle with thinner line for individual objects
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add object label
            label = f"{obj_type}: {obj.confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw shelf regions on top (thicker lines, different color)
        for i, shelf_region in enumerate(shelf_regions):
            x1, y1, x2, y2 = (
                int(shelf_region.bounding_box.x1),
                int(shelf_region.bounding_box.y1),
                int(shelf_region.bounding_box.x2),
                int(shelf_region.bounding_box.y2)
            )
            
            # Draw shelf region with thick white border
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 255), 4)
            
            # Add shelf label
            label = f"MAIN FRAME {i+1} ({detection_method})"
            cv2.putText(vis_image, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add confidence
            conf_label = f"Conf: {shelf_region.bounding_box.confidence:.2f}"
            cv2.putText(vis_image, conf_label, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add comprehensive info panel
        info_y = 30
        cv2.putText(vis_image, f"Image: {width}x{height}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(vis_image, f"Detection Method: {detection_method}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(vis_image, f"Total Objects: {len(detected_objects)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(vis_image, f"Shelf Regions: {len(shelf_regions)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add object type breakdown
        info_y += 35
        cv2.putText(vis_image, "Object Breakdown:", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 20
        for obj_type, count in sorted(object_counts.items()):
            color = color_map.get(obj_type, (128, 128, 128))
            cv2.putText(vis_image, f"  {obj_type}: {count}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            info_y += 18
        
        # Add legend
        legend_y = height - 200
        cv2.putText(vis_image, "Legend:", (width - 200, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 20
        cv2.putText(vis_image, "  White Border = Main Frame", (width - 200, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 15
        cv2.putText(vis_image, "  Colored Boxes = YOLO Objects", (width - 200, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', vis_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vis_image_bytes = buffer.tobytes()
        
        # Return the visualized image with enhanced headers
        return Response(
            content=vis_image_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=visualization.jpg",
                "X-Detection-Method": detection_method,
                "X-Shelf-Count": str(len(shelf_regions)),
                "X-Total-Objects": str(len(detected_objects)),
                "X-Object-Types": ",".join(object_counts.keys())
            }
        )
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.shelf_analyzer.main:app", host="0.0.0.0", port=8000, reload=True)
