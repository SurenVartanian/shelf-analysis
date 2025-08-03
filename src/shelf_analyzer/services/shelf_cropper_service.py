import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from PIL import Image
import io
import base64

from ..models.base_models import ShelfRegion, BoundingBox
from .detection.yolo_detection_strategy import YOLODetectionStrategy
from .image_processing.image_cropper import ImageCropper

logger = logging.getLogger(__name__)


class ShelfCropperService:
    """Service to detect and crop shelf regions from images"""
    
    def __init__(self, yolo_service):
        self.yolo_service = yolo_service
        self.yolo_strategy = YOLODetectionStrategy()
        self.image_cropper = ImageCropper()
    
    async def detect_and_crop_shelves(self, image_data: bytes, image_type: str = "image/jpeg") -> List[Tuple[np.ndarray, ShelfRegion]]:
        """
        Detect shelf regions and crop them into individual images
        
        Args:
            image_data: Raw image bytes
            image_type: MIME type of the image
            
        Returns:
            List of tuples: (cropped_image_array, shelf_region_info)
        """
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            height, width = image.shape[:2]
            logger.info(f"Processing image: {width}x{height}")
            
            # Detect objects using YOLO
            detected_objects = await self.yolo_service.detect_objects(image)
            logger.info(f"YOLO detected {len(detected_objects)} objects")
            
            # Find shelf regions
            shelf_regions, detection_method = await self._identify_shelf_regions(image, detected_objects)
            logger.info(f"Identified {len(shelf_regions)} shelf regions using {detection_method}")
            
            # Crop each shelf region
            cropped_shelves = []
            for i, shelf_region in enumerate(shelf_regions):
                cropped_image = self.image_cropper.crop_shelf_region(image, shelf_region)
                if cropped_image is not None:
                    cropped_shelves.append((cropped_image, shelf_region))
                    logger.info(f"Shelf {i+1}: cropped to {cropped_image.shape[1]}x{cropped_image.shape[0]}")
            
            return cropped_shelves
            
        except Exception as e:
            logger.error(f"Failed to detect and crop shelves: {e}")
            raise
    
    async def _identify_shelf_regions(self, image: np.ndarray, detected_objects: List) -> tuple[List[ShelfRegion], str]:
        """
        Identify the main frame/area for analysis using detection strategies
        
        Args:
            image: OpenCV image array
            detected_objects: List of detected objects from YOLO
            
        Returns:
            List containing single main frame region
        """
        # Use YOLO detection strategy
        shelf_regions, detection_method = await self.yolo_strategy.detect_shelf_regions(image, detected_objects)
        
        if shelf_regions:
            logger.info(f"YOLO strategy detected {len(shelf_regions)} regions using {detection_method}")
            return shelf_regions, detection_method
        else:
            logger.warning("YOLO strategy failed to detect any regions")
            return [], "no_detection"
    
    def _identify_main_display_area(self, image: np.ndarray) -> ShelfRegion:
        """
        Identify the main display area using image analysis when YOLO doesn't find large objects
        
        Args:
            image: OpenCV image array
            
        Returns:
            ShelfRegion representing the main display area
        """
        try:
            height, width = image.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection to find structural elements
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours to identify potential display areas
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and aspect ratio
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Look for large, roughly rectangular areas (potential displays)
                if area > (width * height * 0.1):  # At least 10% of image
                    aspect_ratio = w / h
                    # Prefer vertical displays (typical for retail)
                    if 0.3 < aspect_ratio < 2.0:
                        valid_contours.append((x, y, w, h, area))
            
            if valid_contours:
                # Use the largest valid contour as the main display
                largest_contour = max(valid_contours, key=lambda c: c[4])
                x, y, w, h = largest_contour[:4]
                
                logger.info(f"Found main display area via image analysis: {w}x{h} at ({x},{y})")
                
                return ShelfRegion(
                    bounding_box=BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=0.7
                    ),
                    shelf_level=1,
                    area_percentage=100.0
                )
            else:
                # Enhanced fallback: analyze the image to find the main display structure
                logger.info("Analyzing image structure for main display detection")
                main_display = self._detect_main_display_structure(image)
                return main_display
                
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            # Final fallback: use entire image
            return ShelfRegion(
                bounding_box=BoundingBox(
                    x1=0,
                    y1=0,
                    x2=float(width),
                    y2=float(height),
                    confidence=0.5
                ),
                shelf_level=1,
                area_percentage=100.0
                            )
    
    def _detect_main_display_structure(self, image: np.ndarray) -> ShelfRegion:
        """
        Enhanced method to detect the main display structure using multiple techniques
        
        Args:
            image: OpenCV image array
            
        Returns:
            ShelfRegion representing the main display area
        """
        try:
            height, width = image.shape[:2]
            
            # Method 1: Try to detect the red shelving structure
            red_display = self._detect_red_shelving(image)
            if red_display:
                logger.info("Detected red shelving structure")
                return red_display
            
            # Method 2: Use product density analysis to find the main display area
            density_display = self._detect_by_product_density(image)
            if density_display:
                logger.info("Detected main display via product density")
                return density_display
            
            # Method 3: Smart center area detection (excludes side displays)
            logger.info("Using smart center area detection")
            return self._detect_smart_center_area(image)
            
        except Exception as e:
            logger.warning(f"Enhanced display detection failed: {e}")
            # Final fallback: use entire image
            return ShelfRegion(
                bounding_box=BoundingBox(
                    x1=0,
                    y1=0,
                    x2=float(width),
                    y2=float(height),
                    confidence=0.4
                ),
                shelf_level=1,
                area_percentage=100.0
            )
    
    def _detect_red_shelving(self, image: np.ndarray) -> Optional[ShelfRegion]:
        """Detect red shelving structure in the image"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define red color range (red wraps around in HSV)
            # More specific range for retail shelving red
            lower_red1 = np.array([0, 100, 100])    # More saturated
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])  # More saturated
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for red regions
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in red regions
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest red region (likely the main shelving)
            if contours:
                # Sort by area and try to find the best candidate
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours[:3]:  # Check top 3 largest contours
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    
                    # Ensure it's a reasonable size and shape for shelving
                    if (w > image.shape[1] * 0.2 and h > image.shape[0] * 0.4 and  # Minimum size
                        w < image.shape[1] * 0.8 and h < image.shape[0] * 0.95 and  # Maximum size
                        h > w * 1.5):  # Should be taller than wide (vertical shelving)
                        
                        logger.info(f"Found red shelving structure: {w}x{h} at ({x},{y}) with area {area}")
                        return ShelfRegion(
                            bounding_box=BoundingBox(
                                x1=float(x),
                                y1=float(y),
                                x2=float(x + w),
                                y2=float(y + h),
                                confidence=0.9
                            ),
                            shelf_level=1,
                            area_percentage=100.0
                        )
            
            logger.info("No suitable red shelving structure found")
            return None
            
        except Exception as e:
            logger.warning(f"Red shelving detection failed: {e}")
            return None
    
    def _detect_by_product_density(self, image: np.ndarray) -> Optional[ShelfRegion]:
        """Detect main display area based on product density"""
        try:
            height, width = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to find product areas
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Group contours by vertical position to find the main display area
            if contours:
                # Find the vertical center of the image
                center_y = height // 2
                center_x = width // 2
                
                # Find contours near the center (both horizontally and vertically)
                center_contours = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_center_y = y + h // 2
                    contour_center_x = x + w // 2
                    
                    # If contour is near the center and reasonably sized
                    # More restrictive horizontally, but allow full height for vertical displays
                    if (abs(contour_center_y - center_y) < height * 0.4 and   # Allow more vertical range
                        abs(contour_center_x - center_x) < width * 0.25 and   # Keep horizontal constraint
                        w > 50 and h > 50):
                        center_contours.append((x, y, w, h))
                
                if center_contours:
                    # Calculate the bounding box of all center contours
                    min_x = min(c[0] for c in center_contours)
                    max_x = max(c[0] + c[2] for c in center_contours)
                    min_y = min(c[1] for c in center_contours)
                    max_y = max(c[1] + c[3] for c in center_contours)
                    
                    # Add some padding
                    padding = 20
                    min_x = max(0, min_x - padding)
                    max_x = min(width, max_x + padding)
                    min_y = max(0, min_y - padding)
                    max_y = min(height, max_y + padding)
                    
                    # Ensure the detected area is reasonable (not the entire image)
                    detected_width = max_x - min_x
                    detected_height = max_y - min_y
                    
                    if detected_width < width * 0.9 and detected_height < height * 0.9:
                        logger.info(f"Found main display via product density: {detected_width}x{detected_height}")
                        return ShelfRegion(
                            bounding_box=BoundingBox(
                                x1=float(min_x),
                                y1=float(min_y),
                                x2=float(max_x),
                                y2=float(max_y),
                                confidence=0.7
                            ),
                            shelf_level=1,
                            area_percentage=100.0
                        )
                    else:
                        logger.info(f"Product density detected area too large ({detected_width}x{detected_height}), skipping")
            
            return None
            
        except Exception as e:
            logger.warning(f"Product density detection failed: {e}")
            return None
    
    def _detect_smart_center_area(self, image: np.ndarray) -> ShelfRegion:
        """Smart center area detection that excludes side displays"""
        height, width = image.shape[:2]
        
        # Use a more aggressive center focus (exclude more of the sides)
        # But be less aggressive on top/bottom to include full display height
        center_margin_x = width * 0.15  # 15% margin on each side (more aggressive)
        center_margin_y = height * 0.02  # Only 2% margin top/bottom (less aggressive)
        
        logger.info(f"Using smart center area: {width-2*center_margin_x:.0f}x{height-2*center_margin_y:.0f}")
        
        return ShelfRegion(
            bounding_box=BoundingBox(
                x1=float(center_margin_x),
                y1=float(center_margin_y),
                x2=float(width - center_margin_x),
                y2=float(height - center_margin_y),
                confidence=0.6
            ),
            shelf_level=1,
            area_percentage=100.0
        )
    
    async def _detect_shelves_by_horizontal_lines(self, image: np.ndarray) -> List[ShelfRegion]:
        """
        Detect shelves by finding strong horizontal lines in the image
        Focuses on the main shelf area by filtering out side shelves
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of shelf regions
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect horizontal lines using Hough transform
            # More lenient parameters for real images with promotional materials
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=30)
            
            if lines is None:
                return []
            
            height, width = image.shape[:2]
            horizontal_lines = []
            
            # Filter for horizontal lines (angle close to 0 or 180 degrees)
            logger.info(f"Found {len(lines)} total lines, filtering for horizontal ones...")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Consider lines within ±15 degrees of horizontal
                if abs(angle) < 15 or abs(angle - 180) < 15:
                    horizontal_lines.append((y1, y2))
                    logger.info(f"Horizontal line found at y={y1}-{y2}, angle={angle:.1f}°")
            
            logger.info(f"Filtered to {len(horizontal_lines)} horizontal lines")
            
            if not horizontal_lines:
                return []
            
            # Sort lines by y-coordinate
            horizontal_lines.sort(key=lambda x: min(x[0], x[1]))
            
            # Focus on main shelf area by filtering lines that span most of the image width
            main_shelf_lines = []
            min_width_ratio = 0.6  # Lines must span at least 60% of image width
            
            for line in horizontal_lines:
                y1, y2 = line
                # Check if this line spans a significant portion of the image width
                # We'll assume horizontal lines span the full width for main shelves
                main_shelf_lines.append(line)
            
            logger.info(f"Focusing on {len(main_shelf_lines)} main shelf lines")
            
            # Create shelf regions between consecutive lines
            shelf_regions = []
            for i in range(len(main_shelf_lines) - 1):
                y1 = min(main_shelf_lines[i][0], main_shelf_lines[i][1])
                y2 = min(main_shelf_lines[i + 1][0], main_shelf_lines[i + 1][1])
                
                # Ensure minimum shelf height
                if y2 - y1 > 50:
                    shelf_region = ShelfRegion(
                        bounding_box=BoundingBox(
                            x1=0,
                            y1=float(y1),
                            x2=float(width),
                            y2=float(y2),
                            confidence=0.8
                        ),
                        shelf_level=i + 1,
                        area_percentage=100.0
                    )
                    shelf_regions.append(shelf_region)
            
            # Return whatever we found, let the main method decide on fallback
            return shelf_regions
            
        except Exception as e:
            logger.error(f"Horizontal line detection failed: {e}")
            return []
    
    def _detect_shelves_by_product_density(self, image: np.ndarray) -> List[ShelfRegion]:
        """
        Detect shelves by analyzing product density patterns
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of shelf regions
        """
        try:
            height, width = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding to find product areas
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and position
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter for reasonable product-sized contours
                if area > 1000 and w > 20 and h > 20:
                    valid_contours.append((x, y, w, h, area))
            
            logger.info(f"Found {len(valid_contours)} valid product contours")
            
            if not valid_contours:
                return []
            
            # Group contours by vertical position to identify shelf levels
            shelf_groups = []
            current_group = []
            last_y = -1
            
            # Sort contours by y-position
            valid_contours.sort(key=lambda c: c[1])
            
            for x, y, w, h, area in valid_contours:
                if last_y == -1 or abs(y - last_y) < 100:  # Group within 100px vertically (more realistic)
                    current_group.append((x, y, w, h, area))
                else:
                    if current_group:
                        shelf_groups.append(current_group)
                    current_group = [(x, y, w, h, area)]
                last_y = y
            
            if current_group:
                shelf_groups.append(current_group)
            
            logger.info(f"Grouped products into {len(shelf_groups)} potential shelf groups")
            
            # Create shelf regions from groups
            shelf_regions = []
            for i, group in enumerate(shelf_groups):
                if len(group) < 2:  # Need at least 2 products to be a shelf
                    continue
                
                # Calculate shelf boundaries
                min_y = min(y for x, y, w, h, area in group)
                max_y = max(y + h for x, y, w, h, area in group)
                
                # Add some padding
                min_y = max(0, min_y - 20)
                max_y = min(height, max_y + 20)
                
                # Ensure minimum shelf height (at least 150px)
                shelf_height = max_y - min_y
                if shelf_height < 150:
                    # Extend the shelf to minimum height
                    extension = (150 - shelf_height) // 2
                    min_y = max(0, min_y - extension)
                    max_y = min(height, max_y + extension)
                
                # Check for overlap with existing shelves and merge if needed
                overlapping = False
                for existing_shelf in shelf_regions:
                    existing_min_y = existing_shelf.bounding_box.y1
                    existing_max_y = existing_shelf.bounding_box.y2
                    
                    # Check if shelves overlap significantly
                    if (min_y < existing_max_y and max_y > existing_min_y and 
                        min(max_y, existing_max_y) - max(min_y, existing_min_y) > 50):
                        overlapping = True
                        break
                
                if not overlapping:
                    shelf_region = ShelfRegion(
                        bounding_box=BoundingBox(
                            x1=0,
                            y1=float(min_y),
                            x2=float(width),
                            y2=float(max_y),
                            confidence=0.7
                        ),
                        shelf_level=len(shelf_regions) + 1,  # Sequential numbering
                        area_percentage=100.0
                    )
                    shelf_regions.append(shelf_region)
                    logger.info(f"Created shelf region {len(shelf_regions)}: y={min_y}-{max_y}, products={len(group)}")
                else:
                    logger.info(f"Skipping overlapping shelf region: y={min_y}-{max_y}")
            
            logger.info(f"Product density detection found {len(shelf_regions)} shelf regions")
            return shelf_regions
            
        except Exception as e:
            logger.error(f"Product density detection failed: {e}")
            return []
    
    def _estimate_shelf_regions_grid(self, image: np.ndarray) -> List[ShelfRegion]:
        """
        Fallback: Estimate shelf regions using a simple grid approach
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of shelf regions
        """
        height, width = image.shape[:2]
        
        # Estimate number of shelves based on image height
        # Assume shelves are roughly 150-300px tall each
        estimated_shelves = max(2, min(8, height // 200))  # Between 2-8 shelves
        
        # Focus on the center 80% of the image width to avoid side shelves
        center_margin = width * 0.1  # 10% margin on each side
        center_x1 = center_margin
        center_x2 = width - center_margin
        
        num_shelves = estimated_shelves
        shelf_height = height // num_shelves
        
        logger.info(f"Grid-based detection: focusing on center area ({center_x1:.0f}-{center_x2:.0f}px) with {num_shelves} shelves")
        
        shelf_regions = []
        for i in range(num_shelves):
            y1 = i * shelf_height
            y2 = (i + 1) * shelf_height if i < num_shelves - 1 else height
            
            shelf_region = ShelfRegion(
                bounding_box=BoundingBox(
                    x1=float(center_x1),
                    y1=float(y1),
                    x2=float(center_x2),
                    y2=float(y2),
                    confidence=0.6
                ),
                shelf_level=i + 1,
                area_percentage=100.0
            )
            shelf_regions.append(shelf_region)
        
        return shelf_regions
    

    
    async def debug_shelf_detection(self, image_data: bytes, image_type: str = "image/jpeg") -> dict:
        """
        Debug method to see what shelf detection finds
        
        Args:
            image_data: Raw image bytes
            image_type: MIME type of the image
            
        Returns:
            Detailed debug information
        """
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            height, width = image.shape[:2]
            
            # Detect objects using YOLO
            detected_objects = await self.yolo_service.detect_objects(image)
            
            # Find shelf regions with detailed logging
            shelf_regions, detection_method = await self._identify_shelf_regions(image, detected_objects)
            
            # Crop each shelf region and get debug info
            cropped_shelves = []
            for i, shelf_region in enumerate(shelf_regions):
                cropped_image = self.image_cropper.crop_shelf_region(image, shelf_region)
                if cropped_image is not None:
                    cropped_shelves.append({
                        "shelf_id": i + 1,
                        "bounding_box": {
                            "x1": shelf_region.bounding_box.x1,
                            "y1": shelf_region.bounding_box.y1,
                            "x2": shelf_region.bounding_box.x2,
                            "y2": shelf_region.bounding_box.y2,
                            "width": shelf_region.bounding_box.x2 - shelf_region.bounding_box.x1,
                            "height": shelf_region.bounding_box.y2 - shelf_region.bounding_box.y1
                        },
                        "cropped_size": {
                            "width": cropped_image.shape[1],
                            "height": cropped_image.shape[0]
                        },
                        "confidence": shelf_region.bounding_box.confidence,
                        "shelf_level": shelf_region.shelf_level
                    })
            
            return {
                "image_info": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height
                },
                "yolo_detection": {
                    "total_objects": len(detected_objects),
                    "object_types": list(set([obj.object_type.value for obj in detected_objects])),
                    "objects": [
                        {
                            "type": obj.object_type.value,
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "bounding_box": {
                                "x1": obj.bounding_box.x1,
                                "y1": obj.bounding_box.y1,
                                "x2": obj.bounding_box.x2,
                                "y2": obj.bounding_box.y2
                            }
                        } for obj in detected_objects[:10]  # First 10 objects
                    ]
                },
                "shelf_detection": {
                    "total_shelves": len(shelf_regions),
                    "shelves": cropped_shelves,
                    "detection_method": detection_method
                }
            }
            
        except Exception as e:
            logger.error(f"Debug shelf detection failed: {e}")
            raise 