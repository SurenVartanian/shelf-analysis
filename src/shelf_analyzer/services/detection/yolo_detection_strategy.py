"""
YOLO Detection Strategy

Uses YOLO object detection to identify shelf regions based on detected objects.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .base_detection_strategy import BaseDetectionStrategy
from ...models.base_models import ShelfRegion, BoundingBox

logger = logging.getLogger(__name__)


class YOLODetectionStrategy(BaseDetectionStrategy):
    """YOLO-based shelf detection strategy"""
    
    def __init__(self):
        super().__init__("YOLO Detection")
    
    async def detect_shelf_regions(
        self, 
        image: np.ndarray, 
        detected_objects: List = None
    ) -> Tuple[List[ShelfRegion], str]:
        """
        Detect shelf regions using YOLO object detection
        
        Args:
            image: OpenCV image array
            detected_objects: List of detected objects from YOLO
            
        Returns:
            Tuple of (list of shelf regions, detection method name)
        """
        height, width = image.shape[:2]
        
        if not detected_objects:
            self.logger.warning("No detected objects provided for YOLO detection")
            return [], "yolo_no_objects"
        
        try:
            self.logger.info(f"YOLO detected {len(detected_objects)} objects:")
            for i, obj in enumerate(detected_objects):
                self.logger.info(f"  Object {i+1}: {obj.label} (confidence: {obj.confidence:.2f})")
            
            frame_objects = []
            for obj in detected_objects:
                obj_area = (obj.bounding_box.x2 - obj.bounding_box.x1) * (obj.bounding_box.y2 - obj.bounding_box.y1)
                image_area = height * width
                coverage_ratio = obj_area / image_area
                
                if 0.15 < coverage_ratio < 0.8:  # Between 15% and 80% of image
                    if obj.label.lower() not in ['book', 'suitcase', 'backpack', 'handbag', 'cell phone']:
                        frame_objects.append((obj, coverage_ratio))
                        self.logger.info(f"Main frame candidate: {obj.label} with {coverage_ratio:.1%} coverage")
                    else:
                        self.logger.info(f"Skipping product-like object: {obj.label} with {coverage_ratio:.1%} coverage")
            
            if frame_objects:
                frame_objects.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
                best_object, coverage = frame_objects[0]
                
                self.logger.info(f"Using best frame object: {best_object.label} with {coverage:.1%} coverage")
                
                shelf_region = ShelfRegion(
                    bounding_box=BoundingBox(
                        x1=best_object.bounding_box.x1,
                        y1=best_object.bounding_box.y1,
                        x2=best_object.bounding_box.x2,
                        y2=best_object.bounding_box.y2,
                        confidence=best_object.confidence
                    ),
                    shelf_level=1,
                    area_percentage=coverage * 100.0,
                    detection_method="yolo_object_detection"
                )
                
                self.log_detection_result([shelf_region], "yolo_object_detection")
                return [shelf_region], "yolo_object_detection"
            
            self.logger.info("No suitable frame objects found, trying enhanced detection")
            enhanced_region = self._detect_enhanced_structure(image)
            
            if enhanced_region:
                self.log_detection_result([enhanced_region], "yolo_enhanced_detection")
                return [enhanced_region], "yolo_enhanced_detection"
            
            fallback_region = ShelfRegion(
                bounding_box=BoundingBox(
                    x1=0,
                    y1=0,
                    x2=float(width),
                    y2=float(height),
                    confidence=0.4
                ),
                shelf_level=1,
                area_percentage=100.0,
                detection_method="yolo_fallback"
            )
            
            self.log_detection_result([fallback_region], "yolo_fallback")
            return [fallback_region], "yolo_fallback"
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return [], "yolo_error"
    
    def _detect_enhanced_structure(self, image: np.ndarray) -> Optional[ShelfRegion]:
        """
        Enhanced method to detect the main display structure using multiple techniques
        
        Args:
            image: OpenCV image array
            
        Returns:
            ShelfRegion representing the main display area or None
        """
        try:
            height, width = image.shape[:2]
            
            red_display = self._detect_red_shelving(image)
            if red_display:
                self.logger.info("Detected red shelving structure")
                return red_display
            
            density_display = self._detect_by_product_density(image)
            if density_display:
                self.logger.info("Detected main display via product density")
                return density_display
            
            self.logger.info("Using smart center area detection")
            return self._detect_smart_center_area(image)
            
        except Exception as e:
            self.logger.warning(f"Enhanced display detection failed: {e}")
            return None
    
    def _detect_red_shelving(self, image: np.ndarray) -> Optional[ShelfRegion]:
        """Detect red shelving structure in the image"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Red color range (red wraps around in HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            kernel = np.ones((5,5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours[:3]:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    
                    if (w > image.shape[1] * 0.2 and h > image.shape[0] * 0.4 and
                        w < image.shape[1] * 0.8 and h < image.shape[0] * 0.95 and
                        h > w * 1.5):  # Should be taller than wide
                        
                        self.logger.info(f"Found red shelving structure: {w}x{h} at ({x},{y}) with area {area}")
                        return ShelfRegion(
                            bounding_box=BoundingBox(
                                x1=float(x),
                                y1=float(y),
                                x2=float(x + w),
                                y2=float(y + h),
                                confidence=0.9
                            ),
                            shelf_level=1,
                            area_percentage=100.0,
                            detection_method="red_shelving_detection"
                        )
            
            self.logger.info("No suitable red shelving structure found")
            return None
            
        except Exception as e:
            self.logger.error(f"Red shelving detection failed: {e}")
            return None
    
    def _detect_by_product_density(self, image: np.ndarray) -> Optional[ShelfRegion]:
        """Detect main display area based on product density"""
        try:
            height, width = image.shape[:2]
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            edges = cv2.Canny(gray, 50, 150)
            
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                best_contour = None
                best_density = 0
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if w < width * 0.1 or h < height * 0.1:
                        continue
                    
                    roi_edges = edges[y:y+h, x:x+w]
                    edge_density = np.sum(roi_edges > 0) / (w * h)
                    
                    if edge_density > best_density:
                        best_density = edge_density
                        best_contour = contour
                
                if best_contour is not None:
                    x, y, w, h = cv2.boundingRect(best_contour)
                    self.logger.info(f"Found high-density region: {w}x{h} at ({x},{y}) with density {best_density:.3f}")
                    
                    return ShelfRegion(
                        bounding_box=BoundingBox(
                            x1=float(x),
                            y1=float(y),
                            x2=float(x + w),
                            y2=float(y + h),
                            confidence=0.8
                        ),
                        shelf_level=1,
                        area_percentage=100.0,
                        detection_method="product_density_detection"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Product density detection failed: {e}")
            return None
    
    def _detect_smart_center_area(self, image: np.ndarray) -> ShelfRegion:
        """Detect smart center area, excluding side displays"""
        try:
            height, width = image.shape[:2]
            
            # Define center area (exclude 20% from each side)
            margin_x = int(width * 0.2)
            margin_y = int(height * 0.1)  # Smaller vertical margin
            
            x1 = margin_x
            y1 = margin_y
            x2 = width - margin_x
            y2 = height - margin_y
            
            self.logger.info(f"Using smart center area: {x1},{y1} to {x2},{y2}")
            
            return ShelfRegion(
                bounding_box=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=0.7
                ),
                shelf_level=1,
                area_percentage=100.0,
                detection_method="smart_center_detection"
            )
            
        except Exception as e:
            self.logger.error(f"Smart center detection failed: {e}")
            # Fallback to entire image
            return ShelfRegion(
                bounding_box=BoundingBox(
                    x1=0,
                    y1=0,
                    x2=float(width),
                    y2=float(height),
                    confidence=0.5
                ),
                shelf_level=1,
                area_percentage=100.0,
                detection_method="smart_center_fallback"
            ) 