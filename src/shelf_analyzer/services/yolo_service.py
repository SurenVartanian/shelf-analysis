import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
import logging
import os

from ..models.base_models import DetectedObject, BoundingBox, ObjectType, ShelfRegion

logger = logging.getLogger(__name__)


class YOLOService:
    """YOLO-based object detection service"""
    
    def __init__(self, model_name: str = "yolov8n.pt", use_custom_model: bool = False):
        self.model_name = model_name
        self.use_custom_model = use_custom_model
        self.model = None
        self._ready = False
        
        # Custom model class names (matching our training dataset)
        self.custom_class_names = {
            0: "bottle",
            1: "can", 
            2: "package",
            3: "fruit",
            4: "box"
        }
        
        # Custom model class mapping to our ObjectType enum
        self.custom_class_mapping = {
            0: ObjectType.BOTTLE,    # bottle
            1: ObjectType.CAN,       # can
            2: ObjectType.PACKAGE,   # package
            3: ObjectType.FRUIT,     # fruit
            4: ObjectType.BOX        # box
        }
    
    async def initialize(self):
        """Initialize YOLO model"""
        try:
            if self.use_custom_model:
                # Use our custom trained model
                custom_model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models", "yolo", "shelf_analysis_custom.pt"
                )
                if os.path.exists(custom_model_path):
                    logger.info(f"Loading custom YOLO model: {custom_model_path}")
                    self.model = YOLO(custom_model_path)
                    logger.info("Custom YOLO model loaded successfully")
                else:
                    logger.warning(f"Custom model not found at {custom_model_path}, falling back to standard model")
                    self.use_custom_model = False
                    self.model_name = "yolov8n.pt"
            
            if not self.use_custom_model:
                # Use standard YOLO model
                logger.info(f"Loading standard YOLO model: {self.model_name}")
                self.model = YOLO(self.model_name)
                logger.info("Standard YOLO model loaded successfully")
            
            self._ready = True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._ready and self.model is not None
    
    async def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in the image using YOLO
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of detected objects
        """
        if not self.is_ready():
            raise RuntimeError("YOLO service not initialized")
        
        try:
            # Run YOLO inference with very low confidence threshold to catch everything
            results = self.model(image, conf=0.1, classes=None)  # Much lower threshold
            detected_objects = []
            
            logger.info(f"YOLO inference completed. Results: {len(results)}")
            
            # Process results
            for result in results:
                logger.info(f"Processing result with boxes: {result.boxes is not None}")
                if result.boxes is not None:
                    logger.info(f"Number of boxes in result: {len(result.boxes)}")
                    for box in result.boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Map class to object type based on model type
                        if self.use_custom_model:
                            object_type = self._map_custom_class_to_type(class_id)
                            label = self.custom_class_names.get(class_id, f"class_{class_id}")
                        else:
                            object_type = self._map_class_to_type(class_id)
                            label = self.model.names[class_id]
                        
                        detected_object = DetectedObject(
                            object_type=object_type,
                            bounding_box=BoundingBox(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                                confidence=confidence
                            ),
                            confidence=confidence,
                            label=label
                        )
                        
                        detected_objects.append(detected_object)
            
            logger.info(f"Detected {len(detected_objects)} objects using {'custom' if self.use_custom_model else 'standard'} model")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise
    
    async def detect_empty_spaces(
        self, 
        image: np.ndarray, 
        detected_objects: List[DetectedObject]
    ) -> List[ShelfRegion]:
        """
        Detect empty shelf spaces using a grid-based approach
        This looks for gaps between products rather than completely empty shelves
        
        Args:
            image: OpenCV image array
            detected_objects: Previously detected objects
            
        Returns:
            List of empty shelf regions
        """
        try:
            height, width = image.shape[:2]
            
            # Create a grid to check for product coverage
            grid_size = 80  # Size of each grid cell in pixels
            empty_spaces = []
            
            logger.info(f"Analyzing image {width}x{height} with grid size {grid_size}")
            logger.info(f"Found {len(detected_objects)} detected objects")
            
            # Use ALL detected objects for gap analysis, but with different confidence
            # This includes 'unknown' objects that might be packages YOLO can't classify
            shelf_objects = detected_objects  # Use everything for now
            
            # Remove only the most obvious background objects
            background_labels = ['refrigerator', 'person']
            shelf_objects = [obj for obj in detected_objects if obj.label not in background_labels]
            
            logger.info(f"Using {len(shelf_objects)} objects for gap analysis (including unknown objects)")
            
            # Reality check: If we detect very few shelf objects for a large image,
            # YOLO probably isn't seeing the actual products
            expected_objects = (width * height) / 100000  # Rough heuristic
            if len(shelf_objects) < expected_objects:
                logger.warning(f"Only detected {len(shelf_objects)} objects for {width}x{height} image")
                logger.warning(f"Expected at least {expected_objects:.0f} objects - YOLO likely missing products")
                logger.info("Being conservative: not marking any areas as empty")
                return []
            
            # Debug: Show exactly what objects we're working with
            if len(shelf_objects) == 0:
                logger.warning("NO SHELF OBJECTS FOUND! This will mark everything as empty.")
                logger.info("All detected objects:")
                for obj in detected_objects:
                    logger.info(f"  - {obj.label} (confidence: {obj.confidence:.2f})")
                
                # If we can't detect any shelf products, don't mark anything as empty
                # Better to show no red boxes than wrong red boxes
                logger.info("Returning no empty spaces due to lack of detected products")
                return []
            
            logger.info(f"Working with these shelf objects:")
            for obj in shelf_objects:
                logger.info(f"  - {obj.label} at ({obj.bounding_box.x1:.0f}, {obj.bounding_box.y1:.0f}) confidence: {obj.confidence:.2f}")
            
            # Estimate horizontal shelf bands (typical shelf layout)
            num_shelves = 4
            shelf_height = height // num_shelves
            
            for shelf_idx in range(num_shelves):
                shelf_y1 = shelf_idx * shelf_height
                shelf_y2 = (shelf_idx + 1) * shelf_height
                
                # Look for empty horizontal segments within this shelf band
                empty_segments = self._find_empty_segments_in_shelf(
                    shelf_objects, shelf_y1, shelf_y2, width, grid_size
                )
                
                # Convert segments to ShelfRegion objects
                for segment_idx, (x1, x2) in enumerate(empty_segments):
                    # Only mark as empty if segment is reasonably wide
                    segment_width = x2 - x1
                    if segment_width > grid_size * 3:  # Require larger gaps (3 grid cells)
                        
                        area_percentage = (segment_width * shelf_height) / (width * height) * 100
                        
                        logger.info(f"Found empty segment in shelf {shelf_idx + 1}: ({x1}, {shelf_y1}) to ({x2}, {shelf_y2})")
                        
                        empty_space = ShelfRegion(
                            bounding_box=BoundingBox(
                                x1=float(x1),
                                y1=float(shelf_y1),
                                x2=float(x2),
                                y2=float(shelf_y2),
                                confidence=0.7
                            ),
                            shelf_level=shelf_idx + 1,
                            area_percentage=area_percentage,
                            marked_color="red"
                        )
                        
                        empty_spaces.append(empty_space)
            
            logger.info(f"Found {len(empty_spaces)} empty segments")
            return empty_spaces
            
        except Exception as e:
            logger.error(f"Empty space detection failed: {e}")
            raise
    
    def _map_custom_class_to_type(self, class_id: int) -> ObjectType:
        """Map custom model class ID to our object type"""
        return self.custom_class_mapping.get(class_id, ObjectType.PACKAGE)
    
    def _map_class_to_type(self, class_id: int) -> ObjectType:
        """Map YOLO class ID to our object type"""
        # Focus on deli/convenience store products
        class_mapping = {
            39: ObjectType.BOTTLE,     # bottle
            44: ObjectType.BOTTLE,     # wine glass (bottles)
            46: ObjectType.FRUIT,      # banana (individual fruits)
            47: ObjectType.FRUIT,      # apple (individual fruits)
            51: ObjectType.PACKAGE,    # bowl (containers)
            52: ObjectType.FRUIT,      # orange (individual fruits)
            53: ObjectType.PACKAGE,    # broccoli (packaged vegetables)
            54: ObjectType.PACKAGE,    # carrot (packaged vegetables)
            84: ObjectType.PACKAGE,    # book (rectangular packages like chips)
            # Add more deli-relevant objects
        }
        
        return class_mapping.get(class_id, ObjectType.PACKAGE)  # Default to PACKAGE for unknown items
    
    def _find_empty_segments_in_shelf(
        self, 
        shelf_objects: List[DetectedObject], 
        shelf_y1: int, 
        shelf_y2: int, 
        width: int, 
        grid_size: int
    ) -> List[Tuple[int, int]]:
        """
        Find empty horizontal segments within a shelf band
        
        Args:
            shelf_objects: Objects that could be on shelves (excluding background)
            shelf_y1, shelf_y2: Y coordinates of the shelf band
            width: Image width
            grid_size: Size of grid cells for analysis
            
        Returns:
            List of (x1, x2) tuples representing empty segments
        """
        # Create a boolean array representing occupied horizontal space
        occupied = [False] * width
        
        # Mark areas as occupied where objects exist
        for obj in shelf_objects:
            box = obj.bounding_box
            
            # Check if object overlaps with this shelf band
            if (box.y1 < shelf_y2 and box.y2 > shelf_y1):
                # Mark horizontal range as occupied
                x1 = max(0, int(box.x1))
                x2 = min(width, int(box.x2))
                
                # Add some padding around detected objects
                padding = grid_size // 2  # Larger padding to account for YOLO imprecision
                x1 = max(0, x1 - padding)
                x2 = min(width, x2 + padding)
                
                for x in range(x1, x2):
                    occupied[x] = True
        
        # Find continuous empty segments
        empty_segments = []
        start = None
        
        for x in range(width):
            if not occupied[x]:  # Empty space
                if start is None:
                    start = x
            else:  # Occupied space
                if start is not None:
                    # End of empty segment
                    empty_segments.append((start, x))
                    start = None
        
        # Handle case where empty segment goes to end of image
        if start is not None:
            empty_segments.append((start, width))
        
        return empty_segments
    
    def _conservative_empty_detection(self, height: int, width: int) -> List[ShelfRegion]:
        """
        Very conservative empty space detection when we can't detect any shelf products
        Only marks obviously empty areas
        """
        logger.info("Using conservative empty detection")
        
        # Don't mark anything as empty if we can't detect products
        # Better to miss empty spaces than to mark stocked shelves as empty
        return []
    
    def _estimate_shelf_regions(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Estimate shelf regions based on image dimensions
        This is a simple heuristic - could be improved with shelf detection
        """
        # Assume 3-4 horizontal shelves, but make them smaller regions
        # Focus on center areas where products are typically placed
        shelf_height = height // 4
        margin_x = width // 8  # Smaller horizontal margins
        margin_y = shelf_height // 6  # Add vertical margins to focus on product areas
        
        shelves = []
        for i in range(4):
            y1 = i * shelf_height + margin_y
            y2 = (i + 1) * shelf_height - margin_y
            x1 = margin_x
            x2 = width - margin_x
            
            # Make sure regions are reasonable size
            if y2 > y1 and x2 > x1:
                shelves.append((x1, y1, x2, y2))
        
        return shelves
    
    def _check_shelf_has_objects(
        self, 
        shelf_region: Tuple[int, int, int, int], 
        objects: List[DetectedObject]
    ) -> bool:
        """Check if a shelf region contains any detected objects (excluding large objects like refrigerators/people)"""
        shelf_x1, shelf_y1, shelf_x2, shelf_y2 = shelf_region
        
        # Ignore large objects like refrigerators and people for shelf analysis
        ignored_labels = ['refrigerator', 'person', 'oven', 'microwave']
        
        for obj in objects:
            # Skip large background objects
            if obj.label in ignored_labels:
                continue
                
            box = obj.bounding_box
            
            # Check for overlap
            if (box.x1 < shelf_x2 and box.x2 > shelf_x1 and 
                box.y1 < shelf_y2 and box.y2 > shelf_y1):
                return True
        
        return False
    
    def create_annotated_image(
        self, 
        image: np.ndarray, 
        objects: List[DetectedObject], 
        empty_spaces: List[ShelfRegion]
    ) -> np.ndarray:
        """
        Create annotated image with detected objects and empty spaces
        
        Args:
            image: Original image
            objects: Detected objects
            empty_spaces: Empty shelf regions
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw detected objects (green boxes)
        for obj in objects:
            box = obj.bounding_box
            cv2.rectangle(
                annotated,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                (0, 255, 0),  # Green
                2
            )
            
            # Add label
            label = f"{obj.object_type.value}: {obj.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (int(box.x1), int(box.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Draw empty spaces (red boxes)
        for empty in empty_spaces:
            box = empty.bounding_box
            cv2.rectangle(
                annotated,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                (0, 0, 255),  # Red
                3
            )
            
            # Add "EMPTY" label
            cv2.putText(
                annotated,
                f"EMPTY SHELF {empty.shelf_level}",
                (int(box.x1) + 10, int(box.y1) + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return annotated
