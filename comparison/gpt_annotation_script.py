#!/usr/bin/env python3
"""
GPT-4O Vision Annotation Script
Uses GPT-4O to detect objects in images and generate YOLO training annotations
"""

import asyncio
import json
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple
import litellm
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("gpt_annotations")
CLASSES = ["bottle", "can", "package", "fruit", "box"]

class GPTAnnotator:
    def __init__(self, max_concurrent: int = 5, batch_size: int = 20, batch_delay: int = 30):
        self.class_to_id = {cls: i for i, cls in enumerate(CLASSES)}
        self.id_to_class = {i: cls for cls, i in self.class_to_id.items()}
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "total_objects": 0,
            "batches_completed": 0
        }
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _create_annotation_prompt(self) -> str:
        """Create the prompt for GPT-4O object detection"""
        return f"""
You are an expert in computer vision and retail shelf analysis. Your task is to detect objects in this shelf image and provide precise bounding box coordinates.

DETECT THESE OBJECT TYPES:
- bottle: Beverages, water, soda bottles
- can: Beer, soda, canned goods  
- package: Chips, candy, small snacks
- fruit: Individual fruits (apples, bananas, oranges)
- box: Larger packages (cereal, crackers)

INSTRUCTIONS:
1. Analyze the image carefully
2. Identify all objects that belong to the categories above
3. For each object, provide:
   - class_name: one of {CLASSES}
   - x_center: center X coordinate (0.0 to 1.0)
   - y_center: center Y coordinate (0.0 to 1.0) 
   - width: object width (0.0 to 1.0)
   - height: object height (0.0 to 1.0)

IMPORTANT - BOUNDING BOX PRECISION:
- Draw tight bounding boxes that closely follow the actual product boundaries
- The box should be just large enough to contain the entire product with minimal extra space
- Be precise with coordinates - avoid oversized or undersized boxes
- Only include objects you are confident about

RESPONSE FORMAT:
Return a JSON array of objects like this:
[
  {{"class_name": "bottle", "x_center": 0.25, "y_center": 0.3, "width": 0.1, "height": 0.2}},
  {{"class_name": "package", "x_center": 0.6, "y_center": 0.4, "width": 0.08, "height": 0.12}}
]

If no objects are detected, return an empty array [].
"""

    async def annotate_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """Annotate a single image using GPT-4O"""
        async with self.semaphore:  # Limit concurrent requests
            logger.info(f"Annotating {image_path.name}")
            
            try:
                # Encode image
                image_base64 = self._encode_image(image_path)
                
                # Create messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are a computer vision expert specializing in retail shelf analysis with precise object detection."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._create_annotation_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ]
                
                # Call GPT-4O
                response = await litellm.acompletion(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.1
                )
                
                # Parse response
                content = response.choices[0].message.content
                logger.info(f"GPT-4O response for {image_path.name}: {content[:100]}...")
                
                # Extract JSON from response
                try:
                    # Look for JSON array in the response
                    import re
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        annotations = json.loads(json_match.group())
                    else:
                        logger.warning(f"No JSON array found in response for {image_path.name}")
                        return []
                    
                    logger.info(f"Found {len(annotations)} annotations for {image_path.name}")
                    return annotations
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {image_path.name}: {e}")
                    return []
                    
            except Exception as e:
                logger.error(f"Annotation failed for {image_path.name}: {e}")
                return []
    
    def convert_to_yolo_format(self, annotations: List[Dict[str, Any]]) -> List[str]:
        """Convert GPT annotations to YOLO format"""
        yolo_lines = []
        
        for ann in annotations:
            class_name = ann.get("class_name")
            if class_name not in self.class_to_id:
                logger.warning(f"Unknown class: {class_name}")
                continue
                
            class_id = self.class_to_id[class_name]
            x_center = ann.get("x_center", 0.5)
            y_center = ann.get("y_center", 0.5)
            width = ann.get("width", 0.1)
            height = ann.get("height", 0.1)
            
            # Validate coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                logger.warning(f"Invalid coordinates for {class_name}: {x_center}, {y_center}, {width}, {height}")
                continue
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        return yolo_lines
    
    async def process_single_image(self, image_path: Path) -> bool:
        """Process a single image and return success status"""
        try:
            # Check if already processed
            label_path = OUTPUT_DIR / f"{image_path.stem}.txt"
            json_path = OUTPUT_DIR / f"{image_path.stem}.json"
            
            if label_path.exists() and json_path.exists():
                logger.info(f"Skipping {image_path.name} - already processed")
                return True
            
            # Annotate image
            annotations = await self.annotate_image(image_path)
            
            if annotations:
                # Convert to YOLO format
                yolo_lines = self.convert_to_yolo_format(annotations)
                
                # Save YOLO annotation
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                # Save JSON annotation for reference
                with open(json_path, 'w') as f:
                    json.dump(annotations, f, indent=2)
                
                self.stats["successful"] += 1
                self.stats["total_objects"] += len(annotations)
                logger.info(f"✅ Saved {len(yolo_lines)} annotations to {label_path}")
                return True
            else:
                logger.warning(f"⚠️ No annotations found for {image_path.name}")
                self.stats["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to process {image_path.name}: {e}")
            self.stats["failed"] += 1
            return False
        finally:
            self.stats["processed"] += 1
    
    async def process_batch(self, batch_images: List[Path]) -> None:
        """Process a batch of images"""
        logger.info(f"🔄 Processing batch of {len(batch_images)} images...")
        
        # Process images in batch concurrently
        tasks = []
        for image_path in batch_images:
            task = self.process_single_image(image_path)
            tasks.append(task)
        
        # Wait for all tasks in batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats["batches_completed"] += 1
        logger.info(f"✅ Batch {self.stats['batches_completed']} completed")
        
        # Add delay between batches to avoid rate limiting
        if self.batch_delay > 0:
            logger.info(f"⏳ Waiting {self.batch_delay} seconds before next batch...")
            await asyncio.sleep(self.batch_delay)
    
    async def process_images(self, max_images: int = None):
        """Process multiple images for annotation in batches"""
        # Create output directory
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [f for f in IMAGES_DIR.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.error(f"No images found in {IMAGES_DIR}")
            return
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        # Skip already processed images
        unprocessed_images = []
        for image_path in image_files:
            label_path = OUTPUT_DIR / f"{image_path.stem}.txt"
            json_path = OUTPUT_DIR / f"{image_path.stem}.json"
            if not (label_path.exists() and json_path.exists()):
                unprocessed_images.append(image_path)
        
        logger.info(f"🚀 Starting annotation of {len(unprocessed_images)} unprocessed images")
        logger.info(f"📊 Configuration: {self.max_concurrent} concurrent, {self.batch_size} per batch, {self.batch_delay}s delay")
        start_time = time.time()
        
        # Process images in batches
        for i in range(0, len(unprocessed_images), self.batch_size):
            batch = unprocessed_images[i:i + self.batch_size]
            await self.process_batch(batch)
        
        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"\n📊 ANNOTATION COMPLETE!")
        logger.info(f"⏱️  Total time: {duration:.1f} seconds")
        logger.info(f"📸 Images processed: {self.stats['processed']}")
        logger.info(f"✅ Successful: {self.stats['successful']}")
        logger.info(f"❌ Failed: {self.stats['failed']}")
        logger.info(f"🎯 Total objects detected: {self.stats['total_objects']}")
        if self.stats['successful'] > 0:
            logger.info(f"📈 Average objects per image: {self.stats['total_objects']/self.stats['successful']:.1f}")
        logger.info(f"⚡ Average time per image: {duration/self.stats['processed']:.1f} seconds")
        logger.info(f"📦 Batches completed: {self.stats['batches_completed']}")
        logger.info(f"📁 Results saved to {OUTPUT_DIR}")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-4O Vision Annotation Tool')
    parser.add_argument('--max-images', type=int, default=None, 
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--concurrent', type=int, default=3,
                       help='Number of concurrent requests (default: 3)')
    parser.add_argument('--batch-size', type=int, default=15,
                       help='Number of images per batch (default: 15)')
    parser.add_argument('--batch-delay', type=int, default=30,
                       help='Delay between batches in seconds (default: 30)')
    
    args = parser.parse_args()
    
    print(f"🔧 Configuration:")
    print(f"   Max images: {args.max_images or 'ALL'}")
    print(f"   Concurrent requests: {args.concurrent}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Batch delay: {args.batch_delay}s")
    print(f"   Classes: {CLASSES}")
    print()
    
    annotator = GPTAnnotator(
        max_concurrent=args.concurrent,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay
    )
    await annotator.process_images(args.max_images)

if __name__ == "__main__":
    asyncio.run(main()) 