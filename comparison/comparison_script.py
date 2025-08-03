#!/usr/bin/env python3
"""
Shelf Analysis Comparison Script
Compares YOLO-based vs Direct LLM analysis and uses OpenAI for judgment
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import base64
from datetime import datetime
import numpy as np
import cv2

# Add the src directory to the path so we can import our services
sys.path.append(str(Path(__file__).parent.parent / "src"))

from shelf_analyzer.services.litellm_vision_service import (
    LiteLLMVisionService,
    VisionAnalysisConfig,
    VisionModel,
    VisionAnalysisResult
)
from shelf_analyzer.services.yolo_service import YOLOService
import litellm

# Configuration
COMPARISON_DIR = Path(__file__).parent
IMAGES_DIR = COMPARISON_DIR / "images"
GROUND_TRUTH_DIR = COMPARISON_DIR / "ground_truth"
RESULTS_DIR = COMPARISON_DIR / "results"
YOLO_RESULTS_DIR = RESULTS_DIR / "yolo"
DIRECT_RESULTS_DIR = RESULTS_DIR / "direct"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"

# Model configuration
MODEL_NAME = "gemini-flash-lite"  # Change this as needed
YOLO_MODEL = "gemini-flash-lite"  # YOLO-based analysis

# Batch processing configuration
BATCH_SIZE = 5  # Process 5 images at a time
MAX_IMAGES = None  # Set to a number to limit total images, None for all

class ComparisonRunner:
    def __init__(self):
        self.yolo_service = None
        self.vision_service = None
    
    def calculate_f1_metrics(self, ground_truth_shelves: int, predicted_shelves: int) -> Dict[str, float]:
        """Calculate F1 scores for shelf detection only"""
        
        if ground_truth_shelves == 0 and predicted_shelves == 0:
            return {
                "precision": 1.0, 
                "recall": 1.0, 
                "f1_score": 1.0, 
                "accuracy": 1.0,
                "absolute_error": 0
            }
        
        # Treat each count as binary classification
        true_positives = min(ground_truth_shelves, predicted_shelves)
        false_positives = max(0, predicted_shelves - ground_truth_shelves)
        false_negatives = max(0, ground_truth_shelves - predicted_shelves)
        
        precision = true_positives / max(predicted_shelves, 1)
        recall = true_positives / max(ground_truth_shelves, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        accuracy = 1.0 if ground_truth_shelves == predicted_shelves else 0.0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1_score": f1_score,
            "accuracy": accuracy,
            "absolute_error": abs(ground_truth_shelves - predicted_shelves)
        }
        
    async def initialize_services(self):
        """Initialize all required services"""
        print("🔄 Initializing services...")
        
        # Initialize YOLO service
        self.yolo_service = YOLOService()
        await self.yolo_service.initialize()
        print("✅ YOLO service initialized")
        
        # Initialize vision service for both analyses
        self.vision_service = LiteLLMVisionService(
            VisionAnalysisConfig(model=VisionModel.GEMINI_FLASH_LITE)
        )
        await self.vision_service.initialize()
        print("✅ Vision service initialized")
        
    def get_ground_truth(self, image_name: str) -> Optional[Dict[str, Any]]:
        """Load ground truth for an image if it exists"""
        ground_truth_file = GROUND_TRUTH_DIR / f"{image_name}.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                return json.load(f)
        return None
    
    async def run_yolo_analysis(self, image_path: Path) -> Dict[str, Any]:
        """Run Custom YOLO + Gemini analysis by calling the /analyze-vision-custom endpoint"""
        print(f"🎯 Running Custom YOLO + Gemini analysis on {image_path.name}...")
        
        # Call the custom YOLO endpoint with Gemini
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('image', f, filename=image_path.name, content_type='image/jpeg')
                data.add_field('model', 'gemini-flash-lite')  # Use Gemini, not GPT-4o!
                
                start_time = time.time()
                async with session.post('http://localhost:8000/analyze-vision-custom', data=data) as response:
                    result = await response.json()
                    processing_time = (time.time() - start_time) * 1000
                    
                    result['processing_time_ms'] = processing_time
                    result['analysis_method'] = 'custom_yolo_gemini'
                    return result
    
    async def run_direct_analysis(self, image_path: Path) -> Dict[str, Any]:
        """Run direct GPT-4o analysis by calling the /analyze-vision-direct endpoint"""
        print(f"🎯 Running direct GPT-4o analysis on {image_path.name}...")
        
        # Call the existing /analyze-vision-direct endpoint with GPT-4o
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('image', f, filename=image_path.name, content_type='image/jpeg')
                data.add_field('model', 'gpt-4o')  # Use GPT-4o for direct analysis
                
                start_time = time.time()
                async with session.post('http://localhost:8000/analyze-vision-direct', data=data) as response:
                    result = await response.json()
                    processing_time = (time.time() - start_time) * 1000
                    
                    result['processing_time_ms'] = processing_time
                    result['analysis_method'] = 'direct_gpt4o'
                    return result
    
    async def get_openai_judgment(self, image_path: Path, ground_truth: Dict[str, Any], 
                                 yolo_result: Dict[str, Any], direct_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get OpenAI's judgment comparing the two approaches"""
        print(f"🤖 Getting OpenAI judgment for {image_path.name}...")
        
        # Prepare the comparison prompt
        prompt = self._create_comparison_prompt(ground_truth, yolo_result, direct_result)
        
        # Convert image to base64 for OpenAI
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are an expert in computer vision and retail analytics. Compare two shelf analysis results and provide objective judgment."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        try:
            # Use OpenAI GPT-4 Vision for judgment
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            judgment_text = response.choices[0].message.content
            
            # Try to parse structured judgment if possible
            try:
                # Look for JSON-like structure in the response
                import re
                json_match = re.search(r'\{.*\}', judgment_text, re.DOTALL)
                if json_match:
                    judgment_data = json.loads(json_match.group())
                else:
                    judgment_data = {"raw_judgment": judgment_text}
            except:
                judgment_data = {"raw_judgment": judgment_text}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_path.name,
                "judgment": judgment_data,
                "raw_response": judgment_text
            }
            
        except Exception as e:
            print(f"❌ Error getting OpenAI judgment: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_path.name,
                "error": str(e),
                "raw_response": "Failed to get judgment"
            }
    
    def _create_comparison_prompt(self, ground_truth: Dict[str, Any], 
                                 yolo_result: Dict[str, Any], 
                                 direct_result: Dict[str, Any]) -> str:
        """Create the comparison prompt for OpenAI"""
        
        # Extract key metrics
        yolo_shelves = len(yolo_result.get('shelves', []))
        yolo_products = sum(len(shelf.get('products', [])) for shelf in yolo_result.get('shelves', []))
        yolo_time = yolo_result.get('processing_time_ms', 0)
        
        direct_shelves = len(direct_result.get('shelves', []))
        direct_products = sum(len(shelf.get('products', [])) for shelf in direct_result.get('shelves', []))
        direct_time = direct_result.get('processing_time_ms', 0)
        
        # Build ground truth section dynamically
        ground_truth_section = "GROUND TRUTH:\n"
        if 'expected_shelves' in ground_truth:
            ground_truth_section += f"- Expected shelves: {ground_truth['expected_shelves']}\n"
        if 'expected_products' in ground_truth:
            ground_truth_section += f"- Expected products: {ground_truth['expected_products']}\n"
        if 'notes' in ground_truth:
            ground_truth_section += f"- Notes: {ground_truth['notes']}\n"
        if 'image_quality' in ground_truth:
            ground_truth_section += f"- Image quality: {ground_truth['image_quality']}\n"
        if 'challenges' in ground_truth:
            ground_truth_section += f"- Challenges: {ground_truth['challenges']}\n"
        
        # If no ground truth fields provided, use a generic message
        if len(ground_truth) == 0:
            ground_truth_section = "GROUND TRUTH: No specific ground truth provided. Compare based on general accuracy and quality.\n"
        
        prompt = f"""
Compare these two shelf analysis results{f" against the ground truth" if len(ground_truth) > 0 else ""}:

{ground_truth_section}
ANALYSIS A (YOLO-based preprocessing):
- Shelves detected: {yolo_shelves}
- Products detected: {yolo_products}
- Processing time: {yolo_time:.0f}ms
- Method: YOLO cropping + LLM analysis

ANALYSIS B (Direct LLM):
- Shelves detected: {direct_shelves}
- Products detected: {direct_products}
- Processing time: {direct_time:.0f}ms
- Method: Direct LLM analysis

Please provide a structured judgment in JSON format:

{{
    "yolo_accuracy": {{
        "shelf_count_score": 0-10,
        "product_count_score": 0-10,
        "overall_quality_score": 0-10,
        "comments": "explanation"
    }},
    "direct_accuracy": {{
        "shelf_count_score": 0-10,
        "product_count_score": 0-10,
        "overall_quality_score": 0-10,
        "comments": "explanation"
    }},
    "winner": "yolo" or "direct" or "tie",
    "reasoning": "detailed explanation of which approach is better and why",
    "recommendation": "which approach should be used for this type of image"
}}

Focus on accuracy and quality of the analysis results.
"""
        return prompt
    
    def save_result(self, result: Dict[str, Any], output_dir: Path, filename: str):
        """Save result to JSON file"""
        output_file = output_dir / f"{filename}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Saved result to {output_file}")
    
    async def process_image(self, image_path: Path):
        """Process a single image through the comparison pipeline"""
        image_name = image_path.stem
        print(f"\n{'='*60}")
        print(f"📸 Processing: {image_name}")
        print(f"{'='*60}")
        
        # Check if we already have results
        yolo_result_file = YOLO_RESULTS_DIR / f"{image_name}.json"
        direct_result_file = DIRECT_RESULTS_DIR / f"{image_name}.json"
        judgment_file = JUDGMENTS_DIR / f"{image_name}.json"
        
        # Run analyses if needed
        yolo_result = None
        direct_result = None
        
        if not yolo_result_file.exists():
            yolo_result = await self.run_yolo_analysis(image_path)
            self.save_result(yolo_result, YOLO_RESULTS_DIR, image_name)
        else:
            print(f"📁 Using existing YOLO result for {image_name}")
            with open(yolo_result_file, 'r') as f:
                yolo_result = json.load(f)
        
        if not direct_result_file.exists():
            direct_result = await self.run_direct_analysis(image_path)
            self.save_result(direct_result, DIRECT_RESULTS_DIR, image_name)
        else:
            print(f"📁 Using existing direct result for {image_name}")
            with open(direct_result_file, 'r') as f:
                direct_result = json.load(f)
        
        # Get ground truth
        ground_truth = self.get_ground_truth(image_name)
        if not ground_truth:
            print(f"⚠️  No ground truth found for {image_name}, skipping judgment")
            return
        
        # Calculate F1 metrics if expected_shelves is provided
        if 'expected_shelves' in ground_truth:
            expected_shelves = ground_truth['expected_shelves']
            yolo_shelves = len(yolo_result.get('shelves', []))
            direct_shelves = len(direct_result.get('shelves', []))
            
            yolo_f1 = self.calculate_f1_metrics(expected_shelves, yolo_shelves)
            direct_f1 = self.calculate_f1_metrics(expected_shelves, direct_shelves)
            
            print(f"📊 F1 Metrics for {image_name}:")
            print(f"   Ground Truth: {expected_shelves} shelves")
            print(f"   YOLO: {yolo_shelves} shelves (F1: {yolo_f1['f1_score']:.3f}, Error: {yolo_f1['absolute_error']})")
            print(f"   Direct: {direct_shelves} shelves (F1: {direct_f1['f1_score']:.3f}, Error: {direct_f1['absolute_error']})")
            
            # Add F1 metrics to results for later analysis
            yolo_result['f1_metrics'] = yolo_f1
            direct_result['f1_metrics'] = direct_f1
            
            # Save updated results with F1 metrics
            self.save_result(yolo_result, YOLO_RESULTS_DIR, image_name)
            self.save_result(direct_result, DIRECT_RESULTS_DIR, image_name)
        
        # Get OpenAI judgment if needed
        if not judgment_file.exists():
            judgment = await self.get_openai_judgment(image_path, ground_truth, yolo_result, direct_result)
            self.save_result(judgment, JUDGMENTS_DIR, image_name)
        else:
            print(f"📁 Using existing judgment for {image_name}")
    
    async def run_comparison(self):
        """Run the full comparison pipeline"""
        print("🚀 Starting Shelf Analysis Comparison")
        print(f"📁 Images directory: {IMAGES_DIR}")
        print(f"📁 Results directory: {RESULTS_DIR}")
        print(f"⚙️  Batch size: {BATCH_SIZE}")
        if MAX_IMAGES:
            print(f"⚙️  Max images: {MAX_IMAGES}")
        
        # Ensure directories exist
        for dir_path in [IMAGES_DIR, GROUND_TRUTH_DIR, YOLO_RESULTS_DIR, DIRECT_RESULTS_DIR, JUDGMENTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        await self.initialize_services()
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [f for f in IMAGES_DIR.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No images found in {IMAGES_DIR}")
            print(f"Please add some images to {IMAGES_DIR}")
            return
        
        # Sort images and apply max limit
        image_files = sorted(image_files)
        if MAX_IMAGES:
            image_files = image_files[:MAX_IMAGES]
        
        print(f"📸 Found {len(image_files)} images to process")
        
        # Process images in batches
        total_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
        processed_count = 0
        error_count = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(image_files))
            batch_files = image_files[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"📦 BATCH {batch_num + 1}/{total_batches} ({len(batch_files)} images)")
            print(f"📊 Progress: {processed_count}/{len(image_files)} ({processed_count/len(image_files)*100:.1f}%)")
            print(f"{'='*60}")
            
            # Process images in parallel within the batch
            print(f"🚀 Processing {len(batch_files)} images in parallel...")
            tasks = []
            for image_path in batch_files:
                task = self.process_image(image_path)
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful and failed results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Error processing {batch_files[i].name}: {result}")
                    error_count += 1
                else:
                    processed_count += 1
            
            # Batch summary
            print(f"\n✅ Batch {batch_num + 1} complete!")
            print(f"📊 Total processed: {processed_count}/{len(image_files)}")
            print(f"❌ Total errors: {error_count}")
            
            # Optional: Add a small delay between batches to be nice to the API
            if batch_num < total_batches - 1:
                print("⏳ Waiting 2 seconds before next batch...")
                await asyncio.sleep(2)
        
        print(f"\n✅ Comparison complete! Check results in {RESULTS_DIR}")
        print(f"📊 Final stats: {processed_count} processed, {error_count} errors")
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of the comparison results"""
        print(f"\n{'='*60}")
        print("📊 CUSTOM YOLO + GEMINI vs DIRECT GPT-4O COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Count results
        yolo_results = len(list(YOLO_RESULTS_DIR.glob("*.json")))
        direct_results = len(list(DIRECT_RESULTS_DIR.glob("*.json")))
        
        print(f"🎯 Custom YOLO + Gemini analyses: {yolo_results}")
        print(f"🧠 Direct GPT-4o analyses: {direct_results}")
        print(f"📊 Comparison: Custom YOLO + Gemini Flash Lite vs Direct GPT-4o")
        
        # Custom YOLO vs Direct LLM comparison
        print(f"\n🎯 CUSTOM YOLO vs DIRECT LLM ANALYSIS:")
        print(f"Comparing our custom-trained YOLO model (85 shelf images) vs direct GPT-4o analysis")
        
        # Analyze F1 metrics if available
        print(f"\n🎯 F1 METRICS ANALYSIS:")
        custom_yolo_f1_scores = []
        direct_f1_scores = []
        custom_yolo_errors = []
        direct_errors = []
        
        for yolo_file in YOLO_RESULTS_DIR.glob("*.json"):
            image_name = yolo_file.stem
            direct_file = DIRECT_RESULTS_DIR / f"{image_name}.json"
            
            if direct_file.exists():
                with open(yolo_file, 'r') as f:
                    yolo_result = json.load(f)
                with open(direct_file, 'r') as f:
                    direct_result = json.load(f)
                
                if 'f1_metrics' in yolo_result and 'f1_metrics' in direct_result:
                    custom_yolo_f1_scores.append(yolo_result['f1_metrics']['f1_score'])
                    direct_f1_scores.append(direct_result['f1_metrics']['f1_score'])
                    custom_yolo_errors.append(yolo_result['f1_metrics']['absolute_error'])
                    direct_errors.append(direct_result['f1_metrics']['absolute_error'])
        
        if custom_yolo_f1_scores:
            avg_custom_yolo_f1 = sum(custom_yolo_f1_scores) / len(custom_yolo_f1_scores)
            avg_direct_f1 = sum(direct_f1_scores) / len(direct_f1_scores)
            avg_custom_yolo_error = sum(custom_yolo_errors) / len(custom_yolo_errors)
            avg_direct_error = sum(direct_errors) / len(direct_errors)
            
            print(f"Average Custom YOLO + Gemini F1: {avg_custom_yolo_f1:.3f}")
            print(f"Average Direct GPT-4o F1: {avg_direct_f1:.3f}")
            print(f"F1 Difference: {avg_custom_yolo_f1 - avg_direct_f1:+.3f}")
            print(f"Average Custom YOLO + Gemini Error: {avg_custom_yolo_error:.1f} shelves")
            print(f"Average Direct GPT-4o Error: {avg_direct_error:.1f} shelves")
            
            # Count F1 wins
            custom_yolo_f1_wins = sum(1 for y, d in zip(custom_yolo_f1_scores, direct_f1_scores) if y > d)
            direct_f1_wins = sum(1 for y, d in zip(custom_yolo_f1_scores, direct_f1_scores) if d > y)
            f1_ties = sum(1 for y, d in zip(custom_yolo_f1_scores, direct_f1_scores) if abs(y - d) < 0.001)
            
            print(f"Custom YOLO + Gemini F1 wins: {custom_yolo_f1_wins}")
            print(f"Direct GPT-4o F1 wins: {direct_f1_wins}")
            print(f"F1 ties: {f1_ties}")
        else:
            print("No F1 metrics available (missing ground truth or results)")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shelf Analysis Comparison Tool')
    parser.add_argument('--batch-size', type=int, default=5, 
                       help='Number of images to process in each batch (default: 5)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process only first batch)')
    
    args = parser.parse_args()
    
    # Update configuration based on command line arguments
    global BATCH_SIZE, MAX_IMAGES
    BATCH_SIZE = args.batch_size
    if args.test:
        MAX_IMAGES = BATCH_SIZE
    else:
        MAX_IMAGES = args.max_images
    
    print(f"🔧 Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max images: {MAX_IMAGES or 'all'}")
    print(f"   Test mode: {args.test}")
    print()
    
    runner = ComparisonRunner()
    await runner.run_comparison()

if __name__ == "__main__":
    asyncio.run(main()) 