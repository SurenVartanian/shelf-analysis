#!/usr/bin/env python3
"""
Custom YOLO Model Comparison Script

Compares:
1. Custom YOLO + LLM (/analyze-vision-custom)
2. Standard YOLO + LLM (/analyze-vision) 
3. Direct LLM (/analyze-vision-direct)

Uses the same 15 test images to evaluate performance differences.
"""

import asyncio
import aiohttp
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGES_DIR = Path("test_images")
RESULTS_DIR = Path("custom_comparison_results")
MODEL = "gpt-4o"  # Use GPT-4o for all tests

# Test methods
TEST_METHODS = {
    "custom_yolo": {
        "endpoint": "/analyze-vision-custom",
        "description": "Custom YOLO + LLM",
        "color": "🎯"
    },
    "standard_yolo": {
        "endpoint": "/analyze-vision", 
        "description": "Standard YOLO + LLM",
        "color": "📊"
    },
    "direct_llm": {
        "endpoint": "/analyze-vision-direct",
        "description": "Direct LLM (No YOLO)",
        "color": "🧠"
    }
}

class CustomYOLOComparator:
    def __init__(self):
        self.results = {}
        self.stats = {
            "total_images": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "processing_times": {},
            "object_counts": {},
            "yolo_contexts": {}
        }
    
    async def test_method(self, session: aiohttp.ClientSession, method: str, image_path: Path) -> Dict[str, Any]:
        """Test a specific method on an image"""
        method_config = TEST_METHODS[method]
        
        try:
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('image', f, filename=image_path.name, content_type='image/jpeg')
                data.add_field('model', MODEL)
                
                start_time = time.time()
                
                async with session.post(f"{API_BASE_URL}{method_config['endpoint']}", data=data) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "method": method,
                            "image": image_path.name,
                            "processing_time": processing_time,
                            "result": result,
                            "yolo_context": result.get('yolo_context', {}),
                            "total_objects": result.get('yolo_context', {}).get('total_objects', 0),
                            "object_types": result.get('yolo_context', {}).get('object_types', 'N/A'),
                            "shelves_count": len(result.get('shelves', [])),
                            "total_score": result.get('total_score', 0)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "method": method,
                            "image": image_path.name,
                            "error": f"HTTP {response.status}: {error_text}",
                            "processing_time": processing_time
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "method": method,
                "image": image_path.name,
                "error": str(e),
                "processing_time": 0
            }
    
    async def process_image(self, session: aiohttp.ClientSession, image_path: Path):
        """Process a single image with all methods"""
        print(f"\n🖼️  Processing: {image_path.name}")
        print("-" * 50)
        
        image_results = {}
        
        # Test all methods
        for method in TEST_METHODS.keys():
            method_config = TEST_METHODS[method]
            print(f"{method_config['color']} Testing {method_config['description']}...")
            
            result = await self.test_method(session, method, image_path)
            image_results[method] = result
            
            if result["success"]:
                print(f"   ✅ Success - {result['processing_time']:.2f}s")
                print(f"   📊 Objects: {result['total_objects']}, Shelves: {result['shelves_count']}")
                if result.get('object_types'):
                    print(f"   🏷️  Types: {result['object_types']}")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        return image_results
    
    async def run_comparison(self):
        """Run the full comparison"""
        print("🚀 Custom YOLO Model Comparison")
        print("=" * 60)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Model: {MODEL}")
        print(f"📁 Test images: {TEST_IMAGES_DIR}")
        print()
        
        # Find test images
        if not TEST_IMAGES_DIR.exists():
            print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
            return
        
        test_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
        if not test_images:
            print(f"❌ No test images found in {TEST_IMAGES_DIR}")
            return
        
        print(f"📸 Found {len(test_images)} test images")
        
        # Create results directory
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Clear old results
        for method in TEST_METHODS.keys():
            method_dir = RESULTS_DIR / method
            if method_dir.exists():
                import shutil
                shutil.rmtree(method_dir)
            method_dir.mkdir(exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            # Process each image
            for i, image_path in enumerate(test_images, 1):
                print(f"\n📊 Progress: {i}/{len(test_images)}")
                
                image_results = await self.process_image(session, image_path)
                self.results[image_path.name] = image_results
                
                # Save individual results
                for method, result in image_results.items():
                    if result["success"]:
                        result_file = RESULTS_DIR / method / f"{image_path.stem}.json"
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)
                
                # Small delay between requests
                await asyncio.sleep(1)
        
        # Generate summary report
        await self.generate_report()
    
    async def generate_report(self):
        """Generate comparison report"""
        print("\n📊 Generating Comparison Report")
        print("=" * 60)
        
        # Calculate statistics
        total_images = len(self.results)
        method_stats = {method: {"success": 0, "failed": 0, "times": [], "objects": []} for method in TEST_METHODS.keys()}
        
        for image_name, image_results in self.results.items():
            for method, result in image_results.items():
                if result["success"]:
                    method_stats[method]["success"] += 1
                    method_stats[method]["times"].append(result["processing_time"])
                    method_stats[method]["objects"].append(result["total_objects"])
                else:
                    method_stats[method]["failed"] += 1
        
        # Print summary
        print(f"\n📈 Summary (Total Images: {total_images})")
        print("-" * 40)
        
        for method, stats in method_stats.items():
            method_config = TEST_METHODS[method]
            success_rate = (stats["success"] / total_images) * 100 if total_images > 0 else 0
            
            print(f"\n{method_config['color']} {method_config['description']}")
            print(f"   ✅ Success: {stats['success']}/{total_images} ({success_rate:.1f}%)")
            print(f"   ❌ Failed: {stats['failed']}")
            
            if stats["times"]:
                avg_time = sum(stats["times"]) / len(stats["times"])
                avg_objects = sum(stats["objects"]) / len(stats["objects"])
                print(f"   ⏱️  Avg Time: {avg_time:.2f}s")
                print(f"   🎯 Avg Objects: {avg_objects:.1f}")
        
        # Save comprehensive report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "total_images": total_images,
            "method_stats": method_stats,
            "detailed_results": self.results
        }
        
        report_file = RESULTS_DIR / "comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("🎉 Comparison completed!")

async def main():
    """Main function"""
    comparator = CustomYOLOComparator()
    await comparator.run_comparison()

if __name__ == "__main__":
    print("🔍 Custom YOLO Model Comparison")
    print("Make sure your server is running on http://localhost:8000")
    print()
    
    asyncio.run(main()) 