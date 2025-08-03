#!/usr/bin/env python3
"""
Create ground truth JSON files for all images
"""

import json
from pathlib import Path

# Directories
IMAGES_DIR = Path("images")
GROUND_TRUTH_DIR = Path("ground_truth")

def create_ground_truth_files():
    """Create ground truth JSON files for all images"""
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [f for f in IMAGES_DIR.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images")
    
    # Create ground truth files for each image
    created_count = 0
    existing_count = 0
    
    for image_path in sorted(image_files):
        image_name = image_path.stem
        ground_truth_file = GROUND_TRUTH_DIR / f"{image_name}.json"
        
        if ground_truth_file.exists():
            existing_count += 1
            continue
        
        # Create new ground truth file
        ground_truth_data = {
            "expected_shelves": 0
        }
        
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)
        
        created_count += 1
        print(f"Created: {ground_truth_file}")
    
    print(f"\nSummary:")
    print(f"Created: {created_count} new ground truth files")
    print(f"Already existed: {existing_count} files")
    print(f"Total: {created_count + existing_count} ground truth files")

if __name__ == "__main__":
    create_ground_truth_files() 