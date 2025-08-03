#!/usr/bin/env python3
"""
Visualization script for GPT-4O annotations
Overlays bounding boxes on original images for manual quality checking
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import argparse

# Configuration
IMAGES_DIR = Path("images")
ANNOTATIONS_DIR = Path("gpt_annotations")
OUTPUT_DIR = Path("visualization")

# Color scheme for different classes
CLASS_COLORS = {
    "bottle": (255, 0, 0),      # Red
    "can": (0, 255, 0),         # Green  
    "package": (0, 0, 255),     # Blue
    "fruit": (255, 255, 0),     # Yellow
    "box": (255, 0, 255),       # Magenta
}

CLASS_NAMES = ["bottle", "can", "package", "fruit", "box"]

def load_annotations(json_path: Path) -> list:
    """Load annotations from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def draw_annotations_opencv(image_path: Path, annotations: list, output_path: Path):
    """Draw annotations using OpenCV"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Draw each annotation
    for ann in annotations:
        class_name = ann.get("class_name", "unknown")
        x_center = ann.get("x_center", 0.5)
        y_center = ann.get("y_center", 0.5)
        w = ann.get("width", 0.1)
        h = ann.get("height", 0.1)
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # Get color for this class
        color = CLASS_COLORS.get(class_name, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save annotated image
    cv2.imwrite(str(output_path), image)
    print(f"Saved OpenCV visualization: {output_path}")

def draw_annotations_matplotlib(image_path: Path, annotations: list, output_path: Path):
    """Draw annotations using Matplotlib (nicer looking)"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    height, width = image.shape[:2]
    
    # Draw each annotation
    for ann in annotations:
        class_name = ann.get("class_name", "unknown")
        x_center = ann.get("x_center", 0.5)
        y_center = ann.get("y_center", 0.5)
        w = ann.get("width", 0.1)
        h = ann.get("height", 0.1)
        
        # Convert normalized coordinates to pixel coordinates
        x1 = (x_center - w/2) * width
        y1 = (y_center - h/2) * height
        rect_width = w * width
        rect_height = h * height
        
        # Get color for this class
        color = CLASS_COLORS.get(class_name, (0.5, 0.5, 0.5))
        color_rgb = tuple(c/255 for c in color)
        
        # Create rectangle
        rect = Rectangle((x1, y1), rect_width, rect_height, 
                        linewidth=2, edgecolor=color_rgb, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1 - 5, class_name, fontsize=10, color=color_rgb, 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add legend
    legend_elements = []
    for class_name, color in CLASS_COLORS.items():
        color_rgb = tuple(c/255 for c in color)
        legend_elements.append(patches.Patch(color=color_rgb, label=class_name))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_title(f"GPT-4O Annotations: {image_path.name}", fontsize=14, weight='bold')
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Matplotlib visualization: {output_path}")

def create_summary_report(annotations_dir: Path):
    """Create a summary report of all annotations"""
    summary = {}
    
    for json_file in annotations_dir.glob("*.json"):
        image_name = json_file.stem
        annotations = load_annotations(json_file)
        
        # Count by class
        class_counts = {}
        for ann in annotations:
            class_name = ann.get("class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary[image_name] = {
            "total_objects": len(annotations),
            "by_class": class_counts
        }
    
    # Print summary
    print("\n" + "="*60)
    print("GPT-4O ANNOTATION SUMMARY")
    print("="*60)
    
    total_objects = 0
    total_by_class = {}
    
    for image_name, data in summary.items():
        print(f"\n📸 {image_name}:")
        print(f"   Total objects: {data['total_objects']}")
        for class_name, count in data['by_class'].items():
            print(f"   - {class_name}: {count}")
            total_by_class[class_name] = total_by_class.get(class_name, 0) + count
        total_objects += data['total_objects']
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   Total images: {len(summary)}")
    print(f"   Total objects: {total_objects}")
    print(f"   Average objects per image: {total_objects/len(summary):.1f}")
    print(f"\n   By class:")
    for class_name, count in total_by_class.items():
        print(f"   - {class_name}: {count} ({count/total_objects*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Visualize GPT-4O annotations')
    parser.add_argument('--image', type=str, help='Specific image to visualize (without extension)')
    parser.add_argument('--method', choices=['opencv', 'matplotlib', 'both'], default='both',
                       help='Visualization method')
    parser.add_argument('--summary', action='store_true', help='Show annotation summary')
    
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if args.summary:
        create_summary_report(ANNOTATIONS_DIR)
        return
    
    if args.image:
        # Visualize specific image
        image_name = args.image
        image_path = IMAGES_DIR / f"{image_name}.jpg"
        json_path = ANNOTATIONS_DIR / f"{image_name}.json"
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return
        if not json_path.exists():
            print(f"Annotations not found: {json_path}")
            return
        
        annotations = load_annotations(json_path)
        print(f"Loaded {len(annotations)} annotations for {image_name}")
        
        if args.method in ['opencv', 'both']:
            output_path = OUTPUT_DIR / f"{image_name}_opencv.jpg"
            draw_annotations_opencv(image_path, annotations, output_path)
        
        if args.method in ['matplotlib', 'both']:
            output_path = OUTPUT_DIR / f"{image_name}_matplotlib.png"
            draw_annotations_matplotlib(image_path, annotations, output_path)
    
    else:
        # Visualize all annotated images
        json_files = list(ANNOTATIONS_DIR.glob("*.json"))
        print(f"Found {len(json_files)} annotation files")
        
        for json_file in json_files:
            image_name = json_file.stem
            image_path = IMAGES_DIR / f"{image_name}.jpg"
            
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            
            annotations = load_annotations(json_file)
            print(f"Processing {image_name}: {len(annotations)} annotations")
            
            if args.method in ['opencv', 'both']:
                output_path = OUTPUT_DIR / f"{image_name}_opencv.jpg"
                draw_annotations_opencv(image_path, annotations, output_path)
            
            if args.method in ['matplotlib', 'both']:
                output_path = OUTPUT_DIR / f"{image_name}_matplotlib.png"
                draw_annotations_matplotlib(image_path, annotations, output_path)

if __name__ == "__main__":
    main() 