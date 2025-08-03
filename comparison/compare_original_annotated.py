#!/usr/bin/env python3
"""
Side-by-side comparison of original images with GPT-4O annotations
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json

# Configuration
IMAGES_DIR = Path("images")
ANNOTATIONS_DIR = Path("gpt_annotations")
OUTPUT_DIR = Path("comparison_visuals")

# Color scheme for different classes
CLASS_COLORS = {
    "bottle": (255, 0, 0),      # Red
    "can": (0, 255, 0),         # Green  
    "package": (0, 0, 255),     # Blue
    "fruit": (255, 255, 0),     # Yellow
    "box": (255, 0, 255),       # Magenta
}

def load_annotations(json_path: Path) -> list:
    """Load annotations from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def create_side_by_side_comparison(image_path: Path, annotations: list, output_path: Path):
    """Create side-by-side comparison of original and annotated image"""
    # Read original image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create annotated version
    annotated_image = image_rgb.copy()
    height, width = image.shape[:2]
    
    # Draw annotations on the annotated version
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
        color_rgb = tuple(c/255 for c in color)
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        label = f"{class_name}"
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Background rectangle for text
        cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(annotated_image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(image_rgb)
    ax1.set_title("Original Image", fontsize=16, weight='bold')
    ax1.axis('off')
    
    # Annotated image
    ax2.imshow(annotated_image)
    ax2.set_title(f"GPT-4O Annotations ({len(annotations)} objects)", fontsize=16, weight='bold')
    ax2.axis('off')
    
    # Add legend
    legend_elements = []
    for class_name, color in CLASS_COLORS.items():
        color_rgb = tuple(c/255 for c in color)
        legend_elements.append(patches.Patch(color=color_rgb, label=class_name))
    
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Add statistics
    class_counts = {}
    for ann in annotations:
        class_name = ann.get("class_name", "unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    stats_text = f"Objects detected:\n"
    for class_name, count in class_counts.items():
        stats_text += f"• {class_name}: {count}\n"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {output_path}")

def main():
    """Main function to create comparisons for all annotated images"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get all annotation files
    json_files = list(ANNOTATIONS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files")
    
    for json_file in json_files:
        image_name = json_file.stem
        image_path = IMAGES_DIR / f"{image_name}.jpg"
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        
        annotations = load_annotations(json_file)
        print(f"Creating comparison for {image_name}: {len(annotations)} annotations")
        
        output_path = OUTPUT_DIR / f"{image_name}_comparison.png"
        create_side_by_side_comparison(image_path, annotations, output_path)
    
    print(f"\n✅ All comparisons saved to {OUTPUT_DIR}")
    print("📊 Summary:")
    for json_file in json_files:
        image_name = json_file.stem
        annotations = load_annotations(json_file)
        class_counts = {}
        for ann in annotations:
            class_name = ann.get("class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"   {image_name}: {len(annotations)} objects - {dict(class_counts)}")

if __name__ == "__main__":
    main() 