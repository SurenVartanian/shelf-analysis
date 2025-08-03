#!/usr/bin/env python3
"""
Prepare YOLO Training Dataset
Organizes GPT-4O annotations into train/val/test splits for YOLO training
"""

import os
import shutil
import random
from pathlib import Path
import yaml

# Configuration
IMAGES_DIR = Path("images")
ANNOTATIONS_DIR = Path("gpt_annotations")
YOLO_DATASET_DIR = Path("yolo_dataset")
CLASSES = ["bottle", "can", "package", "fruit", "box"]

def create_yolo_dataset_structure():
    """Create YOLO dataset directory structure"""
    # Create main dataset directory
    YOLO_DATASET_DIR.mkdir(exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (YOLO_DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created YOLO dataset structure in {YOLO_DATASET_DIR}")

def get_annotation_files():
    """Get list of all annotation files"""
    annotation_files = []
    for json_file in ANNOTATIONS_DIR.glob("*.json"):
        if json_file.exists():
            annotation_files.append(json_file.stem)
    
    print(f"📊 Found {len(annotation_files)} annotation files")
    return annotation_files

def split_dataset(annotation_files, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split dataset into train/val/test sets"""
    # Shuffle files for random split
    random.shuffle(annotation_files)
    
    total_files = len(annotation_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    train_files = annotation_files[:train_count]
    val_files = annotation_files[train_count:train_count + val_count]
    test_files = annotation_files[train_count + val_count:]
    
    print(f"📈 Dataset split:")
    print(f"   Train: {len(train_files)} images ({len(train_files)/total_files*100:.1f}%)")
    print(f"   Val: {len(val_files)} images ({len(val_files)/total_files*100:.1f}%)")
    print(f"   Test: {len(test_files)} images ({len(test_files)/total_files*100:.1f}%)")
    
    return train_files, val_files, test_files

def copy_files_to_split(image_names, split_name):
    """Copy images and labels to specified split directory"""
    split_images_dir = YOLO_DATASET_DIR / split_name / 'images'
    split_labels_dir = YOLO_DATASET_DIR / split_name / 'labels'
    
    copied_count = 0
    for image_name in image_names:
        # Copy image
        image_src = IMAGES_DIR / f"{image_name}.jpg"
        image_dst = split_images_dir / f"{image_name}.jpg"
        
        # Copy label
        label_src = ANNOTATIONS_DIR / f"{image_name}.txt"
        label_dst = split_labels_dir / f"{image_name}.txt"
        
        if image_src.exists() and label_src.exists():
            shutil.copy2(image_src, image_dst)
            shutil.copy2(label_src, label_dst)
            copied_count += 1
        else:
            print(f"⚠️ Missing files for {image_name}")
    
    print(f"✅ Copied {copied_count} files to {split_name} split")
    return copied_count

def create_yolo_config():
    """Create YOLO configuration file"""
    config = {
        'path': str(YOLO_DATASET_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    config_path = YOLO_DATASET_DIR / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created YOLO config: {config_path}")
    return config_path

def create_training_script():
    """Create a training script for the custom YOLO model"""
    training_script = f"""#!/usr/bin/env python3
\"\"\"
# YOLO Training Script for Custom Shelf Analysis Model
\"\"\"
from ultralytics import YOLO

# Load a base model
model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano

# Train the model on our custom dataset
results = model.train(
    data='{YOLO_DATASET_DIR}/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='shelf_analysis_custom',
    patience=20,
    save=True,
    device='auto'
)

print("Training completed!")
print(f"Results saved in: runs/detect/shelf_analysis_custom/")
"""
    
    script_path = YOLO_DATASET_DIR / 'train_yolo.py'
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    print(f"✅ Created training script: {script_path}")

def create_dataset_info():
    """Create dataset information file"""
    info = f"""# Custom Shelf Analysis Dataset

## Dataset Information
- **Total Images**: {len(get_annotation_files())}
- **Classes**: {', '.join(CLASSES)}
- **Class IDs**: {dict(enumerate(CLASSES))}

## Directory Structure
```
{YOLO_DATASET_DIR}/
├── train/
│   ├── images/     # Training images
│   └── labels/     # Training labels
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
├── test/
│   ├── images/     # Test images
│   └── labels/     # Test labels
├── dataset.yaml    # YOLO configuration
└── train_yolo.py   # Training script
```

## Training Command
```bash
cd {YOLO_DATASET_DIR}
python train_yolo.py
```

## Model Classes
0: bottle - Beverages, water, soda bottles
1: can - Beer, soda, canned goods
2: package - Chips, candy, small snacks
3: fruit - Individual fruits (apples, bananas)
4: box - Larger packages (cereal, crackers)
"""
    
    info_path = YOLO_DATASET_DIR / 'README.md'
    with open(info_path, 'w') as f:
        f.write(info)
    
    print(f"✅ Created dataset info: {info_path}")

def main():
    """Main function to prepare the YOLO dataset"""
    print("🚀 Preparing YOLO Training Dataset")
    print("=" * 50)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create directory structure
    create_yolo_dataset_structure()
    
    # Get annotation files
    annotation_files = get_annotation_files()
    
    if not annotation_files:
        print("❌ No annotation files found!")
        return
    
    # Split dataset
    train_files, val_files, test_files = split_dataset(annotation_files)
    
    # Copy files to splits
    print("\n📁 Copying files to dataset splits...")
    train_count = copy_files_to_split(train_files, 'train')
    val_count = copy_files_to_split(val_files, 'val')
    test_count = copy_files_to_split(test_files, 'test')
    
    # Create YOLO configuration
    config_path = create_yolo_config()
    
    # Create training script
    create_training_script()
    
    # Create dataset info
    create_dataset_info()
    
    print("\n🎉 YOLO Dataset Preparation Complete!")
    print("=" * 50)
    print(f"📊 Dataset Summary:")
    print(f"   Total images: {len(annotation_files)}")
    print(f"   Train images: {train_count}")
    print(f"   Val images: {val_count}")
    print(f"   Test images: {test_count}")
    print(f"   Classes: {len(CLASSES)} ({', '.join(CLASSES)})")
    print(f"\n📁 Dataset location: {YOLO_DATASET_DIR}")
    print(f"⚙️  Config file: {config_path}")
    print(f"🎯 Training script: {YOLO_DATASET_DIR}/train_yolo.py")
    print(f"\n🚀 To start training:")
    print(f"   cd {YOLO_DATASET_DIR}")
    print(f"   python train_yolo.py")

if __name__ == "__main__":
    main() 