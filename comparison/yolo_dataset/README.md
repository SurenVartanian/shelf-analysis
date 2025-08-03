# Custom Shelf Analysis Dataset

## Dataset Information
- **Total Images**: 85
- **Classes**: bottle, can, package, fruit, box
- **Class IDs**: {0: 'bottle', 1: 'can', 2: 'package', 3: 'fruit', 4: 'box'}

## Directory Structure
```
yolo_dataset/
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
cd yolo_dataset
python train_yolo.py
```

## Model Classes
0: bottle - Beverages, water, soda bottles
1: can - Beer, soda, canned goods
2: package - Chips, candy, small snacks
3: fruit - Individual fruits (apples, bananas)
4: box - Larger packages (cereal, crackers)
