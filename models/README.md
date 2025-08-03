# Models Directory

This directory contains all trained models used by the shelf analysis system.

## Directory Structure

```
models/
├── README.md                 # This file
├── yolo/                     # YOLO object detection models
│   └── shelf_analysis_custom.pt  # Custom YOLO model trained on shelf images
└── [future_models]/          # Other model types (LLMs, etc.)
```

## Model Details

### YOLO Models (`yolo/`)

#### `shelf_analysis_custom.pt`
- **Purpose**: Custom YOLO model trained specifically for retail shelf analysis
- **Training Data**: 85 annotated shelf images
- **Classes**: 5 classes (bottle, can, package, fruit, box)
- **Performance**: Better shelf detection accuracy than standard YOLO
- **Size**: ~5.9MB

#### Model Classes
1. **bottle** - Beverage bottles and containers
2. **can** - Canned products
3. **package** - Packaged goods and boxes
4. **fruit** - Fresh produce and fruits
5. **box** - Larger containers and boxes

## Usage

Models are automatically loaded by the respective services:
- YOLO models: Loaded by `YOLOService` in `src/shelf_analyzer/services/yolo_service.py`
- Custom model path: `models/yolo/shelf_analysis_custom.pt`

## Adding New Models

When adding new models:
1. Create appropriate subdirectory (e.g., `models/llm/` for language models)
2. Update service code to reference new model paths
3. Update this README with model details
4. Add model files to `.gitignore` if they're large (>100MB)

## Model Versioning

- Keep model files in version control for reproducibility
- Use descriptive filenames with version information
- Document model training parameters and performance metrics 