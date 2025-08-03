# 🎯 Shelf Analysis API

A FastAPI-based service for analyzing retail shelf images using YOLO object detection and LLM-powered analysis, with comprehensive comparison tools.

## 🚀 Features

- **YOLO Object Detection**: Detect products on shelves using YOLOv8
- **Custom YOLO Model**: Trained specifically on 85 shelf images
- **Empty Space Detection**: Identify and mark empty shelf areas
- **LLM Analysis**: Detailed analysis using GPT-4o or Gemini Flash Lite
- **FastAPI**: Modern, fast web API with automatic documentation
- **Async Processing**: Efficient handling of multiple requests
- **Comparison Tools**: Comprehensive analysis of different approaches

## 🏆 Key Results

### **Custom YOLO + Gemini vs Direct GPT-4o**
- **3-6x faster** processing (5-10s vs 30s)
- **5-10x cheaper** per image
- **Better shelf detection** (F1: 0.945 vs 0.854)
- **Competitive accuracy** (7.3/10 vs 7.3/10)

**Recommendation**: Use Custom YOLO + Gemini for production applications.

See `comparison/FINAL_COMPARISON_REPORT.md` for detailed analysis.

## 🛠️ Quick Start

### **Option 1: Docker (Recommended)**

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/SurenVartanian/shelf-analysis.git
   cd shelf-analysis
   ```

2. **Set Environment Variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Test the API**:
   - Visit `http://localhost:8000/docs` for interactive API documentation
   - Visit `http://localhost:8000` for the web interface

### **Option 2: Local Development**

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Server**:
   ```bash
   uvicorn src.shelf_analyzer.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Test the API**:
   - Visit `http://localhost:8000/docs` for interactive API documentation
   - Upload an image to `/analyze-vision` endpoint

## 📊 API Endpoints

### **Main Analysis Endpoints**
- `POST /analyze-vision` - Standard YOLO + LLM analysis
- `POST /analyze-vision-custom` - Custom YOLO + LLM analysis
- `POST /analyze-vision-direct` - Direct LLM analysis (no YOLO)
- `POST /analyze-vision-stream` - Streaming versions of above

### **Utility Endpoints**
- `GET /` - Health check
- `GET /health` - Detailed service status
- `GET /api` - API information

## 🎯 Usage Examples

### **Web Interface**
Visit `http://localhost:8000` for the interactive web interface with YOLO model selection.

### **API Usage**
```bash
# Custom YOLO + Gemini (recommended)
curl -X POST "http://localhost:8000/analyze-vision-custom" \
  -F "image=@shelf_image.jpg" \
  -F "model=gemini-flash-lite"

# Direct GPT-4o (high accuracy)
curl -X POST "http://localhost:8000/analyze-vision-direct" \
  -F "image=@shelf_image.jpg" \
  -F "model=gpt-4o"

# Standard YOLO + LLM
curl -X POST "http://localhost:8000/analyze-vision" \
  -F "image=@shelf_image.jpg" \
  -F "model=gpt-4o"
```

## 📈 Comparison Tools

### **Run Performance Comparison**
```bash
cd comparison
python comparison_script.py --batch-size 5 --max-images 15
```

### **Three-Way Comparison**
```bash
cd comparison
python custom_yolo_comparison.py
```

### **Analyze Results**
```bash
cd comparison
python analyze_judgments.py
```

See `comparison/README.md` for detailed comparison documentation.

## 🏗️ Architecture

### **Services**
- **YOLOService**: Object detection with custom/standard YOLO models
- **ShelfCropperService**: Shelf region detection and cropping
- **LiteLLMVisionService**: LLM analysis with multiple model support
- **ImageService**: Image processing utilities

### **Models**
- **Custom YOLO**: Trained on 85 shelf images (5 classes)
- **Standard YOLO**: YOLOv8n with COCO dataset
- **LLM models**: Configured via LiteLLM

## 📁 Project Structure

```
shelf-analysis/
├── src/shelf_analyzer/
│   ├── main.py                 # FastAPI application
│   ├── models/                 # Pydantic models
│   └── services/               # Core services
├── comparison/                 # Comparison tools and results
│   ├── comparison_script.py    # Main comparison
│   ├── custom_yolo_comparison.py # Three-way comparison
│   ├── FINAL_COMPARISON_REPORT.md # Final results
│   └── results/                # Comparison results (gitignored)
├── static/                     # Web interface
├── tests/                      # Test suite
├── models/                     # ML models
│   └── yolo/                   # YOLO models
│       └── shelf_analysis_custom.pt # Custom trained model
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker orchestration
├── docker-start.sh             # Docker entrypoint script
├── .dockerignore               # Docker build exclusions
├── DOCKER.md                   # Docker documentation
├── requirements.txt            # Python dependencies
├── env.example                 # Environment variables template
└── README.md                   # This file
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your_openai_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Optional
LITELLM_LOG=DEBUG
```

### **Model Configuration**
- Custom YOLO model: `models/yolo/shelf_analysis_custom.pt`
- Standard YOLO model: Downloaded automatically
- LLM models: Configured via LiteLLM

## 🐳 Docker Support

### **Docker Features**
- **Multi-stage build** for optimized image size
- **Health checks** for container monitoring
- **Volume mounts** for persistent data and logs
- **Environment variable support** via `.env` file
- **Hot reload** for development

### **Docker Commands**

**Build and Run**:
```bash
# Build and start the application
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

**Individual Docker Commands**:
```bash
# Build the image
docker build -t shelf-analysis .

# Run the container
docker run -p 8000:8000 --env-file .env shelf-analysis

# Run with custom environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json \
  shelf-analysis
```

### **Docker Configuration**
- **Base Image**: Python 3.12
- **Port**: 8000 (exposed)
- **Volumes**: 
  - `./data:/app/data` (persistent data)
  - `./logs:/app/logs` (application logs)
- **Health Check**: `http://localhost:8000/health`

### **Production Docker Setup**
```bash
# Production build (no development dependencies)
docker build --target production -t shelf-analysis:prod .

# Run with production settings
docker run -d \
  --name shelf-analysis-prod \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  shelf-analysis:prod
```

See `DOCKER.md` for detailed Docker documentation.

## 🚀 Production Deployment

### **Recommended Setup**
1. **Use Custom YOLO + Gemini** for speed and cost
2. **Batch processing** for multiple images
3. **GPU acceleration** for YOLO inference
4. **Load balancing** for high traffic

### **Performance Tuning**
- Batch size: 5-10 images
- Processing time: 5-10 seconds per batch
- Memory: 4-8GB RAM recommended
- GPU: Optional but recommended for YOLO

## 📊 Response Format

```json
{
  "model_used": "gemini-flash-lite",
  "processing_time_ms": 8500,
  "is_display_detected": true,
  "shelves": [
    {
      "shelf_position": 1,
      "shelf_visibility": 95,
      "products": [
        {
          "name": "Coca Cola",
          "full_name": "Coca Cola Classic 330ml",
          "count": 12
        }
      ]
    }
  ],
  "empty_spaces": [
    {
      "shelf_position": 2,
      "area_percentage": 15.5
    }
  ],
  "scores": {
    "product_filling": {"value": 85, "comment": "Well stocked"},
    "product_neatness": {"value": 90, "comment": "Very organized"}
  },
  "total_score": 87,
  "general_comment": "Excellent shelf organization and product availability"
}
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run comparison tests
cd comparison
python comparison_script.py --test
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support:
- Check the comparison results in `comparison/FINAL_COMPARISON_REPORT.md`
- Review the API documentation at `http://localhost:8000/docs`
- See `comparison/README.md` for comparison tools usage
