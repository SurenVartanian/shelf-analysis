# 🎯 Shelf Analysis Comparison Tools

This directory contains tools for comparing different approaches to retail shelf analysis.

## 📊 Available Comparison Scripts

### 1. **Main Comparison: Custom YOLO + Gemini vs Direct GPT-4o**
```bash
python comparison_script.py --batch-size 5 --max-images 15
```
- **Purpose**: Compare Custom YOLO + Gemini Flash Lite vs Direct GPT-4o
- **Results**: Speed, accuracy, and cost analysis
- **Output**: `results/` directory with detailed metrics

### 2. **Three-Way Comparison: Custom vs Standard YOLO vs Direct LLM**
```bash
python custom_yolo_comparison.py
```
- **Purpose**: Compare all three approaches (Custom YOLO, Standard YOLO, Direct LLM)
- **Use Case**: Comprehensive performance analysis
- **Output**: `custom_comparison_results/` directory

### 3. **Judgment Analysis**
```bash
python analyze_judgments.py
```
- **Purpose**: Analyze OpenAI judgment results from comparison
- **Output**: Summary statistics and winner analysis

## 📈 Key Results

### **Final Recommendation: Custom YOLO + Gemini**
- **3-6x faster** than Direct GPT-4o
- **5-10x cheaper** than Direct GPT-4o  
- **Better shelf detection** (F1: 0.945 vs 0.854)
- **Competitive accuracy** (7.3/10 vs 7.3/10)

See `FINAL_COMPARISON_REPORT.md` for detailed analysis.

## 🛠️ Setup Requirements

1. **Start the API server**:
   ```bash
   cd .. && uvicorn src.shelf_analyzer.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Install dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Ensure models are loaded**:
   - Custom YOLO model: `models/yolo/shelf_analysis_custom.pt`
   - Standard YOLO model: Will be downloaded automatically

## 📁 Directory Structure

```
comparison/
├── comparison_script.py          # Main Gemini vs GPT-4o comparison
├── custom_yolo_comparison.py     # Three-way comparison
├── analyze_judgments.py          # Judgment analysis
├── FINAL_COMPARISON_REPORT.md    # Final report
├── images/                       # Test images (15 shelf images)
├── ground_truth/                 # Ground truth data
├── results/                      # Comparison results (gitignored)
├── yolo_dataset/                 # YOLO training dataset
├── gpt_annotation_script.py      # GPT-4o annotation tool
├── prepare_yolo_dataset.py       # Dataset preparation
├── visualize_annotations.py      # Annotation visualization
└── compare_original_annotated.py # Side-by-side comparison
```

## 🎯 Usage Examples

### Quick Test (5 images)
```bash
python comparison_script.py --batch-size 5 --test
```

### Full Comparison (15 images)
```bash
python comparison_script.py --batch-size 5 --max-images 15
```

### Custom Batch Size
```bash
python comparison_script.py --batch-size 3 --max-images 10
```

## 📊 Understanding Results

### **F1 Metrics**
- **F1 Score**: Harmonic mean of precision and recall for shelf detection
- **Error**: Absolute difference from ground truth shelf count
- **Winner**: Method with higher F1 score

### **Quality Scores**
- **Shelf Count Score**: 0-10 based on accuracy
- **Product Count Score**: 0-10 based on accuracy  
- **Overall Quality**: 0-10 based on OpenAI judgment

### **Processing Times**
- **Custom YOLO + Gemini**: 5-10 seconds
- **Direct GPT-4o**: 30 seconds
- **Speed Advantage**: 3-6x faster with Custom YOLO + Gemini

## 🔧 Customization

### **Adding New Test Images**
1. Place images in `images/` directory
2. Add ground truth in `ground_truth/` directory
3. Run comparison script

### **Modifying Comparison Parameters**
- Edit `comparison_script.py` for batch size, models, etc.
- Edit `custom_yolo_comparison.py` for three-way comparison

### **Changing Models**
- Update `model` parameter in scripts
- Ensure API endpoints support the model

## 🚀 Production Recommendations

### **For Real-time Applications**
- Use **Custom YOLO + Gemini** for speed and cost
- Batch size: 5-10 images
- Processing time: 5-10 seconds per batch

### **For High-accuracy Requirements**
- Use **Direct GPT-4o** for maximum accuracy
- Consider hybrid approach for different use cases

### **For Cost-sensitive Applications**
- Use **Custom YOLO + Gemini** (5-10x cheaper)
- Monitor API costs and usage

## 📝 Notes

- Results are saved in `results/` directory (gitignored)
- Ground truth data is required for F1 metrics
- OpenAI API key needed for judgment analysis
- Custom YOLO model must be trained and available

## 🔗 Related Files

- `../src/shelf_analyzer/main.py` - API endpoints
- `../src/shelf_analyzer/services/yolo_service.py` - YOLO service
- `../src/shelf_analyzer/services/litellm_vision_service.py` - Vision service
- `../FINAL_COMPARISON_REPORT.md` - Detailed final report 