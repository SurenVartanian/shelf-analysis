# Quick Summary: F1 Score Analysis Results

## 🎯 Key Findings (15 Images)

**YOLO vs Direct LLM Analysis**

| Metric | YOLO | Direct | Difference |
|--------|------|--------|------------|
| **Average F1 Score** | 0.951 | 0.770 | **+0.181** |
| **Average Error** | 0.4 shelves | 1.1 shelves | **-63.6%** |
| **Win Rate** | 33.3% | 13.3% | **+20.0%** |
| **Tie Rate** | 53.3% | - | - |

## 🏆 Performance Breakdown

- **YOLO F1 wins**: 5 images
- **Direct F1 wins**: 2 images  
- **Ties**: 8 images

## 🚨 Critical Cases

**Direct Analysis Complete Failures (0 shelves detected):**
- IMG_20250728_132733: Should be 5, detected 0 (F1: 0.000)
- IMG_20250728_132757: Should be 5, detected 0 (F1: 0.000)  
- IMG_20250728_132845: Should be 4, detected 0 (F1: 0.000)

**YOLO Performance in Same Cases:**
- IMG_20250728_132733: Detected 4/5 (F1: 0.889)
- IMG_20250728_132757: Detected 6/5 (F1: 0.909)
- IMG_20250728_132845: Detected 3/4 (F1: 0.857)

## 📊 Recommendation

**Use YOLO preprocessing for shelf analysis** - it provides:
- 18.1 percentage point improvement in accuracy
- 63.6% reduction in errors
- Better handling of challenging edge cases
- More consistent performance

## 📁 Files

- **Full Report**: `F1_Analysis_Report.md`
- **Results**: `results/yolo/`, `results/direct/`, `results/judgments/`
- **Script**: `comparison_script.py` 