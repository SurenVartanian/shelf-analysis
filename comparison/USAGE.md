# Comparison Script Usage

## Basic Usage

```bash
# Process all images in batches of 5 (default)
python comparison_script.py

# Process only first 5 images (test mode)
python comparison_script.py --test

# Process in batches of 10
python comparison_script.py --batch-size 10

# Process only first 20 images
python comparison_script.py --max-images 20

# Process first 15 images in batches of 3
python comparison_script.py --batch-size 3 --max-images 15
```

## Command Line Options

- `--batch-size N`: Process N images at a time (default: 5)
- `--max-images N`: Limit total images to N (default: all)
- `--test`: Quick test mode (process only first batch)

## Batch Processing Benefits

✅ **Safer**: Process in manageable chunks
✅ **Resumable**: Can stop and restart
✅ **Progress tracking**: See exactly where you are
✅ **Error handling**: Continue even if some images fail
✅ **API friendly**: 2-second delays between batches

## Example Workflows

### Quick Test (5 images)
```bash
python comparison_script.py --test
```

### Medium Test (20 images)
```bash
python comparison_script.py --max-images 20
```

### Full Run (all 93 images)
```bash
python comparison_script.py
```

### Large Batch (10 at a time)
```bash
python comparison_script.py --batch-size 10
```

## Progress Tracking

The script shows:
- 📦 Current batch number
- 📊 Overall progress percentage
- ✅ Batch completion status
- ❌ Error count
- ⏳ Wait time between batches

## Results

All results are saved in:
- `results/yolo/` - YOLO analysis results
- `results/direct/` - Direct analysis results
- `results/judgments/` - OpenAI judgments

The script won't re-process images that already have results! 