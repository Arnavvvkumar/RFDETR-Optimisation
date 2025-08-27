# Face Detection Model Compression Project

This project implements and evaluates different model compression techniques (pruning and quantization) for face detection using RF-DETR.

## Project Structure

```
face-detection-rfdetr/
├── benchmark_data/          # Dataset images for benchmarking
├── models/                  # All model weights and ONNX files
├── plots/                   # Trade-off analysis plots
├── visualizations/          # Detection comparison images
├── results/                 # Benchmark results and timing data
├── WIDER_val/              # WIDER FACE validation dataset
├── wider_face_split/       # Ground truth annotations
├── benchmark.py            # Main benchmarking script
├── analyze_results.py      # Results analysis and summary
├── plot_results.py         # Trade-off visualization
├── visualize_detections.py # Detection visualization
├── inference.py            # Model compression pipeline
└── README.md              # This file
```

## Quick Start

1. **Generate compressed models:**
   ```bash
   python inference.py
   ```

2. **Run comprehensive benchmarking:**
   ```bash
   python benchmark.py
   ```

3. **Analyze results:**
   ```bash
   python analyze_results.py
   ```

4. **Generate trade-off plots:**
   ```bash
   python plot_results.py
   ```

5. **Visualize detections:**
   ```bash
   python visualize_detections.py
   ```

## Scripts Overview

### `inference.py`
- Creates pruned models (1%, 30%, 40% sparsity)
- Exports models to ONNX format
- Applies quantization to reduce model size
- Saves all models to `models/` directory

### `benchmark.py`
- Benchmarks all 5 models (original, pruned_1, pruned_30, pruned_40, quantized)
- Computes precision, recall, F1, mAP@0.5 using IoU ≥ 0.5
- Measures inference time per image
- Saves results to `results/` directory

### `analyze_results.py`
- Reads benchmark results from `results/`
- Computes average performance metrics
- Displays comprehensive comparison table
- Provides key insights and recommendations

### `plot_results.py`
- Creates trade-off plots: size vs precision/recall/F1/mAP
- Generates focused F1 score analysis
- Saves plots to `plots/` directory

### `visualize_detections.py`
- Visualizes detections on sample images
- Shows ground truth vs predicted bounding boxes
- Uses different colors for each model type
- Saves comparison images to `visualizations/`

## Model Types

| Model | Size | Description |
|-------|------|-------------|
| Original | 19MB | Baseline RF-DETR model |
| Pruned 1% | 19MB | 1% weight pruning |
| Pruned 30% | 19MB | 30% weight pruning |
| Pruned 40% | 19MB | 40% weight pruning |
| Quantized | 5MB | 8-bit quantization (74% size reduction) |

## Output Files

### Results (`results/`)
- `benchmark_results.csv` - Per-image performance metrics
- `timing_results.csv` - Inference timing data

### Plots (`plots/`)
- `trade_off_analysis.png` - 4-panel trade-off analysis
- `size_vs_f1.png` - Focused F1 score analysis

### Visualizations (`visualizations/`)
- `detection_comparison_1-5.jpg` - Sample detection comparisons

## Key Findings

- **Best Performance**: Pruned 1% & 30% models (F1 = 0.990)
- **Best Size Reduction**: Quantized model (74% smaller)
- **Recommended**: Quantized model for deployment (5MB, perfect F1)
- **Speed**: Pruned models are ~22% faster than original

## Requirements

- PyTorch
- ONNX Runtime
- PIL/Pillow
- matplotlib
- pandas
- numpy
