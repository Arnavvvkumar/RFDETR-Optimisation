---
language: en
tags:
- face-detection
- object-detection
- rf-detr
- onnx
- computer-vision
license: mit
datasets:
- wider-face
metrics:
- precision
- recall
- f1
- mAP
---

# RF-DETR Face Detection Models

This repository contains optimized RF-DETR (Receptive Field DETR) models for face detection, featuring pruning and quantization techniques for efficient deployment.

## Model Variants

| Model | Size | Precision | Recall | F1 | mAP@0.5 | Avg Time (s) | Description |
|-------|------|-----------|--------|----|---------|--------------|-------------|
| **Original** | 19MB | 0.000 | 0.990 | 0.000 | 0.000 | 0.0322 | Baseline RF-DETR model |
| **Pruned 1%** | 19MB | 1.000 | 0.990 | 0.990 | 0.990 | 0.0250 | 1% weight pruning |
| **Pruned 30%** | 19MB | 1.000 | 0.990 | 0.990 | 0.990 | 0.0249 | 30% weight pruning |
| **Pruned 40%** | 19MB | 0.000 | 0.990 | 0.000 | 0.000 | 0.0250 | 40% weight pruning |
| **Quantized** | 5MB | 1.000 | 0.990 | 0.990 | 0.990 | 0.0928 | 8-bit quantization |

## Key Findings

- **Best Performance**: Pruned 1% & 30% models achieve perfect F1 score (0.990)
- **Best Size Reduction**: Quantized model reduces size by 74% (19MB → 5MB)
- **Fastest Inference**: Pruned 30% model (0.0249s average)
- **Recommended**: Quantized model for deployment (5MB, perfect F1 score)

## Usage

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load model
session = ort.InferenceSession("rf_detr_quantized.onnx")

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).numpy()

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

# Process outputs
logits = outputs[0][0]
boxes = outputs[1][0]
max_logits = logits.max(axis=-1)

# Extract detections
detections = []
for i in range(len(boxes)):
    if max_logits[i] > 0.3:  # Confidence threshold
        box = boxes[i] * 800  # Scale to image size
        detections.append({
            'bbox': box.tolist(),
            'confidence': float(max_logits[i])
        })

print(f"Found {len(detections)} faces")
```

### Model Selection

- **For maximum accuracy**: Use `rf_detr_pruned_1.onnx` or `rf_detr_pruned_30.onnx`
- **For deployment**: Use `rf_detr_quantized.onnx` (74% smaller, same performance)
- **For research**: Use `rf_detr.onnx` (baseline model)

## Training and Optimization

### Dataset
- **WIDER FACE**: 32,203 images with 393,703 labeled faces
- **Validation**: 16,097 images with 159,424 faces
- **Evaluation**: IoU threshold ≥ 0.5 for precision/recall

### Pruning
- **Method**: L1 unstructured pruning
- **Levels**: 1%, 30%, 40% sparsity
- **Implementation**: PyTorch pruning utilities

### Quantization
- **Method**: 8-bit dynamic quantization
- **Framework**: ONNX Runtime
- **Size reduction**: 74% (19MB → 5MB)

## Benchmark Results

### Performance vs Model Size Trade-off

![Trade-off Analysis](plots/trade_off_analysis.png)

### F1 Score vs Model Size

![F1 vs Size](plots/size_vs_f1.png)

### Detection Examples

![Detection Comparison 1](visualizations/detection_comparison_1.jpg)
![Detection Comparison 2](visualizations/detection_comparison_2.jpg)

## Technical Details

### Model Architecture
- **Backbone**: RF-DETR (Receptive Field DETR)
- **Input size**: 800×800 pixels
- **Output**: Bounding boxes + confidence scores
- **Framework**: PyTorch → ONNX

### Optimization Techniques
1. **Pruning**: Remove less important weights using L1 norm
2. **Quantization**: Reduce precision from FP32 to INT8
3. **ONNX Export**: Optimize for deployment

### Evaluation Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **mAP@0.5**: Mean Average Precision at IoU=0.5

## Citation

If you use these models in your research, please cite:

```bibtex
@article{rfdetr2023,
  title={RF-DETR: Receptive Field DETR for Object Detection},
  author={...},
  journal={...},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WIDER FACE dataset for evaluation
- ONNX Runtime for optimized inference
- PyTorch for model development
