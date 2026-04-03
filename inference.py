import torch
import onnx
from onnxruntime.quantization import quantize_dynamic
import os

from pruning import load_pytorch_model, prune_model


def export_to_onnx(model, filename):
    """Export PyTorch model to ONNX format"""
    dummy_input = torch.randn(1, 3, 800, 800)
    
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'boxes'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'boxes': {0: 'batch_size'}
        }
    )

def quantize_onnx_model(input_path, output_path):
    """Quantize ONNX model to 8-bit"""
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=onnx.TensorProto.INT8,
        optimize_model=True
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading base model...")
    base_model = load_pytorch_model("models/rf-detr-base.pth", device)

    os.makedirs("models", exist_ok=True)
    
    prune_percentages = [0.01, 0.30, 0.40]  # 1%, 30%, 40%
    
    for prune_percent in prune_percentages:
        print(f"\nPruning model with {prune_percent*100}% sparsity...")
        
        # Apply pruning
        pruned_model = prune_model(base_model, prune_percent)
        
        prune_suffix = str(int(prune_percent * 100))
        
        # Save pruned PyTorch model
        torch.save({'model_state_dict': pruned_model.state_dict()}, f"models/rf_detr_pruned_{prune_suffix}.pth")
        print(f"Saved pruned PyTorch model: models/rf_detr_pruned_{prune_suffix}.pth")
        
        # Export to ONNX
        onnx_filename = f"models/rf_detr_pruned_{prune_suffix}.onnx"
        export_to_onnx(pruned_model, onnx_filename)
        print(f"Exported to ONNX: {onnx_filename}")
        
        # Quantize ONNX model
        quantized_filename = f"models/rf_detr_pruned_{prune_suffix}_quantized.onnx"
        quantize_onnx_model(onnx_filename, quantized_filename)
        print(f"Quantized ONNX model: {quantized_filename}")
    
    print("\nCreating quantized version of original model...")
    original_onnx = "models/rf_detr.onnx"
    if os.path.exists(original_onnx):
        quantize_onnx_model(original_onnx, "models/rf_detr_quantized.onnx")
        print("Created: models/rf_detr_quantized.onnx")
    
    print("\nAll models created successfully!")

if __name__ == "__main__":
    main()
