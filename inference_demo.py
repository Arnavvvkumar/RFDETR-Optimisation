import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import os
import argparse

def load_model(model_path):
    """Load ONNX model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found")
    return ort.InferenceSession(model_path)

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()
    return input_tensor, image.size

def run_inference(model_path, image_path, confidence_threshold=0.3):
    """Run face detection inference"""
    # Load model
    session = load_model(model_path)
    
    # Preprocess image
    input_tensor, (orig_w, orig_h) = preprocess_image(image_path)
    
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
        if max_logits[i] > confidence_threshold:
            box = boxes[i] * 800  # Scale to 800x800
            # Scale back to original image size
            x1, y1, x2, y2 = box
            x1 = x1 * orig_w / 800
            y1 = y1 * orig_h / 800
            x2 = x2 * orig_w / 800
            y2 = y2 * orig_h / 800
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(max_logits[i])
            })
    
    return detections

def draw_detections(image_path, detections, output_path, model_name="Model"):
    """Draw bounding boxes on image and save"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Color based on model
    colors = {
        "Original": "blue",
        "Pruned 1%": "green", 
        "Pruned 30%": "orange",
        "Quantized": "red"
    }
    color = colors.get(model_name, "blue")
    
    # Draw each detection
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw confidence score
        label = f"{confidence:.2f}"
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text background
        text_bbox = draw.textbbox((x1, y1-20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1-20), label, fill="white", font=font)
    
    image.save(output_path)
    print(f"Result saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Face Detection Demo")
    parser.add_argument("--model", type=str, default="rf_detr_quantized.onnx", 
                       help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to input image")
    parser.add_argument("--output", type=str, default="output.jpg", 
                       help="Path to output image")
    parser.add_argument("--confidence", type=float, default=0.3, 
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Run inference
    print(f"Running inference with {args.model} on {args.image}")
    detections = run_inference(args.model, args.image, args.confidence)
    
    print(f"Found {len(detections)} faces")
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        print(f"  Face {i+1}: bbox={bbox}, confidence={confidence:.3f}")
    
    # Draw and save result
    model_name = os.path.basename(args.model).replace('.onnx', '').replace('rf_detr_', '').replace('_', ' ').title()
    draw_detections(args.image, detections, args.output, model_name)

if __name__ == "__main__":
    main()
