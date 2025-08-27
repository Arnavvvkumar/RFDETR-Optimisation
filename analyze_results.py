import pandas as pd
import os

def analyze_results():
    """Analyze benchmark results and compute average metrics"""
    
    # Check if results files exist
    if not os.path.exists('results/benchmark_results.csv'):
        print("Error: results/benchmark_results.csv not found")
        return
    
    if not os.path.exists('results/timing_results.csv'):
        print("Error: results/timing_results.csv not found")
        return
    
    # Read results
    df = pd.read_csv('results/benchmark_results.csv')
    timing_df = pd.read_csv('results/timing_results.csv')
    
    # Model file sizes (in MB)
    file_sizes = {
        'original': 19.0,
        'pruned_1': 19.0,
        'pruned_30': 19.0,
        'pruned_40': 19.0,
        'quantized': 5.0
    }
    
    # Compute average metrics for each model
    results = []
    
    for model_type in ['original', 'pruned_1', 'pruned_30', 'pruned_40', 'quantized']:
        model_data = df[df['model_type'] == model_type]
        timing_data = timing_df[timing_df['model_type'] == model_type]
        
        if len(model_data) == 0:
            continue
        
        # Compute averages
        precision = model_data['precision'].mean()
        recall = model_data['recall'].mean()
        f1 = model_data['f1'].mean()
        map_50 = model_data['ap@0.5'].mean()
        avg_time = timing_data['inference_time'].mean()
        
        results.append({
            'Model': model_type.replace('_', ' ').title(),
            'Size (MB)': file_sizes[model_type],
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}", 
            'F1': f"{f1:.3f}",
            'mAP@0.5': f"{map_50:.3f}",
            'Avg Time (s)': f"{avg_time:.4f}"
        })
    
    # Create comparison table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Print header
    header = f"{'Model':<15} {'Size (MB)':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'mAP@0.5':<10} {'Avg Time (s)':<12}"
    print(header)
    print("-" * 80)
    
    # Print results
    for result in results:
        row = f"{result['Model']:<15} {result['Size (MB)']:<10.1f} {result['Precision']:<10} {result['Recall']:<10} {result['F1']:<10} {result['mAP@0.5']:<10} {result['Avg Time (s)']:<12}"
        print(row)
    
    print("="*80)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 40)
    
    # Find best F1 score
    best_f1 = max(results, key=lambda x: float(x['F1']))
    print(f"• Best F1 Score: {best_f1['Model']} ({best_f1['F1']})")
    
    # Find fastest model
    fastest = min(results, key=lambda x: float(x['Avg Time (s)']))
    print(f"• Fastest Model: {fastest['Model']} ({fastest['Avg Time (s)']}s)")
    
    # Find smallest model
    smallest = min(results, key=lambda x: x['Size (MB)'])
    print(f"• Smallest Model: {smallest['Model']} ({smallest['Size (MB)']}MB)")
    
    # Quantization benefits
    quantized = next((r for r in results if 'Quantized' in r['Model']), None)
    if quantized:
        size_reduction = ((19.0 - quantized['Size (MB)']) / 19.0) * 100
        print(f"• Quantization reduces model size by {size_reduction:.1f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_results()
