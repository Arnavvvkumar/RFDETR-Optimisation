import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    """Generate trade-off plots comparing model size with performance metrics"""
    
    # Check if results file exists
    if not os.path.exists('results/benchmark_results.csv'):
        print("Error: results/benchmark_results.csv not found")
        return
    
    # Read results
    df = pd.read_csv('results/benchmark_results.csv')
    
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
        
        if len(model_data) == 0:
            continue
        
        # Compute averages
        precision = model_data['precision'].mean()
        recall = model_data['recall'].mean()
        f1 = model_data['f1'].mean()
        map_50 = model_data['ap@0.5'].mean()
        
        results.append({
            'Model': model_type.replace('_', ' ').title(),
            'Size (MB)': file_sizes[model_type],
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'mAP@0.5': map_50
        })
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Size vs Performance Trade-offs', fontsize=16, fontweight='bold')
    
    # Plot 1: Size vs Precision
    axes[0, 0].scatter([r['Size (MB)'] for r in results], [r['Precision'] for r in results], 
                       s=100, alpha=0.7, c=['blue', 'green', 'orange', 'red', 'purple'])
    for i, result in enumerate(results):
        axes[0, 0].annotate(result['Model'], (result['Size (MB)'], result['Precision']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 0].set_xlabel('Model Size (MB)')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Size vs Precision')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Size vs Recall
    axes[0, 1].scatter([r['Size (MB)'] for r in results], [r['Recall'] for r in results], 
                       s=100, alpha=0.7, c=['blue', 'green', 'orange', 'red', 'purple'])
    for i, result in enumerate(results):
        axes[0, 1].annotate(result['Model'], (result['Size (MB)'], result['Recall']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('Model Size (MB)')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Size vs Recall')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Size vs F1
    axes[1, 0].scatter([r['Size (MB)'] for r in results], [r['F1'] for r in results], 
                       s=100, alpha=0.7, c=['blue', 'green', 'orange', 'red', 'purple'])
    for i, result in enumerate(results):
        axes[1, 0].annotate(result['Model'], (result['Size (MB)'], result['F1']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 0].set_xlabel('Model Size (MB)')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Size vs F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Size vs mAP@0.5
    axes[1, 1].scatter([r['Size (MB)'] for r in results], [r['mAP@0.5'] for r in results], 
                       s=100, alpha=0.7, c=['blue', 'green', 'orange', 'red', 'purple'])
    for i, result in enumerate(results):
        axes[1, 1].annotate(result['Model'], (result['Size (MB)'], result['mAP@0.5']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_xlabel('Model Size (MB)')
    axes[1, 1].set_ylabel('mAP@0.5')
    axes[1, 1].set_title('Size vs mAP@0.5')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/trade_off_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/trade_off_analysis.png")
    
    # Create focused plot: Size vs F1
    plt.figure(figsize=(10, 6))
    plt.scatter([r['Size (MB)'] for r in results], [r['F1'] for r in results], 
                s=150, alpha=0.8, c=['blue', 'green', 'orange', 'red', 'purple'])
    
    for i, result in enumerate(results):
        plt.annotate(result['Model'], (result['Size (MB)'], result['F1']), 
                    xytext=(10, 10), textcoords='offset points', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.xlabel('Model Size (MB)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Model Size vs F1 Score Trade-off', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 25)
    plt.ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig('plots/size_vs_f1.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/size_vs_f1.png")
    
    print("\nPlots generated successfully!")

if __name__ == "__main__":
    plot_results()
