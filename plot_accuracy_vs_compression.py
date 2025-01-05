import matplotlib.pyplot as plt
import json

def plot_accuracy_vs_compression(json_path, save_path='accuracy_vs_compression.png'):
    """Plot accuracy vs bits per pixel from compression results JSON."""
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    results = data['compression_results']
    
    # Initialize lists for plotting
    acc_no_ft = []
    acc_ft = []
    bpp = []
    ratios = []
    
    # Process each result
    for result in results:
        ratios.append(result['compression_ratio'])
        bpp.append(result['bpp'])
        
        # Handle baseline (r=1.0) case differently
        if result['ratio'] == 1.0:
            baseline_acc = result['accuracy']
            acc_no_ft.append(baseline_acc)
            acc_ft.append(baseline_acc)
        else:
            acc_no_ft.append(result['accuracy_no_ft'])
            acc_ft.append(result['accuracy_ft'])
    
    plt.figure(figsize=(15, 6))
    
    # Accuracy vs Compressed Bits per Pixel
    plt.subplot(1, 2, 1)
    plt.plot(bpp, acc_no_ft, 'bo-', label='No Fine-tuning')
    plt.plot(bpp, acc_ft, 'ro-', label='With Fine-tuning')
    
    # Add data labels
    for i, (bits_per_pixel, acc) in enumerate(zip(bpp, acc_no_ft)):
        ratio = ratios[i]
        label = f'r={ratio:.2f}\nBPP={bits_per_pixel:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (bits_per_pixel, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    for i, (bits_per_pixel, acc) in enumerate(zip(bpp, acc_ft)):
        ratio = ratios[i]
        label = f'r={ratio:.2f}\nBPP={bits_per_pixel:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (bits_per_pixel, acc), textcoords="offset points", 
                    xytext=(0,-25), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
    
    plt.xlabel('Compressed Bits per Pixel')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Compressed Bits per Pixel')
    plt.legend()
    plt.grid(True)
    
    # Accuracy vs Compression Ratio
    plt.subplot(1, 2, 2)
    plt.plot(ratios, acc_no_ft, 'bo-', label='No Fine-tuning')
    plt.plot(ratios, acc_ft, 'ro-', label='With Fine-tuning')
    
    # Add data labels
    for i, (ratio, acc) in enumerate(zip(ratios, acc_no_ft)):
        bits_per_pixel = bpp[i]
        label = f'r={ratio:.2f}\nBPP={bits_per_pixel:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (ratio, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    for i, (ratio, acc) in enumerate(zip(ratios, acc_ft)):
        bits_per_pixel = bpp[i]
        label = f'r={ratio:.2f}\nBPP={bits_per_pixel:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (ratio, acc), textcoords="offset points", 
                    xytext=(0,-25), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
    
    plt.xlabel('Compression Ratio')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Compression Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_accuracy_vs_compression('results/blocksparse-channelwise 4x4/compression_results.json', 'results/blocksparse-channelwise 4x4/accuracy_vs_compression.png')
    print("Plot saved as accuracy_vs_compression.png")