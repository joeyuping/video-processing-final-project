# Feature Map Compression for ResNet-18 using Block-Sparse Quantization

This project implements a block-sparse quantization approach for compressing feature maps in a ResNet-18 model trained on the CIFAR-100 dataset. The compression is applied after the first module of the ResNet-18 architecture.

## Features

* ResNet-18 model with block-sparse feature map compression
* Configurable block size and compression ratio
* Channel-wise or global block selection
* Huffman coding for further compression
* Comprehensive compression statistics and visualization
* CIFAR-100 dataset classification

## Block-Sparse Compression Method

The compression pipeline consists of three main steps:

1. **Block Partitioning**: 
   - Feature maps are divided into fixed-size blocks (default: 4x4)
   - Border features are zeroed out to reduce artifacts

2. **Block Selection**:
   - Calculates energy (sum of squared values) for each block
   - Selects top K% blocks based on energy values
   - Supports both channel-wise and global selection modes

3. **Advanced Encoding**:
   - Applies Huffman coding to selected block values
   - Uses delta encoding and run-length encoding for block indices
   - Optimizes bit allocation for different components

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training and Evaluation

```bash
python feature_compression-blocksparse.py
```

### Advanced Options

```bash
# Channel-wise compression
python feature_compression-blocksparse.py --channel-wise

# Custom batch size
python feature_compression-blocksparse.py --batch-size 64

# Quick testing with subset of data
python feature_compression-blocksparse.py --subset
```

## Compression Parameters

The code supports various compression configurations:

* **Block Size**: 4x4 (default)
* **Compression Ratios**: 
  - 1.0 (baseline, no compression)
  - 0.5 (keep top 50% blocks)
  - 0.3 (keep top 30% blocks)
  - 0.1 (keep top 10% blocks)
  - 0.05 (keep top 5% blocks)
* **Selection Mode**:
  - Channel-wise: selects top blocks independently for each channel
  - Global: selects top blocks across all channels

## Visualization Outputs

The code generates several visualization files:

1. `feature_distribution.png`: Distribution of feature values with compression thresholds
2. `block_energy_dist.png`: Distribution of block energies
3. `feature_maps_ratio.png`: Visualization of compressed feature maps
4. `accuracy_vs_compression.png`: Accuracy vs compression ratio plots

## Results Analysis

The compression results are saved in `compression_results.json` with:
- Compression ratios and corresponding accuracies
- Bits per pixel (BPP) measurements
- Detailed compression statistics
- Model performance with and without fine-tuning

## Implementation Details

- Uses PyTorch for deep learning framework
- Implements custom compression layers in ResNet-18
- Supports both training and inference modes
- Includes early stopping and learning rate scheduling
- Provides detailed compression statistics and analysis tools

## Performance Metrics

The code tracks several key metrics:
- Classification accuracy
- Compression ratio
- Bits per pixel (BPP)
- Block energy distribution
- Feature value distribution
- Memory usage statistics


