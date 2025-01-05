# Feature Map Compression using Block-Sparse Quantization

This implementation provides a block-sparse compression method for neural network feature maps, with support for different compression modes and fine-tuning capabilities.

## Features

* Block-sparse compression of feature maps with configurable block sizes
* Multiple compression modes: channel-wise, region-wise, and global
* Huffman coding for additional compression
* Support for fine-tuning after compression
* Comprehensive visualization tools for feature maps and compression statistics
* Pre-trained ResNet18 model integration

## Block-Sparse Compression Method

The compression pipeline consists of three main steps:

1. **Block Partitioning**:
   * Feature maps are divided into fixed-size blocks (default: 4x4)
   * Border features are zeroed out to reduce artifacts
2. **Block Selection**:
   * Calculates energy (sum of squared values) for each block
   * Selects top K% blocks based on energy values
   * Supports channel-wise, region-wise and global selection modes
3. **Advanced Encoding**:
   * Applies Huffman coding to selected block values
   * Uses delta encoding and Huffman coding for block indices

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python feature_compression-blocksparse.py --batch-size 128 --mode channel-wise --block-size 4
```

### Arguments

* `--batch-size`: Batch size for training (default: 128)
* `--mode`: Compression mode: channel-wise, region-wise, or global (default: channel-wise)
* `--block-size`: Size of blocks for block-sparse compression (default: 4)
* `--subset`: Use subset of data for testing (optional)

### Compression Modes


1. **Channel-wise**: Compresses each channel independently
2. **Region-wise**: Selects regions across all channels
3. **Global**: Applies compression globally across all channels and spatial dimensions

## Output

The script generates several visualization files:

* `feature_distribution.png`: Distribution of feature values with compression ratio cutoffs
* `block_energy_dist.png`: Distribution of block energies
* `feature_maps_ratio.png`: Visualization of feature maps at different compression ratios
* `accuracy_vs_compression.png`: Plots of accuracy vs. compression metrics
* `compression_results.json`: Detailed results in JSON format

## Example Results

The implementation provides comprehensive metrics including:

* Accuracy with and without fine-tuning
* Bits per pixel (BPP)
* Compression ratios
* Block energy distributions

## Compression Parameters

The code supports various compression configurations:

* **Block Size**: 4x4 (default) , 1x1 reduces to simple sparse implementation
* **Compression Ratios**:
  * 1.0 (baseline, no compression)
  * 0.5 (keep top 50% blocks)
  * 0.3 (keep top 30% blocks)
  * 0.1 (keep top 10% blocks)
  * 0.05 (keep top 5% blocks)

## Visualization Outputs

The code generates several visualization files:

1. `feature_distribution.png`: Distribution of feature values with compression thresholds
2. `block_energy_dist.png`: Distribution of block energies
3. `feature_maps_ratio.png`: Visualization of compressed feature maps
4. `accuracy_vs_compression.png`: Accuracy vs compression ratio plots

## Results Analysis

The compression results are saved in `compression_results.json` with:

* Compression ratios and corresponding accuracies
* Bits per pixel (BPP) measurements
* Detailed compression statistics
* Model performance with and without fine-tuning

## Implementation Details

* Uses PyTorch for deep learning framework
* Implements custom compression layers in ResNet-18
* Supports both training and inference modes
* Includes early stopping and learning rate scheduling
* Provides detailed compression statistics and analysis tools

## Performance Metrics

The code tracks several key metrics:

* Classification accuracy
* Compression ratio
* Bits per pixel (BPP)
* Block energy distribution
* Feature value distribution
* Memory usage statistics


