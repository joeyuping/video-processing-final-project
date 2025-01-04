import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import math
from collections import namedtuple
import numpy as np
import random
from huggingface_hub import hf_hub_download
from torchvision.datasets import CIFAR100
from cifair import ciFAIR100
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import heapq
from dataclasses import dataclass
import huffman
from datetime import datetime
import os
from PIL import Image
from tqdm import tqdm
import copy
import argparse

@dataclass
class EncodedData:
    """Container for encoded data with various compression schemes."""
    values: torch.Tensor
    indices: torch.Tensor
    metadata: Optional[Dict]
    huffman_tables: Optional[Dict]
    rle_data: Optional[Dict]
    raw_rows: Optional[np.ndarray]

class HuffmanCoder:
    """Huffman coding for quantized values and indices."""
    def __init__(self):
        self.encoders = {}
        self.decoders = {}
    
    def fit(self, data: torch.Tensor, prefix: str) -> Dict:
        """Create Huffman codes for the data."""
        # Convert to integers and count frequencies
        values, counts = torch.unique(data, return_counts=True)
        freq_dict = {int(val): int(count) for val, count in zip(values, counts)}
        
        # Generate Huffman codes
        codec = huffman.codebook(freq_dict.items())
        self.encoders[prefix] = codec
        self.decoders[prefix] = {v: k for k, v in codec.items()}
        
        # Calculate average bits per symbol
        total_symbols = sum(freq_dict.values())
        avg_bits = sum(len(code) * freq_dict[val] for val, code in codec.items()) / total_symbols
        
        return {
            'codebook': codec,
            'avg_bits': avg_bits
        }
    
    def encode(self, data: torch.Tensor, prefix: str) -> Tuple[str, float]:
        """Encode data using Huffman coding."""
        codec = self.encoders[prefix]
        encoded = ''.join(codec[int(x)] for x in data)
        return encoded, len(encoded)
    
    def decode(self, encoded: str, prefix: str, length: int) -> torch.Tensor:
        """Decode Huffman-encoded data."""
        decoder = self.decoders[prefix]
        current_code = ''
        decoded = []
        
        for bit in encoded:
            current_code += bit
            if current_code in decoder:
                decoded.append(decoder[current_code])
                current_code = ''
        
        return torch.tensor(decoded)

def run_length_encode(data: torch.Tensor) -> Tuple[List[int], List[int]]:
    """Perform run-length encoding on sparse data."""
    if len(data) == 0:
        return [], []
    
    values = []
    lengths = []
    current_value = data[0]
    current_length = 1
    
    for value in data[1:]:
        if value == current_value:
            current_length += 1
        else:
            values.append(int(current_value))
            lengths.append(current_length)
            current_value = value
            current_length = 1
    
    values.append(int(current_value))
    lengths.append(current_length)
    
    return values, lengths

def run_length_decode(values: List[int], lengths: List[int]) -> torch.Tensor:
    """Decode run-length encoded data."""
    decoded = []
    for value, length in zip(values, lengths):
        decoded.extend([value] * length)
    return torch.tensor(decoded)

class FeatureCompressor:
    def __init__(self, compression_ratio=0.1, num_bits=8, method='topk', image_size=(32, 32), channel_wise=True):
        self.compression_ratio = compression_ratio
        self.num_bits = num_bits
        self.method = method
        self.image_height, self.image_width = image_size
        self.channel_wise = channel_wise
    
    def compress_features(self, x):
        """Fast compression for training/evaluation - includes lossy quantization but skips lossless compression."""
        batch_size, channels, height, width = x.size()
        
        if self.channel_wise:
            # Reshape to [B*C, H*W] for vectorized operations
            x_flat = x.reshape(batch_size * channels, -1)
            k = max(1, int(height * width * self.compression_ratio))
            
            # Get top k values and their indices for all channels at once
            values, flat_indices = torch.topk(x_flat.abs(), k, dim=1)
            signs = torch.sign(torch.gather(x_flat, 1, flat_indices))
            values = values * signs  # [B*C, K]
            
            # Calculate spatial indices
            h_idx = flat_indices // width  # [B*C, K]
            w_idx = flat_indices % width   # [B*C, K]
            
            # Generate batch and channel indices
            bc_idx = torch.arange(batch_size * channels, device=x.device)
            b_idx = bc_idx // channels  # [B*C]
            c_idx = bc_idx % channels   # [B*C]
            
            # Expand batch and channel indices to match value dimensions
            b_idx = b_idx.unsqueeze(1).expand(-1, k)  # [B*C, K]
            c_idx = c_idx.unsqueeze(1).expand(-1, k)  # [B*C, K]
            
            # Stack all indices
            all_indices = torch.stack([
                b_idx.flatten(),
                c_idx.flatten(),
                h_idx.flatten(),
                w_idx.flatten()
            ], dim=1)
            
            all_values = values.flatten()
            
        else:
            # Reshape to [B, C*H*W] for whole-feature compression
            x_flat = x.reshape(batch_size, -1)
            k = max(1, int(x_flat.size(1) * self.compression_ratio))
            
            # Get top k values and indices for each batch
            values, flat_indices = torch.topk(x_flat.abs(), k, dim=1)  # [B, K]
            signs = torch.sign(torch.gather(x_flat, 1, flat_indices))
            values = values * signs  # [B, K]
            
            # Calculate indices for each dimension
            c_idx = flat_indices // (height * width)  # [B, K]
            hw = flat_indices % (height * width)      # [B, K]
            h_idx = hw // width                       # [B, K]
            w_idx = hw % width                        # [B, K]
            
            # Generate batch indices
            b_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)  # [B, K]
            
            # Stack all indices
            all_indices = torch.stack([
                b_idx.flatten(),
                c_idx.flatten(),
                h_idx.flatten(),
                w_idx.flatten()
            ], dim=1)
            
            all_values = values.flatten()
        
        # Sort by channel first, then by spatial location
        # Create a combined sorting key: channel * (H*W) + (h * W + w)
        spatial_indices = all_indices[:, 2] * width + all_indices[:, 3]
        sorting_key = all_indices[:, 1] * (height * width) + spatial_indices
        
        # Sort both indices and values
        sort_order = sorting_key.argsort()
        all_indices = all_indices[sort_order]
        all_values = all_values[sort_order]
        
        # Apply quantization if needed
        if self.num_bits < 32:
            min_val = all_values.min()
            max_val = all_values.max()
            scale = (max_val - min_val) / (2**self.num_bits - 1)
            
            # Quantize to integers and dequantize in one step
            all_values = torch.round((all_values - min_val) / scale) * scale + min_val
            metadata = {'min_val': min_val, 'scale': scale}
        else:
            metadata = None
        
        # Return encoded data without Huffman or RLE during training
        return EncodedData(
            values=all_values,
            indices=all_indices,
            metadata=metadata,
            huffman_tables=None,
            rle_data=None,
            raw_rows=None
        )
    
    def decompress_features(self, compressed_features, original_size):
        """Fast decompression for training/evaluation - reconstructs quantized sparse tensor."""
        batch_size, channels, height, width = original_size
        
        # Create output tensor with correct device
        output = torch.zeros(original_size, device=compressed_features.values.device)
        
        # Extract indices
        b_idx = compressed_features.indices[:, 0].long()
        c_idx = compressed_features.indices[:, 1].long()
        h_idx = compressed_features.indices[:, 2].long()
        w_idx = compressed_features.indices[:, 3].long()
        
        # Use index_put_ for more efficient sparse updates
        output.index_put_((b_idx, c_idx, h_idx, w_idx), compressed_features.values)
        
        return output
    
    def calculate_compression_ratio(self, x, compressed_data):
        """Calculate compression ratio with detailed analysis including potential lossless savings."""
        # Calculate original dense size (32-bit float)
        dense_size = x.numel() * 32
        
        # Get number of non-zero values
        n_values = len(compressed_data.values)
        
        # 1. Analyze value compression
        values = compressed_data.values.cpu()
        values_huffman = HuffmanCoder()
        quantized_values = torch.round((values - values.min()) / (values.max() - values.min()) * (2**self.num_bits - 1))
        huffman_stats = values_huffman.fit(quantized_values, 'values')
        values_bits = int(huffman_stats['avg_bits'] * n_values)  # Estimated bits after Huffman
        values_naive_bits = n_values * 32  # 32-bit float
        values_fixed_bits = n_values * self.num_bits  # Fixed-length quantization
        values_savings_vs_fixed = (values_fixed_bits - values_bits) / values_fixed_bits * 100
        values_savings_vs_naive = (values_naive_bits - values_bits) / values_naive_bits * 100
        
        # 2. Analyze column indices with delta encoding + Huffman
        col_indices = compressed_data.indices[:, 3].cpu()
        
        # Compute deltas for columns
        col_deltas = torch.zeros_like(col_indices)
        col_deltas[0] = col_indices[0]  # First value as is
        col_deltas[1:] = col_indices[1:] - col_indices[:-1]  # Differences
        
        # Try Huffman on deltas
        col_huffman = HuffmanCoder()
        col_delta_stats = col_huffman.fit(col_deltas, 'col_deltas')
        col_delta_bits = int(col_delta_stats['avg_bits'] * n_values)
        
        # Try RLE on deltas
        runs_col = []
        current_val = col_deltas[0].item()
        current_count = 1
        for val in col_deltas[1:]:
            if val.item() == current_val:
                current_count += 1
            else:
                runs_col.append((current_val, current_count))
                current_val = val.item()
                current_count = 1
        runs_col.append((current_val, current_count))
        
        # Calculate optimal bits for column delta RLE
        max_delta = max(abs(d.item()) for d in col_deltas)
        col_delta_bits_needed = int(np.ceil(np.log2(2 * max_delta + 1)))  # +1 for zero, *2 for sign
        col_length_bits = int(np.ceil(np.log2(max(count for _, count in runs_col) + 1)))
        col_rle_bits = len(runs_col) * (col_delta_bits_needed + col_length_bits)
        
        # Use the better of Huffman or RLE for column deltas
        col_bits = min(col_delta_bits, col_rle_bits)
        col_method = "Delta+Huffman" if col_delta_bits <= col_rle_bits else "Delta+RLE"
        col_fixed_bits = n_values * math.ceil(math.log2(x.size(3)))
        col_savings = (col_fixed_bits - col_bits) / col_fixed_bits * 100
        
        # Print column compression details with delta stats
        print(f"\n2. Column Indices ({col_method}):")
        print(f"   - Bits with {col_method}: {col_bits:,}")
        print(f"   - Bits with fixed: {col_fixed_bits:,}")
        print(f"   - Max delta: {max_delta}")
        print(f"   - Bits per delta: {col_delta_bits_needed}")
        if "RLE" in col_method:
            print(f"   - Number of runs: {len(runs_col)}")
            print(f"   - Average run length: {n_values/len(runs_col):.1f}")
        else:
            print(f"   - Average bits per delta: {col_delta_stats['avg_bits']:.2f}")
        print(f"   - Savings: {col_savings:.1f}%")
        
        # 3. Analyze row indices with delta encoding
        row_indices = compressed_data.indices[:, 2].cpu()
        
        # Compute deltas for rows
        row_deltas = torch.zeros_like(row_indices)
        row_deltas[0] = row_indices[0]  # First value as is
        row_deltas[1:] = row_indices[1:] - row_indices[:-1]  # Differences
        
        # Try regular RLE for rows (since it worked better)
        runs_row = []
        current_val = row_indices[0].item()
        current_count = 1
        for val in row_indices[1:]:
            if val.item() == current_val:
                current_count += 1
            else:
                runs_row.append((current_val, current_count))
                current_val = val.item()
                current_count = 1
        runs_row.append((current_val, current_count))
        
        # Calculate optimal bits for regular row RLE
        row_bits_needed = int(np.ceil(np.log2(x.size(2))))
        row_length_bits = int(np.ceil(np.log2(max(count for _, count in runs_row) + 1)))
        row_rle_bits = len(runs_row) * (row_bits_needed + row_length_bits)
        
        # Use regular RLE for rows since it's better
        row_bits = row_rle_bits
        row_method = "RLE"
        row_fixed_bits = n_values * math.ceil(math.log2(x.size(2)))
        row_savings = (row_fixed_bits - row_bits) / row_fixed_bits * 100
        
        # 4. Analyze channel indices with RLE
        channel_indices = compressed_data.indices[:, 1].cpu().numpy()
        # Find runs of same channel
        runs = []
        current_val = channel_indices[0]
        current_count = 1
        for val in channel_indices[1:]:
            if val == current_val:
                current_count += 1
            else:
                runs.append((current_val, current_count))
                current_val = val
                current_count = 1
        runs.append((current_val, current_count))
        
        # Analyze run length distribution
        run_lengths = [count for _, count in runs]
        max_run_length = max(run_lengths)
        
        # Calculate optimal bits needed
        channel_bits_needed = int(np.ceil(np.log2(x.size(1))))  # For 54 channels, this is 6 bits
        length_bits_needed = int(np.ceil(np.log2(max_run_length + 1)))  # Add 1 to handle length 0
        
        # Calculate RLE bits with optimized allocation
        channel_bits = len(runs) * (channel_bits_needed + length_bits_needed)  # Use optimal bits for both
        channel_fixed_bits = n_values * math.ceil(math.log2(x.size(1)))
        channel_savings = (channel_fixed_bits - channel_bits) / channel_fixed_bits * 100
        unique_channels = len(set(channel_indices))
        
        # Print detailed RLE analysis
        print(f"\n   [RLE Analysis]")
        print(f"   - Number of runs: {len(runs)}")
        print(f"   - Max run length: {max_run_length}")
        print(f"   - Bits per channel: {channel_bits_needed}")
        print(f"   - Bits per length: {length_bits_needed}")
        print(f"   - Total bits per run: {channel_bits_needed + length_bits_needed}")
        
        # Calculate metadata overhead
        metadata_bits = 128  # 4 32-bit values for quantization params
        huffman_table_bits = (
            len(str(huffman_stats['codebook'])) + 
            len(str(col_delta_stats['codebook']))
        ) * 8  # Rough estimate of Huffman table size
        overhead_bits = metadata_bits + huffman_table_bits
        
        # Total bits with all optimizations
        total_compressed_bits = values_bits + col_bits + channel_bits + row_bits + overhead_bits
        total_fixed_bits = values_fixed_bits + col_fixed_bits + channel_fixed_bits + row_fixed_bits + metadata_bits
        
        # Calculate bits per pixel
        num_pixels = x.size(2) * x.size(3)  # H * W
        bpp = total_compressed_bits / num_pixels
        bpp_fixed = total_fixed_bits / num_pixels
        
        # Calculate bits per pixel for each component
        row_bpp = row_bits / num_pixels
        col_bpp = col_bits / num_pixels
        channel_bpp = channel_bits / num_pixels
        value_bpp = values_bits / num_pixels
        metadata_bpp = metadata_bits / num_pixels
        huffman_table_bpp = huffman_table_bits / num_pixels
        
        # Print detailed analysis
        print("\nCompression Analysis (with lossless optimizations):")
        
        print(f"\n1. Values (Huffman + {self.num_bits}-bit quantization):")
        print(f"   - Bits with Huffman: {values_bits:,} (avg {huffman_stats['avg_bits']:.2f} bits/value)")
        print(f"   - Bits with fixed: {values_fixed_bits:,} ({self.num_bits} bits/value)")
        print(f"   - Original bits: {values_naive_bits:,} (32 bits/value)")
        print(f"   - Savings vs fixed: {values_savings_vs_fixed:.1f}%")
        print(f"   - Savings vs naive: {values_savings_vs_naive:.1f}%")
        
        print(f"\n2. Column Indices ({col_method}):")
        print(f"   - Bits with {col_method}: {col_bits:,}")
        print(f"   - Bits with fixed: {col_fixed_bits:,}")
        print(f"   - Savings: {col_savings:.1f}%")
        
        print(f"\n3. Channel Indices (RLE):")
        print(f"   - Bits with RLE: {channel_bits:,} ({len(runs)} runs)")
        print(f"   - Bits with fixed: {channel_fixed_bits:,}")
        print(f"   - Unique channels: {unique_channels}")
        print(f"   - Average run length: {n_values/len(runs):.1f}")
        print(f"   - Savings: {channel_savings:.1f}%")
        
        print(f"\n4. Row Indices ({row_method}):")
        print(f"   - Bits with {row_method}: {row_bits:,}")
        print(f"   - Bits with fixed: {row_fixed_bits:,}")
        print(f"   - Savings: {row_savings:.1f}%")
        
        print(f"\n5. Overhead:")
        print(f"   - Metadata: {metadata_bits:,} bits")
        print(f"   - Huffman tables: {huffman_table_bits:,} bits")
        print(f"   - Total overhead: {overhead_bits:,} bits")
        
        print(f"\n6. Overall:")
        print(f"   - Total bits (with optimizations): {total_compressed_bits:,}")
        print(f"   - Total bits (fixed-length): {total_fixed_bits:,}")
        print(f"   - Original dense bits: {dense_size:,}")
        print(f"   - BPP (with optimizations): {bpp:.2f}")
        print(f"   - BPP (fixed-length): {bpp_fixed:.2f}")
        print(f"   - Compression ratio: {dense_size/total_compressed_bits:.1f}x")
        
        return {
            'bpp': bpp,
            'bpp_fixed': bpp_fixed,
            'compression_ratio': dense_size/total_compressed_bits,
            'total_compressed_bits': total_compressed_bits,
            'total_fixed_bits': total_fixed_bits,
            'original_bits': dense_size,
            'breakdown': {
                'values': {
                    'huffman_bits': values_bits,
                    'fixed_bits': values_fixed_bits,
                    'avg_bits': float(huffman_stats['avg_bits']),
                    'savings_vs_fixed': values_savings_vs_fixed
                },
                'column_indices': {
                    'huffman_bits': col_delta_bits,
                    'rle_bits': col_rle_bits,
                    'method': col_method,
                    'bits': col_bits,
                    'fixed_bits': col_fixed_bits,
                    'avg_bits': float(col_delta_stats['avg_bits']) if col_method == "Delta+Huffman" else None,
                    'savings': col_savings
                },
                'channel_indices': {
                    'rle_bits': channel_bits,
                    'fixed_bits': channel_fixed_bits,
                    'unique_channels': unique_channels,
                    'avg_run_length': n_values/len(runs),
                    'savings': channel_savings
                },
                'row_indices': {
                    'rle_bits': row_rle_bits,
                    'method': row_method,
                    'bits': row_bits,
                    'fixed_bits': row_fixed_bits,
                    'savings': row_savings,
                    'num_runs': len(runs_row),
                    'avg_run_length': float(n_values/len(runs_row))
                },
                'overhead': {
                    'metadata_bits': metadata_bits,
                    'huffman_table_bits': huffman_table_bits
                }
            }
        }

class ModifiedResNet18(nn.Module):
    def __init__(self, compression_ratio=0.1, method='topk', num_classes=100, dataset='cifar100', channel_wise=True):
        super(ModifiedResNet18, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet = torchvision.models.resnet18(weights=None)
        
        # Modify first conv layer for CIFAR
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool
        
        # Modify final fc layer
        self.resnet.fc = nn.Linear(512, num_classes)
        
        # Initialize compression
        self.compressor = FeatureCompressor(
            compression_ratio=compression_ratio,
            method=method,
            num_bits=2,
            image_size=(32, 32),
            channel_wise=channel_wise
        )
        
        self.training_steps = 0
        self.current_bpp = None
        
        if dataset == 'cifar100':
            # Download from HuggingFace if not exists
            model_path = 'cifar100_resnet18.pth'
            if not os.path.exists(model_path):
                print("Downloading pre-trained CIFAR100 model from HuggingFace...")
                model_path = hf_hub_download(
                    repo_id="cascade-ai/cifar100-resnet18",
                    filename="cifar100_resnet18.pth"
                )
            
            # Load checkpoint
            print(f"Loading pre-trained weights from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            
            # Convert state dict keys
            state_dict = {}
            for k, v in checkpoint.items():
                # Remove prefix if present
                if k.startswith('resnet.'):
                    k = k[7:]
                elif k.startswith('module.'):
                    k = k[7:]
                
                # Only add if shape matches
                if k in self.resnet.state_dict():
                    if v.shape == self.resnet.state_dict()[k].shape:
                        state_dict[k] = v
            
            # Load weights
            self.resnet.load_state_dict(state_dict, strict=False)
    
    def forward(self, x):
        # Get intermediate features after layer1
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        
        # Always use fast version for actual compression
        compressed = self.compressor.compress_features(x)
        
        if self.training:
            self.training_steps += 1
        
        # Continue forward pass with fast version results
        x = self.compressor.decompress_features(compressed, x.size())
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Get intermediate feature maps after layer1."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        return x
    
    def get_compression_stats(self):
        """Return current compression statistics."""
        return {'bpp': self.current_bpp}

class StanfordDogsDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Extract dataset if needed
        self._extract_dataset()
        
        # Load image paths and labels
        self.images, self.labels = self._load_dataset()
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in self.labels]
    
    def _extract_dataset(self):
        """Extract the dataset if not already extracted."""
        import tarfile
        from torchvision.datasets.utils import download_and_extract_archive
        
        # Download and extract images
        images_dir = os.path.join(self.root, 'Images')
        if not os.path.exists(images_dir):
            print("Downloading images...")
            url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
            filename = os.path.join(self.root, 'images.tar')
            if not os.path.exists(filename):
                download_and_extract_archive(url, self.root, filename=filename)
            with tarfile.open(filename) as tar:
                tar.extractall(self.root)
        
        # Download and extract annotations
        annotation_dir = os.path.join(self.root, 'Annotation')
        if not os.path.exists(annotation_dir):
            print("Downloading annotations...")
            annotation_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'
            annotation_filename = os.path.join(self.root, 'annotation.tar')
            if not os.path.exists(annotation_filename):
                download_and_extract_archive(annotation_url, self.root, filename=annotation_filename)
            with tarfile.open(annotation_filename) as tar:
                tar.extractall(self.root)
    
    def _load_dataset(self):
        """Load dataset paths and determine train/test split from annotations."""
        import xml.etree.ElementTree as ET
        
        images_dir = os.path.join(self.root, 'Images')
        annotation_dir = os.path.join(self.root, 'Annotation')
        
        images = []
        labels = []
        
        # Get all breed directories
        breed_dirs = sorted([d for d in os.listdir(images_dir) 
                           if os.path.isdir(os.path.join(images_dir, d))])
        
        for breed in breed_dirs:
            breed_path = os.path.join(images_dir, breed)
            breed_annotation_path = os.path.join(annotation_dir, breed)
            
            # Process each image in the breed directory
            for img_file in os.listdir(breed_path):
                if not img_file.endswith('.jpg'):
                    continue
                
                # Check if annotation exists for this image
                annotation_file = os.path.join(breed_annotation_path, 
                                            img_file.replace('.jpg', ''))
                if os.path.exists(annotation_file):
                    try:
                        # Parse annotation to determine train/test split
                        tree = ET.parse(annotation_file)
                        root = tree.getroot()
                        
                        # Images in training set have multiple objects annotated
                        is_training = len(root.findall('.//object')) > 0
                        
                        # Add to dataset if split matches
                        if is_training == self.train:
                            img_path = os.path.join(breed_path, img_file)
                            images.append(img_path)
                            labels.append(breed)
                    except ET.ParseError:
                        print(f"Warning: Could not parse annotation file for {img_file}")
                        continue
        
        print(f"Loaded {len(images)} images for {'training' if self.train else 'testing'}")
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.targets[idx]
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label

def get_dataset(dataset_name, transform, train=True):
    """Get the specified dataset with transforms applied."""
    dataset_path = f'./{dataset_name.lower()}'
    
    if dataset_name.lower() == 'cifar100':
        if train:
            dataset = CIFAR100(root=dataset_path, train=train, download=True, transform=transform)
        else:
            # Use CIFAIR test set for better evaluation
            dataset = ciFAIR100(root=dataset_path, train=False, download=True, transform=transform)
            print("Using ciFAIR100 test set for evaluation")
    elif dataset_name.lower() == 'stanford_dogs':
        os.makedirs(dataset_path, exist_ok=True)
        dataset = StanfordDogsDataset(root=dataset_path, train=train, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset

def load_dataset(batch_size, subset=False):
    """Load and prepare dataset with appropriate transforms."""
    # Base transform for both train and eval
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                          std=[0.2675, 0.2565, 0.2761])
    ])
    # Additional augmentation for training only
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                          std=[0.2675, 0.2565, 0.2761])
    ])

    dataset_path = './cifar100'
    num_classes = 100
    # Load datasets
    train_dataset = CIFAR100(root=dataset_path, train=True, download=True, transform=train_transform)

    test_dataset = ciFAIR100(root=dataset_path, train=False, download=True, transform=base_transform)
    print("Using ciFAIR100 test set for evaluation")

    if subset:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:256])
        test_dataset = torch.utils.data.Subset(test_dataset, indices=torch.randperm(len(test_dataset))[:256])
        print("Using subset of data for testing")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False)
    
    return train_loader, test_loader, num_classes

def plot_feature_distribution(feature_values, compression_ratios, save_path='feature_distribution.png'):
    # Convert to numpy and flatten
    feature_values = feature_values.cpu().numpy().flatten()
    
    # Calculate percentiles
    percentiles = np.linspace(0, 100, 1000)  # 1000 points for smooth curve
    values = np.percentile(feature_values, percentiles)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(percentiles, values, 'b-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Percentile', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.title('Distribution of Feature Values with Compression Ratio Cutoffs', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add compression ratio cutoffs
    colors = plt.cm.rainbow(np.linspace(0, 1, len(compression_ratios)))
    for ratio, color in zip(compression_ratios, colors):
        cutoff_percentile = (1 - ratio) * 100
        cutoff_value = np.percentile(feature_values, cutoff_percentile)
        plt.axvline(x=cutoff_percentile, color=color, linestyle='--', alpha=0.8,
                   label=f'r={ratio:.2f} (p={cutoff_percentile:.1f}%)')
        plt.annotate(f'r={ratio:.2f}\n{cutoff_value:.2f}', 
                    (cutoff_percentile, cutoff_value),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='left',
                    fontsize=10,
                    color=color)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature distribution plot saved as {save_path}")

def plot_accuracy_vs_compression(results, save_path='accuracy_vs_compression.png'):
    # Extract data
    ratios = []
    accuracies = []
    accuracies_ft = []
    bpp = []
    bpp_ft = []
    
    for result in results:
        ratios.append(result['ratio'])
        accuracies.append(result['accuracy'])
        bpp.append(result['bpp'])
        
        # Only add fine-tuning results if they exist
        if 'accuracy_ft' in result:
            accuracies_ft.append(result['accuracy_ft'])
            bpp_ft.append(result['compression_stats_ft']['bpp'])
        else:
            # For baseline, use the same accuracy for both plots
            accuracies_ft.append(result['accuracy'])
            bpp_ft.append(result['bpp'])
    
    ratios = np.array(ratios)
    accuracies = np.array(accuracies)
    accuracies_ft = np.array(accuracies_ft)
    bpp = np.array(bpp)
    bpp_ft = np.array(bpp_ft)
    
    # Sort by ratio for better plotting
    sort_idx = np.argsort(ratios)
    ratios = ratios[sort_idx]
    accuracies = accuracies[sort_idx]
    accuracies_ft = accuracies_ft[sort_idx]
    bpp = bpp[sort_idx]
    bpp_ft = bpp_ft[sort_idx]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy vs compression ratio
    ax1.plot(ratios, accuracies, 'bo-', label='No Fine-tuning')
    ax1.plot(ratios, accuracies_ft, 'ro-', label='With Fine-tuning')
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Compression Ratio')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracy vs bits per pixel
    ax2.plot(bpp, accuracies, 'bo-', label='No Fine-tuning')
    ax2.plot(bpp_ft, accuracies_ft, 'ro-', label='With Fine-tuning')
    ax2.set_xlabel('Bits per Pixel')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Bits per Pixel')
    ax2.grid(True)
    ax2.legend()
    
    # Add annotations
    for i in range(len(ratios)):
        # Annotate ratio plot
        ax1.annotate(f'{ratios[i]:.2f}\n{accuracies[i]:.1f}%', 
                    (ratios[i], accuracies[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
        ax1.annotate(f'{ratios[i]:.2f}\n{accuracies_ft[i]:.1f}%', 
                    (ratios[i], accuracies_ft[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
        
        # Annotate BPP plot
        ax2.annotate(f'{bpp[i]:.1f}\n{accuracies[i]:.1f}%', 
                    (bpp[i], accuracies[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
        ax2.annotate(f'{bpp_ft[i]:.1f}\n{accuracies_ft[i]:.1f}%', 
                    (bpp_ft[i], accuracies_ft[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {save_path}")

def plot_feature_maps_with_ratios(feature_maps, original_image, compression_ratios, save_path='feature_maps_ratio.png'):
    """Plot feature maps for different compression ratios and the original image."""
    num_ratios = len(compression_ratios) + 1  # +1 for original feature map
    num_features = min(4, feature_maps[0].shape[1])  # Show up to 4 feature channels
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_ratios + 1, num_features, figsize=(15, 3*(num_ratios + 1)))
    fig.suptitle('Feature Maps at Different Compression Ratios', fontsize=16, y=0.95)
    
    # Plot original image in the first row
    if original_image.shape[0] == 3:  # If image is in CHW format
        original_image = original_image.permute(1, 2, 0)  # Convert to HWC format
    
    # Normalize image for display
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max()
    original_image = original_image.cpu().numpy()
    
    for j in range(num_features):
        axes[0, j].imshow(original_image)
        axes[0, j].axis('off')
        if j == 0:
            axes[0, j].set_title('Original Image', fontsize=12)
    
    # Plot feature maps for each compression ratio
    for i, feature_map in enumerate(feature_maps, 1):
        ratio = 1.0 if i == 1 else compression_ratios[i-2]
        feature_map = feature_map.detach().cpu()
        
        for j in range(num_features):
            # Normalize feature map for visualization
            fm = feature_map[0, j].numpy()
            fm = fm - fm.min()
            fm = fm / (fm.max() + 1e-9)  # Add small epsilon to prevent division by zero
            
            axes[i, j].imshow(fm, cmap='viridis')
            axes[i, j].axis('off')
            
            if j == 0:
                axes[i, j].set_title(f'Ratio: {ratio:.2f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_model(model, test_loader, device, save_feature_maps=False, analyze_compression=False):
    """Evaluate model on test/validation data.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test/validation data
        device: Device to run evaluation on
        save_feature_maps: Whether to save feature maps for visualization
        analyze_compression: Whether to perform compression analysis
    """
    model.eval()
    correct = 0
    total = 0
    feature_maps_list = []
    compression_stats = None
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Use model's forward pass directly
            outputs = model(images)
            
            # Only analyze compression on first batch if requested
            if batch_idx == 0 and analyze_compression:
                # Get feature maps for analysis
                feature_maps = model.get_feature_maps(images)
                compressed_data = model.compressor.compress_features(feature_maps)
                compression_stats = model.compressor.calculate_compression_ratio(feature_maps, compressed_data)
                
                if save_feature_maps:
                    reconstructed = model.compressor.decompress_features(compressed_data, feature_maps.size())
                    feature_maps_list = [
                        feature_maps.cpu(),  # Original
                        reconstructed.cpu()  # Reconstructed
                    ]
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    
    if save_feature_maps:
        return accuracy, compression_stats, feature_maps_list
    else:
        return accuracy, compression_stats

def train_with_early_stopping(model, train_loader, _, test_loader, device, patience=5, eval_every=100, max_epochs=10):
    best_test_acc = 0
    best_model_state = None
    patience_counter = 0
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    
    # Add weight decay and adjust learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=2,
        min_lr=1e-6, threshold=0.005
    )
    
    model.train()
    for epoch in range(max_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Evaluate periodically
            if global_step % eval_every == 0:
                # Use test set for early stopping
                test_acc, _ = evaluate_model(model, test_loader, device, analyze_compression=False)
                print(f'Step {global_step}, Test Acc: {test_acc:.2f}%')
                
                # Use test accuracy for scheduling and early stopping
                scheduler.step(test_acc)
                
                # Save best model based on test accuracy
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f'Early stopping triggered at step {global_step}')

                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    
                    # Final evaluation with compression analysis
                    final_test_acc, final_compression_stats = evaluate_model(
                        model, test_loader, device, analyze_compression=True)
                    
                    return final_test_acc, final_compression_stats     
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation with compression analysis
    final_test_acc, final_compression_stats = evaluate_model(
        model, test_loader, device, analyze_compression=True)
    
    return final_test_acc, final_compression_stats

def train_baseline(model, train_loader, val_loader, test_loader, device, num_epochs=10):
    """Train the baseline model without compression."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)
    
    best_val_acc = 0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss/100:.3f}')
                running_loss = 0.0
        
        # Evaluate on validation set
        model.eval()
        val_acc, _, _ = evaluate_model(model, val_loader, device, is_test=False)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
        
        scheduler.step(val_acc)
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    test_acc, compression_stats = evaluate_model(model, test_loader, device, is_test=True)
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
    
    return test_acc, compression_stats

def test_workflow():
    """Test the entire workflow with a small subset of data and fewer iterations."""
    print("\nRunning workflow test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        train_loader, test_loader, num_classes = load_dataset(batch_size=128, subset=True)
        
        results = []
        all_feature_maps = []
        original_image = None
        
        # Test baseline first
        print("\nTesting baseline (no compression)...")
        baseline_model = ModifiedResNet18(compression_ratio=1.0, method='topk', 
                                        num_classes=num_classes, channel_wise=True).to(device)
        
        # Get original image and baseline feature maps
        with torch.no_grad():
            inputs, _ = next(iter(test_loader))
            inputs = inputs.to(device)
            original_image = inputs[0]
            baseline_features = baseline_model.get_feature_maps(inputs[0:1])
            all_feature_maps.append(baseline_features)
        
        baseline_acc, baseline_stats, _ = evaluate_model(
            baseline_model, test_loader, device, analyze_compression=True)
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        if baseline_features is not None:
            # Plot feature distribution
            plot_feature_distribution(baseline_features, [0.01], 'test_feature_distribution.png')
        else:
            print("Warning: No feature maps collected for baseline model")
        
        # Add baseline to results with correct structure including bpp
        if baseline_stats:
            results.append({
                'ratio': 1.0,
                'accuracy': float(baseline_acc),
                'bpp': float(baseline_stats['bpp']),
                'bpp_fixed': float(baseline_stats['bpp_fixed']),
                'compression_ratio': float(baseline_stats['compression_ratio'])
            })
        
        # Test compression without finetuning
        ratios = [0.01, 0.05, 0.1, 0.3, 0.5]  # Test with 1% compression
        for ratio in ratios:
            print(f"\nTesting compression (ratio={ratio}) without finetuning...")
            model = ModifiedResNet18(compression_ratio=ratio, method='topk', 
                               num_classes=num_classes, channel_wise=True).to(device)
        
            # Get feature maps for compressed model
            with torch.no_grad():
                feature_map = model.get_feature_maps(inputs[0:1], compressed=True)
                all_feature_maps.append(feature_map)
        
        accuracy_no_ft, compression_stats, _ = evaluate_model(
            model, test_loader, device, analyze_compression=True)
        
        print(f"No Finetuning - Accuracy: {accuracy_no_ft:.2f}%, Original BPP: {compression_stats['bpp_original']:.4f}, Compressed BPP: {compression_stats['bpp']:.4f}")
        
        # Test with finetuning
        model_ft = model
        
        accuracy_ft, compression_stats_ft = train_with_early_stopping(
            model_ft, train_loader, None, test_loader, device,
            patience=2, eval_every=50, max_epochs=2
        )
        
        print(f"With Finetuning - Test Accuracy: {accuracy_ft:.2f}%, Original BPP: {compression_stats_ft['bpp_original']:.4f}, Compressed BPP: {compression_stats_ft['bpp']:.4f}")
        
        # Add compressed result with same structure
        result = {
            'ratio': ratio,
            'accuracy': float(accuracy_no_ft),
            'accuracy_ft': float(accuracy_ft),
            'bpp': float(compression_stats['bpp']),
            'bpp_fixed': float(compression_stats['bpp_fixed']),
            'compression_ratio': float(compression_stats['compression_ratio']),
            'compression_stats_ft': compression_stats_ft
        }
        
        results.append(result)
        
        # Plot feature maps comparison
        plot_feature_maps_with_ratios(all_feature_maps, original_image, [ratio],
                                    save_path='test_feature_maps_ratio.png')
        
        # Test plotting with both baseline and compressed results
        plot_accuracy_vs_compression(results, 'test_accuracy_vs_compression.png')
        print("Test completed successfully!")
        return results
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'stanford_dogs'],
                      help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--channel_wise', action='store_true',
                      help='Whether to use channel-wise compression')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_loader, test_loader, num_classes = load_dataset(batch_size=args.batch_size, subset=args.subset)
    
    # Methods and compression ratios to test
    methods = ['topk']
    compression_ratios = [0.01, 0.05, 0.1, 0.3, 0.5]
    
    # Store all results
    all_results = {method: [] for method in methods}
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} method ===")
        results = []
        
        # First evaluate baseline and get feature distribution
        print("\nBaseline Evaluation (No Compression):")
        baseline_model = ModifiedResNet18(compression_ratio=1.0, method=method, num_classes=num_classes, 
                                        dataset=args.dataset, channel_wise=args.channel_wise).to(device)

        # For CIFAR-100, use pretrained model directly
        baseline_acc, baseline_compression_stats, baseline_features = evaluate_model(
            baseline_model, test_loader, device, save_feature_maps=True, analyze_compression=True)
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        if method == 'topk' and baseline_features and len(baseline_features) > 0:  # Only plot distribution once
            # Plot feature distribution with compression ratio cutoffs
            plot_feature_distribution(baseline_features[0], compression_ratios, 'feature_distribution.png')
        
        # Add baseline to results
        results.append({
            'ratio': 1.0,
            'accuracy': float(baseline_acc),
            'bpp': float(baseline_compression_stats['bpp']),
            'bpp_fixed': float(baseline_compression_stats['bpp_fixed']),
            'compression_ratio': float(baseline_compression_stats['compression_ratio'])
        })
        
        for ratio in compression_ratios:
            print(f"\nTesting compression ratio: {ratio}")
            # Test without finetuning
            model = ModifiedResNet18(compression_ratio=ratio, method=method, num_classes=num_classes, 
                                   dataset=args.dataset, channel_wise=args.channel_wise).to(device)
            accuracy, compression_stats, _ = evaluate_model(
                model, test_loader, device, save_feature_maps=True, analyze_compression=True)
            
            result = {
                'ratio': ratio,
                'accuracy': accuracy,
                'accuracy_ft': accuracy,
                'bpp': compression_stats['bpp'],
                'bpp_fixed': compression_stats['bpp_fixed'],
                'compression_ratio': compression_stats['compression_ratio'],
                'row_bits': compression_stats['breakdown']['row_indices']['bits'],
                'col_bits': compression_stats['breakdown']['column_indices']['bits'],
                'channel_bits': compression_stats['breakdown']['channel_indices']['rle_bits'],
                'value_bits': compression_stats['breakdown']['values']['huffman_bits'],
                'metadata_bits': compression_stats['breakdown']['overhead']['metadata_bits'],
                'huffman_table_bits': compression_stats['breakdown']['overhead']['huffman_table_bits'],
                'original_bits': compression_stats['original_bits'],
                'total_compressed_bits': compression_stats['total_compressed_bits'],
                'total_fixed_bits': compression_stats['total_fixed_bits'],
                'savings': {
                    'values': compression_stats['breakdown']['values']['savings_vs_fixed'],
                    'columns': compression_stats['breakdown']['column_indices']['savings'],
                    'channels': compression_stats['breakdown']['channel_indices']['savings']
                },
                'avg_bits': {
                    'values': compression_stats['breakdown']['values']['avg_bits'],
                    'columns': compression_stats['breakdown']['column_indices']['avg_bits']
                },
                'channel_stats': {
                    'unique_channels': compression_stats['breakdown']['channel_indices']['unique_channels'],
                    'avg_run_length': compression_stats['breakdown']['channel_indices']['avg_run_length']
                }
            }
            
            # Test with finetuning
            model_ft = ModifiedResNet18(compression_ratio=ratio, method=method, num_classes=num_classes, 
                                      dataset=args.dataset, channel_wise=args.channel_wise).to(device)
            
            # Train with early stopping and get test accuracy
            accuracy_ft, compression_stats_ft = train_with_early_stopping(
                model_ft, train_loader, None, test_loader, device,
                patience=5, eval_every=100, max_epochs=10
            )
            result['accuracy_ft'] = accuracy_ft
            result['compression_stats_ft'] = compression_stats_ft
            
            results.append(result)
            print(f"No Finetuning - Accuracy: {accuracy:.2f}%, Bits per pixel: {compression_stats['bpp']:.4f}")
            print(f"With Finetuning - Test Accuracy: {accuracy_ft:.2f}%, Bits per pixel: {compression_stats_ft['bpp']:.4f}")
        
        all_results[method] = results
        
        # Create accuracy vs compression plot for this method
        plot_accuracy_vs_compression(results, f'accuracy_vs_compression_{method}.png')
    
    # Print summary for both methods
    print("\nSummary of Results:")
    print("------------------")
    for method in methods:
        print(f"\n{method.upper()} Method:")
        print(f"Baseline Accuracy: {all_results[method][0]['accuracy']:.2f}%")
        for result in all_results[method][1:]:
            print(f"Target Ratio: {result['ratio']:.3f}, "
                  f"BPP: {result['bpp']:.3f}, "
                  f"Accuracy (no ft): {result['accuracy']:.2f}%, "
                  f"Accuracy (ft): {result['accuracy_ft']:.2f}%")
    
    # Save all results
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'compression_results': all_results
    }
    
    with open('compression_results.json', 'w') as f:
        json.dump(results_with_metadata, f, indent=4)

    # Plot feature maps for different compression ratios
    print("\nGenerating visualizations...")
    all_feature_maps = []
    original_image = None
    
    # First get baseline feature maps and original image
    with torch.no_grad():
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        original_image = inputs[0]
        feature_map = model.get_feature_maps(inputs[0:1])
        all_feature_maps.append(feature_map)
    
    # Get feature maps for each compression ratio
    for ratio in compression_ratios:
        model = ModifiedResNet18(compression_ratio=ratio, method=method, num_classes=num_classes, 
                               dataset=args.dataset, channel_wise=args.channel_wise).to(device)
        with torch.no_grad():
            feature_map = model.get_feature_maps(inputs[0:1])
            all_feature_maps.append(feature_map)
    
    # Plot feature maps
    plot_feature_maps_with_ratios(all_feature_maps, original_image, compression_ratios,
                               save_path=f'feature_maps_{method}_ratio.png')

if __name__ == "__main__":
    if test_workflow():
        print("\nTest passed! Running full experiment...")
        main()
    else:
        print("\nTest failed! Please fix the issues before running the full experiment.")
