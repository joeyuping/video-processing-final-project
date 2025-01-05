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
from torch.nn.functional import pad as F

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
        # Convert tensor to flattened list of integers
        data_list = data.flatten().tolist()
        encoded = ''.join(codec[int(round(x))] for x in data_list)
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
    def __init__(self, compression_ratio=0.1, num_bits=8, image_size=(32, 32), mode='channel-wise', block_size=4):
        self.compression_ratio = compression_ratio
        self.num_bits = num_bits
        self.image_height, self.image_width = image_size
        self.block_size = block_size
        self.mode = mode

    def compress_features(self, x, return_block_energy=False):
        """Compress features using block sparsity."""
        batch_size, channels, height, width = x.size()
        
        # Zero out border features
        x_inner = x.clone()
        x_inner[:, :, 0, :] = 0  # Top border
        x_inner[:, :, -1, :] = 0  # Bottom border
        x_inner[:, :, :, 0] = 0  # Left border 
        x_inner[:, :, :, -1] = 0  # Right border
        
        # Calculate padding if needed
        pad_h = (self.block_size - height % self.block_size) % self.block_size
        pad_w = (self.block_size - width % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            x_inner = F.pad(x_inner, (0, pad_w, 0, pad_h))
        
        # Calculate number of blocks
        h_blocks = (height + pad_h) // self.block_size
        w_blocks = (width + pad_w) // self.block_size
        
        # Reshape into blocks
        blocks = x_inner.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        blocks = blocks.contiguous().view(batch_size, channels, h_blocks, w_blocks, self.block_size, self.block_size)
        
        indices_list = []
        values_list = []
        
        if self.mode == 'global':
            # Calculate block energies for all modes
            block_energies = blocks.pow(2).sum(dim=(-1, -2))  # [B, C, H_b, W_b]
            # Calculate total blocks to keep across all channels
            total_blocks = h_blocks * w_blocks * channels
            k = max(1, int(total_blocks * self.compression_ratio))
            
            # Reshape block energies to [B, C*H_b*W_b]
            block_energies_flat = block_energies.reshape(batch_size, -1)
            
            # Get top k blocks globally for each batch
            _, global_indices = torch.topk(block_energies_flat, k=k, dim=1)  # [B, k]
            
            # Convert global indices to (channel, h, w) coordinates
            c_idx = (global_indices // (h_blocks * w_blocks)).long()  # [B, k]
            hw_idx = global_indices % (h_blocks * w_blocks)  # [B, k]
            h_block = (hw_idx // w_blocks).long()  # [B, k]
            w_block = (hw_idx % w_blocks).long()  # [B, k]
            
            # Convert block coordinates to feature coordinates
            h_indices = h_block * self.block_size  # [B, k]
            w_indices = w_block * self.block_size  # [B, k]
            
            # Create batch indices
            b_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)  # [B, k]
            
            # Stack indices
            all_indices = torch.stack([
                b_indices.reshape(-1),
                c_idx.reshape(-1),
                h_indices.reshape(-1),
                w_indices.reshape(-1)
            ])  # [4, B*k]
            
            # Get block values efficiently
            linear_idx = (b_indices * channels * h_blocks * w_blocks + 
                         c_idx * h_blocks * w_blocks +
                         h_block * w_blocks +
                         w_block)  # [B, k]
            
            # Reshape blocks for efficient gathering
            blocks_reshaped = blocks.reshape(-1, self.block_size, self.block_size)  # [B*C*H_b*W_b, block_size, block_size]
            block_values = blocks_reshaped[linear_idx.reshape(-1)]  # [B*k, block_size, block_size]
            
            # Reshape values
            all_values = block_values.reshape(-1, self.block_size * self.block_size)  # [B*k, block_size*block_size]
            
            indices_list.append(all_indices)
            values_list.append(all_values)

        elif self.mode == 'channel-wise':
            # Calculate block energies
            block_energies = blocks.pow(2).sum(dim=(-1, -2))  # [B, C, H_b, W_b]
            
            # Calculate number of blocks to keep
            total_blocks = h_blocks * w_blocks
            k = max(1, int(total_blocks * self.compression_ratio))

            # Process all channels at once
            # Reshape block_energies to [B*C, H_b*W_b]
            block_energies_flat = block_energies.reshape(batch_size * channels, -1)
            
            # Get top k blocks for each batch-channel combination
            _, block_indices = torch.topk(block_energies_flat, k=min(k, total_blocks), dim=1)  # [B*C, k]
            
            # Create batch and channel indices
            batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(channels)  # [B*C]
            channel_idx = torch.arange(channels, device=x.device).repeat(batch_size)  # [B*C]
            
            # Convert to block coordinates
            h_block = block_indices // w_blocks  # [B*C, k]
            w_block = block_indices % w_blocks   # [B*C, k]
            
            # Convert block coordinates to feature coordinates
            h_indices = h_block * self.block_size  # [B*C, k]
            w_indices = w_block * self.block_size  # [B*C, k]
            
            # Expand batch and channel indices to match block indices
            b_indices = batch_idx.unsqueeze(1).expand(-1, block_indices.size(1))    # [B*C, k]
            c_indices = channel_idx.unsqueeze(1).expand(-1, block_indices.size(1))  # [B*C, k]
            
            # Get block values efficiently
            # Convert indices to linear indices for gather operation
            linear_idx = (b_indices * channels * h_blocks * w_blocks + 
                        c_indices * h_blocks * w_blocks +
                        h_block * w_blocks +
                        w_block)  # [B*C, k]
            
            # Reshape blocks for efficient gathering
            blocks_reshaped = blocks.reshape(-1, self.block_size, self.block_size)  # [B*C*H_b*W_b, block_size, block_size]
            block_values = blocks_reshaped[linear_idx.reshape(-1)]  # [B*C*k, block_size, block_size]
            
            # Stack indices and reshape values
            all_indices = torch.stack([
                b_indices.reshape(-1),
                c_indices.reshape(-1),
                h_indices.reshape(-1),
                w_indices.reshape(-1)
            ])  # [4, B*C*k]
            all_values = block_values.reshape(-1, self.block_size * self.block_size)  # [B*C*k, block_size*block_size]
            
            indices_list.append(all_indices)
            values_list.append(all_values)
        elif self.mode == 'region-wise':
            # Calculate block energies
            # Sum energies across channels for each block
            block_energies = blocks.pow(2).sum(dim=(1, 4, 5))  # [B, H_b, W_b]
            
            # Calculate number of blocks to keep
            total_blocks = h_blocks * w_blocks
            k = max(1, int(total_blocks * self.compression_ratio))
            
            # Global block selection across channels
            block_energies_flat = block_energies.view(batch_size, -1)  # [B, H_b*W_b]
            _, indices = torch.topk(block_energies_flat, k=k, dim=1)  # [B, k]
            
            # Convert to block coordinates
            h_block = indices // w_blocks                # [B, k]
            w_block = indices % w_blocks                 # [B, k]
            
            # Convert block coordinates to feature coordinates
            h_indices = h_block * self.block_size      # [B, k]
            w_indices = w_block * self.block_size      # [B, k]
            
            # Create batch indices and expand for all channels
            b_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)  # [B, 1]
            b_indices = b_indices.expand(-1, k)  # [B, k]
            
            # Create channel indices for all selected blocks
            c_indices = torch.arange(channels, device=x.device)  # [C]
            
            # Expand indices for all channels
            b_indices = b_indices.unsqueeze(1).expand(-1, channels, -1)  # [B, C, k]
            h_indices = h_indices.unsqueeze(1).expand(-1, channels, -1)  # [B, C, k]
            w_indices = w_indices.unsqueeze(1).expand(-1, channels, -1)  # [B, C, k]
            c_indices = c_indices.view(1, -1, 1).expand(batch_size, -1, k)  # [B, C, k]
            
            # Reshape indices for gathering
            b_indices = b_indices.reshape(-1)  # [B*C*k]
            c_indices = c_indices.reshape(-1)  # [B*C*k]
            h_indices = h_indices.reshape(-1)  # [B*C*k]
            w_indices = w_indices.reshape(-1)  # [B*C*k]
            
            # Get block values efficiently
            blocks_reshaped = blocks.reshape(-1, self.block_size, self.block_size)  # [B*C*H_b*W_b, block_size, block_size]
            
            # Calculate linear indices for gathering
            linear_idx = (b_indices * channels * h_blocks * w_blocks + 
                        c_indices * h_blocks * w_blocks +
                        (h_indices // self.block_size) * w_blocks +
                        (w_indices // self.block_size))
            
            # Gather block values
            block_values = blocks_reshaped[linear_idx]  # [B*C*k, block_size, block_size]
            
            # Stack indices and reshape values
            all_indices = torch.stack([b_indices, c_indices, h_indices, w_indices])  # [4, B*C*k]
            all_values = block_values.reshape(-1, self.block_size * self.block_size)  # [B*C*k, block_size*block_size]
            
            indices_list.append(all_indices)
            values_list.append(all_values)

        # Concatenate all indices and values
        all_indices = torch.cat(indices_list, dim=1)  # [4, N]
        all_values = torch.cat(values_list, dim=0)  # [N, block_size*block_size]
        
        metadata = {
            'block_size': self.block_size,
            'h_blocks': h_blocks,
            'w_blocks': w_blocks,
            'padding': (pad_h, pad_w),
            'original_shape': (batch_size, channels, height, width)
        }
        
        if return_block_energy:
            return EncodedData(
                values=all_values.to(dtype=torch.float32),
                indices=all_indices,
                metadata=metadata,
                huffman_tables=None,
                rle_data=None,
                raw_rows=None,
            ), block_energies
        else:
            return EncodedData(
                values=all_values.to(dtype=torch.float32),
                indices=all_indices,
                metadata=metadata,
                huffman_tables=None,
                rle_data=None,
                raw_rows=None
            )
    
    def decompress_features(self, compressed_data):
        """Decompress features using block sparsity."""
        metadata = compressed_data.metadata
        original_shape = metadata['original_shape']
        device = compressed_data.values.device
        block_size = metadata['block_size']
        
        # Extract indices
        indices = compressed_data.indices  # [4, N]
        b_idx = indices[0].long()  # [N]
        c_idx = indices[1].long()  # [N]
        h_idx = indices[2].long()  # [N]
        w_idx = indices[3].long()  # [N]
        
        # Create output tensor
        output = torch.zeros(original_shape, device=device, dtype=torch.float32)
        
        # Reshape values into blocks
        block_values = compressed_data.values.view(-1, block_size, block_size)  # [N, block_size, block_size]
        
        # Create block offsets once
        block_offsets = torch.arange(block_size, device=device)
        h_offsets = h_idx.unsqueeze(1) + block_offsets.unsqueeze(0)  # [N, block_size]
        w_offsets = w_idx.unsqueeze(1) + block_offsets.unsqueeze(0)  # [N, block_size]
        
        # Expand indices for block assignment
        b_idx_exp = b_idx.unsqueeze(1).unsqueeze(2).expand(-1, block_size, block_size)  # [N, block_size, block_size]
        c_idx_exp = c_idx.unsqueeze(1).unsqueeze(2).expand(-1, block_size, block_size)  # [N, block_size, block_size]
        h_offsets_exp = h_offsets.unsqueeze(2).expand(-1, -1, block_size)  # [N, block_size, block_size]
        w_offsets_exp = w_offsets.unsqueeze(1).expand(-1, block_size, -1)  # [N, block_size, block_size]
        
        # Assign blocks in one operation
        output[b_idx_exp, c_idx_exp, h_offsets_exp, w_offsets_exp] = block_values
        
        return output
    
    def calculate_compression_ratio(self, compressed_data, original_size):
        """Calculate compression ratio and bits per pixel with optimized bit allocation."""
        batch_size, channels, height, width = original_size
        
        # 1. Quantize values using specified number of bits
        values = compressed_data.values
        value_range = values.max() - values.min()
        scale = (2**self.num_bits - 1) / value_range
        quantized_values = ((values - values.min()) * scale).round().clamp(0, 2**self.num_bits - 1).to(torch.uint8)
        
        # Fast unique value counting using torch operations
        unique_values, counts = torch.unique(quantized_values, return_counts=True)
        freq_dict = {int(val): int(count) for val, count in zip(unique_values, counts)}
        
        # Generate Huffman codes once
        codec = huffman.codebook(freq_dict.items())
        
        # Calculate total bits for values using vectorized operations
        total_symbols = counts.sum().item()
        avg_bits = sum(len(codec[int(val)]) * count.item() for val, count in zip(unique_values, counts)) / total_symbols
        value_bits = int(avg_bits * total_symbols)
        
        # Initialize huffman_bits
        huffman_bits = 0
        
        # 2. Optimize indices encoding based on compression mode
        indices = compressed_data.indices
        if self.mode == 'global':
             # Create Huffman coder for indices
            huffman_coder = HuffmanCoder()
            
            # For global mode, batch indices follow a regular pattern
            # We only need to store:
            # 1. batch_size as metadata
            # 2. number of blocks per batch (k) as metadata
            # 3. Huffman-encoded channel indices
            # 4. Delta-encoded spatial coordinates
            
            # Process channel indices (more variation in global mode)
            channel_indices = indices[1]
            channel_stats = huffman_coder.fit(channel_indices, 'channel')
            encoded_channel, channel_bits = huffman_coder.encode(channel_indices, 'channel')
            
            # Group spatial coordinates by batch
            blocks_per_batch = len(channel_indices) // batch_size
            h_indices = indices[2].reshape(batch_size, blocks_per_batch)  # [B, K]
            w_indices = indices[3].reshape(batch_size, blocks_per_batch)  # [B, K]
            
            # Calculate deltas within each batch
            h_deltas = torch.zeros_like(h_indices)
            w_deltas = torch.zeros_like(w_indices)
            
            # Calculate deltas for first block in each batch
            h_deltas[:, 0] = h_indices[:, 0]
            w_deltas[:, 0] = w_indices[:, 0]
            
            # Calculate deltas for subsequent blocks
            h_deltas[:, 1:] = h_indices[:, 1:] - h_indices[:, :-1]
            w_deltas[:, 1:] = w_indices[:, 1:] - w_indices[:, :-1]
            
            # Encode spatial deltas
            h_stats = huffman_coder.fit(h_deltas.flatten(), 'height')
            w_stats = huffman_coder.fit(w_deltas.flatten(), 'width')
            
            encoded_h, h_bits = huffman_coder.encode(h_deltas.flatten(), 'height')
            encoded_w, w_bits = huffman_coder.encode(w_deltas.flatten(), 'width')
            
            # Add metadata bits
            # - 16 bits for batch_size
            # - 16 bits for blocks_per_batch (k)
            metadata_bits = 32
            
            huffman_bits = channel_bits + h_bits + w_bits + metadata_bits

        elif self.mode == 'channel-wise':
            # Create Huffman coder for indices
            huffman_coder = HuffmanCoder()
            
            # Process each component (batch, channel, height, width)
            components = ['batch', 'channel', 'height', 'width']
            component_bits = {comp: 0 for comp in components}
            
            for component_idx, comp_name in enumerate(components):
                component = indices[component_idx]
                
                # Calculate deltas
                deltas = torch.zeros_like(component)
                deltas[1:] = component[1:] - component[:-1]
                deltas[0] = component[0]
                
                # Convert deltas to integers and fit Huffman coding
                delta_stats = huffman_coder.fit(deltas, f'indices_{comp_name}')
                encoded_data, bits_used = huffman_coder.encode(deltas, f'indices_{comp_name}')
                component_bits[comp_name] = bits_used
                
                huffman_bits += bits_used
        
        elif self.mode == 'region-wise':
            # Create Huffman coder for indices
            huffman_coder = HuffmanCoder()
            
            # For region-wise, we know that each selected region uses all channels
            # So we only need to encode the spatial coordinates (h, w) for each region
            # and store metadata about batch_size and channels
            
            # Group spatial coordinates by region
            h_indices = indices[2].reshape(batch_size, -1)  # [B, R] where R is regions per batch
            w_indices = indices[3].reshape(batch_size, -1)  # [B, R]
            
            # Calculate deltas within each batch
            h_deltas = torch.zeros_like(h_indices)
            w_deltas = torch.zeros_like(w_indices)
            
            # Calculate deltas for first region in each batch
            h_deltas[:, 0] = h_indices[:, 0]
            w_deltas[:, 0] = w_indices[:, 0]
            
            # Calculate deltas for subsequent regions
            h_deltas[:, 1:] = h_indices[:, 1:] - h_indices[:, :-1]
            w_deltas[:, 1:] = w_indices[:, 1:] - w_indices[:, :-1]
            
            # Encode spatial deltas
            h_stats = huffman_coder.fit(h_deltas.flatten(), 'height')
            w_stats = huffman_coder.fit(w_deltas.flatten(), 'width')
            
            encoded_h, h_bits = huffman_coder.encode(h_deltas.flatten(), 'height')
            encoded_w, w_bits = huffman_coder.encode(w_deltas.flatten(), 'width')
            
            # Add minimal metadata bits (assuming 16 bits each for batch_size and channels)
            metadata_bits = 32  # 16 bits each for batch_size and num_channels
            
            huffman_bits = h_bits + w_bits + metadata_bits

        # Calculate final statistics
        total_compressed_bits = value_bits + huffman_bits
        total_pixels = batch_size * height * width
        original_bits = total_pixels * channels * 8
        
        bpp_original = original_bits / total_pixels
        bpp_compressed = total_compressed_bits / total_pixels
        compression_ratio = total_compressed_bits /original_bits
        
        # Return detailed statistics
        stats = {
            'bpp_original': bpp_original,
            'bpp_compressed': bpp_compressed,
            'compression_ratio': compression_ratio,
            'original_bits': original_bits,
            'compressed_bits': total_compressed_bits,
            'breakdown': {
                'values': {
                    'huffman_bits': value_bits,
                    'avg_bits': avg_bits
                },
                'indices': {
                    'total_bits': huffman_bits,
                    'mode': self.mode,
                    'encoding_scheme': 'optimized_block_locations' if self.mode == 'region-wise' else 'delta_huffman'
                }
            }
        }
        
        return compression_ratio, bpp_compressed, stats

class ModifiedResNet18(nn.Module):
    def __init__(self, compression_ratio=0.1, num_classes=100, mode='channel-wise', block_size=4):
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
            num_bits=2,
            image_size=(32, 32),
            mode=mode,
            block_size=block_size
        )
        
        self.training_steps = 0
        self.current_bpp = None
        self.is_baseline = compression_ratio >= 1.0  # Flag to identify baseline model
        
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
        """Forward pass with optional compression between layer1 and layer2."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        
        # Apply compression only if not baseline model
        if not self.is_baseline:
            compressed = self.compressor.compress_features(x)
            if self.training:
                self.training_steps += 1
            x = self.compressor.decompress_features(compressed)
        
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x
    
    def get_feature_maps(self, x, compressed=False):
        """Get intermediate feature maps after layer1 for compression analysis."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        if compressed:
            x, block_energies = self.compressor.compress_features(x, return_block_energy=True)
            x = self.compressor.decompress_features(x)
            return x, block_energies
        else:
            return x

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
        print("Using subset of 1000 images for training and testing")
    
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

def plot_feature_maps_with_ratios(feature_maps, original_image, compression_ratios, save_path='feature_maps_ratio.png'):
    """Plot feature maps for different compression ratios and the original image horizontally."""
    num_maps = len(feature_maps)  # Number of feature maps we actually have
    num_features = min(4, feature_maps[0].shape[1] if isinstance(feature_maps[0], torch.Tensor) else 4)
    
    # Create a figure with subplots - one column for each compression ratio, num_features rows
    # Reduce horizontal size by adjusting figure width
    fig = plt.figure(figsize=(2.5*(num_maps + 1), 10))  # Reduced from 4* to 2.5*
    
    # Add suptitle with more space at top
    fig.suptitle('Feature Maps at Different Compression Ratios', 
                fontsize=20, y=0.95)
    
    # Create subplot grid with more space between plots
    gs = plt.GridSpec(num_features, num_maps + 1, 
                     hspace=0.1,    # Space between rows
                     wspace=0.15)   # Reduced horizontal space between plots (from 0.3)
    
    # Create axes array - transposed for horizontal layout
    axes = [[plt.subplot(gs[i, j]) for j in range(num_maps + 1)] 
            for i in range(num_features)]
    axes = np.array(axes)
    
    # Plot original image in the first column
    if original_image.shape[0] == 3:  # If image is in CHW format
        original_image = original_image.permute(1, 2, 0)  # Convert to HWC format
    
    # Normalize image for display
    original_image = original_image.detach().cpu()
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max()
    original_image = original_image.numpy()
    
    # Plot original image in first column
    for i in range(num_features):
        axes[i, 0].imshow(original_image)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original\nImage', fontsize=14, pad=10)  # Reduced fontsize and padding
    
    # Plot feature maps for each compression ratio
    for j, feature_map in enumerate(feature_maps, 1):
        # Get ratio (1.0 for baseline, compression ratio for others)
        ratio = 1.0 if j == 1 else compression_ratios[min(j-2, len(compression_ratios)-1)]
        
        # Get feature map data based on type
        if isinstance(feature_map, torch.Tensor):
            feature_data = feature_map[0]  # Take first batch
        else:  # EncodedData
            # Decompress the feature map
            feature_data = feature_map.values.view(-1, feature_map.metadata['block_size'], 
                                                 feature_map.metadata['block_size'])
            feature_data = feature_data.reshape(1, -1, 
                                              feature_map.metadata['original_shape'][2],
                                              feature_map.metadata['original_shape'][3])[0]
        
        # Plot each feature channel
        for i in range(num_features):
            if i < feature_data.shape[0]:  # Check if channel exists
                # Normalize feature map for visualization
                fm = feature_data[i].detach().cpu().numpy()
                fm = fm - fm.min()
                fm = fm / (fm.max() + 1e-9)  # Add small epsilon to prevent division by zero
                
                axes[i, j].imshow(fm, cmap='viridis')
                axes[i, j].axis('off')
                
                if i == 0:
                    axes[i, j].set_title(f'r={ratio:.2f}', fontsize=14, pad=10)  # Simplified title, reduced fontsize
            else:
                axes[i, j].axis('off')
    
    # Add channel labels on the left with adjusted position
    for i in range(num_features):
        plt.figtext(0.01, 0.8 - i*0.23, f'Ch {i+1}',  # Shortened "Channel" to "Ch"
                   ha='left', va='center', fontsize=12)  # Reduced fontsize
    
    # Save with higher resolution and tight layout
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)  # Reduced padding
    plt.close()

def plot_accuracy_vs_compression(results, save_path='accuracy_vs_compression.png'):
    """Plot accuracy vs bits per pixel with data labels."""
    acc_no_ft = []
    acc_ft = []
    bpp_no_ft = []
    bpp_ft = []
    bpp_original = []
    
    # Process each result
    for result in results:
        # Handle baseline result (which has different structure)
        if 'ratio' in result and result['ratio'] == 1.0:
            baseline_acc = result['accuracy']
            baseline_bpp = result['bpp']
            acc_no_ft.append(baseline_acc)
            acc_ft.append(baseline_acc)
            bpp_no_ft.append(baseline_bpp)
            bpp_ft.append(baseline_bpp)
            bpp_original.append(baseline_bpp)
        else:
            acc_no_ft.append(result['accuracy_no_ft'])
            acc_ft.append(result['accuracy_ft'])
            bpp_no_ft.append(result['bpp'])
            bpp_ft.append(result['bpp'])
            bpp_original.append(result['bpp'])
    
    plt.figure(figsize=(15, 6))
    
    # Accuracy vs Compressed Bits per Pixel
    plt.subplot(1, 2, 1)
    
    # Plot points
    plt.plot(bpp_no_ft, acc_no_ft, 'bo-', label='No Fine-tuning')
    plt.plot(bpp_ft, acc_ft, 'ro-', label='With Fine-tuning')
    
    # Add data labels for each point
    for i, (bpp, acc) in enumerate(zip(bpp_no_ft, acc_no_ft)):
        ratio = results[i]['ratio'] if 'ratio' in results[i] else 1.0
        label = f'r={ratio:.2f}\nBPP={bpp:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (bpp, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    for i, (bpp, acc) in enumerate(zip(bpp_ft, acc_ft)):
        ratio = results[i]['ratio'] if 'ratio' in results[i] else 1.0
        label = f'r={ratio:.2f}\nBPP={bpp:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (bpp, acc), textcoords="offset points", 
                    xytext=(0,-25), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
    
    plt.xlabel('Compressed Bits per Pixel')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Compressed Bits per Pixel')
    plt.legend()
    plt.grid(True)
    
    # Accuracy vs Compression Ratio
    plt.subplot(1, 2, 2)
    compression_ratios = [result['ratio'] for result in results]
    
    # Plot points
    plt.plot(compression_ratios, acc_no_ft, 'bo-', label='No Fine-tuning')
    plt.plot(compression_ratios, acc_ft, 'ro-', label='With Fine-tuning')
    
    # Add data labels for each point
    for i, (ratio, acc) in enumerate(zip(compression_ratios, acc_no_ft)):
        bpp = bpp_no_ft[i]
        label = f'r={ratio:.2f}\nBPP={bpp:.1f}\nAcc={acc:.1f}%'
        plt.annotate(label, (ratio, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    for i, (ratio, acc) in enumerate(zip(compression_ratios, acc_ft)):
        bpp = bpp_ft[i]
        label = f'r={ratio:.2f}\nBPP={bpp:.1f}\nAcc={acc:.1f}%'
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

def evaluate_model(model, test_loader, calculate_compression_stats=False):
    """Evaluate model accuracy and compression performance."""
    model.eval()
    device = next(model.parameters()).device
    total = 0
    correct = 0
    feature_maps = None
    compression_stats = None
    
    # Add progress bar
    pbar = tqdm(test_loader, desc='Evaluating')
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Get compression stats only on first batch if requested
            if batch_idx == 0 and calculate_compression_stats and not model.is_baseline:
                feature_maps = model.get_feature_maps(images)
                compressed_data = model.compressor.compress_features(feature_maps)
                compression_stats = model.compressor.calculate_compression_ratio(
                    compressed_data, images.size())
            elif batch_idx == 0 and calculate_compression_stats and model.is_baseline:
                # For baseline, use dummy compression stats
                compression_stats = (1.0, 24.0, {
                    'bpp_original': 24.0,
                    'bpp_compressed': 24.0,
                    'compression_ratio': 1.0
                })
                feature_maps = model.get_feature_maps(images)
            
            # Forward pass through entire model
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Update accuracy metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar with current accuracy
            current_acc = 100. * correct / total
            pbar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})
    
    final_accuracy = 100. * correct / total
    return final_accuracy, compression_stats, feature_maps


def train_with_early_stopping(model, train_loader, _, test_loader, device, patience=5, eval_every=100, max_epochs=100):
    """Train model with early stopping and compression."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    best_acc = 0
    patience_counter = 0
    step = 0
    
    for epoch in range(max_epochs):
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # Slightly more efficient than zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # Evaluate periodically
            if step % eval_every == 0:
                model.eval()
                accuracy, compression_stats, _ = evaluate_model(model, test_loader, calculate_compression_stats=False)
                model.train()
                
                # Update learning rate
                scheduler.step(accuracy)
                
                if accuracy > best_acc:
                    best_acc = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at step {step}")
                    return best_acc, compression_stats
    
    # Final evaluation
    model.eval()
    accuracy, compression_stats, _ = evaluate_model(model, test_loader, calculate_compression_stats=True)
    return accuracy, compression_stats


def plot_block_energy_distribution(block_energies, compression_ratios, save_path='block_energy_distribution.png'):
    """Plot distribution of block energies and mark the cutoff thresholds for different ratios."""
    # Detach and convert to numpy safely
    energies = block_energies.detach().flatten().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of energies with counts instead of density
    n, bins, patches = plt.hist(energies, bins=100, density=False, alpha=0.7, color='b', 
                               label=f'Block Count (total: {len(energies):,})')
    
    # Calculate and plot cutoff thresholds for each ratio
    colors = plt.cm.rainbow(np.linspace(0, 1, len(compression_ratios)))
    max_height = plt.gca().get_ylim()[1]  # Get maximum height for annotations
    
    for ratio, color in zip(compression_ratios, colors):
        # Calculate threshold for this ratio
        threshold = np.percentile(energies, (1 - ratio) * 100)
        
        # Calculate retained energy and blocks for this ratio
        blocks_kept = np.sum(energies >= threshold)
        retained_energy = np.sum(energies[energies >= threshold]) / np.sum(energies)
        
        # Add vertical line at threshold
        plt.axvline(x=threshold, color=color, linestyle='--', 
                   label=f'r={ratio:.2f} ({blocks_kept:,} blocks, {retained_energy:.1%} energy)')
        
        # Add annotation
        plt.annotate(f'r={ratio:.2f}\n{threshold:.1e}', 
                    (threshold, max_height),
                    xytext=(5, -5), textcoords='offset points',
                    ha='left', va='top',
                    color=color,
                    fontsize=8)
    
    plt.title('Block Energy Distribution with Multiple Compression Ratios', 
             fontsize=14, pad=10)
    plt.xlabel('Block Energy', fontsize=12)
    plt.ylabel('Block Count', fontsize=12)  # Changed from Density to Count
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--mode', type=str, default='channel-wise',
                       help='Compression mode: channel-wise, region-wise or global')
    parser.add_argument('--subset', action='store_true',
                       help='Use subset of data for testing')
    parser.add_argument('--block-size', type=int, default=4,
                       help='Size of blocks for block-sparse compression')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset with subset option
    train_loader, test_loader, num_classes = load_dataset(args.batch_size, subset=args.subset)
    
    # Compression ratios to test
    compression_ratios = [0.5, 0.3, 0.1, 0.05]
    results = []
    all_feature_maps = []  # For visualization
    original_image = None
    
    # First evaluate baseline and get feature distribution
    print("\nBaseline Evaluation (No Compression):")
    baseline_model = ModifiedResNet18(compression_ratio=1.0, num_classes=num_classes, 
                                    mode=args.mode, block_size=args.block_size).to(device)
    
    # Get original image and baseline feature maps from first batch
    with torch.no_grad():
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        original_image = inputs[1]
        baseline_features = baseline_model.get_feature_maps(inputs[1:2])
        all_feature_maps.append(baseline_features)  # Save for visualization
    
    # Evaluate baseline model
    baseline_acc, baseline_compression_stats, _ = evaluate_model(
        baseline_model, test_loader, calculate_compression_stats=True)
    
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    if baseline_features is not None:
        # Plot feature distribution with compression ratio cutoffs
        plot_feature_distribution(baseline_features, compression_ratios, 'feature_distribution.png')
    
    # Add baseline to results
    compression_ratio, bits_per_pixel, detailed_stats = baseline_compression_stats
    results.append({
        'ratio': 1.0,
        'accuracy': float(baseline_acc),
        'bpp': float(bits_per_pixel),
        'compression_ratio': float(compression_ratio)
    })
    
    print("Plotting feature maps with ratios and block energy distribution")
    for ratio in compression_ratios:
        model = ModifiedResNet18(compression_ratio=ratio, num_classes=num_classes, 
                               mode=args.mode, block_size=args.block_size).to(device)
        
        # Get feature maps for visualization
        with torch.no_grad():
            feature_map, block_energy = model.get_feature_maps(inputs[1:2], compressed=True)
            all_feature_maps.append(feature_map)
    
    plot_block_energy_distribution(block_energy, compression_ratios,
                                 'block_energy_dist.png')
    
    plot_feature_maps_with_ratios(all_feature_maps, original_image, compression_ratios,
                                save_path='feature_maps_ratio.png')
    
    for ratio in compression_ratios:
        print(f"\nTesting compression ratio: {ratio}")
        
        # Test without finetuning
        model = ModifiedResNet18(compression_ratio=ratio, num_classes=num_classes, 
                               mode=args.mode, block_size=args.block_size).to(device)
        # Evaluate model without finetuning
        accuracy_no_ft, compression_stats, _ = evaluate_model(
            model, test_loader, calculate_compression_stats=True)
        
        compression_ratio, bits_per_pixel, detailed_stats = compression_stats
        
        # Test with finetuning
        model_ft = ModifiedResNet18(compression_ratio=ratio, num_classes=num_classes, 
                                  mode=args.mode, block_size=args.block_size).to(device)
        
        # Train with early stopping and get test accuracy
        accuracy_ft, compression_stats_ft = train_with_early_stopping(
            model_ft, train_loader, None, test_loader, device,
            patience=5, eval_every=100, max_epochs=10
        )
        
        # Store results
        result = {
            'ratio': ratio,
            'accuracy_no_ft': accuracy_no_ft,
            'accuracy_ft': accuracy_ft,
            'bpp': bits_per_pixel,
            'compression_ratio': compression_ratio,
            'detailed_stats': detailed_stats
        }
        
        results.append(result)
        print(f"No Finetuning - Accuracy: {accuracy_no_ft:.2f}%, "
              f"Original BPP: {detailed_stats['bpp_original']:.4f}, "
              f"Compressed BPP: {bits_per_pixel:.4f}")
        
        # Check if compression_stats_ft is a tuple (compression_ratio, bpp, stats)
        if isinstance(compression_stats_ft, tuple) and len(compression_stats_ft) == 3:
            ft_bpp = compression_stats_ft[1]
            ft_stats = compression_stats_ft[2]
            print(f"With Finetuning - Test Accuracy: {accuracy_ft:.2f}%, "
                  f"Original BPP: {ft_stats['bpp_original']:.4f}, "
                  f"Compressed BPP: {ft_bpp:.4f}")
        else:
            print(f"With Finetuning - Test Accuracy: {accuracy_ft:.2f}%")
        
        # Update plots after each ratio
        plot_accuracy_vs_compression(results, f'accuracy_vs_compression.png')
    
    # Print final summary
    print("\nSummary of Results:")
    print("------------------")
    print(f"Baseline Accuracy: {results[0]['accuracy']:.2f}%")
    for result in results[1:]:
        print(f"Target Ratio: {result['ratio']:.3f}, "
              f"BPP: {result['bpp']:.3f}, "
              f"Accuracy (no ft): {result['accuracy_no_ft']:.2f}%, "
              f"Accuracy (ft): {result['accuracy_ft']:.2f}%")
    
    # Save results
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'compression_results': results
    }
    
    with open('compression_results.json', 'w') as f:
        json.dump(results_with_metadata, f, indent=4)

if __name__ == "__main__":
    main()
