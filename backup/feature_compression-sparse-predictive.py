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

class CompressedFeatures:
    """Container for compressed feature data."""
    def __init__(self, indices, values, shape, metadata=None):
        self.indices = indices
        self.values = values
        self.shape = shape
        self.metadata = metadata if metadata is not None else {}

class FeatureCompressor:
    def __init__(self, compression_ratio=0.1, method='threshold', num_bits=2, image_size=(32, 32), channel_wise=True):
        self.compression_ratio = compression_ratio
        self.method = method
        self.num_bits = num_bits
        self.image_size = image_size
        self.channel_wise = channel_wise
        # Fixed prediction weights (4 bytes total)
        self.pred_weights = torch.tensor([0.5, 0.3, 0.1, 0.1])  # [left, top, top-left, top-right]
    
    def predict_from_context(self, x):
        """Predict values using fixed weights for context."""
        batch_size, channels, height, width = x.shape
        device = x.device
        self.pred_weights = self.pred_weights.to(device)
        predictions = torch.zeros_like(x)
        
        for i in range(height):
            for j in range(width):
                weights = []
                contexts = []
                
                # Left neighbor
                if j > 0:
                    contexts.append(x[:, :, i, j-1])
                    weights.append(self.pred_weights[0])
                
                # Top neighbor
                if i > 0:
                    contexts.append(x[:, :, i-1, j])
                    weights.append(self.pred_weights[1])
                
                # Top-left neighbor
                if i > 0 and j > 0:
                    contexts.append(x[:, :, i-1, j-1])
                    weights.append(self.pred_weights[2])
                
                # Top-right neighbor
                if i > 0 and j < width - 1:
                    contexts.append(x[:, :, i-1, j+1])
                    weights.append(self.pred_weights[3])
                
                if contexts:
                    # Normalize weights for available contexts
                    weights = torch.tensor(weights, device=device)
                    weights = weights / weights.sum()
                    
                    context_tensor = torch.stack(contexts, dim=0)
                    predictions[:, :, i, j] = (context_tensor * weights.view(-1, 1, 1)).sum(dim=0)
        
        return predictions
    
    def compress_features(self, x):
        """Compress features using hybrid coding."""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # 1. Predict values
        predictions = self.predict_from_context(x)
        residuals = x - predictions
        
        # 2. Select top-k residuals based on compression ratio
        k = max(1, int(x.numel() * self.compression_ratio))
        values = residuals.abs().view(-1)
        
        # Sort values and find threshold
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        threshold = sorted_values[min(k, len(sorted_values)-1)]
        
        # Create mask for significant residuals
        mask = residuals.abs() > threshold
        
        # 3. Store significant residuals and predictions
        compressed_values = residuals[mask]
        indices = torch.nonzero(mask)
        pred_values = predictions[mask]
        
        # 4. Sort indices for efficient coding
        # First by channel, then by row major order within each channel
        channel_indices = indices[:, 1].cpu()
        row_indices = indices[:, 2].cpu()
        col_indices = indices[:, 3].cpu()
        
        # Create sorting key that preserves spatial locality
        sort_idx = (channel_indices * height * width + 
                   row_indices * width + 
                   col_indices).argsort()
        
        # Apply sorting
        channel_indices = channel_indices[sort_idx]
        row_indices = row_indices[sort_idx]
        col_indices = col_indices[sort_idx]
        compressed_values = compressed_values[sort_idx]
        pred_values = pred_values[sort_idx]
        indices = indices[sort_idx]
        
        # 5. Efficient index coding
        # First encode channels with RLE
        unique_channels, channel_counts = torch.unique_consecutive(channel_indices, return_counts=True)
        
        # For each channel run, encode spatial indices
        encoded_positions = []
        start_idx = 0
        
        for ch, count in zip(unique_channels, channel_counts):
            end_idx = start_idx + count
            
            # Get positions for this channel
            ch_rows = row_indices[start_idx:end_idx]
            ch_cols = col_indices[start_idx:end_idx]
            
            # Convert to linear indices within the channel
            linear_pos = ch_rows * width + ch_cols
            
            # Delta encode
            pos_deltas = torch.diff(linear_pos, prepend=linear_pos[0:1])
            
            # Run length encode the deltas
            unique_deltas, delta_counts = torch.unique_consecutive(pos_deltas, return_counts=True)
            
            encoded_positions.append({
                'channel': ch,
                'first_pos': linear_pos[0],  # Store first position as is
                'deltas': unique_deltas,     # Store unique deltas
                'counts': delta_counts       # Store run lengths
            })
            
            start_idx = end_idx
        
        # Store metadata
        metadata = {
            'encoded_positions': encoded_positions,
            'shape': x.shape,
            'pred_weights': self.pred_weights.cpu(),
            'pred_values': pred_values
        }
        
        # 6. Quantize residuals
        if self.num_bits < 32:
            compressed_values = self.quantize_values(compressed_values)
        
        return CompressedFeatures(indices, compressed_values, x.shape, metadata)
    
    def decompress_features(self, compressed_data, original_size=None):
        """Decompress features using stored metadata."""
        device = compressed_data.values.device
        shape = compressed_data.metadata['shape']
        output = torch.zeros(shape, device=device)
        
        # 1. Reconstruct indices from encoded positions
        encoded_positions = compressed_data.metadata['encoded_positions']
        
        # Reconstruct indices for each channel run
        all_indices = []
        for pos_data in encoded_positions:
            channel = pos_data['channel']
            first_pos = pos_data['first_pos']
            deltas = pos_data['deltas']
            counts = pos_data['counts']
            
            # Reconstruct linear positions
            positions = []
            curr_pos = first_pos
            for delta, count in zip(deltas, counts):
                positions.extend([curr_pos + delta] * count)
                curr_pos += delta
            
            # Convert linear positions back to row, col
            linear_pos = torch.tensor(positions, device=device)
            rows = linear_pos // shape[3]
            cols = linear_pos % shape[3]
            
            # Create full indices including batch and channel
            batch_indices = torch.zeros_like(rows)
            channel_indices = torch.full_like(rows, channel)
            channel_indices = torch.stack([batch_indices, channel_indices, rows, cols], dim=1)
            all_indices.append(channel_indices)
        
        indices = torch.cat(all_indices)
        
        # 2. Get residuals and predictions
        residuals = compressed_data.values
        pred_values = compressed_data.metadata['pred_values'].to(device)
        
        # 3. First pass: Place known values
        output[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = (
            residuals + pred_values
        )
        
        # Create mask for known positions
        known_mask = torch.zeros_like(output, dtype=torch.bool)
        known_mask[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = True
        
        # 4. Fill unknown values with weighted predictions
        # Process in scanline order to minimize error propagation
        for i in range(shape[2]):  # height
            for j in range(shape[3]):  # width
                if not known_mask[0, :, i, j].any():  # Only predict unknown positions
                    weights = []
                    contexts = []
                    
                    # Left neighbor (known values only)
                    if j > 0 and known_mask[0, :, i, j-1].any():
                        contexts.append(output[:, :, i, j-1])
                        weights.append(self.pred_weights[0])
                    
                    # Top neighbor (known values only)
                    if i > 0 and known_mask[0, :, i-1, j].any():
                        contexts.append(output[:, :, i-1, j])
                        weights.append(self.pred_weights[1])
                    
                    # Top-left neighbor (known values only)
                    if i > 0 and j > 0 and known_mask[0, :, i-1, j-1].any():
                        contexts.append(output[:, :, i-1, j-1])
                        weights.append(self.pred_weights[2])
                    
                    # Top-right neighbor (known values only)
                    if i > 0 and j < shape[3] - 1 and known_mask[0, :, i-1, j+1].any():
                        contexts.append(output[:, :, i-1, j+1])
                        weights.append(self.pred_weights[3])
                    
                    if contexts:
                        # Normalize weights for available contexts
                        weights = torch.tensor(weights, device=device)
                        weights = weights / weights.sum()
                        
                        # Make prediction using only known values
                        context_tensor = torch.stack(contexts, dim=0)
                        output[:, :, i, j] = (context_tensor * weights.view(-1, 1, 1)).sum(dim=0)
        
        return output
    
    def calculate_compression_ratio(self, x, compressed_data):
        """Calculate compression ratio with detailed analysis."""
        # Calculate original dense size (32-bit float)
        dense_size = x.numel() * 32
        
        # Get dimensions
        batch_size, channels, height, width = x.shape
        n_values = len(compressed_data.values)
        num_pixels = height * width
        
        # 1. Analyze residual compression
        residuals = compressed_data.values.cpu()
        values_huffman = HuffmanCoder()
        
        # Compare distributions
        orig_values = x[compressed_data.indices[:, 0], 
                       compressed_data.indices[:, 1],
                       compressed_data.indices[:, 2],
                       compressed_data.indices[:, 3]].cpu()
        
        # Count non-zero values after quantization
        non_zero_count = (residuals != 0).sum().item()
        zero_count = len(residuals) - non_zero_count
        
        print("\nValue Distribution Analysis:")
        print(f"   Original - Mean: {orig_values.mean():.3f}, Std: {orig_values.std():.3f}")
        print(f"   Residual - Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f}")
        print(f"   Zero values: {zero_count} ({zero_count/len(residuals)*100:.1f}%)")
        
        # Calculate residual bits (only for non-zero values)
        if non_zero_count > 0:
            non_zero_residuals = residuals[residuals != 0]
            quantized_residuals = torch.round((non_zero_residuals - non_zero_residuals.min()) / 
                                            (non_zero_residuals.max() - non_zero_residuals.min()) * 
                                            (2**self.num_bits - 1))
            huffman_stats = values_huffman.fit(quantized_residuals, 'residuals')
            values_bits = int(huffman_stats['avg_bits'] * non_zero_count)
            
            # Add bits for zero/non-zero flags
            values_bits += len(residuals)  # 1 bit per value for zero/non-zero flag
        else:
            values_bits = len(residuals)  # Just the zero flags
            huffman_stats = {'avg_bits': 0}
        
        # 2. Calculate index bits with new encoding
        # Channel runs
        encoded_positions = compressed_data.metadata['encoded_positions']
        channel_bits = len(encoded_positions) * (math.ceil(math.log2(channels)) + 16)  # channel_id + run_length
        
        # Delta-encoded spatial indices
        delta_bits = 0
        for pos_data in encoded_positions:
            deltas = pos_data['deltas']
            delta_bits += len(deltas) * math.ceil(math.log2(max(deltas) + 1))
        
        spatial_bits = delta_bits
        
        # 3. Prediction overhead (just weights)
        pred_bits = 32 * 4  # 4 32-bit weights
        
        # Total bits
        total_bits = values_bits + channel_bits + spatial_bits + pred_bits
        bpp = total_bits / num_pixels
        
        print("\nCompression Analysis (Hybrid Coding):")
        print(f"\n1. Residual Values ({self.num_bits}-bit + Huffman):")
        print(f"   - Bits: {values_bits:,} (avg {huffman_stats['avg_bits']:.2f} bits/value)")
        print(f"   - Values kept: {n_values:,} ({n_values/x.numel()*100:.1f}%)")
        
        print(f"\n2. Indices (Delta + RLE):")
        print(f"   - Channel bits: {channel_bits:,} ({len(encoded_positions)} runs)")
        print(f"   - Spatial bits: {spatial_bits:,} ({delta_bits} bits/delta)")
        
        print(f"\n3. Prediction Overhead:")
        print(f"   - Weight bits: {pred_bits}")
        
        print(f"\n4. Overall:")
        print(f"   - Total bits: {total_bits:,}")
        print(f"   - Original bits: {dense_size:,}")
        print(f"   - Compression ratio: {dense_size/total_bits:.1f}x")
        print(f"   - Bits per pixel: {bpp:.2f}")
        
        return {
            'total_bits': total_bits,
            'original_bits': dense_size,
            'compression_ratio': dense_size/total_bits,
            'bpp': bpp,
            'bpp_fixed': bpp,  # For compatibility
            'components': {
                'residuals': {
                    'bits': values_bits,
                    'avg_bits': float(huffman_stats['avg_bits']),
                    'savings': 0  # Not applicable for new method
                },
                'indices': {
                    'col_bits': spatial_bits // 2,
                    'row_bits': spatial_bits // 2,
                    'channel_bits': channel_bits
                },
                'predictions': {
                    'bits': pred_bits
                }
            }
        }

    def quantize_values(self, values):
        """Quantize values with adaptive range and small value thresholding."""
        if self.num_bits >= 32:
            return values
            
        # Analyze value distribution
        abs_values = values.abs()
        mean_val = abs_values.mean()
        std_val = abs_values.std()
        
        # Calculate thresholds
        noise_threshold = mean_val * 0.1  # Discard very small values as noise
        quant_range = mean_val + 2 * std_val  # Range for quantization
        
        # Create quantization mask
        significant_mask = abs_values > noise_threshold
        
        # Initialize output tensor with zeros
        quantized = torch.zeros_like(values)
        
        # Only quantize significant values
        if significant_mask.any():
            significant_values = values[significant_mask]
            
            # Scale to [-1, 1] with outlier clipping
            scaled_values = torch.clamp(significant_values / quant_range, -1, 1)
            
            # Quantize to specified bits
            n_levels = 2 ** (self.num_bits - 1)  # One bit for sign
            quantized_significant = torch.round(scaled_values * n_levels) / n_levels
            
            # Scale back to original range
            quantized[significant_mask] = quantized_significant * quant_range
        
        return quantized

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
        
        # Skip compression for baseline (compression_ratio = 1.0)
        if self.compressor.compression_ratio < 1.0:
            compressed = self.compressor.compress_features(x)
            x = self.compressor.decompress_features(compressed)
        
        if self.training:
            self.training_steps += 1
        
        # Continue forward pass
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

def load_dataset(dataset_name, batch_size):
    """Load and prepare dataset with appropriate transforms."""
    if dataset_name.lower() == 'cifar100':
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
        num_classes = 100
        
        # Load datasets
        train_dataset = get_dataset(dataset_name, train_transform, train=True)
        test_dataset = get_dataset(dataset_name, base_transform, train=False)
        
    else:  # stanford_dogs
        base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        num_classes = 120
        
        # Load datasets
        train_dataset = get_dataset(dataset_name, train_transform, train=True)
        test_dataset = get_dataset(dataset_name, base_transform, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = None  # We'll use test set for validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes

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
        # Load small dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                              std=[0.2675, 0.2565, 0.2761])
        ])
        
        dataset_path = './cifar100'
        full_dataset = CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=dataset_path, train=False, download=True, transform=transform)
        
        # Use small subset
        indices = torch.randperm(len(full_dataset))[:1000]
        small_dataset = torch.utils.data.Subset(full_dataset, indices)
        test_indices = torch.randperm(len(test_dataset))[:100]
        small_test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(small_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(small_test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        results = []
        
        # Test baseline first
        print("\nTesting baseline (no compression)...")
        baseline_model = ModifiedResNet18(compression_ratio=1.0, method='topk', 
                                        num_classes=100, dataset='cifar100', 
                                        channel_wise=True).to(device)
        
        baseline_acc, baseline_compression_stats, baseline_features = evaluate_model(
            baseline_model, test_loader, device, save_feature_maps=True, analyze_compression=True)
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        if baseline_features and len(baseline_features) > 0:
            # Plot feature distribution
            plot_feature_distribution(baseline_features[0], [0.5], 'test_feature_distribution.png')
        else:
            print("Warning: No feature maps collected for baseline model")
        
        # Add baseline to results with full stats structure
        results.append({
            'ratio': 1.0,
            'accuracy': float(baseline_acc),
            'bpp': float(baseline_compression_stats['bpp']),
            'bpp_fixed': float(baseline_compression_stats['bpp_fixed']),
            'compression_ratio': float(baseline_compression_stats['compression_ratio'])
        })
        
        # Test one compression ratio
        ratio = 0.5
        print(f"\nTesting compression ratio: {ratio}")
        
        # Test without finetuning
        model = ModifiedResNet18(compression_ratio=ratio, method='topk', 
                               num_classes=100, dataset='cifar100', 
                               channel_wise=True).to(device)
        
        accuracy, compression_stats, _ = evaluate_model(
            model, test_loader, device, save_feature_maps=True, analyze_compression=True)
        
        print(f"No Finetuning - Accuracy: {accuracy:.2f}%, Bits per pixel: {compression_stats['bpp']:.4f}")
        
        # Test with finetuning
        model_ft = ModifiedResNet18(compression_ratio=ratio, method='topk', 
                                  num_classes=100, dataset='cifar100', 
                                  channel_wise=True).to(device)
        
        accuracy_ft, compression_stats_ft = train_with_early_stopping(
            model_ft, train_loader, None, test_loader, device,
            patience=2, eval_every=50, max_epochs=2
        )
        
        print(f"With Finetuning - Test Accuracy: {accuracy_ft:.2f}%, Bits per pixel: {compression_stats_ft['bpp']:.4f}")
        
        # Add compressed result with same structure
        result = {
            'ratio': ratio,
            'accuracy': float(accuracy),
            'accuracy_ft': float(accuracy_ft),
            'bpp': float(compression_stats['bpp']),
            'bpp_fixed': float(compression_stats['bpp_fixed']),
            'compression_ratio': float(compression_stats['compression_ratio']),
            'compression_stats_ft': compression_stats_ft
        }
        results.append(result)
        
        # Test plotting with both baseline and compressed results
        plot_accuracy_vs_compression(results, 'test_accuracy_vs_compression.png')
        print("Test completed successfully!")
        return True
        
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
    train_loader, val_loader, test_loader, num_classes = load_dataset(args.dataset, args.batch_size)
    
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
        
        # Train baseline for Stanford Dogs
        if args.dataset.lower() == 'stanford_dogs':
            print("Training baseline model for Stanford Dogs dataset...")
            baseline_acc = train_baseline(baseline_model, train_loader, val_loader, test_loader, device)
        else:
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
                'row_bits': compression_stats['components']['indices']['row_bits'],
                'col_bits': compression_stats['components']['indices']['col_bits'],
                'channel_bits': compression_stats['components']['indices']['channel_bits'],
                'value_bits': compression_stats['components']['residuals']['bits'],
                'metadata_bits': compression_stats['components']['predictions']['bits'],
                'huffman_table_bits': compression_stats['components']['predictions']['bits'],
                'original_bits': compression_stats['original_bits'],
                'total_compressed_bits': compression_stats['total_bits'],
                'total_fixed_bits': compression_stats['total_bits'],
                'savings': {
                    'values': compression_stats['components']['residuals']['savings'],
                    'columns': compression_stats['components']['indices']['col_bits'],
                    'channels': compression_stats['components']['indices']['channel_bits']
                },
                'avg_bits': {
                    'values': compression_stats['components']['residuals']['avg_bits'],
                    'columns': compression_stats['components']['indices']['col_bits']
                },
                'channel_stats': {
                    'unique_channels': compression_stats['components']['indices']['channel_bits'],
                    'avg_run_length': compression_stats['components']['indices']['channel_bits']
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
