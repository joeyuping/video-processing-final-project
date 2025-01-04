import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import os
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import huffman
from datetime import datetime
from cifair import ciFAIR100
from huggingface_hub import hf_hub_download
import argparse
import heapq


@dataclass
class EncodedData:
    """Container for encoded data with various compression schemes."""
    values: torch.Tensor
    indices: torch.Tensor

class HuffmanCoder:
    """Huffman coding for quantized values and indices."""
    def __init__(self):
        self.codebooks = {}
        self.avg_bits = {}
    
    def _build_huffman_tree(self, frequencies):
        """Build a Huffman tree from frequency dict."""
        # Convert frequencies to float to avoid numpy type issues
        frequencies = {float(sym): float(freq) for sym, freq in frequencies.items()}
        
        # Create a heap where each item is [frequency, symbol, code]
        heap = [[freq, sym, ""] for sym, freq in frequencies.items()]
        heapq.heapify(heap)
        
        # If only one symbol, assign it code '0'
        if len(heap) == 1:
            return {heap[0][1]: '0'}
        
        # If no symbols, return empty codebook
        if not heap:
            return {}
        
        # Build tree by combining nodes
        while len(heap) > 1:
            # Get two nodes with lowest frequencies
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create a new internal node with combined frequency
            freq = float(left[0] + right[0])
            
            # Update codes of child nodes
            left_code = left[2] + '0'
            right_code = right[2] + '1'
            
            # Update the nodes with new codes
            left[2] = left_code
            right[2] = right_code
            
            # Push back the modified nodes
            heapq.heappush(heap, left)
            heapq.heappush(heap, right)
        
        # Extract codes from heap
        codes = {}
        for item in heap:
            if isinstance(item[1], (int, float)):
                codes[item[1]] = item[2]
        
        return codes

    def fit(self, data):
        """Build Huffman codebook from data."""
        # Convert tensor to numpy and flatten
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy().flatten()
        
        # Count frequencies
        unique, counts = np.unique(data, return_counts=True)
        frequencies = dict(zip(unique, counts))
        
        # Build Huffman tree and get codes
        codebook = self._build_huffman_tree(frequencies)
        self.codebooks['default'] = codebook
        
        # Calculate average bits per symbol
        total_bits = sum(len(code) * frequencies[float(sym)] for sym, code in codebook.items())
        total_symbols = sum(frequencies.values())
        avg_bits = total_bits / total_symbols if total_symbols > 0 else 0
        self.avg_bits['default'] = avg_bits
        
        return codebook, float(avg_bits)

    def encode(self, data):
        """Encode data using the fitted codebook."""
        if 'default' not in self.codebooks:
            raise ValueError("No codebook found. Call fit() first.")
            
        codebook = self.codebooks['default']
        
        # Convert tensor to numpy and flatten
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy().flatten()
        
        # Convert values to float for consistent lookup
        data = data.astype(np.float64)
        
        # Encode each value
        encoded = ''.join(codebook[float(val)] for val in data)
        
        return {
            'encoded': encoded,
            'total_bits': len(encoded),
            'avg_bits': self.avg_bits['default'],
            'codebook': codebook
        }

def run_length_encode(data: torch.Tensor):
    """Perform run-length encoding on sparse data using vectorized operations.
    
    Args:
        data: Input tensor to encode
    Returns:
        Tuple of (values, lengths) for the runs
    """
    # Convert to numpy for faster operations
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Find positions where values change
    value_changes = np.concatenate(([True], data[1:] != data[:-1]))
    
    # Get the values and calculate run lengths
    values = data[value_changes]
    lengths = np.diff(np.nonzero(value_changes)[0], append=len(data))
    
    return values.tolist(), lengths.tolist()

def run_length_decode(values: List[int], lengths: List[int]):
    """Decode run-length encoded data using vectorized operations.
    
    Args:
        values: List of values
        lengths: List of run lengths
    Returns:
        Decoded tensor
    """
    # Convert to numpy arrays
    values = np.array(values)
    lengths = np.array(lengths)
    
    # Create output array
    total_length = lengths.sum()
    output = np.zeros(total_length)
    
    # Calculate positions where runs start
    positions = np.cumsum(np.concatenate(([0], lengths[:-1])))
    
    # Use advanced indexing to set values
    for value, pos, length in zip(values, positions, lengths):
        output[pos:pos + length] = value
    
    return torch.from_numpy(output)

class FeatureCompressor:
    def __init__(self, compression_ratio=0.1, num_bits=8, method='topk', image_size=(32, 32), channel_wise=True):
        self.compression_ratio = compression_ratio
        self.num_bits = num_bits
        self.method = method
        self.image_height, self.image_width = image_size
        self.channel_wise = channel_wise
    
    def compress_features(self, x):
        """Fast compression for training/evaluation - includes lossy quantization but skips lossless compression."""
        device = x.device
        
        if self.method == 'topk':
            # Get total number of elements
            total_elements = x.numel()
            
            # Calculate number of elements to keep
            k = int(total_elements * self.compression_ratio)
            
            # Get topk values and indices
            if self.channel_wise:
                # Process each channel separately
                values_list = []
                indices_list = []
                
                for c in range(x.size(1)):  # For each channel
                    channel_data = x[:, c, :, :]  # Get channel data
                    channel_flat = channel_data.reshape(x.size(0), -1)  # Flatten spatial dimensions
                    
                    # Get top-k for this channel
                    channel_k = max(1, int(channel_flat.numel() * self.compression_ratio))
                    topk_values, topk_indices = torch.topk(channel_flat.abs(), k=channel_k, dim=1)
                    
                    # Get original values at topk positions
                    row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand(-1, channel_k)
                    values = channel_flat[row_indices, topk_indices]
                    
                    # Convert to 4D indices
                    h = x.size(2)
                    w = x.size(3)
                    n = torch.arange(x.size(0), device=device).view(-1, 1).expand(-1, channel_k)
                    c_idx = torch.full_like(topk_indices, c, device=device)
                    h_idx = (topk_indices // w).long()
                    w_idx = (topk_indices % w).long()
                    
                    # Stack indices
                    indices = torch.stack([n.flatten(), c_idx.flatten(), 
                                        h_idx.flatten(), w_idx.flatten()], dim=1)
                    
                    values_list.append(values.flatten())
                    indices_list.append(indices)
                
                # Concatenate results from all channels
                values = torch.cat(values_list)
                indices = torch.cat(indices_list)
                
            else:
                # Get global top-k
                values_flat = x.reshape(x.size(0), -1)
                topk_values, topk_indices = torch.topk(values_flat.abs(), k=k, dim=1)
                values = values_flat.gather(1, topk_indices)
                
                # Convert to 4D indices
                c = x.size(1)
                h = x.size(2)
                w = x.size(3)
                n = torch.arange(x.size(0), device=device).view(-1, 1).expand(-1, k)
                c_idx = (topk_indices // (h * w)).long()
                hw_idx = topk_indices % (h * w)
                h_idx = (hw_idx // w).long()
                w_idx = (hw_idx % w).long()
                
                # Stack indices
                indices = torch.stack([n.flatten(), c_idx.flatten(), 
                                    h_idx.flatten(), w_idx.flatten()], dim=1)
                values = values.flatten()
            
            return EncodedData(values=values, indices=indices)
            
        elif self.method == 'channel_wise':
            # Process each channel separately
            values_list = []
            indices_list = []
            
            for c in range(x.size(1)):
                channel_data = x[:, c, :, :]
                channel_flat = channel_data.reshape(x.size(0), -1)
                
                # Get threshold for this channel
                k = max(1, int(channel_flat.numel() * self.compression_ratio))
                threshold = torch.topk(channel_flat.abs().flatten(), k=k)[0][-1]
                
                # Get values above threshold
                mask = channel_flat.abs() > threshold
                values = channel_flat[mask]
                
                # Get indices
                indices = mask.nonzero()
                n = indices[:, 0]
                hw = indices[:, 1]
                h = (hw // x.size(3)).long()
                w = (hw % x.size(3)).long()
                c_idx = torch.full_like(n, c, device=device)
                
                # Stack indices
                indices = torch.stack([n, c_idx, h, w], dim=1)
                
                values_list.append(values)
                indices_list.append(indices)
            
            # Concatenate results
            values = torch.cat(values_list)
            indices = torch.cat(indices_list)
            
            return EncodedData(values=values, indices=indices)
        
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
    
    def decompress_features(self, compressed_features, original_size):
        """Fast decompression for training/evaluation - reconstructs quantized sparse tensor."""
        if not compressed_features:
            return torch.zeros(original_size, device=compressed_features.values.device)
        
        # Create empty tensor
        decompressed = torch.zeros(original_size, device=compressed_features.values.device)
        
        # Use index_put_ to place values back in their original positions
        decompressed.index_put_(
            (compressed_features.indices[:, 0],  # batch
             compressed_features.indices[:, 1],  # channel
             compressed_features.indices[:, 2],  # height
             compressed_features.indices[:, 3]), # width
            compressed_features.values
        )
        
        return decompressed
    
    def calculate_compression_ratio(self, x, compressed_data, verbose=False):
        """Calculate compression ratio with detailed analysis including potential lossless savings."""
        # Skip if no data
        if not compressed_data or not hasattr(compressed_data, 'values'):
            return None
        
        # Move tensors to CPU for analysis
        x = x.detach().cpu()
        values = compressed_data.values.detach().cpu()
        indices = compressed_data.indices.detach().cpu()
        
        # Get original dense size in bits
        dense_size = x.numel() * 32  # Each value is 32 bits
        
        # Calculate size of values after quantization
        values_naive_bits = int(values.numel() * self.num_bits)
        
        # Huffman coding analysis
        huffman = HuffmanCoder()
        codebook, avg_bits = huffman.fit(values)
        values_bits = int(values.numel() * avg_bits)
        values_savings_vs_naive = 100 * (1 - values_bits / values_naive_bits)
        
        # RLE analysis
        rle_runs_total = 0
        rle_bits_total = 0
        
        # Process each channel separately
        for channel in range(x.size(1)):
            channel_data = x[:, channel, :, :]
            values, lengths = run_length_encode(channel_data.flatten())
            
            # Calculate bits needed for this channel
            rle_runs_total += len(values)
            rle_bits_total += len(values) * self.num_bits  # bits for values
            rle_bits_total += len(lengths) * 16  # bits for run lengths (16-bit)
        
        # Calculate average run length
        avg_run_length = x.numel() / rle_runs_total if rle_runs_total > 0 else 0
        
        # Calculate overhead
        metadata_bits = 128  # 4 32-bit values for quantization params
        huffman_table_bits = len(str(codebook)) * 8  # Rough estimate of Huffman table size
        overhead_bits = metadata_bits + huffman_table_bits
        
        # Total bits with RLE optimization
        total_compressed_bits = float(rle_bits_total + overhead_bits)
        
        # Calculate bits per pixel
        num_pixels = float(x.size(2) * x.size(3))  # H * W
        bpp = total_compressed_bits / num_pixels
        compression_ratio = float(dense_size) / total_compressed_bits
        
        if verbose:
            print("\nCompression Analysis (with RLE optimization):")
            print(f"\n1. Values (Huffman + {self.num_bits}-bit quantization):")
            print(f"   - Bits with Huffman: {values_bits:,} (avg {avg_bits:.2f} bits/value)")
            print(f"   - Original bits: {values_naive_bits:,} (32 bits/value)")
            print(f"   - Savings vs naive: {values_savings_vs_naive:.1f}%")
            
            print(f"\n2. RLE Analysis:")
            print(f"   - Total RLE bits: {rle_bits_total:,}")
            print(f"   - Number of runs: {rle_runs_total:,}")
            print(f"   - Average run length: {avg_run_length:.1f}")
            print(f"   - Bits per run: {rle_bits_total/rle_runs_total:.1f}")
            
            print(f"\n3. Overhead:")
            print(f"   - Metadata: {metadata_bits:,} bits")
            print(f"   - Huffman tables: {huffman_table_bits:,} bits")
            print(f"   - Total overhead: {overhead_bits:,} bits")
            
            print(f"\n4. Overall:")
            print(f"   - Total compressed bits: {total_compressed_bits:,}")
            print(f"   - Original dense bits: {dense_size:,}")
            print(f"   - BPP: {bpp:.2f}")
            print(f"   - Compression ratio: {compression_ratio:.1f}x")
        
        return {
            'bpp': float(bpp),
            'compression_ratio': float(compression_ratio),
            'total_compressed_bits': float(total_compressed_bits),
            'dense_size': float(dense_size),
            'overhead_bits': float(overhead_bits),
            'rle_bits': float(rle_bits_total),
            'huffman_bits': float(values_bits)
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
            num_bits=8,
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
        if not hasattr(self, 'compressor'):
            return None
        
        # Create random input tensor on the same device as the model
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        
        # Get feature maps
        feature_maps = self.get_feature_maps(dummy_input)
        
        # Compress features
        compressed_data = self.compressor.compress_features(feature_maps)
        
        # Calculate compression ratio
        stats = self.compressor.calculate_compression_ratio(feature_maps, compressed_data)
        
        if stats is None:
            return {
                'bpp': None,
                'compression_ratio': None
            }
        return stats

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
            dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=train, download=True, transform=transform)
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

def run_length_encode_batch(data: torch.Tensor, chunk_size: int = 1000000):
    """Perform run-length encoding on a batch of data using chunking for better memory efficiency.
    
    Args:
        data: Input tensor to encode
        chunk_size: Size of chunks to process at once
    Returns:
        Tuple of (values, lengths) for the runs
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    total_size = data.size
    all_values = []
    all_lengths = []
    
    # Process data in chunks
    for start_idx in range(0, total_size, chunk_size):
        end_idx = min(start_idx + chunk_size, total_size)
        chunk = data[start_idx:end_idx]
        
        # Find positions where values change in this chunk
        value_changes = np.concatenate(([True], chunk[1:] != chunk[:-1]))
        
        # Get the values and calculate run lengths for this chunk
        values = chunk[value_changes]
        lengths = np.diff(np.nonzero(value_changes)[0], append=len(chunk))
        
        all_values.extend(values.tolist())
        all_lengths.extend(lengths.tolist())
    
    return all_values, all_lengths

def evaluate_model(model, test_loader, device, save_feature_maps=False, analyze_compression=False, final_eval=False):
    """Evaluate model on test/validation data.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test/validation data
        device: Device to run evaluation on
        save_feature_maps: Whether to save feature maps for visualization
        analyze_compression: Whether to perform compression analysis
        final_eval: Whether this is the final evaluation (for printing detailed stats)
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    feature_maps = None
    compression_stats = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions first for faster accuracy calculation
            outputs = model.resnet(images)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            # Only save feature maps from first batch
            if save_feature_maps and batch_idx == 0:
                feature_maps = model.get_feature_maps(images)
            
            # Analyze compression if requested
            if analyze_compression:
                stats = model.get_compression_stats()
                if stats and 'bpp' in stats and stats['bpp'] is not None and 'compression_ratio' in stats and stats['compression_ratio'] is not None:
                    # Ensure numeric values are stored as float
                    if 'bpp' in stats and stats['bpp'] is not None:
                        stats['bpp'] = float(stats['bpp'])
                    if 'compression_ratio' in stats and stats['compression_ratio'] is not None:
                        stats['compression_ratio'] = float(stats['compression_ratio'])
                    compression_stats.append(stats)
    
    # Calculate accuracy
    accuracy = 100. * total_correct / total_samples
    
    # Print stats if requested
    if final_eval:
        print(f"\nAccuracy: {accuracy:.2f}%")
        
        # Print compression stats if available
        if compression_stats:
            valid_stats = [stat for stat in compression_stats if stat and 
                         'bpp' in stat and stat['bpp'] is not None and
                         'compression_ratio' in stat and stat['compression_ratio'] is not None]
            if valid_stats:
                avg_bpp = sum(float(stat['bpp']) for stat in valid_stats) / len(valid_stats)
                avg_ratio = sum(float(stat['compression_ratio']) for stat in valid_stats) / len(valid_stats)
                print(f"Average BPP: {avg_bpp:.2f}")
                print(f"Average Compression Ratio: {avg_ratio:.1f}x")
    
    return accuracy, feature_maps, compression_stats

def train_with_early_stopping(model, train_loader, _, test_loader, device, patience=5, eval_every=100, max_epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    patience_counter = 0
    step = 0
    
    print(f"\nTraining with early stopping (patience={patience}, eval_every={eval_every} steps)")
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with training stats
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.3f}',
                'Acc': f'{acc:.2f}%',
                'Step': step
            })
            
            step += 1
            
            # Evaluate periodically
            if step % eval_every == 0:
                test_acc, _, _ = evaluate_model(model, test_loader, device)
                model.train()
                
                print(f"\nStep {step}: Test Accuracy = {test_acc:.2f}%")
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    patience_counter = 0
                    print(f"New best accuracy! Saving model...")
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!")
                    return best_acc
        
        # Print epoch summary
        avg_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training Loss: {avg_loss:.3f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
    
    return best_acc

def train_baseline(model, train_loader, val_loader, test_loader, device, num_epochs=10):
    """Train the baseline model without compression."""
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining baseline model for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.3f}',
                'Acc': f'{acc:.2f}%'
            })
        
        # Print epoch summary
        avg_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training Loss: {avg_loss:.3f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        
        # Evaluate on test set
        test_acc, _, _ = evaluate_model(model, test_loader, device)
        model.train()
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_acc

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
        full_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
        test_dataset = ciFAIR100(root=dataset_path, train=False, download=True, transform=transform)
        
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
        
        baseline_acc, feature_maps, baseline_compression_stats = evaluate_model(
            baseline_model, test_loader, device, save_feature_maps=True, analyze_compression=True, final_eval=True)
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        if feature_maps is not None:
            # Plot feature distribution
            # plot_feature_distribution(feature_maps, [0.5], 'test_feature_distribution.png')
            pass
        else:
            print("Warning: No feature maps collected for baseline model")
        
        # Save results to JSON
        if baseline_compression_stats:
            # Get the average stats across all batches
            avg_stats = {
                'bpp': sum(stat['bpp'] for stat in baseline_compression_stats) / len(baseline_compression_stats),
                'compression_ratio': sum(stat['compression_ratio'] for stat in baseline_compression_stats) / len(baseline_compression_stats),
                'total_compressed_bits': sum(stat['total_compressed_bits'] for stat in baseline_compression_stats) / len(baseline_compression_stats),
                'original_bits': sum(stat['original_bits'] for stat in baseline_compression_stats) / len(baseline_compression_stats)
            }
            
            results = {
                'baseline': {
                    'accuracy': baseline_acc,
                    'bpp': float(avg_stats['bpp']),
                    'compression_ratio': float(avg_stats['compression_ratio']),
                    'total_compressed_bits': float(avg_stats['total_compressed_bits']),
                    'original_bits': float(avg_stats['original_bits'])
                }
            }
            
            with open('test_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            print("Test results saved to test_results.json")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTest failed! Please fix the issues before running the full experiment.")
        return False

def run_experiment(compression_ratios, methods=['topk'], channel_wise=False, num_bits=8):
    """Run compression experiment with different compression ratios and methods."""
    print("\nRunning compression experiment...")
    print(f"Compression ratios: {compression_ratios}")
    print(f"Methods: {methods}")
    print(f"Channel-wise: {channel_wise}")
    print(f"Num bits: {num_bits}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                            shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                           shuffle=False, num_workers=2)
    
    results = []
    
    # Test baseline model first
    print("\nEvaluating baseline model...")
    baseline_model = ModifiedResNet18(compression_ratio=1.0, method='topk', num_classes=100, 
                                   dataset='cifar100', channel_wise=channel_wise).to(device)
    
    baseline_acc, feature_maps, baseline_compression_stats = evaluate_model(
        baseline_model, testloader, device, save_feature_maps=True, analyze_compression=True, final_eval=True)
    
    if baseline_compression_stats:
        # Calculate average stats
        avg_stats = {
            'bpp': sum(stat['bpp'] for stat in baseline_compression_stats) / len(baseline_compression_stats),
            'compression_ratio': sum(stat['compression_ratio'] for stat in baseline_compression_stats) / len(baseline_compression_stats),
            'total_compressed_bits': sum(stat['total_compressed_bits'] for stat in baseline_compression_stats) / len(baseline_compression_stats),
            'original_bits': sum(stat['original_bits'] for stat in baseline_compression_stats) / len(baseline_compression_stats)
        }
        
        # Add baseline to results
        results.append({
            'method': 'baseline',
            'compression_ratio': 1.0,
            'accuracy': baseline_acc,
            'compression_stats': avg_stats
        })
    else:
        results.append({
            'method': 'baseline',
            'compression_ratio': 1.0,
            'accuracy': baseline_acc,
            'compression_stats': None
        })
    
    # Test compression ratios
    for ratio in compression_ratios:
        print(f"\nTesting compression ratio: {ratio}")
        
        # Test without finetuning
        model = ModifiedResNet18(compression_ratio=ratio, method='topk', num_classes=100, 
                               dataset='cifar100', channel_wise=channel_wise).to(device)
        
        accuracy, compression_stats, _ = evaluate_model(
            model, testloader, device, save_feature_maps=True, analyze_compression=True, final_eval=True)
        
        # Test with finetuning
        model_ft = ModifiedResNet18(compression_ratio=ratio, method='topk', num_classes=100, 
                                  dataset='cifar100', channel_wise=channel_wise).to(device)
        
        accuracy_ft, compression_stats_ft = train_with_early_stopping(
            model_ft, trainloader, None, testloader, device,
            patience=5, eval_every=100, max_epochs=10
        )
        
        # Add results to list
        results.append({
            'method': 'topk',
            'compression_ratio': ratio,
            'accuracy': accuracy,
            'accuracy_ft': accuracy_ft,
            'compression_stats': compression_stats,
            'compression_stats_ft': compression_stats_ft
        })
    
    return results

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
                baseline_model, test_loader, device, save_feature_maps=True, analyze_compression=True, final_eval=True)
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        if method == 'topk' and baseline_features and len(baseline_features) > 0:  # Only plot distribution once
            # Plot feature distribution with compression ratio cutoffs
            # plot_feature_distribution(baseline_features, compression_ratios, 'feature_distribution.png')
            pass
        
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
                model, test_loader, device, save_feature_maps=True, analyze_compression=True, final_eval=True)
            
            result = {
                'ratio': ratio,
                'accuracy': accuracy,
                'accuracy_ft': accuracy,
                'bpp': compression_stats['bpp'],
                'bpp_fixed': compression_stats['bpp_fixed'],
                'compression_ratio': compression_stats['compression_ratio'],
                'row_bits': compression_stats['breakdown']['row_indices']['bits'],
                'col_bits': compression_stats['breakdown']['column_indices']['huffman_bits'],
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
        # plot_accuracy_vs_compression(results, f'accuracy_vs_compression_{method}.png')
        pass
    
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
    # plot_feature_maps_with_ratios(all_feature_maps, original_image, compression_ratios,
    #                            save_path=f'feature_maps_{method}_ratio.png')
    pass

if __name__ == "__main__":
    if test_workflow():
        print("\nTest passed! Running full experiment...")
        main()
    else:
        print("\nTest failed! Please fix the issues before running the full experiment.")

def main():
    """Run full experiment with both compression methods."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run feature compression experiments')
    parser.add_argument('--topk', action='store_true', help='Run TopK compression experiment')
    parser.add_argument('--channel_wise', action='store_true', help='Run channel-wise compression experiment')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.5, 0.25, 0.1],
                      help='List of compression ratios to test')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of fine-tuning epochs')
    args = parser.parse_args()
    
    # Initialize results dictionary
    all_results = {}
    
    # Run experiments for each selected method
    if args.topk:
        print("\n=== Running TopK Compression Experiment ===")
        results = run_experiment(method='topk', compression_ratios=args.ratios, num_epochs=args.epochs)
        all_results['topk'] = results
    
    if args.channel_wise:
        print("\n=== Running Channel-wise Compression Experiment ===")
        results = run_experiment(method='channel_wise', compression_ratios=args.ratios, num_epochs=args.epochs)
        all_results['channel_wise'] = results
    
    # If no method specified, run both
    if not (args.topk or args.channel_wise):
        print("\n=== Running TopK Compression Experiment ===")
        results = run_experiment(method='topk', compression_ratios=args.ratios, num_epochs=args.epochs)
        all_results['topk'] = results
        
        print("\n=== Running Channel-wise Compression Experiment ===")
        results = run_experiment(method='channel_wise', compression_ratios=args.ratios, num_epochs=args.epochs)
        all_results['channel_wise'] = results
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'compression_results_{timestamp}.json'
    
    # Process results for JSON serialization
    json_results = {}
    for method, results in all_results.items():
        json_results[method] = []
        for result in results:
            # Convert numpy values to Python types
            processed_result = {
                'ratio': float(result['ratio']),
                'accuracy': float(result['accuracy']),
                'bpp': float(result['bpp']) if result.get('bpp') is not None else None,
                'compression_stats': result.get('compression_stats')
            }
            
            # Add fine-tuning results if present
            if 'accuracy_ft' in result:
                processed_result['accuracy_ft'] = float(result['accuracy_ft'])
            if 'compression_stats_ft' in result:
                processed_result['compression_stats_ft'] = result['compression_stats_ft']
            
            json_results[method].append(processed_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Print summary for both methods
    print("\nSummary of Results:")
    for method, results in all_results.items():
        print(f"\n{method.upper()} Method:")
        for result in results:
            ratio = result['ratio']
            accuracy = result['accuracy']
            accuracy_ft = result.get('accuracy_ft', None)
            
            print(f"\nCompression Ratio: {ratio:.2f}")
            print(f"Accuracy: {accuracy:.2f}%")
            if accuracy_ft is not None:
                print(f"Accuracy after fine-tuning: {accuracy_ft:.2f}%")
            
            if result.get('compression_stats'):
                stats = result['compression_stats']
                print(f"Compression Statistics:")
                print(f"- BPP: {stats.get('bpp', 'N/A')}")
                print(f"- Compression Ratio: {stats.get('compression_ratio', 'N/A')}x")

def run_experiment(method='topk', compression_ratios=[0.5], num_epochs=1):
    """Run compression experiment with specified method and ratios."""
    # Initialize results storage
    results = []
    baseline_features = None
    
    # Get data loaders
    train_loader, val_loader, test_loader, num_classes = get_cifar100_data(batch_size=128)
    
    # Initialize model
    model = CompressedResNet18(num_classes=num_classes, device='cuda')
    print("Loading pre-trained weights from cifar100_resnet18.pth")
    model.load_state_dict(torch.load('cifar100_resnet18.pth'))
    model = model.to('cuda')
    
    # First, evaluate baseline model
    print("\nEvaluating baseline model...")
    accuracy, feature_maps, compression_stats = evaluate_model(
        model, test_loader, 'cuda',
        save_feature_maps=True,
        analyze_compression=True,
        final_eval=True
    )
    baseline_features = feature_maps
    
    # Store baseline results
    if compression_stats and len(compression_stats) > 0:
        baseline_result = {
            'ratio': 1.0,
            'accuracy': float(accuracy),
            'bpp': float(compression_stats[0]['bpp']) if compression_stats[0].get('bpp') is not None else None,
            'compression_stats': compression_stats[0]
        }
    else:
        baseline_result = {
            'ratio': 1.0,
            'accuracy': float(accuracy),
            'bpp': None,
            'compression_stats': None
        }
    
    if method == 'topk' and baseline_features and len(baseline_features) > 0:  # Only plot distribution once
        # Plot feature distribution with compression ratio cutoffs
        # plot_feature_distribution(baseline_features, compression_ratios, 'feature_distribution.png')
        pass
    
    # Add baseline to results
    results.append(baseline_result)
    
    # Test each compression ratio
    for ratio in compression_ratios:
        print(f"\nTesting compression ratio: {ratio}")
        
        # Create new model instance for this ratio
        compressed_model = CompressedResNet18(
            num_classes=num_classes,
            device='cuda',
            compression_ratio=ratio,
            compression_method=method
        )
        compressed_model.load_state_dict(torch.load('cifar100_resnet18.pth'))
        compressed_model = compressed_model.to('cuda')
        
        # Evaluate compressed model
        print("\nEvaluating compressed model...")
        accuracy, _, compression_stats = evaluate_model(
            compressed_model, test_loader, 'cuda',
            analyze_compression=True,
            final_eval=True
        )
        
        # Store results
        if compression_stats and len(compression_stats) > 0:
            result = {
                'ratio': float(ratio),
                'accuracy': float(accuracy),
                'bpp': float(compression_stats[0]['bpp']) if compression_stats[0].get('bpp') is not None else None,
                'compression_stats': compression_stats[0]
            }
        else:
            result = {
                'ratio': float(ratio),
                'accuracy': float(accuracy),
                'bpp': None,
                'compression_stats': None
            }
        
        # Fine-tune if requested
        if num_epochs > 0:
            print(f"\nFine-tuning for {num_epochs} epochs...")
            compressed_model = fine_tune_model(
                compressed_model, train_loader, val_loader, 'cuda',
                num_epochs=num_epochs
            )
            
            # Evaluate after fine-tuning
            print("\nEvaluating after fine-tuning...")
            accuracy_ft, _, compression_stats_ft = evaluate_model(
                compressed_model, test_loader, 'cuda',
                analyze_compression=True,
                final_eval=True
            )
            
            # Add fine-tuning results
            result['accuracy_ft'] = float(accuracy_ft)
            if compression_stats_ft and len(compression_stats_ft) > 0:
                result['compression_stats_ft'] = compression_stats_ft[0]
            else:
                result['compression_stats_ft'] = None
        
        results.append(result)
        
        # Save model if it's better than baseline
        if accuracy > baseline_result['accuracy']:
            torch.save(compressed_model.state_dict(), f'compressed_model_{method}_{ratio}.pth')
    
    return results
