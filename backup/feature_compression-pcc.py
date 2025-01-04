import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100, ImageFolder
import numpy as np
from collections import namedtuple
import math
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from datetime import datetime

class OctreeNode:
    def __init__(self, center, size):
        self.center = center  # (channel, height, width)
        self.size = size
        self.children = [None] * 8  # octants
        self.points = []  # (channel, height, width, value)
        self.is_leaf = True
        
    def get_octant(self, point):
        """Determine which octant a point belongs to."""
        channel, height, width, _ = point
        c, h, w = self.center
        return (int(channel >= c) << 2) | (int(height >= h) << 1) | int(width >= w)

class FeatureCompressor:
    def __init__(self, compression_ratio=0.1, num_bits=8, method='octree', image_size=(32, 32), min_node_size=2):
        self.compression_ratio = compression_ratio
        self.num_bits = num_bits
        self.method = method
        self.image_height, self.image_width = image_size
        self.min_node_size = min_node_size
        
    def _build_octree(self, points, center, size, depth=0, max_points_per_node=8):
        """Recursively build octree from points."""
        node = OctreeNode(center, size)
        
        if len(points) <= max_points_per_node or size <= self.min_node_size or depth > 10:
            node.points = points
            return node
        
        # Sort points into octants
        octants = [[] for _ in range(8)]
        for point in points:
            octant = node.get_octant(point)
            octants[octant].append(point)
        
        # Create child nodes
        half_size = size / 2
        node.is_leaf = False
        for i in range(8):
            if not octants[i]:
                continue
                
            # Calculate new center for this octant
            new_center = [
                center[0] + half_size * ((i & 4) > 0),
                center[1] + half_size * ((i & 2) > 0),
                center[2] + half_size * ((i & 1) > 0)
            ]
            
            node.children[i] = self._build_octree(
                octants[i], new_center, half_size, depth + 1, max_points_per_node
            )
        
        return node
    
    def _serialize_octree(self, node):
        """Serialize octree to compact representation."""
        if node is None:
            return {'type': 'null'}
            
        if node.is_leaf:
            return {
                'type': 'leaf',
                'points': node.points,
                'center': node.center,
                'size': node.size
            }
        
        return {
            'type': 'internal',
            'center': node.center,
            'size': node.size,
            'children': [self._serialize_octree(child) for child in node.children]
        }
    
    def _deserialize_octree(self, data):
        """Deserialize octree from compact representation."""
        if data['type'] == 'null':
            return None
            
        node = OctreeNode(data['center'], data['size'])
        
        if data['type'] == 'leaf':
            node.points = data['points']
            return node
        
        node.is_leaf = False
        node.children = [self._deserialize_octree(child) for child in data['children']]
        return node
    
    def _reconstruct_features(self, node, shape, device):
        """Reconstruct feature tensor from octree."""
        features = torch.zeros(shape, device=device)
        
        def collect_points(node):
            if node is None:
                return []
            if node.is_leaf:
                return node.points
            points = []
            for child in node.children:
                points.extend(collect_points(child))
            return points
        
        # Collect all points first
        points = collect_points(node)
        if not points:
            return features
            
        # Convert to tensor for efficient assignment
        points = torch.tensor(points, device=device)
        features[points[:, 0].long(), points[:, 1].long(), points[:, 2].long()] = points[:, 3]
        
        return features
    
    def compress_features(self, x):
        """Compress features using octree-based point cloud compression."""
        batch_size, channels, height, width = x.shape
        compressed_batch = []
        
        # Process entire batch at once for threshold
        thresholds = torch.quantile(x.abs().reshape(batch_size, -1), 1 - self.compression_ratio, dim=1)
        
        for b in range(batch_size):
            # Get mask for significant points
            mask = x[b].abs() >= thresholds[b]
            
            if mask.sum() == 0:
                # Handle empty case
                compressed_batch.append({
                    'type': 'empty',
                    'shape': (channels, height, width)
                })
                continue
            
            # Get coordinates and values efficiently
            coords = torch.nonzero(mask).float()  # [N, 3]
            values = x[b][mask]  # [N]
            
            # Move to CPU and combine in one operation
            points = torch.cat([coords, values.unsqueeze(1)], dim=1).cpu()
            
            # Quick spatial sort for better octree efficiency
            _, sort_idx = points[:, :3].sum(dim=1).sort()
            points = points[sort_idx].tolist()
            
            # Build octree with sorted points
            center = [channels/2, height/2, width/2]
            size = max(channels, height, width)
            root = self._build_octree(points, center, size)
            
            # Store compressed format
            compressed_batch.append(self._serialize_octree(root))
        
        CompressedFeatures = namedtuple('CompressedFeatures', ['batch', 'shape', 'device'])
        return CompressedFeatures(batch=compressed_batch, shape=x.shape, device=x.device)
    
    def decompress_features(self, compressed_features):
        """Decompress features from octree representation."""
        batch_size, channels, height, width = compressed_features.shape
        decompressed = torch.zeros((batch_size, channels, height, width), device=compressed_features.device)
        
        for b, compressed in enumerate(compressed_features.batch):
            if compressed['type'] == 'empty':
                continue
                
            # Deserialize and reconstruct
            root = self._deserialize_octree(compressed)
            decompressed[b] = self._reconstruct_features(root, (channels, height, width), compressed_features.device)
        
        return decompressed
    
    def calculate_compression_ratio(self, original_features, compressed_features=None):
        """Calculate compression ratio in terms of bits per pixel."""
        if compressed_features is None:
            compressed_features = self.compress_features(original_features)
        
        batch_size, channels, height, width = original_features.shape
        num_pixels = self.image_height * self.image_width
        
        # Calculate original bits (8-bit per channel per pixel)
        original_bits = 8 * 3 * num_pixels
        
        # Estimate compressed size
        total_points = sum(
            sum(len(node.points) for node in self._collect_leaves(self._deserialize_octree(batch)))
            for batch in compressed_features.batch
        )
        
        # Calculate bits for octree structure and point values
        structure_bits = total_points * (
            math.ceil(math.log2(channels)) +  # channel index
            math.ceil(math.log2(height)) +    # height index
            math.ceil(math.log2(width)) +     # width index
            self.num_bits                     # value
        )
        
        # Add overhead for octree structure (estimated)
        overhead_bits = total_points * 2  # 2 bits per point for tree structure
        
        total_bits = structure_bits + overhead_bits
        bpp = total_bits / num_pixels
        
        compression_stats = {
            'bpp': bpp,
            'total_points': total_points,
            'structure_bits': structure_bits,
            'overhead_bits': overhead_bits,
            'original_bits': original_bits,
            'total_compressed_bits': total_bits,
            'num_pixels': num_pixels,
            'compression_ratio': total_bits / original_bits
        }
        
        return compression_stats
    
    def _collect_leaves(self, node):
        """Helper function to collect all leaf nodes."""
        if node is None:
            return []
        if node.is_leaf:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves


class ModifiedResNet18(nn.Module):
    def __init__(self, compression_ratio=0.1, method='octree', finetune=False, num_classes=100, dataset='cifar100'):
        super(ModifiedResNet18, self).__init__()
        # Create base ResNet18 model
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        
        # Set image size based on dataset
        if dataset.lower() == 'cifar100':
            image_size = (32, 32)
            # Modify for CIFAR (32x32 images)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()  # Remove maxpool as CIFAR images are small
            
            # Load pretrained weights if not finetuning
            if not finetune:
                weights_path = hf_hub_download(
                    repo_id="edadaltocg/resnet18_cifar100",
                    filename="pytorch_model.bin"
                )
                state_dict = torch.load(weights_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print("Loaded pretrained CIFAR-100 weights")
        else:
            # Stanford Dogs uses 224x224 images
            image_size = (224, 224)
            print("Initializing model with Kaiming initialization")
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
        
        # Split the model into parts for feature compression
        self.initial_layers = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )
        self.layer1 = self.model.layer1
        self.remaining_layers = nn.Sequential(
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool,
        )
        self.fc = self.model.fc
        
        # Initialize compressor
        self.compressor = FeatureCompressor(
            compression_ratio=compression_ratio,
            method=method,
            image_size=image_size
        )
    
    def forward(self, x):
        x = self.initial_layers(x)
        x = self.layer1(x)
        
        # Compress and decompress features
        compressed = self.compressor.compress_features(x)
        x = self.compressor.decompress_features(compressed)
        
        x = self.remaining_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_feature_maps(self, x):
        x = self.initial_layers(x)
        x = self.layer1(x)
        return x

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
        dataset = CIFAR100(root=dataset_path, train=train, download=True, transform=transform)
    elif dataset_name.lower() == 'stanford_dogs':
        os.makedirs(dataset_path, exist_ok=True)
        dataset = StanfordDogsDataset(root=dataset_path, train=train, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset

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
    bit_ratios = np.array([result['bit_compression'] for result in results])  # Convert to bits per pixel
    accuracies = np.array([result['accuracy'] for result in results])
    ratios = np.array([result['ratio'] for result in results])
    order = np.argsort(-ratios)
    bit_ratios = bit_ratios[order]
    accuracies = accuracies[order]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(bit_ratios, accuracies, 'bo-', linewidth=2, markersize=8, label='No Finetuning')
    
    accuracies_ft = []
    for result in results:
        if 'accuracy_ft' in result:
            accuracies_ft.append(result['accuracy_ft'])
    if len(accuracies_ft) > 0:
        accuracies_ft = np.array(accuracies_ft)
        accuracies_ft = accuracies_ft[order]
        plt.plot(bit_ratios, accuracies_ft, 'ro-', linewidth=2, markersize=8, label='Finetuning')
    
    # Add labels and title
    plt.xlabel('Bit Compression Ratio', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Bit Compression Ratio for Feature Map Compression', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for each point
    for i, (bits, acc) in enumerate(zip(bit_ratios, accuracies)):
        ratio = results[order[i]]['ratio']
        plt.annotate(f'r={ratio:.2f}\n{acc:.1f}%', 
                    (bits, acc),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)
        
        acc_ft = accuracies_ft[i]
        plt.annotate(f'r={ratio:.2f}\n{acc_ft:.1f}%', 
                    (bits, acc_ft),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
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

def evaluate_model(model, test_loader, device, save_feature_maps=False):
    """Evaluate model accuracy and compression stats."""
    model.eval()
    correct = 0
    total = 0
    feature_maps = None
    compression_stats = None
    
    # Get feature maps and compression stats only once at the start
    if save_feature_maps:
        with torch.no_grad():
            inputs = next(iter(test_loader))[0].to(device)
            feature_maps = model.get_feature_maps(inputs)
            compression_stats = model.compressor.calculate_compression_ratio(feature_maps)
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy, compression_stats, feature_maps

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, patience=5, eval_every=50, max_epochs=10):
    best_val_acc = 0
    patience_counter = 0
    global_step = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % eval_every == 0:
                # Evaluate on validation set
                val_acc, _ = evaluate_model(model, val_loader, device)
                print(f'Step {global_step}, Validation Accuracy: {val_acc:.2f}%')
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f'Early stopping at step {global_step}')
                    return best_val_acc
    
    return best_val_acc

def train_baseline(model, train_loader, val_loader, test_loader, device, num_epochs=10):
    """Train the baseline model without compression."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_acc = 0
    best_model_state = None
    
    print("Training baseline model...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, predicted = val_outputs.max(1)
                total += val_labels.size(0)
                correct += predicted.eq(val_labels).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best model
    if best_model_state is not None:
        for k, v in best_model_state.items():
            best_model_state[k] = v.to(device)
        model.load_state_dict(best_model_state)
    
    # Test final model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing baseline'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    print(f'Final Test Acc: {test_acc:.2f}%')
    return test_acc

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Feature compression experiment')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'stanford_dogs'],
                      help='Dataset to use (default: cifar100)')
    parser.add_argument('--channel_wise', action='store_true',
                      help='Whether to do topK per channel (True) or across all channels (False)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset specific transforms
    if args.dataset.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                              std=[0.2675, 0.2565, 0.2761])
        ])
        num_classes = 100
    else:  # stanford_dogs
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        num_classes = 120
    
    # Load dataset
    train_dataset = get_dataset(args.dataset, transform, train=True)
    test_dataset = get_dataset(args.dataset, transform, train=False)
    
    # Split training data into train and validation sets (90-10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Test different compression ratios
    compression_ratios = [0.01,0.05,0.1,0.3,0.5]
    methods = ['octree']
    all_results = {}
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} method ===")
        results = []
        
        # First evaluate baseline and get feature distribution
        print("\nBaseline Evaluation (No Compression):")
        baseline_model = ModifiedResNet18(compression_ratio=1.0, method=method, num_classes=num_classes, 
                                        dataset=args.dataset).to(device)
        
        # Train baseline for Stanford Dogs
        if args.dataset.lower() == 'stanford_dogs':
            print("Training baseline model for Stanford Dogs dataset...")
            baseline_acc = train_baseline(baseline_model, train_loader, val_loader, test_loader, device)
        else:
            # For CIFAR-100, use pretrained model directly
            baseline_acc, baseline_compression_stats, baseline_features = evaluate_model(
                baseline_model, test_loader, device, save_feature_maps=True)
        
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        if method == 'octree':  # Only plot distribution once
            # Plot feature distribution with compression ratio cutoffs
            plot_feature_distribution(baseline_features[0], compression_ratios, 'feature_distribution.png')
        
        # Add baseline to results
        results.append({
            'ratio': 1.0,
            'accuracy': float(baseline_acc),
            'bpp': float(baseline_compression_stats['bpp']),
            'row_index_bits': float(baseline_compression_stats['row_index_bits']),
            'col_index_bits': float(baseline_compression_stats['col_index_bits']),
            'value_bits': float(baseline_compression_stats['value_bits']),
            'original_bits': float(baseline_compression_stats['original_bits']),
            'total_compressed_bits': float(baseline_compression_stats['total_compressed_bits']),
            'num_pixels': float(baseline_compression_stats['num_pixels']),
            'value_ratio': float(baseline_compression_stats['value_ratio'])
        })
        
        for ratio in compression_ratios:
            print(f"\nTesting compression ratio: {ratio}")
            # Test without finetuning
            model = ModifiedResNet18(compression_ratio=ratio, method=method, num_classes=num_classes, 
                                   dataset=args.dataset).to(device)
            accuracy, compression_stats, _ = evaluate_model(
                model, test_loader, device, save_feature_maps=True)
            
            result = {
                'ratio': ratio,
                'accuracy': accuracy,
                'bpp': compression_stats['bpp'],
                'row_index_bits': compression_stats['row_index_bits'],
                'col_index_bits': compression_stats['col_index_bits'],
                'value_bits': compression_stats['value_bits'],
                'metadata_bits': compression_stats['metadata_bits'],
                'original_bits': compression_stats['original_bits'],
                'total_compressed_bits': compression_stats['total_compressed_bits'],
                'num_pixels': compression_stats['num_pixels'],
                'value_ratio': compression_stats['value_ratio']
            }
            
            # Test with finetuning
            model_ft = ModifiedResNet18(compression_ratio=ratio, method=method, finetune=True, num_classes=num_classes, 
                                      dataset=args.dataset).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.0001)
            
            # Train with early stopping
            accuracy_ft = train_with_early_stopping(
                model_ft, train_loader, val_loader, criterion, optimizer, device,
                patience=5, eval_every=50, max_epochs=10
            )
            result['accuracy_ft'] = accuracy_ft
            
            results.append(result)
            print(f"No Finetuning - Accuracy: {accuracy:.2f}%, Bits per pixel: {compression_stats['bpp']:.4f}")
            print(f"With Finetuning - Accuracy: {accuracy_ft:.2f}%")
        
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
        feature_map = baseline_model.get_feature_maps(inputs[0:1])
        all_feature_maps.append(feature_map)
    
    # Get feature maps for each compression ratio
    for ratio in compression_ratios:
        model = ModifiedResNet18(compression_ratio=ratio, method=method, num_classes=num_classes, 
                               dataset=args.dataset).to(device)
        with torch.no_grad():
            feature_map = model.get_feature_maps(inputs[0:1])
            all_feature_maps.append(feature_map)
    
    # Plot feature maps
    plot_feature_maps_with_ratios(all_feature_maps, original_image, compression_ratios,
                               save_path=f'feature_maps_{method}_ratio.png')

if __name__ == "__main__":
    main()
    # compression_results = json.load(open('compression_results.json', 'r'))
    # plot_accuracy_vs_compression(compression_results.get('compression_results').get('topk'), 'accuracy_vs_compression_ratio_topk.png')
