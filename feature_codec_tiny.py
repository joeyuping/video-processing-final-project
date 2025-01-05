import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torchvision
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from torch.distributions import Uniform

class TinyEncoder(nn.Module):
    """Minimal encoder for small feature blocks."""
    def __init__(self, block_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input: [1, block_size, block_size]
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # Output: [4, block_size//2, block_size//2]
        )

    def forward(self, x):
        return self.encoder(x)

class TinyDecoder(nn.Module):
    """Minimal decoder for small feature blocks."""
    def __init__(self, block_size: int):
        super().__init__()
        self.decoder = nn.Sequential(
            # Input: [4, block_size//2, block_size//2]
            nn.ConvTranspose2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            # Output: [1, block_size, block_size]
        )

    def forward(self, x):
        return self.decoder(x)

class TinyHyperprior(nn.Module):
    """Minimal hyperprior network."""
    def __init__(self):
        super().__init__()
        self.analysis = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(2, 2, kernel_size=1, stride=1),
        )
        
        self.synthesis = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(2, 8, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
        )

    def forward(self, y):
        z = self.analysis(y)
        z_hat = self.quantize(z)
        sigma = self.synthesis(z_hat)
        return z, z_hat, sigma

    def quantize(self, x):
        if self.training:
            noise = Uniform(-0.5, 0.5).sample(x.shape).to(x.device)
            return x + noise
        return torch.round(x)

class TinyFeatureCodec(nn.Module):
    """Complete tiny feature codec."""
    def __init__(self, block_size: int):
        super().__init__()
        self.encoder = TinyEncoder(block_size)
        self.decoder = TinyDecoder(block_size)
        self.hyperprior = TinyHyperprior()
        
        # Entropy model parameters
        self.max_value = 10
        self.num_bins = 64
        
    def forward(self, x):
        # Encode
        y = self.encoder(x)
        
        # Get hyperprior
        z, z_hat, sigma = self.hyperprior(y)
        
        # Quantize latents
        y_hat = self.quantize(y, sigma)
        
        # Decode
        x_hat = self.decoder(y_hat)
        
        return {
            'x_hat': x_hat,
            'y': y,
            'y_hat': y_hat,
            'z': z,
            'z_hat': z_hat,
            'sigma': sigma
        }
    
    def quantize(self, y, sigma):
        if self.training:
            noise = Uniform(-0.5, 0.5).sample(y.shape).to(y.device)
            return y + noise
        return torch.round(y)
    
    def compress(self, x):
        with torch.no_grad():
            # Forward pass
            output = self(x)
            
            # Get quantized tensors
            y_hat = output['y_hat']
            z_hat = output['z_hat']
            
            # Simple entropy coding simulation (actual entropy coding would use arithmetic coding)
            y_bits = self.estimate_bits(y_hat)
            z_bits = self.estimate_bits(z_hat)
            
            return output['x_hat'], y_bits + z_bits
    
    def estimate_bits(self, tensor):
        """Estimate bits needed to encode tensor."""
        # Simple entropy estimation
        prob = torch.histc(tensor, bins=self.num_bins, min=-self.max_value, max=self.max_value)
        prob = prob / prob.sum()
        prob = prob[prob > 0]  # Remove zeros
        entropy = -(prob * torch.log2(prob)).sum()
        return entropy * tensor.numel()

class FeatureBlockDataset(Dataset):
    """Dataset for feature blocks extracted from ResNet18."""
    def __init__(self, dataset, block_size: int, device: str = 'cuda'):
        super().__init__()
        self.dataset = dataset
        self.block_size = block_size
        self.device = device
        
        # Load pretrained ResNet18
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 100)
        
        # Load CIFAR100 weights
        checkpoint = torch.load('cifar100_resnet18.pth', map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.resnet.load_state_dict(state_dict, strict=False)
        
        self.resnet = self.resnet.to(device).eval()
        
    def extract_blocks(self, features: torch.Tensor, block_size: int) -> torch.Tensor:
        """Extract non-overlapping blocks from feature maps."""
        B, C, H, W = features.size()
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size
        
        blocks = features.unfold(2, block_size, block_size) \
                        .unfold(3, block_size, block_size) \
                        .reshape(B, C, num_blocks_h, num_blocks_w, block_size, block_size)
        
        blocks = blocks.permute(0, 2, 3, 1, 4, 5) \
                      .reshape(-1, 1, block_size, block_size)
        
        return blocks
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get features from first layer
            x = self.resnet.conv1(image)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            features = self.resnet.layer1(x)
            
            # Extract blocks
            blocks = self.extract_blocks(features, self.block_size)
            
        return blocks

def train_codec(
    codec: TinyFeatureCodec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lambda_rate: float = 0.01,
    num_epochs: int = 10,
    eval_every: int = 1000,
    patience: int = 5
) -> Dict:
    """Train the feature codec with early stopping."""
    optimizer = optim.Adam(codec.parameters(), lr=1e-6)
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'bpp': []}
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    # Create evaluation progress bar
    train_pbar = tqdm(total=eval_every, desc='Steps until evaluation')
    
    for epoch in range(num_epochs):
        codec.train()
        train_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            x = batch.to(device)
            
            # Forward pass
            output = codec(x)
            x_hat = output['x_hat']
            
            # Calculate metrics
            distortion = mse_loss(x_hat, x)
            rate = (codec.estimate_bits(output['y_hat']).mean() + \
                   codec.estimate_bits(output['z_hat']).mean()) / x.numel()
            
            loss = distortion + lambda_rate * rate
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            global_step += 1
            train_pbar.update(1)
            
            # Evaluate periodically
            if global_step % eval_every == 0:
                current_train_loss = train_loss / batch_count
                
                # Reset and show evaluation progress bar
                val_loss, avg_bpp = evaluate_model(
                    codec, val_loader, mse_loss, device, lambda_rate, max_batches=50)
                
                print(f'\nStep {global_step}: '
                      f'Train Loss = {current_train_loss:.6f}, '
                      f'Val Loss = {val_loss:.6f}, '
                      f'BPP = {avg_bpp:.4f}')
                
                history['train_loss'].append(current_train_loss)
                history['val_loss'].append(val_loss)
                history['bpp'].append(avg_bpp)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(codec.state_dict(), 'best_codec.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping triggered after {global_step} steps')
                        codec.load_state_dict(torch.load('best_codec.pth'))
                        return history
                
                train_pbar.reset()
                train_loss = 0
                batch_count = 0
                codec.train()
    
    train_pbar.close()
    return history

def evaluate_model(codec, val_loader, criterion, device, lambda_rate, max_batches=50):
    """Evaluate the model on a subset of validation data."""
    codec.eval()
    val_loss = 0
    total_bpp = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_batches:  # Limit evaluation batches
                break
                
            x = batch.to(device)
            output = codec(x)  # Use forward pass instead of compress for consistent computation
            x_hat = output['x_hat']
            
            # Calculate metrics the same way as training
            distortion = criterion(x_hat, x)
            rate = (codec.estimate_bits(output['y_hat']).mean() + \
                   codec.estimate_bits(output['z_hat']).mean()) / x.numel()
            
            loss = distortion + lambda_rate * rate
            val_loss += loss.item()
            total_bpp += rate.item()
            num_batches += 1
    
    return val_loss / num_batches, total_bpp / num_batches

def plot_training_history(history: Dict, save_path: str = 'training_history.png'):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot BPP
    ax2.plot(history['bpp'], label='BPP')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Bits per Pixel')
    ax2.set_title('Compression Rate')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parameters
    block_size = 4  # Can be 4, 8, or 16
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CIFAR100 dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./cifar100', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR100(
        root='./cifar100', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # Create feature block datasets - now using the original datasets
    train_blocks = FeatureBlockDataset(train_dataset, block_size, device)
    val_blocks = FeatureBlockDataset(val_dataset, block_size, device)
    
    # Use a custom collate function to handle variable number of blocks
    def collate_blocks(batch):
        return torch.cat(batch, dim=0)
    
    block_train_loader = DataLoader(
        train_blocks, 
        batch_size=1,  # Process one image at a time
        shuffle=True,
        collate_fn=collate_blocks
    )
    block_val_loader = DataLoader(
        val_blocks, 
        batch_size=1,  # Process one image at a time
        shuffle=False,
        collate_fn=collate_blocks
    )
    
    # Create and train codec
    codec = TinyFeatureCodec(block_size).to(device)
    history = train_codec(
        codec=codec,
        train_loader=block_train_loader,
        val_loader=block_val_loader,
        device=device,
        lambda_rate=0.01,
        num_epochs=10,
        eval_every=1000,
        patience=5
    )
    
    # Plot results
    plot_training_history(history)
    
    # Save model
    torch.save(codec.state_dict(), f'tiny_codec_b{block_size}.pth')

if __name__ == '__main__':
    main() 