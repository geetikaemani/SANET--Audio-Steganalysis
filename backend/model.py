import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiActivatedAttention(nn.Module):
    """
    Multi-activated channel attention module using Sigmoid, LeakyReLU, and Tanh.
    """
    def __init__(self, channels):
        super(MultiActivatedAttention, self).__init__()
        
        # Reduce dimension (matching paper's description)
        reduction_ratio = 8
        hidden_dim = max(channels // reduction_ratio, 8)
        
        # Shared MLP for channel attention
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels)
        )
        
        # Three activation functions
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: (batch, channels, time_steps)
        batch, channels, time_steps = x.size()
        
        # Global pooling
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, channels)
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, channels)
        
        # Shared MLP
        pooled = avg_pool + max_pool
        attention_features = self.shared_mlp(pooled)  # (batch, channels)
        
        # Apply three different activation functions
        att_sigmoid = self.sigmoid(attention_features)
        att_leaky_relu = self.leaky_relu(attention_features)
        att_tanh = self.tanh(attention_features)
        
        # Apply attention weights to original features
        out_sigmoid = x * att_sigmoid.unsqueeze(-1)
        out_leaky = x * att_leaky_relu.unsqueeze(-1)
        out_tanh = x * att_tanh.unsqueeze(-1)
        
        # Concatenate along channel dimension
        out = torch.cat([out_sigmoid, out_leaky, out_tanh], dim=1)
        return out  # (batch, 3*channels, time_steps)


class MultiSpaceCorrelationAggregation(nn.Module):
    """
    Multi-space Correlation Aggregation (MCA) module.
    Uses multi-activated attention and 1D convolution for feature fusion.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiSpaceCorrelationAggregation, self).__init__()
        
        self.multi_attention = MultiActivatedAttention(in_channels)
        
        # After multi-activation, channels become 3*in_channels
        self.conv1 = nn.Conv1d(
            in_channels=in_channels * 3,
            out_channels=out_channels * 2,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels * 2)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # x shape: (batch, time_steps, channels)
        # Permute to (batch, channels, time_steps)
        x = x.permute(0, 2, 1)
        
        # Apply multi-activated attention
        x = self.multi_attention(x)  # (batch, 3*channels, time_steps)
        
        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x  # (batch, out_channels, time_steps)


class IntraFrameCorrelationEnhancement(nn.Module):
    """
    Intra-frame correlation enhancement using 1D convolution and multi-head attention.
    """
    def __init__(self, feature_dim, num_heads=4):
        super(IntraFrameCorrelationEnhancement, self).__init__()
        
        # Ensure num_heads divides feature_dim
        while feature_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.num_heads = num_heads
        
        self.conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm1d(feature_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, time_steps)
        batch_size, channels, time_steps = x.size()
        
        # Convolution
        x_conv = F.relu(self.bn(self.conv(x)))
        
        # Residual connection
        x = x + x_conv
        
        # Permute for attention: (batch, time_steps, channels)
        x = x.permute(0, 2, 1)
        
        # Multi-head attention
        attn_out, _ = self.multihead_attn(x, x, x)
        
        # Residual and norm
        x = self.norm(x + attn_out)
        
        # Permute back: (batch, channels, time_steps)
        x = x.permute(0, 2, 1)
        
        return x


class InterFrameCorrelationEnhancement(nn.Module):
    """
    Inter-frame correlation enhancement using 1D convolutions.
    """
    def __init__(self, channels):
        super(InterFrameCorrelationEnhancement, self).__init__()
        
        # Expansion factor
        expansion = 2
        hidden_dim = channels * expansion
        
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x shape: (batch, channels, time_steps)
        identity = x
        
        # First convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second convolution block
        x = self.bn2(self.conv2(x))
        
        # Residual connection
        x = x + identity
        
        # Layer normalization (requires [batch, time, channels])
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        
        return x


class FineGrainedLocalCorrelationEnhancement(nn.Module):
    """
    Fine-grained Local Correlation Enhancement (FLCE) module.
    Combines intra-frame and inter-frame correlation enhancement.
    """
    def __init__(self, feature_dim, num_heads=4):
        super(FineGrainedLocalCorrelationEnhancement, self).__init__()
        
        self.intra_frame = IntraFrameCorrelationEnhancement(feature_dim, num_heads)
        self.inter_frame = InterFrameCorrelationEnhancement(feature_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, time_steps)
        x = self.intra_frame(x)
        x = self.inter_frame(x)
        return x


class SANet(nn.Module):
    """
    Speech encoder and steganography Algorithm independent steganalysis Network (SANet).
    Complete architecture for general steganalysis of compressed speech.
    
    Compatible with preprocessing that extracts MFCC (13 dims) and LFB (40 dims).
    """
    def __init__(self, 
                 mfcc_dim=13, 
                 lfb_dim=40,
                 lstm_hidden_dim=128,
                 lstm_layers=2,
                 mca_out_channels=128,
                 flce_num_heads=4,
                 num_classes=2,
                 dropout=0.5):
        """
        Args:
            mfcc_dim: MFCC feature dimension (default: 13)
            lfb_dim: LFB feature dimension (default: 40)
            lstm_hidden_dim: Hidden dimension for LSTM (default: 128)
            lstm_layers: Number of LSTM layers (default: 2)
            mca_out_channels: Output channels for MCA module (default: 128)
            flce_num_heads: Number of attention heads in FLCE (default: 4)
            num_classes: Number of output classes (default: 2 for binary)
            dropout: Dropout rate (default: 0.5)
        """
        super(SANet, self).__init__()
        
        self.mfcc_dim = mfcc_dim
        self.lfb_dim = lfb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        
        # Subspace Correlation Extraction (Bi-LSTM for each subspace)
        self.lstm_mfcc = nn.LSTM(
            input_size=mfcc_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.lstm_lfb = nn.LSTM(
            input_size=lfb_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Total channels after concatenating both LSTM outputs
        # Each LSTM outputs: hidden_dim * 2 (bidirectional)
        # Two LSTMs total: hidden_dim * 2 * 2
        total_lstm_output = lstm_hidden_dim * 2 * 2
        
        # Multi-space Correlation Aggregation
        self.mca = MultiSpaceCorrelationAggregation(
            in_channels=total_lstm_output,
            out_channels=mca_out_channels
        )
        
        # Fine-grained Local Correlation Enhancement
        self.flce = FineGrainedLocalCorrelationEnhancement(
            feature_dim=mca_out_channels,
            num_heads=flce_num_heads
        )
        
        # Classification module
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(mca_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.)
    
    def forward(self, mfcc, lfb):
        """
        Args:
            mfcc: MFCC features (batch, time_steps, mfcc_dim)
            lfb: LFB features (batch, time_steps, lfb_dim)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Validate inputs
        batch_size = mfcc.size(0)
        
        if mfcc.size(-1) != self.mfcc_dim:
            raise ValueError(f"Expected MFCC dimension {self.mfcc_dim}, got {mfcc.size(-1)}")
        
        if lfb.size(-1) != self.lfb_dim:
            raise ValueError(f"Expected LFB dimension {self.lfb_dim}, got {lfb.size(-1)}")
        
        # Step 1: Subspace Correlation Extraction with Bi-LSTM
        mfcc_out, _ = self.lstm_mfcc(mfcc)  # (batch, time_steps, hidden*2)
        lfb_out, _ = self.lstm_lfb(lfb)      # (batch, time_steps, hidden*2)
        
        # Concatenate features from both subspaces
        x = torch.cat([mfcc_out, lfb_out], dim=-1)  # (batch, time_steps, hidden*4)
        
        # Step 2: Multi-space Correlation Aggregation
        x = self.mca(x)  # (batch, mca_out_channels, time_steps)
        
        # Step 3: Fine-grained Local Correlation Enhancement
        x = self.flce(x)  # (batch, mca_out_channels, time_steps)
        
        # Step 4: Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("SANet Model Test")
    print("="*60)
    
    # Model configuration (matching preprocessing defaults)
    config = {
        'mfcc_dim': 13,
        'lfb_dim': 40,
        'lstm_hidden_dim': 128,
        'lstm_layers': 2,
        'mca_out_channels': 128,
        'flce_num_heads': 4,
        'num_classes': 2,
        'dropout': 0.5
    }
    
    # Create model
    model = SANet(**config)
    
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test with sample data
    batch_size = 4
    time_steps = 999  # Typical for 10 seconds at 8kHz with 10ms hop
    
    mfcc_input = torch.randn(batch_size, time_steps, config['mfcc_dim'])
    lfb_input = torch.randn(batch_size, time_steps, config['lfb_dim'])
    
    print(f"\nTest Input Shapes:")
    print(f"  MFCC: {mfcc_input.shape}")
    print(f"  LFB: {lfb_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(mfcc_input, lfb_input)
    
    print(f"\nTest Output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output[0]}")
    
    # Test with softmax
    probs = F.softmax(output, dim=1)
    print(f"  Probabilities: {probs[0]}")
    print(f"  Predicted class: {torch.argmax(probs[0]).item()}")
    
    print("\n" + "="*60)
    print("Model test completed successfully!")
    print("="*60)