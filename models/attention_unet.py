import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    The 'Attention Gate' Mechanism.
    
    In seismic interpretation, >90% of the image is background rock (noise).
    Standard U-Nets copy this noise from Encoder to Decoder via skip connections.
    
    This block acts as a filter:
    1. Takes the coarse, high-level features from the Decoder (g).
    2. Takes the fine-grained features from the Encoder (x).
    3. Calculates a 'Compatibility Score' (Attention Map) between them.
    4. Multiplies the Encoder features by this score (0 to 1).
    
    Result: Only relevant features (fault edges) are passed through; noise is suppressed.
    Reference: Liu et al. (2021), 'Attention-Based 3D Seismic Fault Segmentation'.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Number of feature maps in the Gating signal (Decoder).
            F_l (int): Number of feature maps in the Skip connection (Encoder).
            F_int (int): Intermediate filters for the calculation (usually F_l / 2).
        """
        super(AttentionBlock, self).__init__()
        
        # Transformation for the Gating Signal (from Decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transformation for the Local Feature Signal (from Encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # The 'Psi' transformation: Compresses mixed features into a single attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Normalizes scores between 0 (ignore) and 1 (pay attention)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 1. Project both signals into a common intermediate space
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 2. Combine them and calculate compatibility (ReLU)
        psi = self.relu(g1 + x1)
        
        # 3. Generate the Attention Map (Sigmoid)
        # Result is a probability map: Where are the faults?
        attention_coefficients = self.psi(psi)
        
        # 4. Filter the original Encoder features
        # Multiply element-wise: Background noise gets multiplied by ~0
        return x * attention_coefficients


class ProteanUNet(nn.Module):
    """
    The Reality Ensemble Engine (REE) - Core Vision Model.
    
    Architecture:
    - Modified U-Net with Attention Gates at every upsampling step.
    - Designed for 1-channel Seismic Amplitude input (Grayscale).
    - Outputs a pixel-wise probability map for Fault presence.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(ProteanUNet, self).__init__()
        
        # --- ENCODER PATH (Downsampling) ---
        # Captures context ("Where is the fault structure?")
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # Reduces size, increases receptive field
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK ---
        # The deepest layer, capturing the most abstract geological features
        self.bottleneck = self.conv_block(128, 256)
        
        # --- DECODER PATH (Upsampling) ---
        # Restores resolution ("Where exactly are the edges?")
        
        # Upsampling Block 1
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Critical: The Attention Gate filters the connection from enc2 to dec2
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64) 
        self.dec2 = self.conv_block(256, 128) # 128 (from up) + 128 (from att) = 256 input
        
        # Upsampling Block 2 (Final Resolution)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = self.conv_block(128, 64)
        
        # Final 1x1 Convolution to map features to Fault Probability
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        """Standard Double Convolution Block with Batch Norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Encode
        e1 = self.enc1(x)      # High-res features
        e2 = self.enc2(self.pool1(e1)) # Medium-res features
        
        # 2. Bottleneck
        b = self.bottleneck(self.pool2(e2)) # Low-res, high-context
        
        # 3. Decode with Attention
        d2 = self.up2(b)
        # Apply Attention Gate: Filter 'e2' using the signal from 'd2'
        e2_att = self.att2(g=d2, x=e2) 
        d2 = torch.cat((e2_att, d2), dim=1) # Concatenate filtered skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        # Apply Attention Gate: Filter 'e1' using the signal from 'd1'
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.dec1(d1)
        
        # 4. Output Map (Sigmoid activation for 0-1 probability)
        return torch.sigmoid(self.final(d1))

# Example Usage Check
if __name__ == "__main__":
    # Simulate a batch of 1 seismic slice (1 Channel, 128x128)
    dummy_input = torch.randn(1, 1, 128, 128)
    model = ProteanUNet()
    output = model(dummy_input)
    print(f"Model Build Successful. Output Shape: {output.shape}")