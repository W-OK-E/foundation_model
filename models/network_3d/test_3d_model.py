#!/usr/bin/env python
"""Test script to verify the 3D ElitNet model works correctly."""

import sys
import torch

# Add path for imports
sys.path.append('/mnt/data/omkumar/foundation_phase1/models/network_3d')

from ElitNet_3d import ElitNet3d

def test_3d_model():
    """Test the 3D ElitNet model with 3D input."""
    print("="*70)
    print("Testing 3D ElitNet Model")
    print("="*70)
    
    # Model parameters
    in_channels = 3
    num_classes = 2
    layers = [32, 64, 128, 256, 512]  # Channel dimensions
    kernel_sz = 3
    up_mode = 'up_conv'
    pool = 'pool'
    
    # Create model
    print("\n1. Creating 3D ElitNet model...")
    try:
        model = ElitNet3d(
            in_channels=in_channels,
            num_classes=num_classes,
            layers=layers,
            kernel_sz=kernel_sz,
            up_mode=up_mode,
            pool=pool,
            conv_bridge=True,
            shortcut=True,
            skip_conn=True,
            residual=True,
            causal=True,
            conv_mode='Conv3d'
        )
        print("✓ Model created successfully")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Output classes: {num_classes}")
        print(f"  - Layer dimensions: {layers}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model moved to device: {device}")
    
    # Create 3D input tensor (batch_size, channels, depth, height, width)
    print("\n2. Creating 3D input tensor...")
    batch_size = 2
    depth, height, width = 128,128,128
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width).to(device)
    print(f"✓ Input tensor shape: {input_tensor.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: {in_channels}")
    print(f"  - Spatial dims (D×H×W): {depth}×{height}×{width}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print("✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        print(f"  - Expected: ({batch_size}, {num_classes}, {depth}, {height}, {width})")
        
        # Verify output shape
        expected_shape = (batch_size, num_classes, depth, height, width)
        if output.shape == expected_shape:
            print(f"✓ Output shape matches expected shape")
        else:
            print(f"✗ Output shape mismatch!")
            print(f"  - Expected: {expected_shape}")
            print(f"  - Got: {output.shape}")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with different input sizes
    print("\n4. Testing with different input sizes...")
    test_sizes = [
        # (1, 128, 128, 128),
        # (1, 128, 128, 128),
        (4,3,128,128,128),
        # (1,1,128,128,128)
    ]
    
    for size in test_sizes:
        try:
            test_input = torch.randn(*size).to(device)
            with torch.no_grad():
                test_output = model(test_input)
            print(f"✓ Input {size} → Output {tuple(test_output.shape)}")
        except Exception as e:
            print(f"✗ Failed with input size {size}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*70)
    print("✓ All tests passed! 3D ElitNet is working correctly.")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_3d_model()
    sys.exit(0 if success else 1)
