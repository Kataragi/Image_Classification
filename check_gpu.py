#!/usr/bin/env python3
"""
GPU Configuration Check Script
Verify that PyTorch can properly use GPU
"""

import torch
import sys

def check_gpu():
    """Comprehensive GPU check"""
    print("=" * 80)
    print("GPU Configuration Check")
    print("=" * 80)

    # Python version
    print(f"\nPython Version: {sys.version}")

    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # CUDA availability
    print(f"\nCUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ CUDA is NOT available!")
        print("Training will run on CPU (very slow)")
        return False

    # CUDA version
    print(f"CUDA Version: {torch.version.cuda}")

    # Number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    # GPU information
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")

        # Compute capability
        capability = torch.cuda.get_device_capability(i)
        print(f"  Compute Capability: {capability[0]}.{capability[1]}")

        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory
        print(f"  Total Memory: {total_memory / 1024**3:.2f} GB")

        # Current memory usage
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(f"  Allocated Memory: {allocated / 1024**3:.2f} GB")
        print(f"  Reserved Memory: {reserved / 1024**3:.2f} GB")

    # Test GPU operations
    print("\n" + "=" * 80)
    print("Testing GPU Operations")
    print("=" * 80)

    try:
        # Create tensor on CPU
        cpu_tensor = torch.randn(1000, 1000)
        print(f"\n✓ Created tensor on CPU: {cpu_tensor.device}")

        # Move to GPU
        gpu_tensor = cpu_tensor.cuda()
        print(f"✓ Moved tensor to GPU: {gpu_tensor.device}")

        # Matrix multiplication on GPU
        result = torch.mm(gpu_tensor, gpu_tensor)
        print(f"✓ Matrix multiplication on GPU: {result.device}")

        # Check memory usage
        print(f"\n✓ GPU Memory after operations: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # Clean up
        del cpu_tensor, gpu_tensor, result
        torch.cuda.empty_cache()
        print(f"✓ Cleaned up GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        print("\n" + "=" * 80)
        print("✅ ALL CHECKS PASSED - GPU is ready for training!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ GPU operation failed: {e}")
        print("\nPossible issues:")
        print("1. PyTorch CUDA version doesn't match your GPU")
        print("2. GPU compute capability is too old")
        print("3. NVIDIA drivers need updating")
        print("\nReinstall PyTorch:")
        print("  pip uninstall torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
