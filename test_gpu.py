#!/usr/bin/env python3
"""
Quick GPU/CUDA check for PyTorch
Run this after installing PyTorch to verify GPU support.
"""

try:
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            
        print("\n✅ GPU support is working correctly!")
        print("   You can now run the deep learning assignments with GPU acceleration.")
        
    else:
        print("\n⚠️  No CUDA support detected - PyTorch is running in CPU-only mode")
        print("   The assignments will work but run slower.")
        print("\nTo enable GPU support:")
        print("  1. Check if you have an NVIDIA GPU: nvidia-smi")
        print("  2. Uninstall CPU-only PyTorch: pip uninstall torch torchvision -y")
        print("  3. Install CUDA version: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("  4. See LOCAL_SETUP_STUDENT.md section 12 (Troubleshooting) for detailed instructions")
        
except ImportError:
    print("❌ PyTorch not installed")
    print("\nInstall with:")
    print("  pip install torch torchvision")
    print("  (or see requirements.txt and LOCAL_SETUP_STUDENT.md)")
