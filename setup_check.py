# setup_check.py
import sys
import os
try:
    import numpy
    print(f"NumPy: {numpy.__version__}")
    import gymnasium
    print(f"Gymnasium: {gymnasium.__version__}")
    import torch
    print(f"Torch: {torch.__version__}, CPU-only: {not torch.cuda.is_available()}")
    import stable_baselines3
    print(f"Stable Baselines3: {stable_baselines3.__version__}")
    print("All libraries ready for spectrum simulator!")
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)