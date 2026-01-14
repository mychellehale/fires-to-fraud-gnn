import torch
import numpy as np
import mgwr
import sys

def verify_system():
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📦 Torch:  {torch.__version__}")
    print(f"📦 NumPy:  {np.__version__}")
    
    # Test 1: Torch to NumPy Bridge (The most common crash point)
    try:
        t = torch.tensor([1.0, 2.0])
        n = t.numpy()
        print("✅ Torch-to-NumPy bridge is active.")
    except Exception as e:
        print(f"❌ Torch-to-NumPy failed: {e}")

    # Test 2: MGWR Initialization
    try:
        # Dummy data for MGWR
        coords = np.array([[0, 0], [1, 1]])
        y = np.array([1, 2]).reshape(-1, 1)
        X = np.array([1, 1]).reshape(-1, 1)
        _ = mgwr.gwr.GWR(coords, y, X, bw=1)
        print("✅ MGWR is correctly initialized.")
    except Exception as e:
        print(f"❌ MGWR check failed: {e}")

if __name__ == "__main__":
    verify_system()