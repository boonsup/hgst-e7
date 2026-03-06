import numpy as np
import os

def main():
    if not os.path.exists("ts_200.npy") or not os.path.exists("ts_302.npy"):
        print("Data files not found.")
        return
        
    ts_200 = np.load("ts_200.npy")
    ts_302 = np.load("ts_302.npy")
    
    print(f"Seed 200: R mean = {np.mean(ts_200):.6f}, std = {np.std(ts_200):.6f}")
    print(f"Seed 302: R mean = {np.mean(ts_302):.6f}, std = {np.std(ts_302):.6f}")
    
    # Find divergence
    diff = np.abs(ts_200 - ts_302)
    div_idx = np.argmax(diff > 1e-6)
    
    if diff[div_idx] > 1e-6:
        print(f"Divergence starts at sweep {div_idx}.")
        print(f"Sweep {div_idx}: seed 200 = {ts_200[div_idx]}, seed 302 = {ts_302[div_idx]}")
    else:
        print("No exact divergence found. Time series match within 1e-6.")

if __name__ == "__main__":
    main()
