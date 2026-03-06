import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"d:\boonsup\pro26\pro07-daft\daft-forward\E1E4E7\phase1_empirical_study\src")))
from p1_execute import run_su3_fss_timeseries

def main():
    print("Testing Seed 200:")
    ts_200 = run_su3_fss_timeseries(8, 200, n_therm=2000, n_meas=2000, verbose=True)
    
    print("\nTesting Seed 302:")
    ts_302 = run_su3_fss_timeseries(8, 302, n_therm=2000, n_meas=2000, verbose=True)
    
    # Save the time series for comparison
    np.save("ts_200.npy", ts_200.R)
    np.save("ts_302.npy", ts_302.R)
    print("Saved time series to ts_200.npy and ts_302.npy.")

if __name__ == "__main__":
    main()
