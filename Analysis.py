import os
import sys
import pickle
import numpy as np
from typing import Any, Dict
from optimisedGridNetwork import MixedModularCoder
from PredictionModel import OptimisedRLS
import itertools
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def load_data(filename: str) -> Dict[str, Any]:
    """
    Load the simulation data from a pickle file under Results/.
    """
    filepath = os.path.join(f"Results\\ID {filename}", f"Data {filename}.pickle")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def save_results(filename: str, results: Dict[str, Any]) -> None:
    """
    Save analysis results to a pickle file under Results/.
    """
    out_path = os.path.join(f"Results\\ID {filename}", f"Analysis {filename}.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

def process_single_trial(trial, vel, pos_r, vn, nn, dt, start, nr_gains, range_max, T):
    """Process a single trial with all parameter combinations."""
    # Generate noise for this trial
    np.random.seed(trial)  # Ensure reproducibility while allowing different noise per trial
    v_noise = np.random.normal(0, vn, (T, vel.shape[1]))
    act_noise = np.random.normal(0, nn, T)
    
    # Apply noise to velocity data
    vel_noisy = (vel + v_noise) * dt
    
    # Results for this trial
    trial_results = np.zeros((len(nr_gains), len(range_max)))
    
    # Process all parameter combinations for this trial
    for i, nr in enumerate(nr_gains):
        for j, max_ in enumerate(range_max):
            # Generate gain values for this parameter set
            gains = np.linspace(start, max_, nr)
            
            # Initialize models for this parameter set
            mmc = MixedModularCoder(gains=gains)
            mmc.set_integrator(pos_r[0])
            rls = OptimisedRLS(mmc.ac_size, num_outputs=3)
            
            # Process all time steps
            squared_errors_sum = 0.0
            
            for t in range(T):
                # Update with noisy velocity
                act, true_pos = mmc.update(vel_noisy[t])
                
                # Apply activation noise
                act = np.clip(act + act_noise[t], 0.0, 1.0)
                
                # Predict position and calculate error
                pred_pos = rls.predict(act)
                error = true_pos - pred_pos
                
                # Update RLS model
                rls.update(act, true_pos)
                
                # Accumulate squared error
                squared_errors_sum += np.sum(error**2)
            
            # Calculate MSE for this parameter combination
            trial_results[i, j] = squared_errors_sum / (T * 3)
    
    return trial_results

def optimized_grid_search(data):
    """Efficient grid search with trial-level parallelization."""
    online = data['online'] 
    ID = data['name']
    boundaries = data['boundaries']
    minutes = data['sim time']
    dt_ms = data['dt ms']
    gains = np.array(data['gains'])
    projection = np.array(data['modular projections'])
    projection_operator = np.array(data['module operators'])
    noise = data['noise']
    vel = np.array(data['velocity'])
    pos_r = np.array(data['position'])
    pos_i = np.array(data['position internal'])
    pos_p = np.array(data['position prediction'])
    activity = np.array(data['activity'])
    
    vn, nn = noise
    nr_gains = [3, 4, 5, 6, 7]
    range_max = [1.0, 1.5, 2.0, 2.5, 3.0]
    start = 0.2
    nr_trial = 20
    dt = data.get("dt", 32/1000)
    T = len(vel)
    
    # Pre-allocate result arrays
    mse_trials = np.zeros((len(nr_gains), len(range_max), nr_trial))
    
    # Set up the processing function with fixed parameters
    process_func = partial(
        process_single_trial,
        vel=vel,
        pos_r=pos_r,
        vn=vn,
        nn=nn,
        dt=dt,
        start=start,
        nr_gains=nr_gains,
        range_max=range_max,
        T=T
    )
    
    # Process trials in parallel - this is coarse-grained parallelism
    # Each worker handles one complete trial with all parameter combinations
    num_processes = min(mp.cpu_count(), nr_trial)
    print(f"Running with {num_processes} parallel processes")
    
    with mp.Pool(processes=num_processes) as pool:
        trial_results = list(tqdm(
            pool.imap(process_func, range(nr_trial)),
            total=nr_trial,
            desc="Processing trials"
        ))
    
    # Collect results
    for trial, result in enumerate(trial_results):
        mse_trials[:, :, trial] = result
    
    # Compute statistics over trials
    mse_mean = np.mean(mse_trials, axis=2)
    mse_std = np.std(mse_trials, axis=2)
    
    # Find best gains (lowest mean MSE)
    idx_flat = np.argmin(mse_mean)
    best_idx = np.unravel_index(idx_flat, mse_mean.shape)
    best_gain = np.linspace(start, range_max[best_idx[1]], nr_gains[best_idx[0]])
    best_mse = mse_mean[best_idx]
    
    return {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "best_gain": best_gain,
        "best_mse": best_mse,
        "nr_gains": nr_gains,
        "range_max": range_max,
        "trials": nr_trial
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py \"Data Name\"")
        sys.exit(1)

    fname = sys.argv[1]
    #if not fname.startswith("Data "):
    #    fname = f"Data {fname}"

    print(f"Loading data from Results/ID {fname}/Data {fname}.pickle")
    data = load_data(fname)
    print("Running grid search...")
    results = optimized_grid_search(data)

    outname = fname.replace("Data ", "")
    save_results(outname, results)
    print(f"Saved analysis to Results/ID {outname}/Analysis {outname}.pickle")

if __name__ == "__main__":
    main()
