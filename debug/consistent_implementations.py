# Solutions to make both implementations produce the same results
import numpy as np
import pandas as pd
from src.torch_bmi import bmi_lstm
import torch
from contextlib import contextmanager

# Better approach: Use isolated random states to avoid affecting other models

def ensure_reproducibility_isolated(seed=42):
    """Set seeds without affecting global NumPy random state"""
    # Save current NumPy random state
    numpy_state = np.random.get_state()
    
    # Set PyTorch seed (this is safe - doesn't affect other models)
    torch.manual_seed(seed)
    
    # If using CUDA/GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic CUDNN behavior
    if hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
    
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = False
    
    return numpy_state

def restore_numpy_state(numpy_state):
    """Restore the original NumPy random state"""
    np.random.set_state(numpy_state)

# Alternative: Create isolated random number generator
def create_isolated_rng(seed=42):
    """Create an isolated random number generator that doesn't affect global state"""
    rng = np.random.RandomState(seed)  # Or np.random.default_rng(seed) for newer NumPy
    return rng

# Even better: Context manager for temporary seed setting

@contextmanager
def temporary_seed(seed):
    """Context manager to temporarily set NumPy seed without affecting global state"""
    # Save current state
    original_state = np.random.get_state()
    
    try:
        # Set temporary seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        if hasattr(torch.backends.cudnn, 'deterministic'):
            torch.backends.cudnn.deterministic = True
        
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = False
            
        yield
        
    finally:
        # Always restore original state
        np.random.set_state(original_state)

# Option 1: Modify Array Implementation to Process One Step at a Time
def array_implementation_fixed(config_file, pn, length=5):
    """Array implementation that processes one step at a time like the loop"""
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    lstm = bmi_lstm()
    lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
    print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
    
    T_C_mu_arr, T_C_sd_arr = np.zeros(length), np.zeros(length)
    
    # Get all data upfront but process one step at a time
    unscaled_data_all = lstm.get_unscaled_values(lead_time=length-1)
    
    for i in range(length):
        # Extract single time step from the pre-loaded data
        unscaled_data_single = {}
        for var in lstm.x_vars:
            unscaled_data_single[var] = unscaled_data_all[var][i]  # Get i-th time step
        
        # Set single time step values
        for var in lstm.x_vars:
            lstm.set_value(var, unscaled_data_single[var])
        
        # Process one time step
        T_C_mu, T_C_sd = lstm.update()
        T_C_mu_arr[i], T_C_sd_arr[i] = T_C_mu, T_C_sd
    
    return T_C_mu_arr, T_C_sd_arr

def loop_implementation_original(config_file, pn, length=5):
    """Original loop implementation"""
    # Set seed for reproducibility  
    torch.manual_seed(42)
    
    lstm = bmi_lstm()
    lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
    print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
    
    T_C_mu_loop, T_C_sd_loop = np.zeros(length), np.zeros(length)
    unscaled_data_loop = pd.DataFrame()
    for i in range(length):
        unscaled_data = lstm.get_unscaled_values(lead_time=0)
        unscaled_data_loop = pd.concat([unscaled_data_loop, unscaled_data])
        for var in lstm.x_vars:
            lstm.set_value(var, unscaled_data[var])

        T_C_mu, T_C_sd = lstm.update()
        T_C_mu_loop[i], T_C_sd_loop[i] = T_C_mu, T_C_sd
    
    return T_C_mu_loop, T_C_sd_loop

# Option 2: Use update_until() method (most consistent approach)
def update_until_implementation(config_file, pn, length=5):
    """Use the built-in update_until method for consistent batch processing"""
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    lstm = bmi_lstm()
    lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
    print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
    
    # Get all data for the entire sequence
    unscaled_data = lstm.get_unscaled_values(lead_time=length-1)
    
    # Set all data at once
    for var in lstm.x_vars:
        lstm.set_value(var, unscaled_data[var])
    
    # Use update_until for consistent batch processing
    lstm.update_until(length-1)
    
    # Extract results
    mu_pred = np.zeros(length)
    sd_pred = np.zeros(length)
    lstm.get_value("channel_water_surface_water__mu_max_of_temperature", mu_pred)
    lstm.get_value("channel_water_surface_water__sd_max_of_temperature", sd_pred)
    
    return mu_pred, sd_pred

# Usage example:
# T_C_mu_arr, T_C_sd_arr = array_implementation_fixed(config_file, pn)
# T_C_mu_loop, T_C_sd_loop = loop_implementation_original(config_file, pn)
# T_C_mu_until, T_C_sd_until = update_until_implementation(config_file, pn)
# 
# print("Array vs Loop match:", np.allclose(T_C_mu_arr, T_C_mu_loop))
# print("Array vs Until match:", np.allclose(T_C_mu_arr, T_C_mu_until))

# Usage example for your LSTM comparison:
def safe_lstm_comparison(config_file, pn, length=5):
    """Compare LSTM implementations without affecting other models"""
    
    # Save original state
    original_numpy_state = ensure_reproducibility_isolated(42)
    
    try:
        # Run your LSTM implementations
        T_C_mu_arr, T_C_sd_arr = array_implementation_fixed(config_file, pn, length)
        
        # Reset torch seed for second implementation
        torch.manual_seed(42)
        T_C_mu_loop, T_C_sd_loop = loop_implementation_original(config_file, pn, length)
        
        return T_C_mu_arr, T_C_sd_arr, T_C_mu_loop, T_C_sd_loop
        
    finally:
        # Always restore original NumPy state
        restore_numpy_state(original_numpy_state)

# Example usage of temporary seed context manager
def your_lstm_comparison(config_file, pn, length=5):
    """Example usage of temporary seed context manager"""
    
    # This affects other models - NOT RECOMMENDED
    # np.random.seed(42)
    
    # This is safe - other models unaffected
    with temporary_seed(42):
        T_C_mu_arr, T_C_sd_arr = array_implementation_fixed(config_file, pn, length)
    
    with temporary_seed(42):
        T_C_mu_loop, T_C_sd_loop = loop_implementation_original(config_file, pn, length)
    
    # NumPy random state is automatically restored here
    # Other models will continue with their original random state
    
    return T_C_mu_arr, T_C_sd_arr, T_C_mu_loop, T_C_sd_loop
