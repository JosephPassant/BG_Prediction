import pandas as pd
import numpy as np
import os


def separate_ptid_data(df):
    """
    Separates the data into individual patient time series.
    
    Args:
    - df: pandas DataFrame containing all patient data
    
    Returns:
    - data_dict: Dictionary containing DataFrames for each patient, with PtID as the key
    """
    data_dict = {} # Initialize dictionary to store DataFrames

    for group in df.groupby('PtID'): # Iterate through each patient group
        PtID = group[0] # Extract the PtID
        data_dict[PtID] = group[1].reset_index(drop=True) # Store the DataFrame in the dictionary with PtID as the key

    return data_dict

def align_start_date(data_dict, base_date="2000-01-01"):
    """
    Aligns the start date of the data for each patient
    
    Args:
    data_dict: dictionary containing dataframes for each patient
    base_date: date to align the data to
    
    Returns:
    dictionary containing dataframes for each patient with aligned start date
    """
    
    updated_data_dict = {}
    
    for ptid, df in data_dict.items():
        # Calculate the difference in days between the base date and the start date
        start_date = df['DateTime'].min()
        days_diff = (start_date - pd.to_datetime(base_date)).days
        
        # Update the DateTime column by subtracting the difference in days
        df['DateTime'] = df['DateTime'] - pd.to_timedelta(days_diff, unit='D')
        
        # Store the updated DataFrame in the dictionary
        updated_data_dict[ptid] = df
    
    return updated_data_dict

def undersample_dict(original_dict, sample_size):
    """Returns a new dictionary with `sample_size` random samples from `original_dict`."""
    if len(original_dict) <= sample_size:  # No need to sample if already within limits
        return original_dict  
    sampled_keys = np.random.choice(list(original_dict.keys()), sample_size, replace=False)
    return {k: original_dict[k] for k in sampled_keys}


def get_first_file(directory, file_extension=".pt"):
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    if files:
        return os.path.join(directory, files[0])
    return None