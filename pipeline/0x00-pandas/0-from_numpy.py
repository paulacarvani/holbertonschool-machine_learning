#!/usr/bin/env python3
import pandas as pd
import numpy as np

def from_numpy(array):
    # Get the number of columns in the array
    num_cols = array.shape[1]
    # Get a list of column labels
    col_labels = [chr(i+65) for i in range(num_cols)]
    # Create a dictionary of column names and array columns
    col_dict = {label: array[:,i] for i,label in enumerate(col_labels)}
    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(col_dict)
    return df
