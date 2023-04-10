#!/usr/bin/env python3
import pandas as pd

def from_file(filename, delimiter):
    # Load the data from the file into a pandas DataFrame
    df = pd.read_csv(filename, sep=delimiter)
    return df
