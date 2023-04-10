#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Get the last 10 rows of the High and Close columns and convert them to a numpy.ndarray
A = pd.array(df[['High', 'Close']].tail(10))

print(A)