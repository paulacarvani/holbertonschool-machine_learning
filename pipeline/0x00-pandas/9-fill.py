#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df.drop('Weighted_Price', axis=1, inplace=True)

# Fill in missing values
df[['Close']] = df[['Close']].fillna(method='ffill')
df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'])
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

print(df.head())
print(df.tail())
