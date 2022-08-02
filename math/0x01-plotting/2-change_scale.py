#!/usr/bin/env python3
from cmath import log
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here
plt.plot(x, y)
"""The x-axis should be labeled Time (years)"""
plt.xlabel('Time (years)')
"""The y-axis should be labeled Fraction Remaining"""
plt.ylabel('Fraction Remaining')
"""The title should be Exponential Decay of C-14"""
plt.title('Exponential Decay of C-14')
"""The y-axis should be logarithmically scaled"""
plt.yscale('log')
"""The x-axis should range from 0 to 28650"""
plt.xlim([0, 28650])
"""Displays the graphic"""
plt.show()
