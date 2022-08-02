#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
# your code here
"""r is for red, line by default"""
plt.plot(y, 'r')
"""Range of x"""
plt.xlim([0, 10])
"""Displays the graphic"""
plt.show()
