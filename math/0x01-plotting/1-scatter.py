#!/usr/bin/env python3
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here
"""The x-axis should be labeled Height (in)"""
plt.xlabel('Height (in)')
"""The y-axis should be labeled Weight (lbs)"""
plt.ylabel('Weight (lbs)')
"""The title should be Men's Height vs Weight"""
plt.title('Men\'s Height vs Weight')
"""The data should be plotted as magenta points"""
plt.scatter(x, y, color='magenta')
"""Displays the graphic"""
plt.show()