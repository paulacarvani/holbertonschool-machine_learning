#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code here
plt.scatter(x, y, c=z)

"""The x-axis should be labeled x coordinate (m)"""
plt.xlabel('x coordinate (m)')

"""The y-axis should be labeled y coordinate (m)"""
plt.ylabel('y coordinate (m)')

"""The title should be Mountain Elevation"""
plt.title('Mountain Elevation')

"""A colorbar should be used to display elevation"""
"""The colorbar should be labeled elevation (m)"""
color_bar = plt.scatter(x, y, s=z, c=z)
plt.colorbar(color_bar, label="elevation (m)")

plt.show()
