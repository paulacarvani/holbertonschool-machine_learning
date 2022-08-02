#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here
"""x ↦ y1 should be plotted with a dashed red line"""
C14 = plt.plot(x, y1, 'r--', label='C-14')

"""x ↦ y2 should be plotted with a solid green line"""
Ra226 = plt.plot(x, y2, 'green', label='Ra-226')

"""The x-axis should be labeled Time (years)"""
plt.xlabel('Time (years)')

"""The y-axis should be labeled Fraction Remaining"""
plt.ylabel('Fraction Remaining')

"""The title should be Exponential Decay of Radioactive Elements"""
plt.title('Exponential Decay of Radioactive Elements')

"""The x-axis should range from 0 to 20,000"""
plt.xlim(0, 20000)

"""The y-axis should range from 0 to 1"""
plt.ylim(0, 1)

"""A legend labeling x ↦ y1 as C-14 and x ↦ y2 as Ra-226 should
be placed in the upper right hand corner of the plot"""
plt.legend(loc='upper right')

"""Displays the graphic"""
plt.show()
