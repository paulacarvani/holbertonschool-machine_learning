#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
"""Limits"""
plt.xlim([0, 100])
plt.ylim([0, 30])

"""The x-axis should be labeled Grades"""
plt.xlabel('Grades')

"""The y-axis should be labeled Number of Students"""
plt.ylabel('Number of Students')

"""The x-axis should have bins every 10 units"""
"""The bars should be outlined in black"""
x = np.linspace(0, 100, 11)
plt.hist(student_grades, bins=x, edgecolor='black')
plt.xticks(x)

"""The title should be Project A"""
plt.title('Project A')

"""Displays the graphic"""
plt.show()
