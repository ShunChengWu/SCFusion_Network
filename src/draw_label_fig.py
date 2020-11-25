#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:52:58 2020

@author: sc
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

NYU_13_CLASSEScolour_code = [[  255,   255,   255],
       [  0,   0, 255],
       [232,  88,  47],
       [  0, 217,   0],
       [148,   0, 240],
       [222, 241,  23],
       [255, 205, 205],
       [  0, 223, 228],
       [106, 135, 204],
       [116,  28,  41],
       [240,  35, 235],
       [  0, 166, 156],
       [249, 139,   0],
       [225, 228, 194]]
NYU_13_CLASSES = [(0,'Unknown'),
                  (1,'Bed'),
                  (2,'Books'),
                  (3,'Ceiling'),
                  (4,'Chair'),
                  (5,'Floor'),
                  (6,'Furniture'),
                  (7,'Objects'),
                  (8,'Picture'),
                  (9,'Sofa'),
                  (10,'Table'),
                  (11,'TV'),
                  (12,'Wall'),
                  (13,'Window')
]


NYU_13_float = [[a/255 for a in color] for color in NYU_13_CLASSEScolour_code]

N = len(NYU_13_CLASSEScolour_code)
# make an empty data set
data = np.ones((1, N)) * np.nan
# fill in some fake data
for i in range(N):
    data[0,i] = i
    
# make a figure + axes
fig, ax = plt.subplots(1, 1, tight_layout=True)
# make color map
my_cmap = matplotlib.colors.ListedColormap(NYU_13_float)
# set the 'bad' values (nan) to be white and transparent
my_cmap.set_bad(color='w', alpha=0)
# draw the grid
for x in range(N + 1):
    ax.axhline(x, lw=2, color='k', zorder=5)
    ax.axvline(x, lw=2, color='k', zorder=5)
# draw the boxes
ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N, 0, 1], zorder=0)
# turn off the axis labels
ax.axis('off')
