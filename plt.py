# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:50:23 2024

@author: 91961
"""

import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8]
y = [6,7,8,9,10,11,12,13]

plt.plot(x,y,marker='^',linestyle="dashed")
plt.title("plot title")
plt.xlabel("Days")
plt.ylabel("Count")
plt.show()


