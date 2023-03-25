# reference:
#   S. Chikazumi "Mechanics of Ski Jump"
#    in "New Logergists vol.4" p71-92 (in Japanese)

import numpy as np
from scipy.constants import degree

# lift and drag coefficients (area S = 1 m^2)
data = np.asarray([
    [0.067,0.0123,0.133,0.00357,0.000175], # form 1
    [0.086,0.0088,0.181,0.00287,0.000175], # form 2
    [0.278,0.0077,0.302,0.0124,0.000175],  # form 3
    [0.128,0.0102,0.213,0.0044,0.000175]]) # form 4

# unit of angle is radian
data[:,[1,3]] /= degree
data[:,4] /= degree**2

def lift(a,j): # a = angle of attack / rad
    return data[j,0] + data[j,1]*a

def drag(a,j):
    return data[j,2] + data[j,3]*a + data[j,4]*a**2

# derivatives
def dLift(a,j): return data[j,1]
def dDrag(a,j): return data[j,3] + 2*data[j,4]*a
