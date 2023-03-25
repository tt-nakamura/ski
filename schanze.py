import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

scale = 26.1/(216.137 - 35.1371) # meter / pixel

# length between jump point and left edge of curve
step = scale*(140.113 - 116.887)

a = plt.imread("schanze.png")[:,:,0]
i = np.r_[np.arange(26, 671, 20), 670]
a = np.diff(a[::-1, i], axis=0)
Y = scale*np.argmax(a<-0.23, axis=0)
Y -= Y[0] + step
X = scale*(i - i[0])

h = CubicSpline(X,Y)
dh = h.derivative()
