import numpy as np
import matplotlib.pyplot as plt
from jump import FlightPath
from scipy.constants import degree
from schanze import X,Y

v0 = 20 # initial velocity / m/s
th0 = 5*degree # initial angle
u = 20*degree # angle of attack
rho = 1.274 # air density / kg/m^3
m = 82 # mass of jumper / kg

plt.figure(figsize=(10, 4.88))
plt.axis('equal')

for form in range(4):
    x,y,v,th = FlightPath(form, v0, th0, u, rho/m)
    plt.plot(x,y,label='%d'%(form+1))
    print(x[-1])

plt.plot(X,Y,'k')
plt.plot([0],[0],'k*')
plt.xlabel(r'$x$ / meter')
plt.ylabel(r'$y$ / meter')
plt.legend()
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
