import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import degree,g
from fopt import fopt
from schanze import X,Y
from jump import FlightTime
from OptJump import f,f_q,f_u,phi,psi

v0 = 20 # init speed / m/s
th0 = 5*degree  # init angle
u0 = 20*degree # angle of attack (init guess)
m = 82 # weight / kg
rho = 1.274 # air density / kg/m^3
k = 0.01 # update step size
maxit = 100 # maximum number of iterations
N = 128 # number of grid points in time

H = abs(Y[-1]) # length scale
vc = np.sqrt(g*H) # velocity scale
tc = H/vc # time scale
rhom = rho/m
rhm = rhom*H/2
u0 = np.full(N, u0)

result = []
for form in range(4):
    t = FlightTime(form, v0, th0, u0[0], rhom)
    r = fopt(f, f_q, f_u, phi, psi, u0, t/tc,
                     [0,0,v0/vc,th0], (rhm, form),
                     k=k, maxit=maxit)
    result.append(r)

### fig3 #################################################

plt.figure(figsize=(10, 4.88))
plt.axis('equal')

for form,r in enumerate(result):
    x,y,v,th = r[2]
    plt.plot(x*H, y*H, label='%d'%(form+1))
    print(x[-1]*H)

plt.plot(X, Y, 'k')
plt.plot([0],[0],'k*')
plt.legend()
plt.xlabel(r'$x$ / meter')
plt.ylabel(r'$y$ / meter')
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()

### fig4 #################################################

plt.figure(figsize=(5, 3.75))

for form,r in enumerate(result):
    t,u = r[0],r[1]
    plt.plot(t*tc, u/degree, label='%d'%(form+1))

plt.legend()
plt.xlabel(r'$t=$time / sec')
plt.ylabel(r'$u=$angle of attack / deg')
plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
