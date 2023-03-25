# reference:
#   S. Chikazumi "Mechanics of Ski Jump"
#    in "New Logergists vol.4" p71-92 (in Japanese)

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq,minimize_scalar
from scipy.constants import g
from schanze import h,X
from LiftDrag import lift,drag

def eom(y, t, u, rhom, form):
    """ equation of motion
    y[0:2] = coordinates (x,y) / m
    y[2:4] = polar coordinates (v,th) of
             velocity vector (v in m/s, th in rad)
    t = time / sec
    u = angle of attack (const) / rad
    rhom = rho/m = (air density)/mass / m^-3
    form = 0,1,2,3 (see Chikazumi)
    """
    x,y,v,th = y
    c,s = np.cos(th), np.sin(th)
    alpha = u - th
    p = rhom*v**2/2
    D = p*drag(alpha, form)
    L = p*lift(alpha, form)
    dv = -D - g*s
    dth = (L - g*c)/v
    return v*c, v*s, dv, dth

def FlightTime(form, v0, th0, u, rhom):
    """
    form = 0,1,2,3
    v0 = initial speed / m/s
    th0 = initial angle / rad
    u = angle of attack (const) / rad
    rho = rho/m = (air density)/mass / m^-3
    return flight time / sec
    """
    def fun(t):
        x,y = odeint(eom, [0,0,v0,th0], [0,t],
                     (u, rhom, form))[-1,:2]
        return y - h(x)

    return brentq(fun, 0, X[-1]/v0)

def FlightPath(form, v0, th0, u, rhom, t=None, N=100):
    """
    form = 0,1,2,3
    v0 = initial speed / m/s
    th0 = initial angle / rad
    u = angle of attack (const) / rad
    rhom = rho/m = (air density)/mass / m^-3
    t = array of times / sec
    if t is scalar, t <- np.linspace(0,t,N)
    if t is None, t <- FlightTime(...)
    N = number of points
    if t is 1d-array, N is ignored
    return x,y,v,th, each shape(N,)
    """
    if t is None:
        t = FlightTime(form, v0, th0, u, rhom)
    if np.isscalar(t):
        t = np.linspace(0,t,N)
    return odeint(eom, [0,0,v0,th0], t,
                  (u, rhom, form)).T

def MaxTime(form, v0, th0, rhom):
    """ maximize flight time
    form = 0,1,2,3
    v0 = initial speed / m/s
    th0 = initial angle / rad
    rhom = rho/m = (air density)/mass / m^-3
    return u,t for maximum t
    """
    def fun(u):
        return -FlightTime(form, v0, th0, u, rhom)

    r = minimize_scalar(fun, (0, np.pi/4))
    return r.x, -r.fun
