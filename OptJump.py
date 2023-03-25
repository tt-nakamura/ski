import numpy as np
from LiftDrag import lift,drag,dLift,dDrag
from schanze import h,dh,Y

H = abs(Y[-1]) # length scale

def f(t,q,u,c,form):# equation of motion
    x,y,v,th = q
    ct,st = np.cos(th),np.sin(th)
    alpha = u - th
    cv2 = c*v**2 # c = rho*S*H/2
    D = cv2*drag(alpha, form)
    L = cv2*lift(alpha, form)
    return v*ct, v*st, -D-st, (L-ct)/v

def f_q(t,q,u,c,form): # df/dq
    x,y,v,th = q
    ct,st = np.cos(th),np.sin(th)
    alpha = u - th
    cv2 = c*v**2 # c = rho*S*H/2
    D = cv2*drag(alpha, form)
    L = cv2*lift(alpha, form)
    dD = cv2*dDrag(alpha, form)
    dL = cv2*dLift(alpha, form)
    return [[0, 0, ct, -v*st],
            [0, 0, st, v*ct],
            [0, 0, -2*D/v, dD-ct],
            [0, 0, (L+ct)/v**2, (st-dL)/v]]

def f_u(t,q,u,c,form): # df/du
    x,y,v,th = q
    alpha = u - th
    cv2 = c*v**2 # c = rho*S*H/2
    dD = cv2*dDrag(alpha, form)
    dL = cv2*dLift(alpha, form)
    z = np.zeros_like(t) # vectorize
    return [z, z, -dD, dL/v]

def phi(t,q): # function to minimize
    x,y,v,th = q
    phi = y-x
    phi_t = 0 # d(phi)/dt
    phi_q = [-1,1,0,0] # d(phi)/dq
    return phi, phi_t, phi_q

def psi(t,q): # constraints to be zeroed
    x,y,v,th = q
    psi = y - h(x*H)/H
    psi_t = 0 # d(psi)/dt
    psi_q = [-dh(x*H), 1, 0, 0] # d(psi)/dq
    return psi, psi_t, psi_q
