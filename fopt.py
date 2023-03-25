# referece:
#   A. E. Bryson, "Dynamic Optimization" section 4.5

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def fopt(f, f_q, f_u, phi, psi, u, tf, q0,
         args=(), k=0.2, eta=1, tol=1e-4, maxit=100, verbose=True):
    """ function optimization with
          terminal constraints and open final time
        translated from MATLAB code in Bryson, TABLE4.2
    f : callable
        equation of motion, dq/dt = f(t,q,u,*args)
        t = time, q = state vector (shape(N,))
        u = control variables (shape(K,) or scalar)
        shape of return value is (N,)
    f_q : callable
        derivative of f with respect to q
        df/dq = f_q(t,q,u,*args)
        shape of return value is (N,N)
    f_u : callable
        derivative of f with respect to u
        df/du = f_u(t,q,u,*args)
        f_u must be vectorized so that if
        shapes of t,q,u are (n,),(N,n),(K,n)
        then shape of return value is (N,K,n)
        if K==1, shape of f_u can be (N,n)
    phi : callable
        function to optimize and its derivatives
        phi(t,q) must return ph, ph_t, ph_q where
        ph = objective function (scalar)
        ph_t = derivative of ph wrt t (scalar)
        ph_q = derivative of ph wrt q (shape(,N))
    psi : callable
        terminal constraints and their derivatives
        psi(t,q) must return ps, ps_t, ps_q where
        ps = constraints to be zeroed (shape(M,))
             M = number of constraints
        ps_t = derivative of ps wrt t (shape(M,))
        ps_q = derivative of ps wrt q (shape(M,N))
        if M==1, ps and ps_t can be scalar and
                 shape of ps_q can be (N,)
    u : 2d-array (shape(K,n))
        u[i,j] = j-th control variable at time i*tf/(n-1)
        n = number of time steps
        K = number of control variables
        if K==1, u can be 1d-array of shape(n,)
    tf : scalar
        initial guess for final time
    q0 : 1d-array (shape(N,))
        intial state vector at time t=0
    args : tuple
        extra arguments to be passed to f, f_q, f_u
        (args are not passed to phi and psi)
    k : scalar or 1d-array
        update step size along grad(phi) direction
        k>0 for minimization, k<0 for maximization
        if 1d-array, len(k) must be maxit
    eta : scalar or 1d-array, 0<eta<=1
        update step size along grad(psi) direction
        if eta==1, constraints are satisfied (see Bryson)
        if 1d-array, len(eta) must be maxit
    tol : scalar
        relative error tolerance such that
        if |update in tf| < tol*tf and
           |updates in u| < tol*|u|, then exit 
    maxit : int, maxmum number of update iterations
    verbose : bool, print message or not
    return t,u,q,lam,nu
        t = time sequence (shape(,n))
        u = control variables (same shape as input u)
        q = state vector sequence (shape(N,n))
        lam = adjoint vector sequence (shape(N,n,))
        nu = Lagrange multiplier for constraints (shape(M,))
    assume N>=2, n>=2
    assume all variables are dimensionless and
           their values are of order unity
    """
    u = np.asarray(u)
    N = u.shape[-1]
    if np.isscalar(k): k = np.full(maxit, k)
    if np.isscalar(eta): eta = np.full(maxit, eta)
    if verbose:
        print(' No.'+
              '        OBJ.fun'+
              '    max delta u'+
              '       delta tf')
    for i,(k,eta) in enumerate(zip(k,eta)):
        t = np.linspace(0,tf,N)
        u_interp = interp1d(t,u,'cubic',fill_value='extrapolate')
        def fw(q,t):
            u = u_interp(t)
            return f(t, q, u, *args)

        q = odeint(fw, q0, t).T
        q_interp = interp1d(t,q,'cubic',fill_value='extrapolate')
        def bk(lam,t):
            u = u_interp(t)
            q = q_interp(t)
            return -np.dot(lam, f_q(t, q, u, *args))

        qf = q[:,-1]
        ph,ph_t,ph_q = phi(tf, qf)
        ps,ps_t,ps_q = psi(tf, qf)
        ps   = np.atleast_1d(ps) # shape(M,)
        ps_t = np.atleast_1d(ps_t) # shape(M,)
        ps_q = np.atleast_2d(ps_q) # shape(M,N)
        fu = np.asarray(f_u(t, q, u, *args))
        if fu.ndim==2: fu = np.expand_dims(fu,1) # shape(N,K,n)
        lam = odeint(bk, ph_q, t[::-1])[::-1].T
        lfu = np.einsum('ik,ijk->jk', lam, fu) # shape(K,n)
        mu = [odeint(bk, p, t[::-1])[::-1].T for p in ps_q] # shape(M,N,n)
        mfu = np.einsum('ijk,jlk->ilk', mu, fu) # shape(M,K,n)
        q_dot = f(tf, qf, u[...,-1], *args) # shape(N,)
        phi_dot = ph_t + np.dot(ph_q, q_dot) # scalar
        psi_dot = ps_t + np.dot(ps_q, q_dot) # shape(M,)
        P = np.sum(np.trapz(mfu*lfu, t), axis=-1) # shpae(M,)
        P+= phi_dot*psi_dot # shpae(M,)
        Q = np.einsum('ijk,ljk->iljk', mfu, mfu) # shape(M,M,K,n)
        Q = np.sum(np.trapz(Q,t), axis=-1) # shape(M,M)
        Q+= np.outer(psi_dot, psi_dot) # shape(M,M)
        Q = np.linalg.inv(Q) # shape(M,M)
        nu = np.dot(Q, eta*ps/k - P) # shape(M,)
        du = -k*(lfu + np.einsum('i,ijk', nu, mfu)) # shape(K,n)
        dt = -k*(phi_dot + np.dot(nu, psi_dot)) # scalar
        u += du.reshape(u.shape)
        tf += dt
        du = np.max(np.abs(du))
        if du <= tol * np.max(np.abs(u)) and\
           np.abs(dt) <= tol * np.abs(tf): break
        if verbose:
            print('%4d%15.6g%15.6g%15.6g'%(i+1, ph, du, dt))

    lam += np.einsum('i,ijk', nu, mu)
    return t,u,q,lam,nu
