import numpy as np
from utils import mid


def calculate_mid(c, a, l, u, lambd, M):
    n = len(c)
    x = np.zeros((n, ))
    for i in range(n):
        alpha = (c[i] + lambd*a[i])/M
        x[i] = mid(l[i], alpha, u[i])
    return x

def calculate_r(c, a, b, l, u, lambd, M):
    x = calculate_mid(c, a, l, u, lambd, M)
    return np.dot(a.reshape(1, -1), x.reshape(-1, 1)) - b

def bracketing(c, a, b, l, u, M, lambd=0, delta_lambd=2):
    r = calculate_r(c, a, b, l, u, lambd, M)
    if r < 0:
        lambd_l, r_l = lambd, r
        lambd += delta_lambd

        r = calculate_r(c, a, b, l, u, lambd, M)
        while r < 0:
            lambd_l, r_l = lambd, r
            s = max(r_l/r - 1, 0.1)
            delta_lambd += delta_lambd/s
            lambd += delta_lambd

            r = calculate_r(c, a, b, l, u, lambd, M)

        lambd_u, r_u = lambd, r
    else:
        lambd_u, r_u = lambd, r
        lambd -= delta_lambd

        r = calculate_r(c, a, b, l, u, lambd, M)
        while r > 0:
            lambd_u, r_u = lambd, r
            s = max(r_u/r - 1, 0.1)
            delta_lambd += delta_lambd/s
            lambd -= delta_lambd

            r = calculate_r(c, a, b, l, u, lambd, M)

        lambd_l, r_l = lambd, r

    return lambd_l, lambd_u, delta_lambd, r_l, r_u

def secant(c, a, b, l, u, M, lambd_l, lambd_u, delta_lambd, r_l, r_u):
    s = 1 - r_l/r_u
    delta_lambd /= s
    lambd = lambd_u - delta_lambd

    r = calculate_r(c, a, b, l, u, lambd, M)
    while abs(r) > 1e-2:
        if r > 0:
            if s <= 2:
                lambd_u, r_u = lambd, r
                s = 1 - r_l/r_u
                delta_lambd = (lambd_u - lambd_l)/s
                lambd = lambd_u - delta_lambd
            else:
                s = max(r_u/r - 1, 0.1)
                delta_lambd = (lambd_u - lambd)/s
                lambd_u, r_u = lambd, r
                lambd = max(lambd - delta_lambd, 0.75*lambd_l + 0.25*lambd)
                s = (lambd_u - lambd_l)/(lambd_u - lambd)
        else:
            if s >= 2:
                lambd_l, r_l = lambd, r
                s = 1 - r_l/r_u
                delta_lambd = (lambd_u - lambd_l)/s
                lambd = lambd_u - delta_lambd
            else:
                s = max(r_l/r - 1, 0.1)
                delta_lambd = (lambd - lambd_l)/s
                lambd_l, r_l = lambd, r
                lambd = min(lambd + delta_lambd, 0.75*lambd_u + 0.25*lambd)
                s = (lambd_u - lambd_l)/(lambd_u - lambd)
        r = calculate_r(c, a, b, l, u, lambd, M)

    return calculate_mid(c, a, l, u, lambd, M)

def projection_step(c, a, b, l, u, M, lambd=0, delta_lambd=2):
    """ Algorithms for singly linearly constrained quadratic
        programs subject to lower and upper bounds.
        
        argmin{0.5*x.T*A*x - c.T*x | a.T*x = b, l <= x <= u}
        Case: A = M*I
    """
    lambd_l, lambd_u, delta_lambd, r_l, r_u = bracketing(c, a, b, l, u, M, lambd, delta_lambd)
    
    return secant(c, a, b, l, u, M, lambd_l, lambd_u, delta_lambd, r_l, r_u)

def P(z, L, rho, v, a, b, l, u, M):
    """Method studied by Dai and Flecher to solve P_M(z),
       where P_M(z) = argmin{0.5*z.T*M*z - c.T*z | a.T*z = b, l <= z <= u}
       
       For convergence M >= lambda_max(L)
    """
    c = M*z + rho*v - np.dot(L, z)
#     return projection_step(c, a, b, l, u, L.shape[0]*M, lambd=0, delta_lambd=2)
    return calculate_mid(c, a, l, u, 0, M)

def accelerated_proximal_gradient_algorithm(L, rho, v, a, b, l, u, M, options={'maxiter': 1000}):
    """An accelerated proximal gradient algorithm
       for singly linearly constrained quadratic programs
       with box constraints.
    """
    x = np.zeros((L.shape[0], ))
    for i in range(options['maxiter']):
        x = P(x, L, rho, v, a, b, l, u, M)

    return x

def MPEC_EPM(L, a, b, l, u, rho=0.01, sigma=np.sqrt(10), T=10, options={'maxiter': 100}):
    """MPEC-EPM: An Exact Penalty Method for Solving MPEC Problem.
       
       When rho > 2M (M - Lipschitz constant for f(x) in [-1, 1]),
       the biconvex optimizationhas the same local and global minima
       with the original problem.
       
       For f(x) = 0.5*x.T*L*x: M <= ||L||*sqrt(n)
    """
    n = L.shape[0]
    eigval_max = max(np.linalg.eigvals(L))
    M = np.sqrt(n)*eigval_max
    
    # initialization
    x = np.zeros((n,))
    v = np.zeros((n,))
    
    for t in range(1, options['maxiter'] + 1):
        # x-Subproblem: Accelerated Proximal Gradient Algorithm
        #               for Singly Linearly Constrained Quadratic Programs
        #               with Box Constraints
        x = accelerated_proximal_gradient_algorithm(L, rho, v, a, b, l, u, eigval_max)
        
        # v-Subproblem:
        if np.linalg.norm(x) != 0:
            v = np.sqrt(n)*x/np.linalg.norm(x)
        else:
            v = np.ones((n,))

        # Update the penalty in every T iterations
        if t % T == 0:
            rho = min(2*M, rho*sigma)

    return x

