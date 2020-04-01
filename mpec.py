import numpy as np
from utils import mid, proj


def calculate_mid(c, l, u, a, lambd, M):
    n = len(c)
    x = np.zeros((n, ))
    for i in range(n):
        alpha = (c[i] + lambd*a[i])/M
        x[i] = mid(l[i], alpha, u[i])
    return x

def calculate_r(c, l, u, a, b, lambd, M):
    x = calculate_mid(c, l, u, a, lambd, M)
    return np.dot(a.reshape(1, -1), x.reshape(-1, 1)) - b

def bracketing(c, l, u, a, b, M, lambd=0, delta_lambd=2):
    r = calculate_r(c, l, u, a, b, lambd, M)
    if r < 0:
        lambd_l, r_l = lambd, r
        lambd += delta_lambd

        r = calculate_r(c, l, u, a, b, lambd, M)
        while r < 0:
            lambd_l, r_l = lambd, r
            s = max(r_l/r - 1, 0.1)
            delta_lambd += delta_lambd/s
            lambd += delta_lambd

            r = calculate_r(c, l, u, a, b, lambd, M)

        lambd_u, r_u = lambd, r
    else:
        lambd_u, r_u = lambd, r
        lambd -= delta_lambd

        r = calculate_r(c, l, u, a, b, lambd, M)
        while r > 0:
            lambd_u, r_u = lambd, r
            s = max(r_u/r - 1, 0.1)
            delta_lambd += delta_lambd/s
            lambd -= delta_lambd

            r = calculate_r(c, l, u, a, b, lambd, M)

        lambd_l, r_l = lambd, r

    return lambd_l, lambd_u, delta_lambd, r_l, r_u

def secant(c, l, u, a, b, M, lambd_l, lambd_u, delta_lambd, r_l, r_u):
    s = 1 - r_l/r_u
    delta_lambd /= s
    lambd = lambd_u - delta_lambd

    r = calculate_r(c, l, u, a, b, lambd, M)
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
        r = calculate_r(c, l, u, a, b, lambd, M)

    return calculate_mid(c, l, u, a, lambd, M)

def projection_step(c, l, u, a, b, M, lambd=0, delta_lambd=2):
    """ Algorithms for singly linearly constrained quadratic
        programs subject to lower and upper bounds.
        
        argmin{0.5*x.T*A*x - c.T*x | a.T*x = b, l <= x <= u}
        Case: A = M*I
    """
    lambd_l, lambd_u, delta_lambd, r_l, r_u = bracketing(c, l, u, a, b, M, lambd, delta_lambd)
    
    return secant(c, l, u, a, b, M, lambd_l, lambd_u, delta_lambd, r_l, r_u)

def P(z, L, rho, v, l, u, a, b, M):
    """Method studied by Dai and Flecher to solve P_M(z),
       where P_M(z) = argmin{0.5*z.T*M*z - c.T*z | a.T*z = b, l <= z <= u}
       
       For convergence M >= lambda_max(L)
    """
    c = M*z + rho*v - np.dot(L, z)
    if a is None and b is None:
        return calculate_mid(c, l, u, np.zeros((L.shape[0], )), 0, M)
    return projection_step(c, l, u, a, b, M, lambd=0, delta_lambd=2)

def accelerated_proximal_gradient_algorithm(L, rho, v, l, u, a, b, M, options={'maxiter': 100}):
    """An accelerated proximal gradient algorithm
       for singly linearly constrained quadratic programs
       with box constraints.
    """
    x = np.zeros((L.shape[0], ))
    z = np.zeros((L.shape[0], ))
    t = 1
    for i in range(options['maxiter']):
        x_next = P(z, L, rho, v, l, u, a, b, M)
        t_next = 0.5*(1 + np.sqrt(1 + 4*t**2))
        z = x_next + (x_next - x)*(t - 1)/t_next
        
        x = x_next
        t = t_next

    return x

def MPEC_EPM(L, l, u, a=None, b=None, rho=0.01, sigma=np.sqrt(10), T=10, options={'maxiter': 100, 'disp': False}):
    """MPEC-EPM: An Exact Penalty Method for Solving MPEC Problem.
       
       When rho > 2M (M - Lipschitz constant for f(x) in [-1, 1]),
       the biconvex optimizationhas the same local and global minima
       with the original problem.
       
       For f(x) = 0.5*x.T*L*x: M <= ||L||*sqrt(n)
    """
    n = L.shape[0]
    eigval_max = np.max(np.linalg.eigvals(L))
    M = np.sqrt(n)*eigval_max
    
    # initialization
    x = np.zeros((n,))
    v = np.zeros((n,))
    
    x_best = x
    f_min = np.inf
    for t in range(1, options['maxiter'] + 1):
        # x-Subproblem: Accelerated Proximal Gradient Algorithm
        #               for Singly Linearly Constrained Quadratic Programs
        #               with Box Constraints
        x = accelerated_proximal_gradient_algorithm(L, rho, v, l, u, a, b, M)
        
        # v-Subproblem:
        if np.linalg.norm(x) != 0:
            v = np.sqrt(n)*x/np.linalg.norm(x)
        else:
            v = np.ones((n,))

        f_cur = np.dot(np.dot(x.reshape(1, -1), L), x.reshape(-1, 1))[0][0]
        if f_cur < f_min:
            f_min = f_cur
            x_best = x
        
        # Update the penalty in every T iterations
        if t % T == 0:
            rho = min(2*M, rho*sigma)
        
        error = n - np.dot(x.reshape(1, -1), v.reshape(-1, 1))
        if (t > 10 and error < 1e-2):
            break
    
    x = proj(x_best, l, u)
    f = np.dot(np.dot(x.reshape(1, -1), L), x.reshape(-1, 1))[0][0]
    if options['disp']:
        print('Function value: {}\n'.format(f)+
              'Iterations number: {}'.format(t))
    return x, f

def break_point_search(d, a, s):
    """A polynomially bounded algorithm for a singly constrained quadratic program."""
    n = len(d)

    y = np.sort(np.hstack((-a.flatten() - d.flatten()*np.ones((n,)), -a.flatten())))
    l = 1
    r = 2*n
    L = n
    R = 0
    while 1:
        if (r - l == 1):
            S = (s - L) / (R - L)
            rho = y[l-1] + (y[r-1] - y[l-1]) * S
            break
        else:
            m = int(0.5 * (l + r))
            t = (-a.flatten() - y[m-1]*np.ones((n,)))/d.flatten()
            C = np.sum(np.array([max(0, min(t_i, 1)) for t_i in t]))
            if (C == s):
                rho = y[m-1]
                break
            elif (C > s):
                l = m
                L = C
            else:
                r = m
                R = C
    t = (-a.flatten() - rho)/d.flatten()
    x = np.array([max(0, min(t_i, 1)) for t_i in t])
    
    return x

def original_MPEC_EPM(L, k, rho=0.01, sigma=np.sqrt(10), T=10, options={'maxiter': 100, 'disp': False}):
    """Original realization of MPEC-EPM algorithm by authors Yuan, Ganzhao & Ghanem, Bernard of the paper
       "Binary Optimization via Mathematical Programming with Equilibrium Constraints", 2016.
    """
    n = L.shape[0]
    M = np.max(np.linalg.eigvals(L))
    
    x = np.zeros((n,))
    v = np.zeros((n,))
    
    x_best = x
    f_min = np.inf
    for t in range(1, options['maxiter'] + 1):
        for i in range(50):
            grad = np.dot(L, x) - rho*2*(2*v - 1)
            xt = x
            x = break_point_search(np.ones((n, 1)), -(x - grad/M), k)
            if (i > 5) and (np.linalg.norm(x - xt)/np.linalg.norm(x) < 1e-5):
                break
                
        v = (np.sqrt(n)*(2*x - 1)/np.linalg.norm(2*x - 1) + 1)/2

        f_cur = np.dot(np.dot(x.reshape(1, -1), L), x.reshape(-1, 1))[0][0]
        if f_cur < f_min:
            f_min = f_cur
            x_best = x

        if t % T == 0:
            rho *= sigma
        
        error = n - np.dot((2*x - 1).reshape(1, -1), (2*v - 1).reshape(-1, 1))
        if (t > 10 and error < 1e-2):
            break

    def original_proj(a, k):
        x = np.zeros((len(a), 1))
        x[np.argsort(abs(a))[::-1][:k]] = 1
        return x
    
    x = original_proj(x_best, k)
    f = np.dot(np.dot(x.reshape(1, -1), L), x.reshape(-1, 1))[0][0]
    if options['disp']:
        print('Function value: {}\n'.format(f)+
              'Iterations number: {}'.format(t))
    return x, f
