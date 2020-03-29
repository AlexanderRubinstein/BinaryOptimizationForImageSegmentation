import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint


def SDP_relaxation(L, l, u, options={'maxiter': 100, 'disp': False}):
    """Binary optimization with SDP relaxation."""
    def objective(L):
        def fun(x):
            X = np.dot(x.reshape(-1, 1),
                       x.reshape(1, -1))
            return np.trace(np.dot(L.T, X))
        return fun
    
    def trace_constraint(x):
        return np.diag(np.dot(x.reshape(-1, 1), x.reshape(1, -1))) - np.ones((n,))
    
    n = L.shape[0]
    fun = objective(L)
    
    x_best = np.zeros((n,))
    f_min = np.inf
    for i in range(options['maxiter']):
        x = np.random.choice([-1, 1], (1, n)).ravel()
        res = minimize(fun, x, method='SLSQP',
                       bounds=[(l_i, u_i) for l_i, u_i in zip(l, u)],
                       constraints=[{'type': 'eq', 'fun': trace_constraint}],
                       options={'maxiter': 10})
        
        x = res.x
        f_cur = np.dot(np.dot(x.reshape(1, -1), L), x.reshape(-1, 1))[0][0]
        if f_cur < f_min:
            f_min = f_cur
            x_best = x
    
    x = x_best
    f = np.dot(np.dot(x.reshape(1, -1), L), x.reshape(-1, 1))[0][0]
    if options['disp']:
        print('Function value: {}\n'.format(f)+
              'Iterations number: {}'.format(i))
    return x, f
