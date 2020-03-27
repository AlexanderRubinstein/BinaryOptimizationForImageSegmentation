import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def SDP_relaxation(L, options={'maxiter': 100, 'disp': False}, x0=None, full_output=False):
    """ Binary optimization with SDP relaxation."""
    def objective(L):
        def fun(x):
            X = np.dot(x.reshape(-1, 1),
                       x.reshape(1, -1))
            return np.trace(np.dot(L.T, X))
        return fun
    
    def constraint(x):
        diag = []
        diag.append(x[0])
        for i in range(1, len(x) - 1):
            diag.append(x[i]**2)
        diag.append(-x[-1])
        return diag
    
    n = L.shape[0]
    fun = objective(L)
    if x0 is None:
    	x0 = np.random.choice([-1, 1], (1, n)).ravel()
    res = minimize(fun, x0, method='SLSQP',
                   constraints=[NonlinearConstraint(constraint,
                                                    np.ones((n,)),
                                                    np.ones((n,)))],
                   options=options)
    if full_output:
    	return res
    else:
    	return res.x
