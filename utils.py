import numpy as np


def mid(x1, x2, x3):
    """Componentwise operation that supplies
       the median of three arguments.
    """
    return x1 + x2 + x3 - max([x1, x2, x3]) - min([x1, x2, x3])

def unnormalized_Laplacian(W):
    return np.diag(np.sum(W, axis=1)) - W
