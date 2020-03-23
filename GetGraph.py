@njit
def get_adjacency_matrix(img):
    """
    Compute adjacency matrix for given image with linear kernel.
    njit for accelerating speed
    
    Parameters:
    --------
    img : (h, w, ...) array_like
        Input image for computing
    
    Returns:
    --------
    A : (h*w, h*w) array_like
        Adjacency matrix of img
    
    """
    N = img.shape[0]*img.shape[1]
    X = img.reshape((N, -1))
    S = X@X.T
    b = np.diag(S)
    A = b.reshape(1, -1) - 2*S + b.reshape(-1, 1) #||x_i-x_j||^2=(x_i, x_i)-2*(x_i, x_j)+(x_j, x_j)
    return A

def coord_to_flatten_idx(y, x, h=h, w=w):
    """
    By coordinate of pixel of original image return single index of flatten array
    """
    assert y < h and x < w, 'Wrond indexes'
    return y*w+x

def flatten_idx_to_coors(idx, h=h, w=w):
    """
    By single index of flatten array return coordinate of pixel of original image
    """
    assert idx < h*w, 'Wrong index'
    return (idx//h, idx%w)