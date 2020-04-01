import numpy as np
import cv2


def mid(x1, x2, x3):
    """Componentwise operation that supplies
       the median of three arguments.
    """
    return x1 + x2 + x3 - max([x1, x2, x3]) - min([x1, x2, x3])

def prior_label_constraints(segments, foreground_pixels, background_pixels):
    l = -np.ones((len(np.unique(segments)), ))
    u = np.ones((len(np.unique(segments)), ))
    
    for pixel in foreground_pixels:
        l[segments[pixel[0], pixel[1]]] = 1
    for pixel in background_pixels:
        u[segments[pixel[0], pixel[1]]] = -1
    
    return l, u

def proj(x, l, u):
    for i in range(len(x)):
        if abs(x[i] - l[i]) < abs(x[i] - u[i]):
            x[i] = l[i]
        else:
            x[i] = u[i]
    return x

def find_bounds(segments):
    """Finding boundary pixels of superpixels."""
    bounds = np.full_like(segments, -1)
    for x in range(1, segments.shape[0] - 1):
        for y in range(1, segments.shape[1] - 1):
            current = segments[x, y]

            left = segments[x - 1, y]
            right = segments[x + 1, y]
            up = segments[x, y + 1]
            down = segments[x, y - 1]

            if (left != current) or (right != current) or (up != current) or (down != current):
                bounds[x, y] = current
    
    return bounds

def find_adjacency(segments):
    """Finding neighbors for each superpixel."""
    adjacency = {}
    def add_node(d, key, value):
        if d.get(key) == None:
            d[key] = set()
        d[key].add(value)
    
    for x in range(1, segments.shape[0] - 1):
        for y in range(1, segments.shape[1] - 1):
            current = segments[x, y]

            left = segments[x - 1, y]
            right = segments[x + 1, y]
            up = segments[x, y + 1]
            down = segments[x, y - 1]

            if left != current:
                add_node(adjacency, current, left)
            if right != current:
                add_node(adjacency, current, right)
            if up != current:
                add_node(adjacency, current, up)
            if down != current:
                add_node(adjacency, current, down)
    
    n = len(np.unique(segments))
    adjacency_matrix = np.zeros((n, n))
    for key in adjacency.keys():
        for value in adjacency[key]:
            adjacency_matrix[key, value] = 1
        
    return adjacency_matrix

def superpixels_vectors(image, segments):
    """Сonstructing a matrix composed of
       5-dimensional pixel vectors [r, g, b, x ,y].
    """
    bounds = find_bounds(segments)
    segments_indices = np.unique(segments)
    
    X = np.zeros((len(segments_indices), 5))
    for index in segments_indices:
        coordinates = np.array(np.nonzero(bounds==index))

        polygon = np.array([[x, y] for x, y in zip(coordinates[0], coordinates[1])])
        center = (np.sum(polygon, axis=0)/polygon.shape[0]).astype(int)

        color = np.sum(np.array([image[x, y] for x, y in zip(coordinates[0], coordinates[1])]), axis=0)/coordinates.shape[1]

        X[index, :] = np.hstack((color.reshape(1, -1), center.reshape(1, -1)/image.shape[:2])).ravel()
        
    return X
    
def dist(x, y):
    """Distance function."""
    G = np.eye(5)
    return np.dot(np.dot((x - y).reshape(1, -1), G), (x - y).reshape(-1, 1))

def unnormalized_Laplacian(W):
    """Unnormalized Laplacian matrix сonstruction
       by a graph weight matrix W.
    """
    return np.diag(np.sum(W, axis=1)) - W

def Laplacian(image, segments, normalization='unnormalized'):
    """Laplacian matrix for graph based on image superpixels."""
    X = superpixels_vectors(image, segments)
    
    n = X.shape[0]
    W = find_adjacency(segments)
    for i in range(0, n):
        for j in range(n):
            if j < i:
                W[i, j] = W[j, i]
            elif (j > i) and (W[i, j] == 1):
                W[i, j] = 1/dist(X[i, :], X[j, :])
    
    if normalization == 'unnormalized':
        L = unnormalized_Laplacian(W)
    
    return L

def get_binary_mask(segments, x_opt):
    """Binary mask: is there an foreground (1) or not (0)
       according to the optimal solution of optimization problem.
    """
    segments_indices = np.unique(segments)
    mask = np.zeros_like(segments)
    for index in segments_indices:
        mask[segments==index] = x_opt[index]
    
    mask[mask != 1] = 0
    return mask

def mask_overlay(image, binary_mask, color=[1, 0, 0], alpha=0.3):
    """Mask overlay with color=[r, g, b] and transparency alpha on the image."""
    overlay = image.copy()
    result = image.copy()
    
    for i, c in enumerate(color):
        overlay[:, :, i] = np.logical_not(binary_mask)*overlay[:, :, i] + binary_mask*c
    
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    return result
