import numpy as np

def my_pca(X, n_components):
    """
    Perform PCA on the data X.  
    Returns the projected data, principal components and singular values.
    """
    
    # Center the data to avoid bias
    X = X - np.mean(X, axis=0)

    # Compute SVD
    U, S, V = np.linalg.svd(X, full_matrices=False)

    # Take the first n_components columns of V
    V = V[:n_components]

    # Project the data
    X_proj = X @ V.T

    return X_proj, V, S[:n_components]