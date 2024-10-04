from myPca import my_pca
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA as sk_pca
  
if __name__ == '__main__':
    # Load the Iris dataset
    wine = fetch_ucirepo(id=109)
    features_table = wine.data.features
    #print(wine)

    # Convert the data to numpy array
    X = features_table.values
    
    # Number of samples and featuresi
    n_samples, n_features = X.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(X[:5])

    # Apply custom pca
    my_X_proj, my_Eigenvec, my_Sval = my_pca(X, 2)

    # Apply sklearn pca
    skPca = sk_pca(n_components=2)
    sk_X_Proj = skPca.fit_transform(X)
    sk_Eigenvec = skPca.components_
    sk_Sval = skPca.singular_values_

    # Compare the results
    print("Custom PCA Eigenvec:")
    print(my_Eigenvec)

    print("Sklearn PCA Eigenvec:")
    print(sk_Eigenvec)

    print("Custom PCA Singular values:")
    print(my_Sval)

    print("Sklearn PCA Singular values:")
    print(sk_Sval)

    # Compare custom and sk components on a plot
    plt.figure()
    plt.scatter(my_X_proj[:, 0], my_X_proj[:, 1], label='My PCA')
    plt.scatter(sk_X_Proj[:, 0], sk_X_Proj[:, 1], label='Sklearn PCA')
    plt.legend()
    plt.show()

"""
OUTPUT:

Number of samples: 178
Number of features: 13
[[1.423e+01 1.710e+00 2.430e+00 1.560e+01 1.270e+02 2.800e+00 3.060e+00
  2.800e-01 2.290e+00 5.640e+00 1.040e+00 3.920e+00 1.065e+03]
 [1.320e+01 1.780e+00 2.140e+00 1.120e+01 1.000e+02 2.650e+00 2.760e+00
  2.600e-01 1.280e+00 4.380e+00 1.050e+00 3.400e+00 1.050e+03]
 [1.316e+01 2.360e+00 2.670e+00 1.860e+01 1.010e+02 2.800e+00 3.240e+00
  3.000e-01 2.810e+00 5.680e+00 1.030e+00 3.170e+00 1.185e+03]
 [1.437e+01 1.950e+00 2.500e+00 1.680e+01 1.130e+02 3.850e+00 3.490e+00
  2.400e-01 2.180e+00 7.800e+00 8.600e-01 3.450e+00 1.480e+03]
 [1.324e+01 2.590e+00 2.870e+00 2.100e+01 1.180e+02 2.800e+00 2.690e+00
  3.900e-01 1.820e+00 4.320e+00 1.040e+00 2.930e+00 7.350e+02]]

Custom PCA Eigenvec:
[[-1.65926472e-03  6.81015556e-04 -1.94905742e-04  4.67130058e-03
  -1.78680075e-02 -9.89829680e-04 -1.56728830e-03  1.23086662e-04
  -6.00607792e-04 -2.32714319e-03 -1.71380037e-04 -7.04931645e-04
  -9.99822937e-01]
 [-1.20340617e-03 -2.15498184e-03 -4.59369254e-03 -2.64503930e-02
  -9.99344186e-01 -8.77962152e-04  5.18507284e-05  1.35447892e-03
  -5.00440040e-03 -1.51003530e-02  7.62673115e-04  3.49536431e-03
   1.77738095e-02]]

Sklearn PCA Eigenvec:
[[ 1.65926472e-03 -6.81015556e-04  1.94905742e-04 -4.67130058e-03
   1.78680075e-02  9.89829680e-04  1.56728830e-03 -1.23086662e-04
   6.00607792e-04  2.32714319e-03  1.71380037e-04  7.04931645e-04
   9.99822937e-01]
 [ 1.20340617e-03  2.15498184e-03  4.59369254e-03  2.64503930e-02
   9.99344186e-01  8.77962152e-04 -5.18507284e-05 -1.35447892e-03
   5.00440040e-03  1.51003530e-02 -7.62673115e-04 -3.49536431e-03
  -1.77738095e-02]]

Custom PCA Singular values:
[4190.31224906  174.75337527]

Sklearn PCA Singular values:
[4190.31224906  174.75337527]
"""