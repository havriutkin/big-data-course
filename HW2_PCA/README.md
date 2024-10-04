# PCA

I demonstrate the use of Principal Component Analysis (PCA) for finding principal component space and projecting data onto it.

## Project Structure

- **my_pca.py**: Contains a custom implementation of the PCA algorithm.
- **main.py**: Contains the script to load the dataset, apply the custom PCA, compare results with scikit-learn's PCA, and visualize the results.
- **requirements.txt**: Contains a list of dependencies needed to run the project.
- **README.md**: This file, explaining the project structure and methodology.

## Dataset

The project uses the **Wine Quality** dataset from the UCI Machine Learning Repository. This dataset has **178 samples** with **13 features**, making it suitable for PCA since it has a relatively large number of features compared to the number of samples.

## Implementation

### Custom PCA Implementation (`my_pca.py`)

The custom PCA algorithm follows these steps:

1. **Data Centering**: The dataset is centered by subtracting the mean of each feature. This ensures that the data has zero mean, so eigenvectors show true variance direction
   
2. **Singular Value Decomposition (SVD)**: We use SVD to decompose the centered data matrix into three matrices `U`, `S`, and `V`.
   
3. **Dimensionality Reduction**: We select the first `n_components` columns of the `V` matrix to form the projection matrix.

4. **Projection**: The original data is projected onto the new space formed by the selected principal components.

### Parameter Tuning
To find the optimal number of principal components, I tested different values of `n_components` and compared the results visually and numerically. The following criteria were used to determine the optimal number of components:


#### Final Parameters Chosen
After experimenting with different values for `n_components`, I found that setting `n_components = 2` provided a good balance between reducing dimensionality and retaining information in the data.

### Comparison with Scikit-Learn's PCA
The results of the custom PCA implementation were compared to the results obtained using scikit-learn's PCA. The output can be seen in `main.py`. The comparisons included:

- **Principal Component Vectors**: The principal components obtained by the custom implementation and scikit-learn were compared and found to be identical but inverse. It is probably caused by the fact, that scikit PCA first find covariance matrix, that have non-negative eigenvalues, since it is symmetrical and orthogonal. In my custom PCA, I just apply SVD to the data, which also effectively captures the variance. Despite the fact that eigenvectors are opposite, they still describe the same space. 
- **Singular Values**: The singular values obtained by both implementations were the same.

## Theoretical Considerations
PCA works by finding the directions (principal components) that maximize the variance in the data. These directions are orthogonal to each other, and each successive component explains a smaller amount of the total variance.


## Conclusion
PCA is an effective technique for dimensionality reduction, and my custom implementation has been verified against the scikit-learn implementation to ensure correctness.

