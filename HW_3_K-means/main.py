import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from time import time

# Function to generate (convex and non-convex) data
def generate_data():
    # Convex
    X_convex, y_convex = make_blobs(n_samples=500, centers=3, random_state=42)
    
    # Non-convex 
    X_nonconvex, y_nonconvex = make_moons(n_samples=500, noise=0.05, random_state=42)
    
    return X_convex, y_convex, X_nonconvex, y_nonconvex

# Function to plot clusters and its' centers
def plot_clusters(X, y_kmeans, centroids, title):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.show()

# Function to use K-means and plot clusters
def run_kmeans(X, n_clusters, title):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    plot_clusters(X, y_kmeans, kmeans.cluster_centers_, title)

# Function to test different parameters of K-means
def test_kmeans_params(X, n_clusters, init, max_iter, n_init):
    start_time = time()
    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    end_time = time()
    elapsed_time = end_time - start_time
    
    # Plot results
    plot_clusters(X, y_kmeans, kmeans.cluster_centers_, 
                  f"n_clusters={n_clusters}, init='{init}', max_iter={max_iter}, n_init={n_init}")
    print(f"Execution Time: {elapsed_time:.4f} seconds")

# Function to track and plot inertia over iterations
def track_inertia(X, n_clusters, max_iter):
    inertia_values = []

    # Run K-means for different iterations to track inertia on each iteration
    for i in range(1, max_iter + 1):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=i, n_init=1, random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
        
        # Check for convergence
        if len(inertia_values) > 1 and abs(inertia_values[-2] - inertia_values[-1]) < 1e-4:
            break  # Stop if inertia change is very small
    
    plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Inertia (Within-cluster Sum of Squares)")
    plt.title("Inertia Change Over Iterations")
    plt.show()

# Main function to demonstrate all steps
def main():
    X_convex, y_convex, X_nonconvex, y_nonconvex = generate_data()
    
    # Try K-means on convex and non-convex data
    print("Convex Clusters:")
    run_kmeans(X_convex, n_clusters=3, title="Convex Clusters")
    
    print("Non-Convex Clusters:")
    run_kmeans(X_nonconvex, n_clusters=2, title="Non-Convex Clusters")
    
    # Try different parameters
    print("\nTesting different parameters on convex clusters:")
    test_kmeans_params(X_convex, n_clusters=3, init='k-means++', max_iter=100, n_init=10)
    test_kmeans_params(X_convex, n_clusters=3, init='random', max_iter=300, n_init=10)
    test_kmeans_params(X_convex, n_clusters=4, init='k-means++', max_iter=100, n_init=20)
    
    # Plot inertia over iterations for convex data
    print("\nTracking inertia over iterations for convex clusters:")
    track_inertia(X_convex, n_clusters=3, max_iter=20)

    # Plot inertia over iterations for non-convex data
    print("\nTracking inertia over iterations for non-convex clusters:")
    track_inertia(X_nonconvex, n_clusters=2, max_iter=20)

if __name__ == "__main__":
    main()

"""
Output:

Convex Clusters:
Non-Convex Clusters:

Testing different parameters on convex clusters:
Execution Time: 0.0649 seconds
Execution Time: 0.0590 seconds
Execution Time: 0.1171 seconds

Tracking inertia over iterations for convex clusters:

Tracking inertia over iterations for non-convex clusters:
"""