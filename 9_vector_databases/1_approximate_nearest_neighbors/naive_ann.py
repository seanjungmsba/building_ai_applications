import numpy as np

class NaiveANN:
    """
    A simple Approximate Nearest Neighbor (ANN) search using random sampling.
    Useful for educational or small-scale purposes.
    """

    def __init__(self, data: np.ndarray, sample_ratio: float = 0.1):
        """
        Initialize the ANN searcher.

        Args:
            data (np.ndarray): Dataset of shape (n_samples, n_features).
            sample_ratio (float): Fraction of the dataset to sample during each query.
        """
        self.data = data
        self.sample_ratio = sample_ratio
        self.n_samples, self.n_features = data.shape

    def search(self, query: np.ndarray, k: int = 1):
        """
        Search for k approximate nearest neighbors of the query vector.

        Args:
            query (np.ndarray): Query vector of shape (n_features,).
            k (int): Number of neighbors to return.

        Returns:
            indices (np.ndarray): Indices of the k nearest neighbors.
            distances (np.ndarray): Corresponding distances.
        """
        # Sample a subset of the dataset
        sample_size = max(1, int(self.n_samples * self.sample_ratio))
        sampled_indices = np.random.choice(self.n_samples, sample_size, replace=False)
        sampled_data = self.data[sampled_indices]

        # Compute distances to the sampled points
        distances = np.linalg.norm(sampled_data - query, axis=1)

        # Find the indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Map sampled indices back to original dataset
        return sampled_indices[nearest_indices], distances[nearest_indices]


def main():
    np.random.seed(42)

    # Generate random dataset: 10000 points in 128D
    num_points = 10000
    dimensions = 128
    data = np.random.rand(num_points, dimensions)

    # Create an instance of NaiveANN
    ann = NaiveANN(data=data, sample_ratio=0.05)

    # Create a random query point
    query = np.random.rand(dimensions)

    # Perform ANN search
    k = 5
    indices, distances = ann.search(query, k=k)

    print(f"Query vector: {query[:5]}... (truncated)")
    '''
    Query vector: [0.18171448 0.34181557 0.6398858  0.29247298 0.44219118]... (truncated)
    '''

    print(f"\nTop {k} approximate nearest neighbors:")
    for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"  Rank {rank}: Index={idx}, Distance={dist:.4f}")
    
    '''
    Top 5 approximate nearest neighbors:
        Rank 1: Index=82, Distance=3.8351
        Rank 2: Index=6267, Distance=3.9009
        Rank 3: Index=4436, Distance=3.9216
        Rank 4: Index=4195, Distance=3.9483
        Rank 5: Index=2347, Distance=3.9800
    '''

if __name__ == "__main__":
    main()
