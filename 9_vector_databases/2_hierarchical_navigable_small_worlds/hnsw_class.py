
import heapq
import numpy as np

class Node:
    """Represents a data point in the HNSW graph."""

    def __init__(self, vector: np.ndarray, id: int):
        """
        Initialize a node with a vector and unique ID.

        Args:
            vector (np.ndarray): The high-dimensional vector representing the node.
            id (int): Unique identifier for the node.
        """
        self.vector = vector
        self.id = id
        self.neighbors = [[] for _ in range(5)]  # List of neighbors per layer.

    def add_neighbor(self, neighbor, layer: int):
        """
        Add a neighbor node to a specific layer.

        Args:
            neighbor (Node): Node to connect to.
            layer (int): Layer index where the connection is made.
        """
        self.neighbors[layer].append(neighbor)

    def distance(self, other_node):
        """
        Compute the Euclidean distance to another node.

        Args:
            other_node (Node): The other node to compare to.

        Returns:
            float: Euclidean distance between self and other_node.
        """
        return np.linalg.norm(self.vector - other_node.vector)


class HNSW:
    """Hierarchical Navigable Small Worlds (HNSW) algorithm for approximate nearest neighbor search."""

    def __init__(self, max_layers=5, M=5, ef_construction=100):
        """
        Initialize the HNSW index.

        Args:
            max_layers (int): Maximum number of graph layers.
            M (int): Maximum number of connections per node per layer.
            ef_construction (int): Size of candidate list for neighbor selection during insertion.
        """
        self.max_layers = max_layers
        self.M = M
        self.ef_construction = ef_construction
        self.layers = [[] for _ in range(max_layers)]  # Layered node storage.
        self.entry_point = None  # Entry point for search traversal.

    def add_node(self, vector, id):
        """
        Add a new vector into the HNSW index.

        Args:
            vector (np.ndarray): The vector to index.
            id (int): Unique identifier for the node.
        """
        node = Node(vector, id)
        node.neighbors = [[] for _ in range(self.max_layers)]

        if self.entry_point is None:
            # First node becomes entry point in all layers.
            self.entry_point = node
            for layer in range(self.max_layers):
                self.layers[layer].append(node)
            return

        # Choose the top layer for the new node using a probabilistic model.
        max_layer = self._choose_layer()

        # Start from the entry point and traverse down.
        curr_node = self.entry_point
        curr_dist = node.distance(curr_node)

        for layer in range(self.max_layers - 1, -1, -1):
            if layer > max_layer:
                # Only traverse higher layers to get closer to insertion region.
                changed = True
                while changed:
                    changed = False
                    for neighbor in curr_node.neighbors[layer]:
                        dist = node.distance(neighbor)
                        if dist < curr_dist:
                            curr_dist = dist
                            curr_node = neighbor
                            changed = True
            else:
                # At eligible layer, insert node and connect to nearest neighbors.
                self._insert_into_layer(node, curr_node, layer)
                self.layers[layer].append(node)

    def _choose_layer(self):
        """
        Randomly choose the highest layer for a new node using exponential distribution.

        Returns:
            int: Chosen layer index.
        """
        return min(self.max_layers - 1, int(-np.log(np.random.uniform()) * self.max_layers / np.log(self.M)))

    def _insert_into_layer(self, node, entry_point, layer):
        """
        Insert a node into a specific graph layer.

        Args:
            node (Node): Node to insert.
            entry_point (Node): Starting point for search.
            layer (int): Layer to insert into.
        """
        candidates = []
        visited = set()
        nearest = []

        # Initialize candidate list with the entry point.
        heapq.heappush(candidates, (node.distance(entry_point), entry_point))
        visited.add(entry_point.id)

        while candidates:
            dist, candidate = heapq.heappop(candidates)

            if nearest and dist > nearest[0][0] and len(nearest) >= self.ef_construction:
                break  # Stop when better neighbors unlikely.

            # Add candidate to nearest if list not full or better than worst.
            if len(nearest) < self.ef_construction:
                heapq.heappush(nearest, (-dist, candidate))
            elif -dist > nearest[0][0]:
                heapq.heappushpop(nearest, (-dist, candidate))

            # Explore neighbors to extend candidate list.
            for neighbor in candidate.neighbors[layer]:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    dist = node.distance(neighbor)
                    heapq.heappush(candidates, (dist, neighbor))

        # Connect to top M nearest neighbors.
        nearest_nodes = [n for _, n in sorted([(-d, n) for d, n in nearest])]
        connections = nearest_nodes[:self.M]

        for conn in connections:
            node.add_neighbor(conn, layer)
            if len(conn.neighbors[layer]) < self.M:
                conn.add_neighbor(node, layer)
            else:
                # Replace farthest neighbor if new one is closer.
                farthest = None
                max_dist = -1
                for neighbor in conn.neighbors[layer]:
                    dist = conn.distance(neighbor)
                    if dist > max_dist:
                        max_dist = dist
                        farthest = neighbor
                if conn.distance(node) < max_dist:
                    conn.neighbors[layer].remove(farthest)
                    conn.add_neighbor(node, layer)

    def search(self, query_vector, k=1, ef_search=50):
        """
        Perform an ANN search to retrieve top-k similar vectors.

        Args:
            query_vector (np.ndarray): Query vector.
            k (int): Number of nearest neighbors to return.
            ef_search (int): Size of the search candidate list.

        Returns:
            list: List of tuples (node_id, node_vector) of nearest neighbors.
        """
        query_node = Node(query_vector, id=-1)

        if self.entry_point is None:
            return []  # No data to search.

        # Traverse down from top to layer 1 (skip layer 0 for fast descent).
        curr_node = self.entry_point
        curr_dist = query_node.distance(curr_node)

        for layer in range(self.max_layers - 1, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in curr_node.neighbors[layer]:
                    dist = query_node.distance(neighbor)
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_node = neighbor
                        changed = True

        # Perform more thorough search in layer 0.
        candidates = []
        visited = set()
        nearest = []

        heapq.heappush(candidates, (curr_dist, curr_node))
        heapq.heappush(nearest, (-curr_dist, curr_node))
        visited.add(curr_node.id)

        while candidates:
            dist, candidate = heapq.heappop(candidates)

            if nearest and dist > -nearest[0][0] and len(nearest) >= ef_search:
                break

            for neighbor in candidate.neighbors[0]:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    dist = query_node.distance(neighbor)

                    if len(nearest) < ef_search or dist < -nearest[0][0]:
                        heapq.heappush(candidates, (dist, neighbor))
                        if len(nearest) < ef_search:
                            heapq.heappush(nearest, (-dist, neighbor))
                        else:
                            heapq.heappushpop(nearest, (-dist, neighbor))

        # Extract top-k nearest nodes.
        result = []
        nearest.sort()
        for i in range(min(k, len(nearest))):
            _, node = heapq.heappop(nearest)
            result.append((node.id, node.vector))

        return result
