
import numpy as np
from hnsw_class import HNSW

def main():
    np.random.seed(42)
    vector_dim = 10
    num_vectors = 1000
    query_count = 5
    k = 5

    print(f"Creating {num_vectors} random vectors with {vector_dim} dimensions...")
    vectors = np.random.rand(num_vectors, vector_dim)

    print("Initializing HNSW index...")
    hnsw_index = HNSW(max_layers=4, M=8, ef_construction=100)

    print("Adding vectors to the index...")
    for i, vector in enumerate(vectors):
        hnsw_index.add_node(vector, id=i)
        if (i + 1) % 100 == 0:
            print(f"  Added {i + 1}/{num_vectors} vectors")

    print(f"\nGenerating {query_count} random query vectors...")
    query_vectors = np.random.rand(query_count, vector_dim)

    print("Performing search for each query vector:")
    for i, query in enumerate(query_vectors):
        print(f"\nQuery {i+1}:")
        start_time = np.datetime64('now')
        results = hnsw_index.search(query, k=k, ef_search=50)
        end_time = np.datetime64('now')
        search_time = (end_time - start_time) / np.timedelta64(1, 'ms')
        print(f"  Search completed in {search_time:.2f} ms")
        print(f"  Top {len(results)} nearest neighbors:")
        for rank, (node_id, node_vector) in enumerate(results):
            distance = np.linalg.norm(query - node_vector)
            print(f"    Rank {rank+1}: ID={node_id}, Distance={distance:.4f}")

        if num_vectors <= 5000:
            print("  Verifying with brute force search...")
            distances = [np.linalg.norm(query - v) for v in vectors]
            true_nearest = np.argsort(distances)[:k]
            found_ids = [node_id for node_id, _ in results]
            recall = sum(1 for id in true_nearest if id in found_ids) / k
            print(f"  Recall@{k}: {recall:.2f}")
            print(f"  True nearest IDs: {list(true_nearest)}")
            print(f"  HNSW found IDs:   {found_ids}")

if __name__ == "__main__":
    main()

'''
Creating 1000 random vectors with 10 dimensions...
Initializing HNSW index...
Adding vectors to the index...
  Added 100/1000 vectors
  Added 200/1000 vectors
  Added 300/1000 vectors
  Added 400/1000 vectors
  Added 500/1000 vectors
  Added 600/1000 vectors
  Added 700/1000 vectors
  Added 800/1000 vectors
  Added 900/1000 vectors
  Added 1000/1000 vectors

Generating 5 random query vectors...
Performing search for each query vector:

Query 1:
  Search completed in 0.00 ms
  Top 5 nearest neighbors:
    Rank 1: ID=413, Distance=0.8652
    Rank 2: ID=498, Distance=0.8596
    Rank 3: ID=662, Distance=0.8581
    Rank 4: ID=381, Distance=0.8564
    Rank 5: ID=737, Distance=0.8509
  Verifying with brute force search...
  Recall@5: 0.00
  True nearest IDs: [915, 22, 831, 565, 496]
  HNSW found IDs:   [413, 498, 662, 381, 737]

Query 2:
  Search completed in 0.00 ms
  Top 5 nearest neighbors:
    Rank 1: ID=80, Distance=0.9014
    Rank 2: ID=746, Distance=0.9010
    Rank 3: ID=249, Distance=0.9006
    Rank 4: ID=884, Distance=0.8949
    Rank 5: ID=25, Distance=0.8915
  Verifying with brute force search...
  Recall@5: 0.00
  True nearest IDs: [752, 537, 628, 18, 718]
  HNSW found IDs:   [80, 746, 249, 884, 25]

Query 3:
  Search completed in 0.00 ms
  Top 5 nearest neighbors:
    Rank 1: ID=968, Distance=0.9235
    Rank 2: ID=99, Distance=0.9117
    Rank 3: ID=250, Distance=0.9064
    Rank 4: ID=78, Distance=0.9062
    Rank 5: ID=918, Distance=0.9048
  Verifying with brute force search...
  Recall@5: 0.00
  True nearest IDs: [95, 885, 630, 504, 822]
  HNSW found IDs:   [968, 99, 250, 78, 918]

Query 4:
  Search completed in 0.00 ms
  Top 5 nearest neighbors:
    Rank 1: ID=512, Distance=0.9457
    Rank 2: ID=783, Distance=0.9446
    Rank 3: ID=356, Distance=0.9411
    Rank 4: ID=495, Distance=0.9386
    Rank 5: ID=635, Distance=0.9363
  Verifying with brute force search...
  Recall@5: 0.00
  True nearest IDs: [979, 808, 765, 149, 140]
  HNSW found IDs:   [512, 783, 356, 495, 635]

Query 5:
  Search completed in 0.00 ms
  Top 5 nearest neighbors:
    Rank 1: ID=838, Distance=0.8700
    Rank 2: ID=63, Distance=0.8689
    Rank 3: ID=194, Distance=0.8628
    Rank 4: ID=787, Distance=0.8623
    Rank 5: ID=736, Distance=0.8517
  Verifying with brute force search...
  Recall@5: 0.00
  True nearest IDs: [445, 284, 235, 105, 582]
  HNSW found IDs:   [838, 63, 194, 787, 736]
'''
