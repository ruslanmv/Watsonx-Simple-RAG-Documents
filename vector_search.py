import json
import numpy as np
from step2_generate_embeddings_ibm import generate_ibm_embedding, token

# Load stored data
with open("vector_store.json", "r", encoding="utf-8") as f:
    store = json.load(f)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0  # Handle zero vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_chunks(query, top_k=3):
    query_emb = generate_ibm_embedding(query, token)
    
    # Ensure the query embedding is a valid numpy array
    if not isinstance(query_emb, np.ndarray):
        raise ValueError("The generated embedding for the query is not a valid numpy array.")
    
    similarities = [
        (cosine_similarity(query_emb, item["embedding"]), item["text"])
        for item in store
    ]
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

# Example usage
# results = search_similar_chunks("your query here")
# print(results)
