import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from ibm_auth import get_ibm_access_token
import requests
import os
from dotenv import load_dotenv

# ─── Load and validate environment variables ───────────────────────────────────
load_dotenv()
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")



MODEL_ID = "ibm/slate-30m-english-rtrvr"

# Load vector store
with open("vector_store.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Get IBM access token
token = get_ibm_access_token(API_KEY)

# IBM embedding API
def embed_question(question):
    #url = f"https://{REGION}.ml.cloud.ibm.com/ml/v1/text/embeddings?version=2024-05-01"
    url = f"{WATSONX_URL}/ml/v1/text/embeddings?version=2024-05-01"  # Using the environment vari
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [question],
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID
    }
    res = requests.post(url, headers=headers, json=payload)
    return res.json()["results"][0]["embedding"]

# Find similar chunks
def search_similar_chunks(question, top_k=1):
    q_emb = embed_question(question)
    doc_embs = [item["embedding"] for item in data]
    scores = cosine_similarity([q_emb], doc_embs)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], data[i]["text"]) for i in top_indices]

'''
# === CLI Interface ===
query = input("❓ Ask your AI Tutor: ")
results = search_similar_chunks(query)
for score, chunk in results:
    print(f"\n🔍 Similarity: {score:.4f}")
    print(f"📘 Top Match:\n{chunk[:500]}...")  # Truncate for preview
    
'''
