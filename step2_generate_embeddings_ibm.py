import requests
from ibm_auth import get_ibm_access_token
from step1_extract_chunks import chunks  # Assuming this now produces token-limited chunks
import os
from dotenv import load_dotenv


MODEL_ID = "ibm/slate-30m-english-rtrvr"  # Using a supported model

# ─── Load and validate environment variables ───────────────────────────────────
load_dotenv()
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")



# Get token
token = get_ibm_access_token(API_KEY)

def generate_ibm_embedding(text, token):
    #url = f"https://{REGION}.ml.cloud.ibm.com/ml/v1/text/embeddings?version=2024-05-01"  # Corrected URL
    url = f"{WATSONX_URL}/ml/v1/text/embeddings?version=2024-05-01"  # Using the environment variable
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": [text],  # Corrected to 'inputs' (plural)
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        # THIS IS THE CRUCIAL LINE TO ADD/CHECK
        full_response_json = response.json()
        print("--- Full API Response (for debugging) ---")
        print(full_response_json)
        print("-----------------------------------------")

        # Check for 'results' and 'embedding'
        if "results" in full_response_json and isinstance(full_response_json["results"], list) and len(full_response_json["results"]) > 0:
            # Access the 'embedding' key from the first dictionary in the 'results' list
            embedding = full_response_json["results"][0].get("embedding")
            if embedding is not None:
                return embedding
            else:
                raise Exception("❌ 'embedding' key not found in the first result object.")
        else:
            raise Exception("❌ 'results' key or expected structure not found in the response.")
    else:
        raise Exception("❌ Failed to get embedding: " + response.text)

# Loop over chunks and embed
all_embeddings = []  # FIXED: Initialized as an empty list

print("🔄 Generating embeddings for chunks...")
for idx, chunk in enumerate(chunks):
    print(f"📄 Chunk {idx + 1}/{len(chunks)}")
    try:
        emb = generate_ibm_embedding(chunk, token)
        all_embeddings.append(emb)
    except Exception as e:
        print(f"Error processing chunk {idx + 1}: {e}")
        # You might want to break or continue depending on how you want to handle errors
        break  # Breaking for now to see the first error clearly

print("✅ All embeddings generated.")
