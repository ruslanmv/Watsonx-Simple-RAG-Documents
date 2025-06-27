import json
from step1_extract_chunks import chunks
from step2_generate_embeddings_ibm import all_embeddings

# Check if chunks and embeddings have the same length
if len(chunks) != len(all_embeddings):
    raise ValueError("The number of chunks and embeddings must be the same.")

# Combine chunks and their embeddings
data = []
for i in range(len(chunks)):
    data.append({
        "text": chunks[i],
        "embedding": all_embeddings[i]
    })

# Save to JSON file
with open("vector_store.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Embeddings stored in vector_store.json")
