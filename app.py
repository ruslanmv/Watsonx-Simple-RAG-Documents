import os
import streamlit as st
import requests
from dotenv import load_dotenv
from step3_vector_search import search_similar_chunks
from ibm_auth import get_ibm_access_token

# ─── Load and validate environment variables ───────────────────────────────────
load_dotenv()
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

if not all([API_KEY, PROJECT_ID, WATSONX_URL]):
    st.error("❌ Missing one or more required environment variables: "
             "WATSONX_API_KEY, PROJECT_ID, WATSONX_URL")
    st.stop()

# ─── Fetch IBM WatsonX access token (cached) ────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_token():
    try:
        return get_ibm_access_token(API_KEY)
    except Exception as e:
        st.error(f"❌ Authentication failed: {e}")
        st.stop()

token = get_token()

# ─── Generate an answer with IBM watsonx (cached) ───────────────────────────────
@st.cache_data(show_spinner=False, max_entries=20)
def generate_answer_ibm(question: str, context: str) -> str:
    """
    Calls the WatsonX text generation endpoint with your question + context.
    """
    MODEL_ID = "ibm/granite-13b-instruct-v2"
    endpoint = f"{WATSONX_URL}/ml/v1/text/generation?version=2024-05-01"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "input": f"Context:\n{context}\n\nQuestion: {question}",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300
        }
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if resp.ok:
        return resp.json()["results"][0]["generated_text"]
    else:
        raise RuntimeError(f"Generation failed ({resp.status_code}): {resp.text}")

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="📘 AI Tutor", layout="centered")
st.title("📘 AI Tutor (powered by IBM watsonx)")
st.markdown("Ask a question about your document (vector store preloaded).")

question = st.text_input("❓ Your question:")
if st.button("🧠 Get Answer"):
    if not question.strip():
        st.warning("Please enter a non-empty question.")
    else:
        with st.spinner("🔍 Searching for relevant document chunks..."):
            try:
                results = search_similar_chunks(question)
            except Exception as e:
                st.error(f"⚠️ Vector search error: {e}")
                st.stop()

            if not results:
                st.error("No relevant information found in the document.")
            else:
                # Optional: preview top-3 chunks
                st.markdown("**Top relevant chunks:**")
                for i, (score, chunk) in enumerate(results[:3], 1):
                    st.markdown(f"{i}. (score: {score:.3f}) {chunk}")

                with st.spinner("🤖 Generating answer with IBM watsonx..."):
                    try:
                        answer = generate_answer_ibm(question, results[0][1])
                        st.success("✅ Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"⚠️ Generation error: {e}")
