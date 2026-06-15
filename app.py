import streamlit as st
import engram 
from sentence_transformers import SentenceTransformer
import ollama
import json
import os

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
INDEX_FILE = "data/book.bin"
META_FILE = "data/book_meta.json"

@st.cache_resource
def load_resources():
    print("Loading models and index...")
    
    # 1. Load the local embedding model
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return embed_model, None, None

    # 2. Load the custom C++ AVX2 engine
    vector_db = engram.VectorIndex(EMBEDDING_DIM)
    vector_db.load_index(INDEX_FILE)
    
    # 3. Load the document metadata
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return embed_model, vector_db, metadata

# --- UI Setup ---
st.set_page_config(page_title="Engram RAG Chat", page_icon="🧠", layout="wide")

st.title("🧠 Engram: Chat with your PDF")
st.caption("Powered by C++ Custom Vector Engine & Local Llama 3")

with st.sidebar:
    st.header("System Status")
    
    st.divider()
    embed_model, vector_db, metadata = load_resources()
    
    if vector_db:
        st.success(f"Engram Loaded: {vector_db.count} vectors")
        st.info("LLM: Ollama (Llama 3 Local)")
    else:
        st.error("Index not found! Run ingestion pipeline first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Loop ---
if prompt := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not vector_db:
            message_placeholder.error("Error: Database not loaded.")
        else:
            try:
                # 1. Vectorize the user prompt
                query_vec = embed_model.encode(prompt).tolist()
                
                # 2. Fetch top 3 results using optimized C++ max-heap search
                results = vector_db.search(query_vec, k=3)
                
                context_texts = []
                for res_id, dist in results:
                    text_chunk = metadata.get(str(res_id), "Unknown Document Chunk")
                    context_texts.append(f"- {text_chunk}")
                
                context_block = "\n".join(context_texts)
                
                # 3. Build the prompt for the local LLM
                full_prompt = (
                    f"You are a helpful assistant. Use the following context to answer the user's question.\n"
                    f"If the answer is not in the context, say you don't know.\n\n"
                    f"Context from document:\n{context_block}\n\n"
                    f"User Question: {prompt}"
                )
                
                # 4. Generate the response using Ollama
                with st.spinner("Engram is retrieving data & Local Llama3 is thinking..."):
                    response = ollama.chat(model='llama3', messages=[
                        {
                            'role': 'user',
                            'content': full_prompt
                        }
                    ])
                    answer = response['message']['content']
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Display the source context
                with st.expander("View Retrieved Context (Source)"):
                    for t in context_texts:
                        st.text(t)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")