import streamlit as st
import numpy as np
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Use rag.py for RAG logic
import rag

# --- INITIALIZATION ---
load_dotenv()

# Ensure fordham_logo.png is in your directory
FORDHAM_ICON_PATH = "fordham_logo.png" 
st.set_page_config(page_title="Fordham AI Assistant", page_icon=FORDHAM_ICON_PATH)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- LOAD NOTEBOOK DATA ---
@st.cache_resource
def load_rag_data():
    local_embs = np.load('scripts/openai_embs_1.npy', allow_pickle=True).item() 
    with open('scripts/chunks_1.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return local_embs, chunks

local_embs, chunks = load_rag_data()

# --- SESSION STATE INITIALIZATION ---
# Store the chat history so it persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RAG PIPELINE ---
def rag_pipeline(question, alpha, k):
    retrieved_chunks = rag.retrieve_relevant_chunks_hybrid(
        question=question,
        local_embs=local_embs,
        chunks=chunks,
        k=k,
        alpha=alpha
    )
    answer = rag.generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=512
    )
    # Additionally return retrieved_chunks for source viewing
    return answer, retrieved_chunks

# --- UI ELEMENTS ---
st.image(FORDHAM_ICON_PATH, width=80)
st.title("Fordham AI Assistant")
st.markdown("Ask anything about Fordham University (Admissions, Financial Aid, etc.)")

with st.sidebar:
    st.header("Search Settings")
    alpha = st.slider("Hybrid Alpha (0=Semantic, 1=Lexical)", 0.0, 1.0, 0.5)
    k_val = st.number_input("Chunks to retrieve", 1, 20, 5)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- DISPLAY CHAT HISTORY ---
# Re-render all previous messages and their corresponding buttons
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Only show verify and view sources buttons for assistant messages
        if msg["role"] == "assistant":
            cols = st.columns(2)
            # --- VERIFY ACCURACY BUTTON ---
            if cols[0].button("🔍 Verify Accuracy", key=f"verify_{i}"):
                with st.spinner("Checking live web data..."):
                    try:
                        config = types.GenerateContentConfig(
                            tools=[types.Tool(google_search=types.GoogleSearch())]
                        )
                        check_prompt = f"Verify this RAG answer about Fordham: {msg['content']}"
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[check_prompt],
                            config=config
                        )
                        # Display the verification report
                        st.info(f"**Verification Report:**\n\n{response.text}")
                    except Exception as e:
                        st.error(f"Verification failed: {e}")

            # --- VIEW SOURCES BUTTON ---
            if 'retrieved_chunks' in msg:
                if cols[1].button("🔗 View Sources", key=f"view_sources_{i}"):
                    st.markdown("**Retrieved Source Chunks:**")
                    for j, chunk in enumerate(msg['retrieved_chunks']):
                        # If chunk is a dict, use .get; otherwise, treat as string
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get("text") or chunk.get("content") or str(chunk)
                            source = chunk.get("source", None)
                        else:
                            chunk_text = str(chunk)
                            source = None
                        # If the chunk has a "source" field, make it a link if it's a URL
                        if source and (isinstance(source, str)) and (source.startswith("http://") or source.startswith("https://")):
                            st.markdown(f'[{source}]({source})', unsafe_allow_html=True)
                        st.markdown(f"> {chunk_text}")

# --- NEW CHAT INPUT ---
if prompt := st.chat_input("Ask me anything about Fordham!"):
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate, display, and store assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching Fordham records..."):
            answer, retrieved_chunks = rag_pipeline(prompt, alpha, k_val)
            st.markdown(answer)
            # Store the retrieved_chunks in the message for later "View Sources"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "retrieved_chunks": retrieved_chunks
            })
            # Rerun so the buttons appear immediately under the new message
            st.rerun()