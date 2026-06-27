import streamlit as st
import pymupdf4llm
from llama_cpp import Llama

st.title("DocFlow: Lightweight Document QA")

# 1. Setup Local LLM (Update this path to your actual GGUF file)
@st.cache_resource
def load_llm():
    return Llama(
        model_path="models/model.gguf",
        n_ctx=512,           # Keep the context small to save memory
        n_gpu_layers=0,      # Disable GPU offloading to keep RAM usage predictable
        use_mmap=True,
        use_mlock=False,
        verbose=False        # Suppress the massive log output that eats up terminal memory
    )
llm = load_llm()

# 2. Extract Document Content
def process_pdf(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc_content = process_pdf("temp.pdf")

    query = st.text_input("Ask a question about the document:")

    if query:
        # Prompt Engineering for Structured QA
        prompt = f"""
        You are an expert assistant. Use the following document content to answer the question.

        Document:
        {doc_content[:15000]} # Truncate to fit context window

        Question: {query}

        Answer:"""

        with st.spinner("DocFlow is thinking..."):
            response = llm(prompt, max_tokens=500)
            st.write(response['choices'][0]['text'])
