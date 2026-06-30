import streamlit as st
import pymupdf4llm
import google.generativeai as genai

st.title("DocFlow: Document QA")

# Setup Gemini API
# Note: st.secrets looks for a key named GEMINI_API_KEY
# You will set this in the Streamlit Cloud Dashboard "Secrets"
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Gemini API key not found in secrets! Please set it in your App Settings.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# Extract Document Content
def process_pdf(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save temporarily to process
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc_content = process_pdf("temp.pdf")
    query = st.text_input("Ask a question about the document:")

    if query:
        # Prompt Engineering
        prompt = f"""
        You are an expert assistant. Use the following document content to answer the question.
        
        Document Content:
        {doc_content[:15000]}

        Question: {query}
        """

        with st.spinner("DocFlow is thinking..."):
            try:
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error communicating with Gemini: {e}")
