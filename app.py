import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

# ---------------- LOAD ENV ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ---------------- PDF PROCESSING ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# ---------------- GEMINI QUERY WITH RETRY ----------------
def ask_gemini_with_retry(question, context, retries=3, delay=5):
    for i in range(retries):
        try:
            response = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(
                f"Answer the question based ONLY on the context below.\n\nContext: {context}\nQuestion: {question}"
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay)
            else:
                return f"❌ Error: {e}"
    return "❌ Error: Too many requests. Please try again later."

# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")
    st.title("📚 Chat with Multiple PDFs (Gemini 2.0 + RAG)")

    # Sidebar: API Key
    st.sidebar.header("⚙️ Settings")
    api_key_input = st.sidebar.text_input("Google API Key:", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        genai.configure(api_key=api_key_input)

    # Upload PDFs
    pdf_docs = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)

    if pdf_docs and st.button("🚀 Process PDFs"):
        with st.spinner("🔄 Processing PDFs..."):
            text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(text)
            build_vector_store(chunks)
            st.session_state.pdf_text = text
            st.session_state.pdf_docs = pdf_docs
            st.session_state.pdf_processed = True
            st.success("✅ PDFs processed!")

    # Initialize chat history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Chat interface
    if st.session_state.get("pdf_processed", False):
        st.subheader("💬 Ask Questions")

        # Show last 5 messages
        for chat in st.session_state.conversation_history[-5:]:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        # User input
        user_question = st.chat_input("Ask a question about your PDFs...")
        if user_question:
            # Load FAISS with safe pickle option
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True  # ✅ fix for pickle loading
            )
            # Retrieve top 5 relevant chunks
            docs = db.similarity_search(user_question, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            # Ask Gemini with retry
            answer = ask_gemini_with_retry(user_question, context)

            # Save to history
            st.session_state.conversation_history.append({
                "question": user_question,
                "answer": answer,
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "pdfs": ", ".join([pdf.name for pdf in st.session_state.pdf_docs])
            })

            # Display chat
            with st.chat_message("user"):
                st.write(user_question)
            with st.chat_message("assistant"):
                st.write(answer)

            # CSV download
            df = pd.DataFrame(st.session_state.conversation_history)
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="📥 Download Chat History",
                data=csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    else:
        st.info("👆 Upload PDFs and click 'Process PDFs' to start chatting!")

if __name__ == "__main__":
    main()                                     