# python -m venv myenv
# myenv/scripts/activate

import streamlit as st
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ✅ This MUST be the first Streamlit command
st.set_page_config(page_title="RAG LLM Q&A", page_icon="🧠")

st.title("🧠 Retrieval-Augmented Generation (RAG) with Wikipedia")
st.write("Ask questions on any topic using Wikipedia as an external knowledge source.")

# Load models with caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_qa_pipeline():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

embedding_model = load_embedding_model()
qa_pipeline = load_qa_pipeline()

# Function to fetch Wikipedia content
def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        st.warning("No Wikipedia page found for this topic.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        st.warning(f"Ambiguous topic. Please be more specific. Options: {e.options}")
        return None

# Split text into chunks
def split_text(text, tokenizer, chunk_size=256, chunk_overlap=20):
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

# UI: Input topic
topic = st.text_input("🔍 Enter a topic to retrieve knowledge from Wikipedia:", value="Apple Computers")

if topic:
    with st.spinner("Retrieving content from Wikipedia..."):
        document = get_wikipedia_content(topic)

    if document:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        chunks = split_text(document, tokenizer)

        st.success(f"Retrieved Wikipedia article and split into {len(chunks)} chunks.")

        with st.spinner("Generating embeddings..."):
            embeddings = embedding_model.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

        # UI: Input user question
        query = st.text_input("💬 Ask a question about this topic:")

        if query:
            query_embedding = embedding_model.encode([query])
            k = 3
            distances, indices = index.search(np.array(query_embedding), k)
            retrieved_chunks = [chunks[i] for i in indices[0]]

            st.subheader("📄 Retrieved Relevant Chunks:")
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk[:300]}...")

            # Answering
            with st.spinner("Generating answer using QA model..."):
                context = " ".join(retrieved_chunks)
                answer = qa_pipeline(question=query, context=context)

            st.subheader("✅ Answer:")
            st.markdown(f"**{answer['answer']}**")
