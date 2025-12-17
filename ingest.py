import os
import glob
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()

# --- CONFIG ---
ROOT_DIR = Path("data_store")
PDF_PATH = ROOT_DIR / "pdf_files"

DB_PATH = os.getenv("VECTOR_DB", "./vector_storage")
DB_COLLECTION = "knowledge_store"

# Ensure folders exist
ROOT_DIR.mkdir(exist_ok=True)
PDF_PATH.mkdir(parents=True, exist_ok=True)


# --- PDF EXTRACTOR ---
def read_pdf(file_path: Path) -> list[str]:
    """Reads PDF and extracts cleaned text chunks."""
    chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    page_chunks = chunk_text(text)
                    chunks.extend(page_chunks)

    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")

    return chunks


# --- TEXT CHUNKING ---
def chunk_text(text: str, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks suitable for embeddings."""
    if not text:
        return []

    chunks = []
    start = 0
    text = text.replace("\n", " ").strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # slide window with overlap
        if start < 0:
            start = 0

    return chunks


# --- MAIN INGESTION ---
def run_ingestion(reset=False):
    print("\n🚀 Starting ingestion...\n")

    print("📌 Loading embedding model (MiniLM)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.\n")

    all_chunks = []
    metas = []
    ids = []

    # Read PDFs from folder
    pdf_files = glob.glob(str(PDF_PATH / "*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in data_store/pdf_files")
        return

    for pdf_file in pdf_files:
        print(f"📄 Reading PDF: {pdf_file}")
        chunks = read_pdf(Path(pdf_file))

        if not chunks:
            print(f"⚠️ No readable text found in: {pdf_file}")
            continue

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metas.append({
                "source": os.path.basename(pdf_file),
                "type": "pdf",
                "chunk_id": idx
            })
            ids.append(str(uuid.uuid4()))

    if not all_chunks:
        print("❌ No chunks created. Aborting ingestion.")
        return

    print(f"\n📦 Total text chunks created: {len(all_chunks)}\n")

    print("⚙️ Generating embeddings...")
    vectors = embedder.encode(all_chunks, show_progress_bar=True).tolist()
    print("✅ Embeddings created.\n")

    # Store in ChromaDB
    print(f"🗄️ Saving to ChromaDB at: {DB_PATH}")

    client = chromadb.PersistentClient(path=DB_PATH)

    if reset:
        try:
            client.delete_collection(DB_COLLECTION)
            print("🗑 Old collection deleted.\n")
        except Exception:
            pass

    collection = client.get_or_create_collection(DB_COLLECTION)

    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=metas,
        embeddings=vectors
    )

    print("🎉 Ingestion complete!")

    with open("vector_summary.json", "w") as f:
        json.dump({
            "chunks": len(all_chunks),
            "collection": DB_COLLECTION
        }, f, indent=2)

    print("📄 Summary saved to vector_summary.json\n")


# --- RUN SCRIPT ---
if __name__ == "__main__":
    run_ingestion(reset=False)
