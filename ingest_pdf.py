import os
import json
import engram
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# הגדרות
PDF_FILE = "my_book.pdf"
INDEX_FILE = "data/book.bin"
META_FILE = "data/book_meta.json"
EMBEDDING_DIM = 384

def ingest():
    if not os.path.exists("data"):
        os.makedirs("data")

    reader = PdfReader(PDF_FILE)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    db = engram.VectorIndex(EMBEDDING_DIM)
    metadata = {}

    print(f"Ingesting {len(texts)} pages...")

    for i, text in enumerate(texts):
        vec = model.encode(text).tolist()
        db.add_vector(vec)
        metadata[str(i)] = text

    db.save_index(INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
        
    print("Ingestion complete! Database ready.")

if __name__ == "__main__":
    ingest()