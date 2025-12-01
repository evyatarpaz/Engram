import engram
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import json
import os

# Settings
PDF_PATH = "my_book.pdf"       # Put the path to a real PDF file you have
INDEX_FILE = "data/book.bin"   # Where to save the vector index
META_FILE = "data/book_meta.json" # Where to save the texts
CHUNK_SIZE = 100               # How many words per chunk (not too big or too small)

def main():
    PDF_PATH = input("Enter path to PDF file (default: my_book.pdf): ") or PDF_PATH
    # 1. Load the model
    print("🧠 Loading AI Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Read the PDF
    print(f"📖 Reading PDF: {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        print("❌ Error: Please put a PDF file in the folder and rename it to 'my_book.pdf'")
        return

    reader = PdfReader(PDF_PATH)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    print(f"   Extracted {len(full_text)} characters.")

    # 3. Chunking
    # We can't put a whole book into a single vector. Split into paragraphs/chunks.
    words = full_text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))
            current_chunk = [] # reset (you can implement overlap to improve results)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"   Split into {len(chunks)} chunks.")

    # 4. Create the index and insert data
    print("🚀 Indexing to Engram...")
    db = engram.VectorIndex(384) # model dimension
    metadata = {} # dictionary to save original text: ID -> Text

    for i, chunk in enumerate(chunks):
        # convert to vector
        vec = model.encode(chunk).tolist()
        # save in the engine
        db.add_vector(vec)
        # save the original text separately
        metadata[i] = chunk
        
        if i % 10 == 0:
            print(f"   Processed {i}/{len(chunks)} chunks...", end="\r")

    # 5. Save to disk (Persistence)
    # create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # save the vectors (Engram)
    db.save_index(INDEX_FILE)
    
    # save the texts (JSON)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Done! Saved index to '{INDEX_FILE}' and metadata to '{META_FILE}'")

if __name__ == "__main__":
    main()