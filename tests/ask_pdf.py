import engram
from sentence_transformers import SentenceTransformer
import json
import os

INDEX_FILE = "data/book.bin"
META_FILE = "data/book_meta.json"

def main():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        print("❌ Index not found. Please run ingest_pdf.py first.")
        return

    print("🧠 Loading Model & Index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # טעינת המנוע הווקטורי
    # שים לב: חייבים ליצור את האובייקט עם המימד הנכון (384) לפני הטעינה
    db = engram.VectorIndex(384)
    db.load_index(INDEX_FILE)

    # טעינת הטקסטים המקוריים
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"✅ Loaded {db.count} chunks from the book.")
    print("-" * 50)

    while True:
        query = input("\n🔍 Ask a question about the book (or 'q'): ")
        if query.lower() == 'q':
            break

        # 1. המרה לוקטור
        query_vec = model.encode(query).tolist()

        # 2. חיפוש
        results = db.search(query_vec, k=3) # תביא את ה-3 הכי רלוונטיים

        # 3. הצגת התשובות
        print(f"\n--- Best Answers from Engram ---")
        for res_id, dist in results:
            # שליפת הטקסט לפי ה-ID שהמנוע החזיר
            text_snippet = metadata[str(res_id)] 
            print(f"\n[Score: {dist:.4f}]")
            print(f"...{text_snippet}...")

if __name__ == "__main__":
    main()