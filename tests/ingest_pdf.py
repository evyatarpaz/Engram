import engram
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import json
import os

# הגדרות
PDF_PATH = "my_book.pdf"       # שים פה שם של קובץ PDF אמיתי שיש לך
INDEX_FILE = "data/book.bin"   # איפה נשמור את המנוע הווקטורי
META_FILE = "data/book_meta.json" # איפה נשמור את הטקסטים
CHUNK_SIZE = 100               # כמה מילים בכל חתיכה (לא גדול מדי ולא קטן מדי)

def main():
    PDF_PATH = input("Enter path to PDF file (default: my_book.pdf): ") or PDF_PATH
    # 1. טעינת המודל
    print("🧠 Loading AI Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. קריאת ה-PDF
    print(f"📖 Reading PDF: {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        print("❌ Error: Please put a PDF file in the folder and rename it to 'my_book.pdf'")
        return

    reader = PdfReader(PDF_PATH)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    print(f"   Extracted {len(full_text)} characters.")

    # 3. חיתוך לחתיכות (Chunking)
    # אנחנו לא יכולים להכניס ספר שלם לוקטור אחד. חותכים לפסקאות.
    words = full_text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))
            current_chunk = [] # איפוס (אפשר לעשות חפיפה - Overlap - לשיפור תוצאות)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"   Split into {len(chunks)} chunks.")

    # 4. יצירת האינדקס והכנסת הנתונים
    print("🚀 Indexing to Engram...")
    db = engram.VectorIndex(384) # מימד המודל
    metadata = {} # מילון לשמירת הטקסט המקורי: ID -> Text

    for i, chunk in enumerate(chunks):
        # המרה לוקטור
        vec = model.encode(chunk).tolist()
        # שמירה במנוע
        db.add_vector(vec)
        # שמירת הטקסט המקורי בצד
        metadata[i] = chunk
        
        if i % 10 == 0:
            print(f"   Processed {i}/{len(chunks)} chunks...", end="\r")

    # 5. שמירה לדיסק (Persistence)
    # יוצרים תיקיית data אם לא קיימת
    os.makedirs("data", exist_ok=True)
    
    # שומרים את הוקטורים (Engram)
    db.save_index(INDEX_FILE)
    
    # שומרים את הטקסטים (JSON)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Done! Saved index to '{INDEX_FILE}' and metadata to '{META_FILE}'")

if __name__ == "__main__":
    main()