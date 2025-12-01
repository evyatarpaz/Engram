import engram # your C++ engine
from sentence_transformers import SentenceTransformer
import time

def main():
    print("--- 🧠 Initializing AI Model ---")
    # Load a small, fast model that converts text to vectors (384 dims)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = 384 

    print("--- 🚀 Initializing Engram C++ Engine ---")
    # Create the engine with the correct embedding dimension of the model
    db = engram.VectorIndex(embedding_dim)

    # Simple Python-side database (to remember which text corresponds to which ID)
    documents = [
        "The quick brown fox jumps over the lazy dog",  # ID 0
        "Artificial Intelligence is changing the world", # ID 1
        "I love eating pizza and pasta",                 # ID 2
        "Python is a great programming language",        # ID 3
        "Stock market crashed yesterday",                # ID 4
        "My dog loves to play in the park"               # ID 5 (related to 0)
    ]

    print(f"--- 📥 Indexing {len(documents)} documents ---")
    
    # Ingestion phase
    start_time = time.time()
    for i, doc in enumerate(documents):
        # 1. AI converts text to a vector
        vector = model.encode(doc) 
        
        # 2. Send to your C++ engine
        db.add_vector(vector.tolist())
        
        print(f"Indexed doc {i}: '{doc}'")

    print(f"Done! Engine contains {db.count} vectors. Time: {time.time() - start_time:.4f}s")
    print("-" * 50)

    # Search phase
    while True:
        query_text = input("\nEnter a search query (or 'q' to quit): ")
        if query_text.lower() == 'q':
            break

        # 1. Convert the query to a vector
        query_vector = model.encode(query_text).tolist()

        # 2. Your engine searches for the nearest vector
        results = db.search(query_vector, k=1)

        if results:
            best_id, dist = results[0]
            print(f"\n✅ Best Match Found (ID: {best_id}, Dist: {dist:.4f}):")
            print(f"   --> \"{documents[best_id]}\"")
        else:
            print("❌ No match found.")

if __name__ == "__main__":
    main()