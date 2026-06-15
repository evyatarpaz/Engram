import os
import json
import engram
from sentence_transformers import SentenceTransformer

os.makedirs("data", exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
db = engram.VectorIndex(384) # 384 dimensions for all-MiniLM-L6-v2

# Sample data mimicking your target domain
texts = [
    "NVIDIA BlueField DPUs offload networking, storage, and security.",
    "AVX2 instructions allow processing 8 single-precision floats simultaneously.",
    "Zero-trust security models require ephemeral data storage."
]
metadata = {}

for i, text in enumerate(texts):
    vec = model.encode(text).tolist()
    db.add_vector(vec)
    metadata[str(i)] = text

db.save_index("data/book.bin")
with open("data/book_meta.json", "w") as f:
    json.dump(metadata, f)

print(f"Engine initialized. Ingested {db.count} vectors successfully.")