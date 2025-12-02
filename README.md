# 🧠 Engram — High-Performance Local Vector Database

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg?style=flat&logo=python)
![Architecture](https://img.shields.io/badge/Optimized-AVX2_SIMD-red)
![Build](https://img.shields.io/badge/Build-CMake-green)

**Engram** is a high-performance, embedded vector search engine built in modern C++17.  
Designed for **local RAG (Retrieval-Augmented Generation) pipelines**, Engram combines low-level AVX2 SIMD optimizations with clean, pythonic usability through `pybind11`.

Engram eliminates network latency, minimizes memory overhead, and delivers **24.5× faster** similarity search compared to standard Python implementations.

---

## 🚀 Features

### ⚡ High Performance
- Fully optimized **Euclidean distance using AVX2 SIMD** (`_mm256_fmadd_ps`).
- Processes **8 float dimensions per CPU cycle**.
- Custom contiguous memory layout for optimal CPU cache locality.

### 🧩 Architecture
- **Core Engine:** C++17, STL-only, zero external dependencies.
- **Bindings:** Python integration via pybind11.
- **Persistence:** Custom binary snapshot format for instant load/save (`save_index`, `load_index`).
- **Deterministic & lightweight:** Ideal for embedded or offline AI workflows.

---

## 🛠️ Installation & Build

### Prerequisites
- C++17 compliant compiler (MSVC, GCC, Clang)
- CMake 3.10+
- Python 3.8+

### Build Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/evyatarpaz/Engram.git
   cd Engram
   ```

2. **Configure and Build:**
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

3. **Install the Python Module:**
   Ensure the compiled `.pyd` (Windows) or `.so` (Linux) file is in your Python path or project root.

---

## 🐍 Python API

```python
import engram

# Initialize index with vector dimension (e.g., 384 for all-MiniLM-L6-v2)
db = engram.VectorIndex(384)

# Add vectors (must match dimension)
db.add_vector([0.1, 0.2, ...])

# Search for k-nearest neighbors
# Returns list of (id, distance) tuples
results = db.search(query_vector, k=5)

# Persistence
db.save_index("index.bin")
db.load_index("index.bin")

# Properties
print(db.count)      # Number of vectors
print(db.dimension)  # Vector dimension
```

---

## 📚 RAG Pipeline Example (PDF Chat)

Engram includes a full example of a "Chat with your PDF" pipeline using `sentence-transformers`.

### 1. Install Dependencies
```bash
pip install sentence-transformers pypdf
```

### 2. Ingest a PDF
Use `tests/ingest_pdf.py` to read a PDF, chunk the text, generate embeddings, and save them to Engram.

```bash
python tests/ingest_pdf.py
# Follow the prompts to provide a PDF path (default: my_book.pdf)
```
This creates `data/book.bin` (vector index) and `data/book_meta.json` (text chunks).

### 3. Ask Questions
Use `tests/ask_pdf.py` to load the index and chat with the document.

```bash
python tests/ask_pdf.py
```

---

## 📊 Benchmark

Benchmark: 100,000 vectors, 128-dimensional.

| Implementation        | Time (Seconds) | Speedup |
|----------------------|----------------|---------|
| Python (NumPy/List)  | 1.61s          | 1×      |
| **Engram (C++ AVX2)** | **0.06s**      | **≈24.5×** |

> Engram delivers hardware-level speedups through hand-tuned AVX2 intrinsics and zero-overhead Python bindings.

---

## 🏛️ System Architecture

```
[ Python Application Layer ]
           ↓
[ pybind11 Binding Layer ]
           ↓
[ C++ Core Engine (AVX2) ]
           ↓
[ Flat Buffer Memory Layout ]
```
