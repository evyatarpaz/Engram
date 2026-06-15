# 🧠 Engram — High-Performance Local Vector Database

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg?style=flat&logo=python)
![Architecture](https://img.shields.io/badge/Optimized-AVX2_SIMD-red)
![Build](https://img.shields.io/badge/Build-CMake-green)

**Engram** is a high-performance, embedded vector search engine built in modern C++17.  
Designed for **local RAG (Retrieval-Augmented Generation) pipelines**, Engram combines aggressive AVX2 SIMD vectorization with custom memory alignment and hand-tuned algorithms for **314× faster** similarity search compared to standard Python implementations.

Engram eliminates network latency, minimizes memory overhead, and delivers production-grade performance for offline AI workflows.

---

## 🚀 Features

### ⚡ Extreme Performance

- **Fused Multiply-Add (FMA) SIMD**: `_mm256_fmadd_ps` processes 8 float dimensions per CPU cycle (~314× faster than Python)
- **32-byte memory alignment**: Ensures efficient L1 cache line utilization and unaligned load penalties
- **Dimension padding**: Rounds vector dimensions to multiples of 8 for SIMD safety and correct zero-padding
- **Custom aligned allocator**: Zero-copy allocation via `_mm_malloc()` with guaranteed cache alignment
- **O(N log k) max-heap search**: Efficient k-NN retrieval without full dataset scans

### 🧩 Architecture

- **Core Engine:** C++17, STL-only, zero external dependencies
- **Memory Layout:** Flat contiguous buffer with calculated stride for optimal prefetch and cache locality
- **Distance Metric:** Squared Euclidean (no sqrt—preserves ranking while avoiding expensive floating-point operations)
- **Bindings:** Python integration via pybind11 with zero-overhead function calls
- **Persistence:** Custom binary format with padding preservation for direct disk-load compatibility
- **O(1) Deletion:** Swap-and-pop strategy avoids memory shifting
- **Deterministic & lightweight:** Ideal for embedded or offline AI workflows

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

Benchmark: 100,000 vectors, 128-dimensional, single k-NN query.

| Implementation        | Time (Seconds) | Speedup    |
| --------------------- | -------------- | ---------- |
| Python (NumPy/List)   | 1.0625s        | 1×         |
| **Engram (C++ AVX2)** | **0.0034s**    | **314.2×** |

### What's Responsible for the Speedup?

1. **Fused Multiply-Add (FMA)**: Each `_mm256_fmadd_ps` instruction computes `distance += (a - b)²` in a single CPU cycle instead of separate load/multiply/add operations
2. **8-float parallelism**: AVX2 processes 8 float32 values simultaneously (256 bits ÷ 32 bits/float)
3. **32-byte alignment**: Eliminates unaligned load penalties and improves L1 cache utilization
4. **Dimension padding**: Vectors padded to multiples of 8, ensuring SIMD loads never overflow
5. **Max-heap k-NN**: O(N log k) search complexity instead of O(N) sorting
6. **Cache-friendly layout**: Flat contiguous memory with prefetch-optimized sequential access

> **Result**: 314× speedup through hardware-level vectorization and algorithmic efficiency.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────┐
│   Python Application Layer              │
│  (app.py, tests/ask_pdf.py, etc.)      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   pybind11 Binding Layer                │
│  (Zero-overhead function marshaling)    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  C++17 Core Engine (VectorIndex)        │
│  ├─ AVX2 SIMD Vectorization             │
│  ├─ FMA (Fused Multiply-Add)            │
│  ├─ Max-Heap k-NN Search                │
│  └─ O(1) Deletion (Swap-and-Pop)        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Aligned Memory Layout (32-byte)        │
│  ├─ Dimension Padding (multiples of 8)  │
│  ├─ Custom AlignedAllocator             │
│  └─ Flat Contiguous Buffer              │
└─────────────────────────────────────────┘
```

**Key Design Decisions:**

- **No external dependencies**: Pure C++17 + STL + pybind11 (header-only for bindings)
- **Immutable during search**: Thread-safe for read-heavy workloads
- **Single-threaded performance focus**: Extreme speed through SIMD, not parallelism
- **Memory-mapped persistence**: Load indices in O(1) time (binary format)
