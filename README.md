# 🧠 Engram — High-Performance Local Vector Database

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg?style=flat&logo=python)
![Architecture](https://img.shields.io/badge/Optimized-AVX2_SIMD-red)
![Build](https://img.shields.io/badge/Build-CMake-green)

**Engram** is a high-performance, embedded vector search engine built in modern C++17.  
Designed for **local RAG pipelines**, Engram combines low-level AVX2 SIMD optimizations with clean, pythonic usability through `pybind11`.

Engram eliminates network latency, minimizes memory overhead, and delivers **24.5× faster** similarity search compared to Python implementations.

---

## 🚀 Features

### ⚡ High Performance
- Fully optimized **Euclidean distance using AVX2 SIMD** (`_mm256_fmadd_ps`).
- Processes **8 float dimensions per CPU cycle**.
- Custom contiguous memory layout for optimal CPU cache locality.

### 🧩 Architecture
- **Core Engine:** C++17, STL-only, zero external dependencies.
- **Bindings:** Python integration via pybind11.
- **Serialization:** Custom binary snapshot format for instant load/save.
- **Deterministic & lightweight:** Ideal for embedded or offline AI workflows.

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
Python Application Layer
      ↓
pybind11 Binding Layer
      ↓
C++ Core Engine
 ├── VectorIndex        (memory, indexing, persistence)
 └── MathUtils          (AVX2 SIMD math kernels)
```

---

## 🔧 Installation

### Requirements
- C++17 compiler  
- CMake ≥ 3.10  
- Python ≥ 3.8  

### Build Instructions

```bash
git clone https://github.com/your-username/engram.git
cd engram

# Configure
cmake -S . -B build

# Build (Release mode required for SIMD)
cmake --build build --config Release
```

### Install Python binding (dev mode)

```bash
python setup_dev.py
```

---

## 🧪 Usage Examples

### Basic Example

```python
import engram

db = engram.VectorIndex(3)

db.add_vector([1.0, 0.0, 0.0])
db.add_vector([0.0, 1.0, 0.0])

query = [0.9, 0.1, 0.0]
results = db.search(query, k=1)

print(results)
# → [(0, 0.1413)]
```

### Semantic Search (RAG)

```python
import engram
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = engram.VectorIndex(384)

docs = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta"
]

for doc in docs:
    db.add_vector(model.encode(doc).tolist())

query = model.encode("Italian food").tolist()
result = db.search(query, k=1)

print("Best match:", docs[result[0][0]])
# → "I love pasta"
```

---

## 📂 Project Layout

```
Engram/
├── include/            # C++ Headers (public API)
├── src/                # C++ Implementations
│   ├── VectorIndex.cpp
│   └── bindings.cpp
├── tests/              # Benchmarks & demos
│   ├── benchmark.py
│   └── demo_rag.py
├── CMakeLists.txt
└── README.md
```