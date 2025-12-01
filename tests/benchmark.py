import engram
import time
import numpy as np
import random

def run_benchmark():
    # 1. Benchmark settings
    NUM_VECTORS = 100_000   # one hundred thousand vectors!
    DIMENSION = 128         # standard dimension
    
    print(f"--- ⚔️  BENCHMARK: C++ (Engram) vs Python List ---")
    print(f"Generating {NUM_VECTORS} random vectors (dim={DIMENSION})...")
    
    # create fake data (NumPy is fast at this)
    # we create a huge array of random numbers
    data = np.random.rand(NUM_VECTORS, DIMENSION).astype(np.float32)
    query = np.random.rand(DIMENSION).astype(np.float32).tolist()

    # --- Competitor 1: Python List (naive approach) ---
    print("\n1. Preparing Python List...")
    python_list = data.tolist() # convert to regular Python lists
    
    print("   Running Python Search (Linear Scan)...")
    start_py = time.time()
    
    # naive Python search (compute distance for each)
    min_dist = float('inf')
    best_idx = -1
    for i, vec in enumerate(python_list):
        # compute Euclidean distance in pure Python
        dist = sum((v1 - v2) ** 2 for v1, v2 in zip(vec, query)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            
    end_py = time.time()
    time_py = end_py - start_py
    print(f"   🐍 Python Time: {time_py:.4f} seconds")

    # --- Competitor 2: Engram C++ (your engine) ---
    print("\n2. Preparing Engram C++ Engine...")
    db = engram.VectorIndex(DIMENSION)
    
    # loading data into the engine (this takes a moment but is one-time)
    # in a real benchmark you should measure only the search
    for vec in python_list:
        db.add_vector(vec)
        
    print("   Running C++ Search (SIMD Optimized)...")
    start_cpp = time.time()
    
    # the actual search
    results = db.search(query, 1)
    
    end_cpp = time.time()
    time_cpp = end_cpp - start_cpp
    print(f"   🚀 C++ Time:    {time_cpp:.4f} seconds")

    # --- Results ---
    print("-" * 40)
    speedup = time_py / time_cpp
    print(f"🏆 Speedup Factor: {speedup:.1f}x FASTER")
    
    if results[0][0] == best_idx:
        print("✅ Validation: Results match perfectly!")
    else:
        print("❌ Validation: Results differ! (Check math)")

if __name__ == "__main__":
    run_benchmark()