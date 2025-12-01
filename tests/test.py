import engram

print("--- Testing Engram in Debug Mode ---")
try:
    db = engram.VectorIndex(3)
    db.add_vector([1.0, 0.0, 0.0])
    print(f"SUCCESS! DB initialized with {db.count} vectors.")
except Exception as e:
    print(f"FAILED: {e}")