import time
import sys
import db
import model

if len(sys.argv) != 2:
    print("Usage: python verify.py <audio_path>")
    sys.exit(1)

path = sys.argv[1]

start = time.time()
test_emb = model.get_embedding(path)
embed_time = time.time() - start

results = []
for name, ref_emb in db.get_all().items():
    score = model.compare(ref_emb, test_emb)
    results.append((name, score))

results.sort(key=lambda x: x[1], reverse=True)

print(f"\n⏱️  Embedding: {embed_time:.2f}s | Total: {time.time()-start:.2f}s\n")
for name, score in results:
    match = "✅" if score > 0.25 else "❌"
    print(f"{name:15} {score:.3f} {match}")