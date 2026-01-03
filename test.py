import time
import db
import model

# === KONFIGURACJA ===

# Dodaj nowe osoby (odkomentuj gdy potrzebne)
# start = time.time()
# db.add("Wojtek", model.get_embedding("twoja_osoba.wav"))
# db.add("Ola", model.get_embedding("ola.wav"))
# print(f"✅ {name} added in {time.time()-start:.2f}s")

# # Sprawdź osobę
TEST_FILE = "wojtek3.wav"

# === WERYFIKACJA ===

start = time.time()
test_emb = model.get_embedding(TEST_FILE)
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