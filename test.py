import time
import db
import model
from record import AudioRecorder
from threading import Thread
from queue import Empty

# === KONFIGURACJA ===

# Dodaj nowe osoby do bazy (odkomentuj gdy potrzebne)
# db.add("Wojtek", model.get_embedding("twoja_osoba.wav"))
# db.add("Ola", model.get_embedding("ola.wav"))

THRESHOLD = 0.25  # próg rozpoznania
DEBOUNCE_COUNT = 2  # ile kolejnych chunków musi pasować
MIN_ENERGY = 0.01  # minimalny poziom audio


# === FUNKCJE ===

def process_chunk(chunk_path):
    """Przetwórz jeden chunk audio"""
    try:
        # Pobierz embedding
        test_emb = model.get_embedding(chunk_path)

        # Porównaj z bazą
        results = []
        for name, ref_emb in db.get_all().items():
            score = model.compare(ref_emb, test_emb)
            results.append((name, score))

        results.sort(key=lambda x: x[1], reverse=True)

        # Zwróć najlepszy wynik
        if results and results[0][1] > THRESHOLD:
            return results[0]  # (name, score)
        return None

    except Exception as e:
        print(f"❌ Błąd przetwarzania {chunk_path}: {e}")
        return None


def process_chunks_worker(recorder):
    """Wątek przetwarzający chunki w tle"""
    current_speaker = None
    speaker_count = 0

    print("🔍 Processor uruchomiony\n")

    while True:
        try:
            # Pobierz chunk z kolejki
            chunk_path = recorder.get_chunk(block=True, timeout=1)

            # Przetwórz chunk
            start = time.time()
            result = process_chunk(chunk_path)
            process_time = time.time() - start

            if result:
                name, score = result

                # Debouncing - zmiana mówcy po kilku chunkach
                if name == current_speaker:
                    speaker_count += 1
                else:
                    speaker_count = 1
                    if speaker_count >= DEBOUNCE_COUNT:
                        current_speaker = name
                        print(f"\n🎤 MÓWI: {name} | Score: {score:.3f} | Time: {process_time:.2f}s")
                    else:
                        print(f"   {name}: {score:.3f} (potwierdzenie {speaker_count}/{DEBOUNCE_COUNT})")
            else:
                if current_speaker:
                    print(f"   🔇 Brak rozpoznania")
                    current_speaker = None
                    speaker_count = 0

        except Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Błąd w processorze: {e}")


def main():
    """Main - nagrywanie + real-time processing"""

    # Sprawdź bazę
    speakers = db.get_all()
    if not speakers:
        print("⚠️  UWAGA: Baza głosów jest pusta!")
        print("   Dodaj osoby przez: db.add('Imie', model.get_embedding('plik.wav'))")
        return

    print(f"📊 Załadowano {len(speakers)} głosów: {list(speakers.keys())}")
    print("=" * 60)

    # Utwórz recorder
    recorder = AudioRecorder(
        device_id=4,
        samplerate=48000,
        target_samplerate=16000,
        chunk_duration=1.5,  # 1.5s chunki dla lepszej jakości
        output_dir="chunks"
    )

    # Uruchom processor w tle
    processor = Thread(target=process_chunks_worker, args=(recorder,), daemon=True)
    processor.start()

    # Uruchom nagrywanie
    try:
        recorder.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n🛑 Zatrzymywanie...")
        recorder.stop()
        time.sleep(1)  # Daj czas na dokończenie przetwarzania
        print("✅ Zakończono")


if __name__ == "__main__":
    main()