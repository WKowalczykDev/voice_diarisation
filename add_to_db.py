import os
import sys
import glob
import db
import model

def add_folder_to_db(folder_path="."):
    print(f"🔍 Przeszukiwanie folderu: {folder_path} w poszukiwaniu plików .wav...")
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    
    if not wav_files:
        print("⚠️  Nie znaleziono żadnych plików .wav!")
        return
        
    for file_path in wav_files:
        # Pobieranie nazwy pliku bez rozszerzenia, np. "lucjan.wav" -> "Lucjan"
        basename = os.path.basename(file_path)
        name = os.path.splitext(basename)[0].capitalize()
        
        print(f"⚙️  Przetwarzanie i dodawanie: {name} ({basename})...")
        try:
            embedding = model.get_embedding(file_path)
            db.add(name, embedding)
            print(f"✅ Dodano {name} do bazy!")
        except Exception as e:
            print(f"❌ Błąd podczas dodawania {file_path}: {e}")
            
    print("\n📊 Aktualna zawartość bazy:")
    for name in db.get_all().keys():
        print(f" - {name}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    add_folder_to_db(folder)