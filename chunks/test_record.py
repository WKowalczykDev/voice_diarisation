import sounddevice as sd
import soundfile as sf
import numpy as np

# Ustaw działające urządzenie
sd.default.device = [4, sd.default.device[1]]

SAMPLERATE = 16000
DURATION = 3  # sekundy
FILENAME = "test.wav"

print(f"🎙️  Nagrywam {DURATION}s do {FILENAME}...")
print(f"🎤 Używam urządzenia: {sd.query_devices(4)['name']}")
print("Mów coś!")

recording = sd.rec(int(DURATION * SAMPLERATE),
                   samplerate=SAMPLERATE,
                   channels=1,
                   dtype='float32')

sd.wait()

sf.write(FILENAME, recording, SAMPLERATE)

print(f"✅ Zapisano. Odtwórz plik: {FILENAME}")
print(f"📊 Kształt: {recording.shape}, Min: {recording.min():.4f}, Max: {recording.max():.4f}")

if recording.max() > 0.001:
    print("🎉 Mikrofon działa prawidłowo!")
else:
    print("⚠️  Ostrzeżenie: Bardzo cichy sygnał lub cisza")