import sounddevice as sd

print("🎤 DOSTĘPNE URZĄDZENIA AUDIO:")
print("=" * 60)
print(sd.query_devices())
print("\n" + "=" * 60)
print(f"\n🔧 DOMYŚLNE URZĄDZENIE WEJŚCIOWE: {sd.default.device[0]}")
print(f"🔊 DOMYŚLNE URZĄDZENIE WYJŚCIOWE: {sd.default.device[1]}")

# Test nagrywania z domyślnego urządzenia
print("\n🎙️  Test nagrywania 2s...")
import numpy as np

recording = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

print(f"📊 Min: {recording.min():.6f}, Max: {recording.max():.6f}, Mean: {np.abs(recording).mean():.6f}")

if recording.max() == 0.0:
    print("\n❌ PROBLEM: Mikrofon nic nie nagrywa!")
    print("\n💡 ROZWIĄZANIA:")
    print("1. Sprawdź uprawnienia mikrofonu w systemie")
    print("2. Wybierz inne urządzenie wejściowe:")
    print("   sd.default.device = [NUMER_Z_LISTY_POWYŻEJ, sd.default.device[1]]")
    print("3. Linux: sudo apt install portaudio19-dev python3-pyaudio")
else:
    print("\n✅ Mikrofon działa!")