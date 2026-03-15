import sounddevice as sd

device_id = 4
common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]

print(f"🎤 Testowanie sample rate dla urządzenia {device_id}")
print(f"   {sd.query_devices(device_id)['name']}")
print("=" * 60)

working_rates = []

for rate in common_rates:
    try:
        # Spróbuj otworzyć stream z tym sample rate
        stream = sd.InputStream(
            device=device_id,
            samplerate=rate,
            channels=1
        )
        stream.close()
        print(f"✅ {rate} Hz - DZIAŁA")
        working_rates.append(rate)
    except Exception as e:
        print(f"❌ {rate} Hz - nie działa")

print("\n" + "=" * 60)
if working_rates:
    print(f"✅ DZIAŁAJĄCE SAMPLE RATES: {working_rates}")
    print(f"\n💡 Użyj w test.py:")
    print(f"   recorder = AudioRecorder(")
    print(f"       device_id=4,")
    print(f"       samplerate={working_rates[0]},  # <-- ZMIEŃ NA TO")
    print(f"       target_samplerate=16000,")
    print(f"       chunk_duration=1.5")
    print(f"   )")

    # Dodatkowo test nagrywania
    print(f"\n🎙️  Test nagrywania 2s z {working_rates[0]} Hz...")
    import numpy as np

    recording = sd.rec(int(2 * working_rates[0]),
                       samplerate=working_rates[0],
                       channels=1,
                       device=device_id,
                       dtype='float32')
    sd.wait()

    if recording.max() > 0.001:
        print(f"✅ Nagrywanie działa! Max: {recording.max():.4f}")
    else:
        print(f"⚠️  Nagrywanie nie działa (cisza)")
else:
    print("❌ Żaden sample rate nie działa!")
    print("\n🔧 Spróbuj:")
    print("   1. Innego urządzenia z listy")
    print("   2. pavucontrol - sprawdź ustawienia")