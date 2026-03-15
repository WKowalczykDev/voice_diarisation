import sounddevice as sd
import numpy as np

print("🎤 TESTOWANIE WSZYSTKICH MIKROFONÓW")
print("=" * 60)

devices = sd.query_devices()
working_devices = []

for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"\n🔍 Testuję urządzenie {i}: {device['name']}")

        try:
            recording = sd.rec(
                int(1.5 * 16000),
                samplerate=16000,
                channels=1,
                device=i,
                dtype='float32'
            )
            sd.wait()

            max_val = recording.max()
            mean_val = np.abs(recording).mean()

            print(f"   Max: {max_val:.6f}, Mean: {mean_val:.6f}")

            if max_val > 0.001:
                print(f"   ✅ DZIAŁA!")
                working_devices.append((i, device['name']))
            else:
                print(f"   ❌ Cisza")

        except Exception as e:
            print(f"   ⚠️  Błąd: {e}")

print("\n" + "=" * 60)
if working_devices:
    print(f"\n✅ DZIAŁAJĄCE MIKROFONY:")
    for dev_id, dev_name in working_devices:
        print(f"   {dev_id}: {dev_name}")
    print(f"\n💡 Użyj w kodzie:")
    print(f"   sd.default.device = [{working_devices[0][0]}, sd.default.device[1]]")
else:
    print("\n❌ Żaden mikrofon nie działa!")
    print("\n🔧 Sprawdź:")
    print("   1. Uprawnienia mikrofonu (Settings > Privacy)")
    print("   2. alsamixer - czy mikrofon nie jest wyciszony")
    print("   3. pavucontrol - czy aplikacja ma dostęp do mikrofonu")