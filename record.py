import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
from queue import Queue
from threading import Thread


class AudioRecorder:
    """Klasa do nagrywania i dzielenia audio na chunki"""

    def __init__(self, device_id=None, samplerate=48000, target_samplerate=16000,
                 chunk_duration=1.0, output_dir="chunks"):
        self.device_id = device_id
        self.samplerate = samplerate
        self.target_samplerate = target_samplerate
        self.chunk_duration = chunk_duration
        self.output_dir = output_dir

        self.audio_buffer = []
        self.chunk_counter = 0
        self.chunk_queue = Queue()
        self.is_recording = False

        os.makedirs(output_dir, exist_ok=True)
        if device_id is not None:
            sd.default.device = [device_id, sd.default.device[1]]

    def _callback(self, indata, frames, time_info, status):
        """Callback do zbierania audio"""
        if status:
            print(f"⚠️  {status}")
        # Bierzemy tylko pierwszy kanał, żeby uniknąć problemów z przesunięciem fazy
        # między mikrofonami (co może niszczyć rozpoznawanie mowy)
        mono_data = indata[:, 0]
        self.audio_buffer.append(mono_data.copy())

    def _resample(self, audio_data, original_sr, target_sr):
        """Prosty resample przez interpolację"""
        if original_sr == target_sr:
            return audio_data

        duration = len(audio_data) / original_sr
        target_samples = int(duration * target_sr)

        indices = np.linspace(0, len(audio_data) - 1, target_samples)
        resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)

        return resampled

    def _save_chunk(self, audio_data, chunk_id):
        """Zapisz chunk do pliku"""
        filename = os.path.join(self.output_dir, f"chunk_{chunk_id:04d}.wav")

        # Resample do target samplerate
        resampled = self._resample(audio_data, self.samplerate, self.target_samplerate)

        sf.write(filename, resampled.reshape(-1, 1), self.target_samplerate)

        max_val = resampled.max()
        mean_val = np.abs(resampled).mean()

        return filename, max_val, mean_val

    def _process_buffer(self):
        """Wątek przetwarzający buffer"""
        chunk_samples = int(self.samplerate * self.chunk_duration)

        while self.is_recording or len(self.audio_buffer) > 0:
            time.sleep(0.1)

            if len(self.audio_buffer) > 0:
                total_samples = sum(len(chunk) for chunk in self.audio_buffer)

                if total_samples >= chunk_samples:
                    audio_data = np.concatenate(self.audio_buffer)
                    chunk_data = audio_data[:chunk_samples]

                    filename, max_val, mean_val = self._save_chunk(
                        chunk_data, self.chunk_counter
                    )

                    print(f"💾 {os.path.basename(filename)} | Max: {max_val:.4f} | Mean: {mean_val:.4f}")

                    # Dodaj do kolejki do przetwarzania
                    self.chunk_queue.put(filename)
                    self.chunk_counter += 1

                    remaining = audio_data[chunk_samples:]
                    self.audio_buffer = [remaining] if len(remaining) > 0 else []

    def start(self):
        """Rozpocznij nagrywanie"""
        device_id_to_query = self.device_id if self.device_id is not None else sd.default.device[0]
        device_info = sd.query_devices(device_id_to_query)
        
        # Pobieramy natywne parametry dla tego urządzenia
        native_samplerate = int(device_info.get('default_samplerate', self.samplerate))
        # Aktualizujemy własny samplerate, aby funkcja _process_buffer działała poprawnie
        self.samplerate = native_samplerate
        
        # Spróbujemy użyć tylu kanałów ile mikrofon podaje za domyślne (np. 2 zamiast 1)
        native_channels = int(device_info.get('max_input_channels', 1))
        # Jeśli max to 0, awaryjnie bierzemy 1 (choć to byłby dziwny przypadek dla mikrofonu)
        if native_channels < 1:
            native_channels = 1
            
        print(f"🎙️  NAGRYWANIE - chunki po {self.chunk_duration}s")
        print(f"🎤 Urządzenie: {device_info['name']}")
        print(f"⚙️  Natywne parametry wejścia: {native_samplerate} Hz, Kanały: {native_channels}")
        print(f"📁 Output: {self.output_dir}/")
        print(f"🔄 Resample: {self.samplerate} Hz → {self.target_samplerate} Hz")
        print("=" * 60)

        self.is_recording = True

        # Start processing thread
        processor = Thread(target=self._process_buffer, daemon=True)
        processor.start()

        # Start recording stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=native_channels,
                callback=self._callback
            )
            self.stream.start()
        except Exception as e:
            print(f"❌ BŁĄD: Nie można uruchomić strumienia mikrofonu: {e}")
            self.is_recording = False
            return

        print("🔴 Nagrywam... Ctrl+C żeby zakończyć\n")

    def stop(self):
        """Zatrzymaj nagrywanie"""
        self.is_recording = False
        self.stream.stop()
        self.stream.close()

        # Zapisz ostatni fragment
        if len(self.audio_buffer) > 0:
            remaining_data = np.concatenate(self.audio_buffer)
            if len(remaining_data) > 0:
                filename, _, _ = self._save_chunk(remaining_data, self.chunk_counter)
                self.chunk_queue.put(filename)
                print(f"💾 Ostatni fragment: {os.path.basename(filename)}")

        print(f"\n✅ Zakończono. Zapisano {self.chunk_counter} chunków")

    def get_chunk(self, block=True, timeout=None):
        """Pobierz następny chunk z kolejki"""
        return self.chunk_queue.get(block=block, timeout=timeout)

    def has_chunks(self):
        """Sprawdź czy są chunki do przetworzenia"""
        return not self.chunk_queue.empty()


def main():
    """Standalone mode - tylko nagrywanie"""
    # Pobierz urządzenie ze zmiennej środowiskowej, domyślnie None
    device_env = os.environ.get("AUDIO_DEVICE_ID")
    device_id = int(device_env) if device_env and device_env.strip() else None

    recorder = AudioRecorder(
        device_id=device_id,
        samplerate=48000,
        target_samplerate=16000,
        chunk_duration=1.0,
        output_dir="chunks"
    )

    try:
        recorder.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        recorder.stop()


if __name__ == "__main__":
    main()