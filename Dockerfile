FROM python:3.11-slim

# Instalacja pakietów systemowych potrzebnych do obsługi audio (PortAudio/ALSA/PulseAudio)
RUN apt-get update && apt-get install -y \
    gcc \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    libasound2-plugins \
    pulseaudio \
    alsa-utils \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopiowanie konfiguracji ALSA do wsparcia PulseAudio
COPY asound.conf /etc/asound.conf

# Kopiowanie plików zależności (najpierw, żeby wykorzystać cache Dockera)
COPY requirements.txt .

# Instalacja bibliotek Pythona
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Kopiowanie reszty plików projektu
COPY . .

# Domyślne polecenie do odpalenia aplikacji (możesz zmienić np. na record.py lub model.py)
CMD ["python", "test.py"]
