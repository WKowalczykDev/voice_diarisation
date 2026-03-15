#!/bin/bash

# Ten skrypt ustawia odpowiednie zmienne w zależności od środowiska i uruchamia Dockera
echo "=========================================="
echo "    Inicjalizacja środowiska dla Audio    "
echo "=========================================="

if [ -n "$TERMUX_VERSION" ]; then
    echo "📱 Wykryto środowisko: Termux (Tablet)"
    
    export PULSE_SOCKET_PATH="/tmp/pulseaudio.socket"
    
    # Termux nie potrzebuje ciasteczka z auth-anonymous=1
    # Tworzymy pusty plik jako atrapę by Docker nie tworzył pustego folderu
    touch /tmp/pulse_cookie_dummy
    export PULSE_COOKIE_PATH="/tmp/pulse_cookie_dummy"
    
    # Upewniamy się, że w Termuxie działa serwer dźwięku
    if [ ! -S "$PULSE_SOCKET_PATH" ]; then
        echo "Uruchamiam serwer PulseAudio dla Termuxa..."
        pulseaudio --start --load="module-native-protocol-unix auth-anonymous=1 socket=$PULSE_SOCKET_PATH"
    fi

else
    echo "💻 Wykryto środowisko: Standardowy Linux (Laptop/PC)"
    
    # Większość nowych dystrybucji Linux używa /run/user/ID/pulse/native (dotyczy również PipeWire-Pulse)
    USER_ID=$(id -u)
    export PULSE_SOCKET_PATH="/run/user/$USER_ID/pulse/native"
    
    # Ścieżka do ciasteczka PulseAudio
    export PULSE_COOKIE_PATH="$HOME/.config/pulse/cookie"
    
    # Tworzymy puste ciasteczko w przypadku gdy użytkownik ma PipeWire i nie ma pliku pulse/cookie
    if [ ! -f "$PULSE_COOKIE_PATH" ]; then
        mkdir -p "$HOME/.config/pulse"
        touch "$PULSE_COOKIE_PATH"
    fi
    
    if [ ! -S "$PULSE_SOCKET_PATH" ]; then
        echo "⚠️  OSTRZEŻENIE: Nie znaleziono socketu PulseAudio/PipeWire w $PULSE_SOCKET_PATH"
        echo "Audio może nie działać prawidłowo w kontenerze!"
    fi
fi

# Jeśli przekazujesz argumenty do skryptu np. "./run_docker.sh build" albo "./run_docker.sh up -d"
CMD=${1:-"run"}

if [ "$CMD" = "run" ]; then
    echo "🚀 Uruchamiam aplikację (docker compose run --rm app)..."
    docker compose run --rm app
else
    echo "⚙️  Uruchamiam: docker compose $@"
    docker compose "$@"
fi
