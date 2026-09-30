#!/usr/bin/env python3
"""
Laptop speech-to-text test: press Enter, talk, and it talks back.

    pip install vosk sounddevice numpy piper-tts

The first run downloads a small English Vosk model automatically.
You also need a Piper voice (.onnx + .onnx.json), see PIPER_VOICE below.
Say "quit" or "exit" to stop.
"""

import datetime
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
import wave
import RPi.GPIO as GPIO
from time import sleep

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

# Path to the Piper voice model; the matching .onnx.json must sit next to it.
# To change voices: download a new one (see README/setup notes) and update
# this filename to match — nothing else in the script needs to change.

#gpio stuff
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


#setup led
led_pin = 16
GPIO.setup(led_pin, GPIO.OUT)

PIPER_VOICE = os.path.expanduser("~/models/piper/en_US-libritts_r-medium.onnx")


OUTPUT_DEVICE = 0   #bt headphones are 0 index in sound devices on rpi

SAMPLE_RATE = 16000          # what Vosk requires
MIC_SAMPLE_RATE = 48000      # what the mic hardware actually supports; USB mics
                             # (e.g. Blue Yeti) often reject 16000 directly. Check
                             # with: python3 -m sounddevice
BLOCK_SIZE = int(MIC_SAMPLE_RATE * 0.25)   # 0.25 s of audio per block
LISTEN_TIMEOUT_S = 8.0       # give up if no full phrase in this long

SetLogLevel(-1)
print("Loading speech model (downloads on first run)...")
model = Model(lang="en-us")

def gpio_on(pin):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(led_pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)
    GPIO.cleanup()
        
    
def gpio_off(pin):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(led_pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()
        
    

def speak(text: str) -> None:
    gpio_on(led_pin)
    print(f"[assistant] {text}")
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        # Run Piper through the current interpreter so it works inside a venv.
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", PIPER_VOICE, "-f", wav_path],
            input=text.encode(),
            check=True,
        )
        # Play via paplay (PipeWire/PulseAudio) rather than sd.play(), since
        # this system's PortAudio build has no Pulse host API and can only
        # see raw ALSA hw: devices -- it can't reach a Bluetooth sink at all.
        # paplay talks to PipeWire/Pulse directly and uses whatever sink is
        # currently set as default (see: pactl set-default-sink).
        subprocess.run(["paplay", wav_path], check=True)
    finally:
        os.remove(wav_path)
    gpio_off(led_pin)
    


def listen() -> str:
    """Record from the default mic until Vosk detects the end of a phrase."""
    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(bytes(indata))

    rec = KaldiRecognizer(model, SAMPLE_RATE)
    deadline = time.monotonic() + LISTEN_TIMEOUT_S

    with sd.RawInputStream(
        samplerate=MIC_SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("Listening...")
        while time.monotonic() < deadline:
            try:
                raw = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # Downsample from the mic's native rate to the 16 kHz Vosk expects.
            samples = np.frombuffer(raw, dtype=np.int16)
            factor = MIC_SAMPLE_RATE // SAMPLE_RATE
            data = samples[::factor].tobytes()
            if rec.AcceptWaveform(data):
                return json.loads(rec.Result()).get("text", "")
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial:
                print(f"  ...{partial}", end="\r")
    return json.loads(rec.FinalResult()).get("text", "")


def respond(text: str) -> str:
    words = set(text.split())
    if "time" in words:
        return datetime.datetime.now().strftime("It is %I:%M %p.")
    if "hello" in words or "hi" in words:
        return "Hello Mike. Speech to text is working."
    return f"You said: {text}"


def main() -> None:
    print(f"Using mic: {sd.query_devices(sd.default.device[0])['name']}")
    speak("Ready.")
    while True:
        input("\nPress Enter, then speak... ")
        text = listen().strip()
        print(f"[heard] {text!r}")
        if not text:
            speak("I didn't catch that.")
            continue
        if text in ("quit", "exit"):
            speak("Goodbye.")
            break
        speak(respond(text))


if __name__ == "__main__":
    main()