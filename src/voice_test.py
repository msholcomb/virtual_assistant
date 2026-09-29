2#!/usr/bin/env python3
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

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

# Path to the Piper voice model; the matching .onnx.json must sit next to it.
PIPER_VOICE = os.path.expanduser("~\models\piper\en_US-danny-low.onnx")

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000            # 0.25 s of audio per block
LISTEN_TIMEOUT_S = 8.0       # give up if no full phrase in this long

SetLogLevel(-1)
print("Loading speech model (downloads on first run)...")
model = Model(lang="en-us")


def speak(text: str) -> None:
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
        with wave.open(wav_path, "rb") as w:
            rate, channels = w.getframerate(), w.getnchannels()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        sd.play(audio.reshape(-1, channels), samplerate=rate)
        sd.wait()
    finally:
        os.remove(wav_path)


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
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("Listening...")
        while time.monotonic() < deadline:
            try:
                data = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
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
    if "twin" in words: 
            return "What up twin whats popping"
    if "down" in words: 
        return "shit im down, im trying to get some puh twin"
    
    
    if "apologize" in words:
        return "my apologies avery you are hella tough and mike should come visit you"
    
    return f"You said: {text}"
    
    



def spec_responses():
    response_map = {"Hello": "Hello", "Time": datetime.datetime.now().strftime("It is %I:%M %p.")}
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