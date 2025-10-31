# src/realtime_detect.py
import sounddevice as sd
import numpy as np
import tempfile
import wavio
from predict import predict_emotion

def record_audio(duration=3, fs=44100):
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return recording, fs

def save_temp_wav(recording, fs):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_file.name, recording, fs, sampwidth=2)
    return temp_file.name

if __name__ == "__main__":
    rec, fs = record_audio()
    file_path = save_temp_wav(rec, fs)
    emotion = predict_emotion(file_path)
    print(f"Predicted Emotion: {emotion}")
