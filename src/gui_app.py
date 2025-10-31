# src/gui_app.py
import tkinter as tk
from tkinter import messagebox
from realtime_detect import record_audio, save_temp_wav
from predict import predict_emotion

def detect_emotion_gui():
    rec, fs = record_audio()
    file_path = save_temp_wav(rec, fs)
    emotion = predict_emotion(file_path)
    messagebox.showinfo("Detected Emotion", f"Predicted Emotion: {emotion}")

root = tk.Tk()
root.title("Emotion Recognition from Speech")
root.geometry("400x200")

label = tk.Label(root, text="Press the button to record your voice\nand detect your emotion.", font=("Arial", 12))
label.pack(pady=20)

btn = tk.Button(root, text="Record & Detect", command=detect_emotion_gui, font=("Arial", 12), bg="lightblue")
btn.pack(pady=10)

root.mainloop()
