# src/predict.py
import numpy as np
import tensorflow as tf
import joblib
from feature_extraction import extract_features

# Load model and label encoder
model = tf.keras.models.load_model("models/emotion_recognition_model.h5")
le = joblib.load("models/label_encoder.pkl")

def predict_emotion(file_path):
    """Predict emotion from a WAV file."""
    mfcc = extract_features(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = mfcc[..., np.newaxis]
    prediction = model.predict(mfcc)
    predicted_emotion = le.inverse_transform([np.argmax(prediction)])
    return predicted_emotion[0]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <audio_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    emotion = predict_emotion(file_path)
    print(f"Predicted Emotion: {emotion}")
