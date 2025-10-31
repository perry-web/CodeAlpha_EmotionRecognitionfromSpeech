# Emotion Recognition from Speech 

## Overview
This project focuses on recognizing human emotions (such as **happy**, **sad**, **angry**, **neutral**, etc.) from speech audio using **deep learning** and **speech signal processing techniques**.  
It leverages **feature extraction (MFCCs)** and **neural networks (CNN/LSTM)** to classify emotional states based on speech characteristics.

---

## Objectives
- Recognize and classify human emotions from speech signals.
- Explore and analyze audio features using visualization.
- Build and train a deep learning model using extracted MFCCs.
- Evaluate the model's performance and predict emotions for unseen audio.

---

## Dataset
The project uses the **Toronto Emotional Speech Set (TESS)** dataset, which contains recordings of 7 emotions:  
**Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad**

Instead of manually downloading the dataset, it is programmatically retrieved using **KaggleHub**:

```python
import kagglehub

# Download latest version of the TESS dataset
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
print("Path to dataset files:", path)
````

---

## Project Structure

```
Emotion_Recognition/
│
├── data/
│   └── (TESS dataset downloaded automatically here)
│
├── Emotion_Recognition_Notebook.ipynb
│
├── models/
│   └── emotion_model.keras          # Trained deep learning model
│
├── app/
│   └── predict.py                   # Script for emotion prediction
│   └── utils.py                     # Feature extraction utilities
│
├── README.md
└── requirements.txt
```

---

## Environment Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate # On Linux/Mac
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Required Libraries**

   * tensorflow / keras
   * librosa
   * numpy
   * pandas
   * seaborn
   * matplotlib
   * scikit-learn
   * kagglehub

---

## Exploratory Data Analysis (EDA)

After downloading and confirming the dataset, we explored it using visual analysis tools.

### 1. Create DataFrame

Each `.wav` file path and its corresponding emotion label were stored in a Pandas DataFrame.

```python
file_paths = glob.glob(os.path.join(path, "**/*.wav"), recursive=True)
def get_emotion_from_path(file_path):
    folder = os.path.basename(os.path.dirname(file_path))
    return folder.split('_')[-1].lower()

df = pd.DataFrame({
    'path': file_paths,
    'emotion': [get_emotion_from_path(fp) for fp in file_paths]
})
```

### 2. Emotion Distribution

A **Seaborn countplot** was used to visualize dataset balance:

```python
sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='viridis')
```

### 3. Waveform and Spectrogram Analysis

Custom functions were created to visualize time-domain and frequency-domain characteristics of sample audio.

```python
def plot_waveform(file_path):
    y, sr = librosa.load(file_path, sr=None)
    librosa.display.waveshow(y, sr=sr)

def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
```

These plots helped visualize how emotion affects energy, tone, and frequency ranges.

---

## Feature Extraction

To transform raw audio into a numerical representation suitable for deep learning, **Mel-Frequency Cepstral Coefficients (MFCCs)** were extracted.

```python
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)
```

All features were compiled into NumPy arrays (`X`) and labels (`y`), followed by label encoding and train-test splitting.

---

## Model Architecture

### CNN-LSTM Hybrid Model

The model combines convolutional layers for feature extraction with LSTM layers for temporal sequence modeling.

```python
model = Sequential([
    Reshape((40, 1), input_shape=(40,)),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

**Compilation**

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Training

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)
```

---

## Evaluation

Accuracy and loss curves were plotted to evaluate performance.

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()
```

A confusion matrix was also used to assess classification performance per emotion.

---

## Prediction Function

After training, a helper function was created to predict emotions for any new `.wav` file:

```python
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]
```

Example usage:

```python
sample_audio = random.choice(glob.glob(os.path.join(path, "**/*.wav"), recursive=True))
print("File:", sample_audio)
print("Predicted Emotion:", predict_emotion(sample_audio))
```

---

## Results

* The CNN-LSTM model achieved strong accuracy on the validation set.
* Waveform and spectrogram analysis revealed consistent differences between emotions.
* The project demonstrates a full pipeline from **raw audio → feature extraction → model training → prediction**.

---


## References

* TESS Dataset: [https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
* Librosa Documentation: [https://librosa.org/doc/latest/](https://librosa.org/doc/latest/)
* TensorFlow/Keras: [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
* Audio Emotion Recognition Tutorials: Krish Naik (YouTube)

---

## Author

**Perry NDZIE KOKO**

* Email: [perrykoko@engineer.com](mailto:perrykoko@engineer.com)
