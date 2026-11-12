# Serves the trained model as a small web API
# POST a WAV file to /predict and it returns the predicted class with the three most likely classes

# For data handling
import numpy as np

# For loading the trained model (runs on TensorFlow)
import keras

# For the web API
from fastapi import FastAPI, File, HTTPException, UploadFile

# For turning WAV files into log-mel spectrograms, exactly as in training
from features import CLASS_NAMES, wav_to_spectrogram


# Load the model saved by train.py (once, when the server starts)
model = keras.models.load_model('model.keras')

app = FastAPI(title="ESC-10 Sound Classifier")


# Classify an uploaded WAV file
@app.post('/predict')
def predict(file: UploadFile = File(...)): # File(...) makes the upload a required form field
    try:
        spectrogram = wav_to_spectrogram(file.file.read())
    except Exception: # TensorFlow raises an error when the upload is not a readable WAV file
        raise HTTPException(status_code=400, detail="Could not read the upload as a 16-bit, 44.1 kHz WAV file.")

    # Add a batch axis, since the model expects a batch of clips
    probabilities = model.predict(spectrogram[np.newaxis, ...], verbose=0)[0] # [0] selects the only clip in the batch

    # Report the three most likely classes, most likely first
    top_indices = np.argsort(probabilities)[::-1][:3]
    return {
        'predicted_class': CLASS_NAMES[top_indices[0]],
        'top_predictions': [
            {'class': CLASS_NAMES[i], 'probability': round(float(probabilities[i]), 4)} for i in top_indices
        ]
    }
