# Tests for the audio code in features.py and the API in app.py

# For generating a test tone
import numpy as np
import tensorflow as tf

# For calling the API without starting a server
from fastapi.testclient import TestClient

from app import app
from features import CLASS_NAMES, wav_to_spectrogram

client = TestClient(app)

# A 5-second 440 Hz tone as the contents of a 16-bit WAV file
time_points = np.arange(44100 * 5) / 44100
tone = (0.5 * np.sin(2 * np.pi * 440 * time_points)).astype(np.float32)
tone_wav = tf.audio.encode_wav(tone[:, np.newaxis], 44100).numpy() # encode_wav expects the shape (samples, channels)


def test_spectrogram_has_the_expected_shape():
    assert wav_to_spectrogram(tone_wav).shape == (214, 64, 1)


def test_predict_returns_one_of_the_classes():
    response = client.post('/predict', files={'file': ('tone.wav', tone_wav, 'audio/wav')})
    assert response.status_code == 200
    assert response.json()['predicted_class'] in CLASS_NAMES
    assert len(response.json()['top_predictions']) == 3


def test_predict_rejects_files_that_are_not_wav():
    response = client.post('/predict', files={'file': ('notes.txt', b'this is not audio', 'text/plain')})
    assert response.status_code == 400
