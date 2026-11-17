# Environmental Sound Classification with a CNN (ESC-10)

Small CNN in TensorFlow/Keras that classifies 10 environmental sounds from log-mel spectrograms, served through a FastAPI service that runs in Docker.
Personal project (2026).

## Results

- **Test accuracy is 0.7875** (63 of 80 clips) on fold 5 of the dataset, which is not used for training. Guessing would give 0.10.
- **Batch-normalisation momentum matters in a short training run.** With the Keras default (`momentum=0.99`), test accuracy is 0.7250 and train accuracy is 0.7562. With `momentum=0.9`, they are 0.7875 and 0.8812. Training has only about 400 steps, so the slowly updated running statistics of the default lag behind the weights.

## Method

- **Features (`features.py`):** each clip is cut or zero-padded to 5 seconds and turned into a 64-band log-mel spectrogram with TensorFlow's built-in audio functions.
- **Model (`train.py`):** three convolutional blocks (16, 32 and 64 filters), global average pooling, dropout and a softmax layer. Trained on folds 1 to 4 with Adam for 40 epochs with fixed random seeds.
- **API (`app.py`):** the trained model is served with FastAPI. `POST /predict` takes a 16-bit, 44.1 kHz WAV file and returns the three most likely classes.

## Usage

Developed with Python 3.13.5 and TensorFlow 2.21.0.

```bash
git clone https://github.com/fakhemm/esc10-sound-classifier.git
cd esc10-sound-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the API with the included model, then open http://127.0.0.1:8000/docs
uvicorn app:app

# Or run it in Docker
docker build -t esc10-sound-classifier .
docker run --rm -p 8000:8000 esc10-sound-classifier

# Reproduce the training (downloads about 600 MB)
curl -L -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip -q esc50.zip -d data
python train.py

# Run the tests
python -m pytest -q test_app.py
```

## Use of AI tools

The code was written with an AI coding assistant (Claude). I ran everything on my own machine and checked the results: the tests, the accuracy on the training and test clips, and the responses of the API inside the Docker container. The batch-normalisation problem under Results was found during these checks.

## Limitations

- **Small test set.** The result comes from a single test fold of 80 clips, so one clip is worth 1.25 percentage points.
- **No tuning.** The model and training settings are simple defaults. Only the batch-normalisation momentum was changed, and it was compared on the test fold, so the reported accuracy may be slightly optimistic.
- **Closed set of 10 classes.** Any sound is assigned to one of the 10 classes, including sounds that belong to none of them.

## Acknowledgements

- Dataset: ESC-10, from K. J. Piczak (2015), *ESC: Dataset for Environmental Sound Classification* ([GitHub](https://github.com/karolpiczak/ESC-50)), licensed under Creative Commons Attribution
