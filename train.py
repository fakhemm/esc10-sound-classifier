# Trains a small CNN on ESC-10 (10 environmental sound classes) and saves it for the API
# Folds 1 to 4 of the dataset are used for training and fold 5 for testing

# For data handling
import pandas as pd
import numpy as np

# For reading the audio files and building the neural network
import tensorflow as tf
import keras

# For turning WAV files into log-mel spectrograms
from features import CLASS_NAMES, wav_to_spectrogram


# Set key experiment configuration
CONFIG = {
    'data_dir': 'data/ESC-50-master', # Make sure the unzipped ESC-50 download is in this folder
    'test_fold': 5,
    'random_state': 42,
    'num_epochs': 40,
    'batch_size': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.3,
    'bn_momentum': 0.9 # Keras' default (0.99) updates the running statistics too slowly for only about 400 training steps
}

print("--- Environmental Sound Classification on ESC-10 with a Small CNN ---")

# Fix the random seeds of Python, NumPy and TensorFlow so results are reproducible
keras.utils.set_random_seed(CONFIG['random_state'])


# Part 1: Load Data and Compute Spectrograms

print("\nPart 1: Loading audio and computing spectrograms...")

# Load the metadata file and keep only the clips of the 10-class ESC-10 subset
df = pd.read_csv(f"{CONFIG['data_dir']}/meta/esc50.csv")
df = df[df['esc10']] # The column esc10 is True for these clips

# Turn every clip into a log-mel spectrogram (the "image" the CNN sees) and every class name into an integer label
X = np.stack([wav_to_spectrogram(tf.io.read_file(f"{CONFIG['data_dir']}/audio/{filename}")) for filename in df['filename']])
y = df['category'].map(CLASS_NAMES.index).to_numpy()

# Split by the dataset's official folds, which keep clips from the same recording together
is_test = (df['fold'] == CONFIG['test_fold']).to_numpy()
X_train, y_train = X[~is_test], y[~is_test]
X_test, y_test = X[is_test], y[is_test]

# Checks
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# Part 2: Train the Model

print("\nPart 2: Training the model...")

# Build a small CNN: three convolutional blocks, global average pooling, and one dense output layer
model = keras.Sequential([keras.Input(shape=X.shape[1:])])
for num_filters in (16, 32, 64):
    model.add(keras.layers.Conv2D(num_filters, 3, padding='same', use_bias=False)) # use_bias=False since batch normalisation follows
    model.add(keras.layers.BatchNormalization(momentum=CONFIG['bn_momentum']))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(2)) # Halves the height and width
model.add(keras.layers.GlobalAveragePooling2D()) # Averages each feature map into a single number
model.add(keras.layers.Dropout(CONFIG['dropout_rate']))
model.add(keras.layers.Dense(len(CLASS_NAMES), activation='softmax'))
model.compile(
    optimizer=keras.optimizers.Adam(CONFIG['learning_rate']),
    loss='sparse_categorical_crossentropy', # The labels are integers, not one-hot vectors
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'], verbose=0) # verbose=0 hides the per-epoch log


# Part 3: Evaluate and Save the Model

print("\nPart 3: Evaluating on the test fold...")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1) # argmax picks the class with the highest probability
y_pred_train = model.predict(X_train, verbose=0).argmax(axis=1)

print(f"Test accuracy: {(y_pred == y_test).mean():.4f} ({(y_pred == y_test).sum()} of {len(y_test)} clips)")
print(f"Train accuracy: {(y_pred_train == y_train).mean():.4f}")

model.save("model.keras")
print("\nModel saved to 'model.keras'")
