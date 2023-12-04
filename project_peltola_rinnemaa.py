import numpy as np
import librosa as lb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Example classifier, use any desired classifier
import sounddevice as sd
import soundfile as sf

# 1. Data collection (Assuming you have a list of audio files)
# You might use a library like librosa to handle audio data.
audio1, fs1 = lb.load('Tallenna-001.wav', sr=None)
audio2, fs2 = lb.load('Tallenna-002.wav', sr=None)
audio3, fs3 = lb.load('Tallenna-003.wav', sr=None)
audio4, fs4 = lb.load('Tallenna-004.wav', sr=None)
audio5, fs5 = lb.load('Tallenna-005.wav', sr=None)

audio_files = [audio1, audio2, audio3, audio4, audio5]  # List of paths to audio files

print(audio_files)
# 2. Data conversion to wav file (If necessary, convert to WAV format)
# This step may not be needed if your audio files are already in WAV format.

# 3. Normalize the data
# Normalize audio data using librosa or other signal processing libraries.
normalized_data = []
for audio in audio_files:
    #data, sr = librosa.load(file)
    normalized = lb.util.normalize(audio)
    normalized_data.append(normalized)

print(normalized_data)



hop_length = 256
frame_length = 512

# Extracting time-domain and frequency-domain features
# Features: energy, RMS, spectrograms, log-spectrograms, mel-spectrograms, logmel-spectrograms, MFCCs, CQT spectrograms

energy_data = []
rms_data = []
spectrogram_data = []
log_spectrogram_data = []
mel_spectrogram_data = []
logmel_spectrogram_data = []
mfccs_data = []
CQT_spectrogram_data = []

for audio in normalized_data:

    # Energy
    energy = np.array([
        sum(abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio), hop_length)
    ])

    energy_data.append(energy)

    # RMS
    rms = lb.feature.rms(y=audio)

    rms_data.append(rms)

    # Spectrogram

"""

# 5. Extract frequency-domain features
# Use Fourier Transform to convert data to frequency domain and extract features.
frequency_features = []
for data in normalized_data:
    # Apply Fourier Transform and extract frequency-domain features
    # Append features to the frequency_features list.
    stft = np.abs(librosa.stft(data))

    # Extracting frequency-domain features (e.g., spectral centroid, bandwidth)
    spectral_centroids = librosa.feature.spectral_centroid(S=stft).ravel()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft).ravel()

    # Append extracted features to the frequency_features list
    frequency_features.append([spectral_centroids, spectral_bandwidth])

# 6. Plot spectrograms and MFCC from some of the files and compare them
# Use matplotlib or other plotting libraries to visualize spectrograms and MFCC.
# This requires computing Mel-frequency cepstral coefficients (MFCC) and spectrograms.
# Compare and plot them to observe differences.

# 7. Implement the classifier (Choose a machine learning model)


# Prepare feature matrix
X = []  # Combine time and frequency features into X
y = [...]  # Labels for your data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = SVC()
classifier.fit(X_train, y_train)

# 8. Train the model (Already part of the previous step)

# 9. Calculate results and analyze them
# Use the trained model to predict on test data and analyze results
predictions = classifier.predict(X_test)
# Perform evaluation, calculate accuracy, confusion matrix, etc.

# 10. Write report
# Summarize your findings, results, and analysis in a report format using markdown or a document editor.

"""