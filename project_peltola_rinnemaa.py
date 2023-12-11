import numpy as np
import librosa as lb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import sounddevice as sd
import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import os

# Reading audio data and storing samples in separate lists

def read_file(file):
    audio, fs = lb.load(file, sr=None)
    return audio, fs   

def get_bus_samples():
    bus_samp = []

    for filename in os.listdir('bus_samples'):
        file_path = os.path.join('bus_samples', filename)
        if os.path.isfile(file_path):
            audio, fs = read_file(file_path)
            bus_samp.append((audio, fs))

    return bus_samp

def get_tram_samples():
    tram_samp = []

    for filename in os.listdir('tram_samples'):
        file_path = os.path.join('tram_samples', filename)
        if os.path.isfile(file_path):
            audio, fs = read_file(file_path)
            tram_samp.append((audio, fs))
    return tram_samp


tram_samples = get_tram_samples()
bus_samples = get_bus_samples()

print(len(tram_samples))
print(len(bus_samples))


# Normalize the data

normalized_tram = []
normalized_bus = []

for sample in tram_samples:
    normalized = lb.util.normalize(sample[0])
    normalized_tram.append((normalized, sample[1]))

for sample in bus_samples:
    normalized = lb.util.normalize(sample[0])
    normalized_bus.append((normalized, sample[1]))


# Feature: energy

def get_energy(normalized_data):

    # En tiiä onks nää arvot hyvät/täytyyks olla ees hop_length
    hop_length = 256
    frame_length = 512
    win_size = int(0.1*44100)

    energy_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]

        energy = np.array([
            sum(abs(signal[i:i+frame_length]**2))
            for i in range(0, len(signal), hop_length)
        ])

        energy_data.append((energy, sample_rate))
    
    return energy_data

tram_energy = get_energy(normalized_tram)
bus_energy = get_energy(normalized_bus)

plt.figure()
plt.hist(tram_energy[2][0], bins=20, color='red', edgecolor='black', alpha=0.7, label='Tram')
plt.hist(bus_energy[2][0], bins=20, color='blue', edgecolor='black', alpha=0.7, label='Bus')
plt.legend()


plt.figure()
plt.hist(tram_energy[10][0], bins=20, color='red', edgecolor='black', alpha=0.7, label='Tram')
plt.hist(bus_energy[6][0], bins=20, color='blue', edgecolor='black', alpha=0.7, label='Bus')
plt.legend()


# Feature: RMS

def get_rms(normalized_data):

    rms_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]

        rms = lb.feature.rms(y=signal)
        rms_data.append((rms, sample_rate))

    return rms_data

tram_rms = get_rms(normalized_tram)
bus_rms = get_rms(normalized_bus)

plt.figure()
plt.hist(tram_rms[10][0].flatten(), bins=20, color='red', edgecolor='black', alpha=0.7, label='Tram')
plt.hist(bus_rms[6][0].flatten(), bins=20, color='blue', edgecolor='black', alpha=0.7, label='Bus')
plt.legend()



# For spectrograms

def plot_spectrogram(spec, sample_rate, if_truncate=False):
    sr = sample_rate
    plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(spec,origin='lower',aspect='auto')
    locs, labels = plt.xticks()
    locs_=[np.round((i/locs[-1]*len(spec)/sr),decimals=1) for i in locs]
    plt.xticks(locs[1:-1], locs_[1:-1])
    locs, labels = plt.yticks()
    if if_truncate:
      locs_=[int((i/locs[-1]*sr//16)) for i in locs]   # truncate the plot by a factor of 8
    else:
      locs_=[int((i/locs[-1]*sr//2)) for i in locs]
    plt.yticks(locs[1:-1], locs_[1:-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Fre (Hz)')



# Feature: spectrogram and log spectrogram

def get_spectrogram(normalized_data):

    spectrogram_data = []
    log_spectrogram_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]

        spectrogram = np.abs(lb.stft(signal, n_fft=512))
        spectrogram_data.append(spectrogram)

        log = 10 * np.log10(spectrogram)
        log_spectrogram_data.append(log)

    return spectrogram_data, log_spectrogram_data


tram_spec, tram_logspec = get_spectrogram(normalized_tram)
bus_spec, bus_logspec = get_spectrogram(normalized_bus)

# Ei toimi nää plottaukset 

plot_spectrogram(tram_spec[6], sample_rate=tram_samples[6][1])
plot_spectrogram(tram_logspec[6], sample_rate=tram_samples[6][1])

plot_spectrogram(bus_spec[6], sample_rate=bus_samples[6][1])
plot_spectrogram(bus_logspec[6], sample_rate=bus_samples[6][1])


# Feature: mel-spectrograms and logmel-spectrograms

def get_mel_spectrogram(normalized_data):

    mel_spectrogram_data = []
    logmel_spectrogram_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]

        mel_spectrogram = lb.feature.melspectrogram(y=signal, sr=sample_rate)
        mel_spectrogram_data.append(mel_spectrogram)

        logmel_spectrogram = 10*np.log10(mel_spectrogram)
        logmel_spectrogram_data.append(logmel_spectrogram)

    return mel_spectrogram_data, logmel_spectrogram_data


tram_melspec, tram_logmelspec = get_mel_spectrogram(normalized_tram)
bus_melspec, bus_logmelspec = get_mel_spectrogram(normalized_bus)


# Eikä nämäkään

plot_spectrogram(tram_melspec[6], sample_rate=tram_samples[6][1])
plot_spectrogram(tram_logmelspec[6], sample_rate=tram_samples[6][1])

plot_spectrogram(bus_melspec[6], sample_rate=bus_samples[6][1])
plot_spectrogram(bus_logmelspec[6], sample_rate=bus_samples[6][1])



# Feature: MFCC

def get_mfcc(normalized_data):

    mfccs_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]
        
        mfccs = lb.feature.mfcc(y=signal, sr=sample_rate)
        mfccs_data.append((mfccs, sample_rate))
    
    return mfccs_data


tram_mfccs = get_mfcc(normalized_tram)
bus_mfccs = get_mfcc(normalized_bus)

plot_spectrogram(tram_mfccs[6][0], sample_rate=tram_samples[6][1])
plot_spectrogram(bus_mfccs[6][0], sample_rate=bus_samples[6][1])


# Feature: CQT spectrogram

def get_cqt(normalized_data):

    cqt_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]
        
        cqt_spectrogram = lb.feature.chroma_cqt(y=signal, sr=sample_rate)
        cqt_data.append(cqt_spectrogram)
    
    return cqt_data


tram_cqts = get_cqt(normalized_tram)
bus_cqts = get_cqt(normalized_bus)

# Eikä muuten tässäkään
plot_spectrogram(tram_cqts[6], sample_rate=tram_samples[6][1])
plot_spectrogram(bus_cqts[6], sample_rate=bus_samples[6][1])



# Function to pad MFCCs to a fixed size
def pad_mfccs(mfccs_data, pad_size):
    padded_data = []
    for mfccs, _ in mfccs_data:
        pad_width = max(0, pad_size - mfccs.shape[1])
        mfccs_padded = np.pad(mfccs, pad_width=((0,0), (0,pad_width)), mode='constant')
        padded_data.append(mfccs_padded.flatten()[:pad_size])
    return padded_data

# Determine the maximum size needed for padding
max_size = max(max(mfccs.shape[1] for mfccs, _ in tram_mfccs),
               max(mfccs.shape[1] for mfccs, _ in bus_mfccs))

padded_tram_mfccs = pad_mfccs(tram_mfccs, max_size)
padded_bus_mfccs = pad_mfccs(bus_mfccs, max_size)

# Labeling
tram_labels = [0] * len(padded_tram_mfccs)
bus_labels = [1] * len(padded_bus_mfccs)


X = padded_bus_mfccs + padded_tram_mfccs  
y = tram_labels + bus_labels 

# Train SVM
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

predictions = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

