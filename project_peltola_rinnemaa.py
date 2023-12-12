
import numpy as np
import librosa as lb
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix
import sounddevice as sd
import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

class AudioProcessor:
    def __init__(self, hop_length=256, frame_length=512):
        self.hop_length = hop_length
        self.frame_length = frame_length

    def read_file(self, file):
        try:
            audio, fs = lb.load(file, sr=None)
            return audio, fs
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return None, None

    def read_files_from_folder(self, folder_name):
        samples = []
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            if os.path.isfile(file_path):
                audio, fs = self.read_file(file_path)
                if audio is not None:
                    samples.append((audio, fs))
        return samples

    def normalize_sample(self, samples):
        normalized_samples = []
        for sample in samples:
            normalized = lb.util.normalize(sample[0])
            normalized_samples.append((normalized, sample[1]))
        return normalized_samples

    def get_energy(self, normalized_data):
        energy_data = []
        for audio in normalized_data:
            signal, sample_rate = audio
            energy = np.array([
                sum(abs(signal[i:i+self.frame_length]**2))
                for i in range(0, len(signal), self.hop_length)
            ])
            energy_data.append(energy)
        return energy_data
    
    def get_mfcc(self, normalized_data):
        mfccs_data = []

        for audio in normalized_data:

            signal = audio[0]
            sample_rate = audio[1]
            
            mfccs = lb.feature.mfcc(y=signal, sr=sample_rate)
            mfccs_data.append((mfccs, sample_rate))
        
        return mfccs_data
    
    def pad_mfccs(self, mfccs_data, pad_size):
        padded_data = []
        for mfccs, _ in mfccs_data:
            pad_width = max(0, pad_size - mfccs.shape[1])
            mfccs_padded = np.pad(mfccs, pad_width=((0,0), (0,pad_width)), mode='constant')
            padded_data.append(mfccs_padded.flatten()[:pad_size])
        
        return padded_data
    
    

# Example usage
processor = AudioProcessor()
train_bus = processor.read_files_from_folder('37007__aleksin__bussound')
train_tram = processor.read_files_from_folder('36822__hnminh__tram_class')
val_bus = processor.read_files_from_folder('39926__thespicychip__tampere_bus_audio_data')
val_tram = processor.read_files_from_folder('39927__thespicychip__tampere_tram_audio_data')
test_bus = processor.read_files_from_folder('bus_samples')
test_tram = processor.read_files_from_folder('tram_samples')

norm_train_bus = processor.normalize_sample(train_bus)
norm_train_tram = processor.normalize_sample(train_tram)
norm_val_bus = processor.normalize_sample(val_bus)
norm_val_tram = processor.normalize_sample(val_tram)
norm_test_bus = processor.normalize_sample(test_bus)
norm_test_tram = processor.normalize_sample(test_tram)

# Feature: energy

tram_energy = processor.get_energy(norm_test_tram)
bus_energy = processor.get_energy(norm_test_bus)

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

tram_rms = get_rms(norm_test_tram)
bus_rms = get_rms(norm_test_bus)

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


tram_spec, tram_logspec = get_spectrogram(norm_test_tram)
bus_spec, bus_logspec = get_spectrogram(norm_test_bus)

# Ei toimi nää plottaukset 

plot_spectrogram(tram_spec[6], sample_rate=test_tram[6][1])
plot_spectrogram(tram_logspec[6], sample_rate=test_tram[6][1])

plot_spectrogram(bus_spec[6], sample_rate=test_bus[6][1])
plot_spectrogram(bus_logspec[6], sample_rate=test_bus[6][1])


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


tram_melspec, tram_logmelspec = get_mel_spectrogram(norm_test_tram)
bus_melspec, bus_logmelspec = get_mel_spectrogram(norm_test_tram)


# Eikä nämäkään

plot_spectrogram(tram_melspec[6], sample_rate=test_tram[6][1])
plot_spectrogram(tram_logmelspec[6], sample_rate=test_tram[6][1])

plot_spectrogram(bus_melspec[6], sample_rate=test_bus[6][1])
plot_spectrogram(bus_logmelspec[6], sample_rate=test_bus[6][1])

tram_mfccs_test = processor.get_mfcc(norm_test_tram)
bus_mfccs_test = processor.get_mfcc(norm_test_bus)
tram_mfccs_train = processor.get_mfcc(norm_train_tram)
bus_mfccs_train = processor.get_mfcc(norm_train_bus)
tram_mfccs_val = processor.get_mfcc(norm_val_tram)
bus_mfccs_val = processor.get_mfcc(norm_val_bus)


plot_spectrogram(tram_mfccs_test[6][0], sample_rate=test_tram[6][1])
plot_spectrogram(bus_mfccs_test[6][0], sample_rate=test_bus[6][1])


# Feature: CQT spectrogram

def get_cqt(normalized_data):

    cqt_data = []

    for audio in normalized_data:

        signal = audio[0]
        sample_rate = audio[1]
        
        cqt_spectrogram = lb.feature.chroma_cqt(y=signal, sr=sample_rate)
        cqt_data.append(cqt_spectrogram)
    
    return cqt_data


tram_cqts = get_cqt(norm_test_tram)
bus_cqts = get_cqt(norm_test_bus)

# Eikä muuten tässäkään
plot_spectrogram(tram_cqts[6], sample_rate=test_tram[6][1])
plot_spectrogram(bus_cqts[6], sample_rate=test_bus[6][1])

# Determine the maximum size needed for padding
max_size = max(max(mfccs.shape[1] for mfccs, _ in tram_mfccs_test),
               max(mfccs.shape[1] for mfccs, _ in bus_mfccs_test))

padded_tram_mfccs_train = processor.pad_mfccs(tram_mfccs_train, max_size)
padded_bus_mfccs_train = processor.pad_mfccs(bus_mfccs_train, max_size)
padded_tram_mfccs_test = processor.pad_mfccs(tram_mfccs_test, max_size)
padded_bus_mfccs_test = processor.pad_mfccs(bus_mfccs_test, max_size)
padded_tram_mfccs_val = processor.pad_mfccs(tram_mfccs_val, max_size)
padded_bus_mfccs_val = processor.pad_mfccs(bus_mfccs_val, max_size)


# Labeling
def label_data(bus_data, tram_data, bus_label, tram_label):
    bus_labels = [bus_label] * len(bus_data)
    tram_labels = [tram_label] * len(tram_data)
    X = bus_data + tram_data
    y = bus_labels + tram_labels
    return X, y

# Label datasets
X_train, y_train = label_data(padded_bus_mfccs_train, padded_tram_mfccs_train, 1, 0)
X_val, y_val = label_data(padded_bus_mfccs_val, padded_tram_mfccs_val, 1, 0)
X_test, y_test = label_data(padded_bus_mfccs_test, padded_tram_mfccs_test, 1, 0)


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Validation evaluation
val_predictions = svm_classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_predictions).ravel()
val_precision_score = val_tp / (val_tp + val_fp)
val_recall_score = val_tp / (val_tp + val_fn)

print("Validation Accuracy: " + "{:.2f}".format(val_accuracy))
print("Validation Precision: " + "{:.2f}".format(val_precision_score))
print("Validation Recall score: " + "{:.2f}".format(val_recall_score))

# Evaluating on test set
test_predictions = svm_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
test_precision_score = tp / (tp + fp)
test_recall_score = tp / (tp + fn)

print("Test Accuracy: " + "{:.2f}".format(test_accuracy))
print("Test Precision: " + "{:.2f}".format(test_precision_score))
print("Test Recall score: " + "{:.2f}".format(test_recall_score))
