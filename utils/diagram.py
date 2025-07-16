import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


ravdess_female = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\dataset\\ravdes\\audio_speech_actors_01-24\\Actor_02\\03-01-01-01-01-01-02.wav"
ravdess_male = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\dataset\\ravdes\\audio_speech_actors_01-24\\Actor_03\\03-01-01-01-01-01-03.wav"
emodb_female = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\dataset\\emodb\\wav\\09b03Fd.wav"
emodb_male = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\dataset\\emodb\\wav\\10a04Wa.wav"
emovo_female = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\dataset\\EMOVO\\f1\\dis-f1-b3.wav"
emovo_male = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\dataset\\EMOVO\\m1\\dis-m1-l1.wav"

output_folder = "C:\\Users\\Aruay\\Desktop\\ra application\\project\\diagram\\sample_waveform.eps"


fig, axes = plt.subplots(6, 1, figsize=(10, 15))
axes = axes.flatten()

signal, sr = librosa.load(ravdess_male, sr=None)
librosa.display.waveshow(signal, sr=sr, ax=axes[0])
axes[0].set_title(f"RAVDESS, male")
axes[0].set_xlabel("")

signal, sr = librosa.load(ravdess_female, sr=None)
librosa.display.waveshow(signal, sr=sr, ax=axes[1])
axes[1].set_title(f"RAVDESS, female")
axes[1].set_xlabel("")

signal, sr = librosa.load(emodb_male, sr=None)
librosa.display.waveshow(signal, sr=sr, ax=axes[2])
axes[2].set_title(f"EMO-DB, male")
axes[2].set_xlabel("")

signal, sr = librosa.load(emodb_female, sr=None)
librosa.display.waveshow(signal, sr=sr, ax=axes[3])
axes[3].set_title(f"EMO-DB, female")
axes[3].set_xlabel("")

signal, sr = librosa.load(emovo_male, sr=None)
librosa.display.waveshow(signal, sr=sr, ax=axes[4])
axes[4].set_title(f"EMOVO, male")
axes[4].set_xlabel("")

signal, sr = librosa.load(emovo_female, sr=None)
librosa.display.waveshow(signal, sr=sr, ax=axes[5])
axes[5].set_title(f"EMOVO, female")
axes[5].set_xlabel("Time (seconds)")

    
plt.subplots_adjust(hspace=0.7)
plt.savefig(output_folder, bbox_inches='tight')
plt.close()
