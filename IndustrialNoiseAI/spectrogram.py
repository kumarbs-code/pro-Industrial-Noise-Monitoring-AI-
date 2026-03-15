import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path):

    y, sr = librosa.load(audio_path, duration=3)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(librosa.power_to_db(spectrogram), sr=sr)
    plt.axis('off')

    image_path = "uploads/spec.png"
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return image_path