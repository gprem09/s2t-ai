import librosa
import soundfile as sf

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    sf.write(file_path, y, sr)
    return y, sr

if __name__ == "__main__":
    preprocess_audio("/Users/gprem/Desktop/s2t-ai/dataset/wavs/LJ001-0001.wav")