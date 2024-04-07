import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from torchaudio.transforms import Resample
import numpy as np

def transcribe_audio_with_wav2vec(audio_data, sample_rate):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    if audio_data.size(0) > 1:
        audio_data = audio_data[0, :].unsqueeze(0)

    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        audio_data = resampler(audio_data)

    inputs = processor(audio_data.squeeze(), return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    return transcription

def capture_live_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating microphone... DONT TALK")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Calibration complete. SPEAK NOW!")
        audio = recognizer.listen(source)
        print("Recording stopped. Processing...")

    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    audio_data = torch.from_numpy(audio_data).float() / 32768.0
    audio_data = audio_data.unsqueeze(0) 

    return audio_data, audio.sample_rate

def main():
    audio_data, sample_rate = capture_live_audio()
    transcription = transcribe_audio_with_wav2vec(audio_data, sample_rate)
    print("Text:", transcription)

if __name__ == "__main__":
    main()
