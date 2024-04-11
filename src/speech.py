import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import transformers
transformers.logging.set_verbosity_error()

class AudioTranscriber:
    def __init__(self, model_path, processor_path):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.eval()

    def transcribe(self, audio_data, sample_rate):
        if audio_data.size(0) > 1:
            audio_data = audio_data[0, :].unsqueeze(0)
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            audio_data = resampler(audio_data)
        inputs = self.processor(audio_data.squeeze(), return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
        return transcription.lower()

class LiveAudioCapture:    
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.recognizer = sr.Recognizer()

    def capture_and_transcribe(self):
        with sr.Microphone() as source:
            print("Checking for microphone... SILENCE!")
            self.recognizer.adjust_for_ambient_noise(source, duration=5)
            print("TALK!")
            while True:
                try:
                    audio = self.recognizer.listen(source)
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).copy()
                    audio_data = torch.from_numpy(audio_data).float() / 32768.0
                    audio_data = audio_data.unsqueeze(0)
                    transcription = self.transcriber.transcribe(audio_data, audio.sample_rate)
                    print(transcription)
                except Exception as e:
                    print(f"Error transcribing audio: {e}")

if __name__ == "__main__":
    model_path = '/Users/gprem/Desktop/s2t-ai/src/data/model'
    processor_path = '/Users/gprem/Desktop/s2t-ai/src/data/processor'
    transcriber = AudioTranscriber(model_path, processor_path)
    live_audio = LiveAudioCapture(transcriber)
    try:
        live_audio.capture_and_transcribe()
    except KeyboardInterrupt:
        print("\nExiting")
