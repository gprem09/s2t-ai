import torch
import torchaudio
import numpy as np
import speech_recognition as sr
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import transformers

transformers.logging.set_verbosity_error()

class AudioProcessor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()

    def _resample(self, audio_data, orig_sample_rate, target_sample_rate=16000):
        if orig_sample_rate != target_sample_rate:
            resampler = Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
            audio_data = resampler(audio_data)
        return audio_data

    def _preprocess_audio(self, audio_data, sample_rate):
        if audio_data.size(0) > 1:
            audio_data = audio_data[0, :].unsqueeze(0)
        audio_data = self._resample(audio_data, sample_rate)
        return audio_data

    def transcribe(self, audio_data, sample_rate):
        audio_data = self._preprocess_audio(audio_data, sample_rate)
        inputs = self.processor(audio_data.squeeze(), return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
        return transcription

class LiveAudioTranscriber:
    def __init__(self, audio_processor):
        self.recognizer = sr.Recognizer()
        self.audio_processor = audio_processor

    def capture_and_transcribe(self):
        with sr.Microphone() as source:
            print("Checking for microphone... QUIET!")
            self.recognizer.adjust_for_ambient_noise(source, duration=5)
            print("SPEAK!")
            while True:
                audio = self.recognizer.listen(source)
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).copy()
                audio_data = torch.from_numpy(audio_data).float() / 32768.0
                audio_data = audio_data.unsqueeze(0)
                transcription = self.audio_processor.transcribe(audio_data, audio.sample_rate)
                print("Text:", transcription.lower())

if __name__ == "__main__":
    try:
        audio_processor = AudioProcessor()
        transcriber = LiveAudioTranscriber(audio_processor)
        transcriber.capture_and_transcribe()
    except KeyboardInterrupt:
        print("\nExiting...")
