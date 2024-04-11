import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import transformers
import jiwer
import sacrebleu
import re
transformers.logging.set_verbosity_error()

class AudioTranscriber:
    def __init__(self, model_path, processor_path):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.eval()

    def correct_sentence(self, text):
        text = text.strip().lower()

        def capitalize_sentence(m):
            return m.group(0).capitalize()

        text = re.sub(r'(?:^|(?<=\.\s))(?P<first_letter>[a-z])', capitalize_sentence, text)
        text = re.sub(r'(?<!\.)\Z', '.', text)
        return text

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
        corrected_transcription = self.correct_sentence(transcription)
        return corrected_transcription

class LiveAudioCapture:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.recognizer = sr.Recognizer()

    def capture_and_transcribe(self):
        with sr.Microphone() as source:
            print("Checking for microphone... Please remain silent!")
            self.recognizer.adjust_for_ambient_noise(source, duration=5)
            print("Speak now!")
            while True:
                audio = self.recognizer.listen(source)
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).copy()
                audio_data = torch.from_numpy(audio_data).float() / 32768.0
                audio_data = audio_data.unsqueeze(0)
                transcription = self.transcriber.transcribe(audio_data, audio.sample_rate)
                print("Transcription:", transcription)
                # ground_truth = "He kept telling himself that one day it would all somehow make sense."
                # bleu = sacrebleu.raw_corpus_bleu([transcription], [[ground_truth]], .01).score
                # print(f"BLEU Score: {bleu:.2f}")
                # error = jiwer.wer(ground_truth, transcription)
                # print(f"Word Error Rate: {error:.2%}")
                # cer = jiwer.cer(ground_truth, transcription)
                # print(f"Character Error Rate: {cer:.2%}")
                

if __name__ == "__main__":
    model_path = '/Users/gprem/Desktop/s2t-ai/src/data/model'
    processor_path = '/Users/gprem/Desktop/s2t-ai/src/data/processor'
    transcriber = AudioTranscriber(model_path, processor_path)
    live_audio = LiveAudioCapture(transcriber)
    try:
        live_audio.capture_and_transcribe()
    except KeyboardInterrupt:
        print("\nExiting")
