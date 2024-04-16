import torch
import re
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pathlib import Path
from tqdm import tqdm
import librosa
import transformers
transformers.logging.set_verbosity_error()
import os

csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'metadata.csv')
audio_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'wavs')
model_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'model')
processor_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processor')

class TextFormatter:
    @staticmethod
    def correct_sentence(text):
        text = text.strip().lower()

        def capitalize_sentence(m):
            return m.group(0).capitalize()

        text = re.sub(r'(?:^|(?<=\.\s))(?P<first_letter>[a-z])', capitalize_sentence, text)
        text = re.sub(r'(?<!\.)\Z', '.', text)
        return text

class Transcriber:
    def __init__(self, model_path, processor_path):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.eval()

    def transcribe(self, audio_dir, metadata_path, output_file):
        ground_truths = {}
        predictions = {}

        with open(metadata_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                filename = parts[0]
                transcript = parts[2]
                ground_truths[filename] = TextFormatter.correct_sentence(transcript)

        for filename, transcript in tqdm(list(ground_truths.items())[:5], desc="Transcribing", unit="file"):
            audio_path = audio_dir / f"{filename}.wav"
            try:
                audio, _ = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            input_values = self.processor(audio, return_tensors="pt", sampling_rate=16000).input_values
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            predictions[filename] = TextFormatter.correct_sentence(transcription)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)

        print("Transcriptions saved to:", output_file)

if __name__ == "__main__":
    model_path = model_file_path
    processor_path = processor_file_path
    audio_dir = Path(audio_file_path)
    metadata_path = csv_file_path
    output_file = 'transcriptions_finetuned.json'
    
    transcriber = Transcriber(model_path, processor_path)
    transcriber.transcribe(audio_dir, metadata_path, output_file)