import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from pathlib import Path
from tqdm import tqdm
import json
import transformers
import re
transformers.logging.set_verbosity_error()

finetuned_model = '/Users/gprem/Desktop/s2t-ai/src/data/model'
finetuned_processor = '/Users/gprem/Desktop/s2t-ai/src/data/processor'

def sentence_correction(text):
    text = text.strip().lower()
    def capitalize_sentence(m):
        return m.group(0).capitalize()
    text = re.sub(r'(?:^|(?<=\.\s))(?P<first_letter>[a-z])', capitalize_sentence, text)
    text = re.sub(r'(?<!\.)\Z', '.', text)
    return text

def transcribe_and_save(audio_dir, metadata_path, output_file):
    processor = Wav2Vec2Processor.from_pretrained(finetuned_processor)
    model = Wav2Vec2ForCTC.from_pretrained(finetuned_model)
    model.eval()

    ground_truths = {}
    predictions = {}

    with open(metadata_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            filename = parts[0]
            transcript = parts[2]
            ground_truths[filename] = sentence_correction(transcript)

    for filename, transcript in tqdm(list(ground_truths.items())[:50], desc="Transcribing", unit="file"):
        audio_path = audio_dir / f"{filename}.wav"
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        predictions[filename] = sentence_correction(transcription)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("Transcriptions saved to:", output_file)

audio_dir = Path('/Users/gprem/Desktop/s2t-ai/dataset/wavs')
metadata_path = '/Users/gprem/Desktop/s2t-ai/dataset/metadata.csv'
output_file = 'transcriptions_finetuned.json'

transcribe_and_save(audio_dir, metadata_path, output_file)
