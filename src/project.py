import torch
import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from pathlib import Path
from tqdm import tqdm
import json
transformers.logging.set_verbosity_error()

def transcribe_and_save(audio_dir, metadata_path, output_file):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    ground_truths = {}
    predictions = {}

    with open(metadata_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            filename = parts[0]
            transcript = parts[2]
            ground_truths[filename] = transcript

    for filename, transcript in tqdm(ground_truths.items(), desc="Transcribing", unit="file"):
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

        predictions[filename] = transcription

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("Transcriptions saved to:", output_file)

audio_dir = Path('/Users/gprem/Desktop/s2t-ai/dataset/wavs')
metadata_path = '/Users/gprem/Desktop/s2t-ai/dataset/metadata.csv'
output_file = 'transcriptions.json'

transcribe_and_save(audio_dir, metadata_path, output_file)
