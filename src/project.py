import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
import librosa
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import numpy as np

# nltk.download('punkt')

def load_ground_truths(metadata_path):
   ground_truths = {}
   with open(metadata_path, 'r', encoding='utf-8') as file:
       for line in file:
           parts = line.strip().split('|')
           filename = parts[0]
           transcript = parts[2]
           ground_truths[filename] = transcript
   return ground_truths

def compute_bleu(model_path, audio_dir, metadata_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    ground_truths = load_ground_truths(metadata_path)
    scores = []

    for filename, transcript in tqdm(ground_truths.items(), desc="Transcribing", unit="file"): 
        audio_path = audio_dir / f"{filename}.wav"
        audio, _ = librosa.load(audio_path, sr=16000)
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        reference_tokens = word_tokenize(transcript.lower())
        candidate_tokens = word_tokenize(transcription.lower())

        score = sentence_bleu([reference_tokens], candidate_tokens)
        scores.append(score)
    
    average_bleu_score = np.mean(scores)
    return average_bleu_score

model_path = '/Users/gprem/Desktop/s2t-ai/src/data/default.pt'
audio_dir = Path('/Users/gprem/Desktop/s2t-ai/dataset/wavs')
metadata_path = '/Users/gprem/Desktop/s2t-ai/dataset/metadata.csv'

average_bleu_score = compute_bleu(model_path, audio_dir, metadata_path)
print(f"BLEU Score: {average_bleu_score}")
