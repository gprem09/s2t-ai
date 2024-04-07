import pandas as pd
import os
from datasets import Dataset
from transformers import Wav2Vec2Processor
import soundfile as sf

def load_ljspeech_data(dataset_dir: str, processor: Wav2Vec2Processor, num_examples: int = None):
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path, sep='|', header=None, names=['filename', 'transcription', 'normalized_transcription'])

    # Optionally select a subset of the data
    if num_examples is not None:
        metadata = metadata.head(num_examples)

    def preprocess_function(examples):
        # print(f"Transcriptions: {examples['normalized_transcription'][:5]}")
        audio_files = [sf.read(os.path.join(dataset_dir, 'wavs', f"{file_name}.wav"))[0] for file_name in examples["filename"]]
        inputs = processor(audio_files, sampling_rate=16000, return_tensors="pt", padding=True)
        
        examples["input_values"] = inputs.input_values.tolist()
        examples["labels"] = [processor.tokenizer.encode(transcription) if transcription is not None else [] for transcription in examples.get("normalized_transcription", [])]
        return examples


    dataset = Dataset.from_pandas(metadata)
    dataset = dataset.map(preprocess_function, batched=True)
    
    return dataset
