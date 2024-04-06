import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AdamW
from torch.utils.data import DataLoader
from dataset import load_ljspeech_data 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model.to(device)

dataset_dir = "/Users/gprem/Desktop/s2t-ai/dataset"
dataset = load_ljspeech_data(dataset_dir, processor)

train_dataset = dataset["train"]

def collate_fn(batch):
    input_values = torch.tensor([item['input_values'] for item in batch], dtype=torch.float).to(device)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long).to(device)
    return {"input_values": input_values, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_values=input_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.5f}")

model.save_pretrained("./wav2vec2_finetuned")
processor.save_pretrained("./wav2vec2_finetuned_processor")
