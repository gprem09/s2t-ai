import torch
from torch.utils.data import DataLoader, random_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AdamW
from dataset import load_ljspeech_data  
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

dataset_dir = "/Users/gprem/Desktop/s2t-ai/dataset"
model_name = "facebook/wav2vec2-large-960h-lv60-self"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 3
learning_rate = 1e-5

# Load processor and dataset
processor = Wav2Vec2Processor.from_pretrained(model_name)
dataset = load_ljspeech_data(dataset_dir, processor, num_examples=3600)

subset_size = 3600
subset_dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

def collate_fn(batch):
    input_values = [torch.tensor(item['input_values']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=processor.tokenizer.pad_token_id).to(device)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100).to(device) 
    return {"input_values": input_values_padded, "labels": labels_padded}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        input_values = batch["input_values"]
        labels = batch["labels"]
        outputs = model(input_values, labels=labels).loss  
        loss = outputs
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.5f}")

model.save_pretrained("./wav2vec2_finetuned_model")
processor.save_pretrained("./wav2vec2_finetuned_processor")
