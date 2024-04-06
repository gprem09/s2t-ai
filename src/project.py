import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
import librosa

model_path = '/Users/gprem/Desktop/s2t-ai/src/default.pt'
audio_input = '/Users/gprem/Desktop/s2t-ai/dataset/wavs/LJ001-0001.wav'

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC(config)

model.load_state_dict(torch.load(model_path))
model.eval()

audio, _ = librosa.load(audio_input, sr=16000)
input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print(transcription)
