import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import transformers
transformers.logging.set_verbosity_error()

def transcribe_audio_with_wav2vec(audio_data, sample_rate):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    # if the audio has more than one channel
    if audio_data.size(0) > 1:
        audio_data = audio_data[0, :].unsqueeze(0)

    # the audio to 16 kHz
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        audio_data = resampler(audio_data)

    # process the audio
    inputs = processor(audio_data.squeeze(), return_tensors="pt", sampling_rate=16000)

    # perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # decode 
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    return transcription

def capture_live_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("checking for microphone... SILENT!")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("SPEAK!")
        while True:
            audio = recognizer.listen(source)
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).copy()
            audio_data = torch.from_numpy(audio_data).float() / 32768.0
            audio_data = audio_data.unsqueeze(0)

            transcription = transcribe_audio_with_wav2vec(audio_data, audio.sample_rate)
            print("s2t:", transcription.lower())

if __name__ == "__main__":
    try:
        capture_live_audio()
    except KeyboardInterrupt:
        print("\nexit")