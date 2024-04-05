import librosa
import pandas as pd
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments

