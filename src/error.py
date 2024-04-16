import json
from compute_error import wer_cer
import os

csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'metadata.csv')

def load_data_and_compute_bleu(metadata_path, transcriptions_file):
    with open(metadata_path, 'r', encoding='utf-8') as file:
        ground_truths = {line.split('|')[0]: line.split('|')[2].strip() for line in file}

    with open(transcriptions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    gt_list = [ground_truths[filename] for filename in predictions if filename in ground_truths]
    pred_list = [predictions[filename] for filename in predictions if filename in ground_truths]

    wer_cer(gt_list, pred_list)

metadata_path = csv_file_path
transcriptions_file = 'transcriptions_finetuned.json'

load_data_and_compute_bleu(metadata_path, transcriptions_file)