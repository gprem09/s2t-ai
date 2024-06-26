import json
from compute_bleu import bleu
from compute_error import wer_cer

def load_data_and_score(metadata_path, transcriptions_file):
    with open(metadata_path, 'r', encoding='utf-8') as file:
        ground_truths = {line.split('|')[0]: line.split('|')[2].strip() for line in file}

    with open(transcriptions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    gt_list = [ground_truths[filename] for filename in predictions if filename in ground_truths]
    pred_list = [predictions[filename] for filename in predictions if filename in ground_truths]

    bleu(gt_list, pred_list)
    wer_cer(gt_list, pred_list)

metadata_path = '/Users/gprem/Desktop/s2t-ai/dataset/metadata.csv'
transcriptions_file = 'transcriptions_finetuned.json'

load_data_and_score(metadata_path, transcriptions_file)
