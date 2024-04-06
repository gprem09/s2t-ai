import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize

def compute_bleu(ground_truths, predictions):
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        reference_tokens = word_tokenize(gt)
        candidate_tokens = word_tokenize(pred)

        score = sentence_bleu([reference_tokens], candidate_tokens)
        scores.append(score)
        # print(f"sentence_bleu: {score * 100:.2f}")

    average_bleu_score = np.mean(scores) * 100
    print(f"BLEU Score: {average_bleu_score:.2f}")
