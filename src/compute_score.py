from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
import jiwer

def score(ground_truths, predictions):
    scores = {
        'bleu': [],
        'unigram': [],
        'bigram': [],
        'trigram': [],
        'wer': [],
        'cer': []
    }
    smoothing = SmoothingFunction().method1
    
    for gt, pred in zip(ground_truths, predictions):
        reference_tokens = [word_tokenize(gt)]
        candidate_tokens = word_tokenize(pred)
        
        # Calculate BLEU scores
        scores['bleu'].append(sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing))
        scores['unigram'].append(sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing))
        scores['bigram'].append(sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
        scores['trigram'].append(sentence_bleu(reference_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing))

        # Calculate WER and CER
        scores['wer'].append(jiwer.wer(gt, pred))
        scores['cer'].append(jiwer.cer(gt, pred))

    averages = {key: sum(value) / len(value) * 100 for key, value in scores.items()}
    print(f"scores: {averages}")