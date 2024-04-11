import jiwer

def wer_cer(ground_truths, predictions):
    scores = {
        'wer': [],
        'cer': []
    }
    
    for gt, pred in zip(ground_truths, predictions):
        scores['wer'].append(jiwer.wer(gt, pred))
        scores['cer'].append(jiwer.cer(gt, pred))

    averages = {key: sum(value) / len(value) * 100 for key, value in scores.items()}
    print(f"scores: {averages}")
