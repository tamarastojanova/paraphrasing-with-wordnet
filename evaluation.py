from evaluate import load
import numpy as np


def evaluate_model(predictions, references, experiment_name, lang='en'):
    print(f" Evaluating Experiment: {experiment_name}")

    bertscore_metric = load('bertscore')
    bertscore_result = bertscore_metric.compute(predictions=predictions, references=references, lang=lang)
    mean_precision = np.mean(bertscore_result['precision'])
    mean_recall = np.mean(bertscore_result['recall'])
    mean_f1 = np.mean(bertscore_result['f1'])

    print("\n BERTScore Metrics ")
    print(f"Precision: {mean_precision} | Recall: {mean_recall} | F1: {mean_f1}")

    rouge_metric = load('rouge')
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)

    print("\n ROUGE Scores ")
    print("\n".join([f"{key}: {value}" for key, value in rouge_result.items()]))

    bleu_metric = load('bleu')
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)

    print("\n BLEU Score ")
    print("\n".join([f"{key}: {value}" for key, value in bleu_result.items()]))

    meteor_metric = load('meteor')
    meteor_result = meteor_metric.compute(predictions=predictions, references=references)

    print("\n METEOR Score ")
    print(f"meteor: {meteor_result['meteor']}")
