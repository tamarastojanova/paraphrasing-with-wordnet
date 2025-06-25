import pandas as pd
import torch
import nltk

from retrieve_context import *
from combinations import *
from generate_paraphrases import *
from evaluation import *
from format_prompt import *
from load_models_and_tokenizer import *

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

data_train = pd.read_csv("../datasets/train.txt", sep="\t")
data_val = pd.read_csv("../datasets/val.txt", sep="\t")
data_test = pd.read_csv("../datasets/test.txt", sep="\t")

from huggingface_hub import login

login(token='MY TOKEN HERE')

model, _, tokenizer = load_model_and_tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"
ground_truths = data_test['Sentence 2'].tolist()

# 1. LLM - без fine-tuning, без WordNet контекст

print("Zero shot - no context\n")
paraphrased_sentences_no_context = generate_paraphrases(format_prompt_without_context, data_test, tokenizer, model,
                                                        device)
evaluate_model(paraphrased_sentences_no_context, ground_truths, "Zero shot - no context")

# 2. LLM - без fine-tuning, без WordNet контекст, со few-shots

few_shots = get_few_shots_array(data_val, 2)
print("Few shots - no context\n")
paraphrased_sentences_no_context_fs = generate_paraphrases(format_prompt_without_context, data_test, tokenizer, model,
                                                           device, few_shots=few_shots)
evaluate_model(paraphrased_sentences_no_context_fs, ground_truths, "Few shots - no context")

# 3. LLM - без fine-tuning, со WordNet контекст

for combination_name, combination_settings in combinations1.items():
    print(f"Zero shot - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings)
    evaluate_model(paraphrased_sentences, ground_truths, f"Zero shot - with context - combinations {combination_name}")

for combination_name, combination_settings in combinations2.items():
    print(f"Zero shot - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings)
    evaluate_model(paraphrased_sentences, ground_truths, f"Zero shot - with context - combinations {combination_name}")

for combination_name, combination_settings in combinations3.items():
    print(f"Zero shot - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings)
    evaluate_model(paraphrased_sentences, ground_truths, f"Zero shot - with context - combinations {combination_name}")

# 4. LLM - без fine-tuning, со WordNet контекст и few shots

for combination_name, combination_settings in combinations1.items():
    print(f"Few shots - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings, few_shots=few_shots)
    evaluate_model(paraphrased_sentences, ground_truths, f"Few shots - with context - combinations {combination_name}")

for combination_name, combination_settings in combinations2.items():
    print(f"Few shots - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings, few_shots=few_shots)
    evaluate_model(paraphrased_sentences, ground_truths, f"Few shots - with context - combinations {combination_name}")

for combination_name, combination_settings in combinations3.items():
    print(f"Few shots - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings, few_shots=few_shots)
    evaluate_model(paraphrased_sentences, ground_truths, f"Few shots - with context - combinations {combination_name}")



