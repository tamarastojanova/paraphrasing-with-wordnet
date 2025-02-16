import pandas as pd
import torch
import nltk

from combinations import *
from model_tokenizer_load import load_model_and_tokenizer
from retrieve_context import *
from generate_paraphrases import *
from evaluation import *
from format_prompt import *

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

data_train = pd.read_csv("datasets/train.txt", sep="\t")
data_val = pd.read_csv("datasets/val.txt", sep="\t")
data_test = pd.read_csv("datasets/test.txt", sep="\t")

from huggingface_hub import login

login(token='MY TOKEN HERE')

model, tokenizer = load_model_and_tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"
ground_truths = data_test['Sentence 2'].tolist()

# 4. LLM - без fine-tuning, со WordNet контекст и few shots

few_shots = get_few_shots_array(data_val, 3)
for combination_name, combination_settings in combinations2.items():
    print(f"Few shots - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings, few_shots=few_shots)
    evaluate_model(paraphrased_sentences, ground_truths, f"Few shots - with context - combinations {combination_name}")
