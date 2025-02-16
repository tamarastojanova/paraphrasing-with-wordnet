import pandas as pd
import torch
import nltk

from retrieve_context import *
from generate_paraphrases import *
from evaluation import *
from format_prompt import *
from model_tokenizer_load import *

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

