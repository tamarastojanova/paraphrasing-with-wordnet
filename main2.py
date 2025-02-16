import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import nltk

from combinations import *
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

login(token='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')

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
padding_size = 1024

tokenizer = AutoTokenizer.from_pretrained(f'{tokenizer_name}',
                                          use_auth_token=True,
                                          model_max_length=padding_size)
tokenizer.pad_token = tokenizer.bos_token
tokenizer.padding_side = 'left'

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16,
                                         bnb_4bit_quant_type='nf4')

model = AutoModelForCausalLM.from_pretrained(f'{model_name}',
                                             device_map='auto',
                                             quantization_config=quantization_config,
                                             offload_folder="./offload")

device = "cuda" if torch.cuda.is_available() else "cpu"
ground_truths = data_test['Sentence 2'].tolist()

# 3. LLM - без fine-tuning, со WordNet контекст

for combination_name, combination_settings in combinations1.items():
    print(f"Zero shot - with context - combinations {combination_name}\n")
    paraphrased_sentences = generate_paraphrases(format_prompt_with_context, data_test, tokenizer, model, device,
                                                 additional_args=combination_settings)
    evaluate_model(paraphrased_sentences, ground_truths, f"Zero shot - with context - combinations {combination_name}")

