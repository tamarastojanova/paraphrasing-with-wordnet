from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch


def load_model_and_tokenizer():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'
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

    transformer = SentenceTransformer("all-MiniLM-L6-v2")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer = transformer.to(device)

    return model, transformer, tokenizer
