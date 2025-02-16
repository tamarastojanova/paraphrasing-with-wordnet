from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_model_and_tokenizer():
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

    return model, tokenizer
