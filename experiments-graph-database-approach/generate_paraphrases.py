import re
from format_prompt import *


def process_output(response):
    response = re.sub(r'["\']', '', response)

    match = re.search(r'<ANSWER>(.*?)</ANSWER>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def generate_paraphrases(data, tokenizer, model, device, combination, similarity_query,
                         driver, transformer, max_new_tokens=100, temperature=0.2):
    paraphrased_sentences = []

    for sentence in data['Sentence 1']:
        formatted_prompt = format_prompt_with_context(sentence, combination, similarity_query, driver, transformer)

        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                                temperature=temperature, do_sample=True)

        generated_text = output[:, input_ids.shape[-1]:]
        response = tokenizer.decode(generated_text[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response = process_output(response)
        print(response)
        paraphrased_sentences.append(response)

    return paraphrased_sentences
