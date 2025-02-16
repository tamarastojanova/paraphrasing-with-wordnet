from retrieve_context import get_wordnet_context_for_sentence


def format_prompt_with_context(
        sentence,
        combination,
        few_shots=None
):
    prompt = ""

    if few_shots:
        prompt += "Here are some examples:\n"
        for idx, (original, paraphrased) in enumerate(few_shots, start=1):
            prompt += (
                f"{idx}. Original sentence: '{original}'\n"
                f"Paraphrased sentence: '{paraphrased}'\n\n"
            )

    context_dict = get_wordnet_context_for_sentence(sentence, combination)

    prompt += "Relevant context:\n"

    for word, context in context_dict.items():
        prompt += f"Word: {word}\n"

        for relation, values in context.items():
            if values:
                if relation == "examples":
                    example_list = [f"  - {synonym}: {' | '.join(example_list) if example_list else 'None'}"
                                    for synonym, example_list in values.items() if example_list]
                    if example_list:
                        prompt += "- Example Sentences by Synonyms:\n" + "\n".join(example_list) + "\n"
                else:
                    prompt += f"- {relation.capitalize()}: {', '.join(values) if values else 'None'}\n"

    prompt += (
        f"Your task is to paraphrase the following sentence: {sentence}. "
        "Provide the paraphrased sentence enclosed in <ANSWER></ANSWER> tags."
        "Do not include any explanations, reasoning, or additional text. "
    )

    return prompt


def format_prompt_without_context(sentence, additional_args=None, few_shots=None):
    prompt = ""
    if few_shots:
        prompt += "Here are some examples:\n"
        for idx, (original, paraphrased) in enumerate(few_shots, start=1):
            prompt += (
                f"{idx}. Original sentence: '{original}'\n"
                f"Paraphrased sentence: '{paraphrased}'\n\n"
            )

    prompt += (
        f"Your task is to paraphrase the following sentence: {sentence}. "
        "Provide the paraphrased sentence enclosed in <ANSWER></ANSWER> tags."
        "Do not include any explanations, reasoning, or additional text. "
    )

    return prompt
