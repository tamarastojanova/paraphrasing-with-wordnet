from retrieve_context import get_wordnet_context_for_sentence


def format_prompt_with_context(
        sentence,
        combination,
        few_shots=None
):
    prompt = (
        "Please paraphrase the sentence provided below while ensuring that the meaning remains unchanged. "
        "The paraphrased sentence should be grammatically correct and enclosed in <ANSWER></ANSWER> tags. "
        "Do not add any explanations or extra text.\n\n"
    )

    if few_shots:
        prompt += "Here are some examples to guide you:\n"
        for idx, (original, paraphrased) in enumerate(few_shots, start=1):
            prompt += (
                f"{idx}. Example: "
                f"Original sentence: '{original}', "
                f"Paraphrased sentence: <ANSWER>{paraphrased}</ANSWER>\n"
            )

    context_dict = get_wordnet_context_for_sentence(sentence, combination)

    prompt += "Relevant context to help you paraphrase:\n"

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
        f"Original sentence: '{sentence}'\n"
        "Paraphrased sentence:"
    )

    return prompt


def format_prompt_without_context(sentence, additional_args=None, few_shots=None):
    prompt = (
        "Please paraphrase the sentence provided below while ensuring that the meaning remains unchanged. "
        "The paraphrased sentence should be grammatically correct and enclosed in <ANSWER></ANSWER> tags. "
        "Do not add any explanations or extra text.\n\n"
    )

    if few_shots:
        prompt += "Here are some examples to guide you:\n"
        for idx, (original, paraphrased) in enumerate(few_shots, start=1):
            prompt += (
                f"{idx}. Example: "
                f"Original sentence: '{original}', "
                f"Paraphrased sentence: <ANSWER>{paraphrased}</ANSWER>\n"
            )

    prompt += (
        f"Original sentence: '{sentence}'\n"
        "Paraphrased sentence:"
    )

    return prompt
