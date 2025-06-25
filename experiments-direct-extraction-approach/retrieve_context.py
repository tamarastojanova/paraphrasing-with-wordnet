from nltk import pos_tag, word_tokenize
from nltk.wsd import lesk
import random


def get_wordnet_context_for_sentence(sentence, context_combo):
    context_dict = {}
    keywords = get_keywords(sentence)

    synsets_cache = {}

    for word in keywords:
        best_synset = get_best_synset(word, sentence, synsets_cache)
        if best_synset:
            word_context = get_word_context(best_synset, context_combo)
            context_dict[word] = word_context

    return context_dict


def get_keywords(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    content_tags = {"NN", "NNS", "VB", "JJ", "JJS", "RB"}
    keywords = [word for word, tag in pos_tags if tag in content_tags]
    return keywords


def get_best_synset(word, sentence, synsets_cache):
    if word not in synsets_cache:
        best_synset = lesk(sentence.split(), word)
        synsets_cache[word] = best_synset
    return synsets_cache.get(word)


def get_word_context(synset, context_combo):
    word_context = {}
    for relation, include in context_combo.items():
        if include:
            word_context[relation] = process_relation(synset, relation)
    return word_context


def process_relation(synset, relation):
    relation_processors = {
        "synonyms": lambda syn: [lemma.name() for lemma in syn.lemmas()],
        "hypernyms": lambda syn: [hypernym.name() for hypernym in syn.hypernyms()],
        "hyponyms": lambda syn: [hyponym.name() for hyponym in syn.hyponyms()],
        "antonyms": lambda syn: [antonym.name() for lemma in syn.lemmas() for antonym in lemma.antonyms()],
        "meronyms": lambda syn: [meronym.name() for meronym in syn.part_meronyms() + syn.substance_meronyms()],
        "examples": lambda syn: {lemma: syn.examples() for lemma in [lemma.name() for lemma in syn.lemmas()]}
    }

    return relation_processors.get(relation, lambda syn: None)(synset)


def get_few_shots_array(dataset, number):
    number = min(number, len(dataset))

    random_indices = random.sample(range(len(dataset)), number)

    few_shots = []
    for idx in random_indices:
        original_sentence = dataset['Sentence 1'].iloc[idx]
        paraphrased_sentence = dataset['Sentence 2'].iloc[idx]
        few_shots.append((original_sentence, paraphrased_sentence))

    return few_shots
