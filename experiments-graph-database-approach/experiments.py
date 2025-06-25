from neo4j import GraphDatabase
import nltk
import torch
import spacy
import requests
import json
from requests.auth import HTTPBasicAuth
import spacy
import pandas as pd
import random

from load_models_and_tokenizer import *
from generate_paraphrases import *
from combinations import *
from cypher_functions import *
from evaluation import *

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

data_train = pd.read_csv("../datasets/train.txt", sep="\t")
data_val = pd.read_csv("../datasets/val.txt", sep="\t")

from huggingface_hub import login

login(token='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')

URI = "neo4j+s://95d45b97.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "SP-oIiL9DXrLFJ3mpdjfDAYjDOAer6zDiWPSv-N4D9w"
QUERY_API_URL = "https://95d45b97.databases.neo4j.io/db/{databaseName}/query/v2"
HEADERS = {
    "Content-Type": "application/json"
}
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, transformer, tokenizer = load_model_and_tokenizer()
paraphrased_sentences_synonyms_hypernyms = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['synonyms_hypernyms'],
    run_similarity_query_wo_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_synonyms_hypernyms, data_train['Sentence 2'],
               "Experimenting with synonyms and hypernyms")

paraphrased_sentences_synonyms_examples = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['synonyms_examples'],
    run_similarity_query_w_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_synonyms_examples, data_train['Sentence 2'],
               "Experimenting with synonyms and examples")

paraphrased_sentences_hypernyms_hyponyms = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['hypernyms_hyponyms'],
    run_similarity_query_wo_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_hypernyms_hyponyms, data_train['Sentence 2'],
               "Experimenting with hypernyms and hyponyms")

paraphrased_sentences_synonyms_antonyms = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['synonyms_antonyms'],
    run_similarity_query_wo_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_synonyms_antonyms, data_train['Sentence 2'],
               "Experimenting with synonyms and antonyms")

paraphrased_sentences_hypernyms_synonyms_examples = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['hypernyms_synonyms_examples'],
    run_similarity_query_w_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_hypernyms_synonyms_examples, data_train['Sentence 2'],
               "Experimenting with synonyms, hypernyms and examples")

paraphrased_sentences_synhypoexampl = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['synonyms_hyponyms_examples'],
    run_similarity_query_w_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_synhypoexampl, data_train['Sentence 2'],
               "Experimenting with synonyms, hyponyms and examples")

paraphrased_sentences_hypernyms_hyponyms_synonyms = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['hypernyms_hyponyms_synonyms'],
    run_similarity_query_wo_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_hypernyms_hyponyms_synonyms, data_train['Sentence 2'],
               "Experimenting with synonyms, hypernyms and hyponyms")

paraphrased_sentences_synonyms_hypernyms_antonyms = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['synonyms_hypernyms_antonyms'],
    run_similarity_query_wo_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_synonyms_hypernyms_antonyms, data_train['Sentence 2'],
               "Experimenting with synonyms, hypernyms and antonyms")

paraphrased_sentences_hypernyms_synonyms_meronyms = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['hypernyms_synonyms_meronyms'],
    run_similarity_query_wo_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_hypernyms_synonyms_meronyms, data_train['Sentence 2'],
               "Experimenting with synonyms, hypernyms and meronyms")

paraphrased_sentences_all_combos = generate_paraphrases(
    data_train,
    tokenizer,
    model,
    device,
    combinations['all_combos'],
    run_similarity_query_w_examples,
    driver,
    transformer,
)
evaluate_model(paraphrased_sentences_all_combos, data_train['Sentence 2'],
               "Experimenting with all combinations")

