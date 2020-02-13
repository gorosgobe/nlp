import lib.data
import lib.embeddings
import numpy as np
import lib.mlp
from lib.utils import get_config, MODEL_PATIENCE
import random
import os
import csv

HYPERPARAM_SEARCH_FILE = "results_mlp.csv"

if __name__ == "__main__":
    print("Loading training data...")
    train_source, train_translation, train_scores = lib.data.load_data(data_type=lib.data.DatasetType.TRAIN, target_language=lib.data.Language.GERMAN)

    sources_tok, source_vocab = lib.data.tokenize(train_source)
    translation_tok, translation_vocab = lib.data.tokenize(train_translation)

    ENGLISH_EMBEDDING_MODEL_BIG = "/vol/bitbucket/eb1816/nlp_cw/embeddings/english/model.bin"
    GERMAN_EMBEDDING_MODEL_BIG  = "/vol/bitbucket/eb1816/nlp_cw/embeddings/german/model.bin"

    print("Loading english embedding model...")
    english_embedding_model = lib.embeddings.load_embedding(ENGLISH_EMBEDDING_MODEL_BIG, lib.embeddings.EmbeddingType.WORD2VEC)
    print("English model loaded")
    print("Loading german embedding model...")
    german_embedding_model = lib.embeddings.load_embedding(GERMAN_EMBEDDING_MODEL_BIG, lib.embeddings.EmbeddingType.WORD2VEC)
    print("German model loaded")

    sources_augmented_tok, translation_augmented_tok, augmented_targets \
     = lib.embeddings.augment_dataset(sources_tok, translation_tok, train_scores, english_embedding_model, german_embedding_model)

    with open("augmented_sources_train.src", "w+") as f:
        for sent in sources_augmented_tok:
            f.write(" ".join(sent))
            f.write("\n")

    with open("translation_augmented_train.mt", "w+") as f:
        for sent in translation_augmented_tok:
            f.write(" ".join(sent))
            f.write("\n")

    with open("augmented_scores.scores", "w+") as f:
        for ta in augmented_targets:
            f.write(str(ta))
            f.write("\n")
        

    print("Size augmented sources: ", len(sources_augmented_tok))

