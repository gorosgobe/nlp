"""
Module to handle word embedding functionality.
"""
import enum
from gensim.models import KeyedVectors
from tensorflow.keras.layers import Embedding
import tensorflow as tf
import numpy as np
import random
from lib.utils import PAD_TOK

class EmbeddingType(enum.Enum):
    WORD2VEC = 0

def load_embedding(path, embedding_type):
    """
    Returns: np.array with embedding weights of dimensionality (|V| x dim)
    """
    if embedding_type == EmbeddingType.WORD2VEC:
        return load_word2vec_embedding(path)

def load_word2vec_embedding(path):
    """
    Load the word2vec model given a path to the binaries.
    """
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    model.add([PAD_TOK], [np.zeros(100)])
    return model

def reduce_word2vec_vocab(input_path, output_path, vocab):
    """
    Downsamples the vocabulary in word2vec embeddings to less storage overhead.
    Given the input path of the embeddings and the vocabulary needed, create
    a new word2vec model removing words not in the voabulary. Save this resulting
    model in the output_path.
    """
    input_model = KeyedVectors.load_word2vec_format(input_path, binary=True)
    output_model = KeyedVectors(100)
    for word in vocab:
        if word in input_model.vocab:
            output_model.add([word], [input_model[word]])

    output_model.save_word2vec_format(output_path, binary=True)

def augment_dataset(source_sentences, translation_sentences, targets, source_big_model,
                    translation_big_model):
    """
    Given the source sentences, translation sentences and targets augement the dataset
    by switching words with their closest word embedding. source_big_model and
    translation_big_model refer to the whole (not downsampled) embeddings model used.

    For every source sentance, find the word which has the the highest similarity
    to another word in the vocabulary based on embeddings distance. Replace this word with
    this similary word, thereby creating a new sentence.
    """
    assert len(source_sentences) == len(translation_sentences)

    augmented_english = [*source_sentences]
    augmented_german  = [*translation_sentences]
    augmented_targets = [*targets]

    for idx, sentence in enumerate(source_sentences):
        chosen_index = 0
        max_similarity = 0

        # find the word in the sentence which has the highest similarity to another word in the
        # whole embedding mode based on embedding distance
        for candidate_index, word in enumerate(sentence):
            if word in source_big_model.vocab and word not in [".", ",", "!", ":", ";", "-"]:
                result = source_big_model.most_similar(positive=[word], topn=1)
                most_similar_word, most_similar_score = result[0]


                if most_similar_score > max_similarity:
                    candidate = most_similar_word
                    og = word
                    max_similarity = most_similar_score
                    chosen_index = candidate_index

        augmented_sentence = sentence[:]
        augmented_sentence[chosen_index] = candidate

        augmented_english.append(augmented_sentence)
        augmented_german.append(translation_sentences[idx])
        augmented_targets.append(targets[idx])

    return augmented_english, augmented_german, augmented_targets



def get_embeddings(model, tokenized_sentences, embedding_type, print_max_length=False):
    """
    Given the tokenised sentences and the embedding model.
    List of lists of embedded words and list of ignored words
    """
    res = []
    ignored = []
    max_len_sentence = 0
    for sentence in tokenized_sentences:
        embedded_sentence = []
        for word in sentence:
            if word in model.vocab:
                embedded_word = model[word]
                embedded_sentence.append(embedded_word)
            else:
                ignored.append(word)
        max_len_sentence = max(max_len_sentence, len(embedded_sentence))
        res.append(embedded_sentence)

    if print_max_length:
        print(f"Max length for sentence: {max_len_sentence}")

    return res, ignored

def get_embedding_input(data_tok, model, max_sent_len=None):
    """
    Given the tokenised data data_tok and the word embedding model,
    return the input to the Keras Embedding layer. That is,
    replace each word with its index in the embedding model and
    padd sentences using the pad token index.
    """
    num_sentences = len(data_tok)
    pad_idx = model.vocab[PAD_TOK].index

    out = []

    for sentence in data_tok:
        out.append([])
        for word in sentence:
            if word in model.vocab:
                out[-1].append(model.vocab[word].index)

    # pad to the max outputlength with the index of the pad token PAD_TOK
    padded_out = tf.keras.preprocessing.sequence.pad_sequences(
        out,
        padding="post",
        value=pad_idx,
        maxlen=max_sent_len,
    )

    return padded_out



def get_keras_embedding(model, trainable=False):
    """
    Get keras embedding layer from the embeddings model.
    If trainable is true, the weights can be updated during training.
    """

    vocab_size = len(model.vocab)

    # create weight matrix
    weights = []
    for i in range(vocab_size):
        weights.append(model[model.index2entity[i]])

    weights = np.array(weights)

    return Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=trainable,
    )


def get_sentence_embeddings(word_embeddings):
    """
    Average the word embeddings in a sentence to get a sentence embedding.
    word_embeddings: [ [word_np, word_np, ...], [word_np, ...], ...]
    """
    return np.array(list(map(lambda sentence: np.array(sentence).mean(0), word_embeddings)))
