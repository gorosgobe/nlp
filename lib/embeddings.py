import enum
from gensim.models import KeyedVectors
from tensorflow.keras.layers import Embedding
import tensorflow as tf
import numpy as np
import random
from lib.utils import PAD_TOK
from lib.data import pad_to_length

class EmbeddingType(enum.Enum):
    WORD2VEC = 0

def get_pos_indices(pos_tags, idx_map):
    result = []
    max_sent_len = max(len(l) for l in pos_tags)

    for pos_tag_s in pos_tags:
        tag_encodings = []
        for tag in pos_tag_s:
            tag_idx = idx_map[tag]
            encoding = [0] * len(idx_map)
            encoding[tag_idx] = 1
            tag_encodings.append(encoding)

        result.append(tag_encodings)

    pad_value = [0] * len(idx_map)
    pad_to_length(result, max_sent_len, pad_value)

    return np.array(result)


def load_embedding(path, embedding_type):
    """
    Returns: np.array with embedding weights of dimensionality (|V| x dim)
    """
    # TODO: think about different dimensionalities and load them accordingly
    if embedding_type == EmbeddingType.WORD2VEC:
        return load_word2vec_embedding(path)

def load_word2vec_embedding(path):
    # path should be .bin
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    # TODO: remove 100 hardcoded
    model.add([PAD_TOK], [np.zeros(100)])
    return model

def reduce_word2vec_vocab(input_path, output_path, vocab):
    # TODO: docstring
    input_model = KeyedVectors.load_word2vec_format(input_path, binary=True)
    # TODO: remove hardcoded value
    output_model = KeyedVectors(100)
    for word in vocab:
        if word in input_model.vocab:
            output_model.add([word], [input_model[word]])

    output_model.save_word2vec_format(output_path, binary=True)

def augment_dataset(source_sentences, translation_sentences, targets, source_big_model, translation_big_model):
    assert len(source_sentences) == len(translation_sentences)
    augmented_english = [*source_sentences]
    augmented_german  = [*translation_sentences]
    augmented_targets = [*targets]

    for idx, sentence in enumerate(source_sentences):
        chosen_index = 0
        max_similarity = 0
        for candidate_index, word in enumerate(sentence):
            if word in source_big_model.vocab and word not in [".", ",", "!", ":", ";", "-"]:
                # flip word
                result = source_big_model.most_similar(positive=[word], topn=1)
                most_similar_word, most_similar_score = result[0]
                if most_similar_score > max_similarity:
                    candidate = most_similar_word
                    og = word
                    max_similarity = most_similar_score
                    chosen_index = candidate_index

        #print("Sentence: ", sentence)
        #print("Candidate word, ", candidate)
        augmented_sentence = sentence[:]
        augmented_sentence[chosen_index] = candidate
        #print("Augmented sentence: ", sentence)
        augmented_english.append(augmented_sentence)
        augmented_german.append(translation_sentences[idx])
        augmented_targets.append(targets[idx])

    return augmented_english, augmented_german, augmented_targets



def get_embeddings(model, tokenized_sentences, embedding_type, print_max_length=False):
    """
    Param: tokenized_sentences: list of lists of words
    Returns: List of lists of embeddings, and list of ignored words
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

def get_embedding_input(data_tok, model, max_sent_len=None, pos_idx_map=None):
    pad_idx = model.vocab[PAD_TOK].index

    use_pos = isinstance(data_tok[0][0], tuple)
    if use_pos:
        out_tags = []
    else:
        out_tags = None

    out = []
    for sentence in data_tok:
        if use_pos:
            tokenised_sent, pos_tags = zip(*sentence)
        else:
            tokenised_sent = sentence

        out.append([])

        if use_pos:
            out_tags.append([])

        for idx, word in enumerate(tokenised_sent):
            if word in model.vocab:
                out[-1].append(model.vocab[word].index)
                if use_pos:
                    out_tags[-1].append(pos_tags[idx])

    # build map for pos tags
    if use_pos and pos_idx_map is not None:
        for out_tag_s in out_tags:
            for tag in out_tag_s:
                if tag not in pos_idx_map:
                    pos_idx_map[tag] = len(pos_idx_map)


    padded_out = tf.keras.preprocessing.sequence.pad_sequences(
        out,
        padding="post",
        value=pad_idx,
        maxlen=max_sent_len,
    )

    return padded_out, out_tags



def get_keras_embedding(model, trainable=False):

    vocab_size = len(model.vocab)
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
    word_embeddings: [ [word_np, word_np, ...], [word_np, ...], ...]
    """
    return np.array(list(map(lambda sentence: np.array(sentence).mean(0), word_embeddings)))
