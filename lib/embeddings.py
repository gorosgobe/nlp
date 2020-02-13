import enum
from gensim.models import KeyedVectors
import numpy as np
import random

class EmbeddingType(enum.Enum):
    WORD2VEC = 0

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
        if idx == 20:
            break
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

def get_sentence_embeddings(word_embeddings):
    """
    word_embeddings: [ [word_np, word_np, ...], [word_np, ...], ...]
    """
    return np.array(list(map(lambda sentence: np.array(sentence).mean(0), word_embeddings)))
