import enum
from gensim.models import KeyedVectors
import numpy as np

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


def get_embeddings(model, tokenized_sentences, path, embedding_type):
    """
    Param: tokenized_sentences: list of lists of words
    Returns: List of lists of embeddings, and list of ignored words
    """
    res = []
    ignored = []
    for sentence in tokenized_sentences:
        embedded_sentence = []
        for word in sentence:
            if word in model.wv.vocab:
                embedded_word = model[word]
                embedded_sentence.append(embedded_word)
            else:
                ignored.append(word)
        res.append(embedded_sentence)

    return res, ignored

def get_sentence_embeddings(word_embeddings):
    """
    word_embeddings: [ [word_np, word_np, ...], [word_np, ...], ...]
    """
    return np.array(list(map(lambda sentence: np.array(sentence).mean(0), word_embeddings)))