import pytest
import lib.embeddings
import lib.data
import lib.mlp
from lib.utils import CONSTANT_MAX_LENGTH_ENGLISH, CONSTANT_MAX_LENGTH_GERMAN, PAD_TOK
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

TRAIN_EN, TRAIN_DE, _ = \
            lib.data.load_data(data_type=lib.data.DatasetType.TRAIN,
                               target_language=lib.data.Language.GERMAN)


@pytest.mark.parametrize(
    "model,data,max_sent_len",
    [
        (lib.embeddings.load_word2vec_embedding("embeddings/en_model_downsampled.bin"), TRAIN_EN,
                                                                    CONSTANT_MAX_LENGTH_ENGLISH),
        (lib.embeddings.load_word2vec_embedding("embeddings/de_model_downsampled.bin"), TRAIN_DE,
                                                                     CONSTANT_MAX_LENGTH_GERMAN)
    ]
)
def test_average_embeddings(model, data, max_sent_len):
    data_tok, _ = lib.data.tokenize(data)

    vectors, _ = lib.embeddings.get_embeddings(
        model,
        data_tok,
        lib.embeddings.EmbeddingType.WORD2VEC,
    )

    avg = lib.embeddings.get_sentence_embeddings(vectors)

    embedding_layer_input = lib.embeddings.get_embedding_input(data_tok, model, max_sent_len)

    avg_embedding_model = lib.mlp.get_average_embedding_model(input_shape=embedding_layer_input.shape[1:],
                                                              w2v_model=model)

    model_avg = avg_embedding_model.predict(embedding_layer_input)

    assert avg.shape == model_avg.shape
    assert np.allclose(avg, model_avg, rtol=0, atol=0.000001)

def test_average_embeddings_custom():
    corpus = [
        ["the", "man", "ran"],
        ["the", "boy"],
        ["the", "man", "boy"],
    ]

    max_sent_len = 3

    model = KeyedVectors(vector_size=1)

    model.add(["the"], [np.array([1])])
    model.add(["man"], [np.array([2])])
    model.add(["ran"], [np.array([3])])
    model.add(["boy"], [np.array([6])])
    model.add([PAD_TOK], [np.array([0])])

    embedding_input = lib.embeddings.get_embedding_input(corpus, model, max_sent_len)

    avg_embedding_model = lib.mlp.get_average_embedding_model(
        input_shape=embedding_input.shape[1:],
        w2v_model=model,
    )

    avg = avg_embedding_model.predict(embedding_input)

    print(avg)

    assert avg[0][0] == (1 + 2 + 3) / 3
    assert avg[1][0] == (1 + 6) / 2
    assert avg[2][0] == (1 + 2 + 6) / 3
