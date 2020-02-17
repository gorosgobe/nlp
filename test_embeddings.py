import pytest
from gensim.models.keyedvectors import KeyedVectors
from lib.utils import PAD_TOK
import numpy as np
import lib.embeddings
from tensorflow.keras import layers

def test_embeddings_input():
   corpus = [
        ["the", "man", "ran", "to", "boy"],
        ["the", "boy", "pig"],
        ["the", "pig", "man", "boy"],
   ]

   # "pig" and "to" is not in the vocabulary
   model = KeyedVectors(vector_size=2)
   model.add(["the"], [np.array([1, 3])]) # idx 0
   model.add(["man"], [np.array([2, 5])]) # idx 1
   model.add(["ran"], [np.array([6, 9])]) # idx 2
   model.add(["boy"], [np.array([4, 20])]) # idx 3
   model.add([PAD_TOK], [np.array([0, 0])]) # idx 4

   embedding_input = lib.embeddings.get_embedding_input(corpus, model)

   assert (embedding_input == np.array([
       [0, 1, 2, 3],
       [0, 3, 4, 4],
       [0, 1, 3, 4],
   ])).all()

   embedding_layer = lib.embeddings.get_keras_embedding(model)

   embedded = embedding_layer(embedding_input)


   assert (embedded.numpy() ==  np.array([[[ 1.,  3.], [ 2.,  5.], [ 6.,  9.], [ 4., 20.]],
                                          [[ 1.,  3.], [ 4., 20.], [ 0.,  0.], [ 0.,  0.]],
                                          [[ 1.,  3.], [ 2.,  5.], [ 4., 20.], [ 0.,  0.]]])).all()
