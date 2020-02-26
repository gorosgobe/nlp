"""
Functions to build, compile and run a Multi-Layer Perceptron (MLP) based architecture
"""

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Lambda, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from lib.utils import MODELS_SAVE_PATH, EVALUATION_METRICS, MODEL_PATIENCE, PAD_TOK
from lib.embeddings import get_keras_embedding
import numpy as np

def build_compile_model(learning_rate, layers, dropout):
    """
    Builds an MLP model, given hyperparameters learning_rate, layers (list of neurons) and dropout (applied to all layers)
    """

    model = Sequential()
    model.add(Dense(units=layers[0], activation="relu", input_dim=200))
    for l in layers[1:]:
        model.add(Dense(units=l, activation="relu"))
        model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=EVALUATION_METRICS
    )
    return model

def get_average_embedding_model(input_shape,
                                w2v_model):
    """
    Computes the embedding of the input via an embedding layer and averages the embedding to obtain an average embedding
    """
    inputs = Input(shape=input_shape)
    embedded = get_keras_embedding(w2v_model)(inputs)
    embedded_sum = Lambda(lambda x: K.sum(x, axis=1), name="sum")(embedded)

    pad_token = w2v_model.vocab[PAD_TOK].index
    sent_lens = Lambda(lambda x: K.sum(K.cast(K.not_equal(x, pad_token), "float32"),
                                       axis=1, keepdims=True), name="sent_len")(inputs)

    avg = Lambda(lambda inputs: inputs[0] / inputs[1], name="avg")([embedded_sum,
                                                                     sent_lens])

    res = Model(inputs=inputs, outputs=avg)
    return res


def build_compile_model_embedding_layer(english_input_shape,
                                        german_input_shape,
                                        english_w2v,
                                        german_w2v,
                                        learning_rate,
                                        layers,
                                        dropout):
    """
    Builds an MLP with average embedding for sentences based on supplied hyperparameters and shape of data.
    """

    english_input = Input(shape=english_input_shape, name="english_input")
    german_input = Input(shape=german_input_shape, name="german_input")

    english_avg = get_average_embedding_model(english_input_shape,
                                              english_w2v)(english_input)
    german_avg = get_average_embedding_model(german_input_shape,
                                             german_w2v)(german_input)

    mlp = Sequential()
    mlp.add(Dense(units=layers[0], activation="relu", input_dim=200))
    for l in layers[1:]:
        mlp.add(Dense(units=l, activation="relu"))
        mlp.add(Dropout(dropout))

    mlp.add(Dense(units=1))

    mlp_out = mlp(concatenate([english_avg, german_avg]))

    model = Model(inputs=[english_input, german_input], outputs=mlp_out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=EVALUATION_METRICS
    )


    return model



def fit_model_embedding_layer(english_x_train, german_x_train, y_train, english_x_val, german_x_val, y_val,
                              english_w2v, german_w2v, batch_size, epochs, learning_rate, name, layers,
                              dropout, verbose=0, seed=0):
    """
    Trains the MLP model on the training data supplied, while fine-tuning the pretrained embeddings. 
    Performs model checkpointing and early stopping, using the hyperparameters provided.
    Returns the trained model and the history of metrics obtained via the training process. 
    This history can be used to select models after a hyperparameter search. 
    """
    K.clear_session()
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)



    model = build_compile_model_embedding_layer(
        english_input_shape=english_x_train.shape[1:],
        german_input_shape=german_x_train.shape[1:],
        english_w2v=english_w2v,
        german_w2v=german_w2v,
        learning_rate=learning_rate,
        layers=layers,
        dropout=dropout,
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MODEL_PATIENCE, verbose=1,
                      restore_best_weights=True),
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss',
                        verbose=verbose, save_best_only=True, save_weights_only=True)
    ]

    validation_data = None
    if english_x_val is not None \
        and german_x_val is not None and \
             y_val is not None:
            validation_data = [{"english_input": english_x_val,
                                "german_input": german_x_val},
                                y_val]

    model.summary()


    assert model.get_layer("model").get_layer("embedding").trainable == False
    assert model.get_layer("model_1").get_layer("embedding_1").trainable == False


    history_freeze = model.fit(x={"english_input": english_x_train, "german_input": german_x_train},
                        y=y_train, batch_size=batch_size, epochs=10, verbose=verbose,
                        validation_data=validation_data, callbacks=callbacks)

    if verbose:
        print("unfreezing embedding layers")


    model.get_layer("model").get_layer("embedding").trainable = True
    model.get_layer("model_1").get_layer("embedding_1").trainable = True

    assert model.get_layer("model").get_layer("embedding").trainable == True
    assert model.get_layer("model_1").get_layer("embedding_1").trainable == True


    model.compile(
        loss='mean_squared_error',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=EVALUATION_METRICS
    )
    model.summary()


    history_tune = model.fit(x={"english_input": english_x_train, "german_input": german_x_train},
                        y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=validation_data, callbacks=callbacks)

    return model, history_tune


def fit_model(x, y, x_val, y_val, batch_size, epochs, learning_rate, name, layers, dropout, seed=2019, verbose=0):
    """
    Builds, compiles and trains model on given dataset, given the supplied hyperparameters. Embedding is fixed.
    x: size (7000, 200)
    y: size (7000,)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)
    model = build_compile_model(learning_rate, layers, dropout)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MODEL_PATIENCE, verbose=1, restore_best_weights=True),
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True)
    ]
    validation_data = None
    if x_val is not None and y_val is not None:
        validation_data = [x_val, y_val]

    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=validation_data, callbacks=callbacks)
    return model, history


def eval_model(x_test, y_test, model):
    """
    Evaluates a supplied model given some test data
    """
    score = model.evaluate(x_test, y_test)
    print(score)
    return score
