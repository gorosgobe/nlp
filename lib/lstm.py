import tensorflow.keras
from tensorflow.keras.models import Sequential, Model, Masking
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, concatenate, Input, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.utils import MODELS_SAVE_PATH, EVALUATION_METRICS, BASE_PADDING, MODEL_PATIENCE, PAD_TOK
from lib.data import pad_to_length
from lib.embeddings import get_keras_embedding

import numpy as np

def build_compile_model(
        english_dimensionality, german_dimensionality, english_w2v, german_w2v, learning_rate,
        layers, dropout, english_lstm_units, german_lstm_units, dropout_lstm, bidirectional=False
    ):
    """
    Builds a LSTM model
    """
    #import pdb; pdb.set_trace()
    # define two sets of inputs
    english_input = Input(shape=(None, english_dimensionality), name='english_input')
    german_input = Input(shape=(None, german_dimensionality), name='german_input')

    # Embedding Layer
    english_input = get_keras_embedding(english_w2v)(english_input)
    german_input = get_keras_embedding(german_w2v)(german_input)

    english_input = Masking(mask_value=english_w2v.vocab[PAD_TOK].index)(english_input)
    german_input = Masking(mask_value=german_w2v.vocab[PAD_TOK].index)(german_input)

    # english branch
    if bidirectional:
        x = Bidirectional(LSTM(english_lstm_units))(english_input)
    else:
        x = LSTM(english_lstm_units)(english_input)
    x = Model(inputs=english_input, outputs=x)

    # german branch
    if bidirectional:
        y = Bidirectional(LSTM(german_lstm_units))(german_input)
    else:
        y = LSTM(german_lstm_units)(german_input)
    y = Model(inputs=german_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = combined
    for units in layers:
        z = Dense(units, activation="relu")(z)
        z = Dropout(dropout)(z)
    z = Dense(1, activation="linear", name='output')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(
        loss='mean_squared_error',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=EVALUATION_METRICS
    )
    return model

def fit_model(english_x, german_x, english_w2v, german_w2v, y, batch_size, epochs, learning_rate,
              layers, dropout, english_lstm_units, german_lstm_units,
              dropout_lstm, english_x_val, german_x_val, y_val, name, bidirectional=False, seed=2019, verbose=0):
    """
    Builds, compiles and trains model on given dataset
    english_x, german_x: size (7000, max_len_sentence, 100)
    y: size (7000,)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)

    # TODO: remove 100 hardcoded
    model = build_compile_model(
        100, 100, english_w2v=english_w2v, german_w2v=german_w2v, learning_rate=learning_rate, layers=layers, dropout=dropout,
        dropout_lstm=dropout_lstm, english_lstm_units=english_lstm_units,
        german_lstm_units=german_lstm_units, bidirectional=bidirectional
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MODEL_PATIENCE, verbose=verbose, restore_best_weights=True),
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True)
    ]

    # train_generator = batch_generator(english_x, german_x, y, batch_size)
    validation_data = None
    if english_x_val is not None and german_x_val is not None and y_val is not None:
        validation_data = { 'english_input': english_x_val, 'german_input': german_x_val }, {'output': y_val}

    history = model.fit({'english_input': english_x, 'german_input': german_x}, y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=validation_data, callbacks=callbacks)

    return model, history

def eval_model(x_test_english, x_test_german, y_test, model):
    score = model.evaluate([x_test_english, x_test_german], y_test)
    print(score)
    return score
