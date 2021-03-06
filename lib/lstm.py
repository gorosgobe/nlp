"""
Functions to build and train LSTM based models.
"""
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, concatenate, Input, LSTM, Dropout, Bidirectional, Masking, Lambda, multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.utils import MODELS_SAVE_PATH, EVALUATION_METRICS, BASE_PADDING, MODEL_PATIENCE, PAD_TOK
from lib.data import pad_to_length
from lib.embeddings import get_keras_embedding
from tensorflow.keras import backend as K

import numpy as np

def average_f(inputs, mask):
    """
    Lambda layer function that averages its input taking a mask into account
    """
    number_mask = K.cast(mask, "float32")
    sums = K.sum(inputs * K.expand_dims(number_mask, -1), axis=1)
    lengths = K.sum(number_mask, axis=1, keepdims=True)
    return sums / lengths

def sum_f(inputs, mask):
    """
    Lambda layer function that sums its input taking a mask into account
    """
    number_mask = K.cast(mask, "float32")
    sums = K.sum(inputs * K.expand_dims(number_mask, -1), axis=1)
    return sums


def build_compile_model(
        english_w2v, german_w2v, learning_rate,
        layers, dropout, english_lstm_units, german_lstm_units, dropout_lstm, bidirectional=False, attention=False, 
        pos_tags_encoded_en=None, pos_tags_encoded_de=None
    ):
    """
    Builds an LSTM model
    """

    english_input = Input(shape=(None, ), name='english_input')
    german_input = Input(shape=(None, ), name='german_input')

    english_encoding = Input(shape=(None, 41), name='english_encoding')
    german_encoding = Input(shape=(None, 51), name='german_encoding')

    # Embedding Layer
    english_embedding_layer = get_keras_embedding(english_w2v)
    english_embedded = english_embedding_layer(english_input)
    german_embedding_layer = get_keras_embedding(german_w2v)
    german_embedded = german_embedding_layer(german_input)

    english_combined_input = concatenate([english_embedded, english_encoding], axis=2)
    german_combined_input = concatenate([german_embedded, german_encoding], axis=2)

    # Masking Layer
    english_masked = Masking(mask_value=0.0, input_shape=(39, 141))(english_combined_input)
    german_masked = Masking(mask_value=0.0, input_shape=(39, 151))(german_combined_input)

    # english branch
    if bidirectional:
        en_repr = Bidirectional(LSTM(english_lstm_units, return_sequences=attention))(english_masked)
    else:
        en_repr = LSTM(english_lstm_units, input_shape=(39, 141), return_sequences=attention)(english_masked)

    # german branch
    if bidirectional:
        de_repr = Bidirectional(LSTM(german_lstm_units, return_sequences=attention))(german_masked)
    else:
        de_repr = LSTM(german_lstm_units, input_shape=(39, 151), return_sequences=attention)(german_masked)

    if attention:
        # compute attention probabilities
        attention_probs_en = LSTM(1, return_sequences=True, activation='softmax')(en_repr)
        # multiply with attention
        attention_mul_en = multiply([en_repr, attention_probs_en])
        # sum to obtain weighted average
        averaged_en = Lambda(sum_f)(attention_mul_en)

        # same as above, but with German
        attention_probs_de = LSTM(1, return_sequences=True, activation='softmax')(de_repr)
        attention_mul_de = multiply([de_repr, attention_probs_de])

        averaged_de = Lambda(sum_f)(attention_mul_de)
        # combine the output of the two branches
        combined = concatenate([averaged_en, averaged_de])
    else:
        combined = concatenate([en_repr, de_repr])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = combined
    for units in layers:
        z = Dense(units, activation="relu")(z)
        z = Dropout(dropout)(z)
    z = Dense(1, activation="linear", name='output')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value

    model = Model(inputs=[english_input, german_input, english_encoding, german_encoding], outputs=z)
    model.compile(
        loss='mean_squared_error',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=EVALUATION_METRICS
    )
    return model

def fit_model(english_x, german_x, english_w2v, german_w2v, y, batch_size, epochs, learning_rate,
              layers, dropout, english_lstm_units, german_lstm_units,
              dropout_lstm, english_x_val, german_x_val, y_val, name, bidirectional=False, seed=2019, verbose=0, attention=False):
    """
    Builds, compiles and trains model on given dataset
    english_x, german_x: size (7000, max_len_sentence, 100)
    y: size (7000,)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)

    model = build_compile_model(
        english_w2v=english_w2v, german_w2v=german_w2v, learning_rate=learning_rate, layers=layers, dropout=dropout,
        dropout_lstm=dropout_lstm, english_lstm_units=english_lstm_units,
        german_lstm_units=german_lstm_units, bidirectional=bidirectional, attention=attention
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MODEL_PATIENCE, verbose=verbose, restore_best_weights=True),
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True)
    ]

    # train_generator = batch_generator(english_x, german_x, y, batch_size)
    validation_data = None
    if english_x_val is not None and german_x_val is not None and y_val is not None:
        validation_data = { 
            'english_input': english_x_val['input'], 
            'german_input': german_x_val['input'], 
            'english_encoding': english_x_val['encoding'], 
            'german_encoding': german_x_val['encoding']
        }, {'output': y_val}

    history = model.fit(
        {
            'english_input': english_x['input'], 
            'german_input': german_x['input'], 
            'english_encoding': english_x['encoding'], 
            'german_encoding': german_x['encoding']
        }, 
        y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=validation_data, 
        callbacks=callbacks
    )

    return model, history

def eval_model(x_test_english, x_test_german, y_test, model):
    """
    Evaluates supplied model on test data
    """
    score = model.evaluate([x_test_english, x_test_german], y_test)
    print(score)
    return score
