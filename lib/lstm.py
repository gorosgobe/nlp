import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, concatenate, Input, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.utils import MODELS_SAVE_PATH, EVALUATION_METRICS, BASE_PADDING, MODEL_PATIENCE
from lib.data import pad_to_length

import numpy as np

def build_compile_model(
        english_dimensionality, german_dimensionality, learning_rate,
        layers, dropout, english_lstm_units, german_lstm_units, dropout_lstm
    ):
    """
    Builds a LSTM model
    """
    #import pdb; pdb.set_trace()
    # define two sets of inputs
    english_input = Input(shape=(None, english_dimensionality), name='english_input')
    german_input = Input(shape=(None, german_dimensionality), name='german_input')

    # english branch
    x = LSTM(english_lstm_units)(english_input)
    x = Model(inputs=english_input, outputs=x)

    # german branch
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

def fit_model(english_x, german_x, y, batch_size, epochs, learning_rate,
              layers, dropout, english_lstm_units, german_lstm_units,
              dropout_lstm, english_x_val, german_x_val, y_val, name, seed=2019):
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
        100, 100, learning_rate=learning_rate, layers=layers, dropout=dropout,
        dropout_lstm=dropout_lstm, english_lstm_units=english_lstm_units,
        german_lstm_units=german_lstm_units
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MODEL_PATIENCE, verbose=1, restore_best_weights=True),
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    ]

    train_generator = batch_generator(english_x, german_x, y, batch_size)
    validation_generator = None

    if english_x_val is not None and german_x_val is not None and y_val is not None:
        validation_generator = batch_generator(english_x_val, german_x_val, y_val, batch_size)

    history = model.fit_generator(train_generator, steps_per_epoch=len(english_x)//batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=len(english_x_val)//batch_size,
                        verbose=0, callbacks=callbacks)

    return model, history

def eval_model(x_test_english, x_test_german, y_test, model):
    score = model.evaluate([x_test_english, x_test_german], y_test)
    print(score)
    return score

def batch_generator(in1, in2, labels, batch_size=128):
    n_batches_per_epoch = len(in1)//batch_size
    while True:
        for i in range(n_batches_per_epoch):
            # Get vectors for batch
            in1_batch_vecs = in1[batch_size * i:batch_size * (i + 1)]
            in2_batch_vecs = in2[batch_size * i:batch_size * (i + 1)]
            y_batch = labels[batch_size * i:batch_size * (i + 1)]

            # Find max len of sentences in batch and pad vectors
            in1_longest_sent_len = max(len(x) for x in in1)
            in2_longest_sent_len = max(len(x) for x in in2)
            pad_to_length(in1_batch_vecs, in1_longest_sent_len, BASE_PADDING)
            pad_to_length(in2_batch_vecs, in2_longest_sent_len, BASE_PADDING)

            en_batch = np.array(in1_batch_vecs)
            de_batch = np.array(in2_batch_vecs)

            yield {'english_input': en_batch, 'german_input': de_batch}, {'output': y_batch}
