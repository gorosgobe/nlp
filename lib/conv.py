import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, concatenate, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from lib.utils import EVALUATION_METRICS, MODELS_SAVE_PATH
import numpy as np

def _get_single_conv(max_len,
                     dim):

    input_ = Input(shape=(max_len, dim))

    filter_sizes = [2,4,8,16,32]
    filters_per_size = 10

    outputs_to_concat = []
    for filter_size in filter_sizes:
        conv_output = Conv1D(
            filters=filters_per_size,
            kernel_size=filter_size,
            activation="relu",
            input_shape=(max_len, dim)
        )(input_)

        max_pool_output = GlobalMaxPooling1D()(conv_output)

        outputs_to_concat.append(max_pool_output)

    output = concatenate(outputs_to_concat)

    return Model(inputs=input_, outputs=output)

def build_word_level_conv_net(max_english_len,
                              english_dim,
                              max_german_len,
                              german_dim):
    """
    Builds a word level convolutional neural network
    """


    english_input = Input(shape=(max_english_len, english_dim), name="english_input")
    german_input = Input(shape=(max_german_len, german_dim), name="german_input")

    english_conv = _get_single_conv(max_english_len, english_dim)
    german_conv = _get_single_conv(max_german_len, german_dim)

    x = concatenate([english_conv(english_input), german_conv(german_input)])
    x = Dense(40, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[english_input, german_input], outputs=output)

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=0.0001),
        metrics=EVALUATION_METRICS,
    )
    model.summary()
    return model


def fit_model(english_x_train, german_x_train, y_train, batch_size, epochs, 
              english_x_val, german_x_val, y_val, name, seed=420):
    """
    x_train shapes: (max_sent_len, dim)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)

    _, max_english_len, english_dim = english_x_train.shape
    _, max_german_len, german_dim = german_x_train.shape

    model = build_word_level_conv_net(max_english_len, english_dim, 
                                      max_english_len, german_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True), 
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    validation_data = None
    if english_x_val is not None \
        and german_x_val is not None and \
             y_val is not None:
            validation_data = [{"english_input": english_x_val,
                                "german_input": german_x_val},
                                y_val]

    model.fit({"english_input": english_x_train, "german_input": german_x_train}, y_train,
                                 batch_size=batch_size, epochs=epochs, verbose=1, validation_data=validation_data, callbacks=callbacks)
    return model



# TODO: remove
if __name__ == "__main__":
    build_word_level_conv_net(50, 100, 50, 100)