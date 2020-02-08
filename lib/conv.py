import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, concatenate, Dense, Dropout, \
    BatchNormalization, GlobalAvgPooling1D, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from lib.utils import EVALUATION_METRICS, MODELS_SAVE_PATH
import numpy as np

def _get_single_conv(max_len, dim, *, stride, 
                                      filter_sizes, 
                                      filter_numbers, 
                                      dropout_rate,
                                      pooling_type):

    assert len(filter_sizes) == len(filter_numbers), "Filter sizes should be the same length as\
                                                      the number of filter"
    assert pooling_type.lower() in ["max", "avg"]

    input_ = Input(shape=(max_len, dim))

    pooling_layers ={
        "max": GlobalMaxPooling1D,
        "avg": GlobalAvgPooling1D,
    }

    outputs_to_concat = []
    for filter_size,  in filter_sizes:
        conv_output = Conv1D(
            filters=filters_per_size,
            kernel_size=filter_size,
            input_shape=(max_len, dim),
            stride=stride,
        )(input_)
        
        normed = BatchNormalization()(conv_output)
        
        activations = ReLU(normed)
        
        pool_output = pooling_layers[pooling_type](activations)

        outputs_to_concat.append(pool_output)

    output = concatenate(outputs_to_concat)

    output = Dropout(dropout_rate)()

    return Model(inputs=input_, outputs=output)

def build_word_level_conv_net(max_english_len,
                            english_dim,
                            max_german_len,
                            german_dim,
                            *,
                            stride, 
                            filter_sizes, 
                            filter_numbers, 
                            dropout_rate,
                            pooling_type,
                            fc_layers):
    """
    Builds a word level convolutional neural network
    """

    english_input = Input(shape=(max_english_len, english_dim), name="english_input")
    german_input = Input(shape=(max_german_len, german_dim), name="german_input")

    english_conv = _get_single_conv(max_english_len, english_dim, 
                                    stride=stride,
                                    filter_sizes=filter_sizes,
                                    filter_numbers=filter_numbers,
                                    dropout_rate=dropout_rate,
                                    pooling_type=pooling_type)   
    german_conv = _get_single_conv(max_german_len, german_dim,
                                   stride=stride,
                                   filter_sizes=filter_sizes,
                                   filter_numbers=filter_numbers,
                                   dropout_rate=dropout_rate,
                                   pooling_type=pooling_type)

    x = concatenate([english_conv(english_input), german_conv(german_input)])

    for layer_size in fc_layers:
        x = Dense(40, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

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
              english_x_val, german_x_val, y_val, name, network_params, seed=420):
    """
    x_train shapes: (max_sent_len, dim)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)

    _, max_english_len, english_dim = english_x_train.shape
    _, max_german_len, german_dim = german_x_train.shape

    model = build_word_level_conv_net(max_english_len, english_dim, 
                                      max_english_len, german_dim, **network_params)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True), 
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
    build_word_level_conv_net(50, 100, 50, 100,
                            stride=1,
                            filter_sizes=[1,2,3],
                            filter_numbers=[3,3,4],
                            dropout_rate=0.5,
                            pooling_type="max",
                            fc_layers=[30,40, 10])