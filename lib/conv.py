import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, concatenate, Dense, Dropout, \
    BatchNormalization, GlobalAveragePooling1D, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from lib.utils import EVALUATION_METRICS, MODELS_SAVE_PATH
import numpy as np

def _get_single_conv(max_len, dim, *, stride,
                                      filter_sizes,
                                      filter_counts,
                                      pooling_type):

    """
    Gets a single CNN for a particular language.
    :param max_len: Maximum sentence length (number of rows)
    :param dim: Dimension of each word
    :param stride: stride of convolution
    :param filter_sizes: A list of sizes of each filter
    :param filter_counts: A list of the number of filter of each size. Should be same length as filter_sizes.
    :param pooling_type: String specifying the pooling type used. "max" for max pooling, "avg", for average pooling.
    :returns: The CNN model
    """

    assert len(filter_sizes) == len(filter_counts), "Filter sizes should be the same length as\
                                                      the number of filter"
    assert pooling_type.lower() in ["max", "avg"]

    input_ = Input(shape=(max_len, dim))

    outputs_to_concat = []
    for filter_size, filter_count in zip(filter_sizes, filter_counts):
        # for each filter_size -> convolve -> batch norm -> activate -> pool

        conv_output = Conv1D(
            filters=filter_count,
            kernel_size=filter_size,
            input_shape=(max_len, dim),
            strides=stride,
        )(input_)

        normed = BatchNormalization()(conv_output)

        activations = ReLU()(normed)

        if pooling_type == "max":
            pool_output = GlobalMaxPooling1D()(activations)
        else:
            pool_output = GlobalAveragePooling1D()(activations)

        outputs_to_concat.append(pool_output)

    # concatenate the outputs
    output = concatenate(outputs_to_concat)

    model =  Model(inputs=input_, outputs=output)
    model.summary()
    return model

def build_word_level_conv_net(max_english_len,
                            english_dim,
                            max_german_len,
                            german_dim,
                            *,
                            stride,
                            filter_sizes,
                            filter_counts,
                            dropout_rate,
                            pooling_type,
                            fc_layers,
                            learning_rate):
    """
    :param max_english_len: Max English sentence length
    :param english_dim: English word dimension
    :param max_german_len: Max German sentence length
    :param german_dim: Max German word dimension
    :param stride: Convolution stride
    :param filter_sizes: A list of sizes of each filter
    :param filter_counts: A list of the number of filter of each size. Should be same length as filter_sizes.
    :param dropout_rate: Dropout rate
    :param pooling_type: String specifying the pooling type used. "max" for max pooling, "avg", for average pooling.
    :param fc_layers: List of nodes in each fully connected layer
    :param learning_rate: Initial learning rate
    """

    english_input = Input(shape=(max_english_len, english_dim), name="english_input")
    german_input = Input(shape=(max_german_len, german_dim), name="german_input")

    english_conv = _get_single_conv(max_english_len, english_dim,
                                    stride=stride,
                                    filter_sizes=filter_sizes,
                                    filter_counts=filter_counts,
                                    pooling_type=pooling_type)(english_input)

    german_conv = _get_single_conv(max_german_len, german_dim,
                                   stride=stride,
                                   filter_sizes=filter_sizes,
                                   filter_counts=filter_counts,
                                   pooling_type=pooling_type)(german_input)


    x = concatenate([english_conv, german_conv])

    x = Dropout(dropout_rate)(x)

    for layer_size in fc_layers:
        x = Dense(layer_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[english_input, german_input], outputs=output)

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=learning_rate),
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
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=1,
                        save_best_only=True, save_weights_only=True)
    ]

    validation_data = None
    if english_x_val is not None \
        and german_x_val is not None and \
             y_val is not None:
            validation_data = [{"english_input": english_x_val,
                                "german_input": german_x_val},
                                y_val]

    history = model.fit({"english_input": english_x_train, "german_input": german_x_train},
              y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=validation_data, callbacks=callbacks)
    return model, history
