import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, concatenate, Dense

def _get_single_conv(max_len,
                     dim):

    input_ = Input(shape=(max_len, dim))

    filter_sizes = [2,3,4]
    filters_per_size = 2

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
    x = Dense(48, activation="relu")(x)
    x = Dense(24, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[english_input, german_input], outputs=output)

    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=["mse", "mae"],
    )
    model.summary()
    return model

if __name__ == "__main__":
    build_word_level_conv_net(50, 100, 50, 100)
