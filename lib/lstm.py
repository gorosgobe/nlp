import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, concatenate, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.utils import MODELS_SAVE_PATH
import numpy as np

def build_compile_model(max_english_len_sentence, max_german_len_sentence, english_dimensionality, german_dimensionality):
    """
    Builds a LSTM model
    """

    # define two sets of inputs
    english_input = Input(shape=(7000, max_english_len_sentence, english_dimensionality))
    german_input = Input(shape=(7000, max_german_len_sentence, german_dimensionality))
    
    # english branch
    x = LSTM(64)(english_input)
    x = Model(inputs=english_input, outputs=x)
    
    # german branch
    y = LSTM(64)(german_input)
    y = Model(inputs=german_input, outputs=y)
    
    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(200, activation="relu")(combined)
    z = Dense(100, activation="relu")(z)
    z = Dense(1, activation="linear")(z)
    
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)
    return model
    
def fit_model(english_x, german_x, y, batch_size, epochs, english_x_val, german_x_val, y_val, name, seed=2019):
    """
    Builds, compiles and trains model on given dataset
    english_x, german_x: size (7000, max_len_sentence, 100)
    y: size (7000,)
    """
    print(english_x.shape)
    print(german_x.shape)
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)
    _, max_len_sentence_english, english_dim = english_x.shape
    _, max_len_sentence_german,  german_dim  = german_x.shape
    
    # TODO: remove 100 hardcoded
    model = build_compile_model(max_len_sentence_english, max_len_sentence_german, 100, 100)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True), 
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    validation_data = None
    if english_x_val is not None and german_x_val is not None and y_val is not None:
        validation_data = [[english_x_val, german_x_val], y_val]

    model.fit([english_x, german_x], y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=validation_data, callbacks=callbacks)
    return model

def eval_model(x_test_english, x_test_german, y_test, model):
    score = model.evaluate([x_test_english, x_test_german], y_test)
    print(score)
    return score