import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, concatenate, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from lib.utils import MODELS_SAVE_PATH
import numpy as np

def build_compile_model(english_dimensionality, german_dimensionality):
    """
    Builds a LSTM model
    """

    # define two sets of inputs
    english_input = Input(shape=(None, english_dimensionality))
    german_input = Input(shape=(None, german_dimensionality))
    
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
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_squared_error', "mae"]
    )
    return model
    
def fit_model(english_x, german_x, y, batch_size, epochs, english_x_val, german_x_val, y_val, name, seed=2019):
    """
    Builds, compiles and trains model on given dataset
    english_x, german_x: size (7000, max_len_sentence, 100)
    y: size (7000,)
    """
    # print(english_x.shape)
    # print(german_x.shape)
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)
    # _, max_len_sentence_english, english_dim = english_x.shape
    # _, max_len_sentence_german,  german_dim  = german_x.shape
    
    # TODO: remove 100 hardcoded
    model = build_compile_model(100, 100)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True), 
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    validation_generator = None
    if english_x_val is not None and german_x_val is not None and y_val is not None:
        validation_generator = DataGenerator(english_x_val, german_x_val, y_val, dim=(), batch_size=batch_size) 

    # model.fit([english_x, german_x], y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=validation_data, callbacks=callbacks)
    model.fit_generator(
        generate_lstm_batches(english_x, german_x, y), 
        epochs=epochs, steps_per_epoch=1, 
        callbacks=callbacks, validation_data=validation_generator, verbose=1
    )

    return model

def eval_model(x_test_english, x_test_german, y_test, model):
    score = model.evaluate([x_test_english, x_test_german], y_test)
    print(score)
    return score


def generate_lstm_batches(english_x, german_x, y):
    print("reached")
    while True:
        english_dim = (64, 10, 100)
        german_dim = (64, 12, 100)
        english_batch = np.zeros(english_dim)
        german_batch = np.zeros(german_dim)
        answer_dim = (64, )
        answer_batch = np.zeros(answer_dim)
        print("Reached")
        yield ([english_batch, german_batch], answer_batch)



class DataGenerator(Sequence):
    
    def __init__(self, src_x, target_x, y, dim=100, batch_size=64, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        ''' Number of batches per epoch '''
        pass # TODO: Implement

    def __getitem__(self, index):
        ''' Generate 1 batch of data '''
        pass # TODO: Implement

    def on_epoch_end(self):
        ''' Updates indexes after each epoch (if shuffle enabled) '''
        if self.shuffle:
            pass # TODO: Implement shuffle

    def __data_generation(self):
        ''' Generates data containing @batch_size samples '''
        pass # TODO: Implement

