import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.utils import MODELS_SAVE_PATH
import numpy as np

def build_compile_model():
    """
    Builds a MLP model
    """
    model = Sequential()
    model.add(Dense(units=1000, activation="relu", input_dim=200))
    model.add(Dense(units=300, activation="relu"))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_squared_error', "mae"]
    )
    return model
    
def fit_model(x, y, batch_size, epochs, x_val, y_val, name, seed=2019):
    """
    Builds, compiles and trains model on given dataset
    x: size (7000, 200)
    y: size (7000,)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)
    model = build_compile_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True), 
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    validation_data = None
    if x_val is not None and y_val is not None:
        validation_data = [x_val, y_val]

    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=validation_data, callbacks=callbacks)
    return model

def eval_model(x_test, y_test, model):
    score = model.evaluate(x_test, y_test)
    print(score)
    return score