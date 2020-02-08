import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.utils import MODELS_SAVE_PATH, EVALUATION_METRICS, MODEL_PATIENCE
import numpy as np


def build_compile_model(learning_rate, layers, dropout):
    """
    Builds a MLP model
    """
    model = Sequential()
    model.add(Dense(units=layers[0], activation="relu", input_dim=200))
    for l in layers[1:]:
        model.add(Dense(units=l, activation="relu"))
        model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=EVALUATION_METRICS
    )
    return model

def fit_model(x, y, x_val, y_val, batch_size, epochs, learning_rate, name, layers, dropout, seed=2019):
    """
    Builds, compiles and trains model on given dataset
    x: size (7000, 200)
    y: size (7000,)
    """
    np.random.seed(seed)
    # apparently, this version is recommended as just set_random_seed is deprecated
    tensorflow.compat.v1.set_random_seed(seed)
    model = build_compile_model(learning_rate, layers, dropout)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MODEL_PATIENCE, verbose=1, restore_best_weights=True),
        ModelCheckpoint(f"{MODELS_SAVE_PATH}/{name}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    ]
    validation_data = None
    if x_val is not None and y_val is not None:
        validation_data = [x_val, y_val]

    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=validation_data, callbacks=callbacks)
    return model, history

def eval_model(x_test, y_test, model):
    score = model.evaluate(x_test, y_test)
    print(score)
    return score
