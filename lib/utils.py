import os
import tensorflow as tf
from tensorflow.keras import backend as K
import random

EPSILON = 10e-6

def pearsonr(x, y):
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / (r_den + EPSILON)
    return K.maximum(K.minimum(r, 1.0), -1.0)

def pearson_loss(x, y):
    return 1 - pearsonr(x, y)


__current_file_path = os.path.dirname(os.path.realpath(__file__))
DATASETS_BASE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'datasets/'))
MODELS_SAVE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'saved_models/'))

CONSTANT_MAX_LENGTH_ENGLISH_TRAIN = 39
CONSTANT_MAX_LENGTH_GERMAN_TRAIN  = 39

CONSTANT_MAX_LENGTH_ENGLISH_VAL   = 37
CONSTANT_MAX_LENGTH_GERMAN_VAL    = 36

CONSTANT_MAX_LENGTH_ENGLISH_TEST   = 34
CONSTANT_MAX_LENGTH_GERMAN_TEST    = 54

# Maximum sequence length for english data
CONSTANT_MAX_LENGTH_ENGLISH = max(
    CONSTANT_MAX_LENGTH_ENGLISH_TRAIN,
    CONSTANT_MAX_LENGTH_ENGLISH_VAL,
    CONSTANT_MAX_LENGTH_ENGLISH_TEST
)

# Maximum sequence length for german data
CONSTANT_MAX_LENGTH_GERMAN = max(
    CONSTANT_MAX_LENGTH_GERMAN_TRAIN,
    CONSTANT_MAX_LENGTH_GERMAN_VAL,
    CONSTANT_MAX_LENGTH_GERMAN_TEST
)

BASE_PADDING = [0.0 for _ in range(100)]

EVALUATION_METRICS = ['mean_squared_error', "mae", tf.keras.metrics.RootMeanSquaredError(), pearsonr]

MODEL_PATIENCE = 25

def get_config(params):
    return {p: params[p][random.randint(0, len(params[p]) - 1)] for p in params}
