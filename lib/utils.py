import os
import tensorflow as tf
from tensorflow.keras import backend as K
import random

def pearsonr(x, y):
    """
    Computes pearson score given predictions and targets
    """
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    return r_num / r_den

    # return K.maximum(K.minimum(r, 1.0), -1.0)

# Paths for models and POS cached results
__current_file_path = os.path.dirname(os.path.realpath(__file__))
DATASETS_BASE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'datasets/'))
MODELS_SAVE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'saved_models/'))
SAVED_POS_BASE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'pos_tags/'))

POS_TAGGERS_BASE_PATH = '/vol/bitbucket/eb1816/nlp_cw/pos/stanford-postagger-full-2018-10-16/'
POS_TAGGERS_JAR_PATH = os.path.join(POS_TAGGERS_BASE_PATH, 'stanford-postagger.jar')
POS_TAGGERS_EN_MODEL_PATH = os.path.join(POS_TAGGERS_BASE_PATH, 'models', 'wsj-0-18-bidirectional-distsim.tagger')
POS_TAGGERS_DE_MODEL_PATH = os.path.join(POS_TAGGERS_BASE_PATH, 'models', 'german-hgc.tagger')

# Max lengths of sentences in datasets after removal of unknown words
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

PAD_TOK = "<PAD>"

def get_config(params):
    return {p: params[p][random.randint(0, len(params[p]) - 1)] for p in params}
