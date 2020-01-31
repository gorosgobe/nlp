import enum
import numpy as np
import os.path
import nltk
from nltk import WordPunctTokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from utils import DATASETS_BASE_PATH

class DatasetType(enum.Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class Language(enum.Enum):
    GERMAN = 0
    ENGLISH = 1
    CHINESE = 2

def load_text(path):
    with open(path) as f:
        read_text = f.read().splitlines()
    
    return np.array(read_text)
    

def load_data(data_type=DatasetType.TRAIN, target_language=Language.GERMAN):
    if target_language == Language.ENGLISH:
        raise ValueError("Target language cannot be english")
    
    base_path = DATASETS_BASE_PATH
    if target_language == Language.GERMAN:
        language_folder = "en-de"
        language = "ende"
        path = os.path.join(base_path, language_folder)
    else:
        language_folder = "en-zh"
        language = "enzh"
        path = os.path.join(base_path, language_folder)

    if data_type == DatasetType.TRAIN:
        prefix = "train"
    elif data_type == DatasetType.VAL:
        prefix = "dev"
    elif data_type == DatasetType.TEST:
        prefix = "test"
    
    src_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.src'))
    translation_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.mt'))

    if data_type != DatasetType.TEST:
        score_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.scores'))
        scores = np.loadtxt(score_file)
    else:
        scores = None
    
    src = load_text(src_file)
    translated = load_text(translation_file)

    
    return src, translated, scores

def tokenize(text_array):
    """
    >>> sentences, voc = tokenize(np.array(["Hello how are you?", "Thank you I'm fine", "yeah me too."]))
    >>> sentences 
    [['hello', 'how', 'are', 'you', '?'], ['thank', 'you', 'i', "'m", 'fine'], ['yeah', 'me', 'too', '.']]
    """

    tokeniser = WordPunctTokenizer()

    sentences = []
    vocabulary = set()
    for sentence in text_array:
        tokens = tokeniser.tokenize(sentence)
        lower_cased_tokens = []
        for tok in tokens:
            tok_lower = tok.lower()
            lower_cased_tokens.append(tok_lower)
            vocabulary.add(tok_lower)
        sentences.append(lower_cased_tokens)

    return sentences, vocabulary

"""
TODO: remove, should be able to run it from pytest
"""
if __name__ == "__main__":
    import doctest
    doctest.testmod()