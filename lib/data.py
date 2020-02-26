"""Handles loading and tokenising of datasets"""

import enum
import numpy as np
import os.path
import pickle
from tqdm import tqdm
import nltk
from nltk import WordPunctTokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from lib.utils import DATASETS_BASE_PATH, SAVED_POS_BASE_PATH
from lib.pos import get_pos_tags

class DatasetType(enum.Enum):
    """
    Represents the type of dataset
    """

    TRAIN = 0
    VAL = 1
    TEST = 2

class Language(enum.Enum):
    """
    Represents the dataset language
    """

    GERMAN = 0
    ENGLISH = 1
    CHINESE = 2

def load_text(path):
    """
    Given a path to csv file, loads the data and 
    returns it as a numpy array
    """

    with open(path) as f:
        read_text = f.read().splitlines()
    
    return np.array(read_text)
    

def load_data(data_type=DatasetType.TRAIN, target_language=Language.GERMAN, augmented=False):
    """
    Given the dataset type, target language and whether or not to use augmented data, 
    loads and returns numpy array representations of the source text, translation text and scores.
    """

    if target_language == Language.ENGLISH:
        raise ValueError("Target language cannot be english")
    
    base_path = DATASETS_BASE_PATH
    if target_language == Language.GERMAN:
        language_folder = "en-de" if not augmented else "en-de-aug"
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

    scores = None
    if data_type != DatasetType.TEST:
        score_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.scores'))
        scores = np.loadtxt(score_file)
    
    src = load_text(src_file)
    translated = load_text(translation_file)
    
    return src, translated, scores

def tokenize(text_array, use_pos=False, data_type=None, lang=None):
    """
    Given an array of sentences, returns:
        If use_pos:
            An array of tokenised sentences (where each tokenised sentence is an array of tokens) 
        else:
            An array of tokenised sentences (where each tokenised sentence is an array of tuples of (token, POS tag))
    NOTE: If use_pos is False, the rest of the kwargs are ignored
    """

    if use_pos:
        # Since POS tags take long to generate, use cached version if exists

        cache_path = None
        
        if data_type == DatasetType.TRAIN:
            cache_path = os.path.join(SAVED_POS_BASE_PATH, f'train-{lang}-pos.pickle')
        elif data_type == DatasetType.VAL:
            cache_path = os.path.join(SAVED_POS_BASE_PATH, f'val-{lang}-pos.pickle')
        elif data_type == DatasetType.TEST:
            cache_path = os.path.join(SAVED_POS_BASE_PATH, f'test-{lang}-pos.pickle')

        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as handle:
                sentences = pickle.load(handle)
            return sentences

    tokeniser = WordPunctTokenizer()

    sentences = []
    with tqdm(total=len(text_array)) as pbar:
        for sentence in text_array:
            tokens = tokeniser.tokenize(sentence)
            lower_cased_tokens = []
            for tok in tokens:
                tok_lower = tok.lower()
                lower_cased_tokens.append(tok_lower)
            
            if use_pos:
                # Store tokenised sentence i.e. arrays of (token, POS_TAG) tuples
                try:
                    sentences.append(get_pos_tags(lower_cased_tokens, lang))
                except:
                    sentences.append([get_pos_tags([tok], lang)[0] for tok in lower_cased_tokens])
            else:
                # Store tokenised sentence
                sentences.append(lower_cased_tokens)
            pbar.update(1)

    if use_pos:
        # Store POS tags to allow faster loading on next invocation
        with open(cache_path, 'wb') as handle:
            pickle.dump(sentences, handle)

    return sentences

def pad_to_length(word_embeddings, length, padding):
    """
    Given some data (word_embeddings or other), of shape (x, variable, dimensionality) 
    returns the data padded in the 2nd dimension to size length i.e. (x, length, dimensionality) 
    """

    for sentence in word_embeddings:
        num_to_append = length - len(sentence)
        assert num_to_append >= 0
        for _ in range(num_to_append):
            sentence.append(padding)
