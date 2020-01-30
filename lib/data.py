import enum
import numpy as np
import os.path

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
    
    base_path = '../datasets/'
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
    else:
        prefix = "test"
    
    src_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.src'))
    translation_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.mt'))
    score_file = os.path.abspath(os.path.join(path, f'{prefix}.{language}.scores'))
    
    src = load_text(src_file)
    translated = load_text(translation_file)
    scores = np.loadtxt(score_file)
    
    return src, translated, scores