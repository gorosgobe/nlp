import os

__current_file_path = os.path.dirname(os.path.realpath(__file__))
DATASETS_BASE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'datasets/'))
MODELS_SAVE_PATH = os.path.abspath(os.path.join(__current_file_path, '..', 'saved_models/'))