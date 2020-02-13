import lib.data
import lib.embeddings
import numpy as np
import lib.lstm
import random
import csv
import os
from lib.utils import get_config, MODEL_PATIENCE

HYPERPARAM_SEARCH_FILE = "results_lstm.csv"

if __name__ == "__main__":

    print("Loading training data...")
    train_source, train_translation, train_scores = lib.data.load_data(data_type=lib.data.DatasetType.TRAIN, target_language=lib.data.Language.GERMAN)

    sources_tok, source_vocab = lib.data.tokenize(train_source)
    translation_tok, translation_vocab = lib.data.tokenize(train_translation)
    # import pdb
    # pdb.set_trace()
    ENGLISH_EMBEDDING_MODEL = "embeddings/en_model_downsampled.bin"
    GERMAN_EMBEDDING_MODEL = "embeddings/de_model_downsampled.bin"

    print("Loading english embedding model...")
    english_embedding_model = lib.embeddings.load_embedding(ENGLISH_EMBEDDING_MODEL, lib.embeddings.EmbeddingType.WORD2VEC)
    print("English model loaded")
    print("Loading german embedding model...")
    german_embedding_model = lib.embeddings.load_embedding(GERMAN_EMBEDDING_MODEL, lib.embeddings.EmbeddingType.WORD2VEC)
    print("German model loaded")

    # training vectors
    print("Training data processing...")
    print("Computing training english word embeddings...")
    english_vectors,  _ignored_english_words = lib.embeddings.get_embeddings(
        english_embedding_model,
        sources_tok,
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    print("Computing training german word embeddings...")
    german_vectors,  _ignored_german_words = lib.embeddings.get_embeddings(
        german_embedding_model,
        translation_tok,
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    # validation vectors
    print("Validation data processing...")
    print("Computing validation english word embeddings...")
    val_source, val_translation, val_scores = lib.data.load_data(data_type=lib.data.DatasetType.VAL, target_language=lib.data.Language.GERMAN)

    val_sources_tok, val_source_vocab = lib.data.tokenize(val_source)
    val_translation_tok, val_translation_vocab = lib.data.tokenize(val_translation)
    val_english_vectors,  ignored_val_english_words = lib.embeddings.get_embeddings(
        english_embedding_model,
        val_sources_tok,
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    print("Computing validation german word embeddings...")
    val_german_vectors,  ignored_val_german_words = lib.embeddings.get_embeddings(
        german_embedding_model,
        val_translation_tok,
        lib.embeddings.EmbeddingType.WORD2VEC
    )
    print(f"Ignored words, english {len(ignored_val_english_words)}, german {len(ignored_val_english_words)}")

    if True:

        print("Hyperparameter search")

        params = {
            "learning_rate": [0.000001 * x for x in range(1000)],
            "epochs": [250],
            "english_lstm": [32, 64, 128, 256, 512],
            "german_lstm": [32, 64, 128, 256, 512],
            "batch_size": [32, 64, 128, 256, 512],
            "dropout": [0.01 * x for x in range(50)],
            "dropout_lstm": [0.01 * x for x in range(50)],
            "layers": [random.sample([64, 128, 256, 512, 1024], random.randint(1, 4)) for _ in range(1000)]
        }

        for _ in range(250):
            sampled_params = get_config(params)
            print("Configuration:")
            print(sampled_params)

            model, history = lib.lstm.fit_model(
                english_x=english_vectors,
                german_x=german_vectors,
                y=train_scores,
                batch_size=sampled_params['batch_size'],
                epochs=sampled_params['epochs'],
                learning_rate=sampled_params['learning_rate'],
                english_x_val=val_english_vectors,
                german_x_val=val_german_vectors,
                y_val=val_scores,
                name='lstm_model_best',
                layers=sampled_params["layers"],
                dropout=sampled_params["dropout"],
                english_lstm_units=sampled_params["english_lstm"],
                german_lstm_units=sampled_params["german_lstm"],
                dropout_lstm=sampled_params["dropout_lstm"],
                bidirectional=True
            )

            print(history.history["val_mean_squared_error"][-MODEL_PATIENCE])
            val_mean_squared_error = history.history["val_mean_squared_error"][-MODEL_PATIENCE]
            val_pearsonr          =  history.history["val_pearsonr"][-MODEL_PATIENCE]
            val_root_mean_squared_error = history.history["val_root_mean_squared_error"][-MODEL_PATIENCE]
            val_mae = history.history["val_mae"][-MODEL_PATIENCE]
            h = {
                "val_mean_squared_error": float(val_mean_squared_error),
                "val_pearsonr": float(val_pearsonr),
                "val_rmse": float(val_root_mean_squared_error),
                "val_mae": float(val_mae),
                "best_epoch": len(history.history["val_mean_squared_error"]) - MODEL_PATIENCE,
                **sampled_params
            }

            file_exists = os.path.exists(HYPERPARAM_SEARCH_FILE)
            with open(HYPERPARAM_SEARCH_FILE, "a") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(h.keys()), dialect="unix")

                if not file_exists:
                    writer.writeheader()
            
                writer.writerow(h)
    else:
        print("Training model")
        model, _ = lib.lstm.fit_model(
                english_x=english_vectors,
                german_x=german_vectors,
                y=train_scores,
                batch_size=256,
                epochs=17,
                learning_rate=0.00094,
                english_x_val=val_english_vectors,
                german_x_val=val_german_vectors,
                y_val=val_scores,
                name='lstm_model_best',
                layers=[256, 64],
                dropout=0.38,
                english_lstm_units=512,
                german_lstm_units=32,
                dropout_lstm=0.28,
                bidirectional=True,
                verbose=1
            )
