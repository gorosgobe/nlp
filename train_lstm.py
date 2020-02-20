import lib.data
import lib.embeddings
import numpy as np
import lib.lstm
import random
import csv
import os
from lib.utils import get_config, MODEL_PATIENCE
import sys

HYPERPARAM_SEARCH_FILE = "results_lstm.csv"

if __name__ == "__main__":

    ENGLISH_EMBEDDING_MODEL = "embeddings/en_model_downsampled.bin"
    GERMAN_EMBEDDING_MODEL = "embeddings/de_model_downsampled.bin"

    print("Loading english embedding model...")
    english_embedding_model = lib.embeddings.load_embedding(ENGLISH_EMBEDDING_MODEL, lib.embeddings.EmbeddingType.WORD2VEC)
    print("English model loaded")
    
    print("Loading german embedding model...")
    german_embedding_model = lib.embeddings.load_embedding(GERMAN_EMBEDDING_MODEL, lib.embeddings.EmbeddingType.WORD2VEC)
    print("German model loaded")

    print("Loading training data...")
    train_source, train_translation, train_scores = lib.data.load_data(data_type=lib.data.DatasetType.TRAIN, target_language=lib.data.Language.GERMAN)

    train_sources_tok = lib.data.tokenize(train_source)
    train_translation_tok = lib.data.tokenize(train_translation)

    train_source_input = lib.embeddings.get_embedding_input(data_tok=train_sources_tok,
                                                            model=english_embedding_model)
    train_translation_input = lib.embeddings.get_embedding_input(data_tok=train_translation_tok,
                                                                 model=german_embedding_model)      


    # validation vectors
    print("Validation data processing...")
    print("Computing validation english word embeddings...")
    val_source, val_translation, val_scores = lib.data.load_data(data_type=lib.data.DatasetType.VAL, target_language=lib.data.Language.GERMAN)

    val_sources_tok = lib.data.tokenize(val_source)
    val_translation_tok = lib.data.tokenize(val_translation)

    val_source_input = lib.embeddings.get_embedding_input(data_tok=val_sources_tok, 
                                                          model=english_embedding_model)
    val_translation_input = lib.embeddings.get_embedding_input(data_tok=val_translation_tok, 
                                                                model=german_embedding_model)


    print("Loading test data...")
    test_source, test_translation, _ = lib.data.load_data(data_type=lib.data.DatasetType.TEST, target_language=lib.data.Language.GERMAN)
    test_src_tok = lib.data.tokenize(test_source)
    test_trans_tok = lib.data.tokenize(test_translation)

    test_source_input = lib.embeddings.get_embedding_input(data_tok=test_src_tok,
                                                            model=english_embedding_model)
    test_translation_input = lib.embeddings.get_embedding_input(data_tok=test_trans_tok,
                                                                 model=german_embedding_model) 

    if 'train' in sys.argv:

        print("Hyperparameter search")

        params = {
            "learning_rate": [0.000001 * x for x in range(1000)],
            "epochs": [500],
            "english_lstm": [32, 64, 128, 256, 512],
            "german_lstm": [32, 64, 128, 256, 512],
            "batch_size": [32, 64, 128, 256, 512],
            "dropout": [0.01 * x for x in range(50)],
            "dropout_lstm": [0.01 * x for x in range(50)],
            "layers": [random.sample([32, 64, 128, 256, 512], random.randint(1, 4)) for _ in range(1000)],
            "bidirectional": [True, False],
            "attention": [True]
        }

        for _ in range(250):
            sampled_params = get_config(params)
            print("Configuration:")
            print(sampled_params)

            model, history = lib.lstm.fit_model(
                english_x=train_source_input,
                german_x=train_translation_input,
                english_w2v=english_embedding_model,
                german_w2v=german_embedding_model,
                y=train_scores,
                batch_size=sampled_params['batch_size'],
                epochs=sampled_params['epochs'],
                learning_rate=sampled_params['learning_rate'],
                english_x_val=val_source_input,
                german_x_val=val_translation_input,
                y_val=val_scores,
                name='lstm_model_best',
                layers=sampled_params["layers"],
                dropout=sampled_params["dropout"],
                english_lstm_units=sampled_params["english_lstm"],
                german_lstm_units=sampled_params["german_lstm"],
                dropout_lstm=sampled_params["dropout_lstm"],
                bidirectional=sampled_params["bidirectional"],
                verbose=1,
                attention=sampled_params["attention"]
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

    elif 'test' in sys.argv:
        print("Evaluation: Combine train and val data for re-training")
        train_source = np.concatenate((train_source, val_source), axis=0)
        train_translation = np.concatenate((train_translation, val_translation), axis=0)
        train_scores = np.concatenate((train_scores, val_scores), axis=0)

        train_sources_tok = lib.data.tokenize(train_source)
        train_translation_tok = lib.data.tokenize(train_translation)

        train_source_input = lib.embeddings.get_embedding_input(data_tok=train_sources_tok,
                                                                model=english_embedding_model)
        train_translation_input = lib.embeddings.get_embedding_input(data_tok=train_translation_tok,
                                                                    model=german_embedding_model)  

        print("Training model")
        model, _ = lib.lstm.fit_model(
            english_x=train_source_input,
            german_x=train_translation_input,
            english_w2v=english_embedding_model,
            german_w2v=german_embedding_model,
            y=train_scores,
            batch_size=128,
            epochs= 59,
            learning_rate=0.000015,
            english_x_val=val_source_input,
            german_x_val=val_translation_input,
            y_val=val_scores,
            name='lstm_model_best',
            layers=[512, 128, 64, 32],
            dropout=0.22,
            english_lstm_units=128,
            german_lstm_units=512,
            dropout_lstm=0.39,
            bidirectional=True,
            verbose=1
        )

        predictions = model.predict({'english_input': test_source_input, 'german_input': test_translation_input})
        np.savetxt('predictions.txt', predictions, delimiter=',', fmt='%f')
