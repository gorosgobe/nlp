import lib.data
import lib.embeddings
import numpy as np
import lib.mlp
from lib.utils import get_config, MODEL_PATIENCE, CONSTANT_MAX_LENGTH_ENGLISH, CONSTANT_MAX_LENGTH_GERMAN
import random
import os
import csv
import ast

HYPERPARAM_SEARCH_FILE = "results_mlp_embeddings.csv"

PEARSON_THRESHOLD = 0.17

if __name__ == "__main__":

    ENGLISH_EMBEDDING_MODEL = "embeddings/en_model_downsampled.bin"
    GERMAN_EMBEDDING_MODEL = "embeddings/de_model_downsampled.bin"

    print("Loading english embedding model...")
    english_embedding_model = lib.embeddings.load_embedding(ENGLISH_EMBEDDING_MODEL, lib.embeddings.EmbeddingType.WORD2VEC)

    print("Loading german embedding model...")
    german_embedding_model = lib.embeddings.load_embedding(GERMAN_EMBEDDING_MODEL, lib.embeddings.EmbeddingType.WORD2VEC)

    print("Loading training data...")
    train_source, train_translation, train_scores = lib.data.load_data(data_type=lib.data.DatasetType.TRAIN, target_language=lib.data.Language.GERMAN)
    train_sources_tok, _ = lib.data.tokenize(train_source)
    train_translation_tok, _ = lib.data.tokenize(train_translation)

    train_source_input = lib.embeddings.get_embedding_input(data_tok=train_sources_tok,
                                                            model=english_embedding_model,
                                                            max_sent_len=CONSTANT_MAX_LENGTH_ENGLISH)
    train_translation_input = lib.embeddings.get_embedding_input(data_tok=train_translation_tok,
                                                                model=german_embedding_model,
                                                                max_sent_len=CONSTANT_MAX_LENGTH_GERMAN)


    print("Loading validation data...")
    val_source, val_translation, val_scores = lib.data.load_data(data_type=lib.data.DatasetType.VAL, target_language=lib.data.Language.GERMAN)
    val_sources_tok, _ = lib.data.tokenize(val_source)
    val_translation_tok, _ = lib.data.tokenize(val_translation)

    val_source_input = lib.embeddings.get_embedding_input(data_tok=val_sources_tok,
                                                          model=english_embedding_model,
                                                          max_sent_len=CONSTANT_MAX_LENGTH_ENGLISH)
    val_translation_input = lib.embeddings.get_embedding_input(data_tok=val_translation_tok,
                                                                model=german_embedding_model,
                                                                max_sent_len=CONSTANT_MAX_LENGTH_GERMAN)

    print("Loading test data...")
    test_source, test_translation, _ = lib.data.load_data(data_type=lib.data.DatasetType.TEST, target_language=lib.data.Language.GERMAN)
    test_src_tok, _ = lib.data.tokenize(test_source)
    test_trans_tok, _ = lib.data.tokenize(test_translation)


    print("Hyperparameter search")

    params = []
    with open("results_mlp_v1_mse.csv") as f:
        reader = csv.DictReader(f, dialect="unix")

        # skip header
        next(reader)
        for row in reader:
            if row["val_pearsonr"] != "nan" and float(row["val_pearsonr"]) >= PEARSON_THRESHOLD:
                params.append({
                    "batch_size": int(row["batch_size"]),
                    "learning_rate": float(row["learning_rate"]),
                    "epochs": 500,
                    "dropout": float(row["dropout"]),
                    "layers": ast.literal_eval(row["layers"]),
                })

    print("{} results with pearson score above {}".format(len(params), PEARSON_THRESHOLD))

    for sampled_params in params:

        model, history = lib.mlp.fit_model_embedding_layer(
            english_x_train=train_source_input,
            german_x_train=train_translation_input,
            y_train=train_scores,
            english_x_val=val_source_input,
            german_x_val=val_translation_input,
            y_val=val_scores,
            english_w2v=english_embedding_model,
            german_w2v=german_embedding_model,
            batch_size=sampled_params["batch_size"],
            epochs=sampled_params["epochs"],
            learning_rate=sampled_params["learning_rate"],
            name="",
            verbose=2,
            layers=sampled_params["layers"],
            dropout=sampled_params["dropout"],
            train_embeddings=True,
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
