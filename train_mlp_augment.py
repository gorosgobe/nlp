import lib.data
import lib.embeddings
import numpy as np
import lib.mlp
from lib.utils import get_config, MODEL_PATIENCE
import random
import os
import csv

HYPERPARAM_SEARCH_FILE = "results_mlp.csv"

if __name__ == "__main__":
    print("Loading training data...")
    train_source, train_translation, train_scores = lib.data.load_data(data_type=lib.data.DatasetType.TRAIN, target_language=lib.data.Language.GERMAN, augmented=True)

    sources_tok, source_vocab = lib.data.tokenize(train_source)
    translation_tok, translation_vocab = lib.data.tokenize(train_translation)

    #TODO: downsample these to augmented vocab
    ENGLISH_EMBEDDING_MODEL = "/vol/bitbucket/eb1816/nlp_cw/embeddings/english/model.bin"
    GERMAN_EMBEDDING_MODEL = "/vol/bitbucket/eb1816/nlp_cw/embeddings/german/model.bin"

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

    english_average_sentence_embeddings = lib.embeddings.get_sentence_embeddings(english_vectors)
    german_average_sentence_embeddings = lib.embeddings.get_sentence_embeddings(german_vectors)
    assert english_average_sentence_embeddings.shape == german_average_sentence_embeddings.shape

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


    val_english_average_sentence_embeddings = lib.embeddings.get_sentence_embeddings(val_english_vectors)
    val_german_average_sentence_embeddings = lib.embeddings.get_sentence_embeddings(val_german_vectors)
    assert val_english_average_sentence_embeddings.shape == val_german_average_sentence_embeddings.shape

    print("Concatenating for training")
    embeddings = np.concatenate((english_average_sentence_embeddings, german_average_sentence_embeddings), axis=1)

    print("Concatenating for validation")
    val_embeddings = np.concatenate((val_english_average_sentence_embeddings, val_german_average_sentence_embeddings), axis=1)

    print("Loading test data...")
    test_source, test_translation, _ = lib.data.load_data(data_type=lib.data.DatasetType.TEST, target_language=lib.data.Language.GERMAN)
    test_src_tok, test_src_vocab = lib.data.tokenize(test_source)
    test_trans_tok, test_trans_vocab = lib.data.tokenize(test_translation)

    print("Computing test english word embeddings")
    test_english_vectors, _ = lib.embeddings.get_embeddings(
        english_embedding_model,
        test_src_tok,
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    print("Computing test german word embeddings")
    test_german_vectors,  _ = lib.embeddings.get_embeddings(
        german_embedding_model,
        test_trans_tok,
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    test_english_avg_sentence_embeddings = lib.embeddings.get_sentence_embeddings(test_english_vectors) 
    test_german_avg_sentence_embeddings = lib.embeddings.get_sentence_embeddings(test_german_vectors)
    assert test_english_avg_sentence_embeddings.shape == test_german_avg_sentence_embeddings.shape

    if True:
        params = {
            "learning_rate": [0.000001 * x for x in range(1000)],
            "epochs": [250],
            "batch_size": [32, 64, 128, 256, 512],
            "dropout": [0.01 * x for x in range(50)],
            "layers": [random.sample([64, 128, 256, 512, 1024], random.randint(1, 4)) for _ in range(1000)]
        }

        print("Hyperparameter search")

        for _ in range(20000):
            sampled_params = get_config(params)
            print("Configuration:")
            print(sampled_params)

            model, history = lib.mlp.fit_model(
                embeddings,
                train_scores,
                batch_size=sampled_params['batch_size'],
                epochs=sampled_params['epochs'],
                learning_rate=sampled_params['learning_rate'],
                x_val=val_embeddings,
                y_val=val_scores,
                name='mlp_model_best',
                layers=sampled_params["layers"],
                dropout=sampled_params["dropout"]
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
        print("Evaluation: Combine train and val data for re-training")
        train_source = np.concatenate((train_source, val_source), axis=0)
        train_translation = np.concatenate((train_translation, val_translation), axis=0)
        train_scores = np.concatenate((train_scores, val_scores), axis=0)

        sources_tok, source_vocab = lib.data.tokenize(train_source)
        translation_tok, translation_vocab = lib.data.tokenize(train_translation)

        # training vectors
        print("Computing combined training english word embeddings...")
        english_vectors,  _ignored_english_words = lib.embeddings.get_embeddings(
            english_embedding_model,
            sources_tok,
            lib.embeddings.EmbeddingType.WORD2VEC
        )

        print("Computing combined training german word embeddings...")
        german_vectors,  _ignored_german_words = lib.embeddings.get_embeddings(
            german_embedding_model,
            translation_tok,
            lib.embeddings.EmbeddingType.WORD2VEC
        )

        english_average_sentence_embeddings = lib.embeddings.get_sentence_embeddings(english_vectors)
        german_average_sentence_embeddings = lib.embeddings.get_sentence_embeddings(german_vectors)
        assert english_average_sentence_embeddings.shape == german_average_sentence_embeddings.shape

        print("Concatenating for training")
        embeddings = np.concatenate((english_average_sentence_embeddings, german_average_sentence_embeddings), axis=1)
        
        print("Concatenating for testing")
        test_embeddings = np.concatenate((test_english_avg_sentence_embeddings, test_german_avg_sentence_embeddings), axis=1)

        model, _ = lib.mlp.fit_model(
            embeddings,
            train_scores,
            batch_size=128,
            epochs=12,
            learning_rate=0.000484,
            x_val=None,
            y_val=None,
            name='mlp_best_tuned_model',
            layers=[512, 128, 64, 256],
            dropout=0.33,
            verbose=1
        )

        predictions = model.predict(test_embeddings)
        np.savetxt('predictions.txt', predictions, delimiter=',', fmt='%f')

