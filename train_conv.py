import lib.data
import lib.embeddings
import numpy as np
import lib.conv
import random
from lib.utils import MODEL_PATIENCE

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

    # TODO: handle test lengths
    lib.data.pad_to_length(english_vectors, lib.utils.CONSTANT_MAX_LENGTH_ENGLISH_TRAIN, [0.0] * 100)
    lib.data.pad_to_length(german_vectors, lib.utils.CONSTANT_MAX_LENGTH_GERMAN_TRAIN, [0.0] * 100)
    lib.data.pad_to_length(val_english_vectors, lib.utils.CONSTANT_MAX_LENGTH_ENGLISH_TRAIN, [0.0] * 100)
    lib.data.pad_to_length(val_german_vectors, lib.utils.CONSTANT_MAX_LENGTH_GERMAN_TRAIN, [0.0] * 100)

    english_x_train = np.array(english_vectors)
    german_x_train = np.array(german_vectors)
    english_x_val = np.array(val_english_vectors)
    german_x_val  = np.array(val_german_vectors)


    for _ in range(3000):
        network_params = {}
        network_params["stride"] = random.choice([1,2])

        num_diff_filters = random.randint(3, 6)
        filter_size_range = range(2, 15)
        num_filter_range = range(1, 5)
        network_params["filter_sizes"] = random.sample(filter_size_range, k=num_diff_filters)
        network_params["filter_counts"] = random.choices(num_filter_range, k=num_diff_filters)

        network_params["dropout_rate"] = random.uniform(0, 0.5)
        network_params["pooling_type"] = random.choice(["max", "avg"])

        num_fc_layers = random.randint(1,3)
        hidden_layer_range = range(10, 100)
        network_params["fc_layers"] = random.choices(hidden_layer_range, k=num_fc_layers)

        network_params["learning_rate"] = 10 ** random.uniform(-2.5, -4)

        print("params: ", network_params)

        model, history = lib.conv.fit_model(
            english_x_train=english_x_train,
            german_x_train=german_x_train,
            y_train=train_scores,
            batch_size=32,
            epochs=500,
            english_x_val=english_x_val,
            german_x_val=german_x_val,
            y_val=val_scores,
            name="conv",
            network_params=network_params,
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
            **network_params
        }
        with open("results_conv.txt", "a") as f:
            f.write(json.dumps(h))
            f.write("\n")
