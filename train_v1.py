
import lib.data
import lib.embeddings
import numpy as np
import lib.mlp

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
        ENGLISH_EMBEDDING_MODEL, 
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    print("Computing training german word embeddings...")
    german_vectors,  _ignored_german_words = lib.embeddings.get_embeddings(
        german_embedding_model,
        translation_tok, 
        GERMAN_EMBEDDING_MODEL, 
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
        ENGLISH_EMBEDDING_MODEL, 
        lib.embeddings.EmbeddingType.WORD2VEC
    )

    print("Computing validation german word embeddings...")
    val_german_vectors,  ignored_val_german_words = lib.embeddings.get_embeddings(
        german_embedding_model,
        val_translation_tok, 
        GERMAN_EMBEDDING_MODEL, 
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

    print("Training model")
    model = lib.mlp.fit_model(embeddings, train_scores, batch_size=64, epochs=500, x_val=val_embeddings, y_val=val_scores, name='mlp_model_best')
