# Setup
1. Install necessary dependencies from ```requirements.txt``` file.
2. Run ```$ git submodule update --init``` to fetch submodules locally.
3. Get the word2vec embeddings from http://vectors.nlpl.eu/repository/
    * English CoNLL17 corpus
: ```$ wget http://vectors.nlpl.eu/repository/20/40.zip```
    * German CoNLL17 corpus
: ```$ wget http://vectors.nlpl.eu/repository/20/45.zip```

# Structure

1. Scripts

We have 3 main scripts: train_mlp.py, train_conv.py and train_lstm.py. These can be used to train the MLP, CNN and LSTM architectures described in our report. 
Particularly:

```$ python3 train_mlp.py ``` is currently set up to perform a hyperparameter search, but can be used to train a specific MLP model.

```$ python3 train_mlp_augment.py ``` is set up in the same way as ```train_mlp.py```, but it searches/trains using the augmented dataset, as described in our report.

```$ python3 train_mlp_embeddings.py ``` is set up in the same way as ```train_mlp.py```, but it searches/trains while performing fine-tuning of the pretrained embeddings, as described in our report.

```$ python3 train_conv.py <train|test> ``` runs a hyperparameter search (```train``` option) or trains and tests (``` test ``` option) a CNN based model as described in our report.

```$ python3 train_lstm.py <train|test> ``` runs a hyperparameter search (```train``` option) or trains and tests (``` test ``` option) an LSTM based model as described in our report.

2. Library (lib/)

 * lib/embeddings
 * lib/utils
 * lib/data
 * lib/mlp
 * lib/conv
 * lib/lstm
 * lib/pos

# Predictions
Predictions for our best models are provided in the predictions_\<test_pearson_score\>.txt files.
