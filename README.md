# Setup
1. Clone the repository.
2. On a lab machine, run ```$ source /vol/bitbucket/eb1816/venv/bin/activate```. We suggest running it on a lab machine since the virtual environment, embedding models and POS taggers are all configured to run on lab machines.  

# Structure

### 1. Scripts

We have 3 main scripts: ```train_mlp.py```, ```train_conv.py``` and ```train_lstm.py```. These can be used to train the MLP, CNN and LSTM architectures described in our report. 

Particularly:

```$ python3 train_mlp.py <train|test> ``` runs a hyperparameter search (```train``` option) or trains and tests (```test``` option) an MLP-based model as described in our report.

```$ python3 train_mlp_augment.py <train|test>``` is set up in the same way as ```train_mlp.py```, but it searches/trains using the augmented dataset, as described in our report.

```$ python3 train_mlp_embeddings.py ``` is set up similarly to ```train_mlp.py```, but it loops through the best MLP configurations to see if trainable embeddings yield substantial performance increases, as described in our report.

```$ python3 train_conv.py <train|test> ``` runs a hyperparameter search (```train``` option) or trains and tests (``` test ``` option) a CNN-based model as described in our report.

```$ python3 train_lstm.py <train|test> ``` runs a hyperparameter search (```train``` option) or trains and tests (``` test ``` option) an LSTM-based model as described in our report.

### 2. Library (lib/)

 * ```lib/embeddings```: Handles word embedding functionality
 * ```lib/utils```: Utilities
 * ```lib/data```: Handles loading and tokenising of datasets
 * ```lib/mlp```: Functions to build, compile and run a Multi-Layer Perceptron (MLP) based architecture
 * ```lib/conv```: Functions to build, compile and run CNN based models
 * ```lib/lstm```: Functions to build, compile and run LSTM based models
 * ```lib/pos```: Handles POS tagging

# Predictions
Predictions for our best models are provided in the predictions_\<test_pearson_score\>.txt files.
