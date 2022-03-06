import os
import pickle
import sys
import numpy as np
from config import config
import tensorflow as tf
from gensim.models import Word2Vec

from keras import regularizers
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def read_csv(path):
    with open(path) as f:
        data = f.readlines()
    return [' '.join(x.strip().split(',')) for x in data]


def get_all_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))

    x_ns_train = read_csv(os.path.join(data_dir, 'train_ns.csv'))
    x_ns_val = read_csv(os.path.join(data_dir, 'val_ns.csv'))
    x_ns_test = read_csv(os.path.join(data_dir, 'test_ns.csv'))

    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]

    y_train = labels[:len(x_train)]
    y_test = labels[len(x_train): len(x_train) + len(x_test)]
    y_val = labels[-len(x_val):]

    y_ns_train = labels[:len(x_ns_train)]
    y_ns_test = labels[len(x_ns_train): len(x_ns_train) + len(x_ns_test)]
    y_ns_val = labels[-len(x_ns_val):]

    data = {
        "with_stopwords": {
            "x": {
                "train": x_train,
                "val": x_val,
                "test": x_test
            },
            "y": {
                "train": y_train,
                "val": y_val,
                "test": y_test
            }
        },
        "no_stopwords": {
            "x": {
                "train": x_ns_train,
                "val": x_ns_val,
                "test": x_ns_test
            },
            "y": {
                "train": y_ns_train,
                "val": y_ns_val,
                "test": y_ns_test
            }
        }
    }

    return data


def build_embedding_matrix(tokenizer, w2v):
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, config['embedding_dim'])) # +1 is b
    for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
        try:
            embeddings_vector = w2v.wv[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector
    return embeddings_matrix


def do_the_pickle(tokenizer):
    path = 'data/tokenizer.pkl'
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)


def main(path_directory, classifier):
    print("Loading data from path: ", path_directory)
    data = get_all_data(path_directory)

    stopwords_choice = "no_stopwords"
    print("Using stopwords choice: {}".format(stopwords_choice))
    x_train = data.get(stopwords_choice).get("x").get("train")
    y_train = data.get(stopwords_choice).get("y").get("train")
    x_test = data.get(stopwords_choice).get("x").get("test")
    y_test = data.get(stopwords_choice).get("y").get("test")
    x_val = data.get(stopwords_choice).get("x").get("val")
    y_val = data.get(stopwords_choice).get("y").get("val")

    # categorize y values into [1,0] or [0,1]s
    y_train = tf.keras.utils.to_categorical(np.asarray(y_train).astype('float32'))
    y_test = tf.keras.utils.to_categorical(np.asarray(y_test).astype('float32'))
    y_val = tf.keras.utils.to_categorical(np.asarray(y_val).astype('float32'))

    tokenizer = Tokenizer(num_words=config['max_vocab_size'])
    tokenizer.fit_on_texts(x_train)

    print("Saving tokenizer...")
    do_the_pickle(tokenizer)

    print("Number of words in vocabulary: {}".format(len(tokenizer.word_index)))

    # Preprocessing text using texts_to_sequences
    x_train = tokenizer.texts_to_sequences(x_train)
    # Padding text
    x_train = pad_sequences(x_train, maxlen=config['max_seq_len'], padding='post', truncating='post')

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=config['max_seq_len'], padding='post', truncating='post')

    x_val = tokenizer.texts_to_sequences(x_val)
    x_val = pad_sequences(x_val, maxlen=config['max_seq_len'], padding='post', truncating='post')

    # load pre-trained w2v model (this model was trained in assignment 3)
    print("Loading pre-train w2v model from assignment 3")
    w2v = Word2Vec.load(os.path.join(os.path.dirname(__file__), '../assignment3/data/w2v.model'))

    print('Building embedding matrix')
    # This matrix will be used to initialze weights in the embedding layer
    embedding_matrix = build_embedding_matrix(tokenizer, w2v)
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))

    print('Building model')

    # Build a sequential model and stacking neural net units
    model = Sequential()

    model.add(Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=config['embedding_dim'],
        input_length=config['max_seq_len'],
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=True,
        name='embedding_layer'))
    # flatted model's layer to into (batch, 1) shape
    model.add(Flatten())
    # add hidden layer with L2 norm regularization
    model.add(Dense(config['hidden_layer_dim'], activation=classifier, kernel_regularizer=regularizers.l2(config['l2_dim']),
                    name='hidden_layer'))
    # add dropout
    model.add(Dropout(config['dropout_rate']))
    # add output layer with softmax for probability distribution
    model.add(Dense(2, activation='softmax', name='output_layer'))

    print(model.summary())

    # use cross entropy as the loss function
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # fit model
    model.fit(x_train, y_train,
              batch_size=config['batch_size'],
              epochs=config['n_epochs'],
              validation_data=(x_val, y_val))

    # evaluate scores and accuracy
    score, acc = model.evaluate(x_test, y_test)
    print("Accuracy on Test Set = {0:4.3f}".format(acc))

    print("Saving model into data/ directory")
    model.save('data/nn_' + classifier + '.model')

    return model


if __name__ == '__main__':
    path_to_splits = sys.argv[1]
    classifier = sys.argv[2]
    main(path_to_splits, classifier)
