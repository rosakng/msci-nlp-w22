import pickle
import sys

import keras.models
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from config import config
import os


def main(path, activation):
    print("Loading tokenizer from data directory...\n")
    tokenizer_pkl = 'data/tokenizer.pkl'
    print("Loading data from {}".format(tokenizer_pkl))
    with open(tokenizer_pkl, 'rb') as f:
        tokenizer = pickle.load(f)

    print("Reading in .txt file from path: {}".format(path))
    with open(path) as f:
        text = f.readlines()
        print(text)

    word_seq = [text_to_word_sequence(sent) for sent in text]
    x_predict = tokenizer.texts_to_sequences([' '.join(seq[:config['max_seq_len']]) for seq in word_seq])
    x_predict = pad_sequences(x_predict, maxlen=config['max_seq_len'], padding='post', truncating='post')

    print("Loading nn_{}.model from data directory...\n".format(activation))
    model_file_name = 'nn_' + activation + '.model'
    model = keras.models.load_model(os.path.join('data/', model_file_name))

    r = model.predict(x_predict)
    print(r)
    classes = np.argmax(r, axis=-1)
    print(classes)
    results = []
    for i in classes:
        if i==0:
            results.append('NEGATIVE')
        else:
            results.append('POSITIVE')
    print(results)
    return classes


if __name__ == '__main__':
    path = sys.argv[1]
    activation = sys.argv[2]
    main(path, activation)
