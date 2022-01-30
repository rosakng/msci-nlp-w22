import re

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
import sys
import random


def run_script():
    path_to_data = sys.argv[1]
    stop_words = stopwords.words('english')

    with open(path_to_data + '/pos.txt', "r") as f:
        pos_content = f.readlines()
    with open(path_to_data + '/neg.txt', "r") as f:
        neg_content = f.readlines()
    content = pos_content + neg_content
    random.shuffle(content)

    out_data = ''
    train_data = ''
    test_data = ''
    val_data = ''

    out_ns_data = ''
    train_ns_data = ''
    test_ns_data = ''
    val_ns_data = ''

    # set train, test, and val test sizes
    train_size = int(len(content) * 0.8)
    test_val_size = int(len(content) * 0.1)

    # read line by line or review by review
    for index, line in enumerate(content):
        tokenized_line = []
        # substitute special characters with spaces
        sub_special_chars = re.sub('[!"#$%&()*+/:;<=>@\[\]\\\\^`{|}~\t\n]', ' ', line.lower())
        # strip line by white spaces and split by white spaces and iterate
        for split in sub_special_chars.strip().split():
            # for each value, split by common punctuations such as ".", ",", "?" and "-"
            split_by_punctuation = re.split('[.*|?*|,*|-]', split)
            for word in split_by_punctuation:
                tokens = []
                # after splitting by punctuation, look for contractions in each word and split
                if len(word) > 0:
                    contractions = '\s|(n\'t)|\'m|(\'ll)|(\'ve)|(\'s)|(\'re)|(\'d)'
                    contracted = re.split(contractions, word)
                    tokens = [x for x in contracted if x]
                # collect all tokens for the tokenized review
                tokenized_line += tokens

        # if the tokenized line is not empty, format values into csv and add to respective data sets
        if tokenized_line:
            csv_line = '{}\n'.format(','.join(tokenized_line))
            out_data += csv_line
            tokenized_line_ns = [token for token in tokenized_line if not token in stop_words]
            csv_ns_line = '{}\n'.format(','.join(tokenized_line_ns))
            out_ns_data += csv_ns_line

            if index < train_size:
                train_data += csv_line
                train_ns_data += csv_ns_line
            elif train_size <= index < train_size + test_val_size:
                test_data += csv_line
                test_ns_data += csv_ns_line
            else:
                val_data += csv_line
                val_ns_data += csv_ns_line

    with open('data/out.csv', "w") as f:
        f.write(out_data)
    with open('data/val.csv', "w") as f:
        f.write(val_data)
    with open('data/train.csv', "w") as f:
        f.write(train_data)
    with open('data/test.csv', "w") as f:
        f.write(test_data)

    with open('data/out_ns.csv', "w") as g:
        g.write(out_ns_data)
    with open('data/val_ns.csv', "w") as g:
        g.write(val_ns_data)
    with open('data/train_ns.csv', "w") as g:
        g.write(train_ns_data)
    with open('data/test_ns.csv', "w") as g:
        g.write(test_ns_data)


if __name__ == '__main__':
    run_script()