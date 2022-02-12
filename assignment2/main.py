import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

# Global variables for classifier types
MNB_UNI = "mnb_uni"
MNB_BI = "mnb_bi"
MNB_UNI_BI = "mnb_uni_bi"
MNB_UNI_NS = "mnb_uni_ns"
MNB_BI_NS = "mnb_bi_ns"
MNB_UNI_BI_NS = "mnb_uni_bi_ns"

CLASSIFIERS_NGRAM = {
    # ngram_range = (x, y) where x is the minimum and y is the maximum size of the ngrams to include
    MNB_UNI: (1, 1),
    MNB_BI: (2, 2),
    MNB_UNI_BI: (1, 2),
    MNB_UNI_NS: (1, 1),
    MNB_BI_NS: (2, 2),
    MNB_UNI_BI_NS: (1, 2)
}


def read_csv(path):
    with open(path) as f:
        data = f.readlines()
    # read csv and return as a joined line
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


def train_data(X_train, X_val_or_test, y_train, classifier_type, tuner):
    ngram_range = CLASSIFIERS_NGRAM[classifier_type]
    count_vector = CountVectorizer(ngram_range=ngram_range)  # converts document to a matrix of token counts
    tfidf_transformer = TfidfTransformer()  # transforms and normalizes token count matrix to tf-idf (smoothing)

    # fit training data documents and transform into a document-term matrix where the row is a document
    # and column is a word with frequency of the word (1 = word in column exists in the row, 0 otherwise)
    # use tfidf transformer on large collection of documents to normalize for tokens that are more common than others
    training_data = tfidf_transformer.fit_transform(count_vector.fit_transform(X_train))

    # transform the validation or test set to be used to calculate score later
    testing_data = tfidf_transformer.transform(count_vector.transform(X_val_or_test))

    # use the Multinomial Naive Bayes classifier and fit the data
    model = MultinomialNB(alpha=tuner).fit(training_data, y_train)

    return model, count_vector, tfidf_transformer, testing_data


def do_the_pickle(name, to_pkl):
    path = 'data/' + name + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(to_pkl, f)


def run_script(use_validation_set, alpha, classifier_type, data):

    # no stop words
    if "_ns" in classifier_type:
        x_ns_train = data.get("no_stopwords").get("x").get("train")
        y_ns_train = data.get("no_stopwords").get("y").get("train")

        y_ns_val = data.get("no_stopwords").get("y").get("val")
        x_ns_val = data.get("no_stopwords").get("x").get("val")

        x_ns_test = data.get("no_stopwords").get("x").get("test")
        y_ns_test = data.get("no_stopwords").get("y").get("test")

        if use_validation_set:
            print("Running model on validation set for no stop words data and classifier type: ", classifier_type)
            model_ns, count_vectorizer_ns, tfidf_transformer_ns, val_data_ns = train_data(x_ns_train, x_ns_val,
                                                                                            y_ns_train, classifier_type,
                                                                                            alpha)
            do_the_pickle(classifier_type, model_ns)
            do_the_pickle("count_vectorizer_" + classifier_type, count_vectorizer_ns)
            do_the_pickle("tfidf_transformer_" + classifier_type, tfidf_transformer_ns)
            return model_ns.score(val_data_ns, y_ns_val),
        else:
            print("Running model on test set for no stop words data and classifier type: ", classifier_type)
            model_ns, count_vectorizer_ns, tfidf_transformer_ns, test_data_ns = train_data(x_ns_train, x_ns_test,
                                                                                                 y_ns_train,
                                                                                                 classifier_type, alpha)
            do_the_pickle(classifier_type, model_ns)
            do_the_pickle("count_vectorizer_" + classifier_type, count_vectorizer_ns)
            do_the_pickle("tfidf_transformer_" + classifier_type, tfidf_transformer_ns)
            return model_ns.score(test_data_ns, y_ns_test)

    else:
        x_train = data.get("with_stopwords").get("x").get("train")
        y_train = data.get("with_stopwords").get("y").get("train")

        y_val = data.get("with_stopwords").get("y").get("val")
        x_val = data.get("with_stopwords").get("x").get("val")

        x_test = data.get("with_stopwords").get("x").get("test")
        y_test = data.get("with_stopwords").get("y").get("test")

        if use_validation_set:
            print("Running model on validation set for data with stop words and classifier type: ", classifier_type)
            model, count_vectorizer, tfidf_transformer, val_data = train_data(x_train, x_val, y_train,
                                                                                classifier_type,
                                                                                alpha)
            do_the_pickle(classifier_type, model)
            do_the_pickle("count_vectorizer_" + classifier_type, count_vectorizer)
            do_the_pickle("tfidf_transformer_" + classifier_type, tfidf_transformer)

            return model.score(val_data, y_val)
        else:
            print("Running model on test set for data with stop words and classifier type: ", classifier_type)
            model, count_vectorizer, tfidf_transformer, test_data = train_data(x_train, x_test, y_train,
                                                                                     classifier_type, alpha)
            do_the_pickle(classifier_type, model)
            do_the_pickle("count_vectorizer_" + classifier_type, count_vectorizer)
            do_the_pickle("tfidf_transformer_" + classifier_type, tfidf_transformer)
            return model.score(test_data, y_test)


if __name__ == '__main__':
    # get all datasets once
    data = get_all_data(sys.argv[1])

    # set this boolean to true when we want to tune our model with the validation set
    # set this boolean to false when we want to use the test dataset to calculate final accuracy scores
    useValidationSet = False
    # set alpha value
    alpha = 0.5
    classification_types = [MNB_UNI, MNB_BI, MNB_UNI_BI, MNB_UNI_NS, MNB_BI_NS, MNB_UNI_BI_NS]

    result_data = []
    for classifier_type in classification_types:
        print("Using alpha: ", alpha, " for classification type: ", classifier_type)
        score = run_script(useValidationSet, alpha, classifier_type, data)
        alpha_result = {
            classifier_type: score
        }
        result_data.append(alpha_result)
    alpha_object = {
        str(alpha): result_data
    }
    print(alpha_object)

    # ex. python3 main.py /Users/rosakang/workspace/msci-nlp-w22/assignment1/data
