import sys
import pickle


def get_classification(line, model, count_vectorizer, tfidf_transformer):
    result = tfidf_transformer.transform(count_vectorizer.transform([line]))
    return 'Positive' if model.predict(result) else 'Negative'


def get_predictions(path, classifier_type):
    with open(path) as f:
        txt = f.readlines()
    txt = [line.strip() for line in txt]

    model_pkl = 'data/' + classifier_type + '.pkl'
    print("Loading data from ", model_pkl)
    with open(model_pkl, 'rb') as f:
        model = pickle.load(f)

    count_vectorizer_pkl = 'data/count_vectorizer_' + classifier_type + '.pkl'
    print("Loading data from ", count_vectorizer_pkl)
    with open(count_vectorizer_pkl, 'rb') as f:
        count_vectorizer = pickle.load(f)

    tfidf_transformer_pkl = 'data/tfidf_transformer_' + classifier_type + '.pkl'
    print("Loading data from ", tfidf_transformer_pkl)
    with open(tfidf_transformer_pkl, 'rb') as f:
        tfidf_transformer = pickle.load(f)

    return ['{} => {}'.format(line, get_classification(line, model, count_vectorizer, tfidf_transformer))
            for line in txt]


if __name__ == '__main__':
    predictions = get_predictions(sys.argv[1], sys.argv[2])
    print('\n'.join(predictions))

# ex. python3 inference.py /Users/rosakang/workspace/MSCI598/MSCI-tutorials/data/raw/sample.txt mnb_uni