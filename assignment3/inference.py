import sys
from gensim.models import Word2Vec


def main(path):
    print("Reading in .txt file...")
    with open(path) as f:
        text = f.readlines()

    words = [w.strip() for w in text]
    print("Loading w2v.model from data directory...\n")
    w2v = Word2Vec.load('data/w2v.model')

    results = []
    for word in words:
        # finding most similar words to word and getting the first 20 words
        most_similar_words = w2v.wv.most_similar([w2v.wv[word]], topn=21)[1:]
        result = '{} => {}'.format(word, [x[0] for x in most_similar_words])
        results.append(result)

    return results


if __name__ == '__main__':
    similar_words = main(sys.argv[1])
    print('\n\n'.join(similar_words))
