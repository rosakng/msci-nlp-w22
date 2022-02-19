import sys
from gensim.models import Word2Vec


# helper function that reads in the path and combines positive and negative text files
def read_data(path_to_data):
    print("Reading in pos.txt and neg.txt files...")
    with open(path_to_data + '/pos.txt', "r") as f:
        pos_content = f.readlines()
    with open(path_to_data + '/neg.txt', "r") as f:
        neg_content = f.readlines()
    all_lines = pos_content + neg_content
    return list(zip(all_lines, [1]*len(pos_content) + [0]*len(neg_content)))


def run_script(path):
    data = read_data(path)
    all_lines = [line[0].strip().split() for line in data]
    print("Training Word2Vec model...")
    w2v = Word2Vec(all_lines, vector_size=100, window=5, min_count=1, workers=4)
    print("Saving model...")
    w2v.save('data/w2v.model')
    print("Saved w2v.model in data directory")


if __name__ == '__main__':
    run_script(sys.argv[1])
