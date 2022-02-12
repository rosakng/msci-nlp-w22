**Instructions**

*Running main.py*

Please cd into `msci-nlp-w22/assignment2` and run `python3 main.py {path}` where `path` is the path to the directory 
containing the split from A1.

*Running inference.py*

Please ensure that the outputs(.pkl) files have already been generated in the `msci-nlp-w22/assignment2/data` directory.
You can do this by running the main script first.

Please cd into `msci-nlp-w22/assignment2` and run `python3 inference.py {path} {classifier_type}` where `path` is the 
path to .txt file and `classifier_type` is the type of classifier to use.
i.e. `mnb_uni` for MultinomialNB classifier trained on unigram features with stopwords for classifying sentences in the .txt file

***
**Alpha: 0.5**

|Stopwords removed | Text features  | Accuracy (test set) |
| :---: | :---: | :---: |
| yes | unigrams | 0.805675 |
| yes | bigrams | 0.7789875 |
| yes | unigrams+bigrams | 0.8235125 |
| no | unigrams | 0.807 |
| no | bigrams | 0.8245875 |
| no | unigrams+bigrams | 0.833775 |