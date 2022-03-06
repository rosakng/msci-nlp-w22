**Instructions**
Please cd into `msci-nlp-w22/assignment4` and run `source venv/bin/activate` and `pip install -r requirements.txt ` to download the modules.

*Running main.py*

Please cd into `msci-nlp-w22/assignment4` and run `python3 main.py {path} {classifier}` where `path` is the path to the directory 
input split files and `classifier` is the activation type. For example: `python3 main.py /Users/rosakang/workspace/msci-nlp-w22/assignment1/data tanh`

*Running inference.py*

Please ensure that the output files `tokenizer.pkl` and classifier models i.e. `nn_sigmoid.model` has already been saved in the `msci-nlp-w22/assignment4/data` directory.
You can do this by running the main script.

Please cd into `msci-nlp-w22/assignment4` and run `python3 inference.py {path} {classifier}` where `path` is the 
path to the sample `.txt` file and `classifier` is the activation type. For example, `python3 inference.py /Users/rosakang/workspace/MSCI598/a4/data/sample.txt relu`

I used dropout of 0.001 and no stop words input for highest accuracy. The results are as follows:

| Sigmoid Accuracy | Tanh Accuracy | Relu Accuracy |
|:----------:|:-------------:|:------:|
| 0.731 |    0.733   |  0.7343 |
    