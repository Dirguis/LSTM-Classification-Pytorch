---
Text Classification based on LSTM on R8 Dataset-[pytorch implementation]
---

### 1. Details of file fold:
- data/
- data/train_txt/*.txt
- data/train_txt.txt
- data/train_label.txt
- data/test_txt/*.txt
- data/test_txt.txt
- data/test_label.txt

### 2. File description:

| file | description|
|---|---|
|data/train_txt/|training text fold|
|data/test_txt/|testing text fold|
|data/train_txt.txt|file name list for training text |
|data/test_txt.txt|file name list for testing text |
|data/train_label.txt|label list for training text|
|data/test_label.txt| label list for testing text|

### 3. Running example:
Environment: python 3

Requirements:
```python
pytorch
```
running example:
```python
python main.py
```
output:
![](./img/LSTM_classifier_example.png)

### 4. Dataset:
We use the following dataset for our example:
R8 Dataset.[link](http://www.cs.umb.edu/~smimarog/textmining/datasets/).

### 5. Notes:
- This code is largely inspired from the original repo: [jiangqy](https://github.com/jiangqy/LSTM-Classification-Pytorch) with a few modifications:
  - Custom dataloader
  - Select the right output after the LSTM/GRU (masking)
  - Attention layer included, from [here](https://github.com/wabyking/TextClassificationBenchmark)
