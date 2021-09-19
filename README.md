# SENT

Codes for ACL2021 paper "SENT: Sentence-level Distant Relation Extraction via Negative Training"

## Environment
Python 3.6.5, Pytorch 1.6.0, Hugging Face Transformers 3.4.0

## Data
1. Original data can be downloaded from https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-ARNOR. 
We offer the pre-processed data we used in the data/ dir. (The training data file is too large and can be downloaded from https://drive.google.com/file/d/1DL-UhXCSqf0qTlDM1_k3YTIggZSG2Khr/view?usp=sharing)

2. Download the pre-trained word vectors glove.6B.50d from https://github.com/stanfordnlp/GloVe , and put it in the data/ dir.

## Usage

Train the model with:
```
sh run.sh
```