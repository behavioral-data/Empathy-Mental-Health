# Empathy in Text-based Mental Health Support
This repository contains codes and dataset access instructions for the [EMNLP 2020 paper](https://arxiv.org/pdf/2009.08441) on understanding empathy expressed in text-based mental health support.

If this code helps you in your research, please cite the following publication:
```bash
@inproceedings{sharma2020empathy,
    title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
    author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2020},
    booktitle={EMNLP}
}
```

### Introduction


### Quickstart

#### 1. Prerequisites

Our framework can be compiled on Python 3.7+ environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```


#### 2. Prepare dataset
A sample raw input data file is available in [dataset/sample_input_ER.csv](dataset/sample_input_ER.csv). This csv file contains five columns:
```
id: A unique identifier
seeker_post: A support seeking post from an online user
response_post: A response/reply posted in response to the seeker_post.
level: Empathy level of the response_post in the context of the seeker_post.
rationale: Portions of the response_post that are supporting evidences or rationales for the identified empathy level. Multiple portions are delimited by '|'
```

This file (and other raw input files of similar format) can be converted into a format that is recognized by the model file using with following command:
```
$ python3 src/process_data.py --input_path dataset/sample_input_ER.csv --output_path dataset/sample_input_model_ER.csv
```

#### 3. Training the model
For training our model on the sample input data, run the following command:
```
./train.sh
```
