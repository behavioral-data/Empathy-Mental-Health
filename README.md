# Empathy in Text-based Mental Health Support
This repository contains codes and dataset access instructions for the [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441) on understanding empathy expressed in text-based mental health support.

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

We present a computational approach to understanding how empathy is expressed in online mental health platforms. We develop a novel unifying theoretically-grounded framework for characterizing the communication of empathy in text-based conversations. We collect and share a corpus of 10k (post, response) pairs annotated using this empathy framework with supporting evidence for annotations (rationales). We develop a multi-task RoBERTa-based bi-encoder model for identifying empathy in conversations and extracting rationales underlying its predictions. Experiments demonstrate that our approach can effectively
identify empathic conversations. We further apply this model to analyze 235k mental health interactions and show that users do not self-learn empathy over time, revealing opportunities for empathy training and feedback.

For a quick overview, please check out [bdata.uw.edu/empathy/](http://bdata.uw.edu/empathy/). For a detailed description of our work, please read our [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441).

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
response_post: A response/reply posted in response to the seeker_post
level: Empathy level of the response_post in the context of the seeker_post
rationale: Portions of the response_post that are supporting evidences or rationales for the identified empathy level. Multiple portions are delimited by '|'
```

This file (and other raw input files of similar format) can be converted into a format that is recognized by the model using with following command:
```
$ python3 src/process_data.py --input_path dataset/sample_input_ER.csv --output_path dataset/sample_input_model_ER.csv
```

#### 3. Training the model
For training our model on the sample input data, run the following command:
```
./train.sh
```
