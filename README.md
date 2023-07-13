# Empathy in Text-based Mental Health Support
This repository contains codes and dataset access instructions for the [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441) on understanding empathy expressed in text-based mental health support.

If this code or dataset helps you in your research, please cite the following publication:
```bash
@inproceedings{sharma2020empathy,
    title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
    author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2020},
    booktitle={EMNLP}
}
```

## Introduction

We present a computational approach to understanding how empathy is expressed in online mental health platforms. We develop a novel unifying theoretically-grounded framework for characterizing the communication of empathy in text-based conversations. We collect and share a corpus of 10k (post, response) pairs annotated using this empathy framework with supporting evidence for annotations (rationales). We develop a multi-task RoBERTa-based bi-encoder model for identifying empathy in conversations and extracting rationales underlying its predictions. Experiments demonstrate that our approach can effectively
identify empathic conversations. We further apply this model to analyze 235k mental health interactions and show that users do not self-learn empathy over time, revealing opportunities for empathy training and feedback.

For a quick overview, check out [bdata.uw.edu/empathy](http://bdata.uw.edu/empathy/). For a detailed description of our work, please read our [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441).

## Quickstart

### 1. Prerequisites

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```


### 2. Prepare dataset
A sample raw input data file is available in [dataset/sample_input_ER.csv](dataset/sample_input_ER.csv). This file (and other raw input files in the [dataset](dataset) folder) can be converted into a format that is recognized by the model using with following command:
```
$ python3 src/process_data.py --input_path dataset/sample_input_ER.csv --output_path dataset/sample_input_model_ER.csv
```

### 3. Training the model
For training our model on the sample input data, run the following command:
```
$ python3 src/train.py \
	--train_path=dataset/sample_input_model_ER.csv \
	--lr=2e-5 \
	--batch_size=32 \
	--lambda_EI=1.0 \
	--lambda_RE=0.5 \
	--save_model \
	--save_model_path=output/sample_ER.pth
```

**Note:** You may need to create an `output` folder in the main directory before running this command.

For training the models on the full Reddit dataset, these are the three commands you can run for Emotional Reactions, Interpretations, and Explorations respectively:

**1. Emotional Reactions**
```
python3 src/train.py \
--train_path=dataset/emotional-reactions-reddit.csv \
--lr=2e-5 \
--batch_size=32 \
--lambda_EI=1.0 \
--lambda_RE=0.5 \
--save_model \
--save_model_path=output/reddit_ER.pth
```

 **2. Interpretations**
```
python3 src/train.py \
--train_path=dataset/interpretations-reddit.csv \
--lr=2e-5 \
--batch_size=32 \
--lambda_EI=1.0 \
--lambda_RE=0.5 \
--save_model \
--save_model_path=output/reddit_IP.pth
```

**3. Explorations**
```
python3 src/train.py \
--train_path=dataset/explorations-reddit.csv \
--lr=2e-5 \
--batch_size=32 \
--lambda_EI=1.0 \
--lambda_RE=0.5 \
--save_model \
--save_model_path=output/reddit_EX.pth
```


### 4. Testing the model
For testing our model on the sample test input, run the following command:
```
$ python3 src/test.py \
	--input_path dataset/sample_test_input.csv \
	--output_path dataset/sample_test_output.csv \
	--ER_model_path output/sample_ER.pth \
	--IP_model_path output/sample_IP.pth \
	--EX_model_path output/sample_EX.pth
```

## Training Arguments

The training script accepts the following arguments: 

Argument | Type | Default value | Description
---------|------|---------------|------------
lr | `float` | `2e-5` | learning rate
lambda_EI | `float` | `0.5` | weight of empathy identification loss 
lambda_RE |  `float` | `0.5` | weight of rationale extraction loss
dropout |  `float` | `0.1` | dropout
max_len | `int` | `64` | maximum sequence length
batch_size | `int` | `32` | batch size
epochs | `int` | `4` | number of epochs
seed_val | `int` | `12` | seed value
train_path | `str` | `""` | path to input training data
dev_path | `str` | `""` | path to input validation data
test_path | `str` | `""` | path to input test data
do_validation | `boolean` | `False` | If set True, compute results on the validation data
do_test | `boolean` | `False` | If set True, compute results on the test data
save_model | `boolean` | `False` | If set True, save the trained model  
save_model_path | `str` | `""` | path to save model 


## Dataset Access Instructions

The Reddit portion of our collected dataset is available inside the [dataset](dataset) folder. The csv files with annotations on the three empathy communication mechanisms are `emotional-reactions-reddit.csv`, `interpretations-reddit.csv`, and `explorations-reddit.csv`. Each csv file contains six columns:
```
sp_id: Seeker post identifier
rp_id: Response post identifier
seeker_post: A support seeking post from an online user
response_post: A response/reply posted in response to the seeker_post
level: Empathy level of the response_post in the context of the seeker_post
rationales: Portions of the response_post that are supporting evidences or rationales for the identified empathy level. Multiple portions are delimited by '|'
```

For accessing the TalkLife portion of our dataset for non-commercial use, please contact the TalkLife team [here](mailto:research@talklife.co). 

