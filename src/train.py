import codecs
import numpy as np
import pandas as pd
import re
import math
import numpy as np
import random
import time
import datetime
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import RobertaTokenizer
from transformers import AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup

from models.models import BiEncoderAttentionWithRationaleClassification
from evaluation_utils import *


parser = argparse.ArgumentParser("BiEncoder")

parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--lambda_EI", default=0.5, type=float, help="lambda_identification")
parser.add_argument("--lambda_RE", default=0.5, type=float, help="lambda_rationale")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
parser.add_argument("--data_percent", default=1, type=float, help="training data percent")
parser.add_argument("--max_len", default=64, type=int, help="maximum sequence length")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=4, type=int, help="number of epochs")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
parser.add_argument("--train_path", type=str, help="path to input training data")
parser.add_argument("--dev_path", type=str, help="path to input validation data")
parser.add_argument("--test_path", type=str, help="path to input test data")
parser.add_argument("--do_validation", action="store_true")
parser.add_argument("--do_test", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--no_domain_pretraining", action="store_true")
parser.add_argument("--no_attention", action="store_true")
parser.add_argument("--save_model_path", type=str, help="path to save model")

args = parser.parse_args()

print("=====================Args====================")

print('lr = ', args.lr)
print('lambda_EI = ', args.lambda_EI)
print('lambda_RE = ', args.lambda_RE)
print('dropout = ', args.dropout)
print('max_len = ', args.max_len)
print('batch_size = ', args.batch_size)
print('epochs = ', args.epochs)
print('seed_val = ', args.seed_val)
print('train_path = ', args.train_path)
print('data_percent = ', args.data_percent)
print('no_domain_pretraining = ', args.no_domain_pretraining)
print('no_attention = ', args.no_attention)
print('do_validation = ', args.do_validation)
print('do_test = ', args.do_test)
print('data_percent = ', args.data_percent)

print("=============================================")


'''
Use GPU if available
'''
if torch.cuda.is_available():
        device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


'''
Load input dataset
'''
if args.train_path:
        df = pd.read_csv(args.train_path, delimiter=',')
        df['rationale_labels'] = df['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
        df = df[:int(len(df)*args.data_percent)]
else:
	print('No input training data specified.')
	print('Exiting...')
	exit(-1)

if args.do_test:
	if args.test_path:
		df_test = pd.read_csv(args.test_path, delimiter=',')
		df_test['rationale_labels'] = df_test['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
	else:
		print('No input test data specified.')
		print('Exiting...')
		exit(-1)

if args.do_validation:
	if args.dev_path:
		df_val = pd.read_csv(args.dev_path, delimiter=',')
		df_val['rationale_labels'] = df_val['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
	else:
		print('No input validation data specified.')
		print('Exiting...')
		exit(-1)


'''
Tokenize input
'''
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

tokenizer_RP = tokenizer.batch_encode_plus(df.response_post, add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_masks=True)
input_ids_RP = torch.tensor(tokenizer_RP['input_ids'])
attention_masks_RP = torch.tensor(tokenizer_RP['attention_mask'])

tokenizer_SP = tokenizer.batch_encode_plus(df.seeker_post, add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_masks=True)
input_ids_SP = torch.tensor(tokenizer_SP['input_ids'])
attention_masks_SP = torch.tensor(tokenizer_SP['attention_mask'])


labels = df.level.values.astype(int)
labels = torch.tensor(labels)
rationales = df.rationale_labels.values.tolist()
rationales = torch.stack(rationales, dim=0)


if args.do_validation:
	val_tokenizer_RP = tokenizer.batch_encode_plus(df_val.response_post, add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_masks=True)
	val_input_ids_RP = torch.tensor(val_tokenizer_RP['input_ids'])
	val_attention_masks_RP = torch.tensor(val_tokenizer_RP['attention_mask'])

	val_tokenizer_SP = tokenizer.batch_encode_plus(df_val.seeker_post, add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_masks=True)
	val_input_ids_SP = torch.tensor(val_tokenizer_SP['input_ids'])
	val_attention_masks_SP = torch.tensor(val_tokenizer_SP['attention_mask'])

	val_labels = torch.tensor(df_val.level.values.astype(int))
	val_rationales = df_val.rationale_labels.values.tolist()
	val_rationales = torch.stack(val_rationales, dim=0)
	val_rationales_trimmed = torch.tensor(df_val.rationale_labels_trimmed.values.astype(int))

if args.do_test:
	test_tokenizer_RP = tokenizer.batch_encode_plus(df_test.response_post, add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_masks=True)
	test_input_ids_RP = torch.tensor(test_tokenizer_RP['input_ids'])
	test_attention_masks_RP = torch.tensor(test_tokenizer_RP['attention_mask'])

	test_tokenizer_SP = tokenizer.batch_encode_plus(df_test.seeker_post, add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_masks=True)
	test_input_ids_SP = torch.tensor(test_tokenizer_SP['input_ids'])
	test_attention_masks_SP = torch.tensor(test_tokenizer_SP['attention_mask'])

	test_labels = torch.tensor(df_test.level.values.astype(int))
	test_rationales = df_test.rationale_labels.values.tolist()
	test_rationales = torch.stack(test_rationales, dim=0)
	test_rationales_trimmed = torch.tensor(df_test.rationale_labels_trimmed.values.astype(int))


'''
Load model
'''
use_domain_pretraining = not args.no_domain_pretraining
use_attention = not args.no_attention
model = BiEncoderAttentionWithRationaleClassification(hidden_dropout_prob=args.dropout, use_domain_pretraining=use_domain_pretraining, use_attention=use_attention)
model = model.to(device)

'''
Do not finetune seeker encoder
'''
params = list(model.named_parameters())
for p in model.seeker_encoder.parameters():
	p.requires_grad = False

# print('Number of parameters', sum(p.numel() for p in model.parameters()))

optimizer = AdamW(model.parameters(),
				  lr = args.lr,
				  eps = 1e-8
				)


'''
Training schedule
'''
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
train_dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP, labels, rationales)
train_size = int(len(train_dataset))
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = args.batch_size, worker_init_fn=seed_worker)

if args.do_validation:
	val_dataset = TensorDataset(val_input_ids_SP, val_attention_masks_SP, val_input_ids_RP, val_attention_masks_RP, val_labels, val_rationales, val_rationales_trimmed)
	validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = args.batch_size)

if args.do_test:
	test_dataset = TensorDataset(test_input_ids_SP, test_attention_masks_SP, test_input_ids_RP, test_attention_masks_RP, test_labels, test_rationales, test_rationales_trimmed)
	test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = args.batch_size)


total_steps = len(train_dataloader) * args.epochs
num_batch = len(train_dataloader)

print('total_steps =', total_steps)
print('num_batch =', num_batch)
print("=============================================")

scheduler = get_linear_schedule_with_warmup(optimizer, 
											num_warmup_steps = 0,
											num_training_steps = total_steps)

random.seed(args.seed_val)
np.random.seed(args.seed_val)
torch.manual_seed(args.seed_val)
torch.cuda.manual_seed_all(args.seed_val)
torch.backends.cudnn.deterministic = True


'''
Do Training
'''
start_time = time.time()
for epoch_i in range(0, args.epochs):
	total_train_loss = 0
	total_train_empathy_loss = 0
	total_train_rationale_loss = 0

	pbar = tqdm(total=num_batch, desc=f"training")

	model.train()

	for step, batch in enumerate(train_dataloader):

		b_input_ids_SP = batch[0].to(device)
		b_input_mask_SP = batch[1].to(device)
		b_input_ids_RP = batch[2].to(device)
		b_input_mask_RP = batch[3].to(device)
		b_labels = batch[4].to(device)
		b_rationales = batch[5].to(device)
		
		model.zero_grad()        

		loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(input_ids_SP = b_input_ids_SP,
																input_ids_RP = b_input_ids_RP, 
																attention_mask_SP=b_input_mask_SP,
																attention_mask_RP=b_input_mask_RP, 
																empathy_labels=b_labels,
																rationale_labels=b_rationales,
																lambda_EI=args.lambda_EI,
																lambda_RE=args.lambda_RE)


		total_train_loss += loss.item()
		total_train_empathy_loss += loss_empathy.item()
		total_train_rationale_loss += loss_rationale.item()

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		optimizer.step()
		scheduler.step()

		pbar.set_postfix_str(
			f"total loss: {float(total_train_loss/(step+1)):.4f} epoch: {epoch_i}")
		pbar.update(1)

	avg_train_loss = total_train_loss / len(train_dataloader)
	avg_train_empathy_loss = total_train_empathy_loss / len(train_dataloader)
	avg_train_rationale_loss = total_train_rationale_loss / len(train_dataloader)

	pbar.close()

	if args.do_validation:
		'''
		Validation
		'''
		print('\n\nRunning validation...\n')
		model.eval()

		total_eval_accuracy_empathy = 0
		total_eval_accuracy_rationale = 0

		total_pos_f1_empathy = 0.0
		total_micro_f1_empathy = 0.0
		total_macro_f1_empathy = 0.0

		total_pos_f1_rationale = 0.0
		total_micro_f1_rationale = 0.0
		total_macro_f1_rationale = 0.0

		total_iou_rationale = 0.0

		total_eval_loss = 0

		for batch in validation_dataloader:
			
			b_input_ids_SP = batch[0].to(device)
			b_input_mask_SP = batch[1].to(device)
			b_input_ids_RP = batch[2].to(device)
			b_input_mask_RP = batch[3].to(device)
			b_labels = batch[4].to(device)
			b_rationales = batch[5].to(device)	
			b_rationales_trimmed = batch[6].to(device)	

			with torch.no_grad():
				loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(input_ids_SP = b_input_ids_SP,
																	input_ids_RP = b_input_ids_RP, 
																	attention_mask_SP=b_input_mask_SP,
																	attention_mask_RP=b_input_mask_RP, 
																	empathy_labels=b_labels,
																	rationale_labels=b_rationales,
																	lambda_EI=args.lambda_EI,
																	lambda_RE=args.lambda_RE)

			total_eval_loss += loss.item()

			logits_empathy = logits_empathy.detach().cpu().numpy()
			logits_rationale = logits_rationale.detach().cpu().numpy()
			
			label_empathy_ids = b_labels.to('cpu').numpy()
			label_rationale_ids = b_rationales.to('cpu').numpy()
			rationale_lens = b_rationales_trimmed.to('cpu').numpy()

			total_eval_accuracy_empathy += flat_accuracy(logits_empathy, label_empathy_ids, axis_=1)
			total_eval_accuracy_rationale += flat_accuracy_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)
			
			pos_f1_empathy, micro_f1_empathy, macro_f1_empathy = compute_f1(logits_empathy, label_empathy_ids, axis_=1)
			macro_f1_rationale = compute_f1_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

			iou_f1_rationale = iou_f1(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

			total_pos_f1_empathy += pos_f1_empathy
			total_micro_f1_empathy += micro_f1_empathy
			total_macro_f1_empathy += macro_f1_empathy

			total_macro_f1_rationale += macro_f1_rationale
			total_iou_rationale += iou_f1_rationale

		avg_val_accuracy_empathy = total_eval_accuracy_empathy / len(validation_dataloader)
		avg_val_accuracy_rationale = total_eval_accuracy_rationale / len(validation_dataloader)

		avg_val_pos_f1_empathy = total_pos_f1_empathy / len(validation_dataloader)
		avg_val_micro_f1_empathy = total_micro_f1_empathy / len(validation_dataloader)
		avg_val_macro_f1_empathy = total_macro_f1_empathy / len(validation_dataloader)

		avg_val_macro_f1_rationale = total_macro_f1_rationale / len(validation_dataloader)
		avg_val_iou_rationale = total_iou_rationale / len(validation_dataloader)

		print("  Accuracy-Empathy: {0:.4f}".format(avg_val_accuracy_empathy))
		print("  macro_f1_empathy: {0:.4f}".format(avg_val_macro_f1_empathy))
		print("  Accuracy-Rationale: {0:.4f}".format(avg_val_accuracy_rationale))

		print("  IOU-F1-Rationale: {0:.4f}".format(avg_val_iou_rationale))
		print("  macro_f1_rationale: {0:.4f}".format(avg_val_macro_f1_rationale))

		avg_val_loss = total_eval_loss / len(validation_dataloader)

		print('\n')




'''
Test
'''
if args.do_test:
	print("\n\nRunning test...\n")

	model.eval()

	total_eval_accuracy_empathy = 0
	total_eval_accuracy_rationale = 0

	total_pos_f1_empathy = 0.0
	total_micro_f1_empathy = 0.0
	total_macro_f1_empathy = 0.0

	total_pos_f1_rationale = 0.0
	total_micro_f1_rationale = 0.0
	total_macro_f1_rationale = 0.0
	total_iou_rationale = 0.0

	total_eval_loss = 0

	for batch in test_dataloader:
		
		b_input_ids_SP = batch[0].to(device)
		b_input_mask_SP = batch[1].to(device)
		b_input_ids_RP = batch[2].to(device)
		b_input_mask_RP = batch[3].to(device)
		b_labels = batch[4].to(device)
		b_rationales = batch[5].to(device)
		b_rationales_trimmed = batch[6].to(device)

		with torch.no_grad():
			loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(input_ids_SP = b_input_ids_SP,
																input_ids_RP = b_input_ids_RP, 
																attention_mask_SP=b_input_mask_SP,
																attention_mask_RP=b_input_mask_RP, 
																empathy_labels=b_labels,
																rationale_labels=b_rationales,
																lambda_EI=args.lambda_EI,
																lambda_RE=args.lambda_RE)

		total_eval_loss += loss.item()

		logits_empathy = logits_empathy.detach().cpu().numpy()
		logits_rationale = logits_rationale.detach().cpu().numpy()
		
		label_empathy_ids = b_labels.to('cpu').numpy()
		label_rationale_ids = b_rationales.to('cpu').numpy()
		rationale_lens = b_rationales_trimmed.to('cpu').numpy()

		total_eval_accuracy_empathy += flat_accuracy(logits_empathy, label_empathy_ids, axis_=1)
		total_eval_accuracy_rationale += flat_accuracy_rationale(logits_rationale, label_rationale_ids,  label_empathy_ids, rationale_lens, axis_=2)
		
		pos_f1_empathy, micro_f1_empathy, macro_f1_empathy = compute_f1(logits_empathy, label_empathy_ids, axis_=1)
		macro_f1_rationale = compute_f1_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

		iou_f1_rationale = iou_f1(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

		total_pos_f1_empathy += pos_f1_empathy
		total_micro_f1_empathy += micro_f1_empathy
		total_macro_f1_empathy += macro_f1_empathy

		total_macro_f1_rationale += macro_f1_rationale
		total_iou_rationale += iou_f1_rationale

	avg_test_accuracy_empathy = total_eval_accuracy_empathy / len(test_dataloader)
	avg_test_accuracy_rationale = total_eval_accuracy_rationale / len(test_dataloader)

	avg_test_pos_f1_empathy = total_pos_f1_empathy / len(test_dataloader)
	avg_test_micro_f1_empathy = total_micro_f1_empathy / len(test_dataloader)
	avg_test_macro_f1_empathy = total_macro_f1_empathy / len(test_dataloader)

	avg_test_macro_f1_rationale = total_macro_f1_rationale / len(test_dataloader)
	avg_test_iou_rationale = total_iou_rationale / len(test_dataloader)

	print("  Accuracy-Empathy: {0:.4f}".format(avg_test_accuracy_empathy))
	print("  macro_f1_empathy: {0:.4f}".format(avg_test_macro_f1_empathy))
	print("  Accuracy-Rationale: {0:.4f}".format(avg_test_accuracy_rationale))

	print("  IOU-F1-Rationale: {0:.4f}".format(avg_test_iou_rationale))
	print("  macro_f1_rationale: {0:.4f}".format(avg_test_macro_f1_rationale))

	avg_test_loss = total_eval_loss / len(test_dataloader)

print('time taken: ', time.time() - start_time)
if args.save_model:
	torch.save(model.state_dict(), args.save_model_path)
