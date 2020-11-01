from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import math
from collections import defaultdict


def flat_accuracy(preds, labels, axis_=2):
	pred_flat = np.argmax(preds, axis=axis_).flatten()
	labels_flat = labels.flatten()

	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_accuracy_rationale(preds, labels, classification_labels, lens, axis_=2):
	preds_li = np.argmax(preds, axis=axis_)

	all_acc = []

	for idx, elem in enumerate(preds_li):

		curr_pred = preds_li[idx]
		curr_pred = curr_pred[1:]
		curr_pred = curr_pred[:lens[idx]]

		curr_label = labels[idx]
		curr_label = curr_label[1:]
		curr_label = curr_label[:lens[idx]]

		curr_acc = np.sum(np.asarray(curr_label) == np.asarray(curr_pred)) / len(curr_label)

		all_acc.append(curr_acc)

	return np.mean(all_acc)


def compute_f1(preds, labels, axis_=2):
	pred_flat = np.argmax(preds, axis=axis_).flatten()
	labels_flat = labels.flatten()

	pos_f1 = f1_score(pred_flat, labels_flat, average = 'weighted')
	micro_f1 = f1_score(pred_flat, labels_flat, average = 'micro')
	macro_f1 = f1_score(pred_flat, labels_flat, average = 'macro')

	return pos_f1, micro_f1, macro_f1 #np.sum(pred_flat == labels_flat) / len(labels_flat)

def compute_f1_rationale(preds, labels, classification_labels, lens, axis_=2):
	preds_li = np.argmax(preds, axis=axis_)

	all_f1 = []

	all_labels = []
	all_f1 = []

	for idx, elem in enumerate(preds_li):

		curr_pred = preds_li[idx]
		curr_pred = curr_pred[1:]
		curr_pred = curr_pred[:lens[idx]]

		curr_label = labels[idx]
		curr_label = curr_label[1:]
		curr_label = curr_label[:lens[idx]]

		macro_f1 = f1_score(curr_pred, curr_label)

		all_f1.append(macro_f1)

	return np.mean(all_f1)

def _f1(_p, _r):
	if _p == 0 or _r == 0:
		return 0
	return 2 * _p * _r / (_p + _r)

def iou_f1(preds, labels, classification_labels, lens, axis_=2, threshold = 0.5):
	preds_li = np.argmax(preds, axis=axis_).tolist()

	all_pred_spans = []
	all_label_spans = []

	all_f1_vals = []

	for idx, elem in enumerate(preds_li):
		curr_pred = preds_li[idx]

		curr_pred = curr_pred[1:]
		curr_pred = curr_pred[:lens[idx]]

		pred_start_idx = -1
		pred_end_idx = -1

		pred_spans = []

		for inner_idx, inner_elem in enumerate(curr_pred):
			
			if inner_elem == 1:
				if pred_start_idx == -1:
					pred_start_idx = inner_idx
				
				else:
					continue 
			
			elif inner_elem == 0:
				if pred_start_idx == -1:
					continue
				else:
					pred_end_idx = inner_idx

					pred_spans.append((pred_start_idx, pred_end_idx))

					pred_start_idx = -1
					pred_end_idx = -1
		
		if pred_start_idx != -1:
			pred_end_idx = inner_idx

			pred_spans.append((pred_start_idx, pred_end_idx))


		# Labels

		curr_label = labels[idx]

		curr_label = curr_label[1:]
		curr_label = curr_label[:lens[idx]]
		
		label_start_idx = -1
		label_end_idx = -1

		label_spans = []

		for inner_idx, inner_elem in enumerate(curr_label):
			
			if inner_elem == 1:
				if label_start_idx == -1:
					label_start_idx = inner_idx
				
				else:
					continue 
			
			elif inner_elem == 0:
				if label_start_idx == -1:
					continue
				else:
					label_end_idx = inner_idx

					label_spans.append((label_start_idx, label_end_idx))

					label_start_idx = -1
					label_end_idx = -1
		
		if label_start_idx != -1:
			label_end_idx = inner_idx

			label_spans.append((label_start_idx, label_end_idx))
		
		ious = defaultdict(dict)
		for p in pred_spans:
			best_iou = 0.0
			for t in label_spans:
				num = len(set(range(p[0], p[1])) & set(range(t[0], t[1])))
				denom = len(set(range(p[0], p[1])) | set(range(t[0], t[1])))
				iou = 0 if denom == 0 else num / denom

				if iou > best_iou:
					best_iou = iou
			ious[idx][p] = best_iou

		threshold_tps = dict()

		for k, vs in ious.items():
			threshold_tps[k] = sum(int(x >= threshold) for x in vs.values())

		micro_r = sum(threshold_tps.values()) / len(label_spans) if len(label_spans) > 0 else 0
		micro_p = sum(threshold_tps.values()) / len(pred_spans) if len(pred_spans) > 0 else 0
		micro_f1 = _f1(micro_r, micro_p)
		all_f1_vals.append(micro_f1)

	return np.mean(all_f1_vals)
