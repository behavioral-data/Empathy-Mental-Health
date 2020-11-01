import codecs
import os
import csv
import re
import numpy as np
import argparse

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

parser = argparse.ArgumentParser("process_data")
parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path to output data")
args = parser.parse_args()

input_file = codecs.open(args.input_path, 'r', 'utf-8')
output_file = codecs.open(args.output_path, 'w', 'utf-8')

csv_reader = csv.reader(input_file, delimiter = ',', quotechar='"')
csv_writer = csv.writer(output_file, delimiter = ',',quotechar='"') 

next(csv_reader, None) # skip the header

csv_writer.writerow(["id","seeker_post","response_post","labels","rationale_labels","rationale_labels_trimmed","response_post_masked"])

for row in csv_reader:
	# "id","seeker_post","response_post","label","rationale"

	seeker_post = row[1].strip()
	response = row[2].strip()

	response_masked = response

	response_tokenized = tokenizer.decode(tokenizer.encode_plus(response, add_special_tokens = True, max_length = 64, pad_to_max_length = True)['input_ids'], clean_up_tokenization_spaces=False)

	response_tokenized_non_padded = tokenizer.decode(tokenizer.encode_plus(response, add_special_tokens = True, max_length = 64, pad_to_max_length = False)['input_ids'], clean_up_tokenization_spaces=False)

	response_words = tokenizer.tokenize(response_tokenized)
	response_non_padded_words = tokenizer.tokenize(response_tokenized_non_padded)

	if len(response_words) != 64:
		continue

	response_words_position = np.zeros((len(response),), dtype=np.int32)

	rationales = row[4].strip().split('|')

	rationale_labels = np.zeros((len(response_words),), dtype=np.int32)


	curr_position = 0

	for idx in range(len(response_words)):
		curr_word = response_words[idx]
		if curr_word.startswith('Ġ'):
			curr_word = curr_word[1:]
		response_words_position[curr_position: curr_position+len(curr_word)+1] = idx
		curr_position += len(curr_word)+1

	

	if len(rationales) == 0 or row[3].strip() == '':
		rationale_labels[1:len(response_non_padded_words)] = 1
		response_masked = ''

	for r in rationales:
		if r == '':
			continue
		try:
			r_tokenizer = tokenizer.decode(tokenizer.encode(r, add_special_tokens = False))
			match = re.search(r_tokenizer , response_tokenized)

			curr_match = response_words_position[match.start(0):match.start(0)+len(r_tokenizer)]
			curr_match = list(set(curr_match))
			curr_match.sort()

			response_masked = response_masked.replace(r, ' ')
			response_masked = re.sub(r' +', ' ', response_masked)

			rationale_labels[curr_match] = 1
		except:
			continue
	
	
	rationale_labels_str = ','.join(str(x) for x in rationale_labels)

	rationale_labels_str_trimmed = ','.join(str(x) for x in rationale_labels[1:len(response_non_padded_words)])


	csv_writer.writerow([row[0], seeker_post, response, row[3], rationale_labels_str, len(rationale_labels_str_trimmed), response_masked])

input_file.close()
output_file.close()