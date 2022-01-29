import csv
import pandas as pd
import argparse
import codecs

import torch
from empathy_classifier import EmpathyClassifier

'''
Example:
'''

parser = argparse.ArgumentParser("Test")
parser.add_argument("--input_path", type=str, help="path to input test data")
parser.add_argument("--output_path", type=str, help="output file path")

parser.add_argument("--ER_model_path", type=str, help="path to ER model")
parser.add_argument("--IP_model_path", type=str, help="path to IP model")
parser.add_argument("--EX_model_path", type=str, help="path to EX model")

args = parser.parse_args()

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


input_df = pd.read_csv(args.input_path, header=0)

ids = input_df.id.astype(str).tolist()
seeker_posts = input_df.seeker_post.astype(str).tolist()
response_posts = input_df.response_post.astype(str).tolist()

empathy_classifier = EmpathyClassifier(device,
						ER_model_path = args.ER_model_path, 
						IP_model_path = args.IP_model_path,
						EX_model_path = args.EX_model_path,)


output_file = codecs.open(args.output_path, 'w', 'utf-8')
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

csv_writer.writerow(['id','seeker_post','response_post','ER_label','IP_label','EX_label', 'ER_rationale', 'IP_rationale', 'EX_rationale'])

for i in range(len(seeker_posts)):
	(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])

	csv_writer.writerow([ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()])

output_file.close()