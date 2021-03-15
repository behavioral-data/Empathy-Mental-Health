import codecs
import os
import csv
import argparse
import random

parser = argparse.ArgumentParser("med_process_data")
parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path to output data")
args = parser.parse_args()

# input_file = codecs.open(args.input_path, 'r', 'utf-8')
input_files = os.listdir(args.input_path)
print(input_files)
print()
output_file = codecs.open(args.output_path+'test_med_data.csv', 'w', 'utf-8')
output_labels = codecs.open(args.output_path+'label_med_data.csv', 'w', 'utf-8')

csv_writer = csv.writer(output_file, delimiter = ',',quotechar='"')
csv_label_writer = csv.writer(output_labels, delimiter = ',',quotechar='"')

csv_writer.writerow(["id","seeker_post","response_post"]) #,"level","rationale_labels","rationale_labels_trimmed","response_post_masked"])
csv_label_writer.writerow(["id","seeker_post","response_post","level","rationale_labels","rationale_labels_trimmed","response_post_masked"])

for i_file in input_files:
    print("Reading from file", i_file)
    input_file = codecs.open(args.input_path+i_file, 'r', 'utf-8')
    seeker = False
    response = False
    rand_samples = [random.randint(0, 250) for i in range(10)]
    # print(rand_samples)
    for line in input_file:
        if 'id=' in line:
            # print('new data found')
            cur_id = line.strip('\ufeffid=')
            seeker_post = ''
            response_post = ''
            seeker = False
            response = False
        elif 'Patient:' == line.strip():
            seeker = True
        elif seeker == True:
            if line == '\n' or line =='\r':
                seeker = False
            elif 'Doctor:' == line.strip():
                response = True
                seeker = False
            else:
                seeker_post += line.strip()
        elif 'Doctor:' == line.strip():
            response = True
        elif response == True:
            response_post += line.strip()
            if line == '\n' or line =='\r':
                response = False
        
        if response_post != '' and seeker_post != '' and seeker == False and response == False:
            # print('writing data to')
            csv_writer.writerow([cur_id, seeker_post, response_post])
        
            if int(cur_id) in rand_samples:
                # print("writing label sample")
                csv_label_writer.writerow([cur_id, seeker_post, response_post, "", "", "", ""])
        
        # print(seeker_post)
        # print(seeker)
        # print(response_post)
        # print(response)
        # print(ascii(line))
        # input()
        
