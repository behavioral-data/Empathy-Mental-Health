import codecs
import os
import csv

hand_label_file = codecs.open('label_med_data.csv', 'r', 'ISO-8859-1')
model_label_file = codecs.open('test_med_data_output.csv', 'r', 'ISO-8859-1')

csv_hand_label_reader = csv.reader(hand_label_file, delimiter = ',', quotechar='"')
csv_model_label_reader = csv.reader(model_label_file, delimiter = ',', quotechar='"')

hand_label_search = []

next(csv_hand_label_reader, None) # skip the header

for row in csv_hand_label_reader:
    cur_id = row[0].strip()
    seeker = row[1].strip()
    response = row[2].strip()
    hand_label_search.append((cur_id, seeker, response))

next(csv_model_label_reader, None) # skip the header
ER = [0, 0, 0]
IR = [0, 0, 0]
EX = [0, 0, 0]
total = 0
for row in csv_model_label_reader:
    # print(row)
    cur_id = row[0].strip()
    if len(row) > 1:
        seeker = row[1].strip()
    else:
        seeker = ''
    if len(row) > 2:
        response = row[2].strip()
    else:
        response = ''
    
    if len(row) > 5:
        ER[int(row[3])] += 1 
        IR[int(row[4])] += 1 
        EX[int(row[5])] += 1 
        total += 1
    
    for hand_labeled in hand_label_search:
        if cur_id == hand_labeled[0] and seeker == hand_labeled[1] and response == hand_labeled[2]:
            print(row)

print(ER)
print(IR)
print(EX)
print(total)