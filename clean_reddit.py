import json
from datetime import datetime
from tqdm import tqdm
import csv
import re
import nltk 


nltk.download('punkt')
from nltk.tokenize import word_tokenize



# Open the JSON file for reading
filtered_data = []

with open('C:/Users/tomas/Downloads/MCFC_comments', 'r', encoding="utf8") as file:
    for line in tqdm(file):
        # Process each line here
        data = json.loads(line) 
        created_utc = data['created_utc']
        timestamp = datetime.utcfromtimestamp(int(created_utc))
        # Format the datetime object as "YYYY MM DD"
        formatted_date = timestamp.strftime("%Y %m %d")
        if int(created_utc) >1628650800 and int(created_utc) < 1653447599 :        # Unix Time ~ 1/1/2021
            filtered_data.append([created_utc,data['body']])


#filter_test = filtered_data

filtered_data = [sublist for sublist in filtered_data if '[deleted]' not in sublist and '[removed]' not in sublist]

with open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/NLP-Proyecto/clean_data/mcfc_filtered.csv', mode='w', encoding="utf-8") as _file:
    _writer = csv.writer(_file, delimiter=',')
    for data in filtered_data:
        _writer.writerow(data)


'''
count = 0
character_count = 0

for line in filter_test:
    words = line[1].split(' ')
    for w in words:
        character_count += len(w)
        count += 1

        
avarage_len = character_count/count

numerador = 0  
for line in filter_test:
    words = line[1].split(' ')
    for w in words:
        numerador += (len(w) - avarage_len)**2
    
varianza = numerador / count
std_dev = varianza ** 0.5


aaa = filter_test[:1000]

to_remove_words = []

for line in tqdm(aaa):
    
    words = word_tokenize(line[1])

    for w in words:
        if len(w) > int(avarage_len + 2*std_dev):
            to_remove_words.append(w)
'''            



