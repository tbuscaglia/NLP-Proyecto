### Generating cleaning data function

import os
import json
import csv
import re
from tqdm import tqdm
from datetime import datetime
import csv
import numpy as np
import html 

#First, we define a url cleaner function to be used later:
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def url_cleaner(text):
    texto_no_urls = re.sub(url_pattern, '', text)
    return texto_no_urls
#Then, we define a another function, that will be used later to eliminate "long" words:
def word_cutter(L):
    words = []
    line_count = 0
    for row in L:
        if line_count > 0:
            words.extend(row[1].lower().split(' '))
        line_count += 1
    word_length = []
    for w in words:
        word_length.append(len(w))
    mean_len = np.mean(word_length)
    std_len = np.std(word_length)
    sentences = {}
    line_count = 0
    for i in range(0,len(L)):
        row = L[i]
        words = row[1].lower().split(' ')
        valid_words = []
        for w in words:
            if (len(w)-mean_len)/std_len <3:
                valid_words.append(w)
        if len(valid_words)>0:
            sentences[i]=(' '.join(valid_words))
    
    return (sentences)
#Note: this function takes as an argument a list, and takes out the long words from the [1] position of each list.
# Then, we generate the reddit comments cleaning function
def process_comments(input_filename, output_filename, zip_filename, name):
    filtered_data = []
    
    with open(input_filename, 'r', encoding="utf8") as file:
        for line in (file):
            data = json.loads(line)
            created_utc = data['created_utc']
            timestamp = datetime.utcfromtimestamp(int(created_utc))
            formatted_date = timestamp.strftime("%Y %m %d")
            if int(created_utc) > 1628650800 and int(created_utc) < 1653447599: #Unix time = we filter data from 2 days before the beggining, to 2 days after the end of the EPL 21/22 season.
                filtered_data.append([created_utc, data['body']])

    filtered_data = [sublist for sublist in filtered_data if '[deleted]' not in sublist and '[removed]' not in sublist]
    
    for line in filtered_data:
        line[1] = url_cleaner(line[1])
    sentences = word_cutter(filtered_data)
    filtered_data_2 = []
    for i in range (len(filtered_data)):
        if i in sentences:
            ROW = [filtered_data[i][0],sentences[i]]
            filtered_data_2.append(ROW)
            
    for row in filtered_data_2:
        row[1] = html.unescape(row[1])
        row[1] = re.sub(' +', ' ', row[1])
    
    filtered_data_3 = []
    sentence_clean = {}
    for i in range (0, len(filtered_data_2)):
        text = ""
        for c in filtered_data_2[i][1]:
            if c.isalnum() or c == " ":
                text += c
        if len(text)>0:
            sentence_clean[i] = text
    for i in range (len(filtered_data_2)):
        if i in sentence_clean:    
            ROW = [filtered_data_2[i][0],sentence_clean[i]]
            filtered_data_3.append(ROW)
                
    with open(output_filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')

        writer.writerow(["Unix Date", "Comment"])

        for row in filtered_data_3:
            writer.writerow([row[0], row[1]])

    cleaned_data = []

    with open(output_filename, mode='r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader, None)

        for row in reader:
            unix_time = int(row[0])
            text = row[1]
            cleaned_data.append([unix_time, text])

    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        print(f"The file '{zip_filename}' has been deleted.")
    else:
        print(f"The file '{zip_filename}' does not exist.")
    
    if os.path.exists(input_filename):
        os.remove(input_filename)
        print(f"The file '{input_filename}' has been deleted.")
    else:
        print(f"The file '{input_filename}' does not exist.")
        
    
    return cleaned_data

### Aplicar funcion ###

'''
Steps for running code:
    1 - Download reddit comments as .zst file
    2 - Run in the console zstd --decompress path/to/file.zst
    3 - Edit folder name with the path of the folder that contains the comments files (both compressed and decompressed)
    4 - Edit file name with the names of the decompressed files
    5 - Run the loop containing the process_comments function
'''
folder = "/Users/julianandelsman/Downloads"
files  = ["NUFC_comments"]
for file in files:
    input = folder + "/" + file
    zip = input + ".zst"
    output = f'{folder}/{file}filtered.csv'
    process_comments(input, output, zip, file)
