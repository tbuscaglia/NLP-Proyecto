import json
from datetime import datetime
from tqdm import tqdm
import csv
import re
import nltk 
import os


nltk.download('punkt')
from nltk.tokenize import word_tokenize



# Import raw data. Store cleaned and filtered data in filtered_data.

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


# Further cleaning of data. Eg: removing deleted comments. 

#filter_test = filtered_data #Comment this line for production and change below

filtered_data = [sublist for sublist in filtered_data if '[deleted]' not in sublist and '[removed]' not in sublist]

subset_filtered_data = filtered_data[:1000]

### Clean data here ###


# Once we have cleaned all of our data we write a new csv file to store it. 

with open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/NLP-Proyecto/clean_data/mcfc_filtered.csv', mode='w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file, delimiter='\t')  # Use a tab as the delimiter

    # Write the header row (optional)
    writer.writerow(["Unix Date", "Comment"])

    # Write each data row
    for row in filtered_data:
        writer.writerow([row[0], row[1]])

# Reading data from the CSV file while preserving paragraph structure
mcfc_clean = []

with open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/NLP-Proyecto/clean_data/mcfc_filtered.csv', mode='r', encoding="utf-8") as file:
    reader = csv.reader(file, delimiter='\t')  # Specify the same delimiter

    # Skip the header row (if present)
    next(reader, None)

    for row in reader:
        unix_time = int(row[0])
        text = row[1]
        mcfc_clean.append([unix_time, text])

subset_clean_mcfc = mcfc_clean[:1000] 

# Delete raw data to save storage

file_path = 'C:/Users/tomas/Downloads/MCFC_comments'

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"The file '{file_path}' has been deleted.")
else:
    print(f"The file '{file_path}' does not exist.")




'''    
#Remove any empty lists              
#mcfc_clean = [sublist for sublist in mcfc_clean if any(element.strip() for element in sublist)]
'''
 

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



