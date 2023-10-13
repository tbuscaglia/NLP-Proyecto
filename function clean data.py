### Testing clean data function

import os
import json
import csv
from tqdm import tqdm
from datetime import datetime

def process_comments(input_filename, output_filename, zip_filename, name):
    filtered_data = []

    with open(input_filename, 'r', encoding="utf8") as file:
        for line in tqdm(file):
            data = json.loads(line)
            created_utc = data['created_utc']
            timestamp = datetime.utcfromtimestamp(int(created_utc))
            formatted_date = timestamp.strftime("%Y %m %d")
            if int(created_utc) > 1628650800 and int(created_utc) < 1653447599:  # Unix Time ~ 1/1/2021
                filtered_data.append([created_utc, data['body']])

    filtered_data = [sublist for sublist in filtered_data if '[deleted]' not in sublist and '[removed]' not in sublist]

    with open(output_filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')

        writer.writerow(["Unix Date", "Comment"])

        for row in filtered_data:
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
    2 - Run in the console !zstd --decompress path/to/file.zst
    3 - Edit input_file, output_file, zip_filename and name acordingly
    4 - Run line with function

    
!zstd --decompress 


'''

# MCFC #

input_file = 'C:/Users/tomas/Downloads/MCFC_comments'
zip_filename = "C:/Users/tomas/Downloads/MCFC_comments.zst"

name = 'mcfc'
output_file = f'C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/NLP-Proyecto/clean_data/{name}_filtered.csv'


mcfc_data = process_comments(input_file, output_file, zip_filename, name)
subset_mcfc_data = mcfc_data[:1000]

# West Ham #

input_file = 'C:/Users/tomas/Downloads/Hammers_comments'
zip_filename = "C:/Users/tomas/Downloads/Hammers_comments.zst"

name = 'WHU'
output_file = f'C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/NLP-Proyecto/clean_data/{name}_filtered.csv'

whu_data = process_comments(input_file, output_file, zip_filename, name)
subset_whu_data = whu_data[:1000]
