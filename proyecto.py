### Proyecto NLP ###

# Importar datos de reddit
import csv

file_path = 'C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/NLP-Proyecto/clean_data/mcfc_filtered.csv'

mcfc_clean = []

with open(file_path, 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_file:
      mcfc_clean.append(row)

abab = mcfc_clean[:1000]




