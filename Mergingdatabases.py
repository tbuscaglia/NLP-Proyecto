import csv
data538 = []

with open('/Users/julianandelsman/Desktop/NLP/Final project/538data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data538.append(row)
data538_1 = [row for row in data538 if row[2] == "2411"]

from datetime import datetime

fecha_inicio = datetime.strptime("2021-08-13", "%Y-%m-%d")
fecha_fin = datetime.strptime("2022-05-22", "%Y-%m-%d")
data538_2 = [row for row in data538_1 if fecha_inicio <= datetime.strptime(row[1], "%Y-%m-%d") <= fecha_fin]

selected_columns = [1, 4, 5, 8, 9, 10, 13, 14, 15, 16]

data538_3 = [[row[i] for i in selected_columns] for row in data538_2]
'''
with open('clean538.csv', 'w', newline='') as file_csv:
    csv_writer = csv.writer(file_csv)
    for i in range(len(data538_3)):
        date = data538_3[i][0]
        team1 = data538_3[i][1]
        team2 = data538_3[i][2]    
        csv_writer.writerow([date, team1, team2])
'''
dataEPL21_22 = []

with open('/Users/julianandelsman/Desktop/NLP/Final project/EPL 21-22.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    rowcount = 0
    for row in csv_reader:
        if rowcount>0:
            dataEPL21_22.append(row)
        rowcount += 1
