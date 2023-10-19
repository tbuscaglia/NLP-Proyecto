import csv
import pandas as pd
import numpy as np

data538 = []

with open('C:/Users/tomas/Downloads/spi_matches (2).csv', 'r') as csv_file:
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

with open('C:/Users/tomas/Downloads/E0.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    rowcount = 0
    for row in csv_reader:
      dataEPL21_22.append(row)



# Transform data into dataframes: 

columns = ['Date', 'HomeTeam', 'AwayTeam', 'prob1', 'prob2', 'probtie', 'importance1', 'importance2',  'score1', 'score2']

df_bets = pd.DataFrame(dataEPL21_22[1:], columns=dataEPL21_22[0])
df_spi = pd.DataFrame(data538_3, columns = columns)

# Standarize the date format

df_spi['Date'] = pd.to_datetime(df_spi['Date']).dt.strftime('%d/%m/%Y')


#Change the team names so that they match

name_to_find = ['Manchester United', 'Leicester City', 'Norwich City', 
                'Tottenham Hotspur', 'Leeds United', 'Manchester City', 
                'Brighton and Hove Albion', 'Wolverhampton', 'West Ham United']

new_name = ['Man United', 'Leicester', 'Norwich', 'Tottenham', 'Leeds', 
            'Man City', 'Brighton', 'Wolves', 'West Ham']

df_spi['HomeTeam'] = df_spi['HomeTeam'].replace(name_to_find, new_name)
df_spi['AwayTeam'] = df_spi['AwayTeam'].replace(name_to_find, new_name)

'''
unique_values_bets = df_bets['HomeTeam'].unique()

# Get unique values from the 'HomeTeam' column in df_spi
unique_values_spi = df_spi['HomeTeam'].unique()
'''

result = df_bets.merge(df_spi, on=['Date', 'HomeTeam'], how='inner')










