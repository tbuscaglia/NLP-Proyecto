import csv
import pandas as pd
#import numpy as np

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

columns = ['Date', 'HomeTeam', 'AwayTeam', 'prob_homewin', 'prob_awaywin', 'prob_tie', 'importance_home', 'importance_away',  'score_home', 'score_away']

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

result = df_bets.merge(df_spi, on=['Date', 'HomeTeam', 'AwayTeam'], how='inner')

match_info = result[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'prob_homewin', 'prob_awaywin', 'prob_tie', 'importance_home', 'importance_away',  'score_home', 'score_away']]

#Convert date and time to unix time

combined_datetime = match_info['Date'] + ' ' + match_info['Time']
match_info['CombinedDateTime'] = pd.to_datetime(combined_datetime, format='%d/%m/%Y %H:%M')


match_info['UnixTimestamp'] = match_info['CombinedDateTime'].apply(lambda x: int(x.timestamp()))

#Start and end match unix time adjusted to GMT time zone
match_info['Unix_start'] = match_info['UnixTimestamp'] - 3600
match_info['Unix_end'] = match_info['UnixTimestamp'] + 3600

#Create gap score ecuation

match_info['R_home'] = (match_info['score_home'] > match_info['score_away']).astype(int) - (match_info['score_home'] < match_info['score_away']).astype(int)
match_info['R_away'] = -match_info['R_home']

match_info['beta_home'] = 0.2 * match_info['R_home']
match_info['beta_away'] = 0.2 * match_info['R_away']

match_info['prob_awaywin'] = match_info['prob_awaywin'].astype(float)
match_info['prob_homewin'] = match_info['prob_homewin'].astype(float)
match_info['score_home'] = match_info['score_home'].astype(int)
match_info['score_away'] = match_info['score_away'].astype(int)

match_info['gamma_home'] = (match_info['prob_awaywin'] * match_info['score_home'] - match_info['prob_homewin'] * match_info['score_away']) / (match_info['score_home'] + match_info['score_away'])

unique_values = match_info['HomeTeam'].unique()


#Offensive and defensive scores 

lista = [['Man City', 3.0, 0.2], ['Liverpool', 3.0, 0.4], ['Chelsea', 2.5, 0.4], ['Arsenal', 2.2, 0.5], 
         ['Tottenham',2.2,0.6], ['West Ham', 2.1, 0.8], ['Man United', 2.2, 0.7], ['Brighton', 1.9, 0.6], 
         ['Leicester', 2.2, 0.9], ['Wolves', 1.7, 0.5], ['Aston Villa', 2.0, 0.8], 
         ['Crystal Palace', 1.9, 0.8], ['Brentford', 1.8, 0.8], ['Everton', 1.9, 0.9],
         ['Southampton', 1.8, 0.9], ['Leeds',1.9, 1.0], ['Burnley',1.8, 1.0], ['Watford', 1.7, 1.1],
         ['Newcastle', 1.7, 1.1], ['Norwich', 1.5, 1.1]]




















