import csv
import pandas as pd
import math
from tqdm import tqdm
import re
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

result = df_bets.merge(df_spi, on=['Date', 'HomeTeam', 'AwayTeam'], how='inner')

match_info = result[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'prob_homewin', 'prob_awaywin', 'prob_tie', 'importance_home', 'importance_away',  'score_home', 'score_away']]

match_info['prob_awaywin'] = match_info['prob_awaywin'].astype(float)
match_info['prob_homewin'] = match_info['prob_homewin'].astype(float)
match_info['importance_home'] = match_info['importance_home'].astype(float)
match_info['importance_away'] = match_info['importance_away'].astype(float)
match_info['score_home'] = match_info['score_home'].astype(int)
match_info['score_away'] = match_info['score_away'].astype(int)

#Convert date and time to unix time

combined_datetime = match_info['Date'] + ' ' + match_info['Time']
match_info['CombinedDateTime'] = pd.to_datetime(combined_datetime, format='%d/%m/%Y %H:%M')


match_info['UnixTimestamp'] = match_info['CombinedDateTime'].apply(lambda x: int(x.timestamp()))

#Start and end match unix time adjusted to GMT time zone
match_info['Unix_start'] = match_info['UnixTimestamp'] - 3600
match_info['Unix_end'] = match_info['UnixTimestamp'] + 3600

#Create gap score ecuation

match_info['gd_home'] = match_info['score_home'] - match_info['score_away']
match_info['gd_away'] = -match_info['gd_home']

match_info['big_win_home'] = (match_info['gd_home'] > 2).astype(int)
match_info['big_loss_away'] = -match_info['big_win_home']
match_info['big_win_away'] = (match_info['gd_away'] > 2).astype(int)
match_info['big_loss_home'] = -match_info['big_win_away']

match_info['R_home'] = (match_info['score_home'] > match_info['score_away']).astype(int) - (match_info['score_home'] < match_info['score_away']).astype(int)
match_info['R_away'] = -match_info['R_home']

match_info['beta_home'] = 0.2 * (match_info['R_home'] + 0.5 * (match_info['big_win_home'] + match_info['big_loss_home']))
match_info['beta_away'] = 0.2 * (match_info['R_away'] + 0.5 * (match_info['big_win_away'] + match_info['big_loss_away']))

#Offensive and defensive scores 

team_scores = [['Man City', 3.0, 0.2], ['Liverpool', 3.0, 0.4], ['Chelsea', 2.5, 0.4], ['Arsenal', 2.2, 0.5], 
         ['Tottenham',2.2,0.6], ['West Ham', 2.1, 0.8], ['Man United', 2.2, 0.7], ['Brighton', 1.9, 0.6], 
         ['Leicester', 2.2, 0.9], ['Wolves', 1.7, 0.5], ['Aston Villa', 2.0, 0.8], 
         ['Crystal Palace', 1.9, 0.8], ['Brentford', 1.8, 0.8], ['Everton', 1.9, 0.9],
         ['Southampton', 1.8, 0.9], ['Leeds',1.9, 1.0], ['Burnley',1.8, 1.0], ['Watford', 1.7, 1.1],
         ['Newcastle', 1.7, 1.1], ['Norwich', 1.5, 1.1]]

team_scores = pd.DataFrame(team_scores, columns=['Team', 'OffensiveScore', 'DefensiveScore'])

offensive_scores_home = []
defensive_scores_home = []
offensive_scores_away = []
defensive_scores_away = []

for index, row in match_info.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    
    home_team_scores = team_scores[team_scores['Team'] == home_team]
    away_team_scores = team_scores[team_scores['Team'] == away_team]
    
    offensive_scores_home.append(home_team_scores['OffensiveScore'].values[0])
    defensive_scores_home.append(home_team_scores['DefensiveScore'].values[0])
    offensive_scores_away.append(away_team_scores['OffensiveScore'].values[0])
    defensive_scores_away.append(away_team_scores['DefensiveScore'].values[0])
    
match_info['off_home'] = offensive_scores_home
match_info['def_home'] = defensive_scores_home
match_info['off_away'] = offensive_scores_away
match_info['def_away'] = defensive_scores_away

match_info['gamma_home'] = (match_info['prob_awaywin'] * match_info['off_home'] - match_info['prob_homewin'] * match_info['def_home']) / (match_info['off_home'] + match_info['def_home'])
match_info['gamma_away'] = (match_info['prob_homewin'] * match_info['off_away'] - match_info['prob_awaywin'] * match_info['def_away']) / (match_info['off_away'] + match_info['def_away'])


match_info['mu_home'] = match_info['importance_home'].apply(lambda x: math.exp((x/100) - 1))
match_info['mu_away'] = match_info['importance_away'].apply(lambda x: math.exp((x/100) - 1))

match_info['gap_score_home'] = match_info['mu_home'] * (match_info['gamma_home'] + match_info['beta_home'])
match_info['gap_score_away'] = match_info['mu_away'] * (match_info['gamma_away'] + match_info['beta_away'])

match_info_clean = match_info[['HomeTeam', 'AwayTeam', 'Unix_start', 'Unix_end', 'gap_score_home', 'gap_score_away']]

#------------------------------------------------------------------------------
#MERGING REDDIT POSTS INTO A SINGLE .PD
team_names = ['Man United', 'Chelsea', 'Everton', 'Leicester',
  'Norwich', 'Newcastle', 'Tottenham', 'Liverpool', 'Aston Villa',
 'Crystal Palace', 'Leeds', 'Man City', 'Brighton', 'Southampton', 'Wolves',
 'Arsenal', 'West Ham']

team_data = {} 
df_names = []

for team in tqdm(team_names):
    file_path = f"/Users/julianandelsman/Desktop/NLP/Final project/Data/{team}.csv"
    df_team = pd.read_csv(file_path, sep='\t')
    team_data[team] = df_team
    df_names.append(re.sub(' ', '_', team))

#We create a list, containing all the dfs for each team, and adding a 3rd column that identifies the team
team_dfs = []
for name, team in zip(df_names, team_data):
    team_data[team]['Team'] = team
    team_dfs.append(team_data[team])
    
#Finally, we concatenate all the dfs into a single one
reddit = pd.concat(team_dfs, ignore_index=True)
reddit = reddit[['Unix Date', 'Comment', 'Team']]

'''
print(len(reddit))
for i in range(len(team_dfs)):
    print(len(team_dfs[i]),team_dfs[i]['Team'][0])

print(reddit)

'''




