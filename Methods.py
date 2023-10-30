import pandas as pd
from tqdm import tqdm
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()

team_names = ['Man United', 'Chelsea', 'Everton', 'Leicester',
  'Norwich', 'Newcastle', 'Tottenham', 'Liverpool', 'Aston Villa',
 'Crystal Palace', 'Leeds', 'Man City', 'Brighton', 'Southampton', 'Wolves',
 'Arsenal', 'West Ham']
team_data = {}
for team in tqdm(team_names):
    file_path = f"/Users/julianandelsman/Desktop/NLP/Final project/Data/{team}.csv"
    df_team = pd.read_csv(file_path, sep='\t')
    team_data[team] = df_team
    
MatchInfo = pd.read_csv('/Users/julianandelsman/Desktop/NLP/Final project/Data/MatchInfo.csv')

##VADER##
# Start VADER
analyzer = SentimentIntensityAnalyzer()
# Define la función para analizar el sentimiento y devolver una etiqueta
def analyze_sentiment(text):
    if isinstance(text, str): 
        sentiment = analyzer.polarity_scores(text)
        compound_score = sentiment['compound']

        if compound_score >= 0.05:
            return 'Positivo'
        elif compound_score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutral'
    else:
        return 'Neutral'

#Defining Quantile separation function
match_info_vader = MatchInfo[['HomeTeam', 'AwayTeam','Unix_start', 'Unix_end', 'gap_score_home','gap_score_away']]

team_played = {}

team_gap = {}
for index, row in tqdm(match_info_vader.iterrows()):
    if row['HomeTeam'] not in ['Brentford', 'Watford', 'Burnley']:
        team = row['HomeTeam']
        gap_score = row['gap_score_home']
        cuantil = gap_score
        match_time = row['Unix_end']
        if team not in team_gap:
            team_gap[team] = {}
        team_gap[team][match_time] = cuantil
        if team not in team_played:
            team_played[team] = []
        team_played[team].append(match_time)
    if row['AwayTeam'] not in ['Brentford', 'Watford', 'Burnley']:
        team = row['AwayTeam']
        gap_score = row['gap_score_away']
        cuantil = gap_score
        match_time = row['Unix_end']
        if team not in team_gap:
            team_gap[team] = {}        
        team_gap[team][match_time] = cuantil        
        if team not in team_played:
            team_played[team] = []
        team_played[team].append(match_time)


data_team_time = {}
for team in team_played:
    data_team_time[team] = {}
    match_starts = []
    for match_time in team_played[team]:
        match_starts.append([match_time,match_time+21600])
        data_team_time[team][match_time] = []
    for idx_2,row_2 in tqdm(team_data[team].iterrows()):  #This runs 18 times instead of 600 times
        if any(item[0] < row_2['Unix Date'] <= item[1] for item in match_starts):
            # Find the corresponding match start time for the comment
            comment_start = next(item[0] for item in match_starts if item[0] <= row_2['Unix Date'] <= item[1])
            data_team_time[team][comment_start].append(row_2['Comment'])

Pom_sentiment = {}
for team in team_names:
    #Pom_sentiment[team] = {}
    for time in team_played[team]:
        Pom_sentiment[team][time] = {}
        compound_sum = 0
        for i in range (len(data_team_time[team][time])):
            if isinstance(data_team_time[team][time][i], str):
                compound_sum += analyzer.polarity_scores(data_team_time[team][time][i])['compound']
        Pom_sentiment[team][time] = compound_sum/len(data_team_time[team][time])
print(team_played['Man United'])

#---------#

data_team_time_pre = {}
for team in team_played:
    data_team_time_pre[team] = {}
    match_starts = []
    for match_time in team_played[team]:
        match_starts.append([match_time - 28800,match_time - 7200])
        data_team_time_pre[team][match_time] = []
    for idx_2,row_2 in tqdm(team_data[team].iterrows()):  #This runs 18 times instead of 600 times
        if any(item[0] < row_2['Unix Date'] <= item[1] for item in match_starts):
            # Find the corresponding match start time for the comment
            comment_start = next(item[1]+7200 for item in match_starts if item[0] <= row_2['Unix Date'] <= item[1])
            data_team_time_pre[team][comment_start].append(row_2['Comment'])

Prem_sentiment = {}
for team in team_names:
    Prem_sentiment[team] = {}
    for time in team_played[team]:
        #Prem_sentiment[team][time] = {}
        compound_sum = 0
        for i in range (len(data_team_time_pre[team][time])):
            if isinstance(data_team_time_pre[team][time][i], str):
                compound_sum += analyzer.polarity_scores(data_team_time_pre[team][time][i])['compound']
        if len(data_team_time_pre[team][time]) == 0:
            Prem_sentiment[team][time] = []
        if len(data_team_time_pre[team][time])>0:
            Prem_sentiment[team][time] = compound_sum/len(data_team_time_pre[team][time])

print(Prem_sentiment['Norwich'][1652301900])
print(team_gap['Leicester'][1646487000])
sent_change = {}
for team in team_names:
    sent_change[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        sent_change[team][time]= {}
        sent_change[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
            if Prem_sentiment[team][time] != 0:
                sent_change[team][time][gap] = ((Pom_sentiment[team][time] - Prem_sentiment[team][time])/Prem_sentiment[team][time])*100


list= []
for team, time_data in sent_change.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))

SentChange = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', 'Percentage change in compound score'])
SentChange.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Data/Sentchange.csv", index=True)

print(SentChange.head())
'''
def positive_calc(team, time):
    positive_count = 0
    total_count = 0
    if team in ['Brentford', 'Watford', 'Burnley']:
        return None
    else: 
        for index, row in team_data[team].iterrows(): 
            if row['Unix Date']> time and row['Unix Date']< (time + 21600):
                total_count += 1
                if row['sentiments'] == "Positivo":
                    positive_count +=1
    if total_count > 0:
        perc_positive = (positive_count/total_count)*100
        return perc_positive
    else: 
        return None

MatchInfo['Local_positive'] = MatchInfo.progress_apply(lambda row: positive_calc(row['HomeTeam'], row['Unix_end']), axis=1)
MatchInfo['Away_positive'] = MatchInfo.progress_apply(lambda row: positive_calc(row['AwayTeam'], row['Unix_end']), axis=1)
def negative_calc(team, time):
    negative_count = 0
    total_count = 0
    if team in ['Brentford', 'Watford', 'Burnley']:
        return None
    else: 
        for index, row in team_data[team].iterrows(): 
            if row['Unix Date']> time and row['Unix Date']< (time + 21600):
                total_count += 1
                if row['sentiments'] == "Negativo":
                    negative_count +=1
    if total_count > 0:
        perc_negative = (negative_count/total_count)*100
        return perc_negative
    else: 
        return None
MatchInfo['Local_negative'] = MatchInfo.progress_apply(lambda row: negative_calc(row['HomeTeam'], row['Unix_end']), axis=1)
MatchInfo['Away_negative'] = MatchInfo.progress_apply(lambda row: negative_calc(row['AwayTeam'], row['Unix_end']), axis=1)
print(MatchInfo.head())
MatchInfo['Local_neutral'] = 100 - MatchInfo['Local_positive'] - MatchInfo['Local_negative']
MatchInfo['Away_neutral'] = 100 - MatchInfo['Away_positive'] - MatchInfo['Away_negative']
MatchInfo.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Data/MatchInfo.csv", index=False)
'''

#------------------------------------------------------------------------------
