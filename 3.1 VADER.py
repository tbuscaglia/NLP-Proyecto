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
print(Pom_sentiment['Man United'])

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

sent_class_post = {}
for team in tqdm(team_names):
    sent_class_post[team] = {}
    for time in tqdm (team_played[team]):
        sent_class_post[team][time] = {}
        positive_count = 0
        negative_count = 0
        for comment in data_team_time[team][time]:
            sent =  analyze_sentiment(comment)
            if sent == "Positivo":
                 positive_count +=1
            if sent == "Negativo":
                 negative_count +=1
        sent_class_post[team][time] =[(positive_count/len(data_team_time[team][time]))*100 , (negative_count/len(data_team_time[team][time]))*100, 100 - (negative_count/len(data_team_time[team][time]))*100 - (positive_count/len(data_team_time[team][time]))*100]

prematch = 0
for team in tqdm(team_names):
    for time in tqdm (team_played[team]):
        prematch += len(data_team_time_pre[team][time])
print(prematch)

sent_class_pre = {}
for team in tqdm(team_names):
    sent_class_pre[team] = {}
    for time in tqdm (team_played[team]):
        sent_class_pre[team][time] = {}
        positive_count = 0
        negative_count = 0
        for comment in data_team_time_pre[team][time]:
            sent =  analyze_sentiment(comment)
            if sent == "Positivo":
                 positive_count +=1
            if sent == "Negativo":
                 negative_count +=1
        if len(data_team_time_pre[team][time])>0:
            sent_class_pre[team][time] =[(positive_count/len(data_team_time_pre[team][time]))*100 , (negative_count/len(data_team_time_pre[team][time]))*100, 100 - (negative_count/len(data_team_time_pre[team][time]))*100 - (positive_count/len(data_team_time_pre[team][time]))*100]

print(sent_class_pre["Norwich"][1651334400])

pos_change = {}
for team in team_names:
    pos_change[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        pos_change[team][time]= {}
        pos_change[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
                pos_change[team][time][gap] = (sent_class_post[team][time][0] - sent_class_pre[team][time][0])
list= []
for team, time_data in pos_change.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))
PosChange = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', 'Change in % positive comments'])
PosChange.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Results/PosChange.csv", index=True)

neg_change = {}
for team in team_names:
    neg_change[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        neg_change[team][time]= {}
        neg_change[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
                neg_change[team][time][gap] = (sent_class_post[team][time][1] - sent_class_pre[team][time][1])
list= []
for team, time_data in neg_change.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))
NegChange = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', 'Change in % negative comments'])
NegChange.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Results/NegChange.csv", index=True)

neu_change = {}
for team in team_names:
    neu_change[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        neu_change[team][time]= {}
        neu_change[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
                neu_change[team][time][gap] = (sent_class_post[team][time][2] - sent_class_pre[team][time][2])
list= []
for team, time_data in neu_change.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))
NeuChange = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', 'Change in % negative comments'])
NeuChange.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Results/NeuChange.csv", index=True)

pos = {}
for team in team_names:
    pos[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        pos[team][time]= {}
        pos[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
                pos[team][time][gap] = (sent_class_post[team][time][0])
list= []
for team, time_data in pos.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))
Positive = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', '% Positive Comments'])
Positive.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Results/POS.csv", index=True)


neg = {}
for team in team_names:
    neg[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        neg[team][time]= {}
        neg[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
                neg[team][time][gap] = (sent_class_post[team][time][1])
list= []
for team, time_data in neg.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))
Negative = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', '% Negative Comments'])
Negative.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Results/NEG.csv", index=True)

neu = {}
for team in team_names:
    neu[team] = {}
    for time in team_played[team]:
        gap = team_gap[team][time]
        neu[team][time]= {}
        neu[team][time][gap] = {}
        if len(data_team_time_pre[team][time]) > 0 and len(data_team_time[team][time])>0:
                neu[team][time][gap] = (sent_class_post[team][time][2])
list= []
for team, time_data in neu.items():
    for time, gap_data in time_data.items():
        for gap, value in gap_data.items():
            list.append((team, time, gap, value))
Neutral = pd.DataFrame(list, columns=['Team', 'Match End', 'Gap Score', '% Neutral Comments'])
Neutral.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Results/NEU.csv", index=True)
