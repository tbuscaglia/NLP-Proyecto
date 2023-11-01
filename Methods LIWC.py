from tqdm import tqdm
import pandas as pd
import numpy as np

#Abriendo datos
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
match_info_liwc = MatchInfo[['HomeTeam', 'AwayTeam', 'Unix_end', 'gap_score_home','gap_score_away']]

#Seteando datos
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

gap_scores_combined = np.concatenate([match_info_liwc['gap_score_home'], match_info_liwc['gap_score_away']])

cuantiles = np.quantile(gap_scores_combined, quantiles)
print(cuantiles)
def encontrar_cuantil(gapscore):
    for i in range(len(cuantiles)):
        if gapscore <= cuantiles[i]:
            return i
    return len(cuantiles)
team_played = {}

team_gap = {}
for index, row in tqdm(match_info_liwc.iterrows()):
    if row['HomeTeam'] not in ['Brentford', 'Watford', 'Burnley']:
        team = row['HomeTeam']
        gap_score = row['gap_score_home']
        cuantil = encontrar_cuantil(gap_score)
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
        cuantil = encontrar_cuantil(gap_score)
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
    for idx_2,row_2 in tqdm(team_data[team].iterrows()):  
        if any(item[0] < row_2['Unix Date'] <= item[1] for item in match_starts):
            comment_start = next(item[0] for item in match_starts if item[0] <= row_2['Unix Date'] <= item[1])
            data_team_time[team][comment_start].append(row_2['Comment'])
#Separando los comentarios x equipo y x cuantil
        
comments_by_quantile = {}

for team in team_names:
    for time in team_played[team]:
        gap = team_gap[team][time] + 1
        if gap not in comments_by_quantile:
            comments_by_quantile[gap]=[]
        comments_by_quantile[gap].extend(data_team_time[team][time])

comments_by_team = {}
for team in team_names:
    if team not in comments_by_team:
            comments_by_team[team] = []
    for time in team_played[team]:
        comments_by_team[team].extend(data_team_time[team][time])

#probando que todos tengan la misma cant de comments
comment1 = 0
for team in team_names:
    comment1 += len(comments_by_team[team])   
print(comment1)

comment2 = 0
for gap in range (1,11):
    comment2 += len(comments_by_quantile[gap])   
print(comment2)

comment3 = 0
for team in team_names:
    for time in team_played[team]:
        comment3 += len(data_team_time[team][time])
print(comment3)
