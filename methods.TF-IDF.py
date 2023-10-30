import pandas as pd
from tqdm import tqdm
import numpy as np
import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#Opening files (Match info and reddits)

MatchInfo = pd.read_csv('/Users/julianandelsman/Desktop/NLP/Final project/Data/MatchInfo.csv')

team_names = ['Man United', 'Chelsea', 'Everton', 'Leicester',
  'Norwich', 'Newcastle', 'Tottenham', 'Liverpool', 'Aston Villa',
 'Crystal Palace', 'Leeds', 'Man City', 'Brighton', 'Southampton', 'Wolves',
 'Arsenal', 'West Ham']
team_data = {}
for team in team_names:
    file_path = f"/Users/julianandelsman/Desktop/NLP/Final project/Data/{team}.csv"
    df_team = pd.read_csv(file_path, sep='\t')
    team_data[team] = df_team
    
#Defining Quantile separation function
match_info_tfidf = MatchInfo[['HomeTeam', 'AwayTeam', 'Unix_end', 'gap_score_home','gap_score_away']]

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

gap_scores_combined = np.concatenate([match_info_tfidf['gap_score_home'], match_info_tfidf['gap_score_away']])

cuantiles = np.quantile(gap_scores_combined, quantiles)
def encontrar_cuantil(gapscore):
    for i in range(len(cuantiles)):
        if gapscore <= cuantiles[i]:
            return i
    return len(cuantiles)
comentarios_cuantil = {}
team_played = {}

team_gap = {}
for index, row in tqdm.tqdm(match_info_tfidf.iterrows()):
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
    for idx_2,row_2 in tqdm.tqdm(team_data[team].iterrows()):  #This runs 18 times instead of 600 times
        if any(item[0] < row_2['Unix Date'] <= item[1] for item in match_starts):
            # Find the corresponding match start time for the comment
            comment_start = next(item[0] for item in match_starts if item[0] <= row_2['Unix Date'] <= item[1])
            data_team_time[team][comment_start].append(row_2['Comment'])


team_cuantile = {}
for team in data_team_time:
    for time in data_team_time[team]:
        gap = team_gap[team][time] + 1
        if gap not in team_cuantile:
            team_cuantile[gap]={}
        if team not in team_cuantile[gap]:
            team_cuantile[gap][team] = []
        texts = data_team_time[team][time]
        team_cuantile[gap][team].extend(texts)

def comment_processing(text_list):
    processed_texts = []
    for text in text_list:
        if isinstance(text, str):
            text = text.lower()
            words = nltk.word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words and word.isalnum()]
            words = [lemmatizer.lemmatize(word) for word in words]
            processed_text = ' '.join(words)
            processed_texts.append(processed_text)
    combined_text = ' '.join(processed_texts)
    return combined_text

from tqdm import tqdm
for gap in tqdm(team_cuantile):
    for team in tqdm(team_cuantile[gap]):
        team_cuantile[gap][team] = comment_processing(team_cuantile[gap][team])

text_by_quantile = {}
for gap in team_cuantile:
    text = []
    for team in team_cuantile[gap]:
        text.append(team_cuantile[gap][team])
    text_by_quantile[gap] = ' '.join(text)

# Initialize the TF-IDF vectorizer for text
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords_list = stopwords.words("english")
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000, stop_words = stopwords_list)
tfidf_matrix = tfidf_vectorizer.fit_transform(text_by_quantile.values())

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=tfidf_feature_names, index=text_by_quantile.keys())

tfidf_df.to_csv("/Users/julianandelsman/Desktop/NLP/Final project/Data/TDIDF.csv", index=True)

#Paso de Df a Dict
 
diccionarios = {}

# Recorre las filas del DataFrame y crea un diccionario por fila
for index, fila in tfidf_df.iterrows():
    diccionario_fila = fila.to_dict()
    # Agrega el diccionario de la fila al diccionario final con el nombre de la fila como clave
    diccionarios[index] = diccionario_fila

i = 0
result = {}
for i in range(1, len(diccionarios) + 1):
    claves_top10 = sorted(diccionarios[i], key=diccionarios[i].get, reverse=True)[:10]
    resultado = [(clave, diccionarios[i][clave]) for clave in claves_top10]
    
    # Crear un DataFrame para este 'i'
    df = pd.DataFrame(resultado, columns=['Word', 'TF-IDF Score'])
    
    # Agregar la columna 'Rank' al DataFrame
    result[i] = df
    

for i in range (1, 11):
    result[i].to_csv(f'/Users/julianandelsman/Desktop/NLP/Final project/Data/Tf-Idf{i}.csv', index=True)
'''
from wordcloud import WordCloud

agg_tfidf = tfidf_df.sum().sort_values(ascending=False)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(agg_tfidf)

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("TF-IDF Values")
wordcloud.to_file('wordcloud_en.png')
plt.show()
'''
