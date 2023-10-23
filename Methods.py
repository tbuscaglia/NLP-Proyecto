import pandas as pd
from tqdm import tqdm
MatchInfo = pd.read_csv('/Users/julianandelsman/Desktop/NLP/Final project/Data/MatchInfo.csv')
Reddit = pd.read_csv('/Users/julianandelsman/Desktop/NLP/Final project/Data/Reddit.csv')
data_combined = pd.concat([MatchInfo['gap_score_home'], MatchInfo['gap_score_away']])
cuantiles_deseados = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
resultados_cuantiles = data_combined.quantile(cuantiles_deseados)
print(resultados_cuantiles)
##VADER##
# Start VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()
# Crear una instancia de SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Define la función para analizar el sentimiento y devolver una etiqueta
def analyze_sentiment(text):
    if isinstance(text, str):  # Verifica si el valor es una cadena
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
Reddit['sentiments'] = Reddit['Comment'].progress_apply(analyze_sentiment)
def sentcalc(team, time):
    positive_count = 0
    total_count = 0
    if team in ['Brenford', 'Watford', 'Burnley']:
        return None
    else: 
        for index, row in Reddit.iterrows(): 
            if row['Unix Date']> time and row['Unix Date']< (time + 21600) and team == row['Team']:
                total_count += 1
                if row['sentiments'] == "Positivo":
                    positive_count +=1
    if total_count > 0:
        perc_positive = (positive_count/total_count)*100
    else: 
        return None
    return perc_positive
MatchInfo['Local_positive'] = MatchInfo.progress_apply(lambda row: sentcalc(row['HomeTeam'], row['Unix_end']), axis=1)
print(MatchInfo.head())