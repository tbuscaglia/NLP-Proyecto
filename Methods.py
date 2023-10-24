import pandas as pd
from tqdm import tqdm
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
for team in team_names:
    team_data[team]['sentiments'] = team_data[team]['Comment'].progress_apply(analyze_sentiment)

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

#------------------------------------------------------------------------------
## TF-IDF ##
'''
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from stop_words import get_stop_words

# Download necessary NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Obtén las stopwords en inglés de NLTK
english_stop_words = set(stopwords.words('english'))

# Load the CSV file with your song data (replace with your data file)
data = {}
with open('english_songs.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Assuming the first row contains column headers
    for row in csv_reader:
        artist = row[0]  # Adjust column indices to match your file
        song = row[1]
        lyrics = row[2]
        if artist not in data:
            data[artist] = []
        data[artist].append(lyrics)

# Initialize the TF-IDF vectorizer for text

tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000, stop_words=combined_stop_words)

# Lemmatize and preprocess the lyrics data
lemmatizer = WordNetLemmatizer()
tfidf_data = {}
for artist in data:
    tfidf_data[artist] = ' '.join([lemmatizer.lemmatize(word.lower()) for lyrics in data[artist] for word in word_tokenize(lyrics) if word.isalpha()])

# Perform TF-IDF analysis
tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_data.values())
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=tfidf_feature_names, index=tfidf_data.keys())

# Create a WordCloud from aggregated TF-IDF values
agg_tfidf = tfidf_df.sum().sort_values(ascending=False)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(agg_tfidf)

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("TF-IDF Values for english songs")
wordcloud.to_file('wordcloud_en.png')
plt.show()
'''
