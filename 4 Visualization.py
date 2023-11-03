import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import webbrowser



'''
Gap Score vs Percent of Positive Comments
'''
positive_data = pd.read_csv('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/POS.csv')

positive_data = positive_data.dropna(subset=['Gap Score', '% Positive Comments'])


mask = positive_data ['% Positive Comments'].str.contains('{}')

positive_data  = positive_data [~mask]

positive_data ['% Positive Comments'] = positive_data ['% Positive Comments'].astype(float)

x = positive_data['Gap Score']
y = positive_data['% Positive Comments']
print(len(x))

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('% of Positive Comments')
plt.title('Gap Score and (%) of Positive Comments')
plt.legend()

plt.show()

r_squared = r_value ** 2
print(f'R-squared: {r_squared:.2f}')

#------------------------------------------------------------------------------

fig = px.scatter(positive_data, x='Gap Score', y='% Positive Comments', color='Team',
                 labels={'Gap Score': 'Gap Score', '% Positive Comments': '% Positive Comments'},
                 title='Gap Score vs Percentage of Post-Match Positive Comments',
                 trendline='ols')

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(label='Show Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, True]}]),  # Show the regression line
                dict(label='Hide Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, False]}])  # Hide the regression line
            ]
        )
    ]
)


fig.update_xaxes(range=[-0.2, 0.5])  # Replace xmin and xmax with your desired range
fig.update_yaxes(range=[15, 90])
fig.write_html("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive Gap vs Positive Comments.html")

webbrowser.open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive Gap vs Positive Comments.html')

'''
Gap Score vs Percent of Neutral Comments
'''
neutral_data = pd.read_csv('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/NEU.csv')

neutral_data  = neutral_data .dropna(subset=['Gap Score', '% Neutral Comments'])


mask = neutral_data  ['% Neutral Comments'].str.contains('{}')

neutral_data   = neutral_data  [~mask]

neutral_data  ['% Neutral Comments'] = neutral_data  ['% Neutral Comments'].astype(float)

x = neutral_data ['Gap Score']
y = neutral_data ['% Neutral Comments']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

y_min_limit = -1  # Define your desired minimum limit
y_max_limit = 50  # Define your desired maximum limit
plt.ylim(y_min_limit, y_max_limit)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('% of Neutral Comments')
plt.title('Gap Score and (%) of Neutral Comments')
plt.legend()

plt.show()

'''
Gap Score vs Percent of Negative Comments
'''
negative_data = pd.read_csv('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/NEG.csv')

negative_data  = negative_data .dropna(subset=['Gap Score', '% Negative Comments'])


mask = negative_data  ['% Negative Comments'].str.contains('{}')

negative_data   = negative_data  [~mask]

negative_data  ['% Negative Comments'] = negative_data  ['% Negative Comments'].astype(float)

x = negative_data ['Gap Score']
y = negative_data ['% Negative Comments']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('% of Negative Comments')
plt.title('Gap Score and (%) of Negative Comments')
plt.legend()

plt.show()
r_squared = r_value ** 2
print(f'R-squared: {r_squared:.2f}')

fig = px.scatter(negative_data, x='Gap Score', y='% Negative Comments', color='Team',
                 labels={'Gap Score': 'Gap Score', '% Negative Comments': '% Negative Comments'},
                 title='Gap Score vs Percentage of Post-Match Negative Comments',
                 trendline='ols')

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(label='Show Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, True]}]),  # Show the regression line
                dict(label='Hide Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, False]}])  # Hide the regression line
            ]
        )
    ]
)


fig.update_xaxes(range=[-0.2, 0.5])  # Replace xmin and xmax with your desired range
fig.update_yaxes(range=[0, 50])
fig.write_html("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive Gap vs Negative Comments.html")

webbrowser.open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive Gap vs Negative Comments.html')

#------------------------------------------------------------------------------

# Amount of comments per day for top 2 teams

man_city = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/Datos limpios/processed comments/Man City.csv")
liverpool = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/Datos limpios/processed comments/Liverpool.csv")

city = man_city['Unix Date'].tolist()
liv = liverpool['Unix Date'].tolist()


'''

Liverpool graphs

'''

days = {}

start  = 1628650800

for date in liv:
    day = (date - start) // 86400  
    if day not in days:
        days[day] = 0
    days[day] += 1
 
liv_df = pd.DataFrame(list(days.items()), columns=['Day', 'Count'])   

plt.figure(figsize=(10, 6))
plt.bar(liv_df['Day'], liv_df['Count'], width=0.5, color='skyblue')
plt.xlabel('Day')
plt.ylabel('Date Count')
plt.title('Date Counts by Day')
plt.xticks(liv_df['Day'])
plt.show()

#Time Series with 7 days rolling average. 

liv_df = liv_df.sort_values(by='Day')

window_size = 7  # Adjust the window size as needed

plt.figure(figsize=(12, 6))
plt.plot(liv_df['Day'], liv_df['Count'], markersize=3, marker='o', linestyle='-', color='b', label='Daily Comments')
plt.xlabel('Day Since Start of Premier League')
plt.ylabel('Daily Comments')
plt.title('Number of Comments per Day in Liverpool Subreddit with 7 Day Rolling Average')
plt.grid(True)
plt.xticks(rotation=45)

rolling_trend = liv_df['Count'].rolling(window=window_size, min_periods=1).mean()
plt.plot(liv_df['Day'], rolling_trend, color='r', linestyle='--', label='7 day rolling average')

plt.legend()
plt.tight_layout()
plt.show()

'''

Manchester City graphs

'''

days = {}

start  = 1628650800

for date in city:
    day = (date - start) // 86400  
    if day not in days:
        days[day] = 0
    days[day] += 1
 
city_df = pd.DataFrame(list(days.items()), columns=['Day', 'Count'])   

plt.figure(figsize=(10, 6))
plt.bar(city_df['Day'], city_df['Count'], width=0.5, color='skyblue')
plt.xlabel('Day')
plt.ylabel('Date Count')
plt.title('Date Counts by Day')
plt.xticks(liv_df['Day'])
plt.show()

#Time Series with 7 days rolling average. 

city_df = city_df.sort_values(by='Day')

window_size = 7  # Adjust the window size as needed

plt.figure(figsize=(12, 6))
plt.plot(city_df['Day'], city_df['Count'], markersize=3, marker='o', linestyle='-', color='b', label='Daily Comments')
plt.xlabel('Day Since Start of Premier League')
plt.ylabel('Daily Comments')
plt.title('Number of Comments per Day in Manchester City Subreddit with 7 Day Rolling Average')
plt.grid(True)
plt.xticks(rotation=45)

rolling_trend = city_df['Count'].rolling(window=window_size, min_periods=1).mean()
plt.plot(city_df['Day'], rolling_trend, color='r', linestyle='--', label='7 day rolling average')

plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------

sent_change = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/Sentchange.csv")

sent_change = sent_change.dropna()

# Calculate Z-scores for your data
z_scores = np.abs(stats.zscore(sent_change['Percentage change in compound score']))

# Define a threshold for the Z-score beyond which data points are considered outliers
threshold = 3  # You can adjust this threshold as needed

# Remove outliers based on the threshold
filtered_data = sent_change[(z_scores < threshold)]

# Extract x and y from the filtered data
x = filtered_data['Gap Score']
y = filtered_data['Percentage change in compound score']

# Perform linear regression on the filtered data
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
regression_line = slope * x + intercept

# Set the Y-axis limits
y_min_limit = -400  # Define your desired minimum limit
y_max_limit = 400  # Define your desired maximum limit
plt.ylim(y_min_limit, y_max_limit)

# Plot the regression line and data points
plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('Percentage change in compound score')
plt.title('Percentage change in compound score')
plt.legend()

plt.show()

fig = px.scatter(filtered_data, x='Gap Score', y='Percentage change in compound score', color='Team',
                 labels={'Gap Score': 'Gap Score', 'Percentage change in compound score': 'Percentage change in compound score'},
                 title='Percentage Change in Compound Score vs Gap Score',
                 trendline='ols')

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(label='Show Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, True]}]),  # Show the regression line
                dict(label='Hide Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, False]}])  # Hide the regression line
            ]
        )
    ]
)


fig.update_xaxes(range=[-0.3, 0.5])  
fig.update_yaxes(range=[-400, 400])
fig.write_html("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive Compound change.html")

webbrowser.open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive Compound change.html')






'''
Change in percentage of comments with POSITIVE gap score from pre-match vs post-match
'''

pos_change = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/PosChange.csv")

pos_change = pos_change.dropna()

mask = pos_change['Change in % positive comments'].str.contains('{}')

pos_change = pos_change[~mask]

pos_change['Change in % positive comments'] = pos_change['Change in % positive comments'].astype(float)

x = pos_change['Gap Score']
y = pos_change['Change in % positive comments']

# Perform linear regression on the filtered data
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
regression_line = slope * x + intercept

# Set the Y-axis limits
y_min_limit = -75  # Define your desired minimum limit
y_max_limit = 75  # Define your desired maximum limit
plt.ylim(y_min_limit, y_max_limit)

# Plot the regression line and data points
plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('Change in % positive comments')
plt.title('Gap Score and Change in % of Positive Comments')
plt.legend()

plt.show()
r_squared = r_value ** 2
print(f'R-squared: {r_squared:.2f}')

#Interactive Plot
fig = px.scatter(pos_change, x='Gap Score', y='Change in % positive comments', color='Team',
                 labels={'Gap Score': 'Gap Score', 'Change in % positive comments': 'Change in % positive comments'},
                 title='Percentage of Positive Comment Change vs Gap Score',
                 trendline='ols')

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(label='Show Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, True]}]),  # Show the regression line
                dict(label='Hide Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, False]}])  # Hide the regression line
            ]
        )
    ]
)


fig.update_xaxes(range=[-0.3, 0.5])  
fig.update_yaxes(range=[-75, 75])
fig.write_html("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive positive percentage change.html")

webbrowser.open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive positive percentage change.html')



'''
Change in percentage of comments with NEUTRAL gap score from pre-match vs post-match
'''

neu_change = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/NeuChange.csv")

neu_change  = neu_change .dropna()

mask = neu_change ['Change in % negative comments'].str.contains('{}')

neu_change  = neu_change [~mask]

neu_change ['Change in % negative comments'] = neu_change ['Change in % negative comments'].astype(float)

x = neu_change ['Gap Score']
y = neu_change ['Change in % negative comments']

# Perform linear regression on the filtered data
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
regression_line = slope * x + intercept

# Set the Y-axis limits
y_min_limit = -75  # Define your desired minimum limit
y_max_limit = 75  # Define your desired maximum limit
plt.ylim(y_min_limit, y_max_limit)

# Plot the regression line and data points
plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('Change in % Neutral Comments')
plt.title('Gap Score and Change in % of Neutral Comments')
plt.legend()

plt.show()


'''
Change in percentage of comments with NEGATIVE gap score from pre-match vs post-match
'''

neg_change = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/NegChange.csv")

neg_change  = neg_change .dropna()

mask = neg_change ['Change in % negative comments'].str.contains('{}')

neg_change  = neg_change [~mask]

neg_change ['Change in % negative comments'] = neg_change ['Change in % negative comments'].astype(float)

x = neg_change ['Gap Score']
y = neg_change ['Change in % negative comments']

# Perform linear regression on the filtered data
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
regression_line = slope * x + intercept

# Set the Y-axis limits
y_min_limit = -75  # Define your desired minimum limit
y_max_limit = 75  # Define your desired maximum limit
plt.ylim(y_min_limit, y_max_limit)

# Plot the regression line and data points
plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score')
plt.ylabel('Change in % Negative Comments')
plt.title('Gap Score and Change in % of Negative Comments')
plt.legend()

plt.show()
r_squared = r_value ** 2
print(f'R-squared: {r_squared:.2f}')

#Interactive Plot
fig = px.scatter(neg_change, x='Gap Score', y='Change in % negative comments', color='Team',
                 labels={'Gap Score': 'Gap Score', 'Change in % negative comments': 'Change in % negative comments'},
                 title='Percentage of Negative Comment Change vs Gap Score',
                 trendline='ols')

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(label='Show Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, True]}]),  # Show the regression line
                dict(label='Hide Regression Lines',
                     method='restyle',
                     args=[{'visible': [True, False]}])  # Hide the regression line
            ]
        )
    ]
)


fig.update_xaxes(range=[-0.3, 0.5])  
fig.update_yaxes(range=[-75, 75])
fig.write_html("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive negative percentage change.html")

webbrowser.open('C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Final Project/Plots/Interactive negative percentage change.html')



#------------------------------------------------------------------------------

team_LIWC = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/team_LIWC.csv")

team_LIWC = team_LIWC.T

team_LIWC.columns = team_LIWC.iloc[0]

# Remove the first row which now contains column names
team_LIWC = team_LIWC[1:]

# Reset the index
team_LIWC.reset_index(drop=True, inplace=True)

correlations = team_LIWC.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, cmap="coolwarm", square=True)
plt.title("Team Correlations")
plt.show()

#------------------------------------------------------------------------------
gap_LIWC = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/gap_LIWC.csv")

gap_LIWC = gap_LIWC.iloc[:, 1:]

gap_LIWC.reset_index(drop=True, inplace=True)

correlations_gap = gap_LIWC.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlations_gap, annot=True, cmap="coolwarm", square=True)
plt.title("Team Correlations")
plt.show()

#------------------------------------------------------------------------------
#WORD CLOUD 

tf = pd.read_csv("C:/Users/tomas/Documents/UdeSA/Tercer Año/Segundo Cuatri/NLP/Datos Proyecto/TFIDF.csv")

tf = tf.iloc[:, 1:]

tf = tf.T

tf.columns = [str(i) for i in range(1, len(tf.columns) + 1)]

for column in tf.columns:
    cuantil = tf[column]

    # Create a dictionary from the words and their scores in the current column
    word_scores = cuantil.to_dict()

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Column {column}')
    plt.axis("off")
    print(column)
    plt.show()

#------------------------------------------------------------------------------





