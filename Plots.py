import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import numpy as np

MatchInfo = pd.read_csv("C:/Users/tomas/Downloads/MatchInfo.csv")

# Local Positive

MatchInfo_plot = MatchInfo.dropna(subset=['gap_score_home', 'Local_positive'])

x = MatchInfo_plot['gap_score_home']
y = MatchInfo_plot['Local_positive']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score Home')
plt.ylabel('Local Positive')
plt.title('Gap Score and (%) of Positive Comments for Home Teams')
plt.legend()

plt.show()

# Local Negative

MatchInfo_plot = MatchInfo.dropna(subset=['gap_score_home', 'Local_negative'])

x = MatchInfo_plot['gap_score_home']
y = MatchInfo_plot['Local_negative']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score Home')
plt.ylabel('Local Negative')
plt.title('Gap Score and (%) of Negative Comments for Home Teams')
plt.legend()

plt.show()

# Away Positive

MatchInfo_plot = MatchInfo.dropna(subset=['gap_score_away', 'Away_positive'])

x = MatchInfo_plot['gap_score_away']
y = MatchInfo_plot['Away_positive']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score Away')
plt.ylabel('Away Positive')
plt.title('Gap Score and (%) of Positive Comments for Away Teams')
plt.legend()

plt.show()

# Away Negative

MatchInfo_plot = MatchInfo.dropna(subset=['gap_score_away', 'Away_negative'])

x = MatchInfo_plot['gap_score_away']
y = MatchInfo_plot['Away_negative']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

regression_line = slope * x + intercept

plt.scatter(x, y, label='Data', s=10)
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('Gap Score Away')
plt.ylabel('Away Negative')
plt.title('Gap Score and (%) of Negative Comments for Away Teams')
plt.legend()

plt.show()

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
