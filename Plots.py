import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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



