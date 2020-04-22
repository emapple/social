import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
DATA_DIR = os.path.abspath('../state_data')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures

from utils.energy_tools import read_region


def df_for_model(state):
    df_tot = pd.DataFrame([])
    """creates a dataframe of the form index: Date, columns : Onehotencoded days, t, t^2, t^3, t^4, Demand """

    states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
    }
    
#     reg_dict = {'California': 'CAL', 'Carolinas': 'CAR',
#                 'Central': 'CENT', 'Florida': 'FLA',
#                 'Mid-Atlantic': 'MIDA', 'Midwest': 'MIDW',
#                 'New England': 'NE', 'New York': 'NY',
#                 'Northwest': 'NW', 'Southeast': 'SE',
#                 'Southwest': 'SW', 'Tennessee': 'TEN',
#                 'Texas': 'TEX'}

    filepath_15to19 = "/Users/NikhilStuff/Desktop/social_backup/state_data/" + state + "_2015-2019_raw.csv"
    filepath_2020 = "/Users/NikhilStuff/Desktop/social_backup/state_data/" + state + "_2020_raw.csv"
    
    T_train = pd.read_csv(filepath_15to19, index_col = 'DATE', parse_dates = True)
    T_test = pd.read_csv(filepath_2020, index_col = 'DATE', parse_dates = True)
    T_all = T_train.append(T_test)
    T_all.dropna(inplace = True)
    
    T_all['DOW'] = T_all.index.dayofweek
    dummies = pd.get_dummies(T_all['DOW'])  
    
    r_avg = (T_all['MEAN_T']
                               .rolling(7, win_type='boxcar')
                               .mean()
                               )
    r_avg.dropna(inplace = True)

    
    X = r_avg.values.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    deg = 5
    
    poly = PolynomialFeatures(deg, include_bias = False)
    Poly4_all= pd.DataFrame(poly.fit_transform(X),
                            index = r_avg.index, 
                            columns = [f'T{i}' for i in range(deg)]) 
    
    Poly4_all= Poly4_all.merge(dummies, on = 'DATE', how = 'left')
    
    #add more features here
    
    energy_all = read_region(states[state])
    rolling_mean_demand = (energy_all
                                    .groupby(level=0)['Demand']
                                    .sum(min_count =24)
                                    .rolling(7, win_type='boxcar')
                                    .mean())
    
    rolling_mean_demand.dropna(inplace =True)
    
    Poly4_all = Poly4_all.merge(rolling_mean_demand.to_frame(),
                                left_index = True, right_index = True, how = 'inner')
    
    df_train = Poly4_all[Poly4_all.index < pd.to_datetime('20200101')]
    df_test = Poly4_all[Poly4_all.index >= pd.to_datetime('20200101')]
    
    return df_train, df_test


def plot_model(state):
    
    """Fits a Ridge regression model to the df and outputs a plot of the model data overlaid on the 
    training data"""

    df_train, df_test = df_for_model(state)

    
    X_train = df_train[df_train.columns[:-1]].values
    y_train = df_train[df_train.columns[-1]].values
    X_test = df_test[df_test.columns[:-1]].values
    y_test = df_test[df_test.columns[-1]].values
    
    model = Ridge(alpha = 1e-3)
    model.fit(X_train,y_train)
    
    print(f'Score : {model.score(X_test, y_test)}')
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)


    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
    fig.set_size_inches(15,5)

    ax[0].plot(df_train.index, y_pred_train/1e3, 'r-', label = 'fit on Training')
    ax[0].plot(df_train.index, y_train/1e3, 'k-', alpha = 0.7, label = 'Training data')
    
    ax[0].plot(df_test.index, y_pred_test/1e3, 'g-', label = 'fit on Training')
    ax[0].plot(df_test.index, y_test/1e3, 'k-', alpha = 0.7, label = 'Testing Data')
    
    ax[1].plot(df_test.index, y_pred_test/1e3, 'g-', label = 'fit on Training')
    ax[1].plot(df_test.index, y_test/1e3, 'k-', alpha = 0.7, label = 'Testing Data')
    
    for a in ax:
        a.legend()
        a.set_ylabel('Demand(GWH)')
    
    ax[0].set_title(f'Electricity demand: expectation vs reality in {state}')
    ax[1].set_title('2020 only')
    xticklabels = df_test.index.astype('str')
    ax[1].set_xticklabels(xticklabels[: :7], rotation = 45)
    plt.tight_layout()
    plt.show()

