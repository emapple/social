import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import matplotlib.dates as mdates

def insert_year(url, year, pattern='xxxx'):
    """Replace "pattern" with "year" in the url"""
    return re.sub(pattern, str(year), url)

def read_eia(year, interval):
    """Read one CSV from eia.gov
    
    year : the year
    interval : 1 or 2 (Jan - Jun or Jul - Dec)
    """
    
    generic_url = 'https://www.eia.gov/realtime_grid/sixMonthFiles/EIA930_BALANCE_xxxx_'
    generic_url += 'Jan_Jun.csv' if interval == 1 else 'Jul_Dec.csv'
    
    df = pd.read_csv(insert_year(generic_url, year), parse_dates=['Data Date'], 
                 index_col=['Balancing Authority', 'Data Date', 'Hour Number'],
                 usecols=['Balancing Authority', 'Data Date', 'Hour Number', 'Demand (MW)'],
                 thousands=',')
    df.rename(columns={'Demand (MW)': 'Demand'}, inplace=True)
    return df

def read_all_eia():
    """Read all available CSVs from eia.gov for Jan-Jun"""
    year_list = np.arange(2016, 2021, 1)
    df_list = []
    for year in year_list:
        df_list.append(read_eia(year, 1))
        if year < 2020:
            df_list.append(read_eia(year, 2))
    df = pd.concat(df_list)
#     df = pd.concat((read_eia(year, 1) for year in year_list))
    df = df.sort_index()
    
    return df


'''Region based datasets'''

def read_region(region):
    """Read the region-specific data"""
    reg_dict = {'California': 'CAL', 'Carolinas': 'CAR', 
                'Central': 'CENT', 'Florida': 'FLA', 
                'Mid-Atlantic': 'MIDA', 'Midwest': 'MIDW', 
                'New England': 'NE', 'New York': 'NY', 
                'Northwest': 'NW', 'Southeast': 'SE', 
                'Southwest': 'SW', 'Tennessee': 'TEN', 
                'Texas': 'TEX'}
    reg_url = 'https://www.eia.gov/realtime_grid/knownissues/xls/Region_xxx.xlsx'
    reg_url = insert_year(reg_url, reg_dict[region], 'xxx')
    df = pd.read_excel(reg_url, parse_dates=['Date', 'Local Time'], 
                       usecols=['Date', 'Local Time', 'D', 'DF'])
    hours = df['Local Time'].dt.hour.values
    hours[hours == 0] = 24
    
    df.index = pd.MultiIndex.from_arrays([df['Date'], hours], names=['Date', 'Hour'])
    df = df.drop(columns=['Date', 'Local Time'])
    df.rename(columns={'D': 'Demand', 'DF': 'Forecast'}, inplace=True)
    
    return df

def read_all_regions():
    """Read data from all regions"""
    reg_list = ['California', 'Carolinas', 
                'Central', 'Florida', 
                'Mid-Atlantic', 'Midwest', 
                'New England', 'New York', 
                'Northwest', 'Southeast', 
                'Southwest', 'Tennessee', 
                'Texas']
    df = pd.concat((read_region(region) for region in reg_list), keys=reg_list)
    df = df.sort_index()
    
    return df



