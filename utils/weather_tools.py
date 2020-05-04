import pandas as pd
import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
DATA_DIR = os.path.abspath('../state_data')



df_stations_US = pd.read_csv(f"{DATA_DIR}/stations_TZ.csv")

def generate_stationID_dict():
    """Generate a dict of state name : stationID key/value pairs

    stationID is the string "USAF identifier-WBAN identifier"
    In the stations DataFrame:
    USAF = Air Force station ID. May contain a letter in the first position.
    WBAN = NCDC WBAN number
    CTRY = FIPS country ID
    ST = State for US stations
    """

    cols_to_use = ['ST', 'StID']
    df_stations = pd.read_csv(f"{DATA_DIR}/stations_TZ.csv", usecols=cols_to_use)

    states = pd.unique(df_stations.ST.values)

    stations_in_state = {}
    for i in range(len(states)):
        stations_in_state[states[i]] = list(df_stations[df_stations.ST == states[i]].StID.values)
    return stations_in_state

stations_in_state = generate_stationID_dict()

key_list = df_stations_US.StID.values
valu_list = df_stations_US.TZone.values 
TZ_dict = {key_list[i] : valu_list[i] for i in range(len(key_list))} # dictionary containing {Station_ID: Time-Zone}

def process_file(f):
    
    StID = f[-20:-8]
    good = ['1','5', 'A', 'C', 'U'] #temperature quality codes which indicate good data
    df = pd.read_csv(f, names = ['data'], compression='gzip', header=None)
    df_no_summary = df[df.data.str.contains('FM-12')].copy()
    if len(df_no_summary) != 0: #check for files not containing FM-12
        df_no_summary.reset_index(drop = True, inplace = True)
        df_no_summary['Date'] = pd.to_datetime(df_no_summary['data'].str[15:27], utc = True)
        df_no_summary['Date'] = df_no_summary.apply(lambda x: x.Date.astimezone(tz=TZ_dict[StID]), axis = 1) #converting to local timezone
        df_no_summary['T_C'] = df_no_summary.apply(lambda row : int(row.data[87:92])/10 if (row.data[92] in good) else np.nan, axis = 1)    
        df_no_summary.drop(columns = ['data'], inplace = True) 
        return df_no_summary.resample('D', on = 'Date').mean()

    else: #skip file. Does not contain FM-12
        return 0

def all_stations_for_state_year(state, year):
    """Reads data from all stations in a given state and year

    Returns: DataFrame with Date index and separate columns for each station-ID
    
    """

    df_list = []

    files_processed = 0
    not_found = 0
    files_wo_FM12 = 0

    tot_files = len(stations_in_state[state])
    print(f"Files to process: {tot_files}")

    for StID in stations_in_state[state]:
        f = "/Volumes/Nikhil hdd/Weather Data/" + year + "/" + StID + "-" + year + ".gz"
        try:
            xx = process_file(f)
            if type(xx) == pd.core.frame.DataFrame :
                print(f"file: {f}")
                xx.rename(columns = {'T_C':StID}, inplace = True)
                df_list.append(xx)
                files_processed += 1
            else:
                files_wo_FM12 += 1
                

        except (FileNotFoundError, EOFError, pd.errors.ParserError) as e:
            not_found += 1
    
    print(f"files processed: {files_processed}")
    print(f"files without FM-12: {files_wo_FM12}")
    print(f"files not_found: {not_found}")  

    StID_0 = stations_in_state[state][0] #representative station ID from the state

    Date = pd.date_range(str(year) + '-01-01', str(year) + '-12-31', tz = TZ_dict[StID_0])
    df = pd.DataFrame(data=Date, columns=['Date'])     

    for frame in df_list:
        df = df.merge(frame, on='Date', how='left')
    df.set_index('Date', inplace=True)
    df.to_csv("/Volumes/Nikhil hdd/Weather Data/" + f"{state}" +f"_{year}" + ".csv")
    

def all_states_files(year):

    """Generate temeprature files for all the states for given year

    returns files named ST_YYYY.csv """

    states = pd.unique(df_stations_US.ST.values)

    for state in states:
        
        try:
            path = "/Volumes/Nikhil hdd/Weather Data/" + f"{state}" + "_" + year + ".csv"
            if not os.path.isfile(path):
                print(state)
                all_stations_for_state_year(state,year)
        except ValueError:
            print(f'Some issue with {state}')

def states_timeseries():

    """ concatenate files for a state from 2015-2019

    generates time series for each station in the state """

    states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IA', 'ID',
 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN','MO', 'MS', 'MT', 'NC',
 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH','OK', 'OR', 'PA', 'RI', 'SC',
 'SD', 'TN', 'TX', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']

    
    years = range(2015,2020,1)
    

    

    for state in states:
        print(f'Processing {state}')

        df_list = []

        for year in years:

            try:

                filepath = f"/Volumes/Nikhil hdd/Weather Data/{year}_processed/{state}_{year}.csv"
                df = pd.read_csv(filepath)
                df_list.append(df)    

            except FileNotFoundError as e:
                print(e)

        df = pd.concat(df_list, axis = 0)
        # df = df_list[0]
        df['MEAN_T'] = df.mean(axis = 1)*(9/5) + 32
        
        try:
            df.rename(columns = {'Date': 'DATE'}, inplace = True)
            df.set_index('DATE', inplace = True)
        except KeyError as e:
            print(e)
            df.rename(index = {'Date': 'DATE'}, inplace = True)
            
        # df['MEAN_T'].to_csv(f"/Volumes/Nikhil hdd/Weather Data/state_data/{state}_2015-2019_raw.csv") 
        df['MEAN_T'].to_csv(f"/Volumes/Nikhil hdd/Weather Data/state_data/{state}_2015-2019_raw.csv") 
        





file_st = f"{DATA_DIR}" + "/" + "states_info.csv"
df_states = pd.read_csv(file_st, index_col = 0, parse_dates = ['LD'])
regions = df_states.columns[2:-3]
states = df_states.index.values
states_in_region = {}

for region in regions:
    state_list = []
    for state in states:
        if not pd.isna(df_states.loc[state][region]):# state is in the region
            state_list.append(state)
    states_in_region[region] = state_list


def weighted_temperature(region):
    
    """Takes all the states in the region and generates a population weighted temperature for the region.
    
    The output frame contains 
    index : DATE
    columns : T_(statename)
              T_WA
    """
    
    states = states_in_region[region]
    print(f"states to process: {states}")
    
    df_list = []
    for state in states:
        filepath_15to19 = f"{DATA_DIR}" + "/" + state + "_2015-2019_raw.csv"
        filepath_2020 = f"{DATA_DIR}" + "/" + state + "_2020_raw.csv"
        T_till_2019 = pd.read_csv(filepath_15to19, index_col = 'DATE', parse_dates = True) #2015-2019 data
        T_2020 = pd.read_csv(filepath_2020, index_col = 'DATE', parse_dates = True) #2020 data

        T_till_2019.index = T_till_2019.index.map(lambda x: pd.to_datetime(x.strftime('%Y-%m-%d')))
        T_2020.index = T_2020.index.map(lambda x: pd.to_datetime(x.strftime('%Y-%m-%d')))

        T_all = T_till_2019.append(T_2020) # 2015-2020 data
        T_all.dropna(inplace = True) #dropping some missing entries
        T_all.rename(columns = {'MEAN_T': f'{state}'}, inplace = True)
        df_list.append(T_all)
        print(f'processed: {state}')
   
    # df = pd.concat(df_list, axis = 1)

    df = df_list[0]

    for frame in df_list[1:]:
        print(f"merging {frame}" )
        df = df.merge(frame, left_index=True, right_index = True, how='outer')
    
    def weighted_average(row):
        """ Calculates Popoulation weighted Average of Temperature. 
        
        Missing Temperature values ignored while calculating weights"""
        
        wm = 0 # weighted mean
        tot_pop = 0 #Population of states with available Temperature data in a particular row
        
        for col in row.index:
            if not pd.isnull(row[col]): 
                wm += row[col]*df_states.loc[col].Population
                tot_pop += df_states.loc[col].Population
        return wm/tot_pop

    print("now applying weighted average.")

    df['T_WA'] = df.apply(weighted_average, axis = 1)
    
    df.to_csv(f'{DATA_DIR}' + "/" + f'Regional_{region}_2015-2020.csv') # stores it in a file
    return df

for region in ['NW']:
    weighted_temperature(region)



