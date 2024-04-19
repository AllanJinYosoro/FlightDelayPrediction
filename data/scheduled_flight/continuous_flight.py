import pandas as pd

def check_continuity(series):
    '''
    传入的series只包含date列。
    if date_i = date_{i-1} + 1 then flag = 1 else flag = 0
    if sum(flag) = len return True else False
    '''
    diff = series.diff().dt.days
    is_continuous = (diff.dropna() == 1).all()

    return is_continuous

def get_continuous_flights_data():
    data = pd.read_parquet('./data/data_preprocessed.parquet')
    data['FLIGHT_ID'] = data['AIRLINE'] + data['FLIGHT_NUMBER'].astype(str)
    data['Date'] = pd.to_datetime(data['Date'])

    continuous_flight = data.groupby('FLIGHT_ID')['Date'].apply(check_continuity).reset_index()
    continuous_flight = continuous_flight.loc[continuous_flight['Date'] == True]
    #print(continuous_flight.head())
    #continuous_flight.to_csv('continuous_flight.csv')

    continuous_flight_ids = continuous_flight['FLIGHT_ID'].tolist()
    continuous_data = data[data['FLIGHT_ID'].isin(continuous_flight_ids)]
    return continuous_data