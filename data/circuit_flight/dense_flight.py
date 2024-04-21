import pandas as pd

def check_dense_flight(series):
    diff = series.diff().dt.days
    is_dense = (diff.dropna() == 0).any()

    return is_dense

def get_dense_flights_data(path = './data/data_preprocessed.parquet'):
    data = pd.read_parquet(path)
    data['FLIGHT_ID'] = data['AIRLINE'] + data['FLIGHT_NUMBER'].astype(str)
    data['Date'] = pd.to_datetime(data['Date'])

    dense_flight = data.groupby('FLIGHT_ID')['Date'].apply(check_dense_flight).reset_index()
    dense_flight = dense_flight.loc[dense_flight['Date'] == True]

    dense_flight_ids = dense_flight['FLIGHT_ID'].tolist()
    dense_data = data[data['FLIGHT_ID'].isin(dense_flight_ids)]
    return dense_data