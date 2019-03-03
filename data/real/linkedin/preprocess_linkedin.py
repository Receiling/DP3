import pandas as pd
from typing import Dict


def preprocess(data, sequence_index: str, domain: Dict):
    """
    from csv to txt, generate file for each field
    """
    for field_name, file_name in domain.items():
        series = data.groupby(sequence_index)[field_name].apply(lambda x: x.tolist())
        series_data = []
        for idx in series.index:
            series_data.append(','.join(str(value) for value in series[idx]) + '\n')
        train_data_num = int(0.7 * len(series_data))
        with open(file_name + "-train.txt", 'w') as fout:
            fout.writelines(series_data[:train_data_num])
        with open(file_name + "-test.txt", 'w') as fout:
            fout.writelines(series_data[train_data_num:])


input_file = "Linkedin.csv"
data = pd.read_csv(input_file)
domain = {"time": "time", "event": "event", "option1": "option1"}
preprocess(data, "id", domain)

# time series
time_series = data.groupby("id").time.apply(lambda x: x.tolist())
with open("time.txt", 'w') as fout_time:
    for idx in time_series.index:
        fout_time.write(','.join(str(time) for time in time_series[idx]) + '\n')

# event series
event_series = data.groupby("id").event.apply(lambda x: x.tolist())
with open("event.txt", 'w') as fout_event:
    for idx in event_series.index:
        fout_event.write(','.join(str(event) for event in event_series[idx]) + '\n')

# option1 series
option1_series = data.groupby("id").option1.apply(lambda x: x.tolist())
with open("option1.txt", 'w') as fout_option1:
    for idx in option1_series.index:
        fout_option1.write(','.join(str(option1) for option1 in option1_series[idx]) + '\n')


