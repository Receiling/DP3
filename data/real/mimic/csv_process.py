import pandas as pd

csv_file = './mimic.csv'
to_csv_file = './mimic_day.csv'
df = pd.read_csv(csv_file)
df.loc[:, 'time'] = df.loc[:, 'time'] / (24 * 3600.0)
df.to_csv(to_csv_file, index=False)