import pickle

# event2index = {'0': {'index': 0, 'cnt': 9204}, '1': {'index': 1, 'cnt': 7767}, '6': {'index': 6, 'cnt': 2226}, '2': {'index': 2, 'cnt': 4082}, '3': {'index': 3, 'cnt': 3371}, '4': {'index': 4, 'cnt': 2525}, '5': {'index': 5, 'cnt': 1485}}
# with open('train_event.txt', 'r') as fin:
#     for line in fin.readlines():
#         for item in line.strip().split(','):
#             if item not in event2index.keys():
#                 event2index[item] = {'index': int(item), 'cnt': 1}
#             else:
#                 event2index[item]['cnt'] += 1
# with open('event_index.json', 'wb') as fout:
#     pickle.dump(event2index, fout)
#
# event2index = pickle.load(open('event_index.json', 'rb'))
# print(event2index)

statistic = pickle.load(open('statistic.json', 'rb'))
print(statistic['event_type_dict'])
import pandas as pd

# csv_file = './ATM.csv'
# to_csv_file = './ATM_day.csv'
# df = pd.read_csv(csv_file, encoding='utf-8')
# df.loc[:, 'time'] = df.loc[:, 'time'] / (24 * 3600.0)
# df.to_csv(to_csv_file, index=False, encoding='utf-8')