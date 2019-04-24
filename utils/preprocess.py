import pandas as pd
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt


"""
    预处理器初始化的时候只需要一个.csv格式的文件
    然后数据按照seq_id, timestamp, event_type分布,event_type可以由多列共同决定
"""
def dataset_statistic(csv_file, domain_dict, dataset_name, dataset_dir):
    dataset = pd.read_csv(csv_file, encoding='utf-8').sort_values(by=[domain_dict['id'], domain_dict['timestamp']],
                                                     ascending=[True, True]).reset_index(drop=True)
    last_id = None
    last_timestamp = None
    sequences_length = {}
    event_interval = []
    event_length = []
    event_type_dict = {}
    length = 0
    for index, row in dataset.iterrows():
        id = row[domain_dict['id']]
        timestamp = row[domain_dict['timestamp']]
        event_type = row[domain_dict['event']]
        if last_id is None or last_id != id:
            last_id = id
            if length != 0:
                sequences_length[length] = 1 if length not in sequences_length.keys() \
                    else sequences_length[length] + 1
                event_length.append(length)
            length = 1
            last_timestamp = timestamp
        else:
            if timestamp - last_timestamp > 0.01:
                event_interval.append(timestamp - last_timestamp)
            last_timestamp = timestamp
            length += 1
        if event_type not in event_type_dict.keys():
            event_type_dict[event_type] = 1
        else:
            event_type_dict[event_type] += 1
    sequences_length[length] = 1 if length not in sequences_length.keys() else sequences_length[length] + 1
    event_interval = np.log10(np.array(event_interval))
    plt.hist(event_interval, bins=80, facecolor='green', edgecolor='black', alpha=0.7)
    plt.xlabel('log event interval')
    plt.ylabel('number')
    # plt.grid(True)
    plt.savefig(dataset_dir + '/event_interval_statistic.png')

    event_length = np.log10(np.array(event_length))
    plt.hist(event_length, bins=20, facecolor='green', edgecolor='black', alpha=0.7)
    plt.xlabel('sequence length')
    plt.ylabel('number')
    plt.savefig(dataset_dir + '/sequence_length_statistic.png')

    statistic_dict = {'event_type_dict': event_type_dict, 'event_interval': event_interval,
                      'sequences_length': sequences_length}
    with open(dataset_dir + '/statistic.json', 'wb') as fout:
        pickle.dump(statistic_dict, fout)
    print('dataset ' + dataset_name + ' statistic:')
    print('event_type_statistic:\n', event_type_dict)
    print('number of event type:\t', len(event_type_dict))
    print('sequences_length_statistic:\n', sequences_length)


def generate_time_sequence(csv_file, domain_dict, time_sequences_file, train_time_sequences_file,
                           dev_time_sequences_file, train_rate=0.7, min_length=3, max_length=50):
    dataset = pd.read_csv(csv_file).sort_values(by=[domain_dict['id'], domain_dict['timestamp']],
                                                ascending=[True, True]).reset_index(drop=True)
    last_id = None
    time_sequence = None
    time_sequences = []
    with open(time_sequences_file, 'w') as time_sequences_fout:
        for index, row in dataset.iterrows():
            id = str(row[domain_dict['id']])
            timestamp = str(row[domain_dict['timestamp']])
            if last_id is None or last_id != id:
                if time_sequence is not None and min_length <= len(time_sequence):
                    for ed in range(min_length - 1, len(time_sequence)):
                        time_sequences.append(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
                        time_sequences_fout.write(
                            ','.join(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]) + '\n')
                last_id = id
                time_sequence = [timestamp]
            else:
                time_sequence.append(timestamp)
        if time_sequence is not None and min_length <= len(time_sequence):
            for ed in range(min_length - 1, len(time_sequence)):
                time_sequences.append(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
                time_sequences_fout.write(
                    ','.join(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]) + '\n')
    n_train_data = int(train_rate * len(time_sequences))
    with open(train_time_sequences_file, 'w') as train_time_fout:
        train_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[:n_train_data]])
    with open(dev_time_sequences_file, 'w') as dev_time_fout:
        dev_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[n_train_data:]])
    print('time_sequences[0]:', time_sequences[0])
    print('time_sequences size:', len(time_sequences))


def generate_event_sequence(csv_file, domain_dict, event_sequences_file, train_event_sequences_file,
                            dev_event_sequences_file, event_index_file, train_rate=0.7, min_length=3, max_length=50):
    dataset = pd.read_csv(csv_file).sort_values(by=[domain_dict['id'], domain_dict['timestamp']],
                                                ascending=[True, True]).reset_index(drop=True)
    last_id = None
    event_sequence = None
    event_sequences = []
    event2index = {}
    event_index = 0
    with open(event_sequences_file, 'w') as event_sequences_fout:
        for index, row in dataset.iterrows():
            id = str(row[domain_dict['id']])
            event = str(row[domain_dict['event']])
            if event not in event2index.keys():
                event2index[event] = {'index': event_index, 'cnt': 0}
                event_index += 1
            if last_id is None or last_id != id:
                if event_sequence is not None and min_length <= len(event_sequence):
                    for ed in range(min_length - 1, len(event_sequence)):
                        event2index[event_sequence[ed]]['cnt'] += 1
                        event_sequences.append(event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
                        event_sequences_fout.write(
                            ','.join(list(map(str, event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]))) + '\n')
                last_id = id
                event_sequence = [event]
            else:
                event_sequence.append(event)
        if event_sequence is not None and min_length <= len(event_sequence):
            for ed in range(min_length - 1, len(event_sequence)):
                event2index[event_sequence[ed]]['cnt'] += 1
                event_sequences.append(event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
                event_sequences_fout.write(
                    ','.join(list(map(str, event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]))) + '\n')

    with open(event_index_file, 'wb') as event_index_fout:
        pickle.dump(event2index, event_index_fout)

    n_train_data = int(train_rate * len(event_sequences))
    with open(train_event_sequences_file, 'w') as train_event_fout:
        train_event_fout.writelines([','.join(event_sequence) + '\n' for event_sequence in event_sequences[:n_train_data]])
    with open(dev_event_sequences_file, 'w') as dev_event_fout:
        dev_event_fout.writelines([','.join(event_sequence) + '\n' for event_sequence in event_sequences[n_train_data:]])
    print('event_sequences[0]:', event_sequences[0])
    print('event_sequences size:', len(event_sequences))
    print('event2index:', event2index)


def generate_sequence(csv_file, domain_dict,
                      time_sequences_file, train_time_sequences_file, dev_time_sequences_file,
                      event_sequences_file, train_event_sequences_file, dev_event_sequences_file,
                      event_index_file, train_rate=0.7, min_length=5, max_length=10, min_event_interval=0.01):
    dataset = pd.read_csv(csv_file).sort_values(by=[domain_dict['id'], domain_dict['timestamp']],
                                                ascending=[True, True]).reset_index(drop=True)
    last_id = None
    time_sequence = None
    time_sequences = []
    event_sequence = None
    event_sequences = []
    event2index = {}
    event_index = 0
    time_sequences_fout = open(time_sequences_file, 'w')
    event_sequences_fout = open(event_sequences_file, 'w')
    for index, row in dataset.iterrows():
        id = str(row[domain_dict['id']])
        timestamp = str(row[domain_dict['timestamp']])
        event = str(row[domain_dict['event']])
        if event not in event2index.keys():
            event2index[event] = {'index': event_index, 'cnt': 0}
            event_index += 1
        if last_id is None or last_id != id:
            if event_sequence is not None and min_length <= len(event_sequence):
                for ed in range(min_length - 1, len(event_sequence)):
                    if float(time_sequence[ed]) - float(time_sequence[ed - 1]) < min_event_interval:
                        continue
                    time_sequences.append(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
                    time_sequences_fout.write(
                        ','.join(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]) + '\n')
                    event2index[event_sequence[ed]]['cnt'] += 1
                    event_sequences.append(event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
                    event_sequences_fout.write(
                        ','.join(list(map(str, event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]))) + '\n')
            last_id = id
            time_sequence = [timestamp]
            event_sequence = [event]
        else:
            time_sequence.append(timestamp)
            event_sequence.append(event)
    if event_sequence is not None and min_length <= len(event_sequence):
        for ed in range(min_length - 1, len(event_sequence)):
            if float(time_sequence[ed]) - float(time_sequence[ed - 1]) < min_event_interval:
                continue
            time_sequences.append(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
            time_sequences_fout.write(
                ','.join(time_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]) + '\n')
            event2index[event_sequence[ed]]['cnt'] += 1
            event_sequences.append(event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed])
            event_sequences_fout.write(
                ','.join(list(map(str, event_sequence[max(ed - min_length + 1, ed - max_length + 1):ed]))) + '\n')
    time_sequences_fout.close()
    event_sequences_fout.close()

    with open(event_index_file, 'wb') as event_index_fout:
        pickle.dump(event2index, event_index_fout)

    n_train_data = int(train_rate * len(event_sequences))
    with open(train_time_sequences_file, 'w') as train_time_fout:
        train_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[:n_train_data]])
    with open(dev_time_sequences_file, 'w') as dev_time_fout:
        dev_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[n_train_data:]])
    print('time_sequences[0]:', time_sequences[0])
    print('time_sequences size:', len(time_sequences))
    with open(train_event_sequences_file, 'w') as train_event_fout:
        train_event_fout.writelines(
            [','.join(event_sequence) + '\n' for event_sequence in event_sequences[:n_train_data]])
    with open(dev_event_sequences_file, 'w') as dev_event_fout:
        dev_event_fout.writelines(
            [','.join(event_sequence) + '\n' for event_sequence in event_sequences[n_train_data:]])
    print('event_sequences[0]:', event_sequences[0])
    print('event_sequences size:', len(event_sequences))
    print('event2index:', event2index)


def load_sequences(time_sequences_file, event_sequences_file):
    time_sequences_fin = open(time_sequences_file, 'r', encoding='utf-8')
    event_sequences_fin = open(event_sequences_file, 'r', encoding='utf-8')
    time_sequences = [line.strip().split(',') for line in time_sequences_fin.readlines()]
    event_sequences = [line.strip().split(',') for line in event_sequences_fin.readlines()]
    time_sequences_fin.close()
    event_sequences_fin.close()
    sequences = []
    for item in zip(time_sequences, event_sequences):
        sequences.append([[float(value[0]), int(value[1])] for value in zip(item[0], item[1])])
    return sequences


if __name__ == '__main__':
    domain_dict = {'id': 'id', 'timestamp': 'time', 'event': 'event'}
    dataset_statistic('../data/real/linkedin/Linkedin.csv', domain_dict, 'linkedin', '../data/real/linkedin')
    generate_time_sequence('../data/real/linkedin/Linkedin.csv', domain_dict,
                           '../data/real/linkedin/time.txt', '../data/real/linkedin/train_time.txt',
                           '../data/real/linkedin/dev_time.txt', train_rate=0.8)
    generate_event_sequence('../data/real/linkedin/Linkedin.csv', domain_dict,
                            '../data/real/linkedin/event.txt', '../data/real/linkedin/train_event.txt',
                            '../data/real/linkedin/dev_event.txt', '../data/real/linkedin/event_index.json', train_rate=0.8)
    sequences = load_sequences('../data/real/linkedin/time.txt', '../data/real/linkedin/event.txt')
    print(sequences[0])
    print(len(sequences))
    statistic_dict = pickle.load(open('../data/real/linkedin/linkedin_statistic.json', 'rb'))
    print(len(statistic_dict['event_type_dict']))
