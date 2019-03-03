import scipy
from scipy import io
import time


def mat2csv(mat_file, csv_file):
    data = scipy.io.loadmat(mat_file)
    with open(csv_file, 'w') as fout:
        fout.write('id,time,event\n')
        sequence_id = 1
        print("starting...")
        for k, v in data.items():
            if 'patient' in k:
                for idx in range(data[k].shape[1]):
                    s = str(sequence_id) + ','
                    if len(data[k][1][idx].tolist()) != 0:
                        st = str(data[k][1][idx].tolist()[0])
                        st = '1' + st[1:]
                        s += (str(time.mktime(time.strptime(st, "%Y-%m-%d %H:%M:%S"))) + ',')
                    else:
                        s += 'N,'
                    if len(data[k][0][idx].tolist()) != 0:
                        s += (str(data[k][0][idx].tolist()[0]) + '\n')
                    else:
                        s += '8\n'
                    fout.write(s)
                sequence_id += 1
                if sequence_id % 1000 == 0:
                    print(sequence_id)


if __name__ == '__main__':
    mat_file = 'C:/Users/wyj/Desktop/ICU_data/train.mat'
    csv_file = './mimic.csv'
    mat2csv(mat_file, csv_file)
