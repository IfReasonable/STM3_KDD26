import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    if dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'METR_LA':
        data_path = os.path.join('./data/METR_LA/metr_la.npz')
        data = np.load(data_path)['data']
    elif dataset == 'Milan_sms':
        data_path = os.path.join('./data/Milan/milan.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'Milan_call':
        data_path = os.path.join('./data/Milan/milan.npz')
        data = np.load(data_path)['data'][:, :, 1]
    elif dataset == 'Milan_internet':
        data_path = os.path.join('./data/Milan/milan.npz')
        data = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'KnowAir':
        data_path = os.path.join('./data/KnowAir/KnowAir_all.npy')
        data = np.load(data_path)
    elif dataset == 'NREL':
        data_path = os.path.join('./data/NREL/X_selected.npy')
        data = np.load(data_path)
    elif dataset == 'ETTh1':
        data_path = os.path.join('./data/ETT-small/ETTh1.csv')
        df_raw = pd.read_csv(data_path)
        cols_data = df_raw.columns[1:]
        data = df_raw[cols_data].values
    elif dataset == 'Electricity':
        data_path = os.path.join('./data/electricity/electricity.csv')
        df_raw = pd.read_csv(data_path)
        cols = list(df_raw.columns)
        cols.remove('OT')
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + ['OT']]
        cols_data = df_raw.columns[1:]
        data = df_raw[cols_data].values
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
