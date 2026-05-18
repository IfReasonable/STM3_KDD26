import torch
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def get_normalizer(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return scaler

def split_data_by_days(data, val_days, test_days, step_per_day=24):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param step_per_day: number of steps per day (e.g., 24 for hourly data)
    :return:
    '''
    T = int(step_per_day)
    x=-T * int(test_days)
    test_data = data[-T*int(test_days):]
    val_data = data[-T*int(test_days + val_days): -T*int(test_days)]
    train_data = data[:-T*int(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def split_data_for_ts_data(data, val_ratio, test_ratio, seq_len):
    data_len = data.shape[0]
    train_ratio = 1 - val_ratio - test_ratio
    if train_ratio < 0:
        raise ValueError("train_ratio < 0, please check val_ratio and test_ratio")
    
    num_train = int(data_len * train_ratio)
    num_test = int(data_len * test_ratio)
    
    num_vali = data_len - num_train - num_test
    border1s = [0, num_train - seq_len, data_len - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, data_len]
    
    train_data = data[border1s[0]:border2s[0]]
    val_data = data[border1s[1]:border2s[1]]
    test_data = data[border1s[2]:border2s[2]]
    return train_data, val_data, test_data

def split_data_for_ETT_hour(data, seq_len):
    '''
    :param data: [B, *]
    :param seq_len: input sequence length
    :return:
    '''
    border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    
    train_data = data[border1s[0]:border2s[0]]
    val_data = data[border1s[1]:border2s[1]]
    test_data = data[border1s[2]:border2s[2]]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
    data = TensorDataset(X, Y)
    dataloader = DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last,
                                             pin_memory=True, num_workers=8)
    return dataloader


class STWindowDataset(Dataset):
    def __init__(self, x_data, y_data, time_in_day, day_in_week, lag, horizon, input_dim):
        self.x_data = x_data
        self.y_data = y_data
        self.time_in_day = time_in_day
        self.day_in_week = day_in_week
        self.lag = lag
        self.horizon = horizon
        self.input_dim = input_dim

        self.T, self.N, _ = self.x_data.shape
        assert self.T == self.y_data.shape[0]
        assert self.T >= lag + horizon

        self.combined_x_features = torch.utils.data._utils.collate.default_collate([
            torch.from_numpy(np.concatenate([
                self.x_data, 
                self.time_in_day, 
                self.day_in_week
            ], axis=-1)).float()
        ])[0]

    def __len__(self):
        return max(0, self.T - self.lag - self.horizon + 1)

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.lag
        x_window = self.combined_x_features[x_start:x_end, :, :]

        y_start = idx + self.lag
        y_end = idx + self.lag + self.horizon
        y_window = self.y_data[y_start:y_end, :, :self.input_dim]

        x_tensor = torch.FloatTensor(x_window)
        y_tensor = torch.FloatTensor(y_window)

        return x_tensor, y_tensor

def get_dataloader(args, normalizer='std', tod=True, dow=True, weather=False, single=True):
    data = load_st_dataset(args.dataset)

    T, N, F = data.shape

    time_ind = [i % args.steps_per_day / args.steps_per_day for i in range(T)]
    time_ind = np.array(time_ind).reshape(-1, 1, 1)
    time_in_day = np.tile(time_ind, (1, N, 1))

    day_in_week = [(i // args.steps_per_day) % args.days_per_week for i in range(T)]
    day_in_week = np.array(day_in_week).reshape(-1, 1, 1)
    day_in_week = np.tile(day_in_week, (1, N, 1))

    combined_raw = data

    if args.dataset == "ETTh1":
        train_data_raw, _, _ = split_data_for_ETT_hour(combined_raw, args.lag)
    elif args.dataset == "Electricity":
        train_data_raw, _, _ = split_data_for_ts_data(combined_raw, args.val_ratio, args.test_ratio, args.lag)
    else:
        if args.test_ratio > 1:
            train_data_raw, _, _ = split_data_by_days(combined_raw, args.val_ratio, args.test_ratio, args.steps_per_day)
        else:
            train_data_raw, _, _ = split_data_by_ratio(combined_raw, args.val_ratio, args.test_ratio)

    scaler = get_normalizer(train_data_raw[..., :args.input_dim], normalizer, args.column_wise)

    normalized_data = combined_raw.copy()
    normalized_data[..., :args.input_dim] = scaler.transform(normalized_data[..., :args.input_dim])

    if args.dataset == "ETTh1":
        x_train_raw, x_val_raw, x_test_raw = split_data_for_ETT_hour(normalized_data, args.lag)
        y_train_raw, y_val_raw, y_test_raw = split_data_for_ETT_hour(normalized_data, args.lag)
        time_in_day_train, time_in_day_val, time_in_day_test = split_data_for_ETT_hour(time_in_day, args.lag)
        day_in_week_train, day_in_week_val, day_in_week_test = split_data_for_ETT_hour(day_in_week, args.lag)
    elif args.dataset == "Electricity":
        x_train_raw, x_val_raw, x_test_raw = split_data_for_ts_data(normalized_data, args.val_ratio, args.test_ratio, args.lag)
        y_train_raw, y_val_raw, y_test_raw = split_data_for_ts_data(normalized_data, args.val_ratio, args.test_ratio, args.lag)
        time_in_day_train, time_in_day_val, time_in_day_test = split_data_for_ts_data(time_in_day, args.val_ratio, args.test_ratio, args.lag)
        day_in_week_train, day_in_week_val, day_in_week_test = split_data_for_ts_data(day_in_week, args.val_ratio, args.test_ratio, args.lag)
    else:
        if args.test_ratio > 1:
            x_train_raw, x_val_raw, x_test_raw = split_data_by_days(normalized_data, args.val_ratio, args.test_ratio, args.steps_per_day)
            y_train_raw, y_val_raw, y_test_raw = split_data_by_days(normalized_data, args.val_ratio, args.test_ratio, args.steps_per_day)
            time_in_day_train, time_in_day_val, time_in_day_test = split_data_by_days(time_in_day, args.val_ratio, args.test_ratio, args.steps_per_day)
            day_in_week_train, day_in_week_val, day_in_week_test = split_data_by_days(day_in_week, args.val_ratio, args.test_ratio, args.steps_per_day)
        else:
            x_train_raw, x_val_raw, x_test_raw = split_data_by_ratio(normalized_data, args.val_ratio, args.test_ratio)
            y_train_raw, y_val_raw, y_test_raw = split_data_by_ratio(normalized_data, args.val_ratio, args.test_ratio)
            time_in_day_train, time_in_day_val, time_in_day_test = split_data_by_ratio(time_in_day, args.val_ratio, args.test_ratio)
            day_in_week_train, day_in_week_val, day_in_week_test = split_data_by_ratio(day_in_week, args.val_ratio, args.test_ratio)

    print(f"Train: x: {x_train_raw.shape}, y: {y_train_raw.shape}, Val: x: {x_val_raw.shape}, y: {y_val_raw.shape}, Test: x: {x_test_raw.shape}, y: {y_test_raw.shape}")

    train_dataset = STWindowDataset(
        x_data=x_train_raw,
        y_data=y_train_raw,
        time_in_day=time_in_day_train,
        day_in_week=day_in_week_train,
        lag=args.lag,
        horizon=args.horizon,
        input_dim=args.input_dim
    )

    val_dataset = STWindowDataset(
        x_data=x_val_raw,
        y_data=y_val_raw,
        time_in_day=time_in_day_val,
        day_in_week=day_in_week_val,
        lag=args.lag,
        horizon=args.horizon,
        input_dim=args.input_dim
    ) if len(x_val_raw) > 0 else None

    test_dataset = STWindowDataset(
        x_data=x_test_raw,
        y_data=y_test_raw,
        time_in_day=time_in_day_test,
        day_in_week=day_in_week_test,
        lag=args.lag,
        horizon=args.horizon,
        input_dim=args.input_dim
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=4
    ) if val_dataset is not None else None

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=4
    )

    return train_loader, val_loader, test_loader, scaler
