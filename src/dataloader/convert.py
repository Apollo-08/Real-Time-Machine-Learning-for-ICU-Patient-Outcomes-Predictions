import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import write_json, write_pkl, load_json


def convert_timeseries_into_mmap(data_dir, save_dir, n_rows=100000):
    """
    Read CSV file and convert time series data into mmap file.
    """
    save_path = Path(save_dir) / 'ts.dat'
    shape = (n_rows, 24, 34)
    write_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)
    ids = []
    n = 0
    info = {'name': 'ts'}

    for split in ['train', 'val', 'test']:
        print('split: ', split)
        csv_path = Path(data_dir) / split / 'timeseries.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path, dtype=float)
        arr = df.values
        if arr.size % (24 * 35) != 0:
            raise ValueError("The number of elements in the array is not divisible by 24 * 35")

        new = np.reshape(arr, (-1, 24, 35))
        pos_to_id = new[:, 0, 0]
        ids.append(pos_to_id)
        new = new[:, :, 1:]  # Remove patient column

        if new.shape[1:] != (24, 34):
            raise ValueError("Reshaped array does not match expected shape (24, 34)")

        write_file[n: n + len(new), :, :] = new
        info[split + '_len'] = len(new)
        n += len(new)
        del new, arr

    info['total'] = n
    info['shape'] = shape
    info['columns'] = list(df.columns)[1:]
    del df

    ids = np.concatenate(ids)
    assert len(set(ids)) == len(ids)

    write_pkl({pid: pos for pos, pid in enumerate(ids)}, Path(save_dir) / 'id2pos.pkl')
    write_pkl({pos: pid for pos, pid in enumerate(ids)}, Path(save_dir) / 'pos2id.pkl')
    write_json(info, Path(save_dir) / 'ts_info.json')
    print(info)


def convert_into_mmap(data_dir, save_dir, csv_name, n_cols=None, n_rows=100000):
    """
    Read CSV file and convert flat data into mmap file.
    """
    csv_to_cols = {'diagnoses': 357, 'diagnoses_1033': 1034, 'labels': 5, 'flat': 93}
    n_cols = (csv_to_cols[csv_name] - 1) if n_cols is None else n_cols
    shape = (n_rows, n_cols)
    save_path = Path(save_dir) / f'{csv_name}.dat'
    write_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)

    info = {'name': csv_name, 'shape': shape}
    n = 0

    for split in ['train', 'val', 'test']:
        print('split: ', split)
        csv_path = Path(data_dir) / split / f'{csv_name}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.replace('Unknown', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(0, inplace=True)

        arr = df.values[:, 1:]  # Remove patient column
        if not np.isfinite(arr).all():
            raise ValueError("Non-numeric or infinite values detected in the array after conversion.")
        if arr.shape[1] != write_file.shape[1]:
            raise ValueError(f"Shape mismatch: arr has {arr.shape[1]} columns, expected {write_file.shape[1]} columns")

        arr_len = len(arr)
        write_file[n: n + arr_len, :] = arr
        info[split + '_len'] = arr_len
        n += arr_len
        del arr

    info['total'] = n
    info['columns'] = list(df.columns)[1:]
    write_json(info, Path(save_dir) / f'{csv_name}_info.json')
    print(info)


def read_mm(datadir, name):
    """
    Read mmap file given its name.
    """
    info = load_json(Path(datadir) / (name + '_info.json'))
    dat_path = Path(datadir) / (name + '.dat')
    data = np.memmap(dat_path, dtype=np.float32, shape=tuple(info['shape']))
    return data, info


if __name__ == '__main__':
    paths = load_json('paths.json')
    data_dir = paths['eICU_path']
    save_dir = paths['data_dir']
    print(f'Load eICU processed data from {data_dir}')
    print(f'Saving mmap data in {save_dir}')
    print('--' * 30)
    Path(save_dir).mkdir(exist_ok=True)
    print('** Converting time series **')
    convert_timeseries_into_mmap(data_dir, save_dir)
    for csv_name in ['flat', 'diagnoses', 'labels']:
        print(f'** Converting {csv_name} **')
        convert_into_mmap(data_dir, save_dir, csv_name)
    print('--' * 30)
    print(f'Done! Saved data in {save_dir}')
