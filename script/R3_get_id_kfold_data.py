import argparse
import glob
import pandas as pd


def read_csv(path):
    data = pd.read_csv(path, sep='\t')
    return data


def remove_row_nan(df):
    df = df.dropna(axis=0)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold_data', type=str,
                        default='../data/cross_validation_data', help='data path')

    opt = parser.parse_args()

    kfold_path = glob.glob(opt.kfold_data + '/*')

    for path in kfold_path:
        train = read_csv(path + '/train.csv')
        train = remove_row_nan(train)
        train['id'].to_csv(path + '/train_id.csv', index=False)

        val = read_csv(path + '/test.csv')
        val = remove_row_nan(val)
        val['id'].to_csv(path+'/val_id.csv', index=False)
