import torch
import random
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler

def set_random_seed(seed=777):
    '''
    Fix random seed for reproducibility of results.
    Args:
        seed: int, random seed, default=777
    Returns:
        None
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 딥러닝 재현성을 위해 추가 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def hyperparameter_setting(**kwargs):
    """
    <Outlier Detection for Time Series with Recurrent Autoencoder Ensembles>
    [4.1 Experimental Setup - Hyperparameters Settings]
    For all deep learning based methods, we use Adadelta [Zeiler, 2012] as the optimizer, 
    and we set their learning rates to 10e-3.
    ...
    we set the number of hidden LSTM units to 8;
    we set the default number of autoencoders N to
    40, and we also study the effect of varying N from 10 to 40;
    and we set λ to 0.005.
    We randomly vary the skip connection jump step size L from 1 to 10.
    ...
    For MP*, we set the pattern size to 10.
    * I think MP is Markov Process, and the pattern size is the state space of MP.

    Args:
        kwargs: dict, hyperparameters
            N: int, default=10
            input_size: int, default=1
            hidden_size: int, default=8
            output_size: int, default=1
            num_layers: int, default=1
            limit_skip_steps: int, default=10
            learning_rate: float, default=1e-3
            l1_lambda: float, default=0.005
            batch_size: int, default=32
            window_size: int, default=10
            num_epochs: int, default=100
    Returns:
        args: dict, hyperparameters
    """
    args = {}
    args['N'] = kwargs['N'] if 'N' in kwargs else 10 # 앙상블 모델 수, 10~40
    args['input_size'] = kwargs['input_size'] if 'input_size' in kwargs else 1  # 단일 시계열
    args['hidden_size'] = kwargs['hidden_size'] if 'hidden_size' in kwargs else 8
    args['output_size'] = kwargs['output_size'] if 'output_size' in kwargs else 1
    args['num_layers'] = kwargs['num_layers'] if 'num_layers' in kwargs else 1 # No mention in the paper
    args['limit_skip_steps'] = kwargs['limit_skip_steps'] if 'limit_skip_steps' in kwargs else 10 # L: 1~10 랜덤
    args['learning_rate'] = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1e-3
    args['l1_lambda'] = kwargs['l1_lambda'] if 'l1_lambda' in kwargs else 0.005
    args['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 256
    args['window_size'] = kwargs['window_size'] if 'window_size' in kwargs else 288
    args['num_epochs'] = kwargs['num_epochs'] if 'num_epochs' in kwargs else 20
    args['random_seed'] = kwargs['random_seed'] if 'random_seed' in kwargs else 777

    return args

def read_dataset(_file_name, _normalize=True):
    '''
    <Outlier Detection for Time Series with Recurrent Autoencoder Ensembles>
    [4.1 Experimental Setup - Data Sets]
    For both repositories (NAB & ECG), ground truth labels of outlier observations are available.
    However, consistent with the unsupervised setting, we do not use these labels for training,
    but only use them for evaluating accuracy.
    '''
    with open('./NAB/labels/combined_windows.json') as data_file:
        json_label = json.load(data_file)
    abnormal = pd.read_csv(_file_name, header=0, index_col=0)
    abnormal['label'] = 1
    relative_path = os.path.relpath(_file_name, './NAB/data')
    relative_path = relative_path.replace(os.sep, '/')  # 경로 구분자를 '/'로 통일
    
    print(f"Processing file: {relative_path}")  # 현재 처리 중인 파일명 출력
    
    list_windows = json_label.get(relative_path)
    for window in list_windows:
        start = window[0]
        end = window[1]
        abnormal.loc[start:end, 'label'] = -1

    abnormal_data = abnormal['value'].values # as_matrix() no longer works.
    # abnormal_preprocessing_data = np.reshape(abnormal_preprocessing_data, (abnormal_preprocessing_data.shape[0], 1))
    abnormal_label = abnormal['label'].values # as_matrix() no longer works.

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    # Normal = 1, Abnormal = -1
    return abnormal_data, abnormal_label

if __name__ == '__main__':
    set_random_seed()
    args = hyperparameter_setting()
    print(args)