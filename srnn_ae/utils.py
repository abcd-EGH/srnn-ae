import torch
import random
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import datetime

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, window_size, stride=1):
        """
        시계열 데이터를 윈도우 단위로 분할하여 Dataset을 구성합니다.

        Args:
            data (np.ndarray): 시계열 데이터, shape (num_samples, input_size)
            labels (np.ndarray): 시계열 데이터 라벨, shape (num_samples,)
            window_size (int): 윈도우 크기
            stride (int): 윈도우 이동 간격
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.num_samples = (len(data) - window_size) // stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        sample = self.data[start:end]
        target = self.data[start:end]  # 오토인코더의 경우 입력과 출력이 동일
        window_labels = self.labels[start:end]
        return (
            torch.tensor(sample, dtype=torch.float32), 
            torch.tensor(target, dtype=torch.float32), 
            torch.tensor(window_labels, dtype=torch.float32)
        )

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
    args['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 18 # No mention in the paper
    args['window_size'] = kwargs['window_size'] if 'window_size' in kwargs else 288
    args['num_epochs'] = kwargs['num_epochs'] if 'num_epochs' in kwargs else 20 # No mention in the paper
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

def data_plot_with_anomalies(train_data_dir = './NAB/data/artificialNoAnomaly', anomaly_label_dir = './NAB/labels/combined_windows.json', max_plot_num=10):
    # 디렉토리 내의 모든 CSV 파일 불러오기
    train_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
    
    # 이상치 라벨 JSON 파일 로드
    with open(anomaly_label_dir, 'r') as f:
        anomaly_labels = json.load(f)
    
    if len(train_files) > max_plot_num:
        train_files = train_files[:max_plot_num]
    
    # 각 파일에 대해 데이터 로드 및 시각화
    for train_file in train_files:
        train_data_file = os.path.join(train_data_dir, train_file)
        df_train = pd.read_csv(train_data_file)
    
        # 타임스탬프를 datetime 형식으로 변환
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_train.set_index('timestamp', inplace=True)
    
        # 해당 파일의 이상치 라벨 가져오기
        # anomaly_labels의 키와 현재 파일 이름을 정확히 매칭
        # 파일의 상위 디렉토리 이름을 포함하여 상대 경로 생성
        dataset_name = os.path.basename(train_data_dir)  # 'realAWSCloudwatch'
        relative_path = f'{dataset_name}/{train_file}'
    
        anomaly_windows = anomaly_labels.get(relative_path, [])
    
        # 학습 데이터 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train['value'], label=train_file)
    
        # 이상치 구간에 대해 영역 표시
        for idx, window in enumerate(anomaly_windows):
            start_time = pd.to_datetime(window[0])
            end_time = pd.to_datetime(window[1])
            plt.axvspan(start_time, end_time, color='red', alpha=0.3, label='Anomaly' if idx == 0 else "")
    
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title(f'Data Plot with Anomalies - {train_file}')
        plt.legend()
        plt.grid(True)
        plt.show()

def train(model, dataloader, criterion, optimizer, device, l1_lambda, num_epochs, h):
    """
    For training model.

    Args:
        model (nn.Module): 학습할 모델
        dataloader (DataLoader): 학습 데이터 로더
        criterion (nn.Module): 손실 함수
        optimizer (optim.Optimizer): 옵티마이저
        device (torch.device): 학습 디바이스
        l1_lambda (float): L1 정규화 계수
        num_epochs (int): 학습 에포크 수

    Returns:
        None
    """
    
    model.train()  # 학습 모드 설정
    epoch_losses = []  # 에포크별 손실을 저장할 리스트
    
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_start_time = time.time()  # 에포크 시작 시간
        
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(device)  # (batch_size, window_size, input_size)
            targets = targets.to(device)  # (batch_size, window_size, input_size)

            # 입력 데이터를 모델의 예상 입력 형태로 변환
            # expected shape of input of model: (seq_len, batch_size, input_size)
            inputs = inputs.permute(1, 0, 2)  # (window_size, batch_size, input_size)
            targets = targets.permute(1, 0, 2)  # (window_size, batch_size, input_size)

            optimizer.zero_grad()

            # 모델의 forward 함수는 입력과 타겟 시퀀스를 필요로 합니다.
            outputs = model(inputs, targets)  # (seq_len, batch_size, output_size)

            # 손실 계산 (각 타임스텝별로 계산)
            loss = criterion(outputs, targets)

            # L1 정규화 추가
            l1_norm = sum(torch.norm(p, 1) for p in model.parameters())
            loss += l1_lambda * l1_norm

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_end_time = time.time()  # 에포크 종료 시간
        duration = epoch_end_time - epoch_start_time  # 에포크 소요 시간
        avg_loss = epoch_loss / len(dataloader)  # 에포크 평균 손실
        epoch_losses.append(avg_loss)
        
        # 현재 시간 (시:분:초)
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        
        # 남은 에포크 수
        remaining_epochs = num_epochs - epoch
        
        # 예상 학습 종료 시간 계산
        estimated_end_time = datetime.datetime.now() + datetime.timedelta(seconds=duration * remaining_epochs)
        estimated_end_time_str = estimated_end_time.strftime('%H:%M:%S')
        
        if num_epochs <= 20:
            # 에포크 수가 20 이하일 때 모든 에포크 출력
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}, "
                  f"(Duration: {duration:.2f}s, Current Time: {current_time}, "
                  f"Estimated End Time: {estimated_end_time_str})")
        elif num_epochs <= 100:
            # 에포크 수가 100 이하일 때 10 에포크마다 출력
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}, "
                      f"(Duration: {duration:.2f}s, Current Time: {current_time}, "
                      f"Estimated End Time: {estimated_end_time_str})")
        else:
            # 에포크 수가 100 초과일 때 50 에포크마다 출력
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}, "
                      f"(Duration: {duration:.2f}s, Current Time: {current_time}, "
                      f"Estimated End Time: {estimated_end_time_str})")

    end_time = time.time()
    print("Training Complete.")

    # 손실 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프를 파일로 저장
    os.makedirs(h, exist_ok=True)
    plot_path = os.path.join(h, 'training_loss.png')
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")

    loss_df = pd.DataFrame({'epoch': range(1, num_epochs + 1), 'loss': epoch_losses})
    loss_path = os.path.join(h, 'training_loss.csv')
    loss_df.to_csv(loss_path, index=False)
    print(f"Training loss saved to {loss_path}")

    # 그래프 표시
    plt.show()

def test(model, dataloader, device, total_length, actual_data, h):
    """
    For testing model.

    Args:
        model (nn.Module): 학습 완료 모델
        dataloader (DataLoader): 테스트 데이터 로더
        device (torch.device): 테스트 디바이스
        total_length (int): 데이터 총 길이
        actual_data (torch.tensor): 실제 데이터 (예측값과 비교)
        h (str): 가설 넘버 e.g. H1

    Returns:
        all_errors (np.array): error 값
        all_labels (np.array): labels
        reconstructed_data (float): 평균 재구성 값
    """
    model.eval()
    all_errors = []
    all_labels = []
    
    reconstructed_sum = np.zeros((total_length,))  # 재구성 값의 합
    reconstructed_counts = np.zeros((total_length,))  # 재구성에 기여한 횟수
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels = labels.numpy()

            # 모델 입력 형태로 변환
            inputs = inputs.permute(1, 0, 2)  # (window_size, batch_size, input_size)
            targets = targets.permute(1, 0, 2)

            # 모델 예측
            outputs = model(inputs, targets)  # (seq_len, batch_size, output_size)

            # 출력 형태 복원
            outputs = outputs.permute(1, 0, 2).cpu().numpy()  # (batch_size, window_size, output_size)
            targets = targets.permute(1, 0, 2).cpu().numpy()

            batch_size = outputs.shape[0]
            for i in range(batch_size):
                start = dataloader.dataset.stride * (batch_idx * dataloader.batch_size + i)
                end = start + dataloader.dataset.window_size
                if end > total_length:
                    # 윈도우가 데이터 범위를 벗어나지 않도록 조정
                    end = total_length
                    start = end - dataloader.dataset.window_size
                    if start < 0:
                        start = 0
                        end = dataloader.dataset.window_size

                # 재구성 값 합산
                reconstructed_sum[start:end] += outputs[i,:,0]
                # 재구성 횟수 카운트
                reconstructed_counts[start:end] += 1

                # MSE 계산
                mse = np.mean((outputs[i] - targets[i])**2)
                all_errors.append(mse)
                all_labels.extend(labels[i].tolist())

    # 평균 재구성 값 계산 (0으로 나누는 것을 방지)
    reconstructed_data = np.divide(
        reconstructed_sum, 
        reconstructed_counts, 
        out=np.zeros_like(reconstructed_sum), 
        where=reconstructed_counts!=0
    )

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)

    print("Testing Complete.")
    # window 및 stride로 인한 actual_data와 reconstructed_data의 길이 불일치 해결
    min_length = min(len(actual_data), len(reconstructed_data))
    actual_data = actual_data[:min_length]
    reconstructed_data = reconstructed_data[:min_length]

    # 플롯
    plt.figure(figsize=(15, 6))
    plt.plot(actual_data, label='Actual Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='orange', alpha=0.7)
    plt.title('Actual Data vs Reconstructed Data')
    plt.xlabel('Timestep Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프를 파일로 저장
    os.makedirs(h, exist_ok=True)
    plot_path = os.path.join(h, 'actual_vs_reconstructed.png')
    plt.savefig(plot_path)
    print(f"Actual vs Reconstructed data plot saved to {plot_path}")
    
    # 그래프 표시
    plt.show()
    
    return all_errors, all_labels, reconstructed_data