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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, 
    cohen_kappa_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
import seaborn as sns
from arch import arch_model

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

def find_anomaly_windows(labels, anomaly_label=-1):
    """
    Find contiguous anomaly windows in the label array.
    """
    windows = []
    in_anomaly = False
    start = 0

    for i, label in enumerate(labels):
        if label == anomaly_label and not in_anomaly:
            in_anomaly = True
            start = i
        elif label != anomaly_label and in_anomaly:
            in_anomaly = False
            end = i - 1
            windows.append((start, end))

    if in_anomaly:
        windows.append((start, len(labels) - 1))

    return windows

def test(model, dataloader, device, total_length, actual_data, h, anomaly_labels, moving_avg_window=20, k=2, threshold_percentile=95, threshold_method='static', scaling_factor=10):
    """
    For testing the anomaly detection model on time-step level labels with various thresholding methods.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Test data loader
        device (torch.device): Device to run the model on
        total_length (int): Total length of the dataset
        actual_data (np.ndarray): Actual data for comparison and plotting
        h (str): Hypothesis number or identifier (e.g., 'H1')
        anomaly_labels (np.ndarray): 1D array of labels (1 for normal, -1 for anomaly)
        moving_avg_window (int): Window size for moving average (default: 20)
        k (float): Multiplier for standard deviation in threshold (default: 2)
        threshold_percentile (float): Percentile to set the static threshold (default: 95)
        threshold_method (str): Thresholding method ('static', 'moving_avg', 'garch')
        scaling_factor (float): Factor to scale reconstruction errors for GARCH modeling (default: 10)

    Returns:
        None
    """
    model.eval()
    all_errors = np.zeros(total_length)  # Reconstruction error per timestep
    error_counts = np.zeros(total_length)  # Number of times each timestep is covered by a window
    reconstructed_data = np.zeros(total_length)  # Reconstructed data per timestep
    reconstructed_counts = np.zeros(total_length)  # Number of times each timestep is reconstructed

    with torch.no_grad():
        for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Assuming labels are per-timestep, flatten if needed
            labels = labels.numpy().flatten()  # Flatten to 1D array if it's 2D

            # Adjust input shape for the model
            inputs = inputs.permute(1, 0, 2)  # (window_size, batch_size, input_size)
            targets = targets.permute(1, 0, 2)

            # Model prediction
            outputs = model(inputs, targets)  # (seq_len, batch_size, output_size)

            # Restore output shape
            outputs = outputs.permute(1, 0, 2).cpu().numpy()  # (batch_size, window_size, output_size)
            targets = targets.permute(1, 0, 2).cpu().numpy()  # (batch_size, window_size, output_size)

            batch_size = outputs.shape[0]
            window_size = dataloader.dataset.window_size
            stride = dataloader.dataset.stride

            for i in range(batch_size):
                # Calculate window start and end indices
                start = stride * (batch_idx * dataloader.batch_size + i)
                end = start + window_size
                if end > total_length:
                    # Adjust window to fit within the data range
                    end = total_length
                    start = end - window_size
                    if start < 0:
                        start = 0
                        end = window_size

                # Calculate per-timestep MSE
                mse_per_timestep = (outputs[i,:,0] - targets[i,:,0]) ** 2  # Assuming output size is 1

                # Accumulate reconstruction error
                current_window_size = end - start
                if len(mse_per_timestep) != current_window_size:
                    print(f"Warning: mse_per_timestep length {len(mse_per_timestep)} does not match window slice length {current_window_size}. Adjusting.")
                    mse_per_timestep = mse_per_timestep[:current_window_size]

                all_errors[start:end] += mse_per_timestep
                error_counts[start:end] += 1

                # Accumulate reconstructed data
                reconstructed_window = outputs[i,:,0]
                if len(reconstructed_window) != current_window_size:
                    print(f"Warning: reconstructed_window length {len(reconstructed_window)} does not match window slice length {current_window_size}. Adjusting.")
                    reconstructed_window = reconstructed_window[:current_window_size]
                reconstructed_data[start:end] += reconstructed_window
                reconstructed_counts[start:end] += 1

    # Calculate average reconstruction error per timestep
    all_errors = np.divide(
        all_errors,
        error_counts,
        out=np.zeros_like(all_errors),
        where=error_counts != 0
    )

    # Calculate average reconstructed data per timestep
    reconstructed_data = np.divide(
        reconstructed_data,
        reconstructed_counts,
        out=np.zeros_like(reconstructed_data),
        where=reconstructed_counts != 0
    )

    # Convert anomaly_labels to binary (1 for anomaly, 0 for normal)
    binary_labels = (anomaly_labels.flatten() == -1).astype(int)

    print("Testing Complete.")

    # Initialize threshold and predictions
    if threshold_method == 'static':
        # Static Threshold based on percentile
        threshold = np.percentile(all_errors, threshold_percentile)
        print(f"Using static Reconstruction Error Threshold ({threshold_percentile}th percentile): {threshold:.6f}")
        predictions = (all_errors > threshold).astype(int)

    elif threshold_method == 'moving_avg':
        # Dynamic Threshold based on moving average and std
        errors_series = pd.Series(all_errors)
        moving_avg = errors_series.rolling(window=moving_avg_window, min_periods=1).mean()
        moving_std = errors_series.rolling(window=moving_avg_window, min_periods=1).std().fillna(0)
        dynamic_threshold = moving_avg + k * moving_std
        threshold = dynamic_threshold.values  # Convert to NumPy array
        print(f"Using dynamic Reconstruction Error Threshold based on {moving_avg_window}-window moving average and {k}*std")
        predictions = (all_errors > threshold).astype(int)

    elif threshold_method == 'garch':
        # Dynamic Threshold based on GARCH model
        print("Fitting GARCH(1,1) model on scaled reconstruction errors...")
        try:
            # 스케일링 적용
            reconstruction_errors_scaled = all_errors * scaling_factor

            # NaN 값 처리
            reconstruction_errors_scaled = np.nan_to_num(reconstruction_errors_scaled, nan=0.0)

            # GARCH 모델 적합 (mean='Constant' 명시적으로 설정)
            am = arch_model(reconstruction_errors_scaled, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
            res = am.fit(disp='off', show_warning=False)

            # 속성 확인
            if hasattr(res, 'conditional_volatility'):
                cond_vol = res.conditional_volatility
            else:
                print("Error: 'conditional_volatility' not found in ARCHModelResult.")

            # 임계값 계산: conditional_mean + k * conditional_volatility
            threshold_scaled = k * cond_vol

            # 임계값을 원래 스케일로 변환
            threshold = threshold_scaled / scaling_factor

            # 임계값의 길이를 데이터 길이에 맞춤
            threshold = threshold[:total_length]

            print(f"Using GARCH-based Dynamic Reconstruction Error Threshold with k={k}")
            predictions = (all_errors > threshold).astype(int)

        except Exception as e:
            print(f"Error fitting GARCH model: {e}")

    else:
        raise ValueError("Invalid threshold_method. Choose from 'static', 'moving_avg', 'garch'.")

    # Calculate classification metrics
    precision = precision_score(binary_labels, predictions, zero_division=0)
    recall = recall_score(binary_labels, predictions, zero_division=0)
    f1 = f1_score(binary_labels, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(binary_labels, all_errors)
        pr_auc = average_precision_score(binary_labels, all_errors)
    except ValueError as e:
        print(f"ROC AUC and PR AUC cannot be calculated: {e}")
        roc_auc = None
        pr_auc = None
    cohen_kappa = cohen_kappa_score(binary_labels, predictions)

    print("[Classification Metrics]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC: Undefined")
    if pr_auc is not None:
        print(f"PR AUC: {pr_auc:.4f}")
    else:
        print("PR AUC: Undefined")
    print(f"Cohen Kappa: {cohen_kappa:.4f}")
    print()

    # Plot Reconstruction Error with Threshold and Predictions
    plt.figure(figsize=(15, 6))
    plt.plot(all_errors, label='Reconstruction Error', color='blue')
    if threshold_method == 'static':
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    elif threshold_method == 'moving_avg':
        plt.plot(dynamic_threshold, color='red', linestyle='--', label='Dynamic Threshold')
    elif threshold_method == 'garch':
        plt.plot(threshold, color='red', linestyle='--', label='GARCH Dynamic Threshold')

    # Highlight Predictions
    TP = (binary_labels == 1) & (predictions == 1)
    FP = (binary_labels == 0) & (predictions == 1)
    FN = (binary_labels == 1) & (predictions == 0)
    # TN = (binary_labels == 0) & (predictions == 0)  # Not used in plotting

    # Ensure TP, FP, FN are NumPy arrays
    TP = TP if isinstance(TP, np.ndarray) else TP.to_numpy()
    FP = FP if isinstance(FP, np.ndarray) else FP.to_numpy()
    FN = FN if isinstance(FN, np.ndarray) else FN.to_numpy()

    # Scatter plots
    plt.scatter(np.where(TP)[0], all_errors[TP], marker='o', color='green', label='True Positives (TP)', s=10)
    plt.scatter(np.where(FP)[0], all_errors[FP], marker='x', color='orange', label='False Positives (FP)', s=10)
    plt.scatter(np.where(FN)[0], all_errors[FN], marker='x', color='purple', label='False Negatives (FN)', s=10)

    plt.title('Reconstruction Error with Anomaly Detection')
    plt.xlabel('Timestep Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show the plot
    os.makedirs(h, exist_ok=True)
    plot_path = os.path.join(h, f'reconstruction_error_{threshold_method}_threshold.png')
    plt.savefig(plot_path)
    print(f"Reconstruction error plot saved to {plot_path}")
    plt.show()

    # Plot Actual Data vs Reconstructed Data
    plt.figure(figsize=(15, 6))
    plt.plot(actual_data, label='Actual Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='orange', alpha=0.7)
    plt.title('Actual Data vs Reconstructed Data')
    plt.xlabel('Timestep Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show the plot
    actual_recon_plot_path = os.path.join(h, 'actual_vs_reconstructed_data.png')
    plt.savefig(actual_recon_plot_path)
    print(f"Actual vs Reconstructed Data plot saved to {actual_recon_plot_path}")
    plt.show()

    # ROC Curve Visualization
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(binary_labels, all_errors)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        roc_plot_path = os.path.join(h, 'roc_curve_all_data.png')
        plt.savefig(roc_plot_path)
        print(f"ROC curve plot saved to {roc_plot_path}")
        plt.show()

    # Precision-Recall Curve Visualization
    if pr_auc is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(binary_labels, all_errors)
        plt.figure(figsize=(10, 6))
        plt.plot(recall_vals, precision_vals, label=f'PR Curve (AUC = {pr_auc:.4f})', color='navy')
        plt.title('Precision-Recall (PR) Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.tight_layout()
        pr_plot_path = os.path.join(h, 'pr_curve_all_data.png')
        plt.savefig(pr_plot_path)
        print(f"PR curve plot saved to {pr_plot_path}")
        plt.show()

    # Confusion Matrix Visualization
    cm = confusion_matrix(binary_labels, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    confusion_plot_path = os.path.join(h, 'confusion_matrix.png')
    plt.savefig(confusion_plot_path)
    print(f"Confusion matrix plot saved to {confusion_plot_path}")
    plt.show()

    # Optional: Plot Anomaly Windows on Reconstruction Error Plot
    # Find anomaly windows from anomaly_labels
    anomaly_windows = find_anomaly_windows(binary_labels, anomaly_label=1)  # Assuming 1 is anomaly

    if anomaly_windows:
        plt.figure(figsize=(15, 6))
        plt.plot(all_errors, label='Reconstruction Error', color='blue')
        if threshold_method == 'static':
            plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        elif threshold_method == 'moving_avg':
            plt.plot(dynamic_threshold, color='red', linestyle='--', label='Dynamic Threshold')
        elif threshold_method == 'garch':
            plt.plot(threshold, color='red', linestyle='--', label='GARCH Dynamic Threshold')

        # Highlight Predictions
        plt.scatter(np.where(TP)[0], all_errors[TP], marker='o', color='green', label='True Positives (TP)', s=10)
        plt.scatter(np.where(FP)[0], all_errors[FP], marker='x', color='orange', label='False Positives (FP)', s=10)
        plt.scatter(np.where(FN)[0], all_errors[FN], marker='x', color='purple', label='False Negatives (FN)', s=10)

        # Highlight Anomaly Windows
        for idx, window in enumerate(anomaly_windows):
            start, end = window
            plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly' if idx == 0 else "")

        plt.title('Reconstruction Error with Anomaly Detection and Anomaly Windows')
        plt.xlabel('Timestep Index')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save and show the plot
        anomaly_plot_path = os.path.join(h, f'reconstruction_error_with_anomalies_{threshold_method}.png')
        plt.savefig(anomaly_plot_path)
        print(f"Reconstruction error with anomalies plot saved to {anomaly_plot_path}")
        plt.show()