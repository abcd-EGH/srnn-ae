import numpy as np
import torch
import torch.nn as nn
import random

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, forget_bias=1.0, dense=None,
                 file_name='h1', type='enc', component=1, partition=1, seed=None, skip_steps=1, use_cell_residual=False):
        super(sLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.file_name = file_name
        self.type = type
        self.component = component
        self.partition = partition
        self.step = 0  # TensorFlow의 _step에 해당
        self.skip_steps = skip_steps  # 스킵 스텝 추가
        self.use_cell_residual = use_cell_residual  # 셀 단위 잔차 연결 사용 여부 플래그

        # 메인 LSTM 셀 초기화
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # LSTM 셀의 가중치를 Xavier 초기화, 편향은 0으로 초기화
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # 스킵 연결을 위한 추가 가중치와 편향 초기화
        self.weight_h_2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h_2 = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.xavier_normal_(self.weight_h_2)
        nn.init.zeros_(self.bias_h_2)

        # 셀 단위 Residual Connection을 사용할 경우, 입력과 은닉 크기가 다르면 선형 변환 레이어 추가
        if self.use_cell_residual and input_size != hidden_size:
            self.cell_residual_transform = nn.Linear(input_size, hidden_size)
        else:
            self.cell_residual_transform = None

        # 활성화 함수는 여전히 Tanh
        self.activation = torch.tanh

        # 재현성을 위한 마스크 생성기 초기화
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        else:
            self.seed = None
            self.rng = np.random.RandomState()

        # 스킵 연결을 위한 과거 은닉 상태 저장 버퍼
        self.hidden_buffer = []

        # 마스크 캐시 초기화
        self.mask_cache = {}  # step: (mask_w1, mask_w2)

    def masked_weight(self):
        # 기존 마스킹 메커니즘 유지
        if self.step in self.mask_cache:
            return self.mask_cache[self.step]

        # 새로운 마스크 생성 및 캐싱
        mask_combined = self.rng.randint(0, 3, size=self.hidden_size)
        masked_W1 = (mask_combined == 0) | (mask_combined == 2)
        masked_W2 = (mask_combined == 1) | (mask_combined == 2)

        masked_W1 = masked_W1.astype(np.float32)
        masked_W2 = masked_W2.astype(np.float32)

        self.mask_cache[self.step] = (masked_W1, masked_W2)

        # 마스크를 torch 텐서로 변환
        masked_W1, masked_W2 = self.mask_cache[self.step]
        tf_mask_W1 = torch.tensor(masked_W1, dtype=torch.float32, device=self.weight_h_2.device)
        tf_mask_W2 = torch.tensor(masked_W2, dtype=torch.float32, device=self.weight_h_2.device)
        return tf_mask_W1, tf_mask_W2

    def forward(self, input, state):
        """
        Args:
            input: Tensor의 형태 (batch_size, input_size)
            state: Tuple of (h, c), each of shape (batch_size, hidden_size)
        Returns:
            h: 새로운 은닉 상태
            new_state: Tuple of (new_h, new_c)
        """
        h, c = state
        self.step += 1

        # LSTM 셀 출력 계산
        new_h_1, new_c = self.lstm(input, (h, c))

        # 스킵 연결을 위한 은닉 상태 업데이트
        self.hidden_buffer.append(h.detach())
        if len(self.hidden_buffer) > self.skip_steps:
            h_skip = self.hidden_buffer.pop(0)
        else:
            h_skip = torch.zeros_like(h)  # 충분한 이전 스텝이 없을 경우 0으로 채움

        # 스킵 연결을 사용하여 new_h_2 계산 (시그모이드 활성화 적용)
        new_h_2 = torch.sigmoid(torch.matmul(h_skip, self.weight_h_2) + self.bias_h_2)

        # 마스크 획득
        mask_w1, mask_w2 = self.masked_weight()

        # 마스크 적용: (1,0), (0,1), (1,1) 매핑
        new_h = new_h_1 * mask_w1 + new_h_2 * mask_w2

        # 셀 단위 Residual Connection 적용 (사용 시)
        if self.use_cell_residual:
            if self.cell_residual_transform:
                cell_residual = self.cell_residual_transform(input)
            else:
                cell_residual = input
            new_h = new_h + cell_residual  # 셀 잔차 추가

        new_state = (new_h, new_c)
        return new_h, new_state

    def reset_step(self):
        """스텝 카운터와 마스크 캐시를 리셋합니다."""
        self.step = 0
        self.mask_cache = {}

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, skip_steps=1, file_name='enc', partition=1, use_cell_residual=False, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # sLSTMCell을 다층 구조로 사용하기 위해 ModuleList로 저장
        self.cells = nn.ModuleList([
            sLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                type='enc',
                component=i+1,
                partition=partition,
                file_name=file_name,
                skip_steps=skip_steps,
                use_cell_residual=use_cell_residual,  # 셀 단위 잔차 연결 플래그 전달
                **kwargs
            )
            for i in range(num_layers)
        ])

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor의 형태 (seq_len, batch_size, input_size)
        Returns:
            outputs: 각 레이어의 마지막 은닉 상태 리스트
            states: 각 레이어의 마지막 (h, c) 튜플 리스트
        """
        batch_size = inputs.size(1)
        seq_len = inputs.size(0)

        # 초기 은닉 상태와 셀 상태를 0으로 초기화
        h = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        states = list(zip(h, c))

        # 각 타임스텝에 대해 순환
        for t in range(seq_len):
            input_t = inputs[t]
            for i, cell in enumerate(self.cells):
                h_i, c_i = states[i]
                h_i, (h_i, c_i) = cell(input_t, (h_i, c_i))
                states[i] = (h_i, c_i)
                input_t = h_i  # 다음 레이어의 입력으로 현재 레이어의 출력을 사용
        outputs = [state[0] for state in states]  # 각 레이어의 마지막 은닉 상태
        return outputs, states

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, skip_steps=1, file_name='dec', partition=1, use_cell_residual=False, **kwargs):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # sLSTMCell을 다층 구조로 사용하기 위해 ModuleList로 저장
        self.cells = nn.ModuleList([
            sLSTMCell(
                output_size if i == 0 else hidden_size,
                hidden_size,
                type='dec',
                component=i+1,
                partition=partition,
                file_name=file_name,
                skip_steps=skip_steps,
                use_cell_residual=use_cell_residual,  # 셀 단위 잔차 연결 플래그 전달
                **kwargs
            )
            for i in range(num_layers)
        ])
        # 최종 출력을 위한 선형 레이어
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, targets, encoder_states):
        """
        Args:
            targets: Tensor의 형태 (seq_len, batch_size, output_size)
            encoder_states: 인코더의 마지막 (h, c) 튜플 리스트
        Returns:
            outputs: 재구성된 출력의 형태 (seq_len, batch_size, output_size)
        """
        batch_size = targets.size(1)
        seq_len = targets.size(0)

        # 인코더의 마지막 상태를 디코더의 초기 상태로 사용
        h = [state[0].detach() for state in encoder_states]  # 인코더로의 그래디언트 흐름 방지를 위해 detach
        c = [state[1].detach() for state in encoder_states]
        states = list(zip(h, c))

        outputs = []
        # 각 타임스텝에 대해 순환
        for t in range(seq_len):
            input_t = targets[t]
            for i, cell in enumerate(self.cells):
                h_i, c_i = states[i]
                h_i, (h_i, c_i) = cell(input_t, (h_i, c_i))
                states[i] = (h_i, c_i)
                input_t = h_i  # 다음 레이어의 입력으로 현재 레이어의 출력을 사용
            output_t = self.output_layer(input_t)
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=0)
        return outputs

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, skip_steps=1,
                 file_name_enc='enc', file_name_dec='dec', partition=1, use_cell_residual=False, **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            skip_steps=skip_steps,
            file_name=file_name_enc,
            partition=partition,
            use_cell_residual=use_cell_residual,  # 셀 단위 잔차 연결 플래그 전달
            **kwargs
        )
        self.decoder = Decoder(
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            skip_steps=skip_steps,
            file_name=file_name_dec,
            partition=partition,
            use_cell_residual=use_cell_residual,  # 셀 단위 잔차 연결 플래그 전달
            **kwargs
        )

    def forward(self, inputs, targets):
        encoder_outputs, encoder_states = self.encoder(inputs)
        outputs = self.decoder(targets, encoder_states)
        return outputs

class ERSLAE(nn.Module):
    def __init__(self, N, input_size, hidden_size, output_size, num_layers=1, limit_skip_steps=2,
                 file_names=None, seed=777, use_cell_residual=False, **kwargs):
        """
        Args:
            N: 앙상블 내 오토인코더 수
            input_size: 입력 특징 크기
            hidden_size: LSTM의 은닉 상태 크기
            output_size: 출력 특징 크기
            num_layers: 각 오토인코더의 레이어 수
            limit_skip_steps: skip_steps의 최대값 (예: 2는 skip_steps가 1 또는 2일 수 있음)
            file_names: 각 오토인코더의 file_name 리스트. 길이는 N이어야 함.
            seed: 랜덤 시드
            use_cell_residual: 오토인코더에 셀 단위 잔차 연결을 사용할지 여부
            **kwargs: sLSTMCell에 대한 추가 키워드 인자
        """
        super(ERSLAE, self).__init__()
        self.N = N
        self.autoencoders = nn.ModuleList()

        # 기본 파일 이름 설정
        if file_names is None:
            file_names = [f'model{i}' for i in range(N)]
        elif len(file_names) != N:
            raise ValueError("file_names의 길이는 N과 같아야 합니다.")

        for idx in range(N):
            # 각 오토인코더에 대해 1부터 limit_skip_steps까지 중 랜덤하게 skip_steps 선택
            random_skip_steps = random.choice([i for i in range(1, limit_skip_steps+1)])

            # 주어진 file_name을 기반으로 인코더와 디코더의 file_name 생성
            file_name_enc = f'{file_names[idx]}_enc'
            file_name_dec = f'{file_names[idx]}_dec'

            # 파티션 번호는 오토인코더 인덱스 +1로 설정
            partition = idx + 1

            # 각 오토인코더마다 시드를 다르게 설정하여 재현성 유지
            autoencoder_seed = seed + idx

            autoencoder = AutoEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                skip_steps=random_skip_steps,
                file_name_enc=file_name_enc,
                file_name_dec=file_name_dec,
                partition=partition,
                use_cell_residual=use_cell_residual,  # 셀 단위 잔차 연결 플래그 전달
                seed=autoencoder_seed,
                **kwargs
            )
            self.autoencoders.append(autoencoder)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor의 형태 (seq_len, batch_size, input_size)
            targets: Tensor의 형태 (seq_len, batch_size, output_size)
        Returns:
            outputs: 재구성된 출력의 평균값, 형태 (seq_len, batch_size, output_size)
        """
        ensemble_outputs = []
        for autoencoder in self.autoencoders:
            output = autoencoder(inputs, targets)
            ensemble_outputs.append(output)
        # 모든 출력들을 스택하고 평균을 냄
        stacked_outputs = torch.stack(ensemble_outputs, dim=0)  # 형태: (N, seq_len, batch_size, output_size)
        averaged_output = torch.mean(stacked_outputs, dim=0)    # 형태: (seq_len, batch_size, output_size)
        return averaged_output

    def reset_steps(self):
        """앙상블 내 모든 오토인코더의 스텝 카운터와 마스크 캐시를 리셋합니다."""
        for autoencoder in self.autoencoders:
            for cell in autoencoder.encoder.cells:
                cell.reset_step()
            for cell in autoencoder.decoder.cells:
                cell.reset_step()
